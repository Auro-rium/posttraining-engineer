"""Crash recovery, human approval and replay on top of LangGraph checkpoints.

* ``run_with_recovery`` - run, and on failure resume the *same thread* from its last
  checkpoint with backoff. Completed steps (e.g. a launched training job) are not re-run.
* ``AWAITING_APPROVAL`` - a run that reaches a human gate (training launch) pauses,
  durably; ``decide`` resumes it with approve/reject, even from a new process.
* ``history`` / ``fork`` - list checkpoints and re-execute from any past one. With a
  ``Cassette`` in ``replay`` mode this is deterministic and offline.
* Every attempt is recorded in a run ledger in the store.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langgraph.store.base import BaseStore
from langgraph.types import Command

from ..llm.backboard_model import CassetteMiss
from ..llm.resilience import is_transient
from ..observability.events import CURRENT_RUN, Journal, emit

LEDGER_NS = ("posttrain", "runs")
FATAL = (CassetteMiss, KeyboardInterrupt, asyncio.CancelledError)


@dataclass
class RunReport:
    thread_id: str
    # COMPLETED | INCOMPLETE (ended without a gate verdict/abort) | FAILED | FAILED_TRANSIENT
    # (provider outage outlasted the retry window; resumable) | AWAITING_APPROVAL
    status: str
    attempts: int
    result: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    last_checkpoint_id: str | None = None
    pending: list[dict[str, Any]] = field(default_factory=list)


OUTAGE_BUDGET_S = 900.0  # ~15 minutes of retrying a transient provider outage
MAX_TRANSIENT_ATTEMPTS = 100


def is_outage(exc: BaseException) -> bool:
    """Transient provider/network trouble worth waiting out (as opposed to a bug or a bad request)."""
    if isinstance(exc, Exception) and is_transient(exc):
        return True
    return isinstance(exc, OSError | TimeoutError | ConnectionError) or \
        type(exc).__module__.startswith("httpx") and "Error" in type(exc).__name__


def _cfg(thread_id: str, checkpoint_id: str | None = None,
         callbacks: list[Any] | None = None) -> dict[str, Any]:
    conf: dict[str, Any] = {"thread_id": thread_id}
    if checkpoint_id:
        conf["checkpoint_id"] = checkpoint_id
    cfg: dict[str, Any] = {"configurable": conf, "recursion_limit": 200,
                           "metadata": {"thread_id": thread_id}}
    if callbacks:
        cfg["callbacks"] = callbacks
    return cfg


async def pending_actions(graph: Any, thread_id: str) -> list[dict[str, Any]]:
    """Actions waiting for a human decision on this thread."""
    state = await graph.aget_state(_cfg(thread_id))
    actions: list[dict[str, Any]] = []
    for task in getattr(state, "tasks", ()) or ():
        for intr in getattr(task, "interrupts", ()) or ():
            value = getattr(intr, "value", None)
            if isinstance(value, dict):
                actions += list(value.get("action_requests", []))
    return actions


async def _pauses(graph: Any, thread_id: str) -> list[tuple[str, int]]:
    """(interrupt id, number of actions) for every paused subagent on the thread."""
    state = await graph.aget_state(_cfg(thread_id))
    out: list[tuple[str, int]] = []
    for task in getattr(state, "tasks", ()) or ():
        for intr in getattr(task, "interrupts", ()) or ():
            value = getattr(intr, "value", None)
            if isinstance(value, dict) and value.get("action_requests"):
                out.append((intr.id, len(value["action_requests"])))
    return out


async def _record(store: BaseStore | None, report: RunReport) -> None:
    if store is not None:
        await store.aput(LEDGER_NS, report.thread_id, {
            "status": report.status, "attempts": report.attempts, "errors": report.errors,
            "last_checkpoint_id": report.last_checkpoint_id, "updated": time.time(),
            "pending": [{"name": a.get("name"), "args": a.get("args")} for a in report.pending],
        })


_ESTIMATE_MARK = "budget estimate: "


def parse_estimate_usd(description: Any) -> float | None:
    """Estimated cost from the JSON after ``budget estimate: `` in an approval description."""
    if not isinstance(description, str) or _ESTIMATE_MARK not in description:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(description.split(_ESTIMATE_MARK, 1)[1])
        value = obj.get("estimated_cost_usd") if isinstance(obj, dict) else None
    except ValueError:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _checkpoint_of(state: Any) -> tuple[str | None, Any]:
    cid = (getattr(state, "config", None) or {}).get("configurable", {}).get("checkpoint_id")
    step = (getattr(state, "metadata", None) or {}).get("step")
    return cid, step


def _emit_pending(journal: Journal | None, thread_id: str, pending: list[dict[str, Any]]) -> None:
    for act in pending:
        args = act.get("args") if isinstance(act.get("args"), dict) else {}
        emit(journal, "approval.pending", {
            "action": act.get("name"), "base_model": args.get("base_model"),
            "dataset_ref": args.get("dataset_ref"),
            "hyperparameters": args.get("hyperparameters"),
            "estimate_usd": parse_estimate_usd(act.get("description")),
            "cli_hint": f"posttrain approve --thread {thread_id}"}, run=thread_id)


async def run_with_recovery(
    graph: Any,
    goal: str,
    thread_id: str,
    *,
    journal: Journal | None = None,
    backend: str | None = None,
    experiment_budget: int | None = None,
    **kw: Any,
) -> RunReport:
    """``_run_with_recovery`` with ``CURRENT_RUN`` bound to ``thread_id`` for the call."""
    token = CURRENT_RUN.set(thread_id)
    try:
        return await _run_with_recovery(graph, goal, thread_id, journal=journal, backend=backend,
                                        experiment_budget=experiment_budget, **kw)
    finally:
        CURRENT_RUN.reset(token)


async def _run_with_recovery(
    graph: Any,
    goal: str,
    thread_id: str,
    *,
    store: BaseStore | None = None,
    max_attempts: int = 3,
    base_delay: float = 2.0,
    resume: Command | None = None,
    callbacks: list[Any] | None = None,
    journal: Journal | None = None,
    backend: str | None = None,
    experiment_budget: int | None = None,
    exit_check: Callable[[str], str | None] | None = None,
    outage_budget_s: float = OUTAGE_BUDGET_S,
    sleep: Callable[[float], Any] = asyncio.sleep,
    jitter: Callable[[], float] = random.random,
) -> RunReport:
    """Run ``goal`` on ``thread_id``; resume from the last checkpoint after failures.

    If the thread already has state the first attempt resumes instead of restarting.
    ``resume`` answers a pending human approval (see ``decide`` ).

    Non-transient errors use the small ``max_attempts`` budget. Transient ones (provider outage,
    DNS, timeouts) are retried with jittered exponential backoff (cap 60 s) for up to
    ``outage_budget_s`` in total, emitting ``run.status RETRYING``; if the outage outlasts that the
    status is ``FAILED_TRANSIENT`` (resumable with ``posttrain resume``). ``exit_check(thread)``
    returns the missing step when the run ended without a gate verdict or abort: the status is then
    ``INCOMPLETE`` rather than ``COMPLETED``.
    """
    cfg = _cfg(thread_id, callbacks=callbacks)
    snapshot = await graph.aget_state(cfg)
    finished = bool(snapshot.values) and not snapshot.next  # a completed thread: nothing to resume
    payload: Any = resume or (None if snapshot.values and not (finished and goal) else
                              {"messages": [{"role": "user", "content": goal}]})
    if resume is None and not goal and finished and exit_check and (why := exit_check(thread_id)):
        payload = {"messages": [{"role": "user", "content": (
            f"Resume: the previous attempt ended without finishing. {why}")}]}
    if resume is None and not snapshot.values:
        emit(journal, "run.started", {"goal_len": len(goal), "experiment_budget": experiment_budget,
                                      "backend": backend}, run=thread_id)
    report = RunReport(thread_id, "FAILED", 0)
    error_types: list[str] = []  # journal gets type names only; full text stays in report.errors
    hard_failures = transient_failures = 0
    waited, last_transient = 0.0, False
    while True:
        report.attempts += 1
        attempt = report.attempts
        await _record(store, report)
        emit(journal, "run.status", {"status": "RUNNING", "attempt": attempt}, run=thread_id)
        try:
            result = await graph.ainvoke(payload, cfg)
        except FATAL as exc:
            emit(journal, "run.status", {"status": "FAILED", "attempt": attempt,
                                         "errors": [type(exc).__name__]}, run=thread_id)
            raise
        except Exception as exc:  # noqa: BLE001 - recorded, then retried from checkpoint
            report.errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            error_types.append(type(exc).__name__)
            payload = None  # every later attempt resumes the saved thread
            state = await graph.aget_state(cfg)
            report.last_checkpoint_id, step = _checkpoint_of(state)
            emit(journal, "checkpoint", {"checkpoint_id": report.last_checkpoint_id,
                                         "step": step}, run=thread_id)
            last_transient = is_outage(exc)
            if last_transient:
                transient_failures += 1
                delay = min(base_delay * 2 ** (transient_failures - 1), 60.0) * (0.5 + jitter())
                if waited + delay > outage_budget_s or transient_failures >= MAX_TRANSIENT_ATTEMPTS:
                    break
                emit(journal, "run.status", {"status": "RETRYING", "attempt": attempt,
                                             "errors": [type(exc).__name__]}, run=thread_id)
                waited += delay
                await sleep(delay)
                continue
            hard_failures += 1
            if hard_failures >= max_attempts:
                break
            await sleep(base_delay * 2 ** (hard_failures - 1))
            continue
        state = await graph.aget_state(cfg)
        report.last_checkpoint_id, step = _checkpoint_of(state)
        emit(journal, "checkpoint", {"checkpoint_id": report.last_checkpoint_id, "step": step},
             run=thread_id)
        report.pending = await pending_actions(graph, thread_id)
        report.status = "AWAITING_APPROVAL" if report.pending else "COMPLETED"
        report.result = result
        if report.status == "COMPLETED" and exit_check is not None:
            missing = exit_check(thread_id)
            if missing:
                report.status = "INCOMPLETE"
                report.errors.append(f"ended without a gate verdict or abort: {missing}")
                error_types.append("ExitInvariant")
        _emit_pending(journal, thread_id, report.pending)
        break
    if report.status == "FAILED" and last_transient:
        report.status = "FAILED_TRANSIENT"
    await _record(store, report)
    emit(journal, "run.status", {"status": report.status, "attempt": report.attempts,
                                 "errors": error_types}, run=thread_id)
    return report


async def decide(
    graph: Any,
    thread_id: str,
    approve: bool,
    *,
    message: str | None = None,
    store: BaseStore | None = None,
    max_attempts: int = 3,
    base_delay: float = 2.0,
    callbacks: list[Any] | None = None,
    journal: Journal | None = None,
    exit_check: Callable[[str], str | None] | None = None,
) -> RunReport:
    """Approve or reject every pending action of a paused thread, then continue the run."""
    pauses = await _pauses(graph, thread_id)
    if not pauses:
        raise ValueError(f"thread {thread_id!r} has nothing awaiting approval")
    decision: dict[str, Any] = {"type": "approve"} if approve else {
        "type": "reject", **({"message": message} if message else {})}

    def answer(n: int) -> dict[str, Any]:
        return {"decisions": [dict(decision) for _ in range(n)]}

    if len(pauses) == 1:  # the common case: a single paused action set
        resume = Command(resume=answer(pauses[0][1]))
    else:  # parallel subagents each paused: answer every interrupt by id
        resume = Command(resume={iid: answer(n) for iid, n in pauses})
    emit(journal, "approval.resolved", {"decision": decision["type"],
                                        "message_len": len(message or "")}, run=thread_id)
    return await run_with_recovery(
        graph, "", thread_id, store=store, max_attempts=max_attempts, base_delay=base_delay,
        resume=resume, callbacks=callbacks, journal=journal, exit_check=exit_check)


async def history(graph: Any, thread_id: str) -> list[dict[str, Any]]:
    """Checkpoints newest first."""
    rows = []
    async for s in graph.aget_state_history(_cfg(thread_id)):
        rows.append({
            "checkpoint_id": s.config["configurable"]["checkpoint_id"],
            "step": (s.metadata or {}).get("step"),
            "source": (s.metadata or {}).get("source"),
            "next": list(s.next),
            "created_at": s.created_at,
            "messages": len((s.values or {}).get("messages", [])),
        })
    return rows


async def fork(graph: Any, thread_id: str, checkpoint_id: str,
               callbacks: list[Any] | None = None) -> dict[str, Any]:
    """Re-execute from a past checkpoint (the original history is preserved)."""
    return await graph.ainvoke(None, _cfg(thread_id, checkpoint_id, callbacks))


async def ledger(store: BaseStore) -> list[dict[str, Any]]:
    items = await store.asearch(LEDGER_NS, limit=100)
    return [{"thread_id": i.key, **i.value} for i in items]
