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
import time
from dataclasses import dataclass, field
from typing import Any

from langgraph.store.base import BaseStore
from langgraph.types import Command

from .backboard_model import CassetteMiss

LEDGER_NS = ("posttrain", "runs")
FATAL = (CassetteMiss, KeyboardInterrupt, asyncio.CancelledError)


@dataclass
class RunReport:
    thread_id: str
    status: str  # COMPLETED | FAILED | AWAITING_APPROVAL
    attempts: int
    result: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    last_checkpoint_id: str | None = None
    pending: list[dict[str, Any]] = field(default_factory=list)


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


async def _record(store: BaseStore | None, report: RunReport) -> None:
    if store is not None:
        await store.aput(LEDGER_NS, report.thread_id, {
            "status": report.status, "attempts": report.attempts, "errors": report.errors,
            "last_checkpoint_id": report.last_checkpoint_id, "updated": time.time(),
            "pending": [{"name": a.get("name"), "args": a.get("args")} for a in report.pending],
        })


async def run_with_recovery(
    graph: Any,
    goal: str,
    thread_id: str,
    *,
    store: BaseStore | None = None,
    max_attempts: int = 3,
    base_delay: float = 2.0,
    resume: Command | None = None,
    callbacks: list[Any] | None = None,
) -> RunReport:
    """Run ``goal`` on ``thread_id``; resume from the last checkpoint after failures.

    If the thread already has state the first attempt resumes instead of restarting.
    ``resume`` answers a pending human approval (see ``decide``).
    """
    cfg = _cfg(thread_id, callbacks=callbacks)
    snapshot = await graph.aget_state(cfg)
    payload: Any = resume or (None if snapshot.values else
                              {"messages": [{"role": "user", "content": goal}]})
    report = RunReport(thread_id, "FAILED", 0)
    for attempt in range(1, max_attempts + 1):
        report.attempts = attempt
        await _record(store, report)
        try:
            result = await graph.ainvoke(payload, cfg)
        except FATAL:
            raise
        except Exception as exc:  # noqa: BLE001 - recorded, then retried from checkpoint
            report.errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            payload = None  # every later attempt resumes the saved thread
            state = await graph.aget_state(cfg)
            report.last_checkpoint_id = (state.config or {}).get("configurable", {}).get(
                "checkpoint_id")
            if attempt < max_attempts:
                await asyncio.sleep(base_delay * 2 ** (attempt - 1))
            continue
        state = await graph.aget_state(cfg)
        report.last_checkpoint_id = (state.config or {}).get("configurable", {}).get(
            "checkpoint_id")
        report.pending = await pending_actions(graph, thread_id)
        report.status = "AWAITING_APPROVAL" if report.pending else "COMPLETED"
        report.result = result
        break
    await _record(store, report)
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
) -> RunReport:
    """Approve or reject every pending action of a paused thread, then continue the run."""
    actions = await pending_actions(graph, thread_id)
    if not actions:
        raise ValueError(f"thread {thread_id!r} has nothing awaiting approval")
    decision: dict[str, Any] = {"type": "approve"} if approve else {
        "type": "reject", **({"message": message} if message else {})}
    return await run_with_recovery(
        graph, "", thread_id, store=store, max_attempts=max_attempts, base_delay=base_delay,
        resume=Command(resume={"decisions": [dict(decision) for _ in actions]}),
        callbacks=callbacks)


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
