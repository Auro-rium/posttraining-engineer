"""Coordinator exit gate and operator-note injection (deterministic, no model involved).

``exit_problem`` decides from durable records (run state, eval ledger, gate log, aborts) whether a
run may end: once a candidate training job exists in the run, the run may only end after a gate
verdict for that candidate or a recorded ``abort_experiment``. ``ExitGate`` enforces it on a
no-tool-call model message (re-prompting up to ``max_reprompts`` times); ``replay`` re-checks it to
decide COMPLETED vs INCOMPLETE. ``require_launch`` (the replay/CLI check) also makes a run that
never launched a job and has no abort INCOMPLETE; the live ExitGate leaves it off by default.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import PrivateStateAttr, hook_config
from langchain_core.messages import HumanMessage
from langgraph.channels.untracked_value import UntrackedValue

from ..experiments import ledger as ledger_mod
from ..experiments import runstate as rs
from ..observability.events import CURRENT_RUN, Journal, emit


def _candidate_problem(state_dir: Path | str, run: str, cand: dict) -> str | None:
    """What is missing for ONE launched candidate job (None when it is gated or aborted)."""
    since = float(cand.get("first_ts", cand.get("ts", 0)))
    if any(float(a.get("ts", 0)) >= since for a in rs.abort_records(state_dir, run)):
        return None
    job, ckpt = cand.get("job_id"), cand.get("checkpoint")
    status = str(cand.get("status", "")).lower()
    for g in rs.gate_records(state_dir, run):
        if (g.get("candidate_job") == job and job) or (g.get("candidate") == ckpt and ckpt):
            if float(g.get("ts", 0)) >= since:
                return None
    if status in rs.TERMINAL_FAILED:
        return (f"training job {job or '(launch)'} ended {status.upper()} and no gate verdict exists. "
                "Call abort_experiment(reason) or revise the plan and relaunch (needs approval).")
    if status != rs.SUCCEEDED or not ckpt:
        return (f"candidate job {job} is {status.upper() or 'UNKNOWN'} with no checkpoint yet. "
                f"Dispatch training_executor with the job id {job} (it must use wait_for_training; "
                "do NOT launch again), or call abort_experiment(reason).")
    errs = []
    for split in ("heldout", "regression"):
        if ledger_mod.latest_ok(state_dir, checkpoint=ckpt, split=split) is None:
            bad = [r for r in ledger_mod.read(state_dir)
                   if r.get("status") == "error" and r.get("split") == split
                   and (r.get("job") == job or r.get("checkpoint") in (ckpt, job))]
            errs.append(f"{split}: " + (str((bad[-1].get("result") or {}).get("error"))[:120]
                                        if bad else "no evaluation recorded"))
    if errs:
        return (f"candidate {ckpt} (job {job}) has no valid evaluation ({'; '.join(errs)}). A BLOCKED "
                f"or FAILED eval is not an outcome: re-dispatch the evaluator with the job id {job}, "
                "or call abort_experiment(reason).")
    return (f"candidate {ckpt} (job {job}) is evaluated on both splits but has no gate verdict. "
            f"Dispatch champion_manager to call run_gate(champion, '{job or ckpt}'), or call "
            "abort_experiment(reason).")


def exit_problem(state_dir: Path | str, run: str, require_launch: bool = True) -> str | None:
    """None when the run may end; otherwise the missing step, phrased as an instruction.

    EVERY candidate job launched by the run needs its own gate verdict (keyed by job id or
    checkpoint) or a later ``abort_experiment``; the first unsatisfied candidate is reported. A
    run that never launched a job and has no abort is not finished either (``require_launch``):
    it must launch a candidate or call ``abort_experiment(reason)`` explaining why not."""
    cands = rs.run_states(state_dir, run or None)
    if not cands:
        if not require_launch or rs.abort_records(state_dir, run):
            return None
        return ("no training job was ever launched in this run and no abort is recorded. Either "
                "launch a candidate (needs approval) and take it through evaluation and run_gate, "
                "or call abort_experiment(reason) stating why no candidate was trained.")
    for cand in cands:
        if problem := _candidate_problem(state_dir, run, cand):
            return problem
    return None


class ExitState(AgentState):
    exit_reprompts: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]


class ExitGate(AgentMiddleware):
    """after_model: a message with no tool calls ends the graph; refuse that until the invariant holds."""

    state_schema = ExitState

    def __init__(self, state_dir: Path | str, max_reprompts: int = 3,
                 require_launch: bool = False) -> None:
        super().__init__()
        self.state_dir, self.max_reprompts = Path(state_dir), max_reprompts
        self.require_launch = require_launch

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        msgs = state["messages"]
        last = msgs[-1] if msgs else None
        if last is None or getattr(last, "type", "") != "ai" or getattr(last, "tool_calls", None):
            return None
        problem = exit_problem(self.state_dir, CURRENT_RUN.get(), self.require_launch)
        n = state.get("exit_reprompts", 0)
        if problem is None or n >= self.max_reprompts:
            return None
        note = HumanMessage(content=(
            f"[exit gate {n + 1}/{self.max_reprompts}] You cannot finish yet: {problem} "
            "Take that step now; do not write a final report first."))
        return {"jump_to": "model", "messages": [note], "exit_reprompts": n + 1}

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.after_model(state, runtime)


class NotesInjector(AgentMiddleware):
    """before_model: deliver unread operator notes (``posttrain note``) once, journaling each."""

    def __init__(self, state_dir: Path | str, journal: Journal | None = None,
                 thread: Callable[[], str] = CURRENT_RUN.get) -> None:
        super().__init__()
        self.state_dir, self.journal, self.thread = Path(state_dir), journal, thread

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        thread = self.thread()
        if not thread:
            return None
        notes = rs.unread_notes(self.state_dir, thread)
        if not notes:
            return None
        for n in notes:
            with contextlib.suppress(Exception):
                emit(self.journal, "operator.note", {"note_id": n.get("id"),
                                                     "text_len": len(str(n.get("text", "")))},
                     run=thread)
        body = "\n".join(f"- {n['text']}" for n in notes)
        return {"messages": [HumanMessage(content=(
            "[operator note] The human operator sent guidance for this run. Follow it unless it "
            f"conflicts with the hard rules:\n{body}"))]}

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.before_model(state, runtime)
