"""Crash recovery and replay on top of LangGraph checkpoints.

* ``run_with_recovery`` - run, and on failure resume the *same thread* from its last
  checkpoint with backoff. Completed steps (e.g. a finished training job) are not re-run.
* ``history`` - list every checkpoint of a thread.
* ``fork`` - re-execute from any past checkpoint. Combined with a ``Cassette`` in
  ``replay`` mode this is deterministic and offline.
* Every attempt is recorded in a run ledger in the store so failures stay inspectable.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from langgraph.store.base import BaseStore

from .backboard_model import CassetteMiss

LEDGER_NS = ("posttrain", "runs")
# Errors where retrying cannot help.
FATAL = (CassetteMiss, KeyboardInterrupt, asyncio.CancelledError)


@dataclass
class RunReport:
    thread_id: str
    status: str  # COMPLETED | FAILED
    attempts: int
    result: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    last_checkpoint_id: str | None = None


def _cfg(thread_id: str, checkpoint_id: str | None = None) -> dict[str, Any]:
    conf: dict[str, Any] = {"thread_id": thread_id}
    if checkpoint_id:
        conf["checkpoint_id"] = checkpoint_id
    return {"configurable": conf, "recursion_limit": 200}


async def _record(store: BaseStore | None, report: RunReport) -> None:
    if store is not None:
        await store.aput(LEDGER_NS, report.thread_id, {
            "status": report.status, "attempts": report.attempts, "errors": report.errors,
            "last_checkpoint_id": report.last_checkpoint_id, "updated": time.time(),
        })


async def run_with_recovery(
    graph: Any,
    goal: str,
    thread_id: str,
    *,
    store: BaseStore | None = None,
    max_attempts: int = 3,
    base_delay: float = 2.0,
) -> RunReport:
    """Run ``goal`` on ``thread_id``; resume from the last checkpoint after failures.

    If the thread already has state, the first attempt resumes instead of restarting.
    """
    snapshot = await graph.aget_state(_cfg(thread_id))
    payload: Any = None if snapshot.values else {"messages": [{"role": "user", "content": goal}]}
    report = RunReport(thread_id, "FAILED", 0)
    for attempt in range(1, max_attempts + 1):
        report.attempts = attempt
        await _record(store, report)
        try:
            result = await graph.ainvoke(payload, _cfg(thread_id))
        except FATAL:
            raise
        except Exception as exc:  # noqa: BLE001 - recorded, then retried from checkpoint
            report.errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            payload = None  # every later attempt resumes the saved thread
            state = await graph.aget_state(_cfg(thread_id))
            report.last_checkpoint_id = (state.config or {}).get("configurable", {}).get(
                "checkpoint_id"
            )
            if attempt < max_attempts:
                await asyncio.sleep(base_delay * 2 ** (attempt - 1))
            continue
        state = await graph.aget_state(_cfg(thread_id))
        report.status, report.result = "COMPLETED", result
        report.last_checkpoint_id = (state.config or {}).get("configurable", {}).get(
            "checkpoint_id"
        )
        break
    await _record(store, report)
    return report


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


async def fork(graph: Any, thread_id: str, checkpoint_id: str) -> dict[str, Any]:
    """Re-execute from a past checkpoint (the original history is preserved)."""
    return await graph.ainvoke(None, _cfg(thread_id, checkpoint_id))


async def ledger(store: BaseStore) -> list[dict[str, Any]]:
    items = await store.asearch(LEDGER_NS, limit=100)
    return [{"thread_id": i.key, **i.value} for i in items]
