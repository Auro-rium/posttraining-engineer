"""Cap how many subagents run at once.

When the coordinator emits several ``task`` calls in one message, LangGraph runs them
concurrently. This middleware gates that with a semaphore so at most ``limit`` subagents
execute at the same time; extra calls wait their turn (they are queued, never dropped).
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware

from ..observability.events import CURRENT_RUN, Journal

TASK_TOOL = "task"


class SubagentConcurrencyMiddleware(AgentMiddleware):
    """Attach to the coordinator only; subagents never spawn subagents."""

    def __init__(self, limit: int = 2, journal: Journal | None = None, run: str = "") -> None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        super().__init__()
        self.limit = limit
        self._async_sem: asyncio.Semaphore | None = None
        self._sync_sem = threading.Semaphore(limit)
        self.peak = 0  # highest concurrency observed; useful in tests and telemetry
        self._active = 0
        self.journal = journal
        self.run = run
        self._slots: set[int] = set()
        self._slot_lock = threading.Lock()

    def _enter(self) -> None:
        self._active += 1
        self.peak = max(self.peak, self._active)

    def _start(self, request: Any) -> tuple[str, int]:
        """Claim the lowest free slot and journal agent.start (best effort)."""
        with self._slot_lock:
            slot = next(i for i in range(1, self.limit + 2) if i not in self._slots)
            self._slots.add(slot)
        role = ""
        with contextlib.suppress(Exception):
            role = str((request.tool_call.get("args") or {}).get("subagent_type", ""))
        self._emit("agent.start", {"role": role, "slot": slot})
        return role, slot

    def _stop(self, role: str, slot: int, result: Any, failed: bool) -> None:
        with self._slot_lock:
            self._slots.discard(slot)
        ok = not failed and getattr(result, "status", None) != "error"
        self._emit("agent.end", {"role": role, "slot": slot, "ok": ok})

    def _emit(self, type: str, data: dict) -> None:
        if self.journal is None:
            return
        with contextlib.suppress(Exception):  # journaling must never fail a run
            self.journal.append(self.run or CURRENT_RUN.get(), type, data)

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        if request.tool_call["name"] != TASK_TOOL:
            return await handler(request)
        if self._async_sem is None:  # created lazily so it binds to the running loop
            self._async_sem = asyncio.Semaphore(self.limit)
        async with self._async_sem:
            self._enter()
            role, slot = self._start(request)
            result, failed = None, True
            try:
                result = await handler(request)
                failed = False
                return result
            finally:
                self._active -= 1
                self._stop(role, slot, result, failed)

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        if request.tool_call["name"] != TASK_TOOL:
            return handler(request)
        with self._sync_sem:
            self._enter()
            role, slot = self._start(request)
            result, failed = None, True
            try:
                result = handler(request)
                failed = False
                return result
            finally:
                self._active -= 1
                self._stop(role, slot, result, failed)
