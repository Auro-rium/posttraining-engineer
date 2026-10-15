"""Cap how many subagents run at once.

When the coordinator emits several ``task`` calls in one message, LangGraph runs them
concurrently. This middleware gates that with a semaphore so at most ``limit`` subagents
execute at the same time; extra calls wait their turn (they are queued, never dropped).
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware

TASK_TOOL = "task"


class SubagentConcurrencyMiddleware(AgentMiddleware):
    """Attach to the coordinator only; subagents never spawn subagents."""

    def __init__(self, limit: int = 2) -> None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        super().__init__()
        self.limit = limit
        self._async_sem: asyncio.Semaphore | None = None
        self._sync_sem = threading.Semaphore(limit)
        self.peak = 0  # highest concurrency observed; useful in tests and telemetry
        self._active = 0

    def _enter(self) -> None:
        self._active += 1
        self.peak = max(self.peak, self._active)

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        if request.tool_call["name"] != TASK_TOOL:
            return await handler(request)
        if self._async_sem is None:  # created lazily so it binds to the running loop
            self._async_sem = asyncio.Semaphore(self.limit)
        async with self._async_sem:
            self._enter()
            try:
                return await handler(request)
            finally:
                self._active -= 1

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        if request.tool_call["name"] != TASK_TOOL:
            return handler(request)
        with self._sync_sem:
            self._enter()
            try:
                return handler(request)
            finally:
                self._active -= 1
