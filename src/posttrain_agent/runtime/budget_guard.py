"""Hard caps on one agent invocation: model calls and tokens. No unbounded loops.

Attach to the coordinator and every subagent. Counters live in graph state (not on the
instance) and reset on each invocation, so one shared instance is safe under concurrency, and
a run resumed from a checkpoint after a crash gets a fresh budget for the resumed step.

When a cap is reached the agent is stopped cleanly with a final message in the JSON shape the
role contracts expect (``status: BUDGET_EXHAUSTED``) instead of raising, so the coordinator can
re-plan. Shortly before the cap the model is told to answer now, without more tool calls.
Token counts are a character/4 estimate of the messages added since the previous call (linear
in history length, not quadratic), not provider usage.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import PrivateStateAttr, hook_config
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.channels.untracked_value import UntrackedValue

from ..observability.events import CURRENT_RUN

WARN_AT = 0.75  # fraction of either cap after which the model is told to wrap up


class BudgetState(AgentState):
    guard_calls: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]
    guard_tokens: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]
    guard_seen: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]  # messages already counted


def estimate_tokens(messages: list[Any]) -> int:
    """Rough token count of a message list (characters / 4)."""
    chars = 0
    for m in messages:
        c = getattr(m, "content", "")
        chars += len(c) if isinstance(c, str) else len(json.dumps(c, default=str))
        chars += len(json.dumps(getattr(m, "tool_calls", None) or [], default=str))
    return chars // 4


class RunTokenLedger:
    """Estimated tokens spent per run across the coordinator and every subagent.

    Persisted to ``path`` after every update so a crash and resume cannot reset the total.
    """

    def __init__(self, path: Path, max_tokens: int) -> None:
        self.path, self.max_tokens = path, max_tokens
        self._lock = threading.Lock()

    def _read(self) -> dict[str, int]:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}

    def total(self, run: str) -> int:
        with self._lock:
            return int(self._read().get(run, 0))

    def add(self, run: str, tokens: int) -> int:
        with self._lock:
            data = self._read()
            data[run] = int(data.get(run, 0)) + tokens
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            tmp.replace(self.path)
            return data[run]


class BudgetGuard(AgentMiddleware):
    state_schema = BudgetState

    def __init__(self, name: str, max_calls: int, max_tokens: int,
                 ledger: RunTokenLedger | None = None) -> None:
        if max_calls < 1 or max_tokens < 1:
            raise ValueError("max_calls and max_tokens must be >= 1")
        super().__init__()
        self.role, self.max_calls, self.max_tokens = name, max_calls, max_tokens
        self.ledger = ledger

    def _exhausted(self, state: Any) -> str | None:
        if self.ledger and self.ledger.total(CURRENT_RUN.get()) >= self.ledger.max_tokens:
            return f"run token cap ~{self.ledger.total(CURRENT_RUN.get())}/{self.ledger.max_tokens}"
        calls, tokens = state.get("guard_calls", 0), state.get("guard_tokens", 0)
        if calls >= self.max_calls:
            return f"model call cap {calls}/{self.max_calls}"
        if tokens >= self.max_tokens:
            return f"token cap ~{tokens}/{self.max_tokens}"
        return None

    def _nearly(self, state: Any) -> bool:
        if self.ledger and self.ledger.total(CURRENT_RUN.get()) >= WARN_AT * self.ledger.max_tokens:
            return True
        return (state.get("guard_calls", 0) >= WARN_AT * self.max_calls
                or state.get("guard_tokens", 0) >= WARN_AT * self.max_tokens)

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        why = self._exhausted(state)
        if why is None:
            return None
        final = json.dumps({"status": "BUDGET_EXHAUSTED", "role": self.role, "limit": why,
                            "note": "stopped by budget; partial work is in /work/ files"})
        return {"jump_to": "end", "messages": [AIMessage(content=final)]}

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.before_model(state, runtime)

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = state["messages"]
        seen = min(state.get("guard_seen", 0), len(messages))
        spent = estimate_tokens(messages[seen:])  # only messages new since the previous call
        if self.ledger:
            self.ledger.add(CURRENT_RUN.get(), spent)
        return {"guard_calls": state.get("guard_calls", 0) + 1,
                "guard_tokens": state.get("guard_tokens", 0) + spent,
                "guard_seen": len(messages)}

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.after_model(state, runtime)

    def _warned(self, request: Any) -> Any:
        if not self._nearly(request.state):
            return request
        note = SystemMessage(content=(
            "BUDGET NEARLY EXHAUSTED. Do not call more tools. Reply now with the single JSON "
            "object your OUTPUT CONTRACT describes, using only what you already have."))
        return request.override(messages=[*request.messages, note])

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return handler(self._warned(request))

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        return await handler(self._warned(request))
