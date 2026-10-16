"""``ChatBackboard``: a LangChain chat model whose brain is the Backboard API.

Design notes
------------
* **Stateless per call.** LangGraph owns conversation state (checkpoints), so every
  call opens a fresh Backboard thread and sends the rendered transcript. Mapping
  graph threads onto Backboard threads would diverge on the first replay, fork or
  isolated subagent. Backboard's *assistant* is the long-lived object: a fixed
  ``assistant_id`` is always passed so assistant-level memory accumulates.
* **Tools are re-sent every turn** (Backboard does not persist per-turn tools) and
  tool results are rendered into the transcript rather than submitted back to a
  pending run.
* **Cassette.** An optional record/replay file keyed by a hash of the request makes
  runs reproducible offline.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from backboard import BackboardAPIError, BackboardClient
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import invalid_tool_call, tool_call
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field


class EmptyResponseError(BackboardAPIError):
    """Backboard returned neither text nor tool calls. Treated as transient."""


class CassetteMiss(RuntimeError):
    """Replay mode asked for a request that was never recorded."""


class Cassette:
    """JSONL record/replay store of model responses.

    ``auto`` replays hits and records misses, ``record`` always calls live,
    ``replay`` fails on a miss.
    """

    def __init__(self, path: Path | str, mode: str = "auto") -> None:
        if mode not in {"auto", "record", "replay"}:
            raise ValueError(f"unknown cassette mode {mode!r}")
        self.path, self.mode = Path(path), mode
        self._entries: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._entries[row["key"]] = row["response"]

    def get(self, key: str) -> dict[str, Any] | None:
        if self.mode == "record":
            return None
        hit = self._entries.get(key)
        if hit is None and self.mode == "replay":
            raise CassetteMiss(f"no recorded response for request {key[:12]}")
        return hit

    def put(self, key: str, response: dict[str, Any]) -> None:
        if self.mode == "replay":
            return
        self._entries[key] = response
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps({"key": key, "response": response}, sort_keys=True) + "\n")


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        parts.append(block if isinstance(block, str) else str(block.get("text", "")))
    return "".join(parts)


def render_messages(messages: Sequence[BaseMessage]) -> tuple[str, str]:
    """Split a LangChain transcript into (system_prompt, content) for Backboard."""
    system = "\n\n".join(_text(m) for m in messages if isinstance(m, SystemMessage))
    rest = [m for m in messages if not isinstance(m, SystemMessage)]
    if len(rest) == 1 and isinstance(rest[0], HumanMessage):
        return system, _text(rest[0])
    lines = ["Conversation so far. Continue as the assistant, answering the final entry."]
    for m in rest:
        if isinstance(m, HumanMessage):
            lines.append(f"[user]\n{_text(m)}")
        elif isinstance(m, AIMessage):
            if _text(m):
                lines.append(f"[assistant]\n{_text(m)}")
            for tc in m.tool_calls:
                lines.append(
                    f"[assistant tool_call id={tc['id']} name={tc['name']}]\n"
                    f"{json.dumps(tc['args'], sort_keys=True)}"
                )
        elif isinstance(m, ToolMessage):
            lines.append(f"[tool_result id={m.tool_call_id} name={m.name or ''}]\n{_text(m)}")
        else:
            lines.append(f"[{m.type}]\n{_text(m)}")
    return system, "\n\n".join(lines)


def _run_sync(coro: Any) -> Any:
    """Run a coroutine from sync code, even if a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class ChatBackboard(BaseChatModel):
    """LangChain chat model backed by Backboard (any provider/model it routes to)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: str = ""
    base_url: str = "https://app.backboard.io/api"
    assistant_id: str | None = None
    llm_provider: str = "anthropic"
    model_name: str = "claude-sonnet-4-5"
    memory: str = "Readonly"
    timeout: int = 120
    max_input_tokens: int = 128_000
    bound_tools: list[dict[str, Any]] = Field(default_factory=list)
    cassette: Cassette | None = Field(default=None, exclude=True)
    client_factory: Callable[[], Any] | None = Field(default=None, exclude=True)

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault(
            "profile",
            {
                "max_input_tokens": kwargs.get("max_input_tokens", 128_000),
                "tool_calling": True,
            },
        )
        super().__init__(**kwargs)

    @property
    def _llm_type(self) -> str:
        return "backboard"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"provider": self.llm_provider, "model": self.model_name}

    def bind_tools(  # type: ignore[override]
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> ChatBackboard:
        # Backboard has no tool_choice control; accepted and ignored for compatibility.
        return self.model_copy(
            update={"bound_tools": [convert_to_openai_tool(t) for t in tools]}
        )

    def _open_client(self) -> Any:
        if self.client_factory is not None:
            return self.client_factory()
        return BackboardClient(self.api_key, base_url=self.base_url, timeout=self.timeout)

    def _cache_key(self, system: str, content: str) -> str:
        blob = json.dumps(
            [self.llm_provider, self.model_name, system, content, self.bound_tools],
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    async def _call(self, system: str, content: str) -> dict[str, Any]:
        key = self._cache_key(system, content)
        if self.cassette and (hit := self.cassette.get(key)) is not None:
            return hit
        async with self._open_client() as client:
            resp = await client.send_message(
                content=content,
                assistant_id=self.assistant_id,
                system_prompt=system or None,
                llm_provider=self.llm_provider,
                model_name=self.model_name,
                memory=self.memory,
                tools=self.bound_tools or None,
                stream=False,
            )
        last = resp.messages[-1] if resp.messages else {}
        status = str(resp.status or "").lower()
        if status in {"failed", "error", "cancelled", "expired"}:
            raise BackboardAPIError(f"Backboard run ended with status {status!r}")
        calls = [
            {
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments or "{}",
            }
            for tc in (resp.tool_calls or [])
        ]
        text = resp.content or ""
        if not text and not calls:
            raise EmptyResponseError("Backboard returned an empty response")
        out = {
            "content": text,
            "tool_calls": calls,
            "usage": {
                "input_tokens": last.get("input_tokens") or 0,
                "output_tokens": last.get("output_tokens") or 0,
            },
        }
        if self.cassette:
            self.cassette.put(key, out)
        return out

    @staticmethod
    def _to_result(payload: dict[str, Any]) -> ChatResult:
        good, bad = [], []
        for raw in payload["tool_calls"]:
            try:
                args = json.loads(raw["arguments"])
                if not isinstance(args, dict):
                    raise ValueError("tool arguments must be a JSON object")
            except ValueError as exc:
                bad.append(invalid_tool_call(
                    name=raw["name"], args=raw["arguments"], id=raw["id"], error=str(exc)
                ))
            else:
                good.append(tool_call(name=raw["name"], args=args, id=raw["id"]))
        usage = payload.get("usage") or {}
        msg = AIMessage(
            content=payload["content"],
            tool_calls=good,
            invalid_tool_calls=bad,
            usage_metadata={
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            },
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system, content = render_messages(messages)
        return self._to_result(await self._call(system, content))

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return _run_sync(self._agenerate(messages, stop, None, **kwargs))
