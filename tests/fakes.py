"""Scriptable stand-in for ``backboard.BackboardClient``."""

from __future__ import annotations

import json
from typing import Any

from backboard import ChatMessagesResponse


def text(content: str) -> dict[str, Any]:
    return {"content": content, "status": "COMPLETED", "input_tokens": 3, "output_tokens": 2}


def call(name: str, args: dict | str, id: str = "call_1") -> dict[str, Any]:
    arguments = args if isinstance(args, str) else json.dumps(args)
    return {
        "content": "",
        "status": "REQUIRES_ACTION",
        "tool_calls": [
            {"id": id, "type": "function", "function": {"name": name, "arguments": arguments}}
        ],
    }


class FakeBackboard:
    """Pops scripted items per ``send_message``; an Exception item is raised.

    ``script`` may be a list (shared across clients) or a callable(kwargs) -> item.
    """

    def __init__(self, script: Any, log: list[dict[str, Any]] | None = None) -> None:
        self.script = script
        self.log = log if log is not None else []

    async def __aenter__(self) -> FakeBackboard:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def aclose(self) -> None:
        return None

    async def send_message(self, **kwargs: Any) -> ChatMessagesResponse:
        self.log.append(kwargs)
        item = self.script(kwargs) if callable(self.script) else self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return ChatMessagesResponse(messages=[item])

    # memory API
    async def add_memory(self, assistant_id: str, content: str, metadata: dict | None = None):
        self.log.append({"add_memory": content, "assistant_id": assistant_id})
        return {"success": True, "memory_id": "m1"}

    async def search_memories(self, assistant_id: str, query: str, limit: int = 5):
        self.log.append({"search_memories": query, "assistant_id": assistant_id})
        return {"memories": [{"id": "m1", "content": "lr 2e-4 regressed"}], "total_count": 1}

    async def create_assistant(self, name: str, **kwargs: Any):
        class _A:
            assistant_id = "asst-123"

        return _A()
