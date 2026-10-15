"""Backboard long-term memory: assistant bootstrap and remember/recall tools.

Memory tiers in this project:
1. thread state    - LangGraph checkpoints (``persistence``)
2. file memory     - ``/memories/`` in the SQLite store, loaded into the prompt as AGENTS.md
3. semantic memory - Backboard assistant memories (below), written explicitly
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from backboard import BackboardClient
from langchain_core.tools import BaseTool, StructuredTool

from .config import Settings


def _factory(settings: Settings, client_factory: Callable[[], Any] | None) -> Callable[[], Any]:
    return client_factory or (
        lambda: BackboardClient(
            settings.backboard_api_key,
            base_url=settings.backboard_base_url,
            timeout=settings.request_timeout,
        )
    )


async def ensure_assistant(
    settings: Settings, client_factory: Callable[[], Any] | None = None
) -> str:
    """Return the persistent assistant id, creating the assistant on first use.

    Precedence: ``BACKBOARD_ASSISTANT_ID`` -> ``<state_dir>/backboard.json`` -> create.
    """
    if settings.assistant_id:
        return settings.assistant_id
    path = Path(settings.state_dir) / "backboard.json"
    if path.exists():
        return json.loads(path.read_text())["assistant_id"]
    async with _factory(settings, client_factory)() as client:
        assistant = await client.create_assistant(
            name="posttrain-agent",
            description="Autonomous post-training engineer",
            system_prompt="You are the reasoning engine of an autonomous post-training engineer.",
        )
    assistant_id = str(assistant.assistant_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"assistant_id": assistant_id}))
    return assistant_id


def memory_tools(
    settings: Settings, assistant_id: str, client_factory: Callable[[], Any] | None = None
) -> tuple[BaseTool, BaseTool]:
    make = _factory(settings, client_factory)

    async def remember(fact: str, kind: str = "lesson") -> str:
        """Store a durable fact (a lesson, failed config, or finding) in long-term memory."""
        async with make() as client:
            await client.add_memory(assistant_id, fact, metadata={"kind": kind})
        return "stored"

    async def recall(query: str, limit: int = 5) -> str:
        """Search long-term memory for facts relevant to the query."""
        async with make() as client:
            result = await client.search_memories(assistant_id, query, limit=limit)
        hits = [m["content"] for m in result.get("memories", [])]
        return json.dumps(hits) if hits else "no relevant memories"

    return (
        StructuredTool.from_function(coroutine=remember, name="remember", description=remember.__doc__),
        StructuredTool.from_function(coroutine=recall, name="recall", description=recall.__doc__),
    )
