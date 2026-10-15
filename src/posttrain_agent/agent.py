"""Assemble the deep agent: Backboard LLM + memory + persistence + fallback."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from deepagents import create_deep_agent

from .backboard_model import Cassette
from .config import Settings
from .domain import ExperimentBackend, LocalStubBackend, make_tools
from .memory import ensure_assistant, memory_tools
from .persistence import SEED_MEMORY, Runtime
from .resilience import build_middleware, build_model
from .subagents import build_subagents
from .web import web_tools

ORCHESTRATOR_PROMPT = """You are the coordinator of an autonomous post-training engineer.

Goal: improve a target checkpoint with at most the experiment budget you are given, and only
promote a candidate when the deterministic gate says PROMOTE.

Loop per experiment (delegate each step with the `task` tool, one specialist at a time):
1. benchmark -> 2. failure_analyst -> 3. researcher -> 4. data_curator ->
5. training_designer -> 6. training_executor -> 7. evaluator (champion AND candidate on the
same split) -> 8. champion_manager.
Plan with write_todos first. Keep specialists' JSON results in files under /work/ so later steps
can read them. Use `recall` before planning and `remember` for durable lessons. Never fabricate
numbers: if a tool result is labelled EXPLANATION, say so in your final report.
"""


async def build_agent(
    settings: Settings,
    runtime: Runtime,
    *,
    experiments: ExperimentBackend | None = None,
    client_factory: Callable[[], Any] | None = None,
    http_client_factory: Callable[[], Any] | None = None,
):
    """Return a compiled deep-agent graph wired to ``runtime``'s checkpointer and store."""
    assistant_id = await ensure_assistant(settings, client_factory)
    settings = _with_assistant(settings, assistant_id)
    extra: dict[str, Any] = {}
    if client_factory:
        extra["client_factory"] = client_factory
    if settings.cassette_path:
        extra["cassette"] = Cassette(settings.cassette_path, settings.cassette_mode)

    middleware = build_middleware(settings, **extra)
    remember, recall = memory_tools(settings, assistant_id, client_factory)
    tools = make_tools(experiments or LocalStubBackend())
    tools.update(web_tools(settings, http_client_factory))
    return create_deep_agent(
        model=build_model(settings, **extra),
        tools=[remember, recall, tools["web_search"]],
        system_prompt=ORCHESTRATOR_PROMPT,
        subagents=build_subagents(tools, (remember, recall), middleware),
        middleware=middleware,
        backend=runtime.backend,
        memory=[SEED_MEMORY],
        checkpointer=runtime.checkpointer,
        store=runtime.store,
        name="posttrain-coordinator",
    )


def _with_assistant(settings: Settings, assistant_id: str) -> Settings:
    from dataclasses import replace

    return replace(settings, assistant_id=assistant_id)
