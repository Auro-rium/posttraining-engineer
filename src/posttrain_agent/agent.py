"""Assemble the deep agent: Backboard LLM + memory + persistence + fallback + approval gate."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from deepagents import create_deep_agent

from . import budget
from .backboard_model import Cassette
from .config import Settings
from .domain import ExperimentBackend, LocalStubBackend, make_tools
from .hf import hf_tools
from .memory import ensure_assistant, memory_tools
from .nebius import NebiusBackend
from .persistence import SEED_MEMORY, Runtime
from .resilience import build_middleware, build_model
from .subagents import build_subagents
from .telemetry import telemetry_tool
from .web import web_tools

ORCHESTRATOR_PROMPT = """You are the coordinator of an autonomous post-training engineer.

Goal: improve a target checkpoint within the experiment and cost budget you are given, and only
promote a candidate when the deterministic gate says PROMOTE.

Loop per experiment (delegate each step with the `task` tool, one specialist at a time):
1. benchmark -> 2. failure_analyst -> 3. researcher -> 4. data_curator ->
5. training_designer (must call estimate_budget) -> 6. training_executor (a human approves the
job before it launches; if rejected, revise the plan, do not retry unchanged) ->
7. evaluator (champion AND candidate on the same split) -> 8. champion_manager.
Plan with write_todos first. Keep specialists' JSON results in files under /work/ so later steps
can read them. Use `recall` before planning and `remember` for durable lessons. Use
`read_telemetry` when calls are slow or failing. Never fabricate numbers: if a tool result is
labelled EXPLANATION, say so in your final report.
"""


def default_backend(settings: Settings) -> ExperimentBackend:
    """Nebius Token Factory when a key is configured, otherwise the labelled simulation."""
    return NebiusBackend(settings) if settings.nebius_api_key else LocalStubBackend()


def approval_description(settings: Settings) -> Callable[..., str]:
    """Text a human sees before approving a training job, including the cost estimate."""

    def describe(tool_call: dict[str, Any], state: Any, runtime: Any) -> str:
        args = tool_call.get("args", {})
        try:
            est = budget.estimate(settings, args.get("dataset_ref", ""),
                                  args.get("hyperparameters", {}))
        except Exception as exc:  # noqa: BLE001 - never block the prompt on a formatting error
            est = {"error": str(exc)}
        return ("Approve launching a fine-tuning job?\n"
                f"base model: {args.get('base_model') or settings.base_model}\n"
                f"dataset: {args.get('dataset_ref')}\n"
                f"hyperparameters: {json.dumps(args.get('hyperparameters'), sort_keys=True)}\n"
                f"budget estimate: {json.dumps(est, sort_keys=True)}")

    return describe


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
    settings = replace(settings, assistant_id=assistant_id)
    extra: dict[str, Any] = {}
    if client_factory:
        extra["client_factory"] = client_factory
    if settings.cassette_path:
        extra["cassette"] = Cassette(settings.cassette_path, settings.cassette_mode)

    middleware = build_middleware(settings, **extra)
    remember, recall = memory_tools(settings, assistant_id, client_factory)
    tools = make_tools(experiments or default_backend(settings), settings)
    tools.update(web_tools(settings, http_client_factory))
    tools.update(hf_tools(settings, http_client_factory))
    tools["read_telemetry"] = telemetry_tool(settings.state_dir / "telemetry.jsonl")

    interrupt_on = None
    if settings.require_approval:
        interrupt_on = {"launch_training": {
            "allowed_decisions": ["approve", "reject"],
            "description": approval_description(settings),
        }}
    return create_deep_agent(
        model=build_model(settings, **extra),
        tools=[remember, recall, tools["web_search"], tools["read_telemetry"]],
        system_prompt=ORCHESTRATOR_PROMPT,
        subagents=build_subagents(tools, (remember, recall), middleware),
        middleware=middleware,
        backend=runtime.backend,
        memory=[SEED_MEMORY],
        interrupt_on=interrupt_on,
        checkpointer=runtime.checkpointer,
        store=runtime.store,
        name="posttrain-coordinator",
    )
