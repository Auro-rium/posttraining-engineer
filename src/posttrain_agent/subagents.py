"""Role contracts -> deepagents subagents."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import resources
from typing import Any

from deepagents import SubAgent

PREAMBLE = """You are one specialist in an autonomous post-training pipeline.
Rules that always apply: held-out inputs are sealed; never invent trajectories, metrics,
provider IDs or promotion decisions; copy evidence labels (LIVE, PRIOR_VERIFIED_RUN,
EXPLANATION) exactly from tool results; finish with the single JSON object your OUTPUT
CONTRACT describes.

"""

# role -> (prompt file, description, tool names, wants long-term memory)
ROLES: dict[str, tuple[str, str, tuple[str, ...], bool]] = {
    "benchmark": ("benchmark.md", "Benchmarks a checkpoint on train-side tasks.",
                  ("run_benchmark",), False),
    "failure_analyst": ("failure_analyst.md", "Clusters benchmark and training failures by behaviour.",
                        ("read_training_logs",), False),
    "researcher": ("research.md", "Turns failure clusters into falsifiable hypotheses; looks up papers and Hub metadata.",
                   ("web_search", "web_extract", "hf_search", "hf_model_info", "hf_dataset_info"), True),
    "data_curator": ("data_curator.md", "Builds and verifies SFT datasets from verified trajectories.",
                     ("generate_game_sft", "verify_dataset", "save_dataset", "inspect_dataset",
                      "check_chat_template", "hf_dataset_info"), False),
    "training_designer": ("training_designer.md", "Chooses bounded LoRA hyperparameters and checks the budget.",
                          ("validate_hparams", "estimate_budget", "web_search", "hf_model_info"), False),
    "training_executor": ("training_executor.md", "Submits one human-approved fine-tuning job and monitors it.",
                          ("launch_training", "wait_for_training", "read_training_logs"), False),
    "evaluator": ("eval.md", "Evaluates champion and candidate on sealed splits.",
                  ("evaluate_checkpoint",), False),
    "champion_manager": ("champion_manager.md", "Runs the deterministic promotion gate over the eval ledger and explains it.",
                         ("run_gate",), False),
}


def load_prompt(name: str) -> str:
    return resources.files("posttrain_agent.prompts").joinpath(name).read_text()


def build_subagents(
    tools: dict[str, Callable[..., Any]],
    memory: Sequence[Any] = (),
    middleware: Sequence[Any] = (),
    models: dict[str, Any] | None = None,
    disable_general_purpose: bool = False,
) -> list[SubAgent]:
    """``memory`` is (remember, recall); researchers get both, everyone else just recall."""
    remember, recall = memory if memory else (None, None)
    specs: list[SubAgent] = []
    for name, (file, description, tool_names, wants_memory) in ROLES.items():
        role_tools: list[Any] = [tools[t] for t in tool_names]
        if recall is not None:
            role_tools.append(recall)
            if wants_memory:
                role_tools.append(remember)
        spec = SubAgent(
            name=name,
            description=description,
            system_prompt=PREAMBLE + load_prompt("mission.md") + "\n" + load_prompt(file),
            tools=role_tools,
            middleware=list(middleware),
        )
        if models and name in models:  # else the subagent inherits the coordinator's model
            spec["model"] = models[name]
        specs.append(spec)
    if disable_general_purpose:
        # an explicit spec overrides deepagents' built-in all-tools general-purpose subagent; the
        # concurrency middleware also refuses `task` calls that name it
        specs.append(SubAgent(
            name="general-purpose",
            description="DISABLED. Do not use; delegate to the named specialists.",
            system_prompt=("This subagent is disabled. Reply only with "
                           '{"status": "BLOCKED", "errors": ["general-purpose is disabled"]}.'),
            tools=[recall] if recall is not None else [],
            middleware=list(middleware)))
    return specs
