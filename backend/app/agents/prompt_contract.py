# ruff: noqa: E501
"""Shared contracts for the reasoning agents used by the live run.

The post-training target is FunctionGemma, while the agents that plan, inspect,
and coordinate the run reason with one fixed Bedrock model.  Keeping the model
and prompt contract here prevents a factory from quietly drifting to a
different model or from emitting an untraceable system prompt.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

NEMOTRON_MODEL_ID: Final[str] = "nvidia.nemotron-super-3-120b"
PROMPT_CONTRACT_VERSION: Final[str] = "nemotron-120b-contract-v2"
PROMPT_LIBRARY_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "prompts"
_LIBRARY_SECTIONS: Final[tuple[str, ...]] = (
    "MISSION",
    "INPUT CONTRACT",
    "OUTPUT CONTRACT",
    "BOUNDED CREATIVITY",
)


@dataclass(frozen=True)
class AgentPromptContract:
    """Immutable prompt and provenance contract for one specialist."""

    agent_key: str
    prompt_file: str

    def _sections(self) -> dict[str, str]:
        """Load and validate this role's reviewable markdown contract."""

        path = PROMPT_LIBRARY_DIR / self.prompt_file
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"prompt library file unavailable: {path}") from exc
        if not source.endswith("\n"):
            raise RuntimeError(f"prompt library file must end with a newline: {path}")
        heading = re.match(r"^# ([^\n]+)\n", source)
        if heading is None or heading.group(1).strip() != self.agent_key:
            raise RuntimeError(f"prompt library heading does not match {self.agent_key}: {path}")
        matches = list(re.finditer(r"^## (.+?)\s*$", source, flags=re.MULTILINE))
        names = tuple(match.group(1).strip() for match in matches)
        if names != _LIBRARY_SECTIONS:
            raise RuntimeError(
                f"prompt library sections for {self.agent_key} must be {_LIBRARY_SECTIONS}; got {names}"
            )
        sections: dict[str, str] = {}
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
            value = source[start:end].strip()
            if not value:
                raise RuntimeError(f"prompt library section is empty: {path} ({names[index]})")
            sections[names[index]] = value
        return sections

    @property
    def mission(self) -> str:
        return self._sections()["MISSION"]

    @property
    def inputs(self) -> str:
        return self._sections()["INPUT CONTRACT"]

    @property
    def outputs(self) -> str:
        return self._sections()["OUTPUT CONTRACT"]

    @property
    def creative_lane(self) -> str:
        return self._sections()["BOUNDED CREATIVITY"]

    def render(self) -> str:
        """Render an explicit, auditable system prompt."""

        return f"""You are {self.agent_key}, a specialist in an autonomous FunctionGemma post-training run.

MISSION
{self.mission}

OPERATING CONTEXT
- The target being improved is FunctionGemma; your reasoning model is NVIDIA Nemotron Super 3 120B.
- The deterministic coordinator and provider adapters are authoritative for state, measurements, job IDs,
  artifacts, approvals, budgets, and promotion. You are an analyst/dispatcher, never the source of truth.
- Work only on the current run and phase. Preserve run_id, run_number, parent_champion_id, suite, version,
  seed, and manifest_hash exactly as supplied. Never silently repair missing context.

AUTONOMOUS EXECUTION LOOP
- First inspect the complete typed context, current phase, prior handoff, budget, and available tool results.
- Form a short plan with the smallest safe next action; prefer idempotent reads before writes.
- Use only injected tools for AWS, objective-worker, artifact, or registry operations. Never emulate a tool result.
- After every action, verify the returned status, identifiers, hashes, and provenance against the request.
- Persist or hand off verified references before moving to the next phase. If a phase cannot be verified, stop.
- Make progress independently within your role, but do not cross role ownership or wait for a human when a safe
  adapter-backed action is available. Escalate only missing authority, missing inputs, policy conflicts, or provider failure.
- Reuse an existing verified checkpoint, dataset, role, table, or bucket only when its immutable identifier and
  provenance match this run. Never create duplicate resources or silently reuse a mismatched artifact.

KNOWLEDGE AND REASONING STANDARD
- Apply current knowledge of SageMaker jobs, S3 versioning, DynamoDB conditional writes, QLoRA, AgentGym/AgentEval,
  tool-calling failure modes, and reproducible ML experiments to interpret the supplied facts.
- Separate OBSERVED facts, INFERRED explanations, and UNKNOWN values in your reasoning and output fields.
- Prefer causal, falsifiable explanations over generic advice. State the exact observation that would disprove a claim.
- Check units, ranges, seed/suite/version alignment, artifact hashes, and cost arithmetic before recommending action.
- When multiple safe paths exist, rank them by expected information gain, reversibility, latency, and cost.
- Never let domain knowledge override provider responses, typed contracts, approval requirements, or deterministic gates.

INPUT CONTRACT
{self.inputs}

OUTPUT CONTRACT
{self.outputs}
- Return one JSON object, with no markdown wrapper and no extra keys.
- Every claim must point to an input field or an adapter/provider artifact reference.
- If an input is absent, malformed, contradictory, or unverifiable, return status BLOCKED with a concise
  reason and required_inputs array. Do not guess a value to make progress.

EVIDENCE AND SAFETY RULES
- Allowed evidence labels are LIVE, PRIOR_VERIFIED_RUN, and EXPLANATION. Use LIVE only for measurements
  returned by the current provider execution; use PRIOR_VERIFIED_RUN only for a previously verified artifact;
  use EXPLANATION for reasoning that is not measurement evidence.
- Never invent a trajectory, metric, score, provider job ID, checkpoint URI, dataset URI, approval token,
  timestamp, cost, latency, or completion status.
- Never turn EXPLANATION into training data or promotion evidence.
- Never reveal, copy, summarize, or route sealed held-out inputs, prompts, completions, or raw trajectories
  into telemetry, training data, or another agent's context. Refer to sealed artifacts by opaque IDs only.
- Do not call a provider, mutate AWS resources, promote a checkpoint, or approve a run unless an injected tool
  explicitly authorizes that operation and returns a verifiable result.
- Deterministic code, not your prose or preference, decides the promotion gate.

BOUNDED CREATIVITY
{self.creative_lane}
Generate crisp alternatives, counterexamples, and testable next steps when the mission permits. Creativity
must remain falsifiable, budget-aware, and provenance-preserving; it may not fabricate evidence or bypass a gate.

HANDOFF DISCIPLINE
Use stable IDs supplied by the coordinator. Keep numerical precision from provider results. State what you
observed, what you inferred, and what remains unknown. A safe BLOCKED result is better than a plausible story.
"""

    @property
    def prompt(self) -> str:
        return self.render()

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()

    def metadata(self) -> dict[str, str]:
        """Return safe provenance fields for manifests and telemetry."""

        return {
            "agent_key": self.agent_key,
            "model_id": NEMOTRON_MODEL_ID,
            "prompt_version": PROMPT_CONTRACT_VERSION,
            "prompt_sha256": self.prompt_sha256,
            "prompt_file": self.prompt_file,
        }


_COMMON_INPUTS = """Required fields: run_id (string), run_number (integer 1..5), suite (string),
suite_version (string), seed (integer), manifest_hash (64-character SHA-256 string), and phase (enum).
Additional fields are listed in the mission-specific contract below. Treat opaque artifact references as
references; do not load or echo sealed content unless an authorized adapter requires it."""


_CONTRACTS: Mapping[str, AgentPromptContract] = {
    "BenchmarkAgent": AgentPromptContract("BenchmarkAgent", "benchmark_agent.md"),
    "FailureAnalystAgent": AgentPromptContract("FailureAnalystAgent", "failure_analyst_agent.md"),
    "ResearchAgent": AgentPromptContract("ResearchAgent", "research_agent.md"),
    "DataCuratorAgent": AgentPromptContract("DataCuratorAgent", "data_curator_agent.md"),
    "TrainingDesignerAgent": AgentPromptContract("TrainingDesignerAgent", "training_designer_agent.md"),
    "TrainingExecutorAgent": AgentPromptContract("TrainingExecutorAgent", "training_executor_agent.md"),
    "EvalAgent": AgentPromptContract("EvalAgent", "eval_agent.md"),
    "ChampionManagerAgent": AgentPromptContract("ChampionManagerAgent", "champion_manager_agent.md"),
}


def get_prompt_contract(agent_key: str) -> AgentPromptContract:
    """Get the immutable contract for a registered specialist."""

    try:
        return _CONTRACTS[agent_key]
    except KeyError as exc:
        raise ValueError(f"unknown post-training agent: {agent_key}") from exc


def resolve_nemotron_model(model: str | None = None) -> str:
    """Resolve the only permitted reasoning model, failing closed on overrides."""

    if model is not None and model != NEMOTRON_MODEL_ID:
        raise ValueError(
            "all post-training reasoning agents must use "
            f"{NEMOTRON_MODEL_ID}; received {model!r}"
        )
    return NEMOTRON_MODEL_ID


def resolve_agent_model(model: str | None = None, *, model_provider: Any = None) -> Any:
    """Resolve the pinned model id or an explicitly configured provider.

    Strands accepts either a Bedrock model instance or a model id.  Live AWS
    startup passes a provider built with an explicit SigV4 boto session so the
    default string shortcut cannot accidentally select bearer authentication.
    """

    resolved = resolve_nemotron_model(model)
    if model_provider is None:
        return resolved
    provider_model_id = getattr(model_provider, "model_id", None)
    if provider_model_id != resolved:
        raise ValueError(
            "the configured Bedrock provider must use "
            f"{resolved}; received {provider_model_id!r}"
        )
    return getattr(model_provider, "model", model_provider)


def agent_prompt_metadata(agent_key: str) -> dict[str, str]:
    """Return safe prompt/model provenance for telemetry and manifests."""

    return get_prompt_contract(agent_key).metadata()


AGENT_KEYS: Final[tuple[str, ...]] = tuple(_CONTRACTS)
