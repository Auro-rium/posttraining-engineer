"""Domain agents used by the bounded autonomous research loop.

The classes in this module are deliberately thin: they give each logical role a
single responsibility while leaving Gemini/ADK, WebShop, and Vertex operations
behind an injectable provider.  This makes local demonstrations deterministic
without pretending that simulated work is a cloud result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from app.models import (
        DatasetManifest,
        EvaluationReport,
        Experiment,
        FailureCluster,
        Hypothesis,
        PromotionDecision,
        QLoRAConfig,
        RunState,
        SFTExample,
        TrainingResult,
        Trajectory,
    )


ALLOWED_LORA_RANKS = frozenset({8, 16, 32})
ALLOWED_LEARNING_RATES = frozenset({5e-5, 1e-4, 2e-4})
ALLOWED_EPOCHS = frozenset({2, 3, 5})
ALLOWED_DROPOUTS = frozenset({0.0, 0.05})


class AgentContractError(ValueError):
    """Raised when a provider returns an artifact that violates a safety boundary."""


@dataclass(frozen=True, slots=True)
class CuratedDataset:
    """A manifest plus the in-memory verifier evidence used to admit it."""

    manifest: DatasetManifest
    examples: tuple[SFTExample, ...]


class DecisionProvider(Protocol):
    """Cloud or deterministic-local implementation of specialist operations."""

    async def benchmark(self, state: RunState) -> list[Trajectory]: ...

    async def analyze_failures(
        self, state: RunState, trajectories: list[Trajectory]
    ) -> list[FailureCluster]: ...

    async def form_hypothesis(self, state: RunState) -> Hypothesis: ...

    async def curate_dataset(self, state: RunState) -> CuratedDataset: ...

    async def design_training(self, state: RunState) -> QLoRAConfig: ...

    async def launch_training(self, state: RunState, experiment: Experiment) -> TrainingResult: ...

    async def evaluate(self, state: RunState, experiment: Experiment) -> EvaluationReport: ...


@dataclass(frozen=True, slots=True)
class BenchmarkRunner:
    provider: DecisionProvider

    async def run(self, state: RunState) -> list[Trajectory]:
        trajectories = await self.provider.benchmark(state)
        if not trajectories:
            raise AgentContractError("benchmark must produce at least one trajectory")
        return trajectories


@dataclass(frozen=True, slots=True)
class FailureAnalyst:
    provider: DecisionProvider

    async def run(self, state: RunState, trajectories: list[Trajectory]) -> list[FailureCluster]:
        clusters = await self.provider.analyze_failures(state, trajectories)
        if not clusters:
            raise AgentContractError("failure analysis must produce at least one cluster")
        return clusters


@dataclass(frozen=True, slots=True)
class ResearchAgent:
    provider: DecisionProvider

    async def run(self, state: RunState) -> Hypothesis:
        hypothesis = await self.provider.form_hypothesis(state)
        if not hypothesis:
            raise AgentContractError("research must produce one hypothesis")
        return hypothesis


@dataclass(frozen=True, slots=True)
class DataCurator:
    provider: DecisionProvider

    async def run(self, state: RunState) -> DatasetManifest:
        curated = await self.provider.curate_dataset(state)
        if not curated.examples:
            raise AgentContractError("curated dataset must contain examples")
        if any(
            not example.verified or example.verifier_reward_after <= example.verifier_reward_before
            for example in curated.examples
        ):
            raise AgentContractError("unverified SFT repairs cannot enter the training dataset")
        if curated.manifest.example_count != len(curated.examples):
            raise AgentContractError("dataset manifest count does not match verified examples")
        return curated.manifest


@dataclass(frozen=True, slots=True)
class TrainingDesigner:
    provider: DecisionProvider

    async def run(self, state: RunState) -> QLoRAConfig:
        config = await self.provider.design_training(state)
        validate_qlora_config(config)
        used_configs = {
            experiment.config.model_dump_json()
            for experiment in state.experiments
            if getattr(experiment, "config", None) is not None
        }
        if config.model_dump_json() in used_configs:
            raise AgentContractError("duplicate QLoRA configurations are not allowed")
        return config


@dataclass(frozen=True, slots=True)
class TrainingExecutor:
    provider: DecisionProvider

    async def run(self, state: RunState, experiment: Experiment) -> TrainingResult:
        return await self.provider.launch_training(state, experiment)


@dataclass(frozen=True, slots=True)
class EvaluationAgent:
    provider: DecisionProvider

    async def run(self, state: RunState, experiment: Experiment) -> EvaluationReport:
        return await self.provider.evaluate(state, experiment)


@dataclass(frozen=True, slots=True)
class ChampionManager:
    """Logical role wrapper around the deterministic, non-LLM promotion gate."""

    def run(self, evaluation: EvaluationReport, provenance_complete: bool) -> PromotionDecision:
        from app.orchestrator import decide_promotion

        return decide_promotion(evaluation, provenance_complete=provenance_complete)


def validate_qlora_config(config: QLoRAConfig) -> None:
    """Apply the server-side experiment whitelist independently of model validation."""

    if config.rank not in ALLOWED_LORA_RANKS:
        raise AgentContractError(f"rank {config.rank} is outside the QLoRA search space")
    if config.learning_rate not in ALLOWED_LEARNING_RATES:
        raise AgentContractError("learning rate is outside the QLoRA search space")
    if config.epochs not in ALLOWED_EPOCHS:
        raise AgentContractError(f"epochs {config.epochs} is outside the QLoRA search space")
    if config.dropout not in ALLOWED_DROPOUTS:
        raise AgentContractError(f"dropout {config.dropout} is outside the QLoRA search space")


AGENT_NAMES = (
    "benchmark_runner",
    "failure_analyst",
    "research_agent",
    "data_curator",
    "training_designer",
    "training_executor",
    "evaluation_agent",
    "champion_manager",
)
