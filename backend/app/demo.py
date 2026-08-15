"""Credential-free deterministic adapters for development and judge walkthroughs.

Local artifacts are always labelled ``EXPLANATION`` and never qualify as cloud
provenance.  The adapter exercises the complete control flow without presenting
generated metrics or job identifiers as evidence from a real training run.
"""

from __future__ import annotations

from hashlib import sha256

from app.agents import CuratedDataset, DecisionProvider
from app.models import (
    ArtifactKind,
    ArtifactRef,
    Citation,
    DatasetManifest,
    DatasetSplit,
    EvaluationReport,
    EvidenceLabel,
    Experiment,
    FailureCluster,
    Hypothesis,
    JobStatus,
    QLoRAConfig,
    RunState,
    SFTExample,
    ToolCall,
    TrainingResult,
    Trajectory,
    TrajectoryStep,
)


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _artifact(kind: ArtifactKind, name: str) -> ArtifactRef:
    body = f"deterministic-local-demo:{kind.value}:{name}"
    return ArtifactRef(
        kind=kind,
        uri=f"demo://{kind.value}/{name}",
        sha256=_digest(body),
        size_bytes=len(body.encode("utf-8")),
    )


class LocalDemoDecisionProvider(DecisionProvider):
    """Repeatable provider for tests and UI-free local API exploration."""

    async def benchmark(self, state: RunState) -> list[Trajectory]:
        return [
            Trajectory(
                trajectory_id=f"{state.run_id}-constraint-failure",
                task_id="train-webshop-001",
                split=DatasetSplit.TRAIN,
                model_version=state.champion.version,
                steps=[
                    TrajectoryStep(
                        index=0,
                        observation="Find a waterproof hiking shoe under $80",
                        action=ToolCall(name="search", arguments={"keywords": "hiking shoe"}),
                        reward=0.0,
                    )
                ],
                reward=0.0,
                success=False,
            ),
            Trajectory(
                trajectory_id=f"{state.run_id}-success",
                task_id="train-webshop-002",
                split=DatasetSplit.TRAIN,
                model_version=state.champion.version,
                steps=[
                    TrajectoryStep(
                        index=0,
                        observation="Find a red cotton shirt",
                        action=ToolCall(name="search", arguments={"keywords": "red cotton shirt"}),
                        reward=1.0,
                    )
                ],
                reward=1.0,
                success=True,
            ),
        ]

    async def analyze_failures(
        self, state: RunState, trajectories: list[Trajectory]
    ) -> list[FailureCluster]:
        failed = [trajectory for trajectory in trajectories if not trajectory.success]
        if not failed:
            # A cloud provider resolves the benchmark artifact after a process
            # restart; local mode requires the in-process deterministic benchmark.
            raise ValueError("local failure analysis requires benchmark trajectories")
        return [
            FailureCluster(
                label="constraint_dropping",
                description="Search reformulation omits price or attribute constraints.",
                trajectory_ids=[trajectory.trajectory_id for trajectory in failed],
                frequency=len(failed),
            )
        ]

    async def form_hypothesis(self, state: RunState) -> Hypothesis:
        if not state.failure_clusters:
            raise ValueError("a measured failure cluster is required")
        cluster = state.failure_clusters[0]
        attempt = state.experiments_used + 1
        return Hypothesis(
            failure_cluster_id=cluster.cluster_id,
            statement=(
                "Verified constraint-preserving search actions will improve WebShop success "
                f"(bounded attempt {attempt})."
            ),
            expected_improvement="At least five percentage points on held-out success.",
            data_strategy="Repair train-side search actions and admit only replay improvements.",
            citations=[
                Citation(
                    document_id="functiongemma-docs",
                    chunk_id="tool-call-format",
                    title="FunctionGemma tool calling format",
                    source_uri="https://ai.google.dev/gemma/docs/functiongemma",
                    excerpt="Use the target model's structured tool-call format.",
                    score=1.0,
                )
            ],
        )

    async def curate_dataset(self, state: RunState) -> CuratedDataset:
        if not state.failure_clusters:
            raise ValueError("curation requires failure evidence")
        source_id = state.failure_clusters[0].trajectory_ids[0]
        example = SFTExample(
            source_trajectory_id=source_id,
            source_step_index=0,
            observation="Find a waterproof hiking shoe under $80",
            target_action=ToolCall(
                name="search",
                arguments={"keywords": "waterproof hiking shoe under $80"},
            ),
            verified=True,
            verifier_reward_before=0.0,
            verifier_reward_after=0.5,
        )
        name = f"{state.run_id}-attempt-{state.experiments_used + 1}"
        manifest = DatasetManifest(
            artifact=_artifact(ArtifactKind.DATASET, name),
            example_count=1,
        )
        return CuratedDataset(manifest=manifest, examples=(example,))

    async def design_training(self, state: RunState) -> QLoRAConfig:
        if state.experiments_used == 0:
            return QLoRAConfig(rank=8, learning_rate=1e-4, epochs=2, dropout=0.0)
        return QLoRAConfig(rank=16, learning_rate=1e-4, epochs=3, dropout=0.05)

    async def launch_training(self, state: RunState, experiment: Experiment) -> TrainingResult:
        name = experiment.experiment_id
        return TrainingResult(
            job_id=f"local-explanation-{name}",
            status=JobStatus.SUCCEEDED,
            checkpoint=_artifact(ArtifactKind.CHECKPOINT, name),
            logs=_artifact(ArtifactKind.TRAINING_LOG, name),
            duration_seconds=1.0,
        )

    async def evaluate(self, state: RunState, experiment: Experiment) -> EvaluationReport:
        candidate_success = 0.40 if state.experiments_used == 1 else 0.43
        return EvaluationReport(
            artifact=_artifact(ArtifactKind.EVALUATION_REPORT, experiment.experiment_id),
            task_count=20,
            champion_success=state.champion.success,
            candidate_success=candidate_success,
            champion_regression_success=state.champion.regression_success,
            candidate_regression_success=state.champion.regression_success - 0.01,
            champion_action_validity=state.champion.action_validity,
            candidate_action_validity=state.champion.action_validity,
            paired_improvement_positive=True,
            provenance_complete=False,
            evidence_label=EvidenceLabel.EXPLANATION,
        )


async def verify_demo(provider: DecisionProvider, state: RunState) -> EvaluationReport:
    """Evaluate the latest candidate without mutating run state."""

    if not state.experiments:
        raise ValueError("demo verification requires at least one candidate experiment")
    return await provider.evaluate(state, state.experiments[-1])
