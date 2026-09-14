"""Behavior tests for the durable autonomous supervisor contract.

These tests intentionally use typed in-memory adapters.  They prove the
control-plane behavior without pretending that a local test is an AWS run.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from app.autonomous.agents import (
    CuratedDatasetPlan,
    FailureCluster,
    QLoRAConfig,
    ResearchHypothesis,
)
from app.autonomous.dispatcher import AutonomousRunDispatcher
from app.autonomous.models import (
    AutonomousRunState,
    AutonomousRunStatus,
    ExperimentStatus,
    RunPhase,
)
from app.autonomous.repository import InMemoryAutonomousRunRepository
from app.autonomous.supervisor import (
    AutonomousRunSupervisor,
    CheckpointArtifact,
    EvaluationEvidence,
    SupervisorBlocked,
)
from app.autonomous.telemetry import DurableTelemetryBridge
from app.posttraining.models import Evidence, EvidenceKind, EvidenceLabel
from app.posttraining.multi_run_gate import MultiRunEvaluation
from app.providers.sagemaker import (
    EvaluationJobRequest,
    JobResult,
    JobStatus,
    TrainingJobRequest,
    TransientProviderError,
)

REVISION = "a" * 40
MANIFEST = "b" * 64


def _evidence(evidence_id: str, *, score: float, run_number: int) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        kind=EvidenceKind.EVALUATION,
        label=EvidenceLabel.LIVE,
        artifact_ids=(f"artifact://{evidence_id}",),
        metrics={"aggregate": score, "checkout": score},
        verified=True,
        benchmark_id="service-recovery-v1",
        suite="AgentGym/AgentEval",
        suite_version="agent-eval-v1",
        manifest_sha256=MANIFEST,
        seed=7,
        model_id="google/functiongemma-270m-it",
    )


def _evaluation(
    eval_id: str, *, score: float, run_number: int, champion: str | None
) -> MultiRunEvaluation:
    return MultiRunEvaluation(
        run_id=eval_id,
        run_number=run_number,
        champion_run_id=champion,
        aggregate_score=score,
        environment_scores={"checkout": score},
        evidence=_evidence(eval_id, score=score, run_number=run_number),
    )


@dataclass(frozen=True)
class _Benchmark:
    evaluation: MultiRunEvaluation
    trajectory_refs: tuple[str, ...]
    artifact_ids: tuple[str, ...]
    cost_usd: float = 0.0


@dataclass(frozen=True)
class _Dataset:
    dataset_id: str
    uri: str
    sha256: str
    artifact_id: str
    cost_usd: float = 0.0


@dataclass(frozen=True)
class _CheckpointArtifactWithManifest:
    artifact_id: str
    uri: str
    sha256: str
    manifest_sha256: str


class _StateStore(InMemoryAutonomousRunRepository):
    """Test-only state patch operation mirroring the future durable adapter."""

    def update_state(
        self,
        run_id: str,
        *,
        expected_version: int,
        updates: Mapping[str, Any],
    ) -> AutonomousRunState:
        current = self.get(run_id)
        assert current is not None
        if current.version != expected_version:
            raise RuntimeError("stale state")
        next_state = AutonomousRunState.model_validate(
            current.model_copy(
                update={"version": current.version + 1, "updated_at": datetime.now(UTC), **updates}
            ).model_dump(mode="python")
        )
        self._states[run_id] = next_state  # type: ignore[attr-defined]
        return next_state


class _Objective:
    def __init__(self, *, candidate_scores: list[float] | None = None) -> None:
        self.candidate_scores = candidate_scores or [0.5, 0.6, 0.7, 0.8, 0.9]
        self.benchmarks: list[tuple[str, int]] = []
        self.datasets: list[str] = []

    def benchmark(
        self, state: AutonomousRunState, *, split: str, experiment_number: int
    ) -> _Benchmark:
        self.benchmarks.append((split, experiment_number))
        if split == "baseline":
            score, n, champion = 0.4, 0, None
            eval_id = f"eval://{state.run_id}/baseline"
        else:
            score, n, champion = 0.4, experiment_number - 1, None
            eval_id = f"eval://{state.run_id}/champion/{experiment_number}"
        return _Benchmark(
            evaluation=_evaluation(eval_id, score=score, run_number=n, champion=champion),
            trajectory_refs=(f"traj://{state.run_id}/{split}/{experiment_number}",),
            artifact_ids=(f"artifact://{state.run_id}/{split}/{experiment_number}",),
        )

    def build_dataset(
        self, state: AutonomousRunState, plan: CuratedDatasetPlan, *, experiment_number: int
    ) -> _Dataset:
        self.datasets.append(plan.plan_id)
        return _Dataset(
            dataset_id=f"dataset-{experiment_number}",
            uri=f"s3://artifacts/dataset-{experiment_number}.jsonl",
            sha256="c" * 64,
            artifact_id=f"dataset://{experiment_number}",
        )

    def verify_dataset(self, dataset: _Dataset, *, run_id: str, experiment_number: int) -> _Dataset:
        assert dataset.sha256 == "c" * 64
        return dataset


class _Agents:
    def __init__(self) -> None:
        self.analysis_calls: list[dict[str, Any]] = []
        self.research_calls: list[dict[str, Any]] = []
        self.curation_calls: list[dict[str, Any]] = []

    def analyze_failures(
        self, refs: Sequence[str], history: Sequence[Any] = (), *, evidence_class: str
    ) -> tuple[FailureCluster, ...]:
        assert refs
        self.analysis_calls.append(
            {"refs": tuple(refs), "history": tuple(history), "evidence_class": evidence_class}
        )
        return (
            FailureCluster(
                cluster_id="cluster-checkout",
                failure_type="dependency_config",
                description="checkout dependency configuration recovery",
                count=1,
                evidence_refs=tuple(refs),
                evidence_class=evidence_class,
            ),
        )

    def research(
        self,
        clusters: Sequence[FailureCluster],
        history: Sequence[Any] = (),
        *,
        run_id: str,
        experiment_number: int,
        verified_evidence_references: Sequence[str],
        verified_evidence_metadata: Mapping[str, Mapping[str, Any]],
    ) -> tuple[ResearchHypothesis, ...]:
        self.research_calls.append(
            {
                "clusters": tuple(clusters),
                "history": tuple(history),
                "run_id": run_id,
                "experiment_number": experiment_number,
                "verified_evidence_references": tuple(verified_evidence_references),
                "verified_evidence_metadata": dict(verified_evidence_metadata),
            }
        )
        return (
            ResearchHypothesis(
                hypothesis_id=f"hyp-{len(history) + 1}",
                cluster_id=clusters[0].cluster_id,
                statement="explicit verification improves service recovery",
                prediction="held-out success rises",
                falsifier="held-out success does not rise",
                evidence_refs=clusters[0].evidence_refs,
                evidence_class="EXPLANATION",
            ),
        )

    def curate(
        self,
        refs: Sequence[str],
        hypotheses: Sequence[ResearchHypothesis] = (),
        experiment_history: Sequence[Any] = (),
        *,
        failure_clusters: Sequence[FailureCluster],
        verified_trajectory_metadata: Mapping[str, Mapping[str, Any]],
    ) -> CuratedDatasetPlan:
        self.curation_calls.append(
            {
                "refs": tuple(refs),
                "hypotheses": tuple(hypotheses),
                "history": tuple(experiment_history),
                "failure_clusters": tuple(failure_clusters),
                "verified_trajectory_metadata": dict(verified_trajectory_metadata),
            }
        )
        return CuratedDatasetPlan(
            plan_id=f"plan-{len(refs)}",
            selected_trajectory_refs=tuple(refs),
            target_failure_classes=tuple(item.failure_type for item in failure_clusters),
            record_count=len(refs),
            evidence_class=next(
                iter(verified_trajectory_metadata.values())
            )["evidence_class"],
        )

    def design_qlora(self, plan: CuratedDatasetPlan, history: tuple[Any, ...] = ()) -> QLoRAConfig:
        return QLoRAConfig(
            rank=8,
            alpha=16,
            dropout=0.0,
            learning_rate=1e-4,
            epochs=1,
            sequence_length=512,
            batch_size=1,
            gradient_accumulation_steps=4,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        )


class _Provider:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.training_submits = 0
        self.evaluation_submits = 0
        self.stops: list[str] = []

    def reconcile_training(self, request: TrainingJobRequest) -> JobResult | None:
        return None

    def submit_training(self, request: TrainingJobRequest) -> JobResult:
        self.training_submits += 1
        return JobResult(
            request.job_name,
            f"train://{request.job_name}",
            JobStatus.COMPLETED,
            "s3://artifacts/checkpoint",
        )

    def get_training_status(self, job_name: str) -> JobResult:
        return JobResult(
            job_name, f"train://{job_name}", JobStatus.COMPLETED, "s3://artifacts/checkpoint"
        )

    def stop_training(self, job_name: str) -> None:
        self.stops.append(job_name)

    def reconcile_evaluation(self, request: EvaluationJobRequest) -> JobResult | None:
        return None

    def submit_evaluation(self, request: EvaluationJobRequest) -> JobResult:
        self.evaluation_submits += 1
        return JobResult(
            request.job_name,
            f"evaljob://{request.job_name}",
            JobStatus.COMPLETED,
            "s3://artifacts/eval",
        )

    def get_evaluation_status(self, job_name: str) -> JobResult:
        return JobResult(
            job_name, f"evaljob://{job_name}", JobStatus.COMPLETED, "s3://artifacts/eval"
        )

    def stop_evaluation(self, job_name: str) -> None:
        self.stops.append(job_name)


class _Factory:
    def training(
        self,
        state: AutonomousRunState,
        *,
        experiment_number: int,
        dataset: _Dataset,
        config: QLoRAConfig,
    ) -> TrainingJobRequest:
        return TrainingJobRequest(
            job_name=f"train-{state.run_id}-{experiment_number}",
            role_arn="arn:role",
            image_uri="123.dkr.ecr/trainer@sha256:" + "a" * 64,
            input_s3_uri=dataset.uri,
            output_s3_uri=f"s3://artifacts/{experiment_number}",
            instance_type="ml.g5.xlarge",
        )

    def evaluation(
        self, state: AutonomousRunState, *, experiment_number: int, candidate: CheckpointArtifact
    ) -> EvaluationJobRequest:
        return EvaluationJobRequest(
            job_name=f"eval-{state.run_id}-{experiment_number}",
            role_arn="arn:role",
            image_uri="123.dkr.ecr/evaluator@sha256:" + "b" * 64,
            input_s3_uri="s3://artifacts/sealed",
            output_s3_uri=f"s3://artifacts/eval-{experiment_number}",
            instance_type="ml.m5.xlarge",
            model_s3_uri=candidate.uri,
        )


class _Artifacts:
    def __init__(self, *, manifest_sha256: str | None = None) -> None:
        self.manifest_sha256 = manifest_sha256

    def verify_checkpoint(
        self, job: JobResult, *, run_id: str, experiment_number: int
    ) -> CheckpointArtifact:
        values = {
            "artifact_id": f"checkpoint://{run_id}/{experiment_number}",
            "uri": job.artifact_uri or "",
            "sha256": "d" * 64,
        }
        if self.manifest_sha256 is not None:
            return cast(
                CheckpointArtifact,
                _CheckpointArtifactWithManifest(
                    **values, manifest_sha256=self.manifest_sha256
                ),
            )
        return CheckpointArtifact(**values)


class _Evaluator:
    def __init__(self) -> None:
        self.calls = 0

    def read_evaluation(
        self, job: JobResult, *, state: AutonomousRunState, experiment_number: int
    ) -> EvaluationEvidence:
        self.calls += 1
        score = 0.5 + 0.1 * (self.calls - 1)
        champion_id = (
            f"eval://{state.run_id}/baseline"
            if experiment_number == 1
            else f"eval://{state.run_id}/candidate/{experiment_number - 1}"
        )
        champion_score = 0.4 if experiment_number == 1 else score - 0.1
        return EvaluationEvidence(
            evaluation=_evaluation(
                f"eval://{state.run_id}/candidate/{experiment_number}",
                score=score,
                run_number=experiment_number,
                champion=champion_id,
            ),
            champion_evaluation=_evaluation(
                champion_id,
                score=champion_score,
                run_number=experiment_number - 1,
                champion=None,
            ),
            artifact_ids=(f"eval-artifact://{experiment_number}",),
            cost_usd=0.1,
        )


def _state(run_id: str = "run-1", **updates: Any) -> AutonomousRunState:
    values: dict[str, Any] = {
        "approval_consumed": True,
        "approval_digest": "e" * 64,
        "metadata": {"checkpoint_uri": "s3://artifacts/functiongemma"},
    }
    values.update(updates)
    return AutonomousRunState(
        run_id=run_id,
        checkpoint_revision=REVISION,
        benchmark_manifest_sha256=MANIFEST,
        **values,
    )


def _supervisor(
    store: _StateStore,
    *,
    max_experiments: int = 1,
    budget: float = 25.0,
    cost_upper_bounds: Mapping[str, float] | None = None,
    phase_cost_estimates: Mapping[str, float] | None = None,
    provider: _Provider | None = None,
    target_score: float | None = None,
    objective: _Objective | None = None,
    agents: _Agents | None = None,
    artifacts: _Artifacts | None = None,
) -> tuple[AutonomousRunSupervisor, _Provider]:
    provider = provider or _Provider([0.5])
    store.create(_state(max_experiments=max_experiments, approved_budget_usd=budget))
    supervisor = AutonomousRunSupervisor(
        repository=store,
        objective=objective or _Objective(),
        agents=agents or _Agents(),
        provider=provider,
        request_factory=_Factory(),
        artifacts=artifacts or _Artifacts(),
        evaluator=_Evaluator(),
        target_score=target_score,
        poll_interval_seconds=0,
        max_polls=2,
        phase_cost_upper_bounds_usd=(
            {"training": 5.0, "evaluation": 1.0}
            if cost_upper_bounds is None
            else cost_upper_bounds
        ),
        phase_cost_estimates=phase_cost_estimates,
    )
    return supervisor, provider


@pytest.mark.asyncio
async def test_one_call_progresses_baseline_training_evaluation_and_promotion() -> None:
    store = _StateStore()
    objective = _Objective()
    supervisor, provider = _supervisor(store, objective=objective)
    result = await supervisor.run_optimization("run-1")
    assert result.status is AutonomousRunStatus.SUCCEEDED, result.stop_reason
    assert result.phase is RunPhase.COMPLETED
    assert result.champion_metrics["aggregate"] == pytest.approx(0.5)
    assert result.baseline_metrics["aggregate"] == pytest.approx(0.4)
    assert provider.training_submits == 1
    assert provider.evaluation_submits == 1
    assert result.experiments[0].status is ExperimentStatus.SUCCEEDED
    assert objective.benchmarks == [("train", 1)]


@pytest.mark.asyncio
async def test_restart_resumes_submitted_operation_without_duplicate_training() -> None:
    store = _StateStore()
    supervisor, provider = _supervisor(store)
    first = await supervisor.run_optimization("run-1")
    assert first.status is AutonomousRunStatus.SUCCEEDED
    # A second worker observing the durable terminal state is a no-op.
    second = await supervisor.run_optimization("run-1")
    assert second.status is AutonomousRunStatus.SUCCEEDED
    assert provider.training_submits == 1


@pytest.mark.asyncio
async def test_expired_lease_recovery_reconciles_persisted_training_intent() -> None:
    class _CreateAcceptedButResponseLostProvider(_Provider):
        def __init__(self) -> None:
            super().__init__([0.5])
            self.accepted_training: JobResult | None = None
            self.training_reconciliations = 0

        def submit_training(self, request: TrainingJobRequest) -> JobResult:
            self.training_submits += 1
            self.accepted_training = JobResult(
                request.job_name,
                f"train://{request.job_name}",
                JobStatus.IN_PROGRESS,
            )
            # Model SageMaker accepting CreateTrainingJob while the coordinator
            # loses the response before persisting its provider ID.
            raise TransientProviderError(
                "training create response was lost",
                job_name=request.job_name,
                operation="create_training_job",
            )

        def reconcile_training(self, request: TrainingJobRequest) -> JobResult | None:
            self.training_reconciliations += 1
            if self.accepted_training is not None:
                assert self.accepted_training.job_name == request.job_name
            return self.accepted_training

    store = _StateStore()
    store.create(
        _state(
            max_experiments=1,
            status=AutonomousRunStatus.QUEUED,
            phase=RunPhase.QUEUED,
            approved_budget_usd=25.0,
        )
    )
    provider = _CreateAcceptedButResponseLostProvider()
    objective = _Objective()
    agents = _Agents()

    def new_supervisor() -> AutonomousRunSupervisor:
        return AutonomousRunSupervisor(
            repository=store,
            objective=objective,
            agents=agents,
            provider=provider,
            request_factory=_Factory(),
            artifacts=_Artifacts(),
            evaluator=_Evaluator(),
            poll_interval_seconds=0,
            max_polls=2,
            phase_cost_upper_bounds_usd={"training": 5.0, "evaluation": 1.0},
        )

    interrupted = await new_supervisor().run_optimization("run-1")

    assert interrupted.status is AutonomousRunStatus.RUNNING
    training_intent = store.get_operation("run-1", "run-1:1:training")
    assert training_intent is not None
    assert training_intent.status.value == "INTENT"
    assert provider.accepted_training is not None
    assert training_intent.result["request"]["job_name"] == provider.accepted_training.job_name
    assert provider.training_submits == 1

    # A hard process loss leaves its durable lease in place until expiry.
    store.claim_lease(
        "run-1",
        "previous-coordinator",
        now=datetime.now(UTC) - timedelta(seconds=5),
        ttl_seconds=1,
    )
    expired = store.get("run-1")
    assert expired is not None and expired.lease_expires_at is not None
    assert expired.lease_expires_at < datetime.now(UTC)

    recovered_supervisor = new_supervisor()
    dispatcher = AutonomousRunDispatcher(
        repository=store,
        supervisor=recovered_supervisor,
        owner="restarted-coordinator",
        lease_ttl_seconds=30,
    )
    processed = await dispatcher.recover_incomplete_runs()

    recovered = store.get("run-1")
    assert processed == ["run-1"]
    assert recovered is not None
    assert recovered.status is AutonomousRunStatus.SUCCEEDED, recovered.stop_reason
    assert provider.training_reconciliations == 2
    assert provider.training_submits == 1


@pytest.mark.asyncio
async def test_promoted_checkpoint_lineage_survives_promotion_recovery() -> None:
    store = _StateStore()
    manifest_sha256 = "f" * 64
    supervisor, _ = _supervisor(
        store, artifacts=_Artifacts(manifest_sha256=manifest_sha256)
    )

    result = await supervisor.run_optimization("run-1")

    assert result.status is AutonomousRunStatus.SUCCEEDED
    assert result.champion_checkpoint_uri == "s3://artifacts/checkpoint"
    assert result.champion_checkpoint_sha256 == "d" * 64
    assert result.metadata["champion_checkpoint_artifact_id"] == "checkpoint://run-1/1"
    assert result.metadata["champion_checkpoint_manifest_sha256"] == manifest_sha256
    promotion = store.get_operation("run-1", "run-1:1:promotion")
    assert promotion is not None
    assert promotion.result["candidate_manifest_sha256"] == manifest_sha256

    recovered = supervisor._complete_promotion("run-1", "run-1:1:promotion")

    assert recovered.champion_checkpoint_uri == "s3://artifacts/checkpoint"
    assert recovered.champion_checkpoint_sha256 == "d" * 64
    assert recovered.metadata["champion_checkpoint_artifact_id"] == "checkpoint://run-1/1"
    assert recovered.metadata["champion_checkpoint_manifest_sha256"] == manifest_sha256


@pytest.mark.asyncio
async def test_max_experiments_is_hard_bounded_and_next_experiment_gets_history() -> None:
    store = _StateStore()
    supervisor, provider = _supervisor(store, max_experiments=2)
    result = await supervisor.run_optimization("run-1")
    assert result.status is AutonomousRunStatus.SUCCEEDED
    assert len(result.experiments) == 2
    assert provider.training_submits == 2


@pytest.mark.asyncio
async def test_budget_exhaustion_blocks_training_before_submission() -> None:
    store = _StateStore()
    supervisor, provider = _supervisor(store, budget=0.5)
    result = await supervisor.run_optimization("run-1")
    assert result.status is AutonomousRunStatus.BLOCKED
    assert "budget" in (result.stop_reason or "")
    assert provider.training_submits == 0


@pytest.mark.asyncio
async def test_training_worst_case_above_remaining_budget_blocks_before_submission() -> None:
    store = _StateStore()
    supervisor, provider = _supervisor(
        store,
        budget=4.99,
        cost_upper_bounds={"training": 5.0, "evaluation": 1.0},
    )

    result = await supervisor.run_optimization("run-1")

    assert result.status is AutonomousRunStatus.BLOCKED
    assert result.stop_reason == "budget exhausted"
    assert provider.training_submits == 0
    assert provider.evaluation_submits == 0


@pytest.mark.asyncio
async def test_evaluation_worst_case_above_remaining_budget_blocks_evaluation_submission() -> None:
    store = _StateStore()
    supervisor, provider = _supervisor(
        store,
        budget=10.0,
        cost_upper_bounds={"training": 5.0, "evaluation": 5.01},
        phase_cost_estimates={"training": 5.0, "evaluation": 1.0},
    )

    result = await supervisor.run_optimization("run-1")

    assert result.status is AutonomousRunStatus.BLOCKED
    assert result.stop_reason == "budget exhausted"
    assert provider.training_submits == 1
    assert provider.evaluation_submits == 0


@pytest.mark.asyncio
async def test_missing_job_cost_bound_blocks_run_before_any_provider_submission() -> None:
    store = _StateStore()
    supervisor, provider = _supervisor(
        store,
        cost_upper_bounds={"training": 5.0},
    )

    result = await supervisor.run_optimization("run-1")

    assert result.status is AutonomousRunStatus.BLOCKED
    assert result.stop_reason == "cost upper bound is unavailable for evaluation"
    assert provider.training_submits == 0
    assert provider.evaluation_submits == 0


@pytest.mark.asyncio
async def test_missing_approval_is_blocked_without_objective_side_effect() -> None:
    store = _StateStore()
    store.create(_state(approval_consumed=False, approval_digest=None))
    supervisor = AutonomousRunSupervisor(
        repository=store,
        objective=_Objective(),
        agents=_Agents(),
        provider=_Provider([0.5]),
        request_factory=_Factory(),
        artifacts=_Artifacts(),
        evaluator=_Evaluator(),
        poll_interval_seconds=0,
    )
    result = await supervisor.run_optimization("run-1")
    assert result.status is AutonomousRunStatus.BLOCKED
    assert result.stop_reason == "approval required"


@pytest.mark.asyncio
async def test_safe_stop_requested_before_next_phase_stops_without_training() -> None:
    store = _StateStore()
    supervisor, provider = _supervisor(store)
    store.update_state("run-1", expected_version=0, updates={"safe_stop_requested": True})
    result = await supervisor.run_optimization("run-1")
    assert result.status is AutonomousRunStatus.STOPPED
    assert result.stop_reason == "safe stop requested"
    assert provider.training_submits == 0


def test_missing_dataset_provenance_is_empty_not_fabricated() -> None:
    plan = CuratedDatasetPlan(
        plan_id="plan-1",
        selected_trajectory_refs=("traj://run-1/train/1",),
        target_failure_classes=("dependency_config",),
        record_count=1,
        evidence_class="LIVE",
    )
    assert "dataset_artifact_ref" not in plan.model_dump(mode="json")


@pytest.mark.asyncio
async def test_strict_agent_handoffs_use_only_verified_benchmark_provenance() -> None:
    store = _StateStore()
    agents = _Agents()
    supervisor, _ = _supervisor(store, agents=agents)

    result = await supervisor.run_optimization("run-1")

    assert result.status is AutonomousRunStatus.SUCCEEDED, result.stop_reason
    ref = "traj://run-1/train/1"
    expected_metadata = {
        ref: {
            "verified": True,
            "run_id": "run-1",
            "experiment_number": 1,
            "measurement_id": ref,
            "evidence_class": "LIVE",
        }
    }
    assert agents.analysis_calls[0]["evidence_class"] == "LIVE"
    assert agents.research_calls[0]["run_id"] == "run-1"
    assert agents.research_calls[0]["experiment_number"] == 1
    assert agents.research_calls[0]["verified_evidence_references"] == (ref,)
    assert agents.research_calls[0]["verified_evidence_metadata"] == expected_metadata
    assert agents.curation_calls[0]["failure_clusters"] == agents.research_calls[0]["clusters"]
    assert agents.curation_calls[0]["verified_trajectory_metadata"] == expected_metadata


@pytest.mark.asyncio
async def test_judgment_agent_schema_failure_retries_without_repeating_provider_work() -> None:
    store = _StateStore()
    agents = _Agents()
    objective = _Objective()
    supervisor, provider = _supervisor(store, objective=objective, agents=agents)
    original_curate = agents.curate
    curation_attempts = 0

    def flaky_curation(*args: Any, **kwargs: Any) -> CuratedDatasetPlan:
        nonlocal curation_attempts
        curation_attempts += 1
        if curation_attempts < 3:
            raise ValueError("schema mismatch")
        return original_curate(*args, **kwargs)

    agents.curate = flaky_curation  # type: ignore[method-assign]

    result = await supervisor.run_optimization("run-1")

    assert result.status is AutonomousRunStatus.SUCCEEDED
    assert curation_attempts == 3
    assert objective.benchmarks == [("train", 1)]
    assert provider.training_submits == 1
    assert provider.evaluation_submits == 1


@pytest.mark.asyncio
async def test_restart_reuses_persisted_benchmark_provenance_before_recuration() -> None:
    store = _StateStore()
    agents = _Agents()
    objective = _Objective()
    supervisor, _ = _supervisor(store, objective=objective, agents=agents)
    original_curate = agents.curate
    curation_attempts = 0

    def interrupt_first_curation(*args: Any, **kwargs: Any) -> CuratedDatasetPlan:
        nonlocal curation_attempts
        curation_attempts += 1
        if curation_attempts == 1:
            raise asyncio.CancelledError()
        return original_curate(*args, **kwargs)

    agents.curate = interrupt_first_curation  # type: ignore[method-assign]
    with pytest.raises(asyncio.CancelledError):
        await supervisor.run_optimization("run-1")

    interrupted = store.get("run-1")
    assert interrupted is not None
    assert interrupted.current_hypothesis is not None
    assert "dataset_plan" not in interrupted.current_hypothesis
    benchmark_operation = store.get_operation("run-1", "run-1:1:benchmark")
    assert benchmark_operation is not None
    assert benchmark_operation.result["trajectory_refs"] == ("traj://run-1/train/1",)

    agents.curate = original_curate  # type: ignore[method-assign]
    resumed = await supervisor.run_optimization("run-1")

    assert resumed.status is AutonomousRunStatus.SUCCEEDED
    assert objective.benchmarks == [("train", 1)]
    assert len(agents.analysis_calls) == 2
    assert len(agents.research_calls) == 1
    assert len(agents.curation_calls) == 1


@pytest.mark.asyncio
async def test_initial_champion_baseline_is_derived_from_the_paired_sealed_evaluator() -> None:
    store = _StateStore()
    objective = _Objective()
    supervisor, _ = _supervisor(store, objective=objective)

    result = await supervisor.run_optimization("run-1")

    assert result.status is AutonomousRunStatus.SUCCEEDED
    assert objective.benchmarks == [("train", 1)]
    assert result.baseline_metrics["aggregate"] == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_safe_stop_after_training_does_not_submit_evaluation() -> None:
    store = _StateStore()
    supervisor, provider = _supervisor(store)
    original = provider.submit_training

    def submit_and_request_stop(request: TrainingJobRequest) -> JobResult:
        result = original(request)
        current = store.get("run-1")
        assert current is not None
        store.update_state(
            "run-1", expected_version=current.version, updates={"safe_stop_requested": True}
        )
        return result

    provider.submit_training = submit_and_request_stop  # type: ignore[method-assign]
    result = await supervisor.run_optimization("run-1")
    assert result.status is AutonomousRunStatus.STOPPED
    assert result.stop_reason == "safe stop requested"
    assert provider.evaluation_submits == 0


@pytest.mark.asyncio
async def test_stopped_training_due_to_cancel_is_cancelled_not_failed() -> None:
    store = _StateStore()
    supervisor, provider = _supervisor(store)

    def submit_stopped(request: TrainingJobRequest) -> JobResult:
        current = store.get("run-1")
        assert current is not None
        store.update_state(
            "run-1", expected_version=current.version, updates={"cancellation_requested": True}
        )
        return JobResult(request.job_name, f"train://{request.job_name}", JobStatus.STOPPED)

    provider.submit_training = submit_stopped  # type: ignore[method-assign]
    result = await supervisor.run_optimization("run-1")
    assert result.status is AutonomousRunStatus.CANCELLED
    assert result.stop_reason == "cancellation requested"


@pytest.mark.asyncio
async def test_failed_training_job_is_failed_not_stopped() -> None:
    store = _StateStore()
    supervisor, provider = _supervisor(store)

    def submit_failed(request: TrainingJobRequest) -> JobResult:
        return JobResult(
            request.job_name,
            f"train://{request.job_name}",
            JobStatus.FAILED,
            failure_reason="trainer failed",
        )

    provider.submit_training = submit_failed  # type: ignore[method-assign]
    result = await supervisor.run_optimization("run-1")
    assert result.status is AutonomousRunStatus.FAILED


@pytest.mark.asyncio
async def test_persisted_expiry_blocks_without_verifier() -> None:
    store = _StateStore()
    supervisor, provider = _supervisor(store)
    current = store.get("run-1")
    assert current is not None
    store.update_state(
        "run-1",
        expected_version=current.version,
        updates={"approval_expires_at": datetime(2020, 1, 1, tzinfo=UTC)},
    )
    result = await supervisor.run_optimization("run-1")
    assert result.status is AutonomousRunStatus.BLOCKED
    assert result.stop_reason == "approval expired"
    assert provider.training_submits == 0


@pytest.mark.asyncio
async def test_cancel_requested_during_design_prevents_provider_submission() -> None:
    store = _StateStore()
    supervisor, provider = _supervisor(store)
    original = supervisor.agents.design_qlora

    def design_and_cancel(plan: CuratedDatasetPlan, history: tuple[Any, ...] = ()) -> QLoRAConfig:
        current = store.get("run-1")
        assert current is not None
        store.update_state(
            "run-1", expected_version=current.version, updates={"cancellation_requested": True}
        )
        return original(plan, history)

    supervisor.agents.design_qlora = design_and_cancel  # type: ignore[method-assign]
    result = await supervisor.run_optimization("run-1")
    assert result.status is AutonomousRunStatus.CANCELLED
    assert provider.training_submits == 0


@pytest.mark.asyncio
async def test_in_progress_terminal_cost_is_reconciled_exactly() -> None:
    store = _StateStore()
    supervisor, provider = _supervisor(store, budget=5.0)
    original = provider.submit_training

    def submit_in_progress(request: TrainingJobRequest) -> JobResult:
        result = original(request)
        return JobResult(
            result.job_name,
            result.provider_job_id,
            JobStatus.IN_PROGRESS,
            result.artifact_uri,
            raw_response={"cost_usd": 0.0},
        )

    provider.submit_training = submit_in_progress  # type: ignore[method-assign]
    provider.get_training_status = lambda name: JobResult(  # type: ignore[method-assign]
        name,
        f"train://{name}",
        JobStatus.COMPLETED,
        "s3://artifacts/checkpoint",
        raw_response={"actual_cost_usd": 4.5},
    )
    result = await supervisor.run_optimization("run-1")
    assert result.status is AutonomousRunStatus.BLOCKED
    assert result.stop_reason == "budget exhausted"
    assert result.spent_budget_usd == pytest.approx(4.5)


@pytest.mark.asyncio
async def test_transient_status_failure_remains_recoverable() -> None:
    store = _StateStore()
    supervisor, provider = _supervisor(store)
    original = provider.get_training_status
    attempts = 0

    def transient_once(name: str) -> JobResult:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TransientProviderError("timeout", job_name=name, operation="describe")
        return original(name)

    def submit_in_progress(request: TrainingJobRequest) -> JobResult:
        provider.training_submits += 1
        return JobResult(request.job_name, f"train://{request.job_name}", JobStatus.IN_PROGRESS)

    provider.submit_training = submit_in_progress  # type: ignore[method-assign]
    provider.get_training_status = transient_once  # type: ignore[method-assign]
    first = await supervisor.run_optimization("run-1")
    assert first.status is AutonomousRunStatus.RUNNING
    second = await supervisor.run_optimization("run-1")
    assert second.status is AutonomousRunStatus.SUCCEEDED
    assert provider.training_submits == 1


@pytest.mark.asyncio
async def test_durable_telemetry_events_are_persisted_by_supervisor() -> None:
    store = _StateStore()
    supervisor, _ = _supervisor(store)
    supervisor.telemetry = DurableTelemetryBridge(store)
    result = await supervisor.run_optimization("run-1")
    assert result.status is AutonomousRunStatus.SUCCEEDED
    events = store.list_events("run-1").items
    assert any(event.event_type == "job.submitted" for event in events)
    assert any(event.event_type == "job.completed" for event in events)
    assert any(event.event_type == "promotion.decided" for event in events)
    event_types = {event.event_type for event in events}
    assert {
        "run.started",
        "phase.started",
        "phase.completed",
        "run.completed",
    } <= event_types


@pytest.mark.asyncio
async def test_supervisor_fails_closed_on_telemetry_validation_error() -> None:
    store = _StateStore()
    supervisor, _ = _supervisor(store)

    class InvalidTelemetry:
        def emit(self, *_: Any, **__: Any) -> None:
            raise ValueError("telemetry contract rejected event")

    supervisor.telemetry = InvalidTelemetry()  # type: ignore[assignment]
    result = await supervisor.run_optimization("run-1")
    assert result.status is AutonomousRunStatus.STOPPED
    assert result.stop_reason == "durable telemetry failure"


def test_foreign_run_identity_is_rejected_even_when_prefix_matches() -> None:
    foreign = _Benchmark(
        evaluation=_evaluation(
            "eval://run-1-foreign/baseline", score=0.4, run_number=0, champion=None
        ),
        trajectory_refs=("traj://run-1-foreign/baseline/0",),
        artifact_ids=("artifact://run-1-foreign/baseline/0",),
    )
    with pytest.raises(SupervisorBlocked, match="provenance"):
        AutonomousRunSupervisor._validate_benchmark(foreign, "run-1", 0)


@pytest.mark.asyncio
async def test_dispatcher_claims_lease_and_releases_it_after_supervisor() -> None:
    store = _StateStore()
    state = _state(status=AutonomousRunStatus.QUEUED, phase=RunPhase.QUEUED)
    store.create(state)
    calls: list[str] = []

    class _StubSupervisor:
        async def run_optimization(self, run_id: str) -> AutonomousRunState:
            calls.append(run_id)
            return store.get(run_id)  # type: ignore[return-value]

    dispatcher = AutonomousRunDispatcher(
        repository=store, supervisor=_StubSupervisor(), owner="worker-a", lease_ttl_seconds=30
    )
    result = await dispatcher.dispatch_once()
    assert calls == ["run-1"]
    assert result == ["run-1"]
    assert store.get("run-1").lease_owner is None  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_dispatcher_cancellation_stops_supervisor_before_releasing_lease() -> None:
    store = _StateStore()
    store.create(_state(status=AutonomousRunStatus.QUEUED, phase=RunPhase.QUEUED))
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class _BlockingSupervisor:
        async def run_optimization(self, run_id: str) -> AutonomousRunState:
            assert run_id == "run-1"
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    dispatcher = AutonomousRunDispatcher(
        repository=store,
        supervisor=_BlockingSupervisor(),
        owner="worker-a",
        lease_ttl_seconds=30,
    )
    dispatch_task = asyncio.create_task(dispatcher.dispatch_once())
    await asyncio.wait_for(started.wait(), timeout=1)

    dispatch_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await dispatch_task

    assert cancelled.is_set()
    assert store.get("run-1").lease_owner is None  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_concurrent_dispatchers_with_same_owner_do_not_run_stale_candidate_twice() -> None:
    class _StaleScanStore(_StateStore):
        def scan_recoverable(self, **kwargs: Any) -> list[AutonomousRunState]:
            del kwargs
            state = self.get("run-1")
            return [state] if state is not None else []

    store = _StaleScanStore()
    store.create(_state(status=AutonomousRunStatus.QUEUED, phase=RunPhase.QUEUED))
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    class _BlockingSupervisor:
        async def run_optimization(self, run_id: str) -> AutonomousRunState:
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            state = store.get(run_id)
            assert state is not None
            return state

    dispatcher = AutonomousRunDispatcher(
        repository=store,
        supervisor=_BlockingSupervisor(),
        owner="same-api-process",
        lease_ttl_seconds=30,
    )
    first = asyncio.create_task(dispatcher.dispatch_once())
    await asyncio.wait_for(started.wait(), timeout=1)

    second = asyncio.create_task(dispatcher.dispatch_once())
    await asyncio.sleep(0)
    assert calls == 1

    release.set()
    await asyncio.gather(first, second)
    assert calls == 1
