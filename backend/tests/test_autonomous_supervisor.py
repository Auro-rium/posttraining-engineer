"""Behavior tests for the durable autonomous supervisor contract.

These tests intentionally use typed in-memory adapters.  They prove the
control-plane behavior without pretending that a local test is an AWS run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

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


class _StateStore(InMemoryAutonomousRunRepository):
    """Test-only state patch operation mirroring the future durable adapter."""

    def update_state(
        self,
        run_id: str,
        *,
        expected_version: int,
        updates: dict[str, Any],
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
    def analyze_failures(
        self, refs: tuple[str, ...], history: tuple[Any, ...] = ()
    ) -> tuple[FailureCluster, ...]:
        assert refs
        return (
            FailureCluster(
                cluster_id="cluster-checkout",
                failure_type="dependency_config",
                description="checkout dependency configuration recovery",
                count=1,
                evidence_refs=refs,
                evidence_class="LIVE",
            ),
        )

    def research(
        self, clusters: tuple[FailureCluster, ...], history: tuple[Any, ...] = (), **kwargs: Any
    ) -> tuple[ResearchHypothesis, ...]:
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
        self, refs: tuple[str, ...], hypotheses: tuple[ResearchHypothesis, ...] = (), **kwargs: Any
    ) -> CuratedDatasetPlan:
        return CuratedDatasetPlan(
            plan_id=f"plan-{len(refs)}",
            selected_trajectory_refs=refs,
            dataset_artifact_ref="dataset://provenance",
            record_count=len(refs),
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
    def verify_checkpoint(
        self, job: JobResult, *, run_id: str, experiment_number: int
    ) -> CheckpointArtifact:
        return CheckpointArtifact(
            artifact_id=f"checkpoint://{run_id}/{experiment_number}",
            uri=job.artifact_uri or "",
            sha256="d" * 64,
        )


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
        return EvaluationEvidence(
            evaluation=_evaluation(
                f"eval://{state.run_id}/candidate/{experiment_number}",
                score=score,
                run_number=experiment_number,
                champion=champion_id,
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
    provider: _Provider | None = None,
    target_score: float | None = None,
) -> tuple[AutonomousRunSupervisor, _Provider]:
    provider = provider or _Provider([0.5])
    store.create(_state(max_experiments=max_experiments, approved_budget_usd=budget))
    supervisor = AutonomousRunSupervisor(
        repository=store,
        objective=_Objective(),
        agents=_Agents(),
        provider=provider,
        request_factory=_Factory(),
        artifacts=_Artifacts(),
        evaluator=_Evaluator(),
        target_score=target_score,
        poll_interval_seconds=0,
        max_polls=2,
    )
    return supervisor, provider


@pytest.mark.asyncio
async def test_one_call_progresses_baseline_training_evaluation_and_promotion() -> None:
    store = _StateStore()
    supervisor, provider = _supervisor(store)
    result = await supervisor.run_optimization("run-1")
    assert result.status is AutonomousRunStatus.SUCCEEDED
    assert result.phase is RunPhase.COMPLETED
    assert result.champion_metrics["aggregate"] == pytest.approx(0.5)
    assert provider.training_submits == 1
    assert provider.evaluation_submits == 1
    assert result.experiments[0].status is ExperimentStatus.SUCCEEDED


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
    state = _state(metadata={"checkpoint_uri": "s3://artifacts/functiongemma"})
    assert AutonomousRunSupervisor._dataset_refs(state) == ()


@pytest.mark.asyncio
async def test_baseline_evidence_survives_new_supervisor_instance() -> None:
    store = _StateStore()
    supervisor, _ = _supervisor(store)
    state = store.get("run-1")
    assert state is not None
    baseline = await supervisor._ensure_baseline(state)
    restarted = AutonomousRunSupervisor(
        repository=store,
        objective=_Objective(),
        agents=_Agents(),
        provider=_Provider([0.5]),
        request_factory=_Factory(),
        artifacts=_Artifacts(),
        evaluator=_Evaluator(),
        poll_interval_seconds=0,
    )
    loaded = restarted._champion_evaluation("run-1", baseline)
    assert loaded.run_id == "eval://run-1/baseline"


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
