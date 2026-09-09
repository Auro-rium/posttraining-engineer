"""Durable, fail-closed autonomous post-training supervisor.

The supervisor is deliberately adapter driven.  It owns ordering, durable
state, bounded policy, and reconciliation; objective workers, Nemotron, S3,
and SageMaker remain injected execution boundaries.  This keeps local tests
honest while allowing the API layer to wire real AWS implementations.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from typing import Any, Protocol, cast

from app.autonomous.agents import (
    CuratedDatasetPlan,
    FailureCluster,
    QLoRAConfig,
    ResearchHypothesis,
)
from app.autonomous.models import (
    AutonomousRunState,
    AutonomousRunStatus,
    ExperimentRecord,
    ExperimentStatus,
    RunOperation,
    RunOperationStatus,
    RunPhase,
)
from app.autonomous.policy import (
    PolicyAction,
    PolicyDecision,
    canonical_job_name,
    canonical_operation_key,
    canonical_request_hash,
    decide_run_control,
)
from app.posttraining.multi_run_gate import (
    MultiRunEvaluation,
    MultiRunPromotionGate,
)
from app.providers.sagemaker import EvaluationJobRequest, JobResult, JobStatus, TrainingJobRequest


class SupervisorError(RuntimeError):
    """Base error for the durable supervisor."""


class SupervisorBlocked(SupervisorError):
    """The run cannot start without a missing or invalid prerequisite."""


class SupervisorProviderFailure(SupervisorError):
    """A provider operation failed and the run must fail closed."""


class SupervisorStopReason(StrEnum):
    APPROVAL_REQUIRED = "approval required"
    READINESS_BLOCKED = "readiness blocked"
    BUDGET_EXHAUSTED = "budget exhausted"
    MAX_EXPERIMENTS = "maximum experiments reached"
    CANCELLATION = "cancellation requested"
    SAFE_STOP = "safe stop requested"
    APPROVAL_EXPIRED = "approval expired"
    PROVIDER_FAILURE = "provider failure"


@dataclass(frozen=True, slots=True)
class BenchmarkEvidence:
    """Verified objective output; no score is accepted without evidence."""

    evaluation: MultiRunEvaluation
    trajectory_refs: tuple[str, ...]
    artifact_ids: tuple[str, ...]
    cost_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class DatasetArtifact:
    dataset_id: str
    uri: str
    sha256: str
    artifact_id: str
    cost_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class CheckpointArtifact:
    artifact_id: str
    uri: str
    sha256: str


@dataclass(frozen=True, slots=True)
class EvaluationEvidence:
    evaluation: MultiRunEvaluation
    artifact_ids: tuple[str, ...]
    cost_usd: float = 0.0


class DurableRunStore(Protocol):
    def get(self, run_id: str) -> AutonomousRunState | None: ...

    def transition(
        self,
        run_id: str,
        *,
        expected_version: int,
        status: AutonomousRunStatus,
        phase: RunPhase,
        reason: str,
        event_type: str = "state.transitioned",
        metadata: Mapping[str, str] | None = None,
    ) -> AutonomousRunState: ...

    def update_state(
        self,
        run_id: str,
        *,
        expected_version: int,
        updates: Mapping[str, Any],
    ) -> AutonomousRunState: ...

    def put_operation_intent(self, operation: RunOperation) -> RunOperation: ...

    def get_operation(
        self, run_id: str, operation_key: str | None = None
    ) -> RunOperation | None: ...

    def record_operation_result(
        self,
        run_id: str,
        operation_key: str,
        *,
        provider_id: str | None = None,
        status: RunOperationStatus | str,
        result: Mapping[str, Any] | None = None,
    ) -> RunOperation: ...

    def add_experiment(self, run_id: str, experiment: ExperimentRecord) -> ExperimentRecord: ...


class ObjectiveAdapter(Protocol):
    def benchmark(
        self, state: AutonomousRunState, *, split: str, experiment_number: int
    ) -> BenchmarkEvidence | Awaitable[BenchmarkEvidence]: ...

    def build_dataset(
        self,
        state: AutonomousRunState,
        plan: CuratedDatasetPlan,
        *,
        experiment_number: int,
    ) -> DatasetArtifact | Awaitable[DatasetArtifact]: ...

    def verify_dataset(
        self, dataset: DatasetArtifact, *, run_id: str, experiment_number: int
    ) -> DatasetArtifact | Awaitable[DatasetArtifact]: ...


class AgentAdapter(Protocol):
    def analyze_failures(
        self, trajectory_references: Sequence[str], experiment_history: Sequence[Any] = ()
    ) -> Sequence[FailureCluster] | Awaitable[Sequence[FailureCluster]]: ...

    def research(
        self,
        failure_clusters: Sequence[FailureCluster],
        experiment_history: Sequence[Any] = (),
        **kwargs: Any,
    ) -> Sequence[ResearchHypothesis] | Awaitable[Sequence[ResearchHypothesis]]: ...

    def curate(
        self,
        verified_trajectory_references: Sequence[str],
        hypotheses: Sequence[ResearchHypothesis] = (),
        experiment_history: Sequence[Any] = (),
        **kwargs: Any,
    ) -> CuratedDatasetPlan | Awaitable[CuratedDatasetPlan]: ...

    def design_qlora(
        self, dataset_plan: CuratedDatasetPlan, experiment_history: Sequence[Any] = ()
    ) -> QLoRAConfig | Awaitable[QLoRAConfig]: ...


class RequestFactory(Protocol):
    def training(
        self,
        state: AutonomousRunState,
        *,
        experiment_number: int,
        dataset: DatasetArtifact,
        config: QLoRAConfig,
    ) -> TrainingJobRequest | Awaitable[TrainingJobRequest]: ...

    def evaluation(
        self,
        state: AutonomousRunState,
        *,
        experiment_number: int,
        candidate: CheckpointArtifact,
    ) -> EvaluationJobRequest | Awaitable[EvaluationJobRequest]: ...


class ArtifactVerifier(Protocol):
    def verify_checkpoint(
        self, job: JobResult, *, run_id: str, experiment_number: int
    ) -> CheckpointArtifact | Awaitable[CheckpointArtifact]: ...


class EvaluationReader(Protocol):
    def read_evaluation(
        self,
        job: JobResult,
        *,
        state: AutonomousRunState,
        experiment_number: int,
    ) -> EvaluationEvidence | Awaitable[EvaluationEvidence]: ...


class ReadinessProbe(Protocol):
    def __call__(self, state: AutonomousRunState) -> bool | Awaitable[bool]: ...


class ApprovalVerifier(Protocol):
    def __call__(
        self, state: AutonomousRunState
    ) -> datetime | Awaitable[datetime | None] | None: ...


class TelemetrySink(Protocol):
    def emit(self, event_type: Any, **kwargs: Any) -> Any: ...


async def _await[T](value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await cast(Awaitable[T], value)
    return value


def _finite_cost(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise SupervisorProviderFailure("provider returned invalid cost")
    if float(value) < 0:
        raise SupervisorProviderFailure("provider returned negative cost")
    return float(value)


def _safe_job_id(job: JobResult) -> str:
    if not isinstance(job.provider_job_id, str) or not job.provider_job_id.strip():
        raise SupervisorProviderFailure("provider job ID is missing")
    return job.provider_job_id


def _require_job_result(value: object) -> JobResult:
    if not isinstance(value, JobResult):
        raise SupervisorProviderFailure("provider returned an invalid job result")
    return value


class AutonomousRunSupervisor:
    """Run one approved top-level optimization autonomously and durably."""

    def __init__(
        self,
        *,
        repository: DurableRunStore,
        objective: ObjectiveAdapter,
        agents: AgentAdapter,
        provider: Any,
        request_factory: RequestFactory,
        artifacts: ArtifactVerifier,
        evaluator: EvaluationReader,
        readiness: ReadinessProbe | None = None,
        approval_verifier: ApprovalVerifier | None = None,
        gate: MultiRunPromotionGate | None = None,
        telemetry: TelemetrySink | None = None,
        target_score: float | None = None,
        max_polls: int = 120,
        poll_interval_seconds: float = 30.0,
        phase_cost_estimates: Mapping[str, float] | None = None,
    ) -> None:
        if max_polls < 1 or poll_interval_seconds < 0:
            raise ValueError("poll settings are invalid")
        self.repository = repository
        self.objective = objective
        self.agents = agents
        self.provider = provider
        self.request_factory = request_factory
        self.artifacts = artifacts
        self.evaluator = evaluator
        self.readiness = readiness
        self.approval_verifier = approval_verifier
        self.gate = gate or MultiRunPromotionGate()
        self.telemetry = telemetry
        self.target_score = target_score
        self.max_polls = max_polls
        self.poll_interval_seconds = poll_interval_seconds
        self.phase_cost_estimates = {
            "baseline": 0.0,
            "benchmark": 0.0,
            "dataset": 0.0,
            "training": 5.0,
            "evaluation": 1.0,
            **dict(phase_cost_estimates or {}),
        }
        self._champion_evaluations: dict[str, MultiRunEvaluation] = {}

    async def run_optimization(self, run_id: str) -> AutonomousRunState:
        state = self.repository.get(run_id)
        if state is None:
            raise SupervisorBlocked("run not found")
        if state.status in {
            AutonomousRunStatus.SUCCEEDED,
            AutonomousRunStatus.FAILED,
            AutonomousRunStatus.BLOCKED,
            AutonomousRunStatus.CANCELLED,
            AutonomousRunStatus.STOPPED,
        }:
            return state
        try:
            await self._verify_start(state)
            state = await self._ensure_baseline(state)
            while True:
                state = self._reload(run_id)
                decision = await self._control(state, active_provider_job=False)
                if decision.should_stop:
                    return self._finish(state, _policy_reason(decision.action), decision.action)
                number = len(state.experiments) + 1
                if number > state.max_experiments:
                    return self._finish(state, SupervisorStopReason.MAX_EXPERIMENTS.value, None)
                state = self._transition(
                    state, AutonomousRunStatus.RUNNING, RunPhase.FAILURE_ANALYSIS, "phase started"
                )
                benchmark = await self._run_benchmark(state, number)
                failures = await _await(
                    self.agents.analyze_failures(benchmark.trajectory_refs, state.experiments)
                )
                if not failures:
                    return self._finish(state, "no valid next experiment", None, blocked=True)
                state = self._transition(
                    state, AutonomousRunStatus.RUNNING, RunPhase.RESEARCH, "phase started"
                )
                hypotheses = await _await(
                    self.agents.research(
                        failures,
                        state.experiments,
                        verified_evidence_references=benchmark.trajectory_refs,
                        verified_evidence_metadata={
                            ref: {"status": "succeeded"} for ref in benchmark.trajectory_refs
                        },
                    )
                )
                if not hypotheses:
                    return self._finish(state, "no valid next experiment", None, blocked=True)
                hypothesis = hypotheses[0]
                state = self._transition(
                    state, AutonomousRunStatus.RUNNING, RunPhase.CURATION, "phase started"
                )
                dataset_refs = self._dataset_refs(state)
                plan = await _await(
                    self.agents.curate(
                        benchmark.trajectory_refs,
                        hypotheses=(hypothesis,),
                        experiment_history=state.experiments,
                        verified_dataset_artifact_references=dataset_refs,
                    )
                )
                dataset = await _await(
                    self.objective.build_dataset(state, plan, experiment_number=number)
                )
                dataset = await _await(
                    self.objective.verify_dataset(dataset, run_id=run_id, experiment_number=number)
                )
                self._validate_dataset(dataset, run_id, number)
                state = self._patch(state, {"current_experiment_number": number})
                state = self._transition(
                    state, AutonomousRunStatus.RUNNING, RunPhase.TRAINING, "phase started"
                )
                config = await _await(self.agents.design_qlora(plan, state.experiments))
                train_request = await _await(
                    self.request_factory.training(
                        state, experiment_number=number, dataset=dataset, config=config
                    )
                )
                training = await self._run_job(
                    state, number, RunPhase.TRAINING, train_request, "training"
                )
                if training.status is not JobStatus.COMPLETED:
                    return self._finish(state, "provider training failed", None)
                state = self._reload(run_id)
                control = await self._control(state, active_provider_job=False)
                if control.action.name == "CANCEL":
                    return self._finish(
                        state, SupervisorStopReason.CANCELLATION.value, control.action
                    )
                candidate = await _await(
                    self.artifacts.verify_checkpoint(
                        training, run_id=run_id, experiment_number=number
                    )
                )
                self._validate_checkpoint(candidate)
                state = self._transition(
                    state, AutonomousRunStatus.RUNNING, RunPhase.EVALUATION, "phase started"
                )
                eval_request = await _await(
                    self.request_factory.evaluation(
                        state, experiment_number=number, candidate=candidate
                    )
                )
                evaluation_job = await self._run_job(
                    state, number, RunPhase.EVALUATION, eval_request, "evaluation"
                )
                if evaluation_job.status is not JobStatus.COMPLETED:
                    return self._finish(state, "provider evaluation failed", None)
                evidence = await _await(
                    self.evaluator.read_evaluation(
                        evaluation_job, state=state, experiment_number=number
                    )
                )
                self._validate_evaluation(evidence, number)
                state = self._transition(
                    state, AutonomousRunStatus.RUNNING, RunPhase.PROMOTION, "phase started"
                )
                champion = self._champion_evaluation(run_id, state)
                promotion = self.gate.evaluate(champion, evidence.evaluation)
                record = ExperimentRecord(
                    experiment_number=number,
                    status=ExperimentStatus.SUCCEEDED
                    if promotion.passed
                    else ExperimentStatus.REJECTED,
                    hypothesis_id=hypothesis.hypothesis_id,
                    dataset_id=dataset.dataset_id,
                    training_config=config.model_dump(mode="json"),
                    provider_job_ids=(_safe_job_id(training), _safe_job_id(evaluation_job)),
                    artifact_ids=tuple(
                        (
                            *dataset.artifact_id.split(),
                            candidate.artifact_id,
                            *evidence.artifact_ids,
                        )
                    ),
                    evidence_ids=(
                        champion.evidence.evidence_id,
                        evidence.evaluation.evidence.evidence_id,
                    ),
                    metrics={
                        "baseline": champion.aggregate_score,
                        "candidate": evidence.evaluation.aggregate_score,
                    },
                    stop_reason=None
                    if promotion.passed
                    else "; ".join(promotion.reasons) or "promotion rejected",
                )
                self.repository.add_experiment(run_id, record)
                self._emit(
                    "promotion.decided",
                    run_id,
                    number,
                    evidence_label="LIVE",
                    status=record.status.value,
                )
                if promotion.passed:
                    self._champion_evaluations[run_id] = evidence.evaluation
                    state = self._patch(
                        self._reload(run_id),
                        {
                            "champion_metrics": {
                                "aggregate": evidence.evaluation.aggregate_score,
                                **evidence.evaluation.environment_scores,
                            },
                            "champion_artifact_ids": (
                                *evidence.artifact_ids,
                                candidate.artifact_id,
                            ),
                        },
                    )
                else:
                    state = self._reload(run_id)
                state = self._patch(state, {"current_experiment_number": None})
                if (
                    promotion.passed
                    and self.target_score is not None
                    and evidence.evaluation.aggregate_score >= self.target_score
                ):
                    return self._finish(self._reload(run_id), "target score reached", None)
                state = self._transition(
                    self._reload(run_id),
                    AutonomousRunStatus.RUNNING,
                    RunPhase.FAILURE_ANALYSIS,
                    "phase transition",
                )
        except SupervisorBlocked as exc:
            return self._finish(self._reload(run_id), str(exc), None, blocked=True)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Do not persist provider/agent exception text; it may contain raw content.
            return self._finish(self._reload(run_id), "provider failure", None)

    async def _verify_start(self, state: AutonomousRunState) -> None:
        if not state.approval_consumed or not state.approval_digest:
            raise SupervisorBlocked(SupervisorStopReason.APPROVAL_REQUIRED.value)
        if self.approval_verifier is not None:
            expiry = await _await(self.approval_verifier(state))
            if expiry is not None and datetime.now(UTC) >= expiry:
                raise SupervisorBlocked(SupervisorStopReason.APPROVAL_EXPIRED.value)
        if self.readiness is not None and not await _await(self.readiness(state)):
            raise SupervisorBlocked(SupervisorStopReason.READINESS_BLOCKED.value)

    async def _ensure_baseline(self, state: AutonomousRunState) -> AutonomousRunState:
        if state.baseline_metrics:
            return state
        state = self._transition(
            state, AutonomousRunStatus.RUNNING, RunPhase.BASELINE, "phase started"
        )
        self._check_budget(state, "baseline")
        benchmark = await _await(
            self.objective.benchmark(state, split="baseline", experiment_number=0)
        )
        self._validate_benchmark(benchmark, state.run_id, 0)
        self._champion_evaluations[state.run_id] = benchmark.evaluation
        return self._patch(
            state,
            {
                "baseline_metrics": {
                    "aggregate": benchmark.evaluation.aggregate_score,
                    **benchmark.evaluation.environment_scores,
                },
                "champion_metrics": {
                    "aggregate": benchmark.evaluation.aggregate_score,
                    **benchmark.evaluation.environment_scores,
                },
                "baseline_artifact_ids": benchmark.artifact_ids,
                "champion_artifact_ids": benchmark.artifact_ids,
                "spent_budget_usd": state.spent_budget_usd + _finite_cost(benchmark.cost_usd),
            },
        )

    async def _run_benchmark(self, state: AutonomousRunState, number: int) -> BenchmarkEvidence:
        self._check_budget(state, "benchmark")
        result = await _await(
            self.objective.benchmark(state, split="train", experiment_number=number)
        )
        self._validate_benchmark(result, state.run_id, number)
        return result

    async def _run_job(
        self,
        state: AutonomousRunState,
        number: int,
        phase: RunPhase,
        request: TrainingJobRequest | EvaluationJobRequest,
        kind: str,
    ) -> JobResult:
        request_payload = asdict(request)
        request_hash = canonical_request_hash(
            {key: value for key, value in request_payload.items() if key != "job_name"}
        )
        operation_key = canonical_operation_key(state.run_id, number, kind)
        deterministic_name = canonical_job_name(operation_key, request_hash)
        # The request factory supplies all mutable parameters; the supervisor
        # owns the final deterministic provider name so a retry/restart cannot
        # accidentally create a second job.
        request.job_name = deterministic_name
        operation = self.repository.put_operation_intent(
            RunOperation(
                operation_key=operation_key,
                run_id=state.run_id,
                experiment_number=number,
                phase=phase,
                provider_name="sagemaker",
                request_digest=request_hash,
            )
        )
        result: JobResult | None = None
        reconcile = getattr(self.provider, f"reconcile_{kind}", None)
        if operation.provider_id:
            result = _require_job_result(
                await _await(getattr(self.provider, f"get_{kind}_status")(request.job_name))
            )
        elif callable(reconcile):
            reconciled = await _await(reconcile(request))
            result = None if reconciled is None else _require_job_result(reconciled)
        if result is None:
            self._check_budget(state, kind)
            result = _require_job_result(
                await _await(getattr(self.provider, f"submit_{kind}")(request))
            )
            self.repository.record_operation_result(
                state.run_id,
                operation_key,
                provider_id=_safe_job_id(result),
                status=_operation_status(result.status),
                result={"job_name": result.job_name},
            )
        else:
            _safe_job_id(result)
            self.repository.record_operation_result(
                state.run_id,
                operation_key,
                provider_id=_safe_job_id(result),
                status=_operation_status(result.status),
                result={"job_name": result.job_name},
            )
        self._emit(
            "job.submitted",
            state.run_id,
            number,
            operation_key=operation_key,
            job_id=_safe_job_id(result),
        )
        if result.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STOPPED}:
            return result
        for _ in range(self.max_polls):
            current = self._reload(state.run_id)
            control = await self._control(current, active_provider_job=True)
            if control.request_provider_stop:
                stop_method = getattr(self.provider, f"stop_{kind}", None)
                if callable(stop_method):
                    await _await(stop_method(request.job_name))
            if control.action.name in {"SAFE_STOP", "APPROVAL_EXPIRED"}:
                # Drain the active provider job, but never submit a later phase.
                pass
            result = _require_job_result(
                await _await(getattr(self.provider, f"get_{kind}_status")(request.job_name))
            )
            if result.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STOPPED}:
                self.repository.record_operation_result(
                    state.run_id,
                    operation_key,
                    provider_id=_safe_job_id(result),
                    status=_operation_status(result.status),
                    result={"job_name": result.job_name},
                )
                return result
            if self.poll_interval_seconds:
                await asyncio.sleep(self.poll_interval_seconds)
        raise SupervisorProviderFailure("provider polling timed out")

    async def _control(
        self, state: AutonomousRunState, *, active_provider_job: bool
    ) -> PolicyDecision:
        expiry = (
            await _await(self.approval_verifier(state))
            if self.approval_verifier is not None
            else None
        )
        remaining = max(0.0, state.approved_budget_usd - state.spent_budget_usd)
        return decide_run_control(
            cancellation_requested=state.cancellation_requested,
            safe_stop_requested=state.safe_stop_requested,
            approval_expires_at=expiry,
            now=datetime.now(UTC) if expiry is not None else None,
            active_provider_job=active_provider_job,
            experiment_count=len(state.experiments),
            max_experiments=state.max_experiments,
            champion_score=(
                state.champion_metrics.get("aggregate") if state.champion_metrics else None
            )
            if self.target_score is not None
            else None,
            target_score=self.target_score,
            remaining_budget_usd=remaining,
        )

    def _check_budget(self, state: AutonomousRunState, phase: str) -> None:
        estimated = _finite_cost(self.phase_cost_estimates.get(phase, 0.0))
        if state.spent_budget_usd + estimated > state.approved_budget_usd:
            raise SupervisorBlocked(SupervisorStopReason.BUDGET_EXHAUSTED.value)

    def _reload(self, run_id: str) -> AutonomousRunState:
        state = self.repository.get(run_id)
        if state is None:
            raise SupervisorBlocked("run not found")
        return state

    def _transition(
        self, state: AutonomousRunState, status: AutonomousRunStatus, phase: RunPhase, reason: str
    ) -> AutonomousRunState:
        return self.repository.transition(
            state.run_id, expected_version=state.version, status=status, phase=phase, reason=reason
        )

    def _patch(self, state: AutonomousRunState, updates: Mapping[str, Any]) -> AutonomousRunState:
        return self.repository.update_state(
            state.run_id, expected_version=state.version, updates=updates
        )

    def _finish(
        self, state: AutonomousRunState, reason: str, action: Any, *, blocked: bool = False
    ) -> AutonomousRunState:
        if state.status in {
            AutonomousRunStatus.SUCCEEDED,
            AutonomousRunStatus.FAILED,
            AutonomousRunStatus.BLOCKED,
            AutonomousRunStatus.CANCELLED,
            AutonomousRunStatus.STOPPED,
        }:
            return state
        if blocked:
            status = AutonomousRunStatus.BLOCKED
        elif action is not None and getattr(action, "name", "") == "CANCEL":
            status = AutonomousRunStatus.CANCELLED
        elif action is not None and getattr(action, "name", "") in {
            "SAFE_STOP",
            "APPROVAL_EXPIRED",
        }:
            status = AutonomousRunStatus.STOPPED
        elif reason in {"maximum experiments reached", "target score reached"}:
            status = AutonomousRunStatus.SUCCEEDED
        elif reason == "provider failure":
            status = AutonomousRunStatus.FAILED
        else:
            status = AutonomousRunStatus.STOPPED
        state = self._patch(state, {"stop_reason": reason})
        phase = RunPhase.COMPLETED if status is AutonomousRunStatus.SUCCEEDED else RunPhase.STOPPED
        transition_reason = {
            AutonomousRunStatus.SUCCEEDED: "run completed",
            AutonomousRunStatus.FAILED: "run failed",
            AutonomousRunStatus.BLOCKED: "run blocked",
            AutonomousRunStatus.CANCELLED: "run cancelled",
            AutonomousRunStatus.STOPPED: "run stopped",
        }[status]
        return self._transition(state, status, phase, transition_reason)

    def _champion_evaluation(self, run_id: str, state: AutonomousRunState) -> MultiRunEvaluation:
        evaluation = self._champion_evaluations.get(run_id)
        if evaluation is not None:
            return evaluation
        loader = getattr(self.repository, "get_champion_evaluation", None)
        if callable(loader):
            loaded = loader(run_id)
            if isinstance(loaded, MultiRunEvaluation):
                self._champion_evaluations[run_id] = loaded
                return loaded
        raise SupervisorBlocked("champion evidence unavailable")

    @staticmethod
    def _dataset_refs(state: AutonomousRunState) -> tuple[str, ...]:
        ref = state.metadata.get("dataset_artifact_ref")
        return (ref,) if isinstance(ref, str) and ref else ("dataset://provenance",)

    @staticmethod
    def _validate_benchmark(result: BenchmarkEvidence, run_id: str, number: int) -> None:
        if (
            result.evaluation.evidence.label.value not in {"LIVE", "PRIOR_VERIFIED_RUN"}
            or not result.evaluation.evidence.verified
        ):
            raise SupervisorBlocked("benchmark evidence is not verified")
        if result.evaluation.run_id == "" or result.evaluation.evidence.manifest_sha256 is None:
            raise SupervisorBlocked("benchmark provenance is incomplete")
        if not result.trajectory_refs or not result.artifact_ids:
            raise SupervisorBlocked("benchmark artifacts are missing")
        _finite_cost(result.cost_usd)

    @staticmethod
    def _validate_dataset(dataset: DatasetArtifact, run_id: str, number: int) -> None:
        if (
            not dataset.dataset_id
            or not dataset.uri
            or not dataset.artifact_id
            or len(dataset.sha256) != 64
        ):
            raise SupervisorBlocked("dataset artifact is incomplete")

    @staticmethod
    def _validate_checkpoint(candidate: CheckpointArtifact) -> None:
        if not candidate.uri or not candidate.artifact_id or len(candidate.sha256) != 64:
            raise SupervisorBlocked("checkpoint artifact is incomplete")

    @staticmethod
    def _validate_evaluation(evidence: EvaluationEvidence, number: int) -> None:
        if (
            evidence.evaluation.run_number != number
            or not evidence.evaluation.evidence.verified
            or evidence.evaluation.evidence.label.value not in {"LIVE", "PRIOR_VERIFIED_RUN"}
        ):
            raise SupervisorBlocked("evaluation evidence is not verified")
        _finite_cost(evidence.cost_usd)

    def _emit(self, event_type: str, run_id: str, number: int, **kwargs: Any) -> None:
        if self.telemetry is None:
            return
        try:
            self.telemetry.emit(
                event_type,
                run_id=run_id,
                run_number=max(1, number),
                experiment_id=f"{run_id}:{number}",
                reason="provider operation completed",
                phase=kwargs.pop("phase", None),
                **kwargs,
            )
        except Exception:
            return


def _operation_status(status: JobStatus) -> RunOperationStatus:
    return {
        JobStatus.SUBMITTED: RunOperationStatus.SUBMITTED,
        JobStatus.IN_PROGRESS: RunOperationStatus.RUNNING,
        JobStatus.COMPLETED: RunOperationStatus.SUCCEEDED,
        JobStatus.FAILED: RunOperationStatus.FAILED,
        JobStatus.STOPPED: RunOperationStatus.CANCELLED,
        JobStatus.UNKNOWN: RunOperationStatus.RUNNING,
    }[status]


def _policy_reason(action: PolicyAction) -> str:
    return {
        PolicyAction.CANCEL: SupervisorStopReason.CANCELLATION.value,
        PolicyAction.SAFE_STOP: SupervisorStopReason.SAFE_STOP.value,
        PolicyAction.APPROVAL_EXPIRED: SupervisorStopReason.APPROVAL_EXPIRED.value,
        PolicyAction.MAX_EXPERIMENTS: SupervisorStopReason.MAX_EXPERIMENTS.value,
        PolicyAction.TARGET_REACHED: "target score reached",
        PolicyAction.BUDGET_EXHAUSTED: SupervisorStopReason.BUDGET_EXHAUSTED.value,
        PolicyAction.CONTINUE: "policy permits next phase",
    }[action]


__all__ = [
    "ApprovalVerifier",
    "ArtifactVerifier",
    "AutonomousRunSupervisor",
    "BenchmarkEvidence",
    "CheckpointArtifact",
    "DatasetArtifact",
    "DurableRunStore",
    "EvaluationEvidence",
    "EvaluationReader",
    "ObjectiveAdapter",
    "RequestFactory",
    "SupervisorBlocked",
    "SupervisorError",
    "SupervisorProviderFailure",
]
