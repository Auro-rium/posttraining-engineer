"""Durable, fail-closed autonomous post-training supervisor.

The supervisor is deliberately adapter driven.  It owns ordering, durable
state, bounded policy, and reconciliation; objective workers, Nemotron, S3,
and SageMaker remain injected execution boundaries.  This keeps local tests
honest while allowing the API layer to wire real AWS implementations.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import inspect
import json
import re
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
from app.providers.sagemaker import (
    EvaluationJobRequest,
    JobResult,
    JobStatus,
    TrainingJobRequest,
    TransientProviderError,
)

from .telemetry import DurableTelemetryError


class SupervisorError(RuntimeError):
    """Base error for the durable supervisor."""


class SupervisorBlocked(SupervisorError):
    """The run cannot start without a missing or invalid prerequisite."""


class SupervisorProviderFailure(SupervisorError):
    """A provider operation failed and the run must fail closed."""


class SupervisorAgentFailure(SupervisorError):
    """A judgment/agent adapter failed; no provider failure is implied."""


class SupervisorArtifactFailure(SupervisorBlocked):
    """A required artifact or provenance contract was invalid."""


class SupervisorRecoverable(SupervisorError):
    """A transient/ambiguous provider outcome left durable work recoverable."""


class SupervisorControlStop(SupervisorError):
    """A stop condition observed before a provider submission."""

    def __init__(self, action: PolicyAction) -> None:
        super().__init__(action.value)
        self.action = action


class SupervisorTelemetryFailure(SupervisorError):
    """Durable telemetry failed and the run must fail closed."""


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
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9:/.@+_=,-]{0,1999}", job.provider_job_id):
        raise SupervisorProviderFailure("provider job ID is unsafe")
    return job.provider_job_id


def _require_job_result(value: object) -> JobResult:
    if not isinstance(value, JobResult):
        raise SupervisorProviderFailure("provider returned an invalid job result")
    return value


def _encode_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")


def _decode_json(value: object) -> Any:
    if not isinstance(value, str) or not value:
        raise SupervisorBlocked("durable evidence is unavailable")
    try:
        return json.loads(base64.urlsafe_b64decode(value.encode("ascii")))
    except (ValueError, UnicodeError, binascii.Error, json.JSONDecodeError) as exc:
        raise SupervisorBlocked("durable evidence is invalid") from exc


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
            state = self._recover_promotion_intents(state)
            while True:
                state = self._reload(run_id)
                decision = await self._control(state, active_provider_job=False)
                if decision.should_stop:
                    return self._finish(state, _policy_reason(decision.action), decision.action)
                number = len(state.experiments) + 1
                if number > state.max_experiments:
                    return self._finish(state, SupervisorStopReason.MAX_EXPERIMENTS.value, None)
                persisted = (
                    state.current_hypothesis if state.current_experiment_number == number else None
                )
                benchmark: BenchmarkEvidence | None = None
                plan: CuratedDatasetPlan | None = None
                if isinstance(persisted, Mapping) and isinstance(
                    persisted.get("hypothesis"), Mapping
                ):
                    try:
                        hypothesis = ResearchHypothesis.model_validate(persisted["hypothesis"])
                    except Exception as exc:
                        raise SupervisorBlocked("persisted hypothesis is invalid") from exc
                    plan_value = persisted.get("dataset_plan")
                    plan = (
                        CuratedDatasetPlan.model_validate(plan_value)
                        if isinstance(plan_value, Mapping)
                        else None
                    )
                else:
                    state = self._transition(
                        state,
                        AutonomousRunStatus.RUNNING,
                        RunPhase.FAILURE_ANALYSIS,
                        "phase started",
                    )
                    benchmark = await self._run_benchmark(state, number)
                    state = self._reload(run_id)
                    try:
                        failures = await _await(
                            self.agents.analyze_failures(
                                benchmark.trajectory_refs, state.experiments
                            )
                        )
                    except Exception as exc:
                        raise SupervisorAgentFailure("agent failure") from exc
                    if not failures:
                        return self._finish(state, "no valid next experiment", None, blocked=True)
                    state = self._transition(
                        state, AutonomousRunStatus.RUNNING, RunPhase.RESEARCH, "phase started"
                    )
                    try:
                        hypotheses = await _await(
                            self.agents.research(
                                failures,
                                state.experiments,
                                verified_evidence_references=benchmark.trajectory_refs,
                                verified_evidence_metadata={
                                    ref: {"status": "succeeded"}
                                    for ref in benchmark.trajectory_refs
                                },
                            )
                        )
                    except Exception as exc:
                        raise SupervisorAgentFailure("agent failure") from exc
                    if not hypotheses:
                        return self._finish(state, "no valid next experiment", None, blocked=True)
                    hypothesis = hypotheses[0]
                    state = self._patch(
                        self._reload(run_id),
                        {
                            "current_experiment_number": number,
                            "current_hypothesis": {
                                "hypothesis": hypothesis.model_dump(mode="json")
                            },
                        },
                    )
                    state = self._transition(
                        state, AutonomousRunStatus.RUNNING, RunPhase.CURATION, "phase started"
                    )
                if plan is None:
                    if benchmark is None:
                        benchmark = await self._run_benchmark(self._reload(run_id), number)
                    dataset_refs = self._dataset_refs(self._reload(run_id))
                    try:
                        plan = await _await(
                            self.agents.curate(
                                benchmark.trajectory_refs,
                                hypotheses=(hypothesis,),
                                experiment_history=self._reload(run_id).experiments,
                                verified_dataset_artifact_references=dataset_refs,
                            )
                        )
                    except Exception as exc:
                        raise SupervisorAgentFailure("agent failure") from exc
                state = self._patch(
                    self._reload(run_id),
                    {
                        "current_hypothesis": {
                            "hypothesis": hypothesis.model_dump(mode="json"),
                            "dataset_plan": plan.model_dump(mode="json"),
                        }
                    },
                )
                dataset = await self._run_dataset(state, plan, number)
                self._validate_dataset(dataset, run_id, number)
                state = self._patch(
                    self._reload(run_id),
                    {
                        "current_experiment_number": number,
                        "current_dataset_uri": dataset.uri,
                        "current_dataset_sha256": dataset.sha256,
                        "metadata": {
                            **self._reload(run_id).metadata,
                            "dataset_artifact_ref": dataset.artifact_id,
                        },
                    },
                )
                state = self._transition(
                    state, AutonomousRunStatus.RUNNING, RunPhase.TRAINING, "phase started"
                )
                persisted_config = (
                    state.current_hypothesis.get("qlora_config")
                    if isinstance(state.current_hypothesis, Mapping)
                    else None
                )
                if isinstance(persisted_config, Mapping):
                    try:
                        config = QLoRAConfig.model_validate(persisted_config)
                    except Exception as exc:
                        raise SupervisorBlocked("persisted QLoRA config is invalid") from exc
                else:
                    try:
                        config = await _await(self.agents.design_qlora(plan, state.experiments))
                    except Exception as exc:
                        raise SupervisorAgentFailure("agent failure") from exc
                state = self._patch(
                    self._reload(run_id),
                    {
                        "current_hypothesis": {
                            **(self._reload(run_id).current_hypothesis or {}),
                            "qlora_config": config.model_dump(mode="json"),
                        }
                    },
                )
                train_request = await _await(
                    self.request_factory.training(
                        state, experiment_number=number, dataset=dataset, config=config
                    )
                )
                training = await self._run_job(
                    state, number, RunPhase.TRAINING, train_request, "training"
                )
                if training.status is JobStatus.STOPPED:
                    current = self._reload(run_id)
                    action = (await self._control(current, active_provider_job=False)).action
                    if action.name == "CANCEL":
                        return self._finish(
                            current, SupervisorStopReason.CANCELLATION.value, action
                        )
                    return self._finish(current, "provider training stopped", None)
                if training.status is JobStatus.FAILED:
                    return self._finish(self._reload(run_id), "provider training failed", None)
                state = self._reload(run_id)
                control = await self._control(state, active_provider_job=False)
                if control.should_stop:
                    return self._finish(state, _policy_reason(control.action), control.action)
                try:
                    candidate = await _await(
                        self.artifacts.verify_checkpoint(
                            training, run_id=run_id, experiment_number=number
                        )
                    )
                except SupervisorError:
                    raise
                except Exception as exc:
                    raise SupervisorArtifactFailure("checkpoint artifact failure") from exc
                self._validate_checkpoint(candidate)
                state = self._patch(
                    self._reload(run_id),
                    {
                        "current_candidate_uri": candidate.uri,
                        "current_candidate_sha256": candidate.sha256,
                        "metadata": {
                            **self._reload(run_id).metadata,
                            "current_candidate_artifact_id": candidate.artifact_id,
                        },
                    },
                )
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
                if evaluation_job.status is JobStatus.STOPPED:
                    current = self._reload(run_id)
                    action = (await self._control(current, active_provider_job=False)).action
                    if action.name == "CANCEL":
                        return self._finish(
                            current, SupervisorStopReason.CANCELLATION.value, action
                        )
                    return self._finish(current, "provider evaluation stopped", None)
                if evaluation_job.status is JobStatus.FAILED:
                    return self._finish(self._reload(run_id), "provider evaluation failed", None)
                state = self._reload(run_id)
                evidence_key = canonical_operation_key(run_id, number, "evaluation-evidence")
                evidence_intent = self.repository.put_operation_intent(
                    RunOperation(
                        operation_key=evidence_key,
                        run_id=run_id,
                        experiment_number=number,
                        phase=RunPhase.EVALUATION,
                        provider_name="evaluator",
                        request_digest=canonical_request_hash(
                            {"job_id": _safe_job_id(evaluation_job), "run_id": run_id}
                        ),
                    )
                )
                if evidence_intent.status is RunOperationStatus.SUCCEEDED:
                    evidence = self._decode_evaluation_evidence(evidence_intent.result)
                else:
                    try:
                        evidence = await _await(
                            self.evaluator.read_evaluation(
                                evaluation_job, state=state, experiment_number=number
                            )
                        )
                    except SupervisorError:
                        raise
                    except Exception as exc:
                        raise SupervisorArtifactFailure("evaluation evidence failure") from exc
                    self.repository.record_operation_result(
                        run_id,
                        evidence_key,
                        status=RunOperationStatus.SUCCEEDED,
                        result=self._encode_evaluation_evidence(evidence),
                    )
                self._validate_evaluation(evidence, number, run_id=run_id, state=state)
                self._reconcile_cost(
                    self._reload(run_id),
                    evidence_key,
                    evidence.cost_usd,
                )
                state = self._reload(run_id)
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
                    artifact_ids=(
                        dataset.artifact_id,
                        candidate.artifact_id,
                        *evidence.artifact_ids,
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
                promotion_key = canonical_operation_key(run_id, number, "promotion")
                promotion_intent = self.repository.put_operation_intent(
                    RunOperation(
                        operation_key=promotion_key,
                        run_id=run_id,
                        experiment_number=number,
                        phase=RunPhase.PROMOTION,
                        provider_name="supervisor",
                        request_digest=canonical_request_hash(
                            {
                                "record": record.model_dump(mode="json"),
                                "candidate": evidence.evaluation.model_dump(mode="json"),
                            }
                        ),
                    )
                )
                if promotion_intent.status is RunOperationStatus.INTENT:
                    self.repository.record_operation_result(
                        run_id,
                        promotion_key,
                        status=RunOperationStatus.INTENT,
                        result={
                            "record": record.model_dump(mode="json"),
                            "candidate_evaluation": evidence.evaluation.model_dump(mode="json"),
                            "champion_evaluation": champion.model_dump(mode="json"),
                            "candidate_artifact_id": candidate.artifact_id,
                            "candidate_uri": candidate.uri,
                            "candidate_sha256": candidate.sha256,
                            "promoted": promotion.passed,
                        },
                    )
                self._complete_promotion(run_id, promotion_key)
                self._emit(
                    "promotion.decided",
                    run_id,
                    number,
                    evidence_label="LIVE",
                    status=record.status.value,
                )
                if promotion.passed:
                    state = self._reload(run_id)
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
        except SupervisorAgentFailure as exc:
            return self._finish(self._reload(run_id), str(exc), None)
        except SupervisorProviderFailure:
            return self._finish(self._reload(run_id), "provider failure", None)
        except SupervisorRecoverable:
            return self._reload(run_id)
        except SupervisorControlStop as exc:
            return self._finish(self._reload(run_id), _policy_reason(exc.action), exc.action)
        except SupervisorTelemetryFailure:
            return self._finish(self._reload(run_id), "durable telemetry failure", None)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Do not persist provider/agent exception text; it may contain raw content.
            return self._finish(self._reload(run_id), "provider failure", None)

    async def _verify_start(self, state: AutonomousRunState) -> None:
        if not state.approval_consumed or not state.approval_digest:
            raise SupervisorBlocked(SupervisorStopReason.APPROVAL_REQUIRED.value)
        expiry = state.approval_expires_at
        if self.approval_verifier is not None:
            verified_expiry = await _await(self.approval_verifier(state))
            if verified_expiry is not None and expiry is not None and verified_expiry != expiry:
                raise SupervisorBlocked("approval expiry mismatch")
            expiry = verified_expiry or expiry
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
        operation_key = canonical_operation_key(state.run_id, 1, "baseline")
        request_digest = canonical_request_hash(
            {"run_id": state.run_id, "split": "baseline", "experiment_number": 0}
        )
        operation = self.repository.put_operation_intent(
            RunOperation(
                operation_key=operation_key,
                run_id=state.run_id,
                experiment_number=1,
                phase=RunPhase.BASELINE,
                provider_name="objective",
                request_digest=request_digest,
            )
        )
        benchmark: BenchmarkEvidence
        if operation.status is RunOperationStatus.SUCCEEDED:
            benchmark = self._decode_benchmark(operation.result)
        else:
            try:
                benchmark = await _await(
                    self.objective.benchmark(state, split="baseline", experiment_number=0)
                )
            except (TransientProviderError, TimeoutError, ConnectionError) as exc:
                raise SupervisorRecoverable("baseline provider request is retryable") from exc
            except Exception as exc:
                raise SupervisorProviderFailure("baseline provider failure") from exc
            self._validate_benchmark(benchmark, state.run_id, 0, state=state)
            self.repository.record_operation_result(
                state.run_id,
                operation_key,
                status=RunOperationStatus.SUCCEEDED,
                result=self._encode_benchmark_result(benchmark),
            )
        self._validate_benchmark(benchmark, state.run_id, 0, state=state)
        state = self._reconcile_cost(state, operation_key, benchmark.cost_usd)
        return self._patch(
            self._reload(state.run_id),
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
                "metadata": {
                    **self._reload(state.run_id).metadata,
                    "champion_evaluation_b64": self._encode_evaluation(benchmark.evaluation),
                },
            },
        )

    async def _run_benchmark(self, state: AutonomousRunState, number: int) -> BenchmarkEvidence:
        self._check_budget(state, "benchmark")
        operation_key = canonical_operation_key(state.run_id, number, "benchmark")
        request_digest = canonical_request_hash(
            {"run_id": state.run_id, "split": "train", "experiment_number": number}
        )
        operation = self.repository.put_operation_intent(
            RunOperation(
                operation_key=operation_key,
                run_id=state.run_id,
                experiment_number=number,
                phase=RunPhase.FAILURE_ANALYSIS,
                provider_name="objective",
                request_digest=request_digest,
            )
        )
        if operation.status is RunOperationStatus.SUCCEEDED:
            result = self._decode_benchmark(operation.result)
        else:
            try:
                result = await _await(
                    self.objective.benchmark(state, split="train", experiment_number=number)
                )
            except (TransientProviderError, TimeoutError, ConnectionError) as exc:
                raise SupervisorRecoverable("benchmark provider request is retryable") from exc
            except Exception as exc:
                raise SupervisorProviderFailure("benchmark provider failure") from exc
            self._validate_benchmark(result, state.run_id, number, state=state)
            self.repository.record_operation_result(
                state.run_id,
                operation_key,
                status=RunOperationStatus.SUCCEEDED,
                result=self._encode_benchmark_result(result),
            )
        self._validate_benchmark(result, state.run_id, number, state=state)
        self._reconcile_cost(self._reload(state.run_id), operation_key, result.cost_usd)
        return result

    async def _run_job(
        self,
        state: AutonomousRunState,
        number: int,
        phase: RunPhase,
        request: TrainingJobRequest | EvaluationJobRequest,
        kind: str,
    ) -> JobResult:
        expected_type = TrainingJobRequest if kind == "training" else EvaluationJobRequest
        if not isinstance(request, expected_type):
            raise SupervisorProviderFailure(f"{kind} request is invalid")
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
        request_payload = asdict(request)
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
        if operation.status is RunOperationStatus.INTENT and not operation.result:
            # Persist the exact request before the provider side effect.  A
            # restarted worker can therefore inspect the intent and reconcile
            # the same deterministic job name without creating another job.
            self.repository.record_operation_result(
                state.run_id,
                operation_key,
                status=RunOperationStatus.INTENT,
                result={"request": request_payload, "job_name": deterministic_name},
            )
            operation = self.repository.get_operation(state.run_id, operation_key) or operation
        result: JobResult | None = None
        reconcile = getattr(self.provider, f"reconcile_{kind}", None)
        if operation.provider_id:
            try:
                result = _require_job_result(
                    await _await(getattr(self.provider, f"get_{kind}_status")(request.job_name))
                )
            except (TransientProviderError, TimeoutError, ConnectionError) as exc:
                raise SupervisorRecoverable(f"provider {kind} status is retryable") from exc
            except Exception as exc:
                raise SupervisorProviderFailure(f"provider {kind} status failure") from exc
        elif callable(reconcile):
            try:
                reconciled = await _await(reconcile(request))
            except (TransientProviderError, TimeoutError, ConnectionError) as exc:
                raise SupervisorRecoverable(f"provider {kind} reconciliation is retryable") from exc
            except Exception as exc:
                raise SupervisorProviderFailure(f"provider {kind} reconciliation failure") from exc
            result = None if reconciled is None else _require_job_result(reconciled)
        if result is None:
            control = await self._control(self._reload(state.run_id), active_provider_job=False)
            if control.should_stop:
                raise SupervisorControlStop(control.action)
            self._check_budget(state, kind)
            try:
                result = _require_job_result(
                    await _await(getattr(self.provider, f"submit_{kind}")(request))
                )
            except (TransientProviderError, TimeoutError, ConnectionError) as exc:
                raise SupervisorRecoverable(f"provider {kind} submission is retryable") from exc
            except SupervisorProviderFailure:
                raise
            except Exception as exc:
                raise SupervisorProviderFailure(f"provider {kind} submission failure") from exc
            self.repository.record_operation_result(
                state.run_id,
                operation_key,
                provider_id=_safe_job_id(result),
                status=_operation_status(result.status),
                result={
                    "job_name": result.job_name,
                    "cost_usd": self._job_cost(result),
                },
            )
        else:
            _safe_job_id(result)
            if operation.status not in {
                RunOperationStatus.SUCCEEDED,
                RunOperationStatus.FAILED,
                RunOperationStatus.CANCELLED,
            }:
                self.repository.record_operation_result(
                    state.run_id,
                    operation_key,
                    provider_id=_safe_job_id(result),
                    status=_operation_status(result.status),
                    result={"job_name": result.job_name, "cost_usd": self._job_cost(result)},
                )
        current = self._reload(state.run_id)
        current_field = (
            "current_training_job_id" if kind == "training" else "current_evaluation_job_id"
        )
        state = self._patch(current, {current_field: _safe_job_id(result)})
        if result.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STOPPED}:
            self._reconcile_cost(state, operation_key, self._job_cost(result))
        self._emit(
            "job.submitted",
            state.run_id,
            number,
            operation_key=operation_key,
            job_id=_safe_job_id(result),
        )
        if result.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STOPPED}:
            self._emit_job_terminal(state.run_id, number, operation_key, result)
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
            try:
                result = _require_job_result(
                    await _await(getattr(self.provider, f"get_{kind}_status")(request.job_name))
                )
            except (TransientProviderError, TimeoutError, ConnectionError) as exc:
                raise SupervisorRecoverable(f"provider {kind} status is retryable") from exc
            except Exception as exc:
                raise SupervisorProviderFailure(f"provider {kind} status failure") from exc
            if result.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STOPPED}:
                if operation.status not in {
                    RunOperationStatus.SUCCEEDED,
                    RunOperationStatus.FAILED,
                    RunOperationStatus.CANCELLED,
                }:
                    self.repository.record_operation_result(
                        state.run_id,
                        operation_key,
                        provider_id=_safe_job_id(result),
                        status=_operation_status(result.status),
                        result={"job_name": result.job_name, "cost_usd": self._job_cost(result)},
                    )
                self._reconcile_cost(
                    self._reload(state.run_id), operation_key, self._job_cost(result)
                )
                self._emit_job_terminal(state.run_id, number, operation_key, result)
                return result
            if self.poll_interval_seconds:
                await asyncio.sleep(self.poll_interval_seconds)
        raise SupervisorRecoverable("provider polling timed out")

    @staticmethod
    def _job_cost(job: JobResult) -> float:
        values = job.raw_response
        if not isinstance(values, Mapping):
            return 0.0
        for key in ("actual_cost_usd", "cost_usd", "cost"):
            if key in values:
                return _finite_cost(values[key])
        return 0.0

    def _emit_job_terminal(
        self, run_id: str, number: int, operation_key: str, result: JobResult
    ) -> None:
        self._emit(
            "job.completed" if result.status is JobStatus.COMPLETED else "job.failed",
            run_id,
            number,
            operation_key=operation_key,
            job_id=_safe_job_id(result),
            cost_usd=self._job_cost(result),
        )

    async def _control(
        self, state: AutonomousRunState, *, active_provider_job: bool
    ) -> PolicyDecision:
        expiry = state.approval_expires_at
        if self.approval_verifier is not None:
            verified_expiry = await _await(self.approval_verifier(state))
            if verified_expiry is not None and expiry is not None and verified_expiry != expiry:
                raise SupervisorBlocked("approval expiry mismatch")
            expiry = verified_expiry or expiry
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

    @staticmethod
    def _encode_evaluation(evaluation: MultiRunEvaluation) -> str:
        return _encode_json(evaluation.model_dump(mode="json"))

    @staticmethod
    def _decode_evaluation(value: object) -> MultiRunEvaluation:
        try:
            return MultiRunEvaluation.model_validate(_decode_json(value))
        except SupervisorBlocked:
            raise
        except Exception as exc:
            raise SupervisorBlocked("durable champion evidence is invalid") from exc

    @staticmethod
    def _encode_evaluation_evidence(evidence: EvaluationEvidence) -> dict[str, Any]:
        return {
            "evaluation": evidence.evaluation.model_dump(mode="json"),
            "artifact_ids": list(evidence.artifact_ids),
            "cost_usd": _finite_cost(evidence.cost_usd),
        }

    @staticmethod
    def _decode_evaluation_evidence(result: Mapping[str, Any]) -> EvaluationEvidence:
        try:
            raw_artifacts = result["artifact_ids"]
            if not isinstance(raw_artifacts, list) or any(
                not isinstance(item, str) or not item.strip() for item in raw_artifacts
            ):
                raise TypeError("evaluation artifact references are invalid")
            return EvaluationEvidence(
                evaluation=MultiRunEvaluation.model_validate(result["evaluation"]),
                artifact_ids=tuple(raw_artifacts),
                cost_usd=_finite_cost(result.get("cost_usd", 0.0)),
            )
        except SupervisorError:
            raise
        except (TransientProviderError, TimeoutError, ConnectionError) as exc:
            raise SupervisorRecoverable("dataset provider request is retryable") from exc
        except Exception as exc:
            raise SupervisorBlocked("durable evaluation evidence is invalid") from exc

    @classmethod
    def _encode_benchmark_result(cls, benchmark: BenchmarkEvidence) -> dict[str, Any]:
        return {
            "evaluation": benchmark.evaluation.model_dump(mode="json"),
            "trajectory_refs": list(benchmark.trajectory_refs),
            "artifact_ids": list(benchmark.artifact_ids),
            "cost_usd": _finite_cost(benchmark.cost_usd),
        }

    @staticmethod
    def _decode_benchmark(result: Mapping[str, Any]) -> BenchmarkEvidence:
        try:
            if not isinstance(result, Mapping):
                raise TypeError("benchmark result must be a mapping")
            raw_refs = result["trajectory_refs"]
            raw_artifacts = result["artifact_ids"]
            if (
                not isinstance(raw_refs, list)
                or not raw_refs
                or any(not isinstance(item, str) or not item.strip() for item in raw_refs)
                or not isinstance(raw_artifacts, list)
                or not raw_artifacts
                or any(not isinstance(item, str) or not item.strip() for item in raw_artifacts)
            ):
                raise TypeError("benchmark references are invalid")
            evaluation = MultiRunEvaluation.model_validate(result["evaluation"])
            refs = tuple(raw_refs)
            artifacts = tuple(raw_artifacts)
            return BenchmarkEvidence(
                evaluation=evaluation,
                trajectory_refs=refs,
                artifact_ids=artifacts,
                cost_usd=_finite_cost(result.get("cost_usd", 0.0)),
            )
        except SupervisorError:
            raise
        except Exception as exc:
            raise SupervisorBlocked("durable benchmark evidence is invalid") from exc

    @staticmethod
    def _decode_dataset(result: Mapping[str, Any]) -> DatasetArtifact:
        try:
            if not isinstance(result, Mapping) or any(
                not isinstance(result.get(key), str) or not result[key].strip()
                for key in ("dataset_id", "uri", "sha256", "artifact_id")
            ):
                raise TypeError("dataset result fields are invalid")
            return DatasetArtifact(
                dataset_id=result["dataset_id"],
                uri=result["uri"],
                sha256=result["sha256"],
                artifact_id=result["artifact_id"],
                cost_usd=_finite_cost(result.get("cost_usd", 0.0)),
            )
        except SupervisorError:
            raise
        except Exception as exc:
            raise SupervisorBlocked("durable dataset artifact is invalid") from exc

    def _reconcile_cost(
        self, state: AutonomousRunState, operation_key: str, actual_cost: object
    ) -> AutonomousRunState:
        cost = _finite_cost(actual_cost)
        marker = "cost." + canonical_request_hash({"operation_key": operation_key})[:32]
        current = self._reload(state.run_id)
        if marker in current.metadata:
            return current
        if current.spent_budget_usd + cost > current.approved_budget_usd:
            raise SupervisorBlocked(SupervisorStopReason.BUDGET_EXHAUSTED.value)
        return self._patch(
            current,
            {
                "spent_budget_usd": current.spent_budget_usd + cost,
                "metadata": {**current.metadata, marker: f"{cost:.8f}"},
            },
        )

    def _complete_promotion(self, run_id: str, operation_key: str) -> AutonomousRunState:
        operation = self.repository.get_operation(run_id, operation_key)
        if operation is None or not operation.result:
            raise SupervisorBlocked("promotion intent is incomplete")
        raw_record = operation.result.get("record")
        try:
            record = ExperimentRecord.model_validate(raw_record)
        except Exception as exc:
            raise SupervisorBlocked("promotion record is invalid") from exc
        state = self._reload(run_id)
        if not any(
            item.experiment_number == record.experiment_number for item in state.experiments
        ):
            self.repository.add_experiment(run_id, record)
            state = self._reload(run_id)
        if bool(operation.result.get("promoted")):
            try:
                candidate = MultiRunEvaluation.model_validate(
                    operation.result["candidate_evaluation"]
                )
            except Exception as exc:
                raise SupervisorBlocked("promoted candidate evidence is invalid") from exc
            artifact_ids = record.artifact_ids
            candidate_artifact = operation.result.get("candidate_artifact_id")
            if not isinstance(candidate_artifact, str) or not candidate_artifact:
                raise SupervisorBlocked("promoted checkpoint artifact is missing")
            # Candidate URI/SHA are kept in the phase fields.  Recovery retains
            # the already persisted values if a crash occurred after patching.
            state = self._patch(
                state,
                {
                    "champion_metrics": {
                        "aggregate": candidate.aggregate_score,
                        **candidate.environment_scores,
                    },
                    "champion_artifact_ids": tuple(artifact_ids),
                    "champion_checkpoint_uri": operation.result.get("candidate_uri"),
                    "champion_checkpoint_sha256": operation.result.get("candidate_sha256"),
                    "metadata": {
                        **state.metadata,
                        "champion_evaluation_b64": self._encode_evaluation(candidate),
                    },
                },
            )
        terminal = self.repository.get_operation(run_id, operation_key)
        if terminal is not None and terminal.status is RunOperationStatus.INTENT:
            self.repository.record_operation_result(
                run_id,
                operation_key,
                status=RunOperationStatus.SUCCEEDED,
                result=dict(terminal.result),
            )
        return self._reload(run_id)

    def _recover_promotion_intents(self, state: AutonomousRunState) -> AutonomousRunState:
        for number in range(1, state.max_experiments + 1):
            key = canonical_operation_key(state.run_id, number, "promotion")
            operation = self.repository.get_operation(state.run_id, key)
            if operation is not None and operation.status is RunOperationStatus.INTENT:
                state = self._complete_promotion(state.run_id, key)
        return state

    async def _run_dataset(
        self, state: AutonomousRunState, plan: CuratedDatasetPlan, number: int
    ) -> DatasetArtifact:
        operation_key = canonical_operation_key(state.run_id, number, "dataset")
        request_digest = canonical_request_hash(plan.model_dump(mode="json"))
        operation = self.repository.put_operation_intent(
            RunOperation(
                operation_key=operation_key,
                run_id=state.run_id,
                experiment_number=number,
                phase=RunPhase.CURATION,
                provider_name="objective",
                request_digest=request_digest,
            )
        )
        if operation.status is RunOperationStatus.SUCCEEDED:
            return self._decode_dataset(operation.result)
        self._check_budget(state, "dataset")
        try:
            dataset = await _await(
                self.objective.build_dataset(state, plan, experiment_number=number)
            )
            dataset = await _await(
                self.objective.verify_dataset(
                    dataset, run_id=state.run_id, experiment_number=number
                )
            )
        except SupervisorError:
            raise
        except Exception as exc:
            raise SupervisorArtifactFailure("dataset artifact failure") from exc
        self._validate_dataset(dataset, state.run_id, number)
        self.repository.record_operation_result(
            state.run_id,
            operation_key,
            status=RunOperationStatus.SUCCEEDED,
            result={
                "dataset_id": dataset.dataset_id,
                "uri": dataset.uri,
                "sha256": dataset.sha256,
                "artifact_id": dataset.artifact_id,
                "cost_usd": _finite_cost(dataset.cost_usd),
            },
        )
        self._reconcile_cost(self._reload(state.run_id), operation_key, dataset.cost_usd)
        return dataset

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
        elif reason in {"provider failure", "agent failure"} or (
            reason.startswith("provider ") and reason.endswith(" failed")
        ):
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
        encoded = state.metadata.get("champion_evaluation_b64")
        if encoded:
            return self._decode_evaluation(encoded)
        loader = getattr(self.repository, "get_champion_evaluation", None)
        if callable(loader):
            loaded = loader(run_id)
            if isinstance(loaded, MultiRunEvaluation):
                return loaded
        raise SupervisorBlocked("champion evidence unavailable")

    @staticmethod
    def _dataset_refs(state: AutonomousRunState) -> tuple[str, ...]:
        ref = state.metadata.get("dataset_artifact_ref")
        return (ref,) if isinstance(ref, str) and ref else ()

    @staticmethod
    def _validate_benchmark(
        result: BenchmarkEvidence,
        run_id: str,
        number: int,
        *,
        state: AutonomousRunState | None = None,
    ) -> None:
        if (
            result.evaluation.evidence.label.value not in {"LIVE", "PRIOR_VERIFIED_RUN"}
            or not result.evaluation.evidence.verified
        ):
            raise SupervisorBlocked("benchmark evidence is not verified")
        if (
            not result.evaluation.run_id
            or run_id not in result.evaluation.run_id
            or result.evaluation.evidence.manifest_sha256 is None
        ):
            raise SupervisorBlocked("benchmark provenance is incomplete")
        if (
            result.evaluation.evidence.benchmark_id != "service-recovery-v1"
            or result.evaluation.evidence.suite != "AgentGym/AgentEval"
            or result.evaluation.evidence.suite_version != "agent-eval-v1"
        ):
            raise SupervisorBlocked("benchmark provenance does not match approved scope")
        if state is not None:
            evidence = result.evaluation.evidence
            if (
                evidence.manifest_sha256 != state.benchmark_manifest_sha256
                or evidence.seed != state.benchmark_seed
                or evidence.model_id != state.model_id
            ):
                raise SupervisorBlocked("benchmark provenance does not match run scope")
        if not result.trajectory_refs or not result.artifact_ids:
            raise SupervisorBlocked("benchmark artifacts are missing")
        _finite_cost(result.cost_usd)

    @staticmethod
    def _validate_dataset(dataset: DatasetArtifact, run_id: str, number: int) -> None:
        if (
            not dataset.dataset_id
            or not dataset.uri.startswith("s3://")
            or any(char.isspace() for char in dataset.uri)
            or not dataset.artifact_id
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9:/.@+_=,-]{0,511}", dataset.artifact_id)
            or not re.fullmatch(r"[0-9a-f]{64}", dataset.sha256)
        ):
            raise SupervisorBlocked("dataset artifact is incomplete")

    @staticmethod
    def _validate_checkpoint(candidate: CheckpointArtifact) -> None:
        if (
            not candidate.uri.startswith("s3://")
            or any(char.isspace() for char in candidate.uri)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9:/.@+_=,-]{0,511}", candidate.artifact_id)
            or not re.fullmatch(r"[0-9a-f]{64}", candidate.sha256)
        ):
            raise SupervisorBlocked("checkpoint artifact is incomplete")

    @staticmethod
    def _validate_evaluation(
        evidence: EvaluationEvidence,
        number: int,
        *,
        run_id: str | None = None,
        state: AutonomousRunState | None = None,
    ) -> None:
        if (
            evidence.evaluation.run_number != number
            or not evidence.evaluation.evidence.verified
            or evidence.evaluation.evidence.label.value not in {"LIVE", "PRIOR_VERIFIED_RUN"}
            or evidence.evaluation.evidence.benchmark_id != "service-recovery-v1"
            or evidence.evaluation.evidence.suite != "AgentGym/AgentEval"
            or evidence.evaluation.evidence.suite_version != "agent-eval-v1"
        ):
            raise SupervisorBlocked("evaluation evidence is not verified")
        if run_id is not None and run_id not in evidence.evaluation.run_id:
            raise SupervisorBlocked("evaluation evidence belongs to another run")
        if state is not None:
            signed = evidence.evaluation.evidence
            if (
                signed.manifest_sha256 != state.benchmark_manifest_sha256
                or signed.seed != state.benchmark_seed
                or signed.model_id != state.model_id
            ):
                raise SupervisorBlocked("evaluation provenance does not match run scope")
        _finite_cost(evidence.cost_usd)

    def _emit(self, event_type: str, run_id: str, number: int, **kwargs: Any) -> None:
        if self.telemetry is None:
            return
        try:
            reason = {
                "job.submitted": "job submitted",
                "job.completed": "job completed",
                "job.failed": "job failed",
                "promotion.decided": "promotion decided",
                "operation.submitted": "operation submitted",
                "operation.completed": "operation completed",
                "operation.failed": "operation failed",
            }.get(event_type, "phase started")
            self.telemetry.emit(
                event_type,
                run_id=run_id,
                run_number=max(1, number),
                experiment_id=f"{run_id}:{number}",
                reason=reason,
                phase=kwargs.pop("phase", self._reload(run_id).phase.value),
                **kwargs,
            )
        except DurableTelemetryError as exc:
            raise SupervisorTelemetryFailure("durable telemetry failure") from exc
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
