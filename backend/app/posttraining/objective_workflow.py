"""Objective benchmark and SageMaker lifecycle boundaries.

The Strands roles can propose what to measure, but this module accepts only
typed results from an objective worker and AWS job provider.  It never creates
scores, trajectories, checkpoint URIs, or provider job identifiers.  A caller
can therefore use the same boundary for a local contract test and for a live
AWS deployment without accidentally turning an explanation into evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.objective.models import TrajectoryReference
from app.posttraining.models import (
    ArtifactReference,
    Evidence,
    EvidenceKind,
    EvidenceLabel,
)
from app.posttraining.run_history import BenchmarkMetrics
from app.providers.sagemaker import (
    EvaluationJobRequest,
    JobResult,
    JobStatus,
    SageMakerProvider,
    TrainingJobRequest,
)


class ObjectiveBenchmarkRequest(BaseModel):
    """Immutable request sent to a sandboxed objective benchmark worker."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    run_id: str = Field(min_length=1)
    model_uri: str = Field(min_length=1)
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    suite: str = Field(min_length=1)
    suite_version: str = Field(min_length=1)
    seed: int
    num_episodes: int = Field(ge=1)
    split: Literal["train", "replay"]
    output_s3_uri: str = Field(min_length=1)

    @field_validator("model_uri")
    @classmethod
    def require_immutable_model_uri(cls, value: str) -> str:
        parsed = urlparse(value)
        versions = parse_qs(parsed.query).get("versionId", [])
        if (
            parsed.scheme != "s3"
            or not parsed.netloc
            or not parsed.path
            or len(versions) != 1
            or not versions[0]
            or versions[0].strip().lower() == "null"
        ):
            raise ValueError("model_uri must be an immutable S3 object version")
        return value


class ObjectiveBenchmarkResult(BaseModel):
    """Provider-owned objective measurements and immutable artifact references."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    benchmark_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    suite: str = Field(min_length=1)
    suite_version: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    split: str = Field(min_length=1)
    metrics: BenchmarkMetrics
    trajectory_artifact: ArtifactReference | None = None
    trajectory_references: tuple[TrajectoryReference, ...] = Field(default_factory=tuple)
    report_artifact: ArtifactReference | None = None
    manifest_sha256: str | None = None
    evidence_label: EvidenceLabel = EvidenceLabel.EXPLANATION
    verified: bool = False

    @field_validator("manifest_sha256")
    @classmethod
    def validate_manifest(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("manifest_sha256 must be a lowercase 64-character digest")
        return value

    @model_validator(mode="after")
    def validate_evidence(self) -> ObjectiveBenchmarkResult:
        references = self.trajectory_references
        reference_ids = tuple(reference.trajectory_id for reference in references)
        if len(set(reference_ids)) != len(reference_ids):
            raise ValueError("trajectory references must have unique trajectory IDs")
        if any(not reference.verified for reference in references):
            raise ValueError("trajectory references must be verifier-confirmed")
        if self.evidence_label in {EvidenceLabel.LIVE, EvidenceLabel.PRIOR_VERIFIED_RUN}:
            if not self.verified:
                raise ValueError("verified objective evidence must set verified=True")
            if self.manifest_sha256 is None:
                raise ValueError("verified objective evidence requires manifest_sha256")
            if self.trajectory_artifact is None and self.report_artifact is None:
                raise ValueError("verified objective evidence requires an artifact reference")
        return self

    def evidence(self, *, kind: EvidenceKind = EvidenceKind.BENCHMARK) -> Evidence:
        """Convert this result into the provenance record consumed by gates."""

        artifacts = tuple(
            artifact.artifact_id
            for artifact in (self.trajectory_artifact, self.report_artifact)
            if artifact is not None
        )
        return Evidence(
            evidence_id=self.benchmark_id,
            kind=kind,
            label=self.evidence_label,
            artifact_ids=artifacts,
            metrics={"aggregate": self.metrics.aggregate, **self.metrics.per_environment},
            verified=self.verified,
            benchmark_id=self.benchmark_id,
            suite=self.suite,
            suite_version=self.suite_version,
            manifest_sha256=self.manifest_sha256,
            seed=self.seed,
            model_id=self.model_id,
        )


class ObjectiveBenchmarkAdapter(Protocol):
    """Execution boundary owned by the isolated objective worker."""

    def execute_benchmark(self, request: ObjectiveBenchmarkRequest) -> ObjectiveBenchmarkResult: ...


def canonical_manifest_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a manifest without including prompts, task contents, or raw outputs."""

    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def execute_objective_benchmark(
    adapter: ObjectiveBenchmarkAdapter, request: ObjectiveBenchmarkRequest
) -> ObjectiveBenchmarkResult:
    """Run and validate one objective benchmark; no fallback is permitted."""

    result = adapter.execute_benchmark(request)
    if result.run_id != request.run_id:
        raise ValueError("objective result run_id does not match request")
    if (
        result.suite != request.suite
        or result.suite_version != request.suite_version
        or result.model_id != request.model_uri
        or result.model_sha256 != request.model_sha256
        or result.seed != request.seed
        or result.split != request.split
    ):
        raise ValueError("objective result provenance does not match request")
    if request.split in {"train", "replay"} and any(
        reference.split.value != request.split for reference in result.trajectory_references
    ):
        raise ValueError("objective trajectory reference split does not match request")
    return result


@dataclass(frozen=True, slots=True)
class JobWaitPolicy:
    """Bounded provider polling policy; callers inject sleep in tests."""

    max_attempts: int = 120
    poll_interval_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds must be non-negative")


class JobWaitTimeout(TimeoutError):
    """Raised when a SageMaker job did not reach a terminal state in the budget."""


class JobExecutionFailed(RuntimeError):
    """Raised when a required training job fails or is stopped."""


_DEFAULT_WAIT_POLICY = JobWaitPolicy()


def wait_for_training_job(
    provider: SageMakerProvider,
    job_name: str,
    *,
    policy: JobWaitPolicy = _DEFAULT_WAIT_POLICY,
    sleep: Callable[[float], object] = lambda _: None,
) -> JobResult:
    """Poll a training job to a terminal state using a bounded retry budget."""

    for attempt in range(policy.max_attempts):
        result = provider.get_training_status(job_name)
        if result.status in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.STOPPED,
        }:
            return result
        if result.status is JobStatus.UNKNOWN:
            raise JobExecutionFailed(f"SageMaker training job {job_name!r} returned UNKNOWN")
        if attempt + 1 < policy.max_attempts:
            sleep(policy.poll_interval_seconds)
    raise JobWaitTimeout(
        f"SageMaker training job {job_name!r} did not finish after {policy.max_attempts} polls"
    )


def wait_for_evaluation_job(
    provider: SageMakerProvider,
    job_name: str,
    *,
    policy: JobWaitPolicy = _DEFAULT_WAIT_POLICY,
    sleep: Callable[[float], object] = lambda _: None,
) -> JobResult:
    """Poll an evaluation processing job to a terminal state."""

    for attempt in range(policy.max_attempts):
        result = provider.get_evaluation_status(job_name)
        if result.status in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.STOPPED,
        }:
            return result
        if result.status is JobStatus.UNKNOWN:
            raise JobExecutionFailed(f"SageMaker evaluation job {job_name!r} returned UNKNOWN")
        if attempt + 1 < policy.max_attempts:
            sleep(policy.poll_interval_seconds)
    raise JobWaitTimeout(
        f"SageMaker evaluation job {job_name!r} did not finish after {policy.max_attempts} polls"
    )


@dataclass(frozen=True, slots=True)
class TrainingEvaluationResult:
    """Completed provider results for one candidate model."""

    training: JobResult
    evaluation: JobResult


def train_then_evaluate(
    provider: SageMakerProvider,
    training_request: TrainingJobRequest,
    evaluation_request: EvaluationJobRequest,
    *,
    policy: JobWaitPolicy = _DEFAULT_WAIT_POLICY,
    sleep: Callable[[float], object] = lambda _: None,
) -> TrainingEvaluationResult:
    """Submit, wait for, and only then evaluate a candidate checkpoint."""

    submitted_training = provider.submit_training(training_request)
    training = wait_for_training_job(
        provider, submitted_training.job_name, policy=policy, sleep=sleep
    )
    if training.status is not JobStatus.COMPLETED:
        raise JobExecutionFailed(
            f"training job {training.job_name!r} ended in {training.status.value}: "
            f"{training.failure_reason or 'no provider reason'}"
        )

    submitted_evaluation = provider.submit_evaluation(evaluation_request)
    evaluation = wait_for_evaluation_job(
        provider, submitted_evaluation.job_name, policy=policy, sleep=sleep
    )
    if evaluation.status is not JobStatus.COMPLETED:
        raise JobExecutionFailed(
            f"evaluation job {evaluation.job_name!r} ended in {evaluation.status.value}: "
            f"{evaluation.failure_reason or 'no provider reason'}"
        )
    return TrainingEvaluationResult(training=training, evaluation=evaluation)
