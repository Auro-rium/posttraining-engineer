"""Guarded live AWS execution for the five-run post-training demonstration.

This module is deliberately small at the control-plane boundary.  It does not
contain a local or simulated fallback: objective measurements must come from
the configured worker, training/evaluation must come from SageMaker, and a
caller must inject the durable run-slot and final-record adapters.  The CLI
scripts are thin wrappers around these contracts.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import time
from base64 import urlsafe_b64decode, urlsafe_b64encode
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from math import ceil, isfinite
from typing import Any, Literal, Protocol, cast
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.observability import EventType, TelemetryRecorder
from app.posttraining.models import ArtifactKind, ArtifactReference, EvidenceKind, EvidenceLabel
from app.posttraining.multi_run_gate import (
    MultiRunEvaluation,
    MultiRunPromotionGate,
    MultiRunPromotionResult,
)
from app.posttraining.objective_workflow import (
    ObjectiveBenchmarkRequest,
    ObjectiveBenchmarkResult,
    execute_objective_benchmark,
    wait_for_evaluation_job,
    wait_for_training_job,
)
from app.posttraining.run_history import (
    MAX_RUNS,
    RunDecision,
    RunHistoryRecord,
    RunStatus,
)
from app.providers.artifacts import (
    ArtifactIntegrityError,
    ArtifactRef,
    ArtifactStore,
    S3ArtifactStore,
)
from app.providers.repository import DynamoDBRunRepository
from app.providers.sagemaker import (
    EvaluationJobRequest,
    JobResult,
    JobStatus,
    SageMakerProvider,
    TrainingJobRequest,
)

NEMOTRON_MODEL_ID = "nvidia.nemotron-super-3-120b"
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ECR_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
OBJECTIVE_WORKER_TIMEOUT_MAX_SECONDS = 600
LIVE_PROVIDER_POLL_INTERVAL_SECONDS = 30.0
SAGEMAKER_PHASES_PER_EXPERIMENT = 2  # one training job and one evaluation job
_ECR_REGISTRY = re.compile(
    r"^(?P<account>[0-9]{12})\.dkr\.ecr\.(?P<region>[a-z0-9-]+)\.amazonaws\.com$"
)


class LiveExecutionBlocked(RuntimeError):
    """A prerequisite or approval gate prevented cloud execution."""

    def __init__(
        self,
        message: str,
        *,
        classification: PreflightClassification | None = None,
    ) -> None:
        super().__init__(message)
        self.classification = classification


class LiveExecutionFailed(RuntimeError):
    """A live provider operation failed after execution began."""


class ApprovalPacket(BaseModel):
    """The exact execution envelope a human approves.

    A token is valid only for this packet.  Keeping the packet content-addressed
    prevents an operator from approving one GPU/cost/checkpoint combination and
    accidentally executing another after a restart or configuration change.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    packet_version: str = "v1"
    run_id: str = Field(min_length=1)
    run_number: int = Field(ge=1, le=MAX_RUNS)
    instance_type: str = Field(min_length=1)
    instance_count: int = Field(ge=1)
    volume_size_gb: int = Field(ge=1)
    max_runtime_seconds: int = Field(ge=1)
    estimated_cost_usd: float = Field(ge=0)
    immutable_model_revision: str
    max_experiments: int = Field(ge=1, le=MAX_RUNS)
    max_cost_usd: float = Field(ge=0, le=25.0)
    # Defaults preserve the low-level signing helper for generic packets;
    # controller validation rejects empty provenance values for execution.
    target_model: str = ""
    benchmark_id: str = "service-recovery-v1"
    objective_suite: str = ""
    objective_suite_version: str = ""
    seed: int = 0
    reasoning_model_id: str = ""
    # Benchmark cardinality is part of the cost/evidence envelope.  Defaults
    # retain compatibility for low-level signing callers; live construction
    # always fills these from the immutable deployment configuration.
    baseline_episodes: int = Field(default=10, ge=1)
    held_out_episodes: int = Field(default=15, ge=1)
    # The URI is included when the API prepares a production packet.  It is
    # optional only for generic packet tests and legacy signing helpers.
    checkpoint_s3_uri: str = ""
    manifest_sha256: str
    checkpoint_sha256: str
    issued_at: datetime
    expires_at: datetime

    @field_validator("instance_type")
    @classmethod
    def validate_instance_type(cls, value: str) -> str:
        if not _SAFE_NAME.fullmatch(value):
            raise ValueError("instance_type must be a safe identifier")
        return value

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        if not _SAFE_NAME.fullmatch(value):
            raise ValueError("run_id must be a safe identifier")
        return value

    @field_validator("estimated_cost_usd", "max_cost_usd")
    @classmethod
    def validate_finite_cost(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("estimated_cost_usd must be finite")
        return value

    @field_validator("immutable_model_revision")
    @classmethod
    def validate_model_revision(cls, value: str) -> str:
        if not _SHA1.fullmatch(value):
            raise ValueError("immutable_model_revision must be a 40-character commit SHA")
        return value

    @field_validator("manifest_sha256", "checkpoint_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("approval hashes must be lowercase 64-character SHA-256 digests")
        return value

    @field_validator("checkpoint_s3_uri")
    @classmethod
    def validate_checkpoint_uri(cls, value: str) -> str:
        if value:
            parsed = urlparse(value)
            if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
                raise ValueError("checkpoint_s3_uri must be an s3:// URI")
        return value

    @field_validator("issued_at", "expires_at")
    @classmethod
    def require_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("approval timestamps must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_window(self) -> ApprovalPacket:
        if self.packet_version != "v1":
            raise ValueError("unsupported approval packet version")
        if self.expires_at <= self.issued_at:
            raise ValueError("approval expiry must be after issuance")
        now = datetime.now(UTC)
        if self.issued_at > now + timedelta(seconds=30):
            raise ValueError("approval issuance cannot be in the future")
        if self.expires_at - self.issued_at > timedelta(days=1):
            raise ValueError("approval expiry window is too long")
        return self

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def issue_approval_token(packet: ApprovalPacket, secret: str) -> str:
    """Create an HMAC-bound token for an operator approval workflow."""

    if not secret:
        raise ValueError("approval secret must not be empty")
    payload = urlsafe_b64encode(packet.canonical_bytes()).decode("ascii").rstrip("=")
    signature = hmac.new(secret.encode("utf-8"), payload.encode("ascii"), hashlib.sha256)
    return f"{packet.packet_version}.{payload}.{signature.hexdigest()}"


def _decode_approval_token(token: str, secret: str) -> ApprovalPacket:
    """Verify and decode a token without trusting any of its packet fields."""

    if not token or not secret:
        raise LiveExecutionBlocked("a signed per-run approval token is required")
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != "v1":
        raise LiveExecutionBlocked("approval token format is invalid")
    _, encoded, supplied_signature = parts
    expected_signature = hmac.new(
        secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected_signature, supplied_signature):
        raise LiveExecutionBlocked("approval token signature is invalid")
    try:
        padded = encoded + "=" * (-len(encoded) % 4)
        packet = ApprovalPacket.model_validate_json(urlsafe_b64decode(padded).decode("utf-8"))
    except Exception as exc:
        raise LiveExecutionBlocked("approval packet payload is invalid") from exc
    if packet.expires_at <= datetime.now(UTC):
        raise LiveExecutionBlocked("approval token is stale or expired")
    return packet


class PreflightStatus(StrEnum):
    READY = "READY"
    BLOCKED = "BLOCKED"


class PreflightClassification(StrEnum):
    """Machine-readable reason a preflight cannot authorize a run."""

    READY = "READY"
    BLOCKED_CONFIGURATION = "BLOCKED_CONFIGURATION"
    BLOCKED_GPU_ALLOWLIST = "BLOCKED_GPU_ALLOWLIST"
    BLOCKED_GPU_QUOTA = "BLOCKED_GPU_QUOTA"
    BLOCKED_GPU_CAPACITY = "BLOCKED_GPU_CAPACITY"
    BLOCKED_PROVIDER = "BLOCKED_PROVIDER"


class GpuQuotaStatus(StrEnum):
    """Result of the read-only Service Quotas lookup."""

    VERIFIED = "VERIFIED"
    INSUFFICIENT = "INSUFFICIENT"
    UNKNOWN = "UNKNOWN"
    NOT_CONFIGURED = "NOT_CONFIGURED"


class GpuCapacityStatus(StrEnum):
    """What the preflight can safely establish about GPU placement."""

    # SageMaker does not expose a non-mutating on-demand placement reservation
    # API.  VERIFIED therefore means the requested capacity is covered by the
    # account's quota and the type is on our explicit allowlist.  Submission
    # remains the provider's final placement check.
    VERIFIED_BY_QUOTA = "VERIFIED_BY_QUOTA"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


class CheckStatus(StrEnum):
    PASSED = "PASSED"
    BLOCKED = "BLOCKED"
    SKIPPED = "SKIPPED"


class LiveExecutionConfig(BaseModel):
    """All mutable deployment inputs required for a real live run.

    The model is intentionally separate from the service's local settings:
    importing a CLI must not accidentally turn on AWS mode or create clients.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    aws_region: str = "us-east-1"
    artifact_bucket: str
    dynamodb_table: str
    training_role_arn: str
    training_image: str
    evaluation_image: str
    objective_worker_url: str
    objective_worker_auth_token: str | None = Field(default=None, repr=False, exclude=True)
    hf_repo_id: str
    hf_revision: str
    target_model: str = "google/functiongemma-270m-it"
    benchmark_id: str = "service-recovery-v1"
    objective_suite: str = "AgentGym/AgentEval"
    objective_suite_version: str = "agent-eval-v1"
    seed: int = 7
    baseline_episodes: int = Field(default=10, ge=1)
    held_out_episodes: int = Field(default=15, ge=1)
    # /api/live generates and pins each experiment dataset independently.
    # The legacy synchronous controller still requires this static prefix.
    training_input_s3_uri: str | None = None
    evaluation_input_s3_uri: str
    checkpoint_s3_uri: str | None = None
    # Required for run 1; later runs derive this from the promoted artifact.
    checkpoint_sha256: str | None = None
    instance_type: str = "ml.g5.xlarge"
    gpu_instance_allowlist: tuple[str, ...] = ("ml.g5.xlarge",)
    sagemaker_gpu_quota_code: str | None = None
    sagemaker_processing_gpu_quota_code: str | None = None
    minimum_gpu_quota: float = Field(default=1.0, gt=0)
    instance_count: int = Field(default=1, ge=1)
    volume_size_gb: int = Field(default=30, ge=1)
    max_runtime_seconds: int = Field(default=120 * 60, ge=1, le=120 * 60)
    # Declared per-instance estimates; actual AWS charges may differ.
    training_hourly_cost_usd: float = Field(default=1.50, ge=0)
    evaluation_hourly_cost_usd: float = Field(default=1.00, ge=0)
    max_runs: int = Field(default=MAX_RUNS, ge=1, le=MAX_RUNS)
    max_cost_usd: float = Field(default=25.0, ge=0, le=25.0)
    artifact_prefix: str = "post-training"
    approval_token_env: str = "LIVE_APPROVAL_TOKEN"
    approval_secret_env: str = "LIVE_APPROVAL_SECRET"
    approval_ttl_seconds: int = Field(default=86400, ge=60, le=86400)
    objective_worker_timeout_seconds: int = Field(
        default=OBJECTIVE_WORKER_TIMEOUT_MAX_SECONDS,
        ge=1,
        le=OBJECTIVE_WORKER_TIMEOUT_MAX_SECONDS,
    )
    preflight_timeout_seconds: float = Field(default=20.0, gt=0, le=120)

    @field_validator("hf_revision")
    @classmethod
    def require_immutable_hf_revision(cls, value: str) -> str:
        if not _SHA1.fullmatch(value):
            raise ValueError("hf_revision must be a 40-character immutable commit SHA")
        return value

    @field_validator("checkpoint_sha256")
    @classmethod
    def validate_checkpoint_sha256(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256.fullmatch(value):
            raise ValueError("checkpoint_sha256 must be a lowercase 64-character SHA-256 digest")
        return value

    @field_validator("checkpoint_s3_uri")
    @classmethod
    def validate_checkpoint_uri(cls, value: str | None) -> str | None:
        if value is not None:
            parsed = urlparse(value)
            if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
                raise ValueError("checkpoint_s3_uri must be an s3:// URI")
        return value

    @field_validator("objective_worker_url")
    @classmethod
    def require_https_worker(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("objective_worker_url must be an HTTPS URL")
        return value.rstrip("/") + "/"

    @field_validator("artifact_bucket", "dynamodb_table", "instance_type")
    @classmethod
    def require_safe_names(cls, value: str) -> str:
        if not _SAFE_NAME.fullmatch(value):
            raise ValueError("deployment names must be safe non-empty identifiers")
        return value

    @field_validator("gpu_instance_allowlist")
    @classmethod
    def validate_gpu_instance_allowlist(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("gpu_instance_allowlist must contain at least one instance type")
        if any(not _SAFE_NAME.fullmatch(item) for item in value):
            raise ValueError("gpu_instance_allowlist contains an unsafe instance type")
        if len(set(value)) != len(value):
            raise ValueError("gpu_instance_allowlist must not contain duplicates")
        return value

    @field_validator("sagemaker_gpu_quota_code", "sagemaker_processing_gpu_quota_code")
    @classmethod
    def validate_quota_code(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[A-Za-z0-9-]{1,128}", value):
            raise ValueError("SageMaker GPU quota codes must be safe Service Quotas codes")
        return value

    @field_validator("approval_secret_env", "approval_token_env")
    @classmethod
    def validate_env_name(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", value):
            raise ValueError("approval environment variable name is invalid")
        return value

    @model_validator(mode="after")
    def validate_cost_ceiling(self) -> LiveExecutionConfig:
        runtime_hours = self.max_runtime_seconds / 3600
        worst_case = (
            self.max_runs
            * self.instance_count
            * runtime_hours
            * (self.training_hourly_cost_usd + self.evaluation_hourly_cost_usd)
        )
        if worst_case > self.max_cost_usd:
            raise ValueError(
                f"worst-case SageMaker estimate ${worst_case:.2f} exceeds "
                f"the ${self.max_cost_usd:.2f} ceiling"
            )
        if self.approval_ttl_seconds < self.minimum_approval_ttl_seconds:
            raise ValueError(
                "approval_ttl_seconds must cover every bounded training/evaluation pair "
                "in the approved experiment window"
            )
        return self

    @property
    def minimum_approval_ttl_seconds(self) -> int:
        """SageMaker-only worst-case window for every approved experiment pair."""

        return (
            self.max_runs
            * SAGEMAKER_PHASES_PER_EXPERIMENT
            * self.max_runtime_seconds
        )

    @property
    def provider_poll_interval_seconds(self) -> float:
        return LIVE_PROVIDER_POLL_INTERVAL_SECONDS

    @property
    def provider_max_polls(self) -> int:
        """Include the terminal poll after a job reaches its configured time limit."""

        return ceil(self.max_runtime_seconds / self.provider_poll_interval_seconds) + 1

    @property
    def estimated_worst_case_cost_usd(self) -> float:
        return round(
            self.max_runs
            * self.instance_count
            * (self.max_runtime_seconds / 3600)
            * (self.training_hourly_cost_usd + self.evaluation_hourly_cost_usd),
            6,
        )

    @property
    def estimated_run_cost_usd(self) -> float:
        """Worst-case cost for one training plus evaluation pair."""

        return round(
            self.instance_count
            * (self.max_runtime_seconds / 3600)
            * (self.training_hourly_cost_usd + self.evaluation_hourly_cost_usd),
            6,
        )

    @property
    def phase_cost_upper_bounds_usd(self) -> dict[str, float]:
        """Declared per-job reservation estimates, not a guaranteed AWS bill cap.

        Each phase uses its configured hourly estimate for the full SageMaker
        runtime and multiplies by the requested instance count. Round up to a
        micro-dollar so precision handling cannot reduce the declared reserve.
        """

        instance_hours = self.max_runtime_seconds * self.instance_count / 3600
        microdollars_per_dollar = 1_000_000
        return {
            "training": (
                ceil(
                    self.training_hourly_cost_usd
                    * instance_hours
                    * microdollars_per_dollar
                )
                / microdollars_per_dollar
            ),
            "evaluation": (
                ceil(
                    self.evaluation_hourly_cost_usd
                    * instance_hours
                    * microdollars_per_dollar
                )
                / microdollars_per_dollar
            ),
        }


# The name used by the live-plan contract remains available to callers.
LiveRunConfig = LiveExecutionConfig


class CheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    status: CheckStatus
    detail: str
    metadata: dict[str, str] = Field(default_factory=dict)
    classification: PreflightClassification | None = None


class PreflightReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: PreflightStatus
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    region: str
    estimated_worst_case_cost_usd: float
    checks: tuple[CheckResult, ...]
    classification: PreflightClassification = PreflightClassification.READY
    blocked_classifications: tuple[PreflightClassification, ...] = ()
    gpu_instance_type: str = ""
    gpu_instance_allowlist: tuple[str, ...] = ()
    gpu_quota_status: GpuQuotaStatus = GpuQuotaStatus.UNKNOWN
    gpu_capacity_status: GpuCapacityStatus = GpuCapacityStatus.UNKNOWN

    @property
    def ready(self) -> bool:
        return self.status is PreflightStatus.READY

    def safe_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def _safe_detail(exc: BaseException) -> str:
    """Convert provider errors to a stable, non-secret status description."""

    code = getattr(exc, "response", None)
    if isinstance(code, Mapping):
        error = code.get("Error")
        if isinstance(error, Mapping) and error.get("Code"):
            return f"provider error {str(error['Code'])[:80]}"
    return f"{exc.__class__.__name__}: operation unavailable"


def _objective_worker_endpoint(base_url: str, endpoint: str) -> str:
    """Join an objective API route to a worker root or a `/v1/` base URL."""

    parsed_base = urlparse(base_url)
    base_path = parsed_base.path.rstrip("/")
    route = endpoint.lstrip("/")
    if route.startswith("v1/") and base_path.rsplit("/", 1)[-1] == "v1":
        route = route.removeprefix("v1/")
    endpoint_path = f"{base_path}/{route}" if base_path else f"/{route}"
    return parsed_base._replace(path=endpoint_path, params="", query="", fragment="").geturl()


class PreflightRunner:
    """Read-only AWS readiness checks for an immutably staged base model.

    Client factories are injectable.  Tests can verify that no write method is
    called, and production uses lazy boto3 clients. Live readiness never
    contacts the Hugging Face Hub; it validates the pinned revision against
    staged S3 checkpoint metadata and content identity.
    """

    def __init__(
        self,
        config: LiveExecutionConfig,
        *,
        clients: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.clients = dict(clients or {})
        self._gpu_quota_status = GpuQuotaStatus.UNKNOWN
        self._gpu_capacity_status = GpuCapacityStatus.UNKNOWN

    def _client(self, name: str) -> Any:
        if name in self.clients:
            return self.clients[name]
        try:
            import boto3  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise LiveExecutionBlocked("boto3 is required for AWS preflight") from exc
        kwargs: dict[str, Any] = {"region_name": self.config.aws_region}
        if name in {"bedrock", "bedrock-runtime"}:
            try:
                from botocore.config import Config  # type: ignore[import-untyped]
            except ImportError as exc:  # pragma: no cover
                raise LiveExecutionBlocked(
                    "botocore is required for SigV4 Bedrock preflight",
                    classification=PreflightClassification.BLOCKED_PROVIDER,
                ) from exc
            # Explicitly select IAM/SigV4 for both catalog and runtime clients.
            # This remains deterministic if AWS_BEARER_TOKEN_BEDROCK is set to
            # a stale or otherwise contaminated value in the process environment.
            kwargs["config"] = Config(signature_version="v4")
        client = boto3.client(name, **kwargs)
        self.clients[name] = client
        return client

    def _check(self, name: str, operation: Callable[[], Mapping[str, Any] | None]) -> CheckResult:
        try:
            metadata = operation() or {}
            return CheckResult(
                name=name,
                status=CheckStatus.PASSED,
                detail="read-only check passed",
                metadata={str(key): str(value) for key, value in metadata.items()},
            )
        except LiveExecutionBlocked as exc:
            classification = getattr(
                exc, "classification", PreflightClassification.BLOCKED_PROVIDER
            )
            return CheckResult(
                name=name,
                status=CheckStatus.BLOCKED,
                detail=_safe_detail(exc),
                classification=classification,
            )
        except Exception as exc:  # provider failures become a blocked report
            return CheckResult(
                name=name,
                status=CheckStatus.BLOCKED,
                detail=_safe_detail(exc),
                classification=PreflightClassification.BLOCKED_PROVIDER,
            )

    def run(self) -> PreflightReport:
        checks: list[CheckResult] = []
        checks.append(
            self._check(
                "aws_identity",
                lambda: {
                    "account": str(self._client("sts").get_caller_identity().get("Account", "")),
                    "region": self.config.aws_region,
                },
            )
        )
        checks.append(
            self._check(
                "bedrock_model_access",
                lambda: self._check_bedrock_readiness(),
            )
        )
        checks.append(
            self._check(
                "configured_base_model_revision",
                self._check_configured_model_revision,
            )
        )
        checks.append(
            self._check(
                "s3_artifact_bucket",
                lambda: self._check_s3_readiness(),
            )
        )
        checks.append(
            self._check(
                "sagemaker_evaluation_input",
                lambda: self._check_input_readiness("evaluation"),
            )
        )
        checks.append(
            self._check(
                "pinned_checkpoint_artifact",
                lambda: self._check_checkpoint_readiness(),
            )
        )
        checks.append(
            self._check(
                "dynamodb_history_table",
                lambda: self._check_dynamodb_readiness(),
            )
        )
        checks.append(
            self._check(
                "sagemaker_role_and_images",
                lambda: self._check_sagemaker_readiness(),
            )
        )
        checks.append(self._check_gpu_readiness())
        checks.append(
            self._check(
                "objective_worker",
                lambda: self._check_worker_readiness(),
            )
        )
        checks.append(self._check_approval_secret())
        checks.append(
            CheckResult(
                name="cost_ceiling",
                status=(
                    CheckStatus.PASSED
                    if self.config.estimated_worst_case_cost_usd <= self.config.max_cost_usd
                    else CheckStatus.BLOCKED
                ),
                detail=(
                    f"worst-case estimate ${self.config.estimated_worst_case_cost_usd:.2f} "
                    f"of ${self.config.max_cost_usd:.2f}"
                ),
                metadata={"currency": "USD"},
                classification=(
                    None
                    if self.config.estimated_worst_case_cost_usd <= self.config.max_cost_usd
                    else PreflightClassification.BLOCKED_CONFIGURATION
                ),
            )
        )
        blocked_classifications = tuple(
            dict.fromkeys(
                item.classification
                for item in checks
                if item.status is CheckStatus.BLOCKED and item.classification is not None
            )
        )
        classification = (
            blocked_classifications[0] if blocked_classifications else PreflightClassification.READY
        )
        status = (
            PreflightStatus.READY
            if all(item.status is CheckStatus.PASSED for item in checks)
            else PreflightStatus.BLOCKED
        )
        return PreflightReport(
            status=status,
            region=self.config.aws_region,
            estimated_worst_case_cost_usd=self.config.estimated_worst_case_cost_usd,
            checks=tuple(checks),
            classification=classification,
            blocked_classifications=blocked_classifications,
            gpu_instance_type=self.config.instance_type,
            gpu_instance_allowlist=self.config.gpu_instance_allowlist,
            gpu_quota_status=self._gpu_quota_status,
            gpu_capacity_status=self._gpu_capacity_status,
        )

    def _check_approval_secret(self) -> CheckResult:
        """Require the one-run approval secret without exposing its value."""

        secret_name = self.config.approval_secret_env
        secret = os.getenv(secret_name, "")
        if not secret:
            return CheckResult(
                name="approval_secret",
                status=CheckStatus.BLOCKED,
                detail="approval secret is not configured",
                classification=PreflightClassification.BLOCKED_CONFIGURATION,
            )
        return CheckResult(
            name="approval_secret",
            status=CheckStatus.PASSED,
            detail="approval secret is configured",
            metadata={"binding": "HMAC-SHA256"},
        )

    def _check_configured_model_revision(self) -> Mapping[str, Any]:
        """Report the immutable model identity configured for the staged bundle."""

        if not _SHA1.fullmatch(self.config.hf_revision):
            raise LiveExecutionBlocked(
                "HF_REVISION must be a 40-character immutable commit SHA",
                classification=PreflightClassification.BLOCKED_CONFIGURATION,
            )
        return {
            "repo_id": self.config.hf_repo_id,
            "revision": self.config.hf_revision,
            "source": "configured_and_staged_s3",
        }

    def _check_s3_readiness(self) -> Mapping[str, Any]:
        client = self._client("s3")
        # Do not call HeadBucket here: it requires s3:ListBucket without a
        # prefix condition, while the deployed coordinator role deliberately
        # grants only prefix-scoped ListBucket access. The bucket-scoped
        # location, encryption, and versioning probes below establish the
        # required bucket properties without widening that IAM policy.
        location = client.get_bucket_location(Bucket=self.config.artifact_bucket).get(
            "LocationConstraint"
        )
        bucket_region = (
            "us-east-1"
            if location in (None, "")
            else "eu-west-1"
            if location == "EU"
            else str(location)
        )
        if bucket_region != self.config.aws_region:
            raise LiveExecutionBlocked(
                "S3 artifact bucket region does not match AWS_REGION",
                classification=PreflightClassification.BLOCKED_CONFIGURATION,
            )
        encryption = client.get_bucket_encryption(Bucket=self.config.artifact_bucket)
        rules = encryption.get("ServerSideEncryptionConfiguration", {}).get("Rules", [])
        algorithms = {
            str(rule.get("ApplyServerSideEncryptionByDefault", {}).get("SSEAlgorithm", ""))
            for rule in rules
            if isinstance(rule, Mapping)
        }
        supported_encryption = algorithms.intersection({"AES256", "aws:kms"})
        if not supported_encryption:
            raise LiveExecutionBlocked(
                "S3 artifact bucket encryption is not SSE-S3 or SSE-KMS",
                classification=PreflightClassification.BLOCKED_CONFIGURATION,
            )
        versioning = client.get_bucket_versioning(Bucket=self.config.artifact_bucket)
        if versioning.get("Status") != "Enabled":
            raise LiveExecutionBlocked(
                "S3 bucket versioning is not enabled",
                classification=PreflightClassification.BLOCKED_CONFIGURATION,
            )
        return {
            "bucket": self.config.artifact_bucket,
            "region": bucket_region,
            "encryption": sorted(supported_encryption)[0],
            "versioning": "Enabled",
        }

    def _check_input_readiness(self, split: str) -> Mapping[str, Any]:
        """Verify the configured SageMaker input prefix exists and is readable."""

        input_fields = {
            "training": "training_input_s3_uri",
            "evaluation": "evaluation_input_s3_uri",
        }
        field_name = input_fields.get(split)
        if field_name is None:
            raise ValueError("unsupported SageMaker input split")
        uri = str(getattr(self.config, field_name))
        parsed = urlparse(uri)
        expected_prefix = self.config.artifact_prefix.strip("/")
        path = parsed.path.strip("/")
        if (
            parsed.scheme != "s3"
            or parsed.netloc != self.config.artifact_bucket
            or not path
            or parsed.query
            or parsed.fragment
            or ".." in path.split("/")
            or not expected_prefix
            or not path.startswith(f"{expected_prefix}/")
        ):
            raise LiveExecutionBlocked(
                f"{split} input must use the configured artifact bucket and prefix",
                classification=PreflightClassification.BLOCKED_CONFIGURATION,
            )

        prefix = path.rstrip("/") + "/"
        response = self._client("s3").list_objects_v2(
            Bucket=self.config.artifact_bucket,
            Prefix=prefix,
            MaxKeys=1,
        )
        contents = response.get("Contents", [])
        if not isinstance(contents, list) or not any(
            isinstance(item, Mapping)
            and isinstance(item.get("Key"), str)
            and type(item.get("Size")) is int
            and item["Size"] > 0
            for item in contents
        ):
            raise LiveExecutionBlocked(
                f"{split} input prefix is empty",
                classification=PreflightClassification.BLOCKED_CONFIGURATION,
            )
        return {"status": "available"}

    def _check_checkpoint_readiness(self) -> Mapping[str, Any]:
        uri = self.config.checkpoint_s3_uri
        digest = self.config.checkpoint_sha256
        if not uri or not digest:
            raise LiveExecutionBlocked(
                "CHECKPOINT_S3_URI and CHECKPOINT_SHA256 are required for a live run",
                classification=PreflightClassification.BLOCKED_CONFIGURATION,
            )
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
            raise LiveExecutionBlocked(
                "checkpoint_s3_uri must be an s3:// URI",
                classification=PreflightClassification.BLOCKED_CONFIGURATION,
            )
        if parsed.netloc != self.config.artifact_bucket:
            raise LiveExecutionBlocked(
                "checkpoint artifact bucket must match the validated artifact bucket",
                classification=PreflightClassification.BLOCKED_CONFIGURATION,
            )
        version_values = parse_qs(parsed.query, keep_blank_values=True).get("versionId", [])
        if len(version_values) != 1 or not version_values[0] or version_values[0] == "null":
            raise LiveExecutionBlocked(
                "checkpoint_s3_uri must include one immutable versionId",
                classification=PreflightClassification.BLOCKED_CONFIGURATION,
            )
        version_id = version_values[0]
        response = self._client("s3").head_object(
            Bucket=parsed.netloc,
            Key=parsed.path.lstrip("/"),
            VersionId=version_id,
        )
        metadata = response.get("Metadata", {})
        observed = str(metadata.get("sha256", "")) if isinstance(metadata, Mapping) else ""
        if observed != digest:
            raise LiveExecutionBlocked(
                "checkpoint S3 metadata digest does not match CHECKPOINT_SHA256",
                classification=PreflightClassification.BLOCKED_CONFIGURATION,
            )
        observed_model_id = (
            str(metadata.get("model-id", "")) if isinstance(metadata, Mapping) else ""
        )
        if observed_model_id != self.config.hf_repo_id:
            raise LiveExecutionBlocked(
                "checkpoint S3 metadata model id does not match HF_REPO_ID",
                classification=PreflightClassification.BLOCKED_CONFIGURATION,
            )
        observed_revision = (
            str(metadata.get("hf-revision", "")) if isinstance(metadata, Mapping) else ""
        )
        if observed_revision != self.config.hf_revision:
            raise LiveExecutionBlocked(
                "checkpoint S3 metadata revision does not match HF_REVISION",
                classification=PreflightClassification.BLOCKED_CONFIGURATION,
            )
        observed_version_id = response.get("VersionId")
        if not observed_version_id or str(observed_version_id) != version_id:
            raise LiveExecutionBlocked(
                "checkpoint artifact version identity could not be verified",
                classification=PreflightClassification.BLOCKED_CONFIGURATION,
            )
        return {
            "checkpoint": "verified",
            "repo_id": observed_model_id,
            "revision": observed_revision,
            "version_id": version_id,
        }

    def _check_bedrock_readiness(self) -> Mapping[str, Any]:
        response = self._client("bedrock").get_foundation_model(modelIdentifier=NEMOTRON_MODEL_ID)
        if not isinstance(response, Mapping):
            raise ValueError("Bedrock catalog returned an invalid response")
        summary = response.get("modelDetails", {})
        runtime_response = self._client("bedrock-runtime").converse(
            modelId=NEMOTRON_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": "Reply with OK."}]}],
            inferenceConfig={"maxTokens": 1, "temperature": 0.0},
        )
        if not isinstance(runtime_response, Mapping) or not runtime_response.get("output"):
            raise ValueError("Bedrock Nemotron runtime probe returned no output")
        return {
            "model_id": NEMOTRON_MODEL_ID,
            "region": self.config.aws_region,
            "provider": str(summary.get("providerName", "unknown")),
            "invocation": "verified",
        }

    def _check_dynamodb_readiness(self) -> Mapping[str, Any]:
        description = self._client("dynamodb").describe_table(TableName=self.config.dynamodb_table)
        table = description.get("Table", {})
        if table.get("TableStatus") != "ACTIVE":
            raise ValueError("DynamoDB table is not ACTIVE")
        raw_schema = table.get("KeySchema", [])
        schema = {
            str(item.get("AttributeName", "")): str(item.get("KeyType", ""))
            for item in raw_schema
            if isinstance(item, Mapping)
        } if isinstance(raw_schema, list) else {}
        if schema != {"pk": "HASH", "sk": "RANGE"} or len(raw_schema) != 2:
            raise LiveExecutionBlocked(
                "DynamoDB table key schema must use pk HASH and sk RANGE",
                classification=PreflightClassification.BLOCKED_CONFIGURATION,
            )
        return {
            "table": self.config.dynamodb_table,
            "status": str(table.get("TableStatus")),
            "key_schema": "pk/sk",
        }

    def _check_sagemaker_readiness(self) -> Mapping[str, Any]:
        identity = self._client("sts").get_caller_identity()
        account_id = str(identity.get("Account", ""))
        if not re.fullmatch(r"[0-9]{12}", account_id):
            raise LiveExecutionBlocked(
                "AWS identity did not return an account id",
                classification=PreflightClassification.BLOCKED_CONFIGURATION,
            )
        role_match = re.match(
            r"^arn:aws:iam::(?P<account>[0-9]{12}):role/.+$", self.config.training_role_arn
        )
        if role_match is None or role_match.group("account") != account_id:
            raise LiveExecutionBlocked(
                "SageMaker training role is not owned by the active AWS account",
                classification=PreflightClassification.BLOCKED_CONFIGURATION,
            )
        role_name = self.config.training_role_arn.split(":role/", 1)[1]
        role_response = self._client("iam").get_role(RoleName=role_name)
        role = role_response.get("Role", {})
        assume_policy = (
            role.get("AssumeRolePolicyDocument", {}) if isinstance(role, Mapping) else {}
        )
        statements = (
            assume_policy.get("Statement", []) if isinstance(assume_policy, Mapping) else []
        )
        if isinstance(statements, Mapping):
            statements = [statements]
        trust_verified = False
        if isinstance(statements, list):
            for statement in statements:
                if not isinstance(statement, Mapping) or statement.get("Effect") != "Allow":
                    continue
                actions = statement.get("Action", [])
                if isinstance(actions, str):
                    actions = [actions]
                principal = statement.get("Principal", {})
                services = principal.get("Service", []) if isinstance(principal, Mapping) else []
                if isinstance(services, str):
                    services = [services]
                if (
                    isinstance(actions, list)
                    and "sts:AssumeRole" in actions
                    and isinstance(services, list)
                    and any(
                        service in {"sagemaker.amazonaws.com", "sagemaker.amazonaws.com.cn"}
                        for service in services
                    )
                ):
                    trust_verified = True
                    break
        if not trust_verified:
            raise LiveExecutionBlocked(
                "SageMaker training role lacks the required SageMaker service trust",
                classification=PreflightClassification.BLOCKED_CONFIGURATION,
            )
        ecr = self._client("ecr")
        for image in (self.config.training_image, self.config.evaluation_image):
            image_match = re.match(
                r"^(?P<registry>[^/]+)/(?P<repository>[^@]+)@"
                r"(?P<digest>sha256:[0-9a-f]{64})$",
                image,
            )
            if image_match is None or not _ECR_DIGEST.fullmatch(image_match.group("digest")):
                raise LiveExecutionBlocked(
                    "SageMaker images must be digest-pinned ECR images",
                    classification=PreflightClassification.BLOCKED_CONFIGURATION,
                )
            registry = image_match.group("registry")
            registry_match = _ECR_REGISTRY.fullmatch(registry)
            if registry_match is None:
                raise LiveExecutionBlocked(
                    "SageMaker image registry is not an AWS ECR registry",
                    classification=PreflightClassification.BLOCKED_CONFIGURATION,
                )
            if registry_match.group("account") != account_id:
                raise LiveExecutionBlocked(
                    "SageMaker image ECR account is not owned by the active AWS account",
                    classification=PreflightClassification.BLOCKED_CONFIGURATION,
                )
            if registry_match.group("region") != self.config.aws_region:
                raise LiveExecutionBlocked(
                    "SageMaker image ECR region does not match AWS_REGION",
                    classification=PreflightClassification.BLOCKED_CONFIGURATION,
                )
            repository = image_match.group("repository")
            digest = image_match.group("digest")
            result = ecr.describe_images(
                repositoryName=repository,
                imageIds=[{"imageDigest": digest}],
            )
            details = result.get("imageDetails", [])
            if not isinstance(details, list) or not details:
                raise LiveExecutionBlocked(
                    "digest-pinned ECR image was not found",
                    classification=PreflightClassification.BLOCKED_CONFIGURATION,
                )
            detail = details[0]
            if not isinstance(detail, Mapping):
                raise ValueError("ECR image metadata is invalid")
            if str(detail.get("imageDigest", "")) != digest:
                raise LiveExecutionBlocked(
                    "ECR image digest identity did not match the requested digest",
                    classification=PreflightClassification.BLOCKED_CONFIGURATION,
                )
            if str(detail.get("repositoryName", "")) != repository:
                raise LiveExecutionBlocked(
                    "ECR image repository identity did not match the requested repository",
                    classification=PreflightClassification.BLOCKED_CONFIGURATION,
                )
            if str(detail.get("registryId", "")) != account_id:
                raise LiveExecutionBlocked(
                    "ECR image registry ownership could not be verified",
                    classification=PreflightClassification.BLOCKED_CONFIGURATION,
                )
        # There is no read-only SageMaker API that reserves on-demand
        # capacity.  GPU allowlist/quota checks below provide the safe,
        # non-mutating readiness signal; create_training_job remains the final
        # provider-side capacity decision after human approval.
        return {"images": "available", "role": "available"}

    def _check_gpu_readiness(self) -> CheckResult:
        """Verify GPU policy and account quota without creating a job.

        Service Quotas reports account-level capacity, not a reservation.  We
        expose that distinction in ``gpu_capacity_status`` so the UI and
        approval packet never imply that SageMaker placement is guaranteed.
        """

        if self.config.instance_type not in self.config.gpu_instance_allowlist:
            self._gpu_quota_status = GpuQuotaStatus.UNKNOWN
            self._gpu_capacity_status = GpuCapacityStatus.UNAVAILABLE
            return CheckResult(
                name="gpu_allowlist",
                status=CheckStatus.BLOCKED,
                detail="requested GPU instance is not in the explicit allowlist",
                metadata={
                    "instance_type": self.config.instance_type,
                    "allowlist": ",".join(self.config.gpu_instance_allowlist),
                },
                classification=PreflightClassification.BLOCKED_GPU_ALLOWLIST,
            )

        quota_codes = (
            ("training", self.config.sagemaker_gpu_quota_code),
            ("processing", self.config.sagemaker_processing_gpu_quota_code),
        )
        missing_quotas = [name for name, code in quota_codes if not code]
        if missing_quotas:
            self._gpu_quota_status = GpuQuotaStatus.NOT_CONFIGURED
            self._gpu_capacity_status = GpuCapacityStatus.UNKNOWN
            return CheckResult(
                name="gpu_quota",
                status=CheckStatus.BLOCKED,
                detail=(
                    "SageMaker GPU Service Quotas code is not configured for: "
                    + ", ".join(missing_quotas)
                ),
                metadata={
                    "instance_type": self.config.instance_type,
                    "missing_quota_types": ",".join(missing_quotas),
                },
                classification=PreflightClassification.BLOCKED_CONFIGURATION,
            )

        quota_values: dict[str, float] = {}
        try:
            client = self._client("service-quotas")
            for quota_type, quota_code in quota_codes:
                if quota_code is None:
                    continue
                response = client.get_service_quota(
                    ServiceCode="sagemaker",
                    QuotaCode=quota_code,
                )
                quota = response.get("Quota", {})
                quota_value = float(quota.get("Value"))
                if not isfinite(quota_value) or quota_value < 0:
                    raise ValueError(
                        f"SageMaker {quota_type} GPU quota value is not finite"
                    )
                quota_values[quota_type] = quota_value
        except Exception as exc:
            self._gpu_quota_status = GpuQuotaStatus.UNKNOWN
            self._gpu_capacity_status = GpuCapacityStatus.UNKNOWN
            return CheckResult(
                name="gpu_quota",
                status=CheckStatus.BLOCKED,
                detail=_safe_detail(exc),
                metadata={
                    "instance_type": self.config.instance_type,
                    **{
                        f"{quota_type}_quota_value": str(value)
                        for quota_type, value in quota_values.items()
                    },
                },
                classification=PreflightClassification.BLOCKED_GPU_QUOTA,
            )

        required_capacity = max(
            self.config.minimum_gpu_quota, float(self.config.instance_count)
        )
        metadata = {
            "instance_type": self.config.instance_type,
            "training_quota_code": quota_codes[0][1] or "",
            "training_quota_value": str(quota_values["training"]),
            "processing_quota_code": quota_codes[1][1] or "",
            "processing_quota_value": str(quota_values["processing"]),
            "required_capacity": str(required_capacity),
        }
        insufficient = [
            quota_type
            for quota_type, value in quota_values.items()
            if value < required_capacity
        ]
        if insufficient:
            self._gpu_quota_status = GpuQuotaStatus.INSUFFICIENT
            self._gpu_capacity_status = GpuCapacityStatus.UNAVAILABLE
            metadata["capacity_status"] = self._gpu_capacity_status.value
            metadata["insufficient_quota_types"] = ",".join(insufficient)
            return CheckResult(
                name="gpu_quota",
                status=CheckStatus.BLOCKED,
                detail="SageMaker GPU quota is insufficient for: " + ", ".join(insufficient),
                metadata=metadata,
                classification=PreflightClassification.BLOCKED_GPU_QUOTA,
            )
        self._gpu_quota_status = GpuQuotaStatus.VERIFIED
        self._gpu_capacity_status = GpuCapacityStatus.VERIFIED_BY_QUOTA
        metadata["capacity_status"] = self._gpu_capacity_status.value
        return CheckResult(
            name="gpu_quota",
            status=CheckStatus.PASSED,
            detail="GPU instance is allowlisted and covered by account quota",
            metadata=metadata,
        )

    def _check_worker_readiness(self) -> Mapping[str, Any]:
        token = self.config.objective_worker_auth_token
        if not token:
            raise LiveExecutionBlocked(
                "objective worker authentication token is required",
                classification=PreflightClassification.BLOCKED_CONFIGURATION,
            )
        response = _http_json(
            _objective_worker_endpoint(self.config.objective_worker_url, "v1/health"),
            method="GET",
            timeout=self.config.preflight_timeout_seconds,
            headers={"Authorization": f"Bearer {token}"},
        )
        if not isinstance(response, Mapping) or str(response.get("status", "")).lower() not in {
            "ok",
            "ready",
            "healthy",
        }:
            raise ValueError("objective worker did not report ready")
        headers = {"Authorization": f"Bearer {token}"}
        try:
            readiness = _http_json(
                _objective_worker_endpoint(self.config.objective_worker_url, "v1/readiness"),
                method="GET",
                timeout=self.config.preflight_timeout_seconds,
                headers=headers,
            )
        except Exception as exc:
            raise LiveExecutionBlocked(
                "objective worker execution readiness could not be verified",
                classification=PreflightClassification.BLOCKED_CONFIGURATION,
            ) from exc
        capabilities = readiness.get("capabilities", {}) if isinstance(readiness, Mapping) else {}
        if (
            not isinstance(readiness, Mapping)
            or readiness.get("status") != "ready"
            or readiness.get("service") != "objective-worker"
            or not isinstance(capabilities, Mapping)
            or capabilities.get("benchmark") is not True
            or capabilities.get("verify-curation") is not True
        ):
            raise LiveExecutionBlocked(
                "objective worker execution readiness is incomplete",
                classification=PreflightClassification.BLOCKED_CONFIGURATION,
            )
        _verify_objective_worker_auth(
            self.config.objective_worker_url,
            timeout=self.config.preflight_timeout_seconds,
            headers=headers,
        )
        return {"endpoint": "configured", "status": "ready", "capabilities": "verified"}


def _http_json(
    url: str,
    *,
    method: str,
    payload: Mapping[str, Any] | None = None,
    timeout: float,
    headers: Mapping[str, str] | None = None,
) -> Any:
    data = None
    request_headers = {"Accept": "application/json"}
    if headers is not None:
        request_headers.update(headers)
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=request_headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        body = response.read()
    if not body:
        return {}
    return json.loads(body.decode("utf-8"))


def _verify_objective_worker_auth(
    base_url: str,
    *,
    timeout: float,
    headers: Mapping[str, str],
) -> None:
    """Prove worker auth with its dedicated read-only probe endpoint."""

    response = _http_json(
        _objective_worker_endpoint(base_url, "v1/auth-probe"),
        method="GET",
        timeout=timeout,
        headers=headers,
    )
    if response != {"status": "authenticated", "service": "objective-worker"}:
        raise LiveExecutionBlocked(
            "objective worker authentication probe returned an unexpected response",
            classification=PreflightClassification.BLOCKED_CONFIGURATION,
        )


class ObjectiveWorkerClient:
    """HTTP adapter for the isolated objective worker; no local fallback."""

    def __init__(
        self,
        base_url: str,
        *,
        auth_token: str | None = None,
        timeout_seconds: float = OBJECTIVE_WORKER_TIMEOUT_MAX_SECONDS,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("objective worker URL must be HTTPS")
        self.base_url = base_url.rstrip("/") + "/"
        if auth_token is not None and not auth_token.strip():
            raise ValueError("objective worker auth token must not be empty")
        if not 0 < timeout_seconds <= OBJECTIVE_WORKER_TIMEOUT_MAX_SECONDS:
            raise ValueError(
                "objective worker timeout_seconds must be greater than zero and at most 600"
            )
        self._auth_token = auth_token
        self.timeout_seconds = timeout_seconds

    def _headers(self) -> dict[str, str]:
        if not self._auth_token:
            raise LiveExecutionBlocked(
                "objective worker authentication token is required",
                classification=PreflightClassification.BLOCKED_CONFIGURATION,
            )
        return {"Authorization": f"Bearer {self._auth_token}"}

    def health(self) -> Mapping[str, Any]:
        response = _http_json(
            _objective_worker_endpoint(self.base_url, "v1/health"),
            method="GET",
            timeout=self.timeout_seconds,
            headers=self._headers(),
        )
        if not isinstance(response, Mapping):
            raise LiveExecutionFailed("objective worker returned a non-object health response")
        if str(response.get("status", "")).lower() not in {"ok", "ready", "healthy"}:
            raise LiveExecutionFailed("objective worker did not report ready")
        self._require_auth_probe()
        return response

    def _require_auth_probe(self) -> None:
        _verify_objective_worker_auth(
            self.base_url,
            timeout=self.timeout_seconds,
            headers=self._headers(),
        )

    def execute_benchmark(self, request: ObjectiveBenchmarkRequest) -> ObjectiveBenchmarkResult:
        response = _http_json(
            _objective_worker_endpoint(self.base_url, "v1/benchmark"),
            method="POST",
            timeout=self.timeout_seconds,
            headers=self._headers(),
            payload=request.model_dump(mode="json"),
        )
        if not isinstance(response, Mapping):
            raise LiveExecutionFailed("objective worker returned a non-object benchmark response")
        return ObjectiveBenchmarkResult.model_validate(response)

    def verify_curation(
        self,
        *,
        run_id: str,
        experiment_id: str,
        trajectory_references: Sequence[Any],
    ) -> Mapping[str, Any]:
        """Ask the isolated objective worker to replay and admit a dataset."""

        from app.objective.models import ObjectiveSplit, TrajectoryReference

        try:
            references = tuple(
                item
                if isinstance(item, TrajectoryReference)
                else TrajectoryReference.model_validate(item)
                for item in trajectory_references
            )
        except Exception as exc:
            raise LiveExecutionFailed(
                "objective curation received malformed trajectory provenance"
            ) from exc
        if not references:
            raise LiveExecutionFailed("objective curation requires selected trajectory references")
        splits = {reference.split for reference in references}
        if len(splits) != 1 or not splits.issubset(
            {ObjectiveSplit.TRAIN, ObjectiveSplit.REPLAY}
        ):
            raise LiveExecutionFailed("objective curation requires one allowed train/replay split")
        if any(not reference.verified for reference in references):
            raise LiveExecutionFailed("objective curation requires verifier-confirmed references")

        response = _http_json(
            _objective_worker_endpoint(self.base_url, "v1/verify-curation"),
            method="POST",
            timeout=self.timeout_seconds,
            headers=self._headers(),
            payload={
                "run_id": run_id,
                "experiment_id": experiment_id,
                "split": references[0].split.value,
                "trajectory_references": [
                    reference.model_dump(mode="json") for reference in references
                ],
            },
        )
        if not isinstance(response, Mapping):
            raise LiveExecutionFailed("objective worker returned a non-object curation response")
        return response

    def replay_corrections(
        self,
        *,
        run_id: str,
        experiment_id: str,
        split: Any,
        proposals: Sequence[Any],
    ) -> Any:
        """Submit untrusted action proposals for deterministic objective replay."""

        from app.objective.models import (
            CorrectionProposal,
            CorrectionReplayRequest,
            CorrectionReplayResponse,
            ObjectiveSplit,
        )

        try:
            scope = ObjectiveSplit(split)
            request = CorrectionReplayRequest(
                run_id=run_id,
                experiment_id=experiment_id,
                split=scope,
                proposals=tuple(
                    item
                    if isinstance(item, CorrectionProposal)
                    else CorrectionProposal.model_validate(item)
                    for item in proposals
                ),
            )
        except Exception as exc:
            raise LiveExecutionFailed("objective correction proposal is malformed") from exc

        response = _http_json(
            _objective_worker_endpoint(self.base_url, "v1/replay-corrections"),
            method="POST",
            timeout=self.timeout_seconds,
            headers=self._headers(),
            payload=request.model_dump(mode="json"),
        )
        try:
            parsed = CorrectionReplayResponse.model_validate(response)
        except Exception as exc:
            raise LiveExecutionFailed(
                "objective worker returned malformed correction replay evidence"
            ) from exc
        if (
            parsed.run_id != run_id
            or parsed.experiment_id != experiment_id
            or parsed.split is not scope
            or tuple(outcome.proposal_id for outcome in parsed.outcomes)
            != tuple(proposal.proposal_id for proposal in request.proposals)
        ):
            raise LiveExecutionFailed("objective correction replay provenance does not match")
        return parsed


class RunSlotStore(Protocol):
    """Durable reservation/finalization boundary supplied by the application."""

    def reserve(
        self,
        *,
        run_id: str,
        run_number: int,
        parent_run_id: str | None,
        champion_run_id: str | None,
    ) -> None: ...

    def finalize(self, record: RunHistoryRecord) -> None: ...


class _RegistrySlotStore:
    """Compatibility adapter for a registry implementation with slot support."""

    def __init__(self, registry: Any) -> None:
        self.registry = registry
        self._reserved: set[str] = set()

    def reserve(
        self,
        *,
        run_id: str,
        run_number: int,
        parent_run_id: str | None,
        champion_run_id: str | None,
    ) -> None:
        reserve_run = getattr(self.registry, "reserve_run", None)
        if callable(reserve_run):
            pending = RunHistoryRecord(
                run_id=run_id,
                run_number=run_number,
                parent_run_id=parent_run_id,
                champion_run_id=champion_run_id,
                status=RunStatus.RUNNING,
            )
            reserve_run(pending, max_runs=MAX_RUNS)
            self._reserved.add(run_id)
            return
        reserve = getattr(self.registry, "reserve_slot", None)
        if not callable(reserve):
            raise LiveExecutionBlocked("durable registry does not expose atomic reserve_slot")
        reserve(
            run_id=run_id,
            run_number=run_number,
            parent_run_id=parent_run_id,
            champion_run_id=champion_run_id,
            max_runs=MAX_RUNS,
        )
        self._reserved.add(run_id)

    def finalize(self, record: RunHistoryRecord) -> None:
        finalize = getattr(self.registry, "finalize_run", None)
        if not callable(finalize):
            raise LiveExecutionBlocked("durable registry does not expose finalize_run")
        finalize(record)


@dataclass(frozen=True, slots=True)
class LiveRunSummary:
    run_id: str
    run_number: int
    status: str
    decision: str | None
    manifest_sha256: str
    training_job_id: str
    evaluation_job_id: str
    candidate_artifact_id: str
    baseline_score: float
    candidate_score: float
    reasons: tuple[str, ...] = ()
    checkpoint_sha256: str | None = None
    approval_packet_sha256: str | None = None
    artifact_refs: tuple[ArtifactReference, ...] = ()
    cleanup_completed: bool = True

    def safe_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_number": self.run_number,
            "status": self.status,
            "decision": self.decision,
            "manifest_sha256": self.manifest_sha256,
            "training_job_id": self.training_job_id,
            "evaluation_job_id": self.evaluation_job_id,
            "candidate_artifact_id": self.candidate_artifact_id,
            "baseline_score": self.baseline_score,
            "candidate_score": self.candidate_score,
            "reasons": list(self.reasons),
            "checkpoint_sha256": self.checkpoint_sha256,
            "approval_packet_sha256": self.approval_packet_sha256,
            "artifact_refs": [item.model_dump(mode="json") for item in self.artifact_refs],
            "cleanup_completed": self.cleanup_completed,
        }


@dataclass(slots=True)
class AutonomousRunController:
    """Run one approved candidate through real objective/SageMaker phases."""

    config: LiveExecutionConfig
    objective_worker: Any
    provider: SageMakerProvider
    artifact_store: ArtifactStore
    slots: RunSlotStore
    telemetry: TelemetryRecorder = field(default_factory=TelemetryRecorder)
    champion_loader: Callable[[int], RunHistoryRecord | None] | None = None
    sleep: Callable[[float], object] = time.sleep
    run_id_factory: Callable[[], str] = lambda: f"run-{uuid4().hex}"
    _owned_job_names: set[str] = field(default_factory=set, init=False, repr=False)
    _used_approval_digests: set[str] = field(default_factory=set, init=False, repr=False)

    def _require_legacy_training_input_uri(self) -> str:
        uri = self.config.training_input_s3_uri
        if not isinstance(uri, str) or not uri.strip():
            raise LiveExecutionBlocked(
                "legacy synchronous execution requires TRAINING_INPUT_S3_URI",
                classification=PreflightClassification.BLOCKED_CONFIGURATION,
            )
        return uri

    def run_once(self, *, run_number: int, approval_token: str) -> LiveRunSummary:
        self._validate_run_number(run_number)
        training_input_s3_uri = self._require_legacy_training_input_uri()
        # Decode and authenticate before touching a provider.  The signed
        # packet carries the run id and manifest digest, so a token cannot be
        # replayed against a different run or configuration.
        packet = self._require_approval(approval_token, run_number=run_number)
        preflight_runner = PreflightRunner(self.config)
        # Normal readiness omits this obsolete static input. Preserve the
        # legacy controller's original gate before writing its run manifest
        # or submitting a static-input SageMaker training job.
        preflight_runner._check_input_readiness("training")
        preflight = preflight_runner.run()
        if not preflight.ready:
            raise LiveExecutionBlocked("preflight is BLOCKED; no cloud job was submitted")

        self._validate_approval_packet(packet, preflight)
        run_id = packet.run_id
        if packet.digest in self._used_approval_digests:
            raise LiveExecutionBlocked("approval token has already been used")

        champion_record = self._load_champion(run_number)
        parent_run_id = champion_record.run_id if champion_record else None
        champion_run_id = champion_record.run_id if champion_record else None
        checkpoint_sha256 = self._checkpoint_sha256(champion_record)
        checkpoint_uri = self._checkpoint_uri(champion_record)
        if packet.checkpoint_sha256 != checkpoint_sha256:
            raise LiveExecutionBlocked(
                "approval packet checkpoint digest does not match the selected artifact"
            )
        manifest_payload = self._manifest_payload(
            run_id=run_id,
            run_number=run_number,
            parent_run_id=parent_run_id,
            champion_run_id=champion_run_id,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_uri=checkpoint_uri,
        )
        manifest_sha256 = hashlib.sha256(
            json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if packet.manifest_sha256 != manifest_sha256:
            raise LiveExecutionBlocked("approval packet does not match the immutable run manifest")
        self.slots.reserve(
            run_id=run_id,
            run_number=run_number,
            parent_run_id=parent_run_id,
            champion_run_id=champion_run_id,
        )
        self._used_approval_digests.add(packet.digest)
        reserved = True
        training_job: JobResult | None = None
        evaluation_job: JobResult | None = None
        manifest_artifact: ArtifactReference | None = None
        job_metadata_artifacts: list[ArtifactReference] = []
        try:
            stored_manifest = self.artifact_store.put_json(
                f"{run_id}/manifest.json",
                manifest_payload,
                metadata={"manifest_sha256": manifest_sha256},
            )
            if stored_manifest.sha256 != manifest_sha256:
                raise LiveExecutionFailed("manifest artifact hash verification failed")
            manifest_artifact = ArtifactReference(
                artifact_id=hashlib.sha256(stored_manifest.version_ref.encode()).hexdigest()[:24],
                kind=ArtifactKind.CONFIGURATION,
                uri=stored_manifest.version_ref,
                sha256=stored_manifest.sha256,
                size_bytes=stored_manifest.size_bytes,
                metadata={"manifest_sha256": manifest_sha256},
            )
            self._event(EventType.RUN_STARTED, run_id, run_number, status="running")
            baseline = self._benchmark(
                run_id=run_id,
                model_uri=checkpoint_uri,
                model_sha256=self._checkpoint_sha256(None),
                split="baseline",
                episodes=self.config.baseline_episodes,
                output_s3_uri=f"s3://{self.config.artifact_bucket}/{self.config.artifact_prefix}/{run_id}/baseline",
                manifest_sha256=manifest_sha256,
            )
            training_request = TrainingJobRequest(
                job_name=f"apt-{run_id}-train",
                role_arn=self.config.training_role_arn,
                image_uri=self.config.training_image,
                input_s3_uri=training_input_s3_uri,
                output_s3_uri=f"s3://{self.config.artifact_bucket}/{self.config.artifact_prefix}/{run_id}/candidate",
                instance_type=self.config.instance_type,
                instance_count=self.config.instance_count,
                volume_size_gb=self.config.volume_size_gb,
                max_runtime_seconds=self.config.max_runtime_seconds,
                environment={"RUN_ID": run_id, "MANIFEST_SHA256": manifest_sha256},
                tags=[{"Key": "project", "Value": "autonomous-post-training"}],
            )
            training_job = self.provider.submit_training(training_request)
            self._require_provider_id(training_job, "training")
            job_metadata_artifacts.append(
                self._persist_job_metadata(
                    run_id,
                    phase="training",
                    job=training_job,
                    manifest_sha256=manifest_sha256,
                )
            )
            self._event(
                EventType.JOB_SUBMITTED,
                run_id,
                run_number,
                phase="training",
                job_id=str(training_job.provider_job_id),
                status="submitted",
            )
            training_job = wait_for_training_job(
                self.provider,
                training_job.job_name,
                policy=self._wait_policy(),
                sleep=self.sleep,
            )
            if training_job.status is not JobStatus.COMPLETED or not training_job.artifact_uri:
                raise LiveExecutionFailed("training did not complete with a model artifact")
            self._event(
                EventType.JOB_COMPLETED,
                run_id,
                run_number,
                phase="training",
                job_id=str(training_job.provider_job_id),
                status="completed",
            )
            candidate_artifact = self._artifact_from_job(training_job, ArtifactKind.CHECKPOINT)

            evaluation_request = EvaluationJobRequest(
                job_name=f"apt-{run_id}-eval",
                role_arn=self.config.training_role_arn,
                image_uri=self.config.evaluation_image,
                input_s3_uri=self.config.evaluation_input_s3_uri,
                output_s3_uri=f"s3://{self.config.artifact_bucket}/{self.config.artifact_prefix}/{run_id}/evaluation",
                model_s3_uri=training_job.artifact_uri,
                instance_type=self.config.instance_type,
                instance_count=self.config.instance_count,
                volume_size_gb=self.config.volume_size_gb,
                max_runtime_seconds=self.config.max_runtime_seconds,
                environment={"RUN_ID": run_id, "MANIFEST_SHA256": manifest_sha256},
                tags=[{"Key": "project", "Value": "autonomous-post-training"}],
            )
            evaluation_job = self.provider.submit_evaluation(evaluation_request)
            self._require_provider_id(evaluation_job, "evaluation")
            job_metadata_artifacts.append(
                self._persist_job_metadata(
                    run_id,
                    phase="evaluation",
                    job=evaluation_job,
                    manifest_sha256=manifest_sha256,
                )
            )
            self._event(
                EventType.JOB_SUBMITTED,
                run_id,
                run_number,
                phase="evaluation",
                job_id=str(evaluation_job.provider_job_id),
                status="submitted",
            )
            evaluation_job = wait_for_evaluation_job(
                self.provider,
                evaluation_job.job_name,
                policy=self._wait_policy(),
                sleep=self.sleep,
            )
            if evaluation_job.status is not JobStatus.COMPLETED:
                raise LiveExecutionFailed("evaluation did not complete")
            self._event(
                EventType.JOB_COMPLETED,
                run_id,
                run_number,
                phase="evaluation",
                job_id=str(evaluation_job.provider_job_id),
                status="completed",
            )
            candidate = self._benchmark(
                run_id=run_id,
                model_uri=training_job.artifact_uri,
                model_sha256=getattr(training_job, "artifact_sha256", ""),
                split="held_out",
                episodes=self.config.held_out_episodes,
                output_s3_uri=f"s3://{self.config.artifact_bucket}/{self.config.artifact_prefix}/{run_id}/held-out",
                manifest_sha256=manifest_sha256,
            )
            if (
                candidate.evidence_label
                not in {EvidenceLabel.LIVE, EvidenceLabel.PRIOR_VERIFIED_RUN}
                or not candidate.verified
            ):
                raise LiveExecutionFailed("objective evaluation is not verified live evidence")
            gate = self._gate(
                baseline,
                candidate,
                run_id,
                run_number,
                champion_run_id=champion_run_id,
                champion_run_number=(champion_record.run_number if champion_record else 0),
            )
            status = RunStatus.COMPLETED if gate.passed else RunStatus.REJECTED
            decision = RunDecision.PROMOTE if gate.passed else RunDecision.REJECT
            self._event(
                EventType.PROMOTION_DECIDED,
                run_id,
                run_number,
                phase="promotion",
                evidence_label=candidate.evidence_label,
                status=decision.value,
                attributes={
                    "decision": decision.value,
                    "baseline_score": baseline.metrics.aggregate,
                    "candidate_score": candidate.metrics.aggregate,
                },
            )
            record = self._record(
                run_id=run_id,
                run_number=run_number,
                status=status,
                decision=decision,
                parent_run_id=parent_run_id,
                champion_run_id=champion_run_id,
                baseline=baseline,
                candidate=candidate,
                candidate_artifact=candidate_artifact,
                manifest_artifact=manifest_artifact,
                manifest_sha256=manifest_sha256,
                gate=gate,
                job_metadata_artifacts=tuple(job_metadata_artifacts),
            )
            self.slots.finalize(record)
            self._event(EventType.RUN_COMPLETED, run_id, run_number, status=status.value)
            return LiveRunSummary(
                run_id=run_id,
                run_number=run_number,
                status=status.value,
                decision=decision.value,
                manifest_sha256=manifest_sha256,
                training_job_id=str(training_job.provider_job_id),
                evaluation_job_id=str(evaluation_job.provider_job_id),
                candidate_artifact_id=candidate_artifact.artifact_id,
                baseline_score=baseline.metrics.aggregate,
                candidate_score=candidate.metrics.aggregate,
                reasons=gate.reasons,
                checkpoint_sha256=checkpoint_sha256,
                approval_packet_sha256=packet.digest,
                artifact_refs=record.artifact_refs,
            )
        except LiveExecutionBlocked:
            if reserved:
                self._finalize_failure(
                    run_id=run_id,
                    run_number=run_number,
                    parent_run_id=parent_run_id,
                    champion_run_id=champion_run_id,
                    manifest_sha256=manifest_sha256,
                    manifest_artifact=manifest_artifact,
                    job_metadata_artifacts=tuple(job_metadata_artifacts),
                )
            self._event(EventType.RUN_FAILED, run_id, run_number, status="blocked")
            raise
        except Exception as exc:
            if reserved:
                self._finalize_failure(
                    run_id=run_id,
                    run_number=run_number,
                    parent_run_id=parent_run_id,
                    champion_run_id=champion_run_id,
                    manifest_sha256=manifest_sha256,
                    manifest_artifact=manifest_artifact,
                    job_metadata_artifacts=tuple(job_metadata_artifacts),
                )
            self._event(EventType.RUN_FAILED, run_id, run_number, status="failed")
            raise LiveExecutionFailed(
                "live run failed; inspect retained metadata artifacts"
            ) from exc
        finally:
            self._cleanup(training_job, evaluation_job, run_id=run_id, run_number=run_number)

    def _finalize_failure(
        self,
        *,
        run_id: str,
        run_number: int,
        parent_run_id: str | None,
        champion_run_id: str | None,
        manifest_sha256: str,
        manifest_artifact: ArtifactReference | None,
        job_metadata_artifacts: tuple[ArtifactReference, ...] = (),
    ) -> None:
        """Close a reserved slot without inventing metrics or evidence."""

        failed = RunHistoryRecord(
            run_id=run_id,
            run_number=run_number,
            parent_run_id=parent_run_id,
            champion_run_id=champion_run_id,
            status=RunStatus.FAILED,
            manifest_sha256=manifest_sha256,
            artifact_refs=tuple(
                item for item in (manifest_artifact, *job_metadata_artifacts) if item is not None
            ),
            decision_reasons=("live execution failed before verified evidence",),
        )
        try:
            self.slots.finalize(failed)
        except Exception:
            # Do not hide the original provider or validation exception. The
            # repository's pending record remains available for reconciliation.
            pass

    def _persist_job_metadata(
        self,
        run_id: str,
        *,
        phase: str,
        job: JobResult,
        manifest_sha256: str,
    ) -> ArtifactReference:
        """Persist provider IDs before polling so recovery can reconcile them."""

        stored = self.artifact_store.put_json(
            f"{run_id}/{phase}-job.json",
            {
                "run_id": run_id,
                "phase": phase,
                "job_name": job.job_name,
                "provider_job_id": str(job.provider_job_id),
                "manifest_sha256": manifest_sha256,
                "status": job.status.value,
            },
        )
        return ArtifactReference(
            artifact_id=hashlib.sha256(stored.version_ref.encode()).hexdigest()[:24],
            kind=ArtifactKind.CONFIGURATION,
            uri=stored.version_ref,
            sha256=stored.sha256,
            size_bytes=stored.size_bytes,
            metadata={"phase": phase, "provider_job_id": str(job.provider_job_id)},
        )

    @staticmethod
    def _require_provider_id(job: JobResult, phase: str) -> None:
        if not job.provider_job_id:
            raise LiveExecutionFailed(f"{phase} provider returned no job ID")

    def build_approval_packet(
        self,
        *,
        run_number: int,
        run_id: str | None = None,
        issued_at: datetime | None = None,
    ) -> ApprovalPacket:
        """Build the operator-facing packet without submitting a cloud job.

        The caller signs the returned packet with :func:`issue_approval_token`
        (or an external approval service).  This method performs only local
        validation and any injected champion read; it never creates AWS
        resources or starts a job.
        """

        self._validate_run_number(run_number)
        actual_run_id = run_id or self.run_id_factory()
        champion_record = self._load_champion(run_number)
        parent_run_id = champion_record.run_id if champion_record else None
        champion_run_id = champion_record.run_id if champion_record else None
        checkpoint_sha256 = self._checkpoint_sha256(champion_record)
        checkpoint_uri = self._checkpoint_uri(champion_record)
        manifest_payload = self._manifest_payload(
            run_id=actual_run_id,
            run_number=run_number,
            parent_run_id=parent_run_id,
            champion_run_id=champion_run_id,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_uri=checkpoint_uri,
        )
        manifest_sha256 = hashlib.sha256(
            json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        now = issued_at or datetime.now(UTC)
        return ApprovalPacket(
            run_id=actual_run_id,
            run_number=run_number,
            instance_type=self.config.instance_type,
            instance_count=self.config.instance_count,
            volume_size_gb=self.config.volume_size_gb,
            max_runtime_seconds=self.config.max_runtime_seconds,
            estimated_cost_usd=self.config.estimated_run_cost_usd,
            immutable_model_revision=self.config.hf_revision,
            max_experiments=self.config.max_runs,
            max_cost_usd=self.config.max_cost_usd,
            target_model=self.config.target_model,
            benchmark_id=self.config.benchmark_id,
            objective_suite=self.config.objective_suite,
            objective_suite_version=self.config.objective_suite_version,
            seed=self.config.seed,
            reasoning_model_id=NEMOTRON_MODEL_ID,
            baseline_episodes=self.config.baseline_episodes,
            held_out_episodes=self.config.held_out_episodes,
            checkpoint_s3_uri=checkpoint_uri,
            manifest_sha256=manifest_sha256,
            checkpoint_sha256=checkpoint_sha256,
            issued_at=now,
            expires_at=now + timedelta(seconds=self.config.approval_ttl_seconds),
        )

    def _load_champion(self, run_number: int) -> RunHistoryRecord | None:
        if run_number <= 1:
            return None
        if self.champion_loader is None:
            raise LiveExecutionBlocked("a persisted champion loader is required after run 1")
        champion_record = self.champion_loader(run_number)
        if champion_record is None or champion_record.decision is not RunDecision.PROMOTE:
            raise LiveExecutionBlocked("no promoted champion is available for this run")
        if not champion_record.candidate_artifact_id:
            raise LiveExecutionBlocked("promoted champion has no checkpoint artifact")
        return champion_record

    def _checkpoint_sha256(self, champion_record: RunHistoryRecord | None) -> str:
        if champion_record is not None:
            return self._champion_artifact(champion_record).sha256
        if self.config.checkpoint_sha256 is None:
            raise LiveExecutionBlocked(
                "a verified checkpoint_sha256 is required before approving run 1"
            )
        return self.config.checkpoint_sha256

    def _manifest_payload(
        self,
        *,
        run_id: str,
        run_number: int,
        parent_run_id: str | None,
        champion_run_id: str | None,
        checkpoint_sha256: str,
        checkpoint_uri: str,
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "run_number": run_number,
            "parent_run_id": parent_run_id,
            "champion_run_id": champion_run_id,
            "target_model": self.config.target_model,
            "benchmark_id": self.config.benchmark_id,
            "hf_repo_id": self.config.hf_repo_id,
            "hf_revision": self.config.hf_revision,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_s3_uri": checkpoint_uri,
            "training_input_s3_uri": self._require_legacy_training_input_uri(),
            "evaluation_input_s3_uri": self.config.evaluation_input_s3_uri,
            "suite": self.config.objective_suite,
            "suite_version": self.config.objective_suite_version,
            "seed": self.config.seed,
            "baseline_episodes": self.config.baseline_episodes,
            "held_out_episodes": self.config.held_out_episodes,
            "model_reasoning_id": NEMOTRON_MODEL_ID,
            "training_role_arn": self.config.training_role_arn,
            "training_image": self.config.training_image,
            "evaluation_image": self.config.evaluation_image,
            "instance_type": self.config.instance_type,
            "instance_count": self.config.instance_count,
            "volume_size_gb": self.config.volume_size_gb,
            "max_runtime_seconds": self.config.max_runtime_seconds,
            "max_experiments": self.config.max_runs,
            "max_cost_usd": self.config.max_cost_usd,
        }

    def _validate_approval_packet(self, packet: ApprovalPacket, preflight: PreflightReport) -> None:
        if packet.instance_type != self.config.instance_type:
            raise LiveExecutionBlocked("approval packet GPU instance does not match configuration")
        if packet.instance_count != self.config.instance_count:
            raise LiveExecutionBlocked("approval packet GPU count does not match configuration")
        if packet.volume_size_gb != self.config.volume_size_gb:
            raise LiveExecutionBlocked("approval packet volume does not match configuration")
        if packet.max_runtime_seconds != self.config.max_runtime_seconds:
            raise LiveExecutionBlocked("approval packet runtime does not match configuration")
        if packet.estimated_cost_usd != self.config.estimated_run_cost_usd:
            raise LiveExecutionBlocked("approval packet cost does not match configuration")
        if (
            packet.immutable_model_revision != self.config.hf_revision
            or packet.max_experiments != self.config.max_runs
            or packet.max_cost_usd != self.config.max_cost_usd
        ):
            raise LiveExecutionBlocked(
                "approval packet bounded optimization scope does not match configuration"
            )
        if (
            packet.target_model != self.config.target_model
            or packet.benchmark_id != self.config.benchmark_id
            or packet.objective_suite != self.config.objective_suite
            or packet.objective_suite_version != self.config.objective_suite_version
            or packet.seed != self.config.seed
            or packet.reasoning_model_id != NEMOTRON_MODEL_ID
            or packet.baseline_episodes != self.config.baseline_episodes
            or packet.held_out_episodes != self.config.held_out_episodes
        ):
            raise LiveExecutionBlocked(
                "approval packet benchmark provenance does not match configuration"
            )
        if not packet.checkpoint_s3_uri:
            raise LiveExecutionBlocked("approval packet checkpoint URI is required")
        checkpoint_uri = urlparse(packet.checkpoint_s3_uri)
        if (
            checkpoint_uri.scheme != "s3"
            or not checkpoint_uri.netloc
            or not checkpoint_uri.path.strip("/")
            or not parse_qs(checkpoint_uri.query).get("versionId", [""])[0]
            or packet.checkpoint_s3_uri != self._checkpoint_uri()
        ):
            raise LiveExecutionBlocked("approval packet checkpoint URI is not version pinned")
        now = datetime.now(UTC)
        if packet.issued_at > now + timedelta(seconds=30):
            raise LiveExecutionBlocked("approval packet issuance is in the future")
        if packet.expires_at <= now:
            raise LiveExecutionBlocked("approval packet is expired")
        if packet.expires_at - packet.issued_at > timedelta(
            seconds=self.config.approval_ttl_seconds
        ):
            raise LiveExecutionBlocked("approval packet expiry exceeds configured approval window")
        if preflight.gpu_capacity_status is not GpuCapacityStatus.VERIFIED_BY_QUOTA:
            raise LiveExecutionBlocked("GPU quota/capacity is not verified for this run")

    def _checkpoint_uri(self, champion_record: RunHistoryRecord | None = None) -> str:
        if champion_record is not None:
            return self._champion_artifact(champion_record).uri
        if not self.config.checkpoint_s3_uri:
            raise LiveExecutionBlocked(
                "a versioned checkpoint_s3_uri is required before starting a run"
            )
        return self.config.checkpoint_s3_uri

    def _benchmark(
        self,
        *,
        run_id: str,
        model_uri: str,
        model_sha256: str,
        split: str,
        episodes: int,
        output_s3_uri: str,
        manifest_sha256: str,
    ) -> ObjectiveBenchmarkResult:
        if split not in {"train", "replay"}:
            raise LiveExecutionFailed(
                "sealed baseline/candidate comparisons must use the paired evaluator"
            )
        request = ObjectiveBenchmarkRequest(
            run_id=run_id,
            model_uri=model_uri,
            model_sha256=model_sha256,
            suite=self.config.objective_suite,
            suite_version=self.config.objective_suite_version,
            seed=self.config.seed,
            num_episodes=episodes,
            split=cast(Literal["train", "replay"], split),
            output_s3_uri=output_s3_uri,
        )
        result = execute_objective_benchmark(self.objective_worker, request)
        if result.manifest_sha256 != manifest_sha256:
            raise LiveExecutionFailed(
                f"objective {split} result manifest does not match the run manifest"
            )
        if (
            result.run_id != run_id
            or result.suite != self.config.objective_suite
            or result.suite_version != self.config.objective_suite_version
            or result.seed != self.config.seed
            or result.split != split
            or result.benchmark_id != self.config.benchmark_id
        ):
            raise LiveExecutionFailed("objective result provenance does not match the run scope")
        if (
            result.evidence_label not in {EvidenceLabel.LIVE, EvidenceLabel.PRIOR_VERIFIED_RUN}
            or not result.verified
        ):
            raise LiveExecutionFailed(f"objective {split} result is not verified live evidence")
        return result

    def _gate(
        self,
        baseline: ObjectiveBenchmarkResult,
        candidate: ObjectiveBenchmarkResult,
        run_id: str,
        run_number: int,
        *,
        champion_run_id: str | None,
        champion_run_number: int | None = None,
    ) -> MultiRunPromotionResult:
        champion = MultiRunEvaluation(
            run_id=champion_run_id or f"baseline-{run_id}",
            run_number=(
                champion_run_number or run_number - 1 if champion_run_id else run_number - 1
            ),
            aggregate_score=baseline.metrics.aggregate,
            environment_scores=dict(baseline.metrics.per_environment),
            evidence=baseline.evidence(kind=EvidenceKind.EVALUATION),
        )
        candidate_eval = MultiRunEvaluation(
            run_id=run_id,
            run_number=run_number,
            champion_run_id=champion.run_id,
            aggregate_score=candidate.metrics.aggregate,
            environment_scores=dict(candidate.metrics.per_environment),
            evidence=candidate.evidence(kind=EvidenceKind.EVALUATION),
        )
        return MultiRunPromotionGate()(champion, candidate_eval)

    @staticmethod
    def _champion_artifact(record: RunHistoryRecord) -> ArtifactReference:
        for artifact in record.artifact_refs:
            if artifact.artifact_id == record.candidate_artifact_id:
                if artifact.kind is not ArtifactKind.CHECKPOINT:
                    raise LiveExecutionBlocked("promoted champion artifact is not a checkpoint")
                if urlparse(artifact.uri).scheme != "s3":
                    raise LiveExecutionBlocked("promoted champion checkpoint is not an S3 URI")
                return artifact
        raise LiveExecutionBlocked("promoted champion artifact reference is missing")

    @staticmethod
    def _champion_uri(record: RunHistoryRecord) -> str:
        """Return the immutable, versioned URI of the promoted checkpoint."""

        return AutonomousRunController._champion_artifact(record).uri

    def _record(
        self,
        *,
        run_id: str,
        run_number: int,
        status: RunStatus,
        decision: RunDecision,
        parent_run_id: str | None,
        champion_run_id: str | None,
        baseline: ObjectiveBenchmarkResult,
        candidate: ObjectiveBenchmarkResult,
        candidate_artifact: ArtifactReference,
        manifest_artifact: ArtifactReference | None,
        manifest_sha256: str,
        gate: MultiRunPromotionResult,
        job_metadata_artifacts: tuple[ArtifactReference, ...] = (),
    ) -> RunHistoryRecord:
        evidence = candidate.evidence(kind=EvidenceKind.EVALUATION)
        return RunHistoryRecord(
            run_id=run_id,
            run_number=run_number,
            parent_run_id=parent_run_id,
            champion_run_id=champion_run_id,
            candidate_artifact_id=candidate_artifact.artifact_id,
            benchmark_id=candidate.benchmark_id,
            suite=candidate.suite,
            suite_version=candidate.suite_version,
            seed=candidate.seed,
            model_id=candidate.model_id,
            status=status,
            decision=decision,
            manifest_sha256=manifest_sha256,
            baseline_metrics=baseline.metrics,
            candidate_metrics=candidate.metrics,
            artifact_refs=tuple(
                item
                for item in (
                    manifest_artifact,
                    candidate_artifact,
                    baseline.trajectory_artifact,
                    baseline.report_artifact,
                    candidate.trajectory_artifact,
                    candidate.report_artifact,
                    *job_metadata_artifacts,
                )
                if item is not None
            ),
            evidence=(evidence,),
            decision_reasons=gate.reasons,
            completed_at=datetime.now(UTC),
        )

    def _artifact_from_job(self, job: JobResult, kind: ArtifactKind) -> ArtifactReference:
        if not job.artifact_uri or not job.provider_job_id:
            raise LiveExecutionFailed("provider returned no artifact URI or job ID")
        parsed = urlparse(job.artifact_uri)
        version_values = parse_qs(parsed.query, keep_blank_values=True).get("versionId", [])
        if len(version_values) > 1 or (
            version_values and (not version_values[0] or version_values[0].lower() == "null")
        ):
            raise LiveExecutionFailed("provider artifact must be an immutable versioned S3 URI")
        try:
            # Versioned output is parsed strictly.  A normal SageMaker output
            # may omit VersionId, but only the scoped canonicalizer below may
            # resolve that mutable path to one exact source version.
            if version_values:
                ArtifactRef.from_live_uri(job.artifact_uri, sha256="0" * 64, size_bytes=0)
        except (ArtifactIntegrityError, ValueError) as exc:
            raise LiveExecutionFailed("provider artifact must be a valid S3 output URI") from exc

        canonicalize = getattr(self.artifact_store, "canonicalize_sagemaker_output", None)
        verify = getattr(self.artifact_store, "verify_immutable", None)
        if not callable(canonicalize) or not callable(verify):
            raise LiveExecutionFailed("artifact store cannot verify and retain provider artifacts")
        try:
            retained = canonicalize(
                job.artifact_uri,
                retained_prefix="checkpoints",
                allowed_source_bucket=self.config.artifact_bucket,
                allowed_source_prefix=self.config.artifact_prefix,
            )
            verified = verify(
                retained,
                expected_sha256=retained.sha256,
                expected_size_bytes=retained.size_bytes,
                allowed_bucket=self.config.artifact_bucket,
                allowed_prefix=self.config.artifact_prefix,
            )
            ArtifactRef.from_live_uri(
                verified.version_ref,
                sha256=verified.sha256,
                size_bytes=verified.size_bytes,
            )
        except (ArtifactIntegrityError, ValueError, TypeError, AttributeError) as exc:
            raise LiveExecutionFailed(
                "provider artifact failed immutable integrity verification"
            ) from exc
        return ArtifactReference(
            artifact_id=hashlib.sha256(verified.version_ref.encode()).hexdigest()[:24],
            kind=kind,
            uri=verified.version_ref,
            sha256=verified.sha256,
            size_bytes=verified.size_bytes,
            metadata={"provider_job_id": str(job.provider_job_id)},
        )

    def _wait_policy(self) -> Any:
        from app.posttraining.objective_workflow import JobWaitPolicy

        return JobWaitPolicy(
            max_attempts=self.config.provider_max_polls,
            poll_interval_seconds=self.config.provider_poll_interval_seconds,
        )

    def _cleanup(
        self,
        training: JobResult | None,
        evaluation: JobResult | None,
        *,
        run_id: str,
        run_number: int,
    ) -> None:
        cleanup_ok = True
        jobs = (
            ("training", training, self.provider.stop_training),
            ("evaluation", evaluation, self.provider.stop_evaluation),
        )
        for phase, job, stop in jobs:
            if job is not None and job.status in {JobStatus.SUBMITTED, JobStatus.IN_PROGRESS}:
                try:
                    stop(job.job_name)
                except Exception:
                    cleanup_ok = False
            if job is not None:
                self._event(
                    EventType.CLEANUP_COMPLETED if cleanup_ok else EventType.CLEANUP_FAILED,
                    run_id,
                    run_number,
                    phase=phase,
                    job_id=str(job.provider_job_id or job.job_name),
                    status="completed" if cleanup_ok else "failed",
                )
        if not any(job is not None for _, job, _ in jobs):
            self._event(
                EventType.CLEANUP_COMPLETED if cleanup_ok else EventType.CLEANUP_FAILED,
                run_id,
                run_number,
                phase="cleanup",
                status="completed" if cleanup_ok else "failed",
            )

    def _event(
        self,
        event_type: EventType,
        run_id: str,
        run_number: int,
        *,
        status: str,
        phase: str | None = None,
        job_id: str | None = None,
        evidence_label: EvidenceLabel | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            self.telemetry.record(
                event_type,
                run_id=run_id,
                run_number=run_number,
                experiment_id=f"{run_id}-experiment",
                phase=phase,
                job_id=job_id,
                evidence_label=evidence_label,
                status=status,
                attributes=attributes,
            )
        except Exception:
            pass

    def _validate_run_number(self, run_number: int) -> None:
        if (
            not isinstance(run_number, int)
            or isinstance(run_number, bool)
            or not 1 <= run_number <= self.config.max_runs
        ):
            raise LiveExecutionBlocked(f"run_number must be between 1 and {self.config.max_runs}")

    def _require_approval(self, token: str, *, run_number: int) -> ApprovalPacket:
        secret = os.getenv(self.config.approval_secret_env, "")
        if not secret:
            raise LiveExecutionBlocked(
                f"{self.config.approval_secret_env} is required for signed approval"
            )
        packet = _decode_approval_token(token, secret)
        if packet.run_number != run_number:
            raise LiveExecutionBlocked("approval token run number does not match request")
        return packet


class LiveObjectiveAdapter:
    """Translate authenticated objective-worker results to supervisor evidence."""

    def __init__(self, client: ObjectiveWorkerClient, config: LiveExecutionConfig) -> None:
        self.client = client
        self.config = config

    def _model_uri(self, state: Any) -> str:
        uri = getattr(state, "champion_checkpoint_uri", None) or getattr(
            state, "base_checkpoint_uri", None
        )
        if not uri:
            uri = getattr(state, "metadata", {}).get("checkpoint_uri")
        if not isinstance(uri, str) or not uri:
            raise LiveExecutionBlocked("a verified checkpoint URI is required")
        return uri

    def _model_sha256(self, state: Any) -> str:
        digest = getattr(state, "champion_checkpoint_sha256", None) or getattr(
            state, "base_checkpoint_sha256", None
        )
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise LiveExecutionBlocked("a verified champion model SHA-256 is required")
        return digest

    @staticmethod
    def _evidence(result: ObjectiveBenchmarkResult, *, run_number: int) -> Any:
        from app.autonomous.supervisor import BenchmarkEvidence
        from app.objective.models import ObjectiveSplit, encode_trajectory_reference

        artifact_ids = tuple(
            artifact.artifact_id
            for artifact in (result.trajectory_artifact, result.report_artifact)
            if artifact is not None
        )
        if not artifact_ids or not result.verified:
            raise LiveExecutionFailed("objective result lacks verified artifacts")
        if not result.trajectory_references:
            raise LiveExecutionFailed("objective result lacks verified trajectory references")
        if result.split not in {ObjectiveSplit.TRAIN.value, ObjectiveSplit.REPLAY.value}:
            raise LiveExecutionFailed(
                "objective trajectory references are outside train/replay scope"
            )
        if any(
            not reference.verified or reference.split.value != result.split
            for reference in result.trajectory_references
        ):
            raise LiveExecutionFailed("objective trajectory reference provenance is inconsistent")
        trajectory_refs = tuple(
            encode_trajectory_reference(reference) for reference in result.trajectory_references
        )
        from app.posttraining.models import Evidence, EvidenceKind
        from app.posttraining.multi_run_gate import MultiRunEvaluation

        evidence = Evidence(
            evidence_id=result.benchmark_id,
            kind=EvidenceKind.BENCHMARK,
            label=result.evidence_label,
            artifact_ids=artifact_ids,
            metrics={"aggregate": result.metrics.aggregate, **result.metrics.per_environment},
            verified=result.verified,
            benchmark_id=result.benchmark_id,
            suite=result.suite,
            suite_version=result.suite_version,
            manifest_sha256=result.manifest_sha256,
            seed=result.seed,
            model_id=result.model_id,
        )
        evaluation = MultiRunEvaluation(
            run_id=result.run_id,
            run_number=run_number,
            aggregate_score=result.metrics.aggregate,
            environment_scores=result.metrics.per_environment,
            evidence=evidence,
        )
        return BenchmarkEvidence(
            evaluation=evaluation,
            trajectory_refs=trajectory_refs,
            artifact_ids=artifact_ids,
        )

    def benchmark(self, state: Any, *, split: str, experiment_number: int) -> Any:
        if split not in {"train", "replay"}:
            raise LiveExecutionFailed(
                "sealed baseline/candidate comparisons must use the paired evaluator"
            )
        request = ObjectiveBenchmarkRequest(
            run_id=state.run_id,
            model_uri=self._model_uri(state),
            model_sha256=self._model_sha256(state),
            suite=self.config.objective_suite,
            suite_version=self.config.objective_suite_version,
            seed=self.config.seed,
            num_episodes=(
                self.config.baseline_episodes
                if split == "train"
                else self.config.held_out_episodes
            ),
            split=cast(Literal["train", "replay"], split),
            output_s3_uri=(
                f"s3://{self.config.artifact_bucket}/{self.config.artifact_prefix}/"
                f"{state.run_id}/{split}/{experiment_number}"
            ),
        )
        return self._evidence(
            self._checked_result(self.client.execute_benchmark(request), request),
            run_number=0 if split == "baseline" else experiment_number,
        )

    def _checked_result(
        self, result: ObjectiveBenchmarkResult, request: ObjectiveBenchmarkRequest
    ) -> ObjectiveBenchmarkResult:
        if (
            result.run_id != request.run_id
            or result.suite != request.suite
            or result.suite_version != request.suite_version
            or result.model_id != request.model_uri
            or result.model_sha256 != request.model_sha256
            or result.seed != request.seed
            or result.split != request.split
        ):
            raise LiveExecutionFailed("objective result provenance does not match the run scope")
        return result

    def build_dataset(self, state: Any, plan: Any, *, experiment_number: int) -> Any:
        from app.autonomous.supervisor import DatasetArtifact
        from app.objective.models import (
            CorrectionReplayResponse,
            Dataset,
            ObjectiveSplit,
            decode_trajectory_reference,
        )

        try:
            references = tuple(
                decode_trajectory_reference(value)
                for value in plan.selected_trajectory_refs
            )
        except Exception as exc:
            raise LiveExecutionFailed(
                "curation plan contains malformed trajectory provenance"
            ) from exc
        if len({reference.trajectory_id for reference in references}) != len(references):
            raise LiveExecutionFailed("curation plan contains duplicate trajectory references")
        proposals = tuple(getattr(plan, "correction_proposals", ()))
        if not references and not proposals:
            raise LiveExecutionFailed("curation plan contains neither selected records nor repairs")
        splits = {reference.split for reference in references} | {
            proposal.split for proposal in proposals
        }
        if len(splits) != 1 or not splits.issubset(
            {ObjectiveSplit.TRAIN, ObjectiveSplit.REPLAY}
        ):
            raise LiveExecutionFailed("curation plan must use one actual train/replay split")

        accepted_references: tuple[Any, ...] = ()
        if proposals:
            replay = self.client.replay_corrections(
                run_id=state.run_id,
                experiment_id=f"{state.run_id}-{experiment_number}",
                split=next(iter(splits)),
                proposals=proposals,
            )
            if not isinstance(replay, CorrectionReplayResponse):
                try:
                    replay = CorrectionReplayResponse.model_validate(replay)
                except Exception as exc:
                    raise LiveExecutionFailed(
                        "objective correction response is not typed replay evidence"
                    ) from exc
            expected_proposal_ids = tuple(proposal.proposal_id for proposal in proposals)
            if (
                replay.run_id != state.run_id
                or replay.experiment_id != f"{state.run_id}-{experiment_number}"
                or replay.split is not next(iter(splits))
                or tuple(outcome.proposal_id for outcome in replay.outcomes)
                != expected_proposal_ids
            ):
                raise LiveExecutionFailed("objective correction replay provenance does not match")
            expected_proposals = {proposal.proposal_id: proposal for proposal in proposals}
            for outcome in replay.outcomes:
                proposal = expected_proposals[outcome.proposal_id]
                if (
                    outcome.source_trajectory_id != proposal.source_trajectory_id
                    or outcome.task_id != proposal.task_id
                    or outcome.split is not proposal.split
                ):
                    raise LiveExecutionFailed("objective correction lineage differs from proposal")
            accepted_references = tuple(
                outcome.trajectory_reference
                for outcome in replay.outcomes
                if outcome.status == "PASS" and outcome.trajectory_reference is not None
            )
        all_references = references + accepted_references
        if not all_references:
            raise LiveExecutionFailed("no selected or verifier-passing correction is available")
        if len({reference.trajectory_id for reference in all_references}) != len(all_references):
            raise LiveExecutionFailed(
                "objective curation references contain duplicate trajectories"
            )

        response = self.client.verify_curation(
            run_id=state.run_id,
            experiment_id=f"{state.run_id}-{experiment_number}",
            trajectory_references=all_references,
        )
        try:
            dataset = Dataset.model_validate(response)
        except Exception as exc:
            raise LiveExecutionFailed(
                "objective curation response is not a verified dataset"
            ) from exc
        expected_provenance = {
            reference.trajectory_id: (reference.task_id, reference.split)
            for reference in all_references
        }
        accepted_lineage = {
            outcome.trajectory_reference.trajectory_id: outcome.source_trajectory_id
            for outcome in replay.outcomes
            if proposals
            and outcome.status == "PASS"
            and outcome.trajectory_reference is not None
        } if proposals else {}
        actual_ids = tuple(row.source_trajectory_id for row in dataset.rows)
        if (
            dataset.manifest.run_id != state.run_id
            or dataset.manifest.experiment_id != f"{state.run_id}-{experiment_number}"
            or dataset.manifest.source_trajectory_ids != actual_ids
            or len(set(actual_ids)) != len(actual_ids)
            or any(
                row.source_trajectory_id not in expected_provenance
                or (row.task_id, row.split)
                != expected_provenance[row.source_trajectory_id]
                or row.repaired_from_trajectory_id
                != accepted_lineage.get(row.source_trajectory_id)
                or row.source_type
                != (
                    "repaired_replay"
                    if row.source_trajectory_id in accepted_lineage
                    else "successful_replay"
                )
                for row in dataset.rows
            )
            or any(
                not row.verifier_confirmed
                or not row.verifier_success
                or row.source_type not in {"successful_replay", "repaired_replay"}
                for row in dataset.rows
            )
        ):
            raise LiveExecutionFailed("objective dataset provenance does not match curation input")
        return DatasetArtifact(
            dataset_id=dataset.manifest.dataset_id,
            uri=dataset.manifest.s3_uri,
            sha256=dataset.manifest.sha256,
            artifact_id=f"dataset://{dataset.manifest.dataset_id}",
        )

    def verify_dataset(self, dataset: Any, *, run_id: str, experiment_number: int) -> Any:
        parsed = urlparse(dataset.uri)
        version_ids = parse_qs(parsed.query, keep_blank_values=True).get("versionId", [])
        key_parts = parsed.path.strip("/").split("/")
        if (
            parsed.scheme != "s3"
            or not parsed.netloc
            or len(version_ids) != 1
            or not version_ids[0]
            or version_ids[0].lower() == "null"
            or set(parse_qs(parsed.query, keep_blank_values=True)) != {"versionId"}
            or len(key_parts) < 3
            or key_parts[-1] != "dataset.jsonl"
            or key_parts[-2] != dataset.sha256
            or dataset.artifact_id != f"dataset://{dataset.dataset_id}"
            or not re.fullmatch(r"[0-9a-f]{64}", dataset.sha256)
        ):
            raise LiveExecutionFailed("objective dataset artifact is not immutable")
        if run_id not in key_parts or f"{run_id}-{experiment_number}" not in key_parts:
            raise LiveExecutionFailed("objective dataset artifact scope does not match run")
        return dataset


class LiveRequestFactory:
    """Build pinned SageMaker requests from durable state and verified handoffs."""

    def __init__(self, config: LiveExecutionConfig) -> None:
        self.config = config

    def _approved(self, state: Any, name: str, configured: Any) -> Any:
        scope = getattr(state, "approval_scope", None)
        if not isinstance(scope, Mapping) or name not in scope:
            raise LiveExecutionBlocked("durable approved execution scope is incomplete")
        value = scope[name]
        if value != configured:
            raise LiveExecutionBlocked(f"approved {name} does not match configuration")
        return value

    @staticmethod
    def _versioned_content_uri(uri: object, digest: object, *, suffix: str) -> str:
        if not isinstance(uri, str) or not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise LiveExecutionBlocked("approved content-addressed artifact is incomplete")
        parsed = urlparse(uri)
        versions = parse_qs(parsed.query, keep_blank_values=True).get("versionId", [])
        if (
            parsed.scheme != "s3"
            or not parsed.netloc
            or parsed.fragment
            or len(versions) != 1
            or not versions[0]
            or versions[0].lower() == "null"
            or set(parse_qs(parsed.query, keep_blank_values=True)) != {"versionId"}
            or parsed.path.rstrip("/").rsplit("/", 1)[-1] != f"{digest}{suffix}"
            or any(char.isspace() for char in uri)
        ):
            raise LiveExecutionBlocked(
                "artifact URI is not an exact versioned content-addressed reference"
            )
        return f"s3://{parsed.netloc}{parsed.path}"

    @classmethod
    def _dataset_prefix(cls, dataset: Any) -> str:
        uri = getattr(dataset, "uri", None)
        digest = getattr(dataset, "sha256", None)
        if not isinstance(uri, str) or not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise LiveExecutionBlocked("approved dataset identity is incomplete")
        parsed = urlparse(uri)
        versions = parse_qs(parsed.query, keep_blank_values=True).get("versionId", [])
        parts = parsed.path.strip("/").split("/")
        if (
            parsed.scheme != "s3"
            or not parsed.netloc
            or parsed.fragment
            or len(versions) != 1
            or not versions[0]
            or versions[0].lower() == "null"
            or set(parse_qs(parsed.query, keep_blank_values=True)) != {"versionId"}
            or len(parts) < 4
            or parts[-1] != "dataset.jsonl"
            or parts[-2] != digest
            or any(char.isspace() for char in uri)
        ):
            raise LiveExecutionBlocked(
                "dataset URI is not an exact versioned content-addressed reference"
            )
        return f"s3://{parsed.netloc}/" + "/".join(parts[:-1])

    @staticmethod
    def _sha(value: object, name: str) -> str:
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise LiveExecutionBlocked(f"approved {name} is missing or invalid")
        return value

    def training(
        self, state: Any, *, experiment_number: int, dataset: Any, config: Any
    ) -> TrainingJobRequest:
        instance_type = self._approved(state, "instance_type", self.config.instance_type)
        instance_count = self._approved(state, "instance_count", self.config.instance_count)
        volume_size_gb = self._approved(state, "volume_size_gb", self.config.volume_size_gb)
        max_runtime_seconds = self._approved(
            state, "max_runtime_seconds", self.config.max_runtime_seconds
        )
        dataset_id = getattr(dataset, "dataset_id", None)
        artifact_id = getattr(dataset, "artifact_id", None)
        dataset_sha = self._sha(getattr(dataset, "sha256", None), "dataset SHA-256")
        if (
            not isinstance(dataset_id, str)
            or not dataset_id
            or artifact_id != f"dataset://{dataset_id}"
        ):
            raise LiveExecutionBlocked(
                "approved dataset artifact identity is missing or mismatched"
            )
        model_id = getattr(state, "model_id", self.config.target_model)
        model_revision = getattr(state, "checkpoint_revision", self.config.hf_revision)
        if model_id != self.config.target_model or model_revision != self.config.hf_revision:
            raise LiveExecutionBlocked("approved base model identity does not match configuration")
        qlora_config = config.model_dump(mode="json")
        if not isinstance(qlora_config, Mapping):
            raise LiveExecutionBlocked("approved QLoRA configuration is invalid")
        experiment_id = f"{state.run_id}-{experiment_number}"
        environment = {
            "RUN_ID": state.run_id,
            "EXPERIMENT_ID": experiment_id,
            "DATASET_ID": dataset_id,
            "DATASET_SHA256": dataset_sha,
            "APPROVED_DATASET_ARTIFACT_ID": artifact_id,
            "BASE_MODEL_ID": model_id,
            "BASE_MODEL_REVISION": model_revision,
            "BASE_MODEL_BUNDLE_SHA256": self._sha(
                getattr(state, "base_checkpoint_sha256", None), "base model bundle SHA-256"
            ),
            "QLORA_CONFIG": json.dumps(dict(qlora_config), sort_keys=True, separators=(",", ":")),
        }
        base_model_uri = self._versioned_content_uri(
            getattr(state, "base_checkpoint_uri", None),
            environment["BASE_MODEL_BUNDLE_SHA256"],
            suffix=".tar.gz",
        )
        parent_uri = None
        if experiment_number > 1:
            metadata = getattr(state, "metadata", {})
            if not isinstance(metadata, Mapping):
                raise LiveExecutionBlocked("approved parent adapter metadata is unavailable")
            parent_uri = self._versioned_content_uri(
                getattr(state, "champion_checkpoint_uri", None),
                getattr(state, "champion_checkpoint_sha256", None),
                suffix=".tar.gz",
            )
            parent_artifact_id = metadata.get("champion_checkpoint_artifact_id")
            if not isinstance(parent_artifact_id, str) or not parent_artifact_id.startswith("checkpoint://"):
                raise LiveExecutionBlocked("approved parent adapter artifact ID is unavailable")
            parent_artifact_sha = self._sha(
                metadata.get("champion_checkpoint_artifact_sha256"),
                "parent artifact SHA-256",
            )
            environment.update(
                {
                    "APPROVED_PARENT_ARTIFACT_ID": parent_artifact_id,
                    "APPROVED_PARENT_MANIFEST_SHA256": self._sha(
                        metadata.get("champion_checkpoint_manifest_sha256"),
                        "parent manifest SHA-256",
                    ),
                    "APPROVED_PARENT_ARTIFACT_SHA256": parent_artifact_sha,
                    "APPROVED_PARENT_ARCHIVE_SHA256": self._sha(
                        getattr(state, "champion_checkpoint_sha256", None), "parent archive SHA-256"
                    ),
                }
            )
        return TrainingJobRequest(
            job_name=f"pending-{state.run_id}-{experiment_number}",
            role_arn=self.config.training_role_arn,
            image_uri=self.config.training_image,
            input_s3_uri=self._dataset_prefix(dataset),
            output_s3_uri=f"s3://{self.config.artifact_bucket}/{self.config.artifact_prefix}/{state.run_id}/train/{experiment_number}",
            instance_type=cast(str, instance_type),
            base_model_s3_uri=base_model_uri,
            parent_adapter_s3_uri=parent_uri,
            instance_count=cast(int, instance_count),
            volume_size_gb=cast(int, volume_size_gb),
            max_runtime_seconds=cast(int, max_runtime_seconds),
            hyperparameters=dict(qlora_config),
            environment=environment,
        )

    def evaluation(
        self, state: Any, *, experiment_number: int, candidate: Any
    ) -> EvaluationJobRequest:
        instance_type = self._approved(state, "instance_type", self.config.instance_type)
        instance_count = self._approved(state, "instance_count", self.config.instance_count)
        volume_size_gb = self._approved(state, "volume_size_gb", self.config.volume_size_gb)
        max_runtime_seconds = self._approved(
            state, "max_runtime_seconds", self.config.max_runtime_seconds
        )
        candidate_uri = self._versioned_content_uri(
            getattr(candidate, "uri", None), getattr(candidate, "sha256", None), suffix=".tar.gz"
        )
        base_model_id = getattr(state, "model_id", self.config.target_model)
        base_model_revision = getattr(state, "checkpoint_revision", self.config.hf_revision)
        if (
            base_model_id != self.config.target_model
            or base_model_revision != self.config.hf_revision
        ):
            raise LiveExecutionBlocked("approved base model identity does not match configuration")
        base_model_sha = self._sha(
            getattr(state, "base_checkpoint_sha256", None), "base model bundle SHA-256"
        )
        base_model_uri = self._versioned_content_uri(
            getattr(state, "base_checkpoint_uri", None), base_model_sha, suffix=".tar.gz"
        )
        champion_kind = "base-model"
        champion_uri = base_model_uri
        champion_sha = base_model_sha
        if experiment_number > 1:
            promoted_uri = getattr(state, "champion_checkpoint_uri", None)
            promoted_sha = getattr(state, "champion_checkpoint_sha256", None)
            if promoted_uri is not None or promoted_sha is not None:
                champion_uri = self._versioned_content_uri(
                    promoted_uri, promoted_sha, suffix=".tar.gz"
                )
                champion_sha = self._sha(promoted_sha, "promoted champion archive SHA-256")
                champion_kind = "qlora-adapter"
        sealed_uri = self.config.evaluation_input_s3_uri
        sealed = urlparse(sealed_uri)
        if (
            sealed.scheme != "s3"
            or not sealed.netloc
            or not sealed.path.strip("/")
            or sealed.query
            or sealed.fragment
        ):
            raise LiveExecutionBlocked("sealed evaluation input must be a query-free S3 URI")
        manifest_sha = self._sha(
            getattr(state, "benchmark_manifest_sha256", None), "evaluation manifest SHA-256"
        )
        suite_version = getattr(state, "benchmark_version", self.config.objective_suite_version)
        seed = getattr(state, "benchmark_seed", self.config.seed)
        return EvaluationJobRequest(
            job_name=f"pending-{state.run_id}-{experiment_number}-eval",
            role_arn=self.config.training_role_arn,
            image_uri=self.config.evaluation_image,
            input_s3_uri=sealed_uri,
            output_s3_uri=f"s3://{self.config.artifact_bucket}/{self.config.artifact_prefix}/{state.run_id}/eval/{experiment_number}",
            model_s3_uri=candidate_uri,
            candidate_s3_uri=candidate_uri,
            champion_s3_uri=champion_uri,
            sealed_s3_uri=sealed_uri,
            base_model_s3_uri=base_model_uri,
            instance_type=cast(str, instance_type),
            instance_count=cast(int, instance_count),
            volume_size_gb=cast(int, volume_size_gb),
            max_runtime_seconds=cast(int, max_runtime_seconds),
            environment={
                "RUN_ID": state.run_id,
                "EXPERIMENT_ID": f"{state.run_id}-{experiment_number}",
                "EVALUATION_MANIFEST_SHA256": manifest_sha,
                "EVALUATION_SUITE_VERSION": suite_version,
                "OBJECTIVE_SEED": str(seed),
                "CANDIDATE_ARCHIVE_SHA256": self._sha(
                    getattr(candidate, "sha256", None), "candidate archive SHA-256"
                ),
                "CHAMPION_ARCHIVE_SHA256": self._sha(champion_sha, "champion archive SHA-256"),
                "CHAMPION_KIND": champion_kind,
                "BASE_MODEL_ID": base_model_id,
                "BASE_MODEL_REVISION": base_model_revision,
                "BASE_MODEL_BUNDLE_SHA256": base_model_sha,
            },
        )


class S3LiveArtifactVerifier:
    """Retain and verify a completed SageMaker output before evidence admission."""

    def __init__(self, store: S3ArtifactStore, config: LiveExecutionConfig) -> None:
        self.store = store
        self.config = config

    def verify_checkpoint(self, job: JobResult, *, run_id: str, experiment_number: int) -> Any:
        import tempfile
        from pathlib import Path

        from app.autonomous.supervisor import CheckpointArtifact
        from workers.trainer.train import _extract_checkpoint_archive, verify_parent_adapter

        if not job.artifact_uri:
            raise LiveExecutionFailed("SageMaker training output URI is missing")
        retained = self.store.canonicalize_sagemaker_output(
            job.artifact_uri,
            retained_prefix=f"{self.config.artifact_prefix}/{run_id}/checkpoints",
            allowed_source_bucket=self.config.artifact_bucket,
            allowed_source_prefix=f"{self.config.artifact_prefix}/{run_id}",
        )
        with tempfile.TemporaryDirectory(prefix="checkpoint-verify-") as scratch:
            scratch_path = Path(scratch)
            archive_path = scratch_path / f"{retained.sha256}.tar.gz"
            archive_path.write_bytes(self.store.get_bytes(retained))
            checkpoint_dir = _extract_checkpoint_archive(
                archive_path, scratch_path / "checkpoint"
            )
            manifest = verify_parent_adapter(checkpoint_dir)
        if (
            manifest.get("run_id") != run_id
            or manifest.get("experiment_id") != f"{run_id}-{experiment_number}"
            or manifest.get("artifact_id", "")
            != f"checkpoint://{manifest.get('artifact_sha256', '')}"
        ):
            raise LiveExecutionFailed(
                "checkpoint manifest provenance does not match the training request"
            )
        return CheckpointArtifact(
            artifact_id=cast(str, manifest["artifact_id"]),
            uri=retained.version_ref,
            sha256=retained.sha256,
            manifest_sha256=cast(str, manifest["manifest_sha256"]),
            artifact_sha256=cast(str, manifest["artifact_sha256"]),
        )


class LiveEvaluationReader:
    """Verify and read the immutable report emitted by the SageMaker evaluator."""

    def __init__(self, artifact_store: S3ArtifactStore, config: LiveExecutionConfig) -> None:
        self.artifact_store = artifact_store
        self.config = config

    def read_evaluation(self, job: JobResult, *, state: Any, experiment_number: int) -> Any:
        import io
        import tarfile

        from app.autonomous.supervisor import EvaluationEvidence
        from app.posttraining.models import ArtifactKind, ArtifactReference, Evidence
        from app.posttraining.run_history import BenchmarkMetrics

        if job.status is not JobStatus.COMPLETED or not job.provider_job_id:
            raise LiveExecutionFailed("SageMaker evaluator job did not complete with a provider ID")
        if not job.artifact_uri:
            raise LiveExecutionFailed("SageMaker evaluator report artifact URI is missing")
        if not state.current_candidate_uri or not state.current_candidate_sha256:
            raise LiveExecutionFailed("verified candidate checkpoint provenance is missing")
        source_prefix = (
            f"{self.config.artifact_prefix}/{state.run_id}/eval/{experiment_number}"
        )
        try:
            artifact = self.artifact_store.canonicalize_sagemaker_processing_output(
                job.artifact_uri,
                retained_prefix=f"{self.config.artifact_prefix}/{state.run_id}/evaluations",
                allowed_source_bucket=self.config.artifact_bucket,
                allowed_source_prefix=source_prefix,
            )
            if artifact.size_bytes > 16 * 1024 * 1024:
                raise LiveExecutionFailed(
                    "SageMaker evaluator artifact exceeds the report size limit"
                )
            artifact_bytes = self.artifact_store.get_bytes(artifact)
        except LiveExecutionFailed:
            raise
        except Exception as exc:
            raise LiveExecutionFailed(
                "SageMaker evaluator artifact could not be retained and verified"
            ) from exc
        try:
            retained_filename = artifact.key.rsplit("/", 1)[-1]
            if retained_filename.endswith(".json"):
                if not artifact_bytes or len(artifact_bytes) > 4 * 1024 * 1024:
                    raise ValueError("evaluation.json must be non-empty and bounded")
                report_bytes = artifact_bytes
            elif retained_filename.endswith(".tar.gz"):
                with tarfile.open(fileobj=io.BytesIO(artifact_bytes), mode="r:gz") as archive:
                    candidates = []
                    for member in archive.getmembers():
                        components = member.name.split("/")
                        if (
                            member.isfile()
                            and components[-1] == "evaluation.json"
                            and not member.name.startswith("/")
                            and all(part not in {"", ".", ".."} for part in components)
                        ):
                            candidates.append(member)
                    if (
                        len(candidates) != 1
                        or candidates[0].size < 1
                        or candidates[0].size > 4 * 1024 * 1024
                    ):
                        raise ValueError("archive must contain one bounded evaluation.json")
                    report_stream = archive.extractfile(candidates[0])
                    if report_stream is None:
                        raise ValueError("evaluation report is unreadable")
                    report_bytes = report_stream.read(4 * 1024 * 1024 + 1)
                    if (
                        len(report_bytes) != candidates[0].size
                        or len(report_bytes) > 4 * 1024 * 1024
                    ):
                        raise ValueError("evaluation report size does not match archive metadata")
            else:
                raise ValueError(
                    "retained Processing object is not a supported evaluation artifact"
                )
            report = json.loads(report_bytes.decode("utf-8"))
            if not isinstance(report, Mapping):
                raise ValueError("evaluation report must be an object")
        except LiveExecutionFailed:
            raise
        except Exception as exc:
            raise LiveExecutionFailed(
                "SageMaker evaluator artifact is not a valid report archive"
            ) from exc

        report_sha256 = report.get("report_sha256")
        unsigned = dict(report)
        unsigned.pop("report_sha256", None)
        expected_report_sha256 = hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        candidate_manifest_sha256 = report.get("candidate_manifest_sha256")
        candidate_artifact_sha256 = report.get("candidate_artifact_sha256")
        champion_manifest_sha256 = report.get("champion_manifest_sha256")
        champion_artifact_sha256 = report.get("champion_artifact_sha256")
        champion_checkpoint_sha256 = getattr(state, "champion_checkpoint_sha256", None) or getattr(
            state, "base_checkpoint_sha256", None
        )
        if (
            report_sha256 != expected_report_sha256
            or report.get("schema_version") != "evaluation-report-v1"
            or report.get("suite") != self.config.objective_suite
            or report.get("suite_version") != self.config.objective_suite_version
            or report.get("evaluation_manifest_sha256") != state.benchmark_manifest_sha256
            or report.get("run_id") != state.run_id
            or report.get("experiment_id") != f"{state.run_id}-{experiment_number}"
            or not isinstance(candidate_manifest_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", candidate_manifest_sha256)
            or not isinstance(candidate_artifact_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", candidate_artifact_sha256)
            or candidate_artifact_sha256 != state.current_candidate_sha256
            or not isinstance(champion_manifest_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", champion_manifest_sha256)
            or not isinstance(champion_artifact_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", champion_artifact_sha256)
            or champion_artifact_sha256 != champion_checkpoint_sha256
        ):
            raise LiveExecutionFailed(
                "SageMaker evaluator report provenance or checksum is invalid"
            )

        try:
            def parse_metrics(prefix: str) -> tuple[BenchmarkMetrics, int, Mapping[str, Any]]:
                metrics_value = report.get(f"{prefix}_metrics")
                if not isinstance(metrics_value, Mapping):
                    raise ValueError(f"{prefix} metrics are missing")
                task_count = metrics_value["task_count"]
                successful_tasks = metrics_value["successful_tasks"]
                success_rate = metrics_value["success_rate"]
                invalid_action_tasks = metrics_value["invalid_action_tasks"]
                environments = metrics_value["by_environment"]
                if (
                    type(task_count) is not int
                    or task_count < 1
                    or type(successful_tasks) is not int
                    or not 0 <= successful_tasks <= task_count
                    or type(invalid_action_tasks) is not int
                    or not 0 <= invalid_action_tasks <= task_count
                    or isinstance(success_rate, bool)
                    or not isinstance(success_rate, (int, float))
                    or not isfinite(float(success_rate))
                    or abs(float(success_rate) - successful_tasks / task_count) > 1e-12
                    or report.get(f"{prefix}_task_count") != task_count
                    or report.get(f"{prefix}_successful_tasks") != successful_tasks
                    or report.get(f"{prefix}_success_rate") != success_rate
                    or report.get(f"{prefix}_invalid_action_tasks") != invalid_action_tasks
                    or not isinstance(environments, Mapping)
                    or not environments
                ):
                    raise ValueError(f"{prefix} metrics are inconsistent")
                environment_scores: dict[str, float] = {}
                environment_counts: dict[str, int] = {}
                environment_tasks = 0
                environment_successes = 0
                for environment, item in environments.items():
                    if not isinstance(environment, str) or not isinstance(item, Mapping):
                        raise ValueError("environment aggregate is malformed")
                    count = item.get("task_count")
                    successes = item.get("successful_tasks")
                    rate = item.get("success_rate")
                    if (
                        type(count) is not int
                        or count < 1
                        or type(successes) is not int
                        or not 0 <= successes <= count
                        or isinstance(rate, bool)
                        or not isinstance(rate, (int, float))
                        or not isfinite(float(rate))
                        or abs(float(rate) - successes / count) > 1e-12
                    ):
                        raise ValueError("environment aggregate is inconsistent")
                    environment_scores[environment] = float(rate)
                    environment_counts[environment] = count
                    environment_tasks += count
                    environment_successes += successes
                if environment_tasks != task_count or environment_successes != successful_tasks:
                    raise ValueError("environment aggregates do not match task totals")
                return (
                    BenchmarkMetrics(
                        aggregate=float(success_rate), per_environment=environment_scores
                    ),
                    task_count,
                    environment_counts,
                )

            candidate_metrics, candidate_count, candidate_environment_counts = parse_metrics(
                "candidate"
            )
            champion_metrics, champion_count, champion_environment_counts = parse_metrics(
                "champion"
            )
            candidate_outcomes = report.get("candidate_task_successes")
            champion_outcomes = report.get("champion_task_successes")
            candidate_environments = report.get("candidate_task_environments")
            champion_environments = report.get("champion_task_environments")
            if (
                candidate_count != champion_count
                or candidate_environment_counts != champion_environment_counts
                or not isinstance(candidate_outcomes, list)
                or not isinstance(champion_outcomes, list)
                or not isinstance(candidate_environments, list)
                or not isinstance(champion_environments, list)
                or len(candidate_outcomes) != candidate_count
                or len(champion_outcomes) != champion_count
                or len(candidate_environments) != candidate_count
                or len(champion_environments) != champion_count
                or any(
                    type(value) is not bool
                    for value in (*candidate_outcomes, *champion_outcomes)
                )
                or any(
                    not isinstance(value, str)
                    for value in (*candidate_environments, *champion_environments)
                )
                or candidate_environments != champion_environments
                or sum(candidate_outcomes) != report.get("candidate_successful_tasks")
                or sum(champion_outcomes) != report.get("champion_successful_tasks")
            ):
                raise ValueError("paired candidate/champion task outcomes are inconsistent")
            for prefix, outcomes, environments in (
                ("candidate", candidate_outcomes, candidate_environments),
                ("champion", champion_outcomes, champion_environments),
            ):
                grouped: dict[str, list[bool]] = {}
                for environment, success in zip(environments, outcomes, strict=True):
                    grouped.setdefault(environment, []).append(success)
                summary = report[f"{prefix}_metrics"]["by_environment"]
                if set(grouped) != set(summary):
                    raise ValueError("paired outcomes do not match environment summaries")
                for environment, results in grouped.items():
                    item = summary[environment]
                    if (
                        item["task_count"] != len(results)
                        or item["successful_tasks"] != sum(results)
                        or abs(item["success_rate"] - sum(results) / len(results)) > 1e-12
                    ):
                        raise ValueError("paired outcomes do not match environment metrics")
            paired_outcomes = {
                "candidate": candidate_outcomes,
                "champion": champion_outcomes,
                "environments": candidate_environments,
            }
            paired_digest = hashlib.sha256(
                json.dumps(paired_outcomes, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            regressions = sum(
                champion and not candidate
                for candidate, champion in zip(candidate_outcomes, champion_outcomes, strict=True)
            )
            improvements = sum(
                candidate and not champion
                for candidate, champion in zip(candidate_outcomes, champion_outcomes, strict=True)
            )
            regression_evidence = {
                "candidate_manifest_sha256": candidate_manifest_sha256,
                "champion_manifest_sha256": champion_manifest_sha256,
                "candidate_artifact_sha256": candidate_artifact_sha256,
                "champion_artifact_sha256": champion_artifact_sha256,
                "task_count": candidate_count,
                "regression_count": regressions,
                "improvement_count": improvements,
                "unchanged_count": candidate_count - regressions - improvements,
                "candidate_task_successes": candidate_outcomes,
                "champion_task_successes": champion_outcomes,
                "candidate_task_environments": candidate_environments,
                "champion_task_environments": champion_environments,
            }
            expected_regression_digest = hashlib.sha256(
                json.dumps(
                    regression_evidence, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            expected_decision = (
                "REGRESSED" if regressions else "IMPROVED" if improvements else "UNCHANGED"
            )
            if (
                report.get("paired_outcomes_sha256") != paired_digest
                or report.get("regression_evidence_sha256") != expected_regression_digest
                or report.get("regression_count") != regressions
                or report.get("improvement_count") != improvements
                or report.get("unchanged_count") != candidate_count - regressions - improvements
                or report.get("regression_decision") != expected_decision
            ):
                raise ValueError("paired sealed evaluation digest is invalid")
        except Exception as exc:
            raise LiveExecutionFailed(
                "SageMaker evaluator report metrics or paired provenance are invalid"
            ) from exc

        artifact_id = f"evaluation-report://{artifact.sha256}"
        artifact_ref = ArtifactReference(
            artifact_id=artifact_id,
            kind=ArtifactKind.REPORT,
            uri=artifact.version_ref,
            sha256=artifact.sha256,
            size_bytes=artifact.size_bytes,
        )
        signed_evidence = Evidence(
            evidence_id=artifact_id,
            kind=EvidenceKind.EVALUATION,
            label=EvidenceLabel.LIVE,
            artifact_ids=(artifact_id,),
            metrics={"aggregate": candidate_metrics.aggregate, **candidate_metrics.per_environment},
            verified=True,
            benchmark_id=state.benchmark_id,
            suite=self.config.objective_suite,
            suite_version=self.config.objective_suite_version,
            manifest_sha256=state.benchmark_manifest_sha256,
            seed=state.benchmark_seed,
            model_id=state.model_id,
        )
        champion_evidence = signed_evidence.model_copy(
            update={
                "metrics": {
                    "aggregate": champion_metrics.aggregate,
                    **champion_metrics.per_environment,
                }
            }
        )
        champion_run_id = f"eval://{state.run_id}/checkpoint/{champion_artifact_sha256}"
        candidate_run_id = f"eval://{state.run_id}/checkpoint/{candidate_artifact_sha256}"
        champion_run_number = 0
        metadata = getattr(state, "metadata", {})
        if isinstance(metadata, Mapping):
            encoded_champion = metadata.get("champion_evaluation_b64")
            if isinstance(encoded_champion, str) and encoded_champion:
                try:
                    stored = json.loads(urlsafe_b64decode(encoded_champion.encode("ascii")))
                    stored_number = (
                        stored.get("run_number") if isinstance(stored, Mapping) else None
                    )
                    if type(stored_number) is not int or stored_number < 0:
                        raise ValueError("stored champion run number is invalid")
                    champion_run_number = stored_number
                except Exception as exc:
                    raise LiveExecutionFailed("stored champion evaluation is invalid") from exc
        if champion_run_number == 0:
            records = getattr(state, "experiments", ())
            champion_run_number = max(
                (
                    item.experiment_number
                    for item in records
                    if getattr(getattr(item, "status", None), "value", None) == "SUCCEEDED"
                ),
                default=0,
            )
        evaluation = MultiRunEvaluation(
            run_id=candidate_run_id,
            run_number=champion_run_number + 1,
            champion_run_id=champion_run_id,
            aggregate_score=candidate_metrics.aggregate,
            environment_scores=candidate_metrics.per_environment,
            evidence=signed_evidence,
        )
        champion_evaluation = MultiRunEvaluation(
            run_id=champion_run_id,
            run_number=champion_run_number,
            aggregate_score=champion_metrics.aggregate,
            environment_scores=champion_metrics.per_environment,
            evidence=champion_evidence,
        )
        return EvaluationEvidence(
            evaluation=evaluation,
            artifact_ids=(artifact_ref.artifact_id,),
            champion_evaluation=champion_evaluation,
            cost_usd=0.0,
        )


@dataclass(frozen=True, slots=True)
class AutonomousLiveComponents:
    """Concrete, lazy-safe AWS components used by the live API lifecycle."""

    repository: Any
    supervisor: Any
    dispatcher: Any


def create_autonomous_live_components(
    config: LiveExecutionConfig,
    *,
    repository: Any,
    telemetry: Any | None = None,
    model: Any | None = None,
    provider: Any | None = None,
    objective_worker: ObjectiveWorkerClient | None = None,
    artifact_store: S3ArtifactStore | None = None,
    owner: str | None = None,
) -> AutonomousLiveComponents:
    """Construct real supervisor adapters without provisioning cloud resources.

    Every dependency is either supplied by the deployment or built from the
    configured AWS SDK wrappers.  No fake objective, model, metric, or
    provider adapter is ever selected when construction fails.
    """

    from app.autonomous.agents import AutonomousAgentAdapters
    from app.autonomous.dispatcher import AutonomousRunDispatcher
    from app.autonomous.supervisor import AutonomousRunSupervisor
    from app.autonomous.telemetry import DurableTelemetryBridge
    from app.providers.bedrock import BedrockStrandsModel

    if repository is None:
        raise LiveExecutionBlocked("durable autonomous repository is required")
    worker = objective_worker or ObjectiveWorkerClient(
        config.objective_worker_url,
        auth_token=config.objective_worker_auth_token,
        timeout_seconds=config.objective_worker_timeout_seconds,
    )
    sagemaker = provider or SageMakerProvider(region_name=config.aws_region)
    store = artifact_store or S3ArtifactStore(
        config.artifact_bucket,
        prefix=config.artifact_prefix,
    )
    reasoning = model or BedrockStrandsModel(
        NEMOTRON_MODEL_ID,
        region_name=config.aws_region,
        auth_mode="sigv4",
    )
    agents = AutonomousAgentAdapters(reasoning)
    objective = LiveObjectiveAdapter(worker, config)
    request_factory = LiveRequestFactory(config)
    artifacts = S3LiveArtifactVerifier(store, config)
    evaluator = LiveEvaluationReader(store, config)
    durable_telemetry = DurableTelemetryBridge(repository, recorder=telemetry)

    def readiness(_: Any) -> bool:
        return PreflightRunner(config).run().ready

    def approval_verifier(state: Any) -> datetime | None:
        expiry = getattr(state, "approval_expires_at", None)
        return expiry if isinstance(expiry, datetime) else None

    supervisor = AutonomousRunSupervisor(
        repository=repository,
        objective=objective,
        agents=cast(Any, agents),
        provider=sagemaker,
        request_factory=request_factory,
        artifacts=artifacts,
        evaluator=evaluator,
        readiness=cast(Any, readiness),
        approval_verifier=approval_verifier,
        telemetry=cast(Any, durable_telemetry),
        max_polls=config.provider_max_polls,
        poll_interval_seconds=config.provider_poll_interval_seconds,
        phase_cost_upper_bounds_usd=config.phase_cost_upper_bounds_usd,
    )
    dispatcher = AutonomousRunDispatcher(
        repository=repository,
        supervisor=supervisor,
        owner=owner or f"api-{uuid4().hex}",
    )
    return AutonomousLiveComponents(
        repository=repository,
        supervisor=supervisor,
        dispatcher=dispatcher,
    )


def config_from_environment(environ: Mapping[str, str] | None = None) -> LiveExecutionConfig:
    """Build live configuration from explicit environment variables."""

    env = dict(os.environ if environ is None else environ)
    required = {
        "artifact_bucket": "S3_ARTIFACT_BUCKET",
        "dynamodb_table": "DYNAMODB_TABLE_NAME",
        "training_role_arn": "SAGEMAKER_TRAINING_ROLE_ARN",
        "training_image": "SAGEMAKER_TRAINING_IMAGE_URI",
        "evaluation_image": "SAGEMAKER_EVALUATION_IMAGE_URI",
        "objective_worker_url": "OBJECTIVE_WORKER_URL",
        "hf_repo_id": "HF_REPO_ID",
        "hf_revision": "HF_REVISION",
        "evaluation_input_s3_uri": "EVALUATION_INPUT_S3_URI",
    }
    missing = [key for key, variable in required.items() if not env.get(variable)]
    if missing:
        raise LiveExecutionBlocked("missing live configuration: " + ", ".join(missing))
    values: dict[str, Any] = {key: env[variable] for key, variable in required.items()}
    values.update(
        {
            "training_input_s3_uri": env.get("TRAINING_INPUT_S3_URI") or None,
            "aws_region": env.get("AWS_REGION", "us-east-1"),
            "objective_worker_auth_token": (
                env.get("OBJECTIVE_WORKER_AUTH_TOKEN") or env.get("OBJECTIVE_WORKER_TOKEN")
            ),
            "target_model": env.get("TARGET_MODEL", "google/functiongemma-270m-it"),
            "objective_suite": env.get("OBJECTIVE_SUITE", "AgentGym/AgentEval"),
            "objective_suite_version": env.get("OBJECTIVE_SUITE_VERSION", "agent-eval-v1"),
            "seed": int(env.get("POSTTRAINING_SEED", "7")),
            "baseline_episodes": int(env.get("BASELINE_EPISODES", "10")),
            "held_out_episodes": int(env.get("HELD_OUT_EPISODES", "15")),
            "max_runs": int(env.get("MAX_EXPERIMENTS", "5")),
            "max_cost_usd": float(env.get("MAX_COST_USD", "25")),
            "max_runtime_seconds": int(env.get("MAX_TRAINING_TIME_MIN", "120")) * 60,
            "artifact_prefix": env.get("S3_ARTIFACT_PREFIX", "post-training"),
            "checkpoint_sha256": env.get("CHECKPOINT_SHA256") or None,
            "checkpoint_s3_uri": env.get("CHECKPOINT_S3_URI") or None,
            "instance_type": env.get("SAGEMAKER_INSTANCE_TYPE", "ml.g5.xlarge"),
            "gpu_instance_allowlist": tuple(
                item.strip()
                for item in env.get("GPU_INSTANCE_ALLOWLIST", "ml.g5.xlarge").split(",")
                if item.strip()
            ),
            "sagemaker_gpu_quota_code": env.get("SAGEMAKER_GPU_QUOTA_CODE") or None,
            "sagemaker_processing_gpu_quota_code": (
                env.get("SAGEMAKER_PROCESSING_GPU_QUOTA_CODE") or None
            ),
            "minimum_gpu_quota": float(env.get("MINIMUM_GPU_QUOTA", "1")),
            "instance_count": int(env.get("SAGEMAKER_INSTANCE_COUNT", "1")),
            "volume_size_gb": int(env.get("SAGEMAKER_VOLUME_SIZE_GB", "30")),
            "approval_secret_env": env.get("LIVE_APPROVAL_SECRET_ENV", "LIVE_APPROVAL_SECRET"),
            "approval_ttl_seconds": int(env.get("LIVE_APPROVAL_TTL_SECONDS", "86400")),
            "objective_worker_timeout_seconds": int(
                env.get("OBJECTIVE_WORKER_TIMEOUT_SECONDS", "600")
            ),
        }
    )
    return LiveExecutionConfig.model_validate(values)


def safe_json_print(value: Mapping[str, Any]) -> None:
    """Print only structured metadata; never provider response bodies."""

    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


def create_aws_controller(config: LiveExecutionConfig) -> AutonomousRunController:
    """Construct real AWS adapters without creating any cloud resources."""

    try:
        import boto3
    except ImportError as exc:  # pragma: no cover
        raise LiveExecutionBlocked("boto3 is required for live execution") from exc
    s3_client = boto3.client("s3", region_name=config.aws_region)
    store = S3ArtifactStore(
        config.artifact_bucket,
        client=s3_client,
        prefix=config.artifact_prefix,
    )
    repository = DynamoDBRunRepository(
        table_name=config.dynamodb_table,
        region_name=config.aws_region,
    )

    def load_champion(run_number: int) -> RunHistoryRecord | None:
        if run_number <= 1:
            return None
        records = repository.list_history_runs(limit=MAX_RUNS)
        promoted = [record for record in records if record.decision is RunDecision.PROMOTE]
        return promoted[-1] if promoted else None

    return AutonomousRunController(
        config=config,
        objective_worker=ObjectiveWorkerClient(
            config.objective_worker_url,
            auth_token=config.objective_worker_auth_token,
        ),
        provider=SageMakerProvider(region_name=config.aws_region),
        artifact_store=store,
        slots=_RegistrySlotStore(repository),
        champion_loader=load_champion,
    )


__all__ = [
    "NEMOTRON_MODEL_ID",
    "ApprovalPacket",
    "AutonomousLiveComponents",
    "AutonomousRunController",
    "CheckResult",
    "CheckStatus",
    "GpuCapacityStatus",
    "GpuQuotaStatus",
    "LiveEvaluationReader",
    "LiveExecutionBlocked",
    "LiveExecutionConfig",
    "LiveExecutionFailed",
    "LiveObjectiveAdapter",
    "LiveRequestFactory",
    "LiveRunConfig",
    "LiveRunSummary",
    "ObjectiveWorkerClient",
    "PreflightReport",
    "PreflightRunner",
    "PreflightStatus",
    "RunSlotStore",
    "S3LiveArtifactVerifier",
    "_RegistrySlotStore",
    "config_from_environment",
    "create_autonomous_live_components",
    "create_aws_controller",
    "issue_approval_token",
    "safe_json_print",
]
