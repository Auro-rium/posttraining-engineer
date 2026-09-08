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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from math import isfinite
from typing import Any, Protocol
from urllib.parse import urljoin, urlparse
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
from app.providers.artifacts import ArtifactRef, ArtifactStore, S3ArtifactStore
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
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


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
    # Defaults preserve the low-level signing helper for generic packets;
    # controller validation rejects empty provenance values for execution.
    target_model: str = ""
    objective_suite: str = ""
    objective_suite_version: str = ""
    seed: int = 0
    reasoning_model_id: str = ""
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

    @field_validator("estimated_cost_usd")
    @classmethod
    def validate_finite_cost(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("estimated_cost_usd must be finite")
        return value

    @field_validator("manifest_sha256", "checkpoint_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("approval hashes must be lowercase 64-character SHA-256 digests")
        return value

    @model_validator(mode="after")
    def validate_window(self) -> ApprovalPacket:
        if self.packet_version != "v1":
            raise ValueError("unsupported approval packet version")
        if self.expires_at <= self.issued_at:
            raise ValueError("approval expiry must be after issuance")
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
    hf_repo_id: str
    hf_revision: str
    target_model: str = "google/functiongemma-270m-it"
    objective_suite: str = "AgentGym/AgentEval"
    objective_suite_version: str = "agent-eval-v1"
    seed: int = 7
    baseline_episodes: int = Field(default=10, ge=1)
    held_out_episodes: int = Field(default=15, ge=1)
    training_input_s3_uri: str
    evaluation_input_s3_uri: str
    checkpoint_s3_uri: str | None = None
    # Required for run 1; later runs derive this from the promoted artifact.
    checkpoint_sha256: str | None = None
    instance_type: str = "ml.g5.xlarge"
    gpu_instance_allowlist: tuple[str, ...] = ("ml.g5.xlarge",)
    sagemaker_gpu_quota_code: str | None = None
    minimum_gpu_quota: float = Field(default=1.0, gt=0)
    instance_count: int = Field(default=1, ge=1)
    volume_size_gb: int = Field(default=30, ge=1)
    max_runtime_seconds: int = Field(default=120 * 60, ge=1, le=120 * 60)
    training_hourly_cost_usd: float = Field(default=1.50, ge=0)
    evaluation_hourly_cost_usd: float = Field(default=1.00, ge=0)
    max_runs: int = Field(default=MAX_RUNS, ge=1, le=MAX_RUNS)
    max_cost_usd: float = Field(default=25.0, ge=0, le=25.0)
    artifact_prefix: str = "post-training"
    approval_token_env: str = "LIVE_APPROVAL_TOKEN"
    approval_secret_env: str = "LIVE_APPROVAL_SECRET"
    approval_ttl_seconds: int = Field(default=900, ge=60, le=86400)
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

    @field_validator("sagemaker_gpu_quota_code")
    @classmethod
    def validate_quota_code(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[A-Za-z0-9-]{1,128}", value):
            raise ValueError("sagemaker_gpu_quota_code must be a safe Service Quotas code")
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
        worst_case = self.max_runs * runtime_hours * (
            self.training_hourly_cost_usd + self.evaluation_hourly_cost_usd
        )
        if worst_case > self.max_cost_usd:
            raise ValueError(
                f"worst-case SageMaker estimate ${worst_case:.2f} exceeds "
                f"the ${self.max_cost_usd:.2f} ceiling"
            )
        return self

    @property
    def estimated_worst_case_cost_usd(self) -> float:
        return round(
            self.max_runs
            * (self.max_runtime_seconds / 3600)
            * (self.training_hourly_cost_usd + self.evaluation_hourly_cost_usd),
            6,
        )

    @property
    def estimated_run_cost_usd(self) -> float:
        """Worst-case cost for one training plus evaluation pair."""

        return round(
            (self.max_runtime_seconds / 3600)
            * (self.training_hourly_cost_usd + self.evaluation_hourly_cost_usd),
            6,
        )


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


class PreflightRunner:
    """Read-only AWS/Hugging Face readiness checks.

    Client factories are injectable.  Tests can verify that no write method is
    called, and production uses lazy boto3/Hugging Face clients.
    """

    def __init__(
        self,
        config: LiveExecutionConfig,
        *,
        clients: Mapping[str, Any] | None = None,
        hf_api: Any | None = None,
    ) -> None:
        self.config = config
        self.clients = dict(clients or {})
        self.hf_api = hf_api
        self._gpu_quota_status = GpuQuotaStatus.UNKNOWN
        self._gpu_capacity_status = GpuCapacityStatus.UNKNOWN

    def _client(self, name: str) -> Any:
        if name in self.clients:
            return self.clients[name]
        try:
            import boto3  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise LiveExecutionBlocked("boto3 is required for AWS preflight") from exc
        client = boto3.client(name, region_name=self.config.aws_region)
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
        checks.append(self._check_hugging_face())
        checks.append(
            self._check(
                "s3_artifact_bucket",
                lambda: self._check_s3_readiness(),
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
            blocked_classifications[0]
            if blocked_classifications
            else PreflightClassification.READY
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

    def _check_hugging_face(self) -> CheckResult:
        def operation() -> Mapping[str, Any]:
            api = self.hf_api
            if api is None:
                try:
                    from huggingface_hub import HfApi  # type: ignore[import-not-found]
                except ImportError as exc:  # pragma: no cover
                    raise LiveExecutionBlocked(
                        "huggingface_hub is required for HF preflight"
                    ) from exc
                api = HfApi()
            info = api.model_info(self.config.hf_repo_id, revision=self.config.hf_revision)
            resolved = str(getattr(info, "sha", ""))
            if resolved.lower() != self.config.hf_revision.lower():
                raise ValueError("Hugging Face revision did not resolve to the requested commit")
            return {"repo_id": self.config.hf_repo_id, "revision": resolved.lower()}

        return self._check("huggingface_pinned_revision", operation)

    def _check_s3_readiness(self) -> Mapping[str, Any]:
        client = self._client("s3")
        client.head_bucket(Bucket=self.config.artifact_bucket)
        versioning = client.get_bucket_versioning(Bucket=self.config.artifact_bucket)
        if versioning.get("Status") != "Enabled":
            raise ValueError("S3 bucket versioning is not enabled")
        return {"bucket": self.config.artifact_bucket, "versioning": "Enabled"}

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
        response = self._client("s3").head_object(
            Bucket=parsed.netloc,
            Key=parsed.path.lstrip("/"),
        )
        metadata = response.get("Metadata", {})
        observed = str(metadata.get("sha256", "")) if isinstance(metadata, Mapping) else ""
        if observed != digest:
            raise LiveExecutionBlocked(
                "checkpoint S3 metadata digest does not match CHECKPOINT_SHA256",
                classification=PreflightClassification.BLOCKED_CONFIGURATION,
            )
        version_id = response.get("VersionId")
        if not version_id:
            raise LiveExecutionBlocked(
                "checkpoint artifact has no S3 version id",
                classification=PreflightClassification.BLOCKED_CONFIGURATION,
            )
        return {
            "checkpoint": "verified",
            "version_id": str(version_id),
        }

    def _check_bedrock_readiness(self) -> Mapping[str, Any]:
        response = self._client("bedrock").get_foundation_model(
            modelIdentifier=NEMOTRON_MODEL_ID
        )
        summary = response.get("modelDetails", {})
        return {
            "model_id": NEMOTRON_MODEL_ID,
            "region": self.config.aws_region,
            "provider": str(summary.get("providerName", "unknown")),
        }

    def _check_dynamodb_readiness(self) -> Mapping[str, Any]:
        description = self._client("dynamodb").describe_table(TableName=self.config.dynamodb_table)
        table = description.get("Table", {})
        if table.get("TableStatus") != "ACTIVE":
            raise ValueError("DynamoDB table is not ACTIVE")
        return {"table": self.config.dynamodb_table, "status": str(table.get("TableStatus"))}

    def _check_sagemaker_readiness(self) -> Mapping[str, Any]:
        self._client("iam").get_role(RoleName=self.config.training_role_arn.rsplit("/", 1)[-1])
        ecr = self._client("ecr")
        for image in (self.config.training_image, self.config.evaluation_image):
            parsed = image.split("/", 1)
            if len(parsed) != 2:
                raise ValueError("image URI must include an ECR registry and repository")
            repository, tag = parsed[1].split(":", 1) if ":" in parsed[1] else (parsed[1], "latest")
            ecr.describe_images(repositoryName=repository, imageIds=[{"imageTag": tag}])
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

        if not self.config.sagemaker_gpu_quota_code:
            self._gpu_quota_status = GpuQuotaStatus.NOT_CONFIGURED
            self._gpu_capacity_status = GpuCapacityStatus.UNKNOWN
            return CheckResult(
                name="gpu_quota",
                status=CheckStatus.BLOCKED,
                detail="SageMaker GPU Service Quotas code is not configured",
                metadata={"instance_type": self.config.instance_type},
                classification=PreflightClassification.BLOCKED_CONFIGURATION,
            )

        try:
            response = self._client("service-quotas").get_service_quota(
                ServiceCode="sagemaker",
                QuotaCode=self.config.sagemaker_gpu_quota_code,
            )
            quota = response.get("Quota", {})
            raw_value = quota.get("Value")
            quota_value = float(raw_value)
            if not isfinite(quota_value) or quota_value < 0:
                raise ValueError("SageMaker GPU quota value is not finite")
        except Exception as exc:
            self._gpu_quota_status = GpuQuotaStatus.UNKNOWN
            self._gpu_capacity_status = GpuCapacityStatus.UNKNOWN
            return CheckResult(
                name="gpu_quota",
                status=CheckStatus.BLOCKED,
                detail=_safe_detail(exc),
                metadata={"instance_type": self.config.instance_type},
                classification=PreflightClassification.BLOCKED_GPU_QUOTA,
            )

        self._gpu_quota_status = GpuQuotaStatus.VERIFIED
        self._gpu_capacity_status = GpuCapacityStatus.VERIFIED_BY_QUOTA
        metadata = {
            "instance_type": self.config.instance_type,
            "quota_code": self.config.sagemaker_gpu_quota_code,
            "quota_value": str(quota_value),
            "required_capacity": str(self.config.minimum_gpu_quota),
            "capacity_status": self._gpu_capacity_status.value,
        }
        if quota_value < self.config.minimum_gpu_quota or quota_value < self.config.instance_count:
            self._gpu_quota_status = GpuQuotaStatus.INSUFFICIENT
            self._gpu_capacity_status = GpuCapacityStatus.UNAVAILABLE
            metadata["capacity_status"] = self._gpu_capacity_status.value
            return CheckResult(
                name="gpu_quota",
                status=CheckStatus.BLOCKED,
                detail="SageMaker GPU quota is below the requested run capacity",
                metadata=metadata,
                classification=PreflightClassification.BLOCKED_GPU_QUOTA,
            )
        return CheckResult(
            name="gpu_quota",
            status=CheckStatus.PASSED,
            detail="GPU instance is allowlisted and covered by account quota",
            metadata=metadata,
        )

    def _check_worker_readiness(self) -> Mapping[str, Any]:
        response = _http_json(
            urljoin(self.config.objective_worker_url, "health"),
            method="GET",
            timeout=self.config.preflight_timeout_seconds,
        )
        if not isinstance(response, Mapping) or str(response.get("status", "")).lower() not in {
            "ok",
            "ready",
            "healthy",
        }:
            raise ValueError("objective worker did not report ready")
        return {"endpoint": "configured", "status": "ready"}


def _http_json(
    url: str,
    *,
    method: str,
    payload: Mapping[str, Any] | None = None,
    timeout: float,
) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        body = response.read()
    if not body:
        return {}
    return json.loads(body.decode("utf-8"))


class ObjectiveWorkerClient:
    """HTTP adapter for the isolated objective worker; no local fallback."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 120.0) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("objective worker URL must be HTTPS")
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout_seconds = timeout_seconds

    def execute_benchmark(self, request: ObjectiveBenchmarkRequest) -> ObjectiveBenchmarkResult:
        response = _http_json(
            urljoin(self.base_url, "v1/benchmark"),
            method="POST",
            timeout=self.timeout_seconds,
            payload=request.model_dump(mode="json"),
        )
        if not isinstance(response, Mapping):
            raise LiveExecutionFailed("objective worker returned a non-object benchmark response")
        return ObjectiveBenchmarkResult.model_validate(response)

    def evaluate(
        self,
        *,
        model_uri: str,
        run_id: str,
        split: str,
        episodes: int,
        output_s3_uri: str,
        suite: str = "AgentGym/AgentEval",
        suite_version: str = "agent-eval-v1",
        seed: int = 7,
    ) -> ObjectiveBenchmarkResult:
        response = _http_json(
            urljoin(self.base_url, "v1/evaluate"),
            method="POST",
            timeout=self.timeout_seconds,
            payload={
                "run_id": run_id,
                "model_uri": model_uri,
                "suite": suite,
                "suite_version": suite_version,
                "seed": seed,
                "num_episodes": episodes,
                "split": split,
                "output_s3_uri": output_s3_uri,
            },
        )
        if not isinstance(response, Mapping):
            raise LiveExecutionFailed("objective worker returned a non-object evaluation response")
        return ObjectiveBenchmarkResult.model_validate(response)


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

    def run_once(self, *, run_number: int, approval_token: str) -> LiveRunSummary:
        self._validate_run_number(run_number)
        # Decode and authenticate before touching a provider.  The signed
        # packet carries the run id and manifest digest, so a token cannot be
        # replayed against a different run or configuration.
        packet = self._require_approval(approval_token, run_number=run_number)
        preflight = PreflightRunner(self.config).run()
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
                model_uri=(
                    self._checkpoint_uri()
                    if champion_record is None
                    else self._champion_uri(champion_record)
                ),
                split="baseline",
                episodes=self.config.baseline_episodes,
                output_s3_uri=f"s3://{self.config.artifact_bucket}/{self.config.artifact_prefix}/{run_id}/baseline",
            )
            training_request = TrainingJobRequest(
                job_name=f"apt-{run_id}-train",
                role_arn=self.config.training_role_arn,
                image_uri=self.config.training_image,
                input_s3_uri=self.config.training_input_s3_uri,
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
            job_metadata_artifacts.append(self._persist_job_metadata(
                run_id,
                phase="training",
                job=training_job,
                manifest_sha256=manifest_sha256,
            ))
            self._event(EventType.JOB_SUBMITTED, run_id, run_number, status="training")
            training_job = wait_for_training_job(
                self.provider,
                training_job.job_name,
                policy=self._wait_policy(),
                sleep=self.sleep,
            )
            if training_job.status is not JobStatus.COMPLETED or not training_job.artifact_uri:
                raise LiveExecutionFailed("training did not complete with a model artifact")
            self._event(EventType.JOB_COMPLETED, run_id, run_number, status="training")
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
            job_metadata_artifacts.append(self._persist_job_metadata(
                run_id,
                phase="evaluation",
                job=evaluation_job,
                manifest_sha256=manifest_sha256,
            ))
            self._event(EventType.JOB_SUBMITTED, run_id, run_number, status="evaluation")
            evaluation_job = wait_for_evaluation_job(
                self.provider,
                evaluation_job.job_name,
                policy=self._wait_policy(),
                sleep=self.sleep,
            )
            if evaluation_job.status is not JobStatus.COMPLETED:
                raise LiveExecutionFailed("evaluation did not complete")
            self._event(EventType.JOB_COMPLETED, run_id, run_number, status="evaluation")
            candidate = self._benchmark(
                run_id=run_id,
                model_uri=training_job.artifact_uri,
                split="held_out",
                episodes=self.config.held_out_episodes,
                output_s3_uri=f"s3://{self.config.artifact_bucket}/{self.config.artifact_prefix}/{run_id}/held-out",
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
    ) -> None:
        """Close a reserved slot without inventing metrics or evidence."""

        failed = RunHistoryRecord(
            run_id=run_id,
            run_number=run_number,
            parent_run_id=parent_run_id,
            champion_run_id=champion_run_id,
            status=RunStatus.FAILED,
            manifest_sha256=manifest_sha256,
            artifact_refs=(manifest_artifact,) if manifest_artifact else (),
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
        manifest_payload = self._manifest_payload(
            run_id=actual_run_id,
            run_number=run_number,
            parent_run_id=parent_run_id,
            champion_run_id=champion_run_id,
            checkpoint_sha256=checkpoint_sha256,
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
            target_model=self.config.target_model,
            objective_suite=self.config.objective_suite,
            objective_suite_version=self.config.objective_suite_version,
            seed=self.config.seed,
            reasoning_model_id=NEMOTRON_MODEL_ID,
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
            for artifact in champion_record.artifact_refs:
                if artifact.artifact_id == champion_record.candidate_artifact_id:
                    return artifact.sha256
            raise LiveExecutionBlocked("promoted champion checkpoint digest is missing")
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
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "run_number": run_number,
            "parent_run_id": parent_run_id,
            "champion_run_id": champion_run_id,
            "target_model": self.config.target_model,
            "hf_repo_id": self.config.hf_repo_id,
            "hf_revision": self.config.hf_revision,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_s3_uri": self.config.checkpoint_s3_uri,
            "training_input_s3_uri": self.config.training_input_s3_uri,
            "evaluation_input_s3_uri": self.config.evaluation_input_s3_uri,
            "suite": self.config.objective_suite,
            "suite_version": self.config.objective_suite_version,
            "seed": self.config.seed,
            "model_reasoning_id": NEMOTRON_MODEL_ID,
            "training_role_arn": self.config.training_role_arn,
            "training_image": self.config.training_image,
            "evaluation_image": self.config.evaluation_image,
            "instance_type": self.config.instance_type,
            "instance_count": self.config.instance_count,
            "volume_size_gb": self.config.volume_size_gb,
            "max_runtime_seconds": self.config.max_runtime_seconds,
        }

    def _validate_approval_packet(
        self, packet: ApprovalPacket, preflight: PreflightReport
    ) -> None:
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
            packet.target_model != self.config.target_model
            or packet.objective_suite != self.config.objective_suite
            or packet.objective_suite_version != self.config.objective_suite_version
            or packet.seed != self.config.seed
            or packet.reasoning_model_id != NEMOTRON_MODEL_ID
        ):
            raise LiveExecutionBlocked(
                "approval packet benchmark provenance does not match configuration"
            )
        if preflight.gpu_capacity_status is not GpuCapacityStatus.VERIFIED_BY_QUOTA:
            raise LiveExecutionBlocked("GPU quota/capacity is not verified for this run")

    def _checkpoint_uri(self) -> str:
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
        split: str,
        episodes: int,
        output_s3_uri: str,
    ) -> ObjectiveBenchmarkResult:
        request = ObjectiveBenchmarkRequest(
            run_id=run_id,
            model_uri=model_uri,
            suite=self.config.objective_suite,
            suite_version=self.config.objective_suite_version,
            seed=self.config.seed,
            num_episodes=episodes,
            split=split,
            output_s3_uri=output_s3_uri,
        )
        return execute_objective_benchmark(self.objective_worker, request)

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
                champion_run_number or run_number - 1
                if champion_run_id
                else run_number - 1
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
    def _champion_uri(record: RunHistoryRecord) -> str:
        for artifact in record.artifact_refs:
            if artifact.artifact_id == record.candidate_artifact_id:
                return artifact.uri
        raise LiveExecutionBlocked("promoted champion artifact reference is missing")

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
        if parsed.scheme != "s3":
            raise LiveExecutionFailed("provider artifact is not an S3 URI")
        head = getattr(self.artifact_store, "head", None)
        if not callable(head):
            raise LiveExecutionFailed("artifact store cannot verify provider artifact metadata")
        ref = ArtifactRef.from_uri(job.artifact_uri, sha256="0" * 64)
        metadata = head(ref).get("Metadata", {})
        digest = str(metadata.get("sha256", "")) if isinstance(metadata, Mapping) else ""
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise LiveExecutionFailed("provider artifact has no verified SHA-256 metadata")
        return ArtifactReference(
            artifact_id=hashlib.sha256(job.artifact_uri.encode()).hexdigest()[:24],
            kind=kind,
            uri=job.artifact_uri,
            sha256=digest,
            metadata={"provider_job_id": str(job.provider_job_id)},
        )

    def _wait_policy(self) -> Any:
        from app.posttraining.objective_workflow import JobWaitPolicy

        return JobWaitPolicy(max_attempts=120, poll_interval_seconds=30.0)

    def _cleanup(
        self,
        training: JobResult | None,
        evaluation: JobResult | None,
        *,
        run_id: str,
        run_number: int,
    ) -> None:
        cleanup_ok = True
        for job, stop in (
            (training, self.provider.stop_training),
            (evaluation, self.provider.stop_evaluation),
        ):
            if job is not None and job.status in {JobStatus.SUBMITTED, JobStatus.IN_PROGRESS}:
                try:
                    stop(job.job_name)
                except Exception:
                    cleanup_ok = False
        self._event(
            EventType.CLEANUP_COMPLETED if cleanup_ok else EventType.CLEANUP_FAILED,
            run_id,
            run_number,
            status="completed" if cleanup_ok else "failed",
        )

    def _event(self, event_type: EventType, run_id: str, run_number: int, *, status: str) -> None:
        try:
            self.telemetry.record(
                event_type,
                run_id=run_id,
                run_number=run_number,
                experiment_id=f"{run_id}-experiment",
                status=status,
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
        "training_input_s3_uri": "TRAINING_INPUT_S3_URI",
        "evaluation_input_s3_uri": "EVALUATION_INPUT_S3_URI",
    }
    missing = [key for key, variable in required.items() if not env.get(variable)]
    if missing:
        raise LiveExecutionBlocked("missing live configuration: " + ", ".join(missing))
    values: dict[str, Any] = {key: env[variable] for key, variable in required.items()}
    values.update(
        {
            "aws_region": env.get("AWS_REGION", "us-east-1"),
            "target_model": env.get("TARGET_MODEL", "google/functiongemma-270m-it"),
            "objective_suite": env.get("OBJECTIVE_SUITE", "AgentGym/AgentEval"),
            "objective_suite_version": env.get("OBJECTIVE_SUITE_VERSION", "agent-eval-v1"),
            "seed": int(env.get("POSTTRAINING_SEED", "7")),
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
            "minimum_gpu_quota": float(env.get("MINIMUM_GPU_QUOTA", "1")),
            "instance_count": int(env.get("SAGEMAKER_INSTANCE_COUNT", "1")),
            "volume_size_gb": int(env.get("SAGEMAKER_VOLUME_SIZE_GB", "30")),
            "approval_secret_env": env.get("LIVE_APPROVAL_SECRET_ENV", "LIVE_APPROVAL_SECRET"),
            "approval_ttl_seconds": int(env.get("LIVE_APPROVAL_TTL_SECONDS", "900")),
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
        objective_worker=ObjectiveWorkerClient(config.objective_worker_url),
        provider=SageMakerProvider(region_name=config.aws_region),
        artifact_store=store,
        slots=_RegistrySlotStore(repository),
        champion_loader=load_champion,
    )


__all__ = [
    "NEMOTRON_MODEL_ID",
    "ApprovalPacket",
    "AutonomousRunController",
    "CheckResult",
    "CheckStatus",
    "GpuCapacityStatus",
    "GpuQuotaStatus",
    "LiveExecutionBlocked",
    "LiveExecutionConfig",
    "LiveExecutionFailed",
    "LiveRunConfig",
    "LiveRunSummary",
    "ObjectiveWorkerClient",
    "PreflightReport",
    "PreflightRunner",
    "PreflightStatus",
    "RunSlotStore",
    "_RegistrySlotStore",
    "config_from_environment",
    "create_aws_controller",
    "issue_approval_token",
    "safe_json_print",
]
