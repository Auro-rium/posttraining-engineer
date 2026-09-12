"""SageMaker training and evaluation provider interfaces.

Training uses a SageMaker training job; evaluation uses a processing job with
the same explicit input/output artifact contract.  The client is injected or
created lazily, so importing this module cannot require AWS credentials.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol
from urllib.parse import parse_qs, urlparse


class OptionalDependencyError(RuntimeError):
    """Raised when a live SageMaker client is requested without boto3."""


class ProviderResponseError(RuntimeError):
    """Raised when SageMaker returns an unsafe or inconsistent response."""


class ProviderReconciliationError(ProviderResponseError):
    """Raised when an existing job cannot be safely matched to a request."""


class JobUnidentifiableError(ProviderReconciliationError):
    """Raised when an existing same-name job lacks identity metadata."""


class JobNameConflictError(ProviderReconciliationError):
    """Raised when a deterministic job name belongs to another request."""


class ProviderJobNotFoundError(ProviderResponseError):
    """Raised when a status is requested for a missing provider job."""


class TransientProviderError(RuntimeError):
    """A transient provider read failure that a supervisor may retry."""

    def __init__(self, message: str, *, job_name: str, operation: str) -> None:
        super().__init__(message)
        self.job_name = job_name
        self.operation = operation


class JobStatus(StrEnum):
    SUBMITTED = "submitted"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class TrainingJobRequest:
    job_name: str
    role_arn: str
    image_uri: str
    input_s3_uri: str
    output_s3_uri: str
    instance_type: str
    parent_adapter_s3_uri: str | None = None
    instance_count: int = 1
    volume_size_gb: int = 30
    max_runtime_seconds: int = 3600
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    tags: list[dict[str, str]] = field(default_factory=list)

    @property
    def request_fingerprint(self) -> str:
        return request_fingerprint(self)

    @property
    def fingerprint(self) -> str:
        """Compatibility alias for callers that use the shorter name."""
        return self.request_fingerprint


@dataclass(slots=True)
class EvaluationJobRequest:
    job_name: str
    role_arn: str
    image_uri: str
    input_s3_uri: str
    output_s3_uri: str
    instance_type: str
    model_s3_uri: str | None = None
    # ProcessingInput names become SM_CHANNEL_* variables in the evaluator.
    # Keep the legacy fields as aliases while allowing the live path to state
    # all three independently and bind each to one exact S3 reference.
    candidate_s3_uri: str | None = None
    champion_s3_uri: str | None = None
    sealed_s3_uri: str | None = None
    instance_count: int = 1
    volume_size_gb: int = 30
    max_runtime_seconds: int = 3600
    environment: dict[str, str] = field(default_factory=dict)
    command: list[str] | None = None
    tags: list[dict[str, str]] = field(default_factory=list)

    @property
    def request_fingerprint(self) -> str:
        return request_fingerprint(self)

    @property
    def fingerprint(self) -> str:
        """Compatibility alias for callers that use the shorter name."""
        return self.request_fingerprint


@dataclass(slots=True)
class JobResult:
    job_name: str
    provider_job_id: str | None
    status: JobStatus
    artifact_uri: str | None = None
    failure_reason: str | None = None
    raw_response: Mapping[str, Any] = field(default_factory=dict)


_JOB_NAME_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
_PROVIDER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:/.@+_=,-]{0,1999}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TRANSIENT_OS_ERRNOS = frozenset(
    {
        errno.ECONNABORTED,
        errno.ECONNREFUSED,
        errno.ECONNRESET,
        errno.EHOSTDOWN,
        errno.EHOSTUNREACH,
        errno.ENETDOWN,
        errno.ENETRESET,
        errno.ENETUNREACH,
        errno.EPIPE,
        errno.ETIMEDOUT,
    }
)
_TRANSIENT_ERROR_CODES = frozenset(
    {
        "InternalError",
        "InternalFailure",
        "InternalServerError",
        "BandwidthLimitExceeded",
        "EC2ThrottledException",
        "LimitExceededException",
        "PriorRequestNotComplete",
        "RequestTimeout",
        "RequestTimeoutException",
        "RequestLimitExceeded",
        "ServiceUnavailable",
        "ServiceUnavailableException",
        "ServiceQuotaExceededException",
        "SlowDown",
        "Throttling",
        "ThrottledException",
        "ThrottlingException",
        "TooManyRequestsException",
    }
)
def _canonical(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical_request_tags(tags: list[dict[str, str]]) -> list[dict[str, str]]:
    filtered = [
        tag
        for tag in tags
        if str(tag.get("Key", "")).lower().replace("_", "-") != "request-fingerprint"
    ]
    return sorted(
        filtered,
        key=lambda tag: json.dumps(_canonical(tag), sort_keys=True, separators=(",", ":")),
    )


def request_fingerprint(request: TrainingJobRequest | EvaluationJobRequest) -> str:
    """Return a stable digest of the provider request, excluding its job name."""

    if isinstance(request, TrainingJobRequest):
        payload: dict[str, object] = {
            "kind": "training",
            "role_arn": request.role_arn,
            "image_uri": request.image_uri,
            "input_s3_uri": request.input_s3_uri,
            "parent_adapter_s3_uri": request.parent_adapter_s3_uri,
            "output_s3_uri": request.output_s3_uri,
            "instance_type": request.instance_type,
            "instance_count": request.instance_count,
            "volume_size_gb": request.volume_size_gb,
            "max_runtime_seconds": request.max_runtime_seconds,
            "hyperparameters": request.hyperparameters,
            "environment": request.environment,
            "tags": _canonical_request_tags(request.tags),
        }
    elif isinstance(request, EvaluationJobRequest):
        payload = {
            "kind": "processing",
            "role_arn": request.role_arn,
            "image_uri": request.image_uri,
            "input_s3_uri": request.input_s3_uri,
            "output_s3_uri": request.output_s3_uri,
            "instance_type": request.instance_type,
            "model_s3_uri": request.model_s3_uri,
            "candidate_s3_uri": request.candidate_s3_uri,
            "champion_s3_uri": request.champion_s3_uri,
            "sealed_s3_uri": request.sealed_s3_uri,
            "instance_count": request.instance_count,
            "volume_size_gb": request.volume_size_gb,
            "max_runtime_seconds": request.max_runtime_seconds,
            "environment": request.environment,
            "command": request.command,
            "tags": _canonical_request_tags(request.tags),
        }
    else:  # pragma: no cover - the type is closed to the two request records
        raise TypeError("unsupported SageMaker request type")
    encoded = json.dumps(_canonical(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def deterministic_job_name(prefix: str, fingerprint: str) -> str:
    """Build a SageMaker-safe deterministic name from a request digest."""

    if not isinstance(fingerprint, str) or not _SHA256_PATTERN.fullmatch(fingerprint):
        raise ValueError("fingerprint must be a lowercase SHA-256 digest")
    prefix = _require(prefix, "prefix").strip("-")
    if not prefix or not re.fullmatch(r"[A-Za-z0-9-]+", prefix):
        raise ValueError("prefix must contain only letters, digits, and hyphens")
    suffix = fingerprint[:24]
    return f"{prefix[: 63 - len(suffix) - 1]}-{suffix}"


class TrainingProvider(Protocol):
    def submit_training(self, request: TrainingJobRequest) -> JobResult: ...

    def get_training_status(self, job_name: str) -> JobResult: ...

    def stop_training(self, job_name: str) -> None: ...


class EvaluationProvider(Protocol):
    def submit_evaluation(self, request: EvaluationJobRequest) -> JobResult: ...

    def get_evaluation_status(self, job_name: str) -> JobResult: ...

    def stop_evaluation(self, job_name: str) -> None: ...


def _require(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _validate_job_name(value: str, name: str = "job_name") -> str:
    value = _require(value, name)
    if len(value) > 63 or not _JOB_NAME_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a SageMaker-safe name of at most 63 characters")
    return value


def _validate_provider_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ProviderResponseError("SageMaker response contained an invalid provider job ID")
    if len(value) > 2_000 or _PROVIDER_ID_PATTERN.fullmatch(value) is None:
        raise ProviderResponseError("SageMaker response contained an invalid provider job ID")
    return value


def _error_code(exc: BaseException) -> str | None:
    response = getattr(exc, "response", None)
    if isinstance(response, Mapping):
        error = response.get("Error")
        if isinstance(error, Mapping) and error.get("Code"):
            return str(error["Code"])
    return None


def _error_message(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, Mapping):
        error = response.get("Error")
        if isinstance(error, Mapping) and error.get("Message"):
            return str(error["Message"])
    return str(exc)


def is_transient_describe_error(exc: BaseException) -> bool:
    """Classify only provider read failures safe for bounded supervisor retry."""

    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, OSError):
        return getattr(exc, "errno", None) in _TRANSIENT_OS_ERRNOS
    if exc.__class__.__name__ in {
        "ConnectTimeoutError",
        "ConnectionClosedError",
        "EndpointConnectionError",
        "ReadTimeoutError",
        "SSLError",
    }:
        return True
    if _error_code(exc) in _TRANSIENT_ERROR_CODES:
        return True
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code in {429, 500, 502, 503, 504}:
        return True
    response = getattr(exc, "response", None)
    if isinstance(response, Mapping):
        metadata = response.get("ResponseMetadata")
        if isinstance(metadata, Mapping):
            status = metadata.get("HTTPStatusCode")
            return isinstance(status, int) and status in {429, 500, 502, 503, 504}
    return False


def _is_not_found_error(exc: BaseException) -> bool:
    code = _error_code(exc)
    if code == "ValidationException":
        message = _error_message(exc).lower()
        return any(
            token in message
            for token in ("not found", "does not exist", "cannot be found", "no such")
        )
    return code in {"ResourceNotFound", "ResourceNotFoundException"}


def _is_already_terminal_stop_error(exc: BaseException) -> bool:
    if _error_code(exc) != "ValidationException":
        return False
    message = _error_message(exc).lower()
    return any(token in message for token in ("already", "not in", "cannot stop", "completed"))


def _is_already_exists_error(exc: BaseException) -> bool:
    code = _error_code(exc)
    if code in {"ConflictException", "ResourceInUse", "ResourceInUseException"}:
        return True
    if code == "ValidationException":
        message = _error_message(exc).lower()
        return any(token in message for token in ("already exists", "already in use", "duplicate"))
    return False


def _safe_failure_reason(value: object) -> str | None:
    if value is None:
        return None
    # Provider failure text is untrusted and can contain credentials, prompts,
    # task contents, or container logs. Preserve only a fixed safe message.
    return "provider reported terminal failure"


def _safe_raw_response(
    response: Mapping[str, Any], *, kind: str, status: JobStatus
) -> dict[str, Any]:
    """Retain only provider metadata needed by local callers."""

    name_key = "TrainingJobName" if kind == "training" else "ProcessingJobName"
    arn_key = "TrainingJobArn" if kind == "training" else "ProcessingJobArn"
    status_key = "TrainingJobStatus" if kind == "training" else "ProcessingJobStatus"
    safe: dict[str, Any] = {key: response[key] for key in (name_key, arn_key) if key in response}
    if status_key in response:
        safe[status_key] = status.value
    if "FailureReason" in response:
        safe["FailureReason"] = _safe_failure_reason(response.get("FailureReason"))
    # Keep known billing/timing metadata under stable, metadata-only keys.
    # Reject malformed values instead of silently accounting a terminal job as
    # zero cost in the supervisor budget ledger.
    cost_value: object = None
    for key in (
        "actual_cost_usd",
        "ActualCostUsd",
        "ActualCostUSD",
        "cost_usd",
        "CostUsd",
        "CostUSD",
        "cost",
        "Cost",
    ):
        if key in response:
            cost_value = response[key]
            break
    if cost_value is not None:
        if (
            isinstance(cost_value, bool)
            or not isinstance(cost_value, (int, float))
            or not math.isfinite(float(cost_value))
            or float(cost_value) < 0
        ):
            raise ProviderResponseError("SageMaker response contained invalid cost metadata")
        safe["actual_cost_usd"] = float(cost_value)
    for key in (
        "BillableTimeInSeconds",
        "TrainingTimeInSeconds",
        "ProcessingTimeInSeconds",
    ):
        if key in response:
            value = response[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ProviderResponseError("SageMaker response contained invalid timing metadata")
            safe[key] = float(value)
    for key in (
        "TrainingStartTime",
        "TrainingEndTime",
        "ProcessingStartTime",
        "ProcessingEndTime",
    ):
        if key in response:
            value = response[key]
            isoformat = getattr(value, "isoformat", None)
            if callable(isoformat):
                value = isoformat()
            if isinstance(value, str) and value.strip():
                safe[key] = value[:128]
    return safe


def _validate_artifact_uri(value: object) -> str:
    if not isinstance(value, str):
        raise ProviderResponseError("SageMaker response contained an invalid artifact URI")
    parsed = urlparse(value)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ProviderResponseError("SageMaker response contained an invalid artifact URI")
    if (
        len(value) > 2_048
        or any(character.isspace() or ord(character) < 32 for character in value)
        or parsed.fragment
    ):
        raise ProviderResponseError("SageMaker response contained an invalid artifact URI")
    versions = parse_qs(parsed.query, keep_blank_values=True).get("versionId", [])
    if len(versions) > 1 or (versions and not versions[0].strip()):
        raise ProviderResponseError("SageMaker response contained an invalid artifact URI")
    return value


def _validate_input_uri(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an S3 URI")
    parsed = urlparse(value)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"{name} must be a non-empty S3 URI")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{name} contains an unsupported S3Uri query or fragment")
    if any(character.isspace() or ord(character) < 32 for character in value):
        raise ValueError(f"{name} must be a valid S3 URI")
    return value


def _validate_versioned_input_uri(value: object, name: str) -> str:
    # SageMaker cannot receive an S3 VersionId through S3Uri.  The caller must
    # first materialize and verify the exact bytes under a digest-addressed
    # key/prefix, then submit that query-free location.
    uri = _validate_input_uri(value, name)
    basename = urlparse(uri).path.rstrip("/").rsplit("/", 1)[-1]
    if re.fullmatch(r"[0-9a-f]{64}(?:\.[A-Za-z0-9._-]+)?", basename) is None:
        raise ValueError(f"{name} must be a content-addressed S3 URI")
    return uri


def _validate_digest_prefix(value: object, digest: object, name: str) -> str:
    uri = _validate_input_uri(value, name)
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{name} requires a lowercase SHA-256 digest")
    basename = urlparse(uri).path.rstrip("/").rsplit("/", 1)[-1]
    if basename != digest:
        raise ValueError(f"{name} must end in its expected SHA-256 digest")
    return uri


def _validate_digest_archive(value: object, digest: object, name: str) -> str:
    uri = _validate_input_uri(value, name)
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{name} requires a lowercase SHA-256 digest")
    basename = urlparse(uri).path.rstrip("/").rsplit("/", 1)[-1]
    if basename != f"{digest}.tar.gz":
        raise ValueError(f"{name} must match its expected archive SHA-256")
    return uri


def _required_environment(request: object, names: tuple[str, ...]) -> Mapping[str, str]:
    environment = getattr(request, "environment", None)
    if not isinstance(environment, Mapping):
        raise ValueError("SageMaker environment must be a mapping")
    missing = [
        name
        for name in names
        if not isinstance(environment.get(name), str) or not environment[name].strip()
    ]
    if missing:
        raise ValueError("SageMaker request is missing required environment: " + ", ".join(missing))
    return environment


def _status(value: object) -> JobStatus:
    normalized = str(value or "").strip().lower()
    if normalized in {"submitted", "pending", "starting"}:
        return JobStatus.SUBMITTED
    if normalized in {"inprogress", "in_progress", "stopping", "downloading", "uploading"}:
        return JobStatus.IN_PROGRESS
    if normalized in {"completed", "succeeded", "success"}:
        return JobStatus.COMPLETED
    if normalized in {"failed", "failure"}:
        return JobStatus.FAILED
    if normalized in {"stopped", "stopping"}:
        return JobStatus.STOPPED
    return JobStatus.UNKNOWN


class SageMakerProvider(TrainingProvider, EvaluationProvider):
    """Concrete SageMaker adapter for training and evaluation jobs."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        region_name: str | None = None,
    ) -> None:
        self._client = client
        self.region_name = region_name

    def _client_or_create(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import boto3  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise OptionalDependencyError("Install boto3 to use SageMakerProvider") from exc
        self._client = boto3.client("sagemaker", region_name=self.region_name)
        return self._client

    @staticmethod
    def _validate_training(request: TrainingJobRequest) -> None:
        for value, name in (
            (request.job_name, "job_name"),
            (request.role_arn, "role_arn"),
            (request.image_uri, "image_uri"),
            (request.input_s3_uri, "input_s3_uri"),
            (request.output_s3_uri, "output_s3_uri"),
            (request.instance_type, "instance_type"),
        ):
            _require(value, name)
        _validate_job_name(request.job_name)
        environment = _required_environment(
            request,
            (
                "RUN_ID",
                "EXPERIMENT_ID",
                "DATASET_ID",
                "DATASET_SHA256",
                "APPROVED_DATASET_ARTIFACT_ID",
                "BASE_MODEL_ID",
                "BASE_MODEL_REVISION",
                "QLORA_CONFIG",
            ),
        )
        _validate_digest_prefix(
            request.input_s3_uri, environment["DATASET_SHA256"], "input_s3_uri"
        )
        if environment["APPROVED_DATASET_ARTIFACT_ID"] != f"dataset://{environment['DATASET_ID']}":
            raise ValueError("APPROVED_DATASET_ARTIFACT_ID is not bound to DATASET_ID")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", environment["BASE_MODEL_REVISION"]):
            raise ValueError("BASE_MODEL_REVISION must be a 40-character immutable revision")
        try:
            qlora = json.loads(environment["QLORA_CONFIG"])
        except json.JSONDecodeError as exc:
            raise ValueError("QLORA_CONFIG must be a JSON object") from exc
        if not isinstance(qlora, Mapping):
            raise ValueError("QLORA_CONFIG must be a JSON object")
        if request.parent_adapter_s3_uri is not None:
            _validate_digest_archive(
                request.parent_adapter_s3_uri,
                environment.get("APPROVED_PARENT_ARCHIVE_SHA256"),
                "parent_adapter_s3_uri",
            )
            _required_environment(
                request,
                (
                    "APPROVED_PARENT_ARTIFACT_ID",
                    "APPROVED_PARENT_MANIFEST_SHA256",
                "APPROVED_PARENT_ARTIFACT_SHA256",
                "APPROVED_PARENT_ARCHIVE_SHA256",
                ),
            )
        elif any(
            name in environment
            for name in (
                "APPROVED_PARENT_ARTIFACT_ID",
                "APPROVED_PARENT_MANIFEST_SHA256",
                "APPROVED_PARENT_ARTIFACT_SHA256",
            )
        ):
            raise ValueError("approved parent metadata requires a parent adapter channel")
        _validate_input_uri(request.output_s3_uri, "output_s3_uri")
        if (
            request.instance_count < 1
            or request.volume_size_gb < 1
            or request.max_runtime_seconds < 1
        ):
            raise ValueError("SageMaker resource sizes and runtime must be positive")

    @staticmethod
    def _validate_evaluation(request: EvaluationJobRequest) -> None:
        for value, name in (
            (request.job_name, "job_name"),
            (request.role_arn, "role_arn"),
            (request.image_uri, "image_uri"),
            (request.input_s3_uri, "input_s3_uri"),
            (request.output_s3_uri, "output_s3_uri"),
            (request.instance_type, "instance_type"),
        ):
            _require(value, name)
        _validate_job_name(request.job_name)
        _validate_input_uri(request.input_s3_uri, "input_s3_uri")
        _validate_input_uri(request.output_s3_uri, "output_s3_uri")
        environment = _required_environment(
            request,
            (
                "RUN_ID",
                "EXPERIMENT_ID",
                "EVALUATION_MANIFEST_SHA256",
                "EVALUATION_SUITE_VERSION",
                "OBJECTIVE_SEED",
                "CANDIDATE_ARCHIVE_SHA256",
                "CHAMPION_ARCHIVE_SHA256",
            ),
        )
        for name in ("EVALUATION_MANIFEST_SHA256",):
            if _SHA256_PATTERN.fullmatch(environment[name]) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not re.fullmatch(r"-?[0-9]+", environment["OBJECTIVE_SEED"]):
            raise ValueError("OBJECTIVE_SEED must be an integer")
        if (
            request.candidate_s3_uri is None
            or request.champion_s3_uri is None
            or request.sealed_s3_uri is None
        ):
            raise ValueError("candidate, champion, and sealed evaluation channels are required")
        if request.model_s3_uri is not None:
            _require(request.model_s3_uri, "model_s3_uri")
            _validate_input_uri(request.model_s3_uri, "model_s3_uri")
        if request.candidate_s3_uri is not None:
            _validate_digest_archive(
                request.candidate_s3_uri,
                environment["CANDIDATE_ARCHIVE_SHA256"],
                "candidate_s3_uri",
            )
        if request.champion_s3_uri is not None:
            _validate_digest_archive(
                request.champion_s3_uri,
                environment["CHAMPION_ARCHIVE_SHA256"],
                "champion_s3_uri",
            )
        if request.sealed_s3_uri is not None:
            _validate_input_uri(request.sealed_s3_uri, "sealed_s3_uri")
        if (
            request.instance_count < 1
            or request.volume_size_gb < 1
            or request.max_runtime_seconds < 1
        ):
            raise ValueError("SageMaker resource sizes and runtime must be positive")

    @staticmethod
    def _fingerprint_from_tags(tags: object) -> str | None:
        if not isinstance(tags, list):
            return None
        for item in tags:
            if not isinstance(item, Mapping):
                continue
            key = str(item.get("Key", "")).lower().replace("_", "-")
            if key in {"request-fingerprint", "fingerprint"} and item.get("Value"):
                return str(item["Value"])
        return None

    def _response_fingerprint(self, response: Mapping[str, Any]) -> str | None:
        for key in ("RequestFingerprint", "request_fingerprint", "Fingerprint", "fingerprint"):
            if response.get(key):
                return str(response[key])
        return self._fingerprint_from_tags(response.get("Tags"))

    def _resource_tags(self, provider_id: str, response: Mapping[str, Any]) -> str | None:
        direct = self._response_fingerprint(response)
        if direct is not None:
            return direct
        list_tags = getattr(self._client_or_create(), "list_tags", None)
        if not callable(list_tags):
            return None
        try:
            tag_response = list_tags(ResourceArn=provider_id)
        except BaseException as exc:
            if _is_not_found_error(exc):
                return None
            if is_transient_describe_error(exc):
                raise TransientProviderError(
                    f"transient SageMaker tag lookup failed for {provider_id!r}",
                    job_name=provider_id.rsplit("/", 1)[-1],
                    operation="list_tags",
                ) from exc
            raise
        if isinstance(tag_response, Mapping):
            return self._fingerprint_from_tags(tag_response.get("Tags"))
        return None

    def _describe(
        self,
        operation: str,
        job_name: str,
        **kwargs: object,
    ) -> Mapping[str, Any] | None:
        try:
            response = getattr(self._client_or_create(), operation)(**kwargs)
        except BaseException as exc:
            if _is_not_found_error(exc):
                return None
            if is_transient_describe_error(exc):
                raise TransientProviderError(
                    f"transient SageMaker describe failed for {job_name!r}",
                    job_name=job_name,
                    operation=operation,
                ) from exc
            raise
        if not isinstance(response, Mapping):
            raise ProviderResponseError(f"SageMaker {operation} returned a non-mapping response")
        return response

    @staticmethod
    def _response_name(response: Mapping[str, Any], *, kind: str) -> str | None:
        key = "TrainingJobName" if kind == "training" else "ProcessingJobName"
        value = response.get(key)
        if value is None:
            return None
        return _validate_job_name(value, key)

    @staticmethod
    def _response_provider_id(
        response: Mapping[str, Any], *, kind: str, fallback: str | None
    ) -> str | None:
        key = "TrainingJobArn" if kind == "training" else "ProcessingJobArn"
        if key in response:
            return _validate_provider_id(response.get(key))
        if fallback is not None:
            return _validate_provider_id(fallback)
        return None

    def _result(
        self,
        response: Mapping[str, Any],
        *,
        job_name: str,
        kind: str,
        require_provider_id: bool,
        status_override: JobStatus | None = None,
    ) -> JobResult:
        response_name = self._response_name(response, kind=kind)
        if response_name is not None and response_name != job_name:
            raise ProviderResponseError(
                f"SageMaker response job name {response_name!r} does not match {job_name!r}"
            )
        provider_job_id = self._response_provider_id(
            response,
            kind=kind,
            fallback=None,
        )
        if require_provider_id and provider_job_id is None:
            key = "TrainingJobArn" if kind == "training" else "ProcessingJobArn"
            raise ProviderResponseError(f"SageMaker response omitted provider job ID ({key})")
        status_key = "TrainingJobStatus" if kind == "training" else "ProcessingJobStatus"
        status = status_override or _status(response.get(status_key))
        artifact_uri: str | None = None
        if kind == "training":
            model_artifacts = response.get("ModelArtifacts")
            if isinstance(model_artifacts, Mapping) and "S3ModelArtifacts" in model_artifacts:
                artifact_uri = _validate_artifact_uri(model_artifacts["S3ModelArtifacts"])
        else:
            outputs = response.get("ProcessingOutputConfig")
            if isinstance(outputs, Mapping):
                listed = outputs.get("Outputs", [])
                if isinstance(listed, list) and listed and isinstance(listed[0], Mapping):
                    s3_output = listed[0].get("S3Output")
                    if isinstance(s3_output, Mapping) and "S3Uri" in s3_output:
                        artifact_uri = _validate_artifact_uri(s3_output["S3Uri"])
        failure_reason = None
        if status in {JobStatus.FAILED, JobStatus.STOPPED}:
            failure_reason = _safe_failure_reason(response.get("FailureReason"))
        return JobResult(
            job_name=job_name,
            provider_job_id=provider_job_id,
            status=status,
            artifact_uri=artifact_uri,
            failure_reason=failure_reason,
            raw_response=_safe_raw_response(response, kind=kind, status=status),
        )

    def reconcile_training(self, request: TrainingJobRequest) -> JobResult | None:
        """Find a same-name training job only when its request digest matches."""

        self._validate_training(request)
        response = self._describe(
            "describe_training_job",
            request.job_name,
            TrainingJobName=request.job_name,
        )
        if response is None:
            return None
        provider_id = response.get("TrainingJobArn")
        if provider_id is None:
            raise JobUnidentifiableError(
                f"SageMaker training job {request.job_name!r} exists but cannot be reconciled"
            )
        provider_id = self._validate_reconcile_provider_id(provider_id)
        fingerprint = self._resource_tags(provider_id, response)
        if fingerprint is None:
            raise JobUnidentifiableError(
                f"SageMaker training job {request.job_name!r} has no request fingerprint"
            )
        if fingerprint != request.request_fingerprint:
            raise JobNameConflictError(
                f"SageMaker training job {request.job_name!r} fingerprint does not match request"
            )
        return self._result(
            response, job_name=request.job_name, kind="training", require_provider_id=False
        )

    def reconcile_evaluation(self, request: EvaluationJobRequest) -> JobResult | None:
        """Find a same-name processing job only when its request digest matches."""

        self._validate_evaluation(request)
        response = self._describe(
            "describe_processing_job",
            request.job_name,
            ProcessingJobName=request.job_name,
        )
        if response is None:
            return None
        provider_id = response.get("ProcessingJobArn")
        if provider_id is None:
            raise JobUnidentifiableError(
                f"SageMaker processing job {request.job_name!r} exists but cannot be reconciled"
            )
        provider_id = self._validate_reconcile_provider_id(provider_id)
        fingerprint = self._resource_tags(provider_id, response)
        if fingerprint is None:
            raise JobUnidentifiableError(
                f"SageMaker processing job {request.job_name!r} has no request fingerprint"
            )
        if fingerprint != request.request_fingerprint:
            raise JobNameConflictError(
                f"SageMaker processing job {request.job_name!r} fingerprint does not match request"
            )
        return self._result(
            response, job_name=request.job_name, kind="processing", require_provider_id=False
        )

    @staticmethod
    def _validate_reconcile_provider_id(value: object) -> str:
        return _validate_provider_id(value)

    def submit_training(self, request: TrainingJobRequest) -> JobResult:
        self._validate_training(request)
        reconciled = self.reconcile_training(request)
        if reconciled is not None:
            return reconciled
        tags = [
            tag
            for tag in request.tags
            if str(tag.get("Key", "")).lower().replace("_", "-") != "request-fingerprint"
        ]
        tags.append({"Key": "request-fingerprint", "Value": request.request_fingerprint})
        try:
            response = self._client_or_create().create_training_job(
                TrainingJobName=request.job_name,
                RoleArn=request.role_arn,
                AlgorithmSpecification={
                    "TrainingImage": request.image_uri,
                    "TrainingInputMode": "File",
                },
                InputDataConfig=[
                    {
                        "ChannelName": "train",
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": request.input_s3_uri,
                                "S3DataDistributionType": "FullyReplicated",
                            }
                        },
                    }
                ]
                + ([
                    {
                        "ChannelName": "parent_adapter",
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": request.parent_adapter_s3_uri,
                                "S3DataDistributionType": "FullyReplicated",
                            }
                        },
                    }
                ] if request.parent_adapter_s3_uri is not None else []),
                OutputDataConfig={"S3OutputPath": request.output_s3_uri},
                ResourceConfig={
                    "InstanceType": request.instance_type,
                    "InstanceCount": request.instance_count,
                    "VolumeSizeInGB": request.volume_size_gb,
                },
                StoppingCondition={"MaxRuntimeInSeconds": request.max_runtime_seconds},
                HyperParameters={str(k): str(v) for k, v in request.hyperparameters.items()},
                Environment=request.environment,
                Tags=tags,
            )
        except BaseException as exc:
            if not _is_already_exists_error(exc):
                raise
            reconciled = self.reconcile_training(request)
            if reconciled is None:
                raise ProviderReconciliationError(
                    f"SageMaker training job {request.job_name!r} exists but cannot be reconciled"
                ) from exc
            return reconciled
        if not isinstance(response, Mapping):
            raise ProviderResponseError(
                "SageMaker create_training_job returned a non-mapping response"
            )
        return self._result(
            response,
            job_name=request.job_name,
            kind="training",
            require_provider_id=True,
            status_override=JobStatus.SUBMITTED,
        )

    def get_training_status(self, job_name: str) -> JobResult:
        job_name = _validate_job_name(job_name)
        response = self._describe(
            "describe_training_job", job_name, TrainingJobName=job_name
        )
        if response is None:
            raise ProviderJobNotFoundError(f"SageMaker training job {job_name!r} was not found")
        return self._result(
            response, job_name=job_name, kind="training", require_provider_id=False
        )

    def stop_training(self, job_name: str) -> None:
        job_name = _validate_job_name(job_name)
        try:
            self._client_or_create().stop_training_job(TrainingJobName=job_name)
        except BaseException as exc:
            if _is_not_found_error(exc) or _is_already_terminal_stop_error(exc):
                return
            raise

    def submit_training_job(self, request: TrainingJobRequest) -> JobResult:
        """Compatibility alias with the AWS API's job-oriented naming."""
        return self.submit_training(request)

    def get_training_job(self, job_name: str) -> JobResult:
        return self.get_training_status(job_name)

    def reconcile_training_job(self, request: TrainingJobRequest) -> JobResult | None:
        """Compatibility alias for callers using job-oriented method names."""
        return self.reconcile_training(request)

    def submit_evaluation(self, request: EvaluationJobRequest) -> JobResult:
        self._validate_evaluation(request)
        reconciled = self.reconcile_evaluation(request)
        if reconciled is not None:
            return reconciled
        # Processing input names become SM_CHANNEL_* variables.  The
        # evaluator rejects every name except candidate/champion/sealed.
        candidate_uri = request.candidate_s3_uri or request.model_s3_uri or request.input_s3_uri
        sealed_uri = request.sealed_s3_uri or request.input_s3_uri
        input_specs: list[tuple[str, str, str]] = [
            ("candidate", candidate_uri, "/opt/ml/processing/input/candidate"),
            ("sealed", sealed_uri, "/opt/ml/processing/input/sealed"),
        ]
        if request.champion_s3_uri:
            input_specs.insert(
                1,
                ("champion", request.champion_s3_uri, "/opt/ml/processing/input/champion"),
            )
        processing_inputs = [
            {
                "InputName": input_name,
                "S3Input": {
                    # Keep a versioned checkpoint reference intact.  The
                    # evaluator entrypoint extracts the resulting tar.gz into
                    # this worker-required directory before parsing it.
                    "S3Uri": uri,
                    "LocalPath": local_path,
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3CompressionType": "None",
                },
            }
            for input_name, uri, local_path in input_specs
        ]
        app_spec: dict[str, Any] = {"ImageUri": request.image_uri}
        app_spec["ContainerEntrypoint"] = request.command or [
            "python",
            "-c",
            (
                "import pathlib,runpy,tarfile; "
                "[tarfile.open(str(a), 'r:*').extractall(str(root), filter='data') "
                "for root in (pathlib.Path('/opt/ml/processing/input/candidate'), "
                "pathlib.Path('/opt/ml/processing/input/champion')) if root.is_dir() "
                "for a in root.rglob('*.tar.gz')]; "
                "runpy.run_path('/opt/ml/code/evaluate.py', run_name='__main__')"
            ),
        ]
        tags = [
            tag
            for tag in request.tags
            if str(tag.get("Key", "")).lower().replace("_", "-") != "request-fingerprint"
        ]
        tags.append({"Key": "request-fingerprint", "Value": request.request_fingerprint})
        try:
            response = self._client_or_create().create_processing_job(
                ProcessingJobName=request.job_name,
                RoleArn=request.role_arn,
                AppSpecification=app_spec,
                ProcessingInputs=processing_inputs,
                ProcessingOutputConfig={
                    "Outputs": [
                        {
                            "OutputName": "evaluation",
                            "S3Output": {
                                "S3Uri": request.output_s3_uri,
                                "LocalPath": "/opt/ml/processing/output",
                                "S3UploadMode": "EndOfJob",
                            },
                        }
                    ]
                },
                ProcessingResources={
                    "ClusterConfig": {
                        "InstanceType": request.instance_type,
                        "InstanceCount": request.instance_count,
                        "VolumeSizeInGB": request.volume_size_gb,
                    }
                },
                StoppingCondition={"MaxRuntimeInSeconds": request.max_runtime_seconds},
                Environment=request.environment,
                Tags=tags,
            )
        except BaseException as exc:
            if not _is_already_exists_error(exc):
                raise
            reconciled = self.reconcile_evaluation(request)
            if reconciled is None:
                raise ProviderReconciliationError(
                    f"SageMaker processing job {request.job_name!r} exists but cannot be reconciled"
                ) from exc
            return reconciled
        if not isinstance(response, Mapping):
            raise ProviderResponseError(
                "SageMaker create_processing_job returned a non-mapping response"
            )
        return self._result(
            response,
            job_name=request.job_name,
            kind="processing",
            require_provider_id=True,
            status_override=JobStatus.SUBMITTED,
        )

    def get_evaluation_status(self, job_name: str) -> JobResult:
        job_name = _validate_job_name(job_name)
        response = self._describe(
            "describe_processing_job", job_name, ProcessingJobName=job_name
        )
        if response is None:
            raise ProviderJobNotFoundError(f"SageMaker processing job {job_name!r} was not found")
        return self._result(
            response, job_name=job_name, kind="processing", require_provider_id=False
        )

    def stop_evaluation(self, job_name: str) -> None:
        job_name = _validate_job_name(job_name)
        try:
            self._client_or_create().stop_processing_job(ProcessingJobName=job_name)
        except BaseException as exc:
            if _is_not_found_error(exc) or _is_already_terminal_stop_error(exc):
                return
            raise

    def submit_evaluation_job(self, request: EvaluationJobRequest) -> JobResult:
        return self.submit_evaluation(request)

    def get_evaluation_job(self, job_name: str) -> JobResult:
        return self.get_evaluation_status(job_name)

    def reconcile_evaluation_job(self, request: EvaluationJobRequest) -> JobResult | None:
        """Compatibility alias for callers using job-oriented method names."""
        return self.reconcile_evaluation(request)


SageMakerTrainingProvider = SageMakerProvider
SageMakerEvaluationProvider = SageMakerProvider
