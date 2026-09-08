"""Validated, immutable records for autonomous run persistence.

The records contain control-plane metadata only.  Prompts, hidden tasks, and
model output are deliberately represented by safe IDs or content hashes.
"""

from __future__ import annotations

import re
from copy import deepcopy
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from time import time_ns
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")
_UNSAFE_CONTENT = re.compile(
    r"(?:raw\s+prompt|raw\s+content|full\s+text|trajectory\b|hidden\b|secret\b|"
    r"task\s+content|private\b|answer\b|response\b|completion\b)",
    re.IGNORECASE,
)
_CONTROL_REASON_WORDS = frozenset(
    "approval packet stale worker queued baseline failure analysis research curation training "
    "checkpoint validation evaluation promotion promoted rejected succeeded failed blocked "
    "cancelled cancel safe stop lease claimed renewed released operation submitted provider "
    "experiment phase transition request completed available unavailable run job timed out "
    "started requested tick".split()
)
_OPAQUE_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/#@+\-]{0,511}$")


class FrozenDict(dict[str, Any]):
    """Dict-compatible mapping that rejects mutation after validation."""

    def _immutable(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("mapping is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable  # type: ignore[assignment]
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable  # type: ignore[assignment]

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenDict:
        clone = FrozenDict()
        memo[id(self)] = clone
        for key, value in self.items():
            dict.__setitem__(clone, deepcopy(key, memo), deepcopy(value, memo))
        return clone


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return FrozenDict({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_deep_freeze(item) for item in value)
    return value


_EVENT_METADATA_KEYS = frozenset(
    {
        "approval_digest",
        "artifact_id",
        "cost_usd",
        "evidence_label",
        "event_id",
        "experiment_id",
        "latency_ms",
        "operation_key",
        "phase",
        "provider_id",
        "reason_code",
        "run_id",
        "status",
    }
)


def _validate_event_metadata(value: dict[str, str]) -> FrozenDict:
    for key, item in value.items():
        if key not in _EVENT_METADATA_KEYS:
            raise ValueError(f"event metadata key {key!r} is not allow-listed")
        if (
            not item.strip()
            or len(item) > 512
            or (_UNSAFE_CONTENT.search(item) and key != "artifact_id")
            or not _OPAQUE_VALUE.fullmatch(item)
        ):
            raise ValueError("event metadata must contain only safe opaque references")
        if key in {"approval_digest"} and not re.fullmatch(r"[0-9a-f]{64}", item):
            raise ValueError(f"event metadata {key!r} must be a lowercase SHA-256 digest")
        if key == "status" and item not in {status.value for status in AutonomousRunStatus}:
            raise ValueError("event status metadata must use a known run status")
        if key == "phase" and item not in {phase.value for phase in RunPhase}:
            raise ValueError("event phase metadata must use a known run phase")
        if key == "evidence_label" and item not in {"LIVE", "PRIOR_VERIFIED_RUN", "EXPLANATION"}:
            raise ValueError("event evidence label metadata is invalid")
        if key in {"cost_usd", "latency_ms"}:
            try:
                numeric = float(item)
            except ValueError as exc:
                raise ValueError(f"event metadata {key!r} must be numeric") from exc
            if not isfinite(numeric):
                raise ValueError(f"event metadata {key!r} must be finite")
    return FrozenDict(value)


def utc_now() -> datetime:
    return datetime.now(UTC)


class RunPhase(StrEnum):
    PREPARED = "PREPARED"
    QUEUED = "QUEUED"
    BASELINE = "BASELINE"
    FAILURE_ANALYSIS = "FAILURE_ANALYSIS"
    ANALYSIS = "FAILURE_ANALYSIS"
    RESEARCH = "RESEARCH"
    CURATION = "CURATION"
    TRAINING = "TRAINING"
    CHECKPOINT_VALIDATION = "CHECKPOINT_VALIDATION"
    EVALUATION = "EVALUATION"
    PROMOTION = "PROMOTION"
    COMPLETED = "COMPLETED"
    STOPPED = "STOPPED"


class AutonomousRunStatus(StrEnum):
    PREPARED = "PREPARED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    SAFE_STOP_REQUESTED = "SAFE_STOP_REQUESTED"
    STOPPED = "STOPPED"


class ExperimentStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class RunOperationStatus(StrEnum):
    INTENT = "INTENT"
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class ExperimentRecord(ContractModel):
    """One bounded candidate experiment and its verified references."""

    experiment_number: int = Field(ge=1, le=5)
    status: ExperimentStatus = ExperimentStatus.PENDING
    hypothesis_id: str | None = Field(default=None, min_length=1)
    dataset_id: str | None = Field(default=None, min_length=1)
    training_config: dict[str, Any] = Field(default_factory=dict)
    provider_job_ids: tuple[str, ...] = Field(default_factory=tuple)
    artifact_ids: tuple[str, ...] = Field(default_factory=tuple)
    evidence_ids: tuple[str, ...] = Field(default_factory=tuple)
    metrics: dict[str, float] = Field(default_factory=dict)
    stop_reason: str | None = Field(default=None, min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("created_at", "updated_at")
    @classmethod
    def require_aware_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must include a timezone")
        return value

    @field_validator("provider_job_ids", "artifact_ids", "evidence_ids")
    @classmethod
    def reject_blank_references(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("references cannot be blank")
        return value

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: dict[str, float]) -> dict[str, float]:
        for key, metric in value.items():
            if not key.strip() or not isinstance(metric, (int, float)) or not isfinite(metric):
                raise ValueError("metrics require non-empty names and finite numeric values")
        return value

    @field_validator("training_config", "metrics", mode="after")
    @classmethod
    def freeze_mappings(cls, value: dict[str, Any]) -> FrozenDict:
        return cast(FrozenDict, _deep_freeze(value))

    @model_validator(mode="after")
    def freeze_nested_values(self) -> ExperimentRecord:
        object.__setattr__(self, "training_config", _deep_freeze(self.training_config))
        object.__setattr__(self, "metrics", _deep_freeze(self.metrics))
        return self


class RunOperation(ContractModel):
    """Persisted intent/result used to reconcile an at-least-once provider call."""

    operation_key: str = Field(min_length=1, max_length=512)
    run_id: str = Field(min_length=1)
    experiment_number: int = Field(ge=1, le=5)
    phase: RunPhase
    provider_name: str = Field(min_length=1, max_length=512)
    provider_id: str | None = Field(default=None, max_length=2_000)
    status: RunOperationStatus = RunOperationStatus.INTENT
    request_digest: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    version: int = Field(default=0, ge=0)

    @field_validator("request_digest")
    @classmethod
    def validate_request_digest(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256.fullmatch(value):
            raise ValueError("request_digest must be a lowercase SHA-256 digest")
        return value

    @field_validator("created_at", "updated_at")
    @classmethod
    def require_aware_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must include a timezone")
        return value

    @field_validator("result", mode="after")
    @classmethod
    def freeze_result(cls, value: dict[str, Any]) -> FrozenDict:
        return cast(FrozenDict, _deep_freeze(value))

    @model_validator(mode="after")
    def freeze_nested_values(self) -> RunOperation:
        object.__setattr__(self, "result", _deep_freeze(self.result))
        return self


class RunEventRecord(ContractModel):
    """Append-only metadata event.  Sequence numbers are repository-owned."""

    run_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    event_id: str = Field(default_factory=lambda: f"evt-{time_ns()}", min_length=1)
    event_type: str = Field(min_length=1, max_length=200)
    from_status: AutonomousRunStatus | None = None
    to_status: AutonomousRunStatus
    from_phase: RunPhase | None = None
    to_phase: RunPhase
    reason: str = Field(min_length=1, max_length=2_000)
    metadata: dict[str, str] = Field(default_factory=dict)
    occurred_at: datetime = Field(default_factory=utc_now)

    @field_validator("occurred_at")
    @classmethod
    def require_aware_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone")
        return value

    @field_validator("reason")
    @classmethod
    def reject_unsafe_reason(cls, value: str) -> str:
        words = re.findall(r"[a-z0-9]+", value.lower())
        if _UNSAFE_CONTENT.search(value) or not any(
            word in _CONTROL_REASON_WORDS for word in words
        ):
            raise ValueError("event reason may not contain raw or sealed content")
        return value

    @field_validator("metadata", mode="after")
    @classmethod
    def validate_metadata(cls, value: dict[str, str]) -> FrozenDict:
        return _validate_event_metadata(value)


class AutonomousRunState(ContractModel):
    """Durable optimistic-concurrency state for one top-level live run."""

    run_id: str = Field(min_length=1, max_length=200)
    model_id: str = Field(default="google/functiongemma-270m-it", min_length=1)
    checkpoint_revision: str
    checkpoint_id: str | None = Field(default=None, min_length=1)
    benchmark_id: str = Field(default="service-recovery-v1", min_length=1)
    benchmark_manifest_sha256: str
    max_experiments: int = Field(default=5, ge=1, le=5)
    approved_budget_usd: float = Field(default=25.0, gt=0, le=25)
    status: AutonomousRunStatus = AutonomousRunStatus.PREPARED
    phase: RunPhase = RunPhase.PREPARED
    version: int = Field(default=0, ge=0)
    event_sequence: int = Field(default=0, ge=0)
    approval_digest: str | None = None
    approval_consumed: bool = False
    approval_consumed_at: datetime | None = None
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    cancellation_requested: bool = False
    safe_stop_requested: bool = False
    spent_budget_usd: float = Field(default=0.0, ge=0)
    baseline_metrics: dict[str, float] = Field(default_factory=dict)
    champion_metrics: dict[str, float] = Field(default_factory=dict)
    baseline_artifact_ids: tuple[str, ...] = Field(default_factory=tuple)
    champion_artifact_ids: tuple[str, ...] = Field(default_factory=tuple)
    current_experiment_number: int | None = Field(default=None, ge=1, le=5)
    experiments: list[ExperimentRecord] = Field(default_factory=list)
    stop_reason: str | None = Field(default=None, min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("checkpoint_revision")
    @classmethod
    def require_immutable_revision(cls, value: str) -> str:
        if not _REVISION.fullmatch(value):
            raise ValueError("checkpoint_revision must be a 40-character immutable revision")
        return value.lower()

    @field_validator("benchmark_manifest_sha256", "approval_digest")
    @classmethod
    def require_sha256(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256.fullmatch(value):
            raise ValueError("digest must be a lowercase 64-character SHA-256 value")
        return value

    @field_validator("created_at", "updated_at", "approval_consumed_at", "lease_expires_at")
    @classmethod
    def require_aware_datetime(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("timestamps must include a timezone")
        return value

    @field_validator("baseline_metrics", "champion_metrics")
    @classmethod
    def validate_metrics(cls, value: dict[str, float]) -> dict[str, float]:
        for key, metric in value.items():
            if not key.strip() or not isinstance(metric, (int, float)) or not isfinite(metric):
                raise ValueError("metrics require non-empty names and finite numeric values")
        return value

    @field_validator("baseline_artifact_ids", "champion_artifact_ids")
    @classmethod
    def reject_blank_artifacts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("artifact references cannot be blank")
        return value

    @field_validator("experiments", mode="after")
    @classmethod
    def freeze_experiments(cls, value: list[ExperimentRecord]) -> tuple[ExperimentRecord, ...]:
        return tuple(value)

    @field_validator("metadata", "baseline_metrics", "champion_metrics", mode="after")
    @classmethod
    def freeze_state_mappings(cls, value: dict[str, Any]) -> FrozenDict:
        return cast(FrozenDict, _deep_freeze(value))

    @model_validator(mode="after")
    def validate_budget_and_scope(self) -> AutonomousRunState:
        if self.spent_budget_usd > self.approved_budget_usd:
            raise ValueError("spent budget cannot exceed approved budget")
        if len(self.experiments) > self.max_experiments:
            raise ValueError("experiment history exceeds approved maximum")
        if self.approval_consumed and not self.approval_digest:
            raise ValueError("consumed approval requires approval_digest")
        object.__setattr__(self, "experiments", tuple(self.experiments))
        object.__setattr__(self, "metadata", _deep_freeze(self.metadata))
        object.__setattr__(self, "baseline_metrics", _deep_freeze(self.baseline_metrics))
        object.__setattr__(self, "champion_metrics", _deep_freeze(self.champion_metrics))
        return self

    @property
    def state_version(self) -> int:
        """Compatibility spelling used by older coordinator adapters."""

        return self.version

    @property
    def approval_packet_sha256(self) -> str | None:
        return self.approval_digest


def copy_for_storage[ModelT: ContractModel](model: ModelT) -> ModelT:
    """Make a deep defensive snapshot without exposing repository internals."""

    return model.model_copy(deep=True)


__all__ = [
    "AutonomousRunState",
    "AutonomousRunStatus",
    "ExperimentRecord",
    "ExperimentStatus",
    "RunEventRecord",
    "RunOperation",
    "RunOperationStatus",
    "RunPhase",
    "copy_for_storage",
    "utc_now",
]
