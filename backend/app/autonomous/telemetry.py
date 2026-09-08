"""Durable, metadata-only telemetry for autonomous runs.

The repository is the source of truth for lifecycle events.  The existing
``TelemetryRecorder`` is an optional observer: it receives the same event
after persistence and an exporter/logger failure is intentionally isolated.
This module owns the small, strict metadata boundary between those systems.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Final, Protocol

from app.observability import EventType as ObserverEventType
from app.observability import TelemetryRecorder

from .models import (
    AutonomousRunState,
    AutonomousRunStatus,
    RunEventRecord,
    RunPhase,
)


class AutonomousEventType(StrEnum):
    """Allow-listed lifecycle vocabulary for autonomous control-plane events."""

    RUN_STARTED = "run.started"
    RUN_QUEUED = "run.queued"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCEL_REQUESTED = "run.cancel_requested"
    RUN_CANCELLED = "run.cancelled"
    RUN_SAFE_STOP_REQUESTED = "run.safe_stop_requested"
    RUN_STOPPED = "run.stopped"
    APPROVAL_CONSUMED = "approval.consumed"
    PHASE_STARTED = "phase.started"
    PHASE_COMPLETED = "phase.completed"
    PHASE_FAILED = "phase.failed"
    JOB_SUBMITTED = "job.submitted"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    OPERATION_INTENT = "operation.intent"
    OPERATION_SUBMITTED = "operation.submitted"
    OPERATION_COMPLETED = "operation.completed"
    OPERATION_FAILED = "operation.failed"
    PROMOTION_DECIDED = "promotion.decided"
    ARTIFACT_RECORDED = "artifact.recorded"
    CLEANUP_COMPLETED = "cleanup.completed"
    CLEANUP_FAILED = "cleanup.failed"


# Compatibility names keep the bridge easy to discover for supervisor callers.
TelemetryEventType = AutonomousEventType
LifecycleEventType = AutonomousEventType
EventType = AutonomousEventType


class DurableTelemetryError(RuntimeError):
    """A lifecycle event could not be written to the durable repository."""


TelemetryPersistenceError = DurableTelemetryError


class DurableEventRepository(Protocol):
    """Repository subset needed by :class:`DurableTelemetryBridge`."""

    def append_event(
        self,
        run_id: str,
        *,
        event_type: str,
        reason: str,
        metadata: Mapping[str, str] | None = None,
    ) -> RunEventRecord: ...

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


class OptionalTelemetryRecorder(Protocol):
    """Minimal recorder protocol accepted by the bridge and tests."""

    def record(self, event_type: Any, **kwargs: Any) -> Any: ...


SafeMetadataValue = str | int | float
SafeTelemetryMetadata = dict[str, SafeMetadataValue]

SAFE_METADATA_KEYS: Final[frozenset[str]] = frozenset(
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

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/#@+\-]{0,511}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SENSITIVE = re.compile(
    r"(?:prompt|completion|trajectory|hidden|sealed|secret|credential|password|token|"
    r"authorization|api[_-]?key|access[_-]?key|private[_-]?key|raw[_-]?output|task[_-]?content)",
    re.IGNORECASE,
)
_ALLOWED_LABELS = frozenset({"LIVE", "PRIOR_VERIFIED_RUN", "EXPLANATION"})


def _safe_token(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty safe identifier")
    normalized = value.strip()
    if _SENSITIVE.search(normalized) or not _SAFE_TOKEN.fullmatch(normalized):
        raise ValueError(f"{name} must be an opaque allow-listed identifier")
    return normalized


def _safe_measurement(name: str, value: int | float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and non-negative")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


def validate_safe_metadata(
    metadata: Mapping[str, SafeMetadataValue] | None,
) -> SafeTelemetryMetadata:
    """Validate and normalize the durable event metadata allow-list.

    Unknown keys and free-form strings are rejected rather than redacted.  A
    rejected value must never reach either persistence or an optional sink.
    """

    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    normalized: SafeTelemetryMetadata = {}
    for raw_key, raw_value in metadata.items():
        if not isinstance(raw_key, str):
            raise ValueError("metadata keys must be strings")
        key = raw_key.strip().lower()
        if key not in SAFE_METADATA_KEYS:
            raise ValueError(f"metadata key {raw_key!r} is not allow-listed")
        if key in {"latency_ms", "cost_usd"}:
            if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
                raise ValueError(f"{key} must be finite and non-negative")
            normalized[key] = _safe_measurement(key, raw_value)
            continue
        if not isinstance(raw_value, str):
            raise ValueError(f"metadata value for {key} must be a safe string")
        value = _safe_token(key, raw_value)
        if key == "approval_digest" and not _SHA256.fullmatch(value):
            raise ValueError("approval_digest must be a lowercase SHA-256 digest")
        if key in {"status", "phase"}:
            value = value.upper()
            allowed = (
                {item.value for item in AutonomousRunStatus}
                if key == "status"
                else {item.value for item in RunPhase}
            )
            if value not in allowed:
                raise ValueError(f"event {key} metadata must use a known autonomous value")
        if key == "evidence_label" and value not in _ALLOWED_LABELS:
            raise ValueError("evidence_label must be LIVE, PRIOR_VERIFIED_RUN, or EXPLANATION")
        normalized[key] = value
    return normalized


def _durable_metadata(
    *,
    metadata: Mapping[str, SafeMetadataValue] | None,
    run_id: str,
    experiment_id: str,
    phase: str | None,
    status: str | None,
    job_id: str | None,
    evidence_label: str | None,
    latency_ms: int | float | None,
    cost_usd: int | float | None,
    operation_key: str | None,
) -> tuple[dict[str, str], SafeTelemetryMetadata]:
    """Merge explicit event correlation fields with caller metadata."""

    values = validate_safe_metadata(metadata)
    fields: dict[str, SafeMetadataValue | None] = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "phase": phase,
        "status": status,
        "provider_id": job_id,
        "evidence_label": evidence_label,
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
        "operation_key": operation_key,
    }
    for key, value in fields.items():
        if value is None:
            continue
        candidate: SafeMetadataValue
        if key in {"latency_ms", "cost_usd"}:
            candidate = _safe_measurement(key, value)  # type: ignore[arg-type]
        else:
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a safe string")
            candidate = _safe_token(key, value)
            if key in {"status", "phase"}:
                candidate = candidate.upper()
                allowed = (
                    {item.value for item in AutonomousRunStatus}
                    if key == "status"
                    else {item.value for item in RunPhase}
                )
                if candidate not in allowed:
                    raise ValueError(f"event {key} metadata must use a known autonomous value")
            if key == "evidence_label" and candidate not in _ALLOWED_LABELS:
                raise ValueError(
                    "evidence_label must be LIVE, PRIOR_VERIFIED_RUN, or EXPLANATION"
                )
        existing = values.get(key)
        if existing is not None and existing != candidate:
            raise ValueError(f"metadata {key} conflicts with the event field")
        values[key] = candidate

    # RunEventRecord stores metadata as strings; the observer receives the
    # typed numeric measurements so latency/cost retain their useful type.
    durable: dict[str, str] = {
        key: (str(value) if isinstance(value, (int, float)) else value)
        for key, value in values.items()
    }
    observer: SafeTelemetryMetadata = dict(values)
    return durable, observer


class DurableTelemetryBridge:
    """Write durable events first, then mirror them to optional telemetry."""

    def __init__(
        self,
        repository: DurableEventRepository,
        *,
        recorder: OptionalTelemetryRecorder | TelemetryRecorder | None = None,
    ) -> None:
        self.repository = repository
        self.recorder = recorder

    def emit(
        self,
        event_type: AutonomousEventType | str,
        *,
        run_id: str,
        run_number: int,
        experiment_id: str,
        reason: str,
        phase: str | None = None,
        status: str | None = None,
        job_id: str | None = None,
        evidence_label: str | None = None,
        latency_ms: int | float | None = None,
        cost_usd: int | float | None = None,
        operation_key: str | None = None,
        metadata: Mapping[str, SafeMetadataValue] | None = None,
    ) -> RunEventRecord:
        """Append one durable event and best-effort mirror it to the recorder."""

        normalized_type = self._event_type(event_type)
        self._validate_common(run_id, run_number, experiment_id, reason)
        durable_metadata, observer_metadata = _durable_metadata(
            metadata=metadata,
            run_id=run_id,
            experiment_id=experiment_id,
            phase=phase,
            status=status,
            job_id=job_id,
            evidence_label=evidence_label,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            operation_key=operation_key,
        )
        try:
            persisted = self.repository.append_event(
                run_id,
                event_type=normalized_type.value,
                reason=reason,
                metadata=durable_metadata,
            )
        except Exception as exc:
            raise DurableTelemetryError(
                f"durable telemetry event {normalized_type.value!r} could not be persisted"
            ) from exc
        self._mirror(
            normalized_type,
            run_id=run_id,
            run_number=run_number,
            experiment_id=experiment_id,
            phase=phase,
            job_id=job_id,
            evidence_label=evidence_label,
            status=status,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            attributes=observer_metadata,
        )
        return persisted

    record = emit
    append_event = emit
    emit_event = emit
    record_event = emit
    persist_event = emit

    def transition(
        self,
        event_type: AutonomousEventType | str,
        *,
        run_id: str,
        expected_version: int,
        status: AutonomousRunStatus,
        phase: RunPhase,
        reason: str,
        run_number: int,
        experiment_id: str,
        status_metadata: str | None = None,
        telemetry_status: str | None = None,
        job_id: str | None = None,
        evidence_label: str | None = None,
        latency_ms: int | float | None = None,
        cost_usd: int | float | None = None,
        operation_key: str | None = None,
        metadata: Mapping[str, SafeMetadataValue] | None = None,
    ) -> AutonomousRunState:
        """Atomically persist a state transition and then mirror telemetry."""

        normalized_type = self._event_type(event_type)
        self._validate_common(run_id, run_number, experiment_id, reason)
        if (
            not isinstance(expected_version, int)
            or isinstance(expected_version, bool)
            or expected_version < 0
        ):
            raise ValueError("expected_version must be a non-negative integer")
        if not isinstance(status, AutonomousRunStatus):
            raise ValueError("status must be an AutonomousRunStatus")
        if not isinstance(phase, RunPhase):
            raise ValueError("phase must be a RunPhase")
        durable_metadata, observer_metadata = _durable_metadata(
            metadata=metadata,
            run_id=run_id,
            experiment_id=experiment_id,
            phase=phase.value,
            status=status_metadata or telemetry_status or status.value,
            job_id=job_id,
            evidence_label=evidence_label,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            operation_key=operation_key,
        )
        try:
            transitioned = self.repository.transition(
                run_id,
                expected_version=expected_version,
                status=status,
                phase=phase,
                reason=reason,
                event_type=normalized_type.value,
                metadata=durable_metadata,
            )
        except Exception as exc:
            raise DurableTelemetryError(
                f"durable telemetry transition {normalized_type.value!r} could not be persisted"
            ) from exc
        self._mirror(
            normalized_type,
            run_id=run_id,
            run_number=run_number,
            experiment_id=experiment_id,
            phase=phase.value,
            job_id=job_id,
            evidence_label=evidence_label,
            status=status_metadata or telemetry_status or status.value,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            attributes=observer_metadata,
        )
        return transitioned

    record_transition = transition
    transition_run = transition
    persist_transition = transition

    @staticmethod
    def _event_type(event_type: AutonomousEventType | str) -> AutonomousEventType:
        try:
            return (
                event_type
                if isinstance(event_type, AutonomousEventType)
                else AutonomousEventType(event_type)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown autonomous event type: {event_type!r}") from exc

    @staticmethod
    def _validate_common(run_id: str, run_number: int, experiment_id: str, reason: str) -> None:
        _safe_token("run_id", run_id)
        if (
            not isinstance(run_number, int)
            or isinstance(run_number, bool)
            or not 1 <= run_number <= 5
        ):
            raise ValueError("run_number must be between 1 and 5")
        _safe_token("experiment_id", experiment_id)
        if not isinstance(reason, str) or not reason.strip() or _SENSITIVE.search(reason):
            raise ValueError("reason must be a non-empty safe lifecycle reason")
        if len(reason) > 2_000:
            raise ValueError("reason is too long")

    def _mirror(
        self,
        event_type: AutonomousEventType,
        *,
        run_id: str,
        run_number: int,
        experiment_id: str,
        phase: str | None,
        job_id: str | None,
        evidence_label: str | None,
        status: str | None,
        latency_ms: int | float | None,
        cost_usd: int | float | None,
        attributes: Mapping[str, SafeMetadataValue],
    ) -> None:
        if self.recorder is None:
            return
        observer_type: ObserverEventType | str
        try:
            observer_type = ObserverEventType(event_type.value)
        except ValueError:
            observer_type = event_type.value
        try:
            self.recorder.record(
                observer_type,
                run_id=run_id,
                run_number=run_number,
                experiment_id=experiment_id,
                phase=phase,
                job_id=job_id,
                evidence_label=evidence_label,
                status=status,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                attributes=dict(attributes),
            )
        except Exception:
            # Durable state already succeeded.  The optional observer cannot
            # retroactively make the run fail or force a retry/duplicate event.
            return


AutonomousTelemetry = DurableTelemetryBridge
TelemetryBridge = DurableTelemetryBridge


__all__ = [
    "SAFE_METADATA_KEYS",
    "AutonomousEventType",
    "AutonomousTelemetry",
    "DurableEventRepository",
    "DurableTelemetryBridge",
    "DurableTelemetryError",
    "EventType",
    "LifecycleEventType",
    "SafeTelemetryMetadata",
    "TelemetryBridge",
    "TelemetryEventType",
    "TelemetryPersistenceError",
    "validate_safe_metadata",
]
