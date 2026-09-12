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
    validate_event_reason,
)


class AutonomousEventType(StrEnum):
    """Allow-listed lifecycle vocabulary for autonomous control-plane events."""

    RUN_STARTED = "run.started"
    RUN_QUEUED = "run.queued"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_BLOCKED = "run.blocked"
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


CANONICAL_EVENT_TYPES: Final[dict[AutonomousEventType, ObserverEventType]] = {
    AutonomousEventType.RUN_STARTED: ObserverEventType.RUN_STARTED,
    AutonomousEventType.RUN_QUEUED: ObserverEventType.RUN_STARTED,
    AutonomousEventType.RUN_COMPLETED: ObserverEventType.RUN_COMPLETED,
    AutonomousEventType.RUN_FAILED: ObserverEventType.RUN_FAILED,
    AutonomousEventType.RUN_BLOCKED: ObserverEventType.RUN_FAILED,
    AutonomousEventType.RUN_CANCEL_REQUESTED: ObserverEventType.RUN_FAILED,
    AutonomousEventType.RUN_CANCELLED: ObserverEventType.RUN_COMPLETED,
    AutonomousEventType.RUN_SAFE_STOP_REQUESTED: ObserverEventType.RUN_FAILED,
    AutonomousEventType.RUN_STOPPED: ObserverEventType.RUN_COMPLETED,
    AutonomousEventType.APPROVAL_CONSUMED: ObserverEventType.RUN_STARTED,
    AutonomousEventType.PHASE_STARTED: ObserverEventType.PHASE_STARTED,
    AutonomousEventType.PHASE_COMPLETED: ObserverEventType.PHASE_COMPLETED,
    AutonomousEventType.PHASE_FAILED: ObserverEventType.PHASE_FAILED,
    AutonomousEventType.JOB_SUBMITTED: ObserverEventType.JOB_SUBMITTED,
    AutonomousEventType.JOB_COMPLETED: ObserverEventType.JOB_COMPLETED,
    AutonomousEventType.JOB_FAILED: ObserverEventType.JOB_FAILED,
    AutonomousEventType.OPERATION_INTENT: ObserverEventType.JOB_SUBMITTED,
    AutonomousEventType.OPERATION_SUBMITTED: ObserverEventType.JOB_SUBMITTED,
    AutonomousEventType.OPERATION_COMPLETED: ObserverEventType.JOB_COMPLETED,
    AutonomousEventType.OPERATION_FAILED: ObserverEventType.JOB_FAILED,
    AutonomousEventType.PROMOTION_DECIDED: ObserverEventType.PROMOTION_DECIDED,
    AutonomousEventType.ARTIFACT_RECORDED: ObserverEventType.PHASE_COMPLETED,
    AutonomousEventType.CLEANUP_COMPLETED: ObserverEventType.CLEANUP_COMPLETED,
    AutonomousEventType.CLEANUP_FAILED: ObserverEventType.CLEANUP_FAILED,
}


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
        "durable_event_type",
        "evidence_label",
        "event_id",
        "experiment_id",
        "latency_ms",
        "operation_key",
        "phase",
        "provider_id",
        "reason_code",
        "run_id",
        "run_number",
        "status",
    }
)

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/#@+\-]{0,511}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SENSITIVE = re.compile(
    r"(?:prompt|completion|trajectory|hidden|sealed|secret|credential|password|token|"
    r"authorization|api[_-]?key|access[_-]?key|private[_-]?key|raw[_-]?output|"
    r"raw[_-]?content|task[_-]?content|answer|response|private)",
    re.IGNORECASE,
)
_ALLOWED_LABELS = frozenset({"LIVE", "PRIOR_VERIFIED_RUN", "EXPLANATION"})
_REQUIRED_CORRELATION_FIELDS: Final[dict[AutonomousEventType, frozenset[str]]] = {
    event: frozenset({"phase"})
    for event in (
        AutonomousEventType.PHASE_STARTED,
        AutonomousEventType.PHASE_COMPLETED,
        AutonomousEventType.PHASE_FAILED,
    )
}
_REQUIRED_CORRELATION_FIELDS.update(
    {
        event: frozenset({"job_id"})
        for event in (
            AutonomousEventType.JOB_SUBMITTED,
            AutonomousEventType.JOB_COMPLETED,
            AutonomousEventType.JOB_FAILED,
        )
    }
)
_REQUIRED_CORRELATION_FIELDS.update(
    {
        event: frozenset({"operation_key"})
        for event in (
            AutonomousEventType.OPERATION_INTENT,
            AutonomousEventType.OPERATION_SUBMITTED,
            AutonomousEventType.OPERATION_COMPLETED,
            AutonomousEventType.OPERATION_FAILED,
        )
    }
)
_REQUIRED_CORRELATION_FIELDS.update(
    {
        AutonomousEventType.PROMOTION_DECIDED: frozenset({"evidence_label"}),
        AutonomousEventType.ARTIFACT_RECORDED: frozenset({"artifact_id"}),
        AutonomousEventType.APPROVAL_CONSUMED: frozenset({"approval_digest"}),
    }
)
for _terminal_event in (
    AutonomousEventType.RUN_QUEUED,
    AutonomousEventType.RUN_COMPLETED,
    AutonomousEventType.RUN_FAILED,
    AutonomousEventType.RUN_BLOCKED,
    AutonomousEventType.RUN_CANCEL_REQUESTED,
    AutonomousEventType.RUN_CANCELLED,
    AutonomousEventType.RUN_SAFE_STOP_REQUESTED,
    AutonomousEventType.RUN_STOPPED,
):
    _REQUIRED_CORRELATION_FIELDS[_terminal_event] = frozenset({"status"})


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
        if key == "run_number":
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, str)):
                raise ValueError("run_number must be between 1 and 5")
            number = int(raw_value) if isinstance(raw_value, int) else int(raw_value.strip())
            if str(number) != str(raw_value).strip() or not 1 <= number <= 5:
                raise ValueError("run_number must be between 1 and 5")
            normalized[key] = number
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
        if key == "durable_event_type" and value not in {
            event.value for event in AutonomousEventType
        }:
            raise ValueError("durable_event_type must use the autonomous event vocabulary")
        normalized[key] = value
    return normalized


def _durable_metadata(
    *,
    metadata: Mapping[str, SafeMetadataValue] | None,
    run_id: str,
    experiment_id: str,
    run_number: int,
    phase: str | None,
    status: str | None,
    job_id: str | None,
    evidence_label: str | None,
    approval_digest: str | None,
    artifact_id: str | None,
    reason_code: str | None,
    latency_ms: int | float | None,
    cost_usd: int | float | None,
    operation_key: str | None,
) -> tuple[dict[str, str], SafeTelemetryMetadata]:
    """Merge explicit event correlation fields with caller metadata."""

    values = validate_safe_metadata(metadata)
    fields: dict[str, SafeMetadataValue | None] = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "run_number": run_number,
        "phase": phase,
        "status": status,
        "provider_id": job_id,
        "evidence_label": evidence_label,
        "approval_digest": approval_digest,
        "artifact_id": artifact_id,
        "reason_code": reason_code,
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
        elif key == "run_number":
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
                raise ValueError("run_number must be between 1 and 5")
            candidate = value
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
        approval_digest: str | None = None,
        artifact_id: str | None = None,
        reason_code: str | None = None,
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
            run_number=run_number,
            phase=phase,
            status=status,
            job_id=job_id,
            evidence_label=evidence_label,
            approval_digest=approval_digest,
            artifact_id=artifact_id,
            reason_code=reason_code,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            operation_key=operation_key,
        )
        self._require_correlation(normalized_type, durable_metadata)
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
            event_id=persisted.event_id,
        )
        return persisted

    def record(self, *args: Any, **kwargs: Any) -> RunEventRecord:
        """Explicit alias for :meth:`emit` with the same argument order."""

        return self.emit(*args, **kwargs)

    def emit_event(self, *args: Any, **kwargs: Any) -> RunEventRecord:
        """Explicit alias for :meth:`emit` with the same argument order."""

        return self.emit(*args, **kwargs)

    def record_event(self, *args: Any, **kwargs: Any) -> RunEventRecord:
        """Explicit alias for :meth:`emit` with the same argument order."""

        return self.emit(*args, **kwargs)

    def append_event(
        self,
        run_id: str,
        *,
        event_type: AutonomousEventType | str,
        run_number: int,
        experiment_id: str,
        reason: str,
        **kwargs: Any,
    ) -> RunEventRecord:
        """Repository-shaped wrapper with explicit telemetry correlation."""

        return self.emit(
            event_type,
            run_id=run_id,
            run_number=run_number,
            experiment_id=experiment_id,
            reason=reason,
            **kwargs,
        )

    def persist_event(self, *args: Any, **kwargs: Any) -> RunEventRecord:
        """Explicit alias for :meth:`emit` with the same argument order."""

        return self.emit(*args, **kwargs)

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
        approval_digest: str | None = None,
        artifact_id: str | None = None,
        reason_code: str | None = None,
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
            run_number=run_number,
            phase=phase.value,
            status=status_metadata or telemetry_status or status.value,
            job_id=job_id,
            evidence_label=evidence_label,
            approval_digest=approval_digest,
            artifact_id=artifact_id,
            reason_code=reason_code,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            operation_key=operation_key,
        )
        self._require_correlation(normalized_type, durable_metadata)
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
            event_id=self._transition_event_id(
                run_id,
                event_sequence=getattr(transitioned, "event_sequence", expected_version + 1),
            ),
        )
        return transitioned

    def record_transition(self, run_id: str, **kwargs: Any) -> AutonomousRunState:
        """Repository-shaped transition wrapper with run ID first."""

        return self.transition(run_id=run_id, **kwargs)

    def transition_run(self, run_id: str, **kwargs: Any) -> AutonomousRunState:
        """Repository-shaped transition wrapper with run ID first."""

        return self.transition(run_id=run_id, **kwargs)

    def persist_transition(self, run_id: str, **kwargs: Any) -> AutonomousRunState:
        """Repository-shaped transition wrapper with run ID first."""

        return self.transition(run_id=run_id, **kwargs)

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
        validate_event_reason(reason)

    @staticmethod
    def _require_correlation(event_type: AutonomousEventType, metadata: Mapping[str, str]) -> None:
        aliases = {
            "job_id": "provider_id",
            "artifact_id": "artifact_id",
            "approval_digest": "approval_digest",
            "evidence_label": "evidence_label",
            "operation_key": "operation_key",
            "phase": "phase",
            "status": "status",
        }
        missing = [
            field
            for field in _REQUIRED_CORRELATION_FIELDS.get(event_type, frozenset())
            if aliases[field] not in metadata
        ]
        if missing:
            raise ValueError(
                f"event {event_type.value} requires correlation field(s): "
                f"{', '.join(sorted(missing))}"
            )

    def _transition_event_id(self, run_id: str, *, event_sequence: object) -> str | None:
        """Resolve the repository-generated event ID without inventing one."""

        list_events = getattr(self.repository, "list_" + "events", None)
        if isinstance(event_sequence, int) and callable(list_events):
            try:
                page = list_events(
                    run_id, after_sequence=max(0, event_sequence - 1), limit=1
                )
                items = getattr(page, "items", page)
                if items:
                    event = items[0]
                    event_id = getattr(event, "event_id", None)
                    if isinstance(event_id, str) and event_id:
                        return event_id
            except Exception:
                pass
        # Injected repositories may expose only transition() in tests.  Do not
        # fabricate an ID that could be mistaken for a durable event identity.
        return None

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
        event_id: str | None,
    ) -> None:
        if self.recorder is None:
            return
        observer_type: ObserverEventType | str
        observer_type = CANONICAL_EVENT_TYPES[event_type]
        mirrored_attributes: SafeTelemetryMetadata = dict(attributes)
        mirrored_attributes["durable_event_type"] = event_type.value
        if event_id is not None:
            mirrored_attributes["event_id"] = event_id
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
                attributes=mirrored_attributes,
            )
        except Exception:
            # Durable state already succeeded.  The optional observer cannot
            # retroactively make the run fail or force a retry/duplicate event.
            return


AutonomousTelemetry = DurableTelemetryBridge
TelemetryBridge = DurableTelemetryBridge


__all__ = [
    "CANONICAL_EVENT_TYPES",
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
