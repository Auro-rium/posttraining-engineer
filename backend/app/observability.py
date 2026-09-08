"""Metadata-only telemetry for autonomous post-training runs.

This module is deliberately independent of AWS clients and application state.
Callers inject an exporter and logger, so local runs can collect events without
AWS credentials while a deployed coordinator can forward the same contract to
CloudWatch, OTLP, or a durable event store.  The event payload is an explicit
allow-list of operational metadata; arbitrary model content is never accepted
as a top-level field and sensitive metadata is redacted recursively.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol
from uuid import uuid4

from app.posttraining.models import EvidenceLabel

_otel_trace: Any = None
try:  # OpenTelemetry is optional at runtime despite being a supported extra.
    from opentelemetry import trace as _otel_trace
except ImportError:  # pragma: no cover - exercised in environments without the extra
    _otel_trace = None


class EventType(StrEnum):
    """Lifecycle events emitted by the run control plane."""

    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    PHASE_STARTED = "phase.started"
    PHASE_COMPLETED = "phase.completed"
    PHASE_FAILED = "phase.failed"
    JOB_SUBMITTED = "job.submitted"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    PROMOTION_DECIDED = "promotion.decided"
    CLEANUP_COMPLETED = "cleanup.completed"
    CLEANUP_FAILED = "cleanup.failed"


class EventExporter(Protocol):
    """Dependency-injected event exporter."""

    def export(self, event: Mapping[str, object]) -> None:
        """Export one already-sanitized event."""


class EventLogger(Protocol):
    """Minimal logger contract accepted by :class:`TelemetryRecorder`."""

    def info(self, message: str, *args: object, **kwargs: object) -> None:
        """Log one already-sanitized event."""


EventSink = EventExporter | Callable[[dict[str, object]], None] | None

_REDACTED = "[REDACTED]"
_SENSITIVE_KEY = re.compile(
    r"(?:prompt|completion|raw[_-]?output|model[_-]?output|secret|token|password|"
    r"authorization|api[_-]?key|access[_-]?key|credential|private[_-]?key|cookie|"
    r"held[_-]?out|task(?:[_-]?(?:content|text|input))?|trajectory|traceback)",
    re.IGNORECASE,
)
_SENSITIVE_VALUE = (
    re.compile(r"bearer\s+\S+", re.IGNORECASE),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\b(?:eyJ[A-Za-z0-9_-]{10,}\.){2}[A-Za-z0-9_-]{10,}\b"),
    re.compile(
        r"\b(?:api[_ -]?key|access[_ -]?token|secret(?:[_ -]?key)?|password)\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:private\s+)?prompt\b|\braw\s+(?:model\s+)?output\b", re.IGNORECASE),
    re.compile(r"\bheld[-_ ]?out\b|\bsealed\s+task\b", re.IGNORECASE),
)
_SAFE_IDENTIFIER = re.compile(r"^[^\x00-\x1f\x7f]{1,256}$")
_OPERATIONAL_METADATA_KEYS = frozenset(
    {
        "artifact_id",
        "artifact_sha256",
        "attempt",
        "baseline_score",
        "benchmark_id",
        "candidate_score",
        "checkpoint_version",
        "component",
        "cost_usd",
        "dataset_version",
        "decision",
        "duration_ms",
        "environment",
        "episode_count",
        "error_code",
        "evaluation_status",
        "event_id",
        "experiment_id",
        "job_id",
        "job_type",
        "latency_ms",
        "manifest_sha256",
        "metric",
        "metric_name",
        "metric_value",
        "mode",
        "model_id",
        "model_family",
        "operation",
        "outcome",
        "phase",
        "provider",
        "region",
        "regression",
        "resource_type",
        "retry_count",
        "role",
        "service",
        "status",
        "suite",
        "suite_version",
        "task_count",
        "training_status",
    }
)


def _metadata_key(key: str) -> str:
    return re.sub(r"[- ]+", "_", key.strip().lower())


def _redact_value(key: str, value: Any) -> Any:
    """Return JSON-safe metadata with sensitive content replaced."""

    if _SENSITIVE_KEY.search(key):
        return _REDACTED
    if isinstance(value, str):
        if (
            _metadata_key(key) not in _OPERATIONAL_METADATA_KEYS
            or any(pattern.search(value) for pattern in _SENSITIVE_VALUE)
        ):
            return _REDACTED
        return value
    if isinstance(value, Mapping):
        return {
            str(child_key): _redact_value(str(child_key), child_value)
            for child_key, child_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_value(key, child) for child in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    # Do not call repr/str on arbitrary objects: their representation can leak
    # prompt text, credentials, or provider response bodies.
    return _REDACTED


def _sanitize_attributes(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    if attributes is None:
        return {}
    if not isinstance(attributes, Mapping):
        raise ValueError("attributes must be a mapping")
    sanitized = {
        str(key): _redact_value(str(key), value)
        for key, value in attributes.items()
    }
    # Ensure the injected sinks only receive JSON-safe data.  This also catches
    # non-finite values without serializing arbitrary provider objects.
    try:
        json.dumps(sanitized, allow_nan=False)
    except (TypeError, ValueError):
        return {"metadata_error": _REDACTED}
    return sanitized


def _freeze_value(value: Any) -> Any:
    """Recursively make metadata immutable after event creation."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_value(child) for key, child in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(child) for child in value)
    return value


def _thaw_value(value: Any) -> Any:
    """Create a JSON-friendly mutable copy for sink serialization."""

    if isinstance(value, Mapping):
        return {str(key): _thaw_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_value(child) for child in value]
    return value


def _validate_identifier(name: str, value: str) -> str:
    if not isinstance(value, str) or not _SAFE_IDENTIFIER.fullmatch(value.strip()):
        raise ValueError(f"{name} must be a non-empty safe identifier")
    if any(pattern.search(value) for pattern in _SENSITIVE_VALUE):
        raise ValueError(f"{name} cannot contain sensitive material")
    return value


def _optional_identifier(name: str, value: str | None) -> str | None:
    return None if value is None else _validate_identifier(name, value)


def _default_logger(payload: dict[str, object]) -> None:
    logging.getLogger("app.observability").info(
        "post-training telemetry", extra={"telemetry": payload}
    )


def _default_tracer() -> Any:
    if _otel_trace is None:
        return None
    try:
        return _otel_trace.get_tracer("autonomous-post-training-engineer")
    except Exception:  # pragma: no cover - protects optional integrations
        return None


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    """A sanitized, correlation-ready lifecycle event."""

    event_type: EventType
    run_id: str
    run_number: int
    experiment_id: str
    event_id: str = field(default_factory=lambda: uuid4().hex)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    phase: str | None = None
    job_id: str | None = None
    evidence_label: EvidenceLabel | None = None
    status: str | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Frozen dataclasses protect field reassignment but not nested dicts;
        # freeze the sanitized tree so callers cannot mutate an emitted event.
        object.__setattr__(
            self,
            "attributes",
            _freeze_value(_sanitize_attributes(self.attributes)),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the event without including arbitrary object values."""

        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "occurred_at": self.occurred_at.astimezone(UTC).isoformat(),
            "run_id": self.run_id,
            "run_number": self.run_number,
            "experiment_id": self.experiment_id,
            "phase": self.phase,
            "job_id": self.job_id,
            "evidence_label": self.evidence_label.value if self.evidence_label else None,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "attributes": _thaw_value(self.attributes),
        }


class TelemetryRecorder:
    """Record metadata-only events to injected sinks and optional OTel spans."""

    def __init__(
        self,
        *,
        exporter: EventSink = None,
        logger: EventSink = _default_logger,
        tracer: Any = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._exporter = exporter
        self._logger = logger
        self._tracer = _default_tracer() if tracer is None else tracer
        self._clock = clock or (lambda: datetime.now(UTC))

    def record(
        self,
        event_type: EventType | str,
        *,
        run_id: str,
        run_number: int,
        experiment_id: str,
        phase: str | None = None,
        job_id: str | None = None,
        evidence_label: EvidenceLabel | str | None = None,
        status: str | None = None,
        latency_ms: float | None = None,
        cost_usd: float | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> TelemetryEvent:
        """Validate, sanitize, emit, and return one lifecycle event.

        Sink failures are intentionally isolated from the run: observability
        must not turn a successful local or cloud phase into a failed phase.
        """

        try:
            normalized_type = (
                event_type if isinstance(event_type, EventType) else EventType(event_type)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown event_type: {event_type!r}") from exc
        if not isinstance(run_number, int) or isinstance(run_number, bool) or run_number < 1:
            raise ValueError("run_number must be a positive integer")
        if latency_ms is not None and (
            not isinstance(latency_ms, (int, float))
            or isinstance(latency_ms, bool)
            or not math.isfinite(float(latency_ms))
            or latency_ms < 0
        ):
            raise ValueError("latency_ms must be finite and non-negative")
        if cost_usd is not None and (
            not isinstance(cost_usd, (int, float))
            or isinstance(cost_usd, bool)
            or not math.isfinite(float(cost_usd))
            or cost_usd < 0
        ):
            raise ValueError("cost_usd must be finite and non-negative")
        if normalized_type is EventType.PROMOTION_DECIDED and evidence_label is None:
            raise ValueError("promotion events require evidence_label")

        normalized_label: EvidenceLabel | None = None
        if evidence_label is not None:
            try:
                normalized_label = (
                    evidence_label
                    if isinstance(evidence_label, EvidenceLabel)
                    else EvidenceLabel(evidence_label)
                )
            except ValueError as exc:
                raise ValueError(
                    "evidence_label must be LIVE, PRIOR_VERIFIED_RUN, or EXPLANATION"
                ) from exc

        event = TelemetryEvent(
            event_type=normalized_type,
            run_id=_validate_identifier("run_id", run_id),
            run_number=run_number,
            experiment_id=_validate_identifier("experiment_id", experiment_id),
            occurred_at=self._clock().astimezone(UTC),
            phase=_optional_identifier("phase", phase),
            job_id=_optional_identifier("job_id", job_id),
            evidence_label=normalized_label,
            status=_optional_identifier("status", status),
            latency_ms=float(latency_ms) if latency_ms is not None else None,
            cost_usd=float(cost_usd) if cost_usd is not None else None,
            attributes=_sanitize_attributes(attributes),
        )
        payload = event.to_dict()
        self._safe_sink(self._exporter, payload)
        self._safe_sink(self._logger, payload)
        self._record_otel(event)
        return event

    def _safe_sink(self, sink: EventSink, payload: dict[str, object]) -> None:
        if sink is None:
            return
        try:
            if callable(sink):
                sink(payload)
            elif hasattr(sink, "export"):
                sink.export(payload)
            elif hasattr(sink, "emit"):
                sink.emit(payload)
            elif hasattr(sink, "info"):
                sink.info("post-training telemetry", extra={"telemetry": payload})
        except Exception:
            # Telemetry is best effort and never changes application outcomes.
            return

    def _record_otel(self, event: TelemetryEvent) -> None:
        if self._tracer is None:
            return
        attrs: dict[str, str | int | float] = {
            "event.type": event.event_type.value,
            "run.id": event.run_id,
            "run.number": event.run_number,
            "experiment.id": event.experiment_id,
        }
        for key, value in (
            ("phase", event.phase),
            ("job.id", event.job_id),
            ("evidence.label", event.evidence_label.value if event.evidence_label else None),
            ("status", event.status),
        ):
            if value is not None:
                attrs[key] = value
        if event.latency_ms is not None:
            attrs["latency_ms"] = event.latency_ms
        if event.cost_usd is not None:
            attrs["cost_usd"] = event.cost_usd
        try:
            span = self._tracer.start_span(event.event_type.value, attributes=attrs)
            span.end()
        except Exception:
            return


__all__ = ["EventExporter", "EventLogger", "EventType", "TelemetryEvent", "TelemetryRecorder"]
