"""Contract tests for metadata-only run telemetry."""

from __future__ import annotations

import json
import math
from types import MappingProxyType

import pytest

from app.observability import EventType, TelemetryRecorder


def test_record_emits_correlated_event_with_latency_cost_and_evidence() -> None:
    exported: list[dict[str, object]] = []
    logged: list[dict[str, object]] = []
    recorder = TelemetryRecorder(exporter=exported.append, logger=logged.append)

    event = recorder.record(
        EventType.PHASE_COMPLETED,
        run_id="run-001",
        run_number=1,
        experiment_id="experiment-001",
        phase="benchmark",
        job_id="job-001",
        evidence_label="LIVE",
        status="succeeded",
        latency_ms=123.5,
        cost_usd=0.47,
        attributes={"suite": "agentgym", "environment": "webshop"},
    )

    assert event.event_type is EventType.PHASE_COMPLETED
    assert event.run_id == "run-001"
    assert event.run_number == 1
    assert event.experiment_id == "experiment-001"
    assert event.latency_ms == pytest.approx(123.5)
    assert event.cost_usd == pytest.approx(0.47)
    assert event.evidence_label == "LIVE"
    assert exported == [event.to_dict()]
    assert logged == [event.to_dict()]


def test_redacts_sensitive_keys_recursively_and_preserves_safe_metadata() -> None:
    exported: list[dict[str, object]] = []
    recorder = TelemetryRecorder(exporter=exported.append)

    recorder.record(
        EventType.JOB_COMPLETED,
        run_id="run-001",
        run_number=1,
        experiment_id="experiment-001",
        phase="training",
        attributes={
            "suite": "agentgym",
            "prompt": "private prompt text",
            "raw_model_output": "private completion text",
            "authorization": "Bearer private-token",
            "nested": {
                "api_key": "secret-value",
                "held_out_task_content": "sealed task",
                "status": "succeeded",
            },
        },
    )

    payload = json.dumps(exported[0], sort_keys=True)
    for secret in (
        "private prompt text",
        "private completion text",
        "private-token",
        "secret-value",
        "sealed task",
    ):
        assert secret not in payload
    assert exported[0]["attributes"] == {
        "suite": "agentgym",
        "prompt": "[REDACTED]",
        "raw_model_output": "[REDACTED]",
        "authorization": "[REDACTED]",
        "nested": {
            "api_key": "[REDACTED]",
            "held_out_task_content": "[REDACTED]",
            "status": "succeeded",
        },
    }


def test_redacts_secret_like_values_even_when_the_key_is_not_sensitive() -> None:
    exported: list[dict[str, object]] = []
    recorder = TelemetryRecorder(exporter=exported.append)

    recorder.record(
        EventType.RUN_STARTED,
        run_id="run-001",
        run_number=1,
        experiment_id="experiment-001",
        attributes={
            "provider": "bedrock",
            "detail": "Bearer very-secret-value",
            "access_key": "AKIAIOSFODNN7EXAMPLE",
        },
    )

    payload = json.dumps(exported[0], sort_keys=True)
    assert "very-secret-value" not in payload
    assert "AKIAIOSFODNN7EXAMPLE" not in payload
    assert exported[0]["attributes"] == {
        "provider": "bedrock",
        "detail": "[REDACTED]",
        "access_key": "[REDACTED]",
    }


def test_rejects_invalid_correlation_and_measurements() -> None:
    recorder = TelemetryRecorder()

    with pytest.raises(ValueError, match="run_number"):
        recorder.record(
            EventType.RUN_STARTED,
            run_id="run-001",
            run_number=0,
            experiment_id="experiment-001",
        )
    with pytest.raises(ValueError, match="latency_ms"):
        recorder.record(
            EventType.RUN_STARTED,
            run_id="run-001",
            run_number=1,
            experiment_id="experiment-001",
            latency_ms=-1,
        )
    with pytest.raises(ValueError, match="latency_ms"):
        recorder.record(
            EventType.RUN_STARTED,
            run_id="run-001",
            run_number=1,
            experiment_id="experiment-001",
            latency_ms=math.inf,
        )
    with pytest.raises(ValueError, match="cost_usd"):
        recorder.record(
            EventType.RUN_STARTED,
            run_id="run-001",
            run_number=1,
            experiment_id="experiment-001",
            cost_usd=math.nan,
        )
    with pytest.raises(ValueError, match="evidence_label"):
        recorder.record(
            EventType.PROMOTION_DECIDED,
            run_id="run-001",
            run_number=1,
            experiment_id="experiment-001",
            evidence_label="SIMULATED",
        )


def test_identifiers_and_allow_listed_string_metadata_reject_free_form_content() -> None:
    exported: list[dict[str, object]] = []
    recorder = TelemetryRecorder(exporter=exported.append, logger=None, tracer=None)

    with pytest.raises(ValueError, match="run_id"):
        recorder.record(
            EventType.RUN_STARTED,
            run_id="run-001 contains words",
            run_number=1,
            experiment_id="experiment-001",
        )
    with pytest.raises(ValueError, match="run_id"):
        recorder.record(
            EventType.RUN_STARTED,
            run_id="private_prompt",
            run_number=1,
            experiment_id="experiment-001",
        )

    event = recorder.record(
        EventType.RUN_STARTED,
        run_id="run-001",
        run_number=1,
        experiment_id="experiment-001",
        attributes={
            "provider": "some harmless but free form text",
            "region": "sealed-task",
        },
    )
    assert event.attributes["provider"] == "[REDACTED]"
    assert event.attributes["region"] == "[REDACTED]"


def test_promotion_events_require_evidence_label() -> None:
    recorder = TelemetryRecorder()

    with pytest.raises(ValueError, match="evidence_label"):
        recorder.record(
            EventType.PROMOTION_DECIDED,
            run_id="run-001",
            run_number=1,
            experiment_id="experiment-001",
        )


def test_exporter_and_logger_failures_do_not_break_local_runs() -> None:
    calls: list[str] = []

    def broken_exporter(_: dict[str, object]) -> None:
        calls.append("exporter")
        raise RuntimeError("exporter unavailable")

    def broken_logger(_: dict[str, object]) -> None:
        calls.append("logger")
        raise RuntimeError("logger unavailable")

    recorder = TelemetryRecorder(exporter=broken_exporter, logger=broken_logger)
    event = recorder.record(
        EventType.RUN_STARTED,
        run_id="run-001",
        run_number=1,
        experiment_id="experiment-001",
    )

    assert event.run_id == "run-001"
    assert calls == ["exporter", "logger"]


def test_redacts_unknown_metadata_strings_and_freezes_nested_attributes() -> None:
    exported: list[dict[str, object]] = []
    recorder = TelemetryRecorder(exporter=exported.append, logger=None, tracer=None)

    event = recorder.record(
        EventType.PHASE_COMPLETED,
        run_id="run-001",
        run_number=1,
        experiment_id="experiment-001",
        attributes={
            "suite": "agentgym",
            "unknown_string": "do not retain arbitrary content",
            "nested": {
                "environment": "webshop",
                "freeform": "do not retain nested content",
            },
            "attempt": 2,
            "items": [1, 2],
            "durable_event_type": "unregistered.event",
            "evidence_label": "FAKE",
        },
    )

    assert exported[0]["attributes"] == {
        "suite": "agentgym",
        "unknown_string": "[REDACTED]",
        "nested": {"environment": "webshop", "freeform": "[REDACTED]"},
        "attempt": 2,
        "items": [1, 2],
        "durable_event_type": "[REDACTED]",
        "evidence_label": "[REDACTED]",
    }
    assert isinstance(event.attributes, MappingProxyType)
    assert isinstance(event.attributes["nested"], MappingProxyType)
    assert event.attributes["items"] == (1, 2)
    with pytest.raises(TypeError):
        event.attributes["suite"] = "changed"
    with pytest.raises(TypeError):
        event.attributes["nested"]["environment"] = "changed"

    payload = event.to_dict()
    assert isinstance(payload["attributes"], dict)
    payload["attributes"]["suite"] = "changed"
    assert event.attributes["suite"] == "agentgym"


def test_otel_span_carries_event_identity_and_allow_listed_semantics() -> None:
    spans: list[dict[str, object]] = []

    class Span:
        def end(self) -> None:
            return None

    class Tracer:
        def start_span(self, name: str, *, attributes: dict[str, object]) -> Span:
            spans.append({"name": name, **attributes})
            return Span()

    from app.autonomous.models import AutonomousRunState
    from app.autonomous.repository import InMemoryAutonomousRunRepository
    from app.autonomous.telemetry import AutonomousEventType, DurableTelemetryBridge

    repository = InMemoryAutonomousRunRepository()
    repository.create(
        AutonomousRunState(
            run_id="run-otel",
            checkpoint_revision="a" * 40,
            benchmark_manifest_sha256="b" * 64,
        )
    )
    recorder = TelemetryRecorder(exporter=None, logger=None, tracer=Tracer())
    persisted = DurableTelemetryBridge(repository, recorder=recorder).emit(
        AutonomousEventType.APPROVAL_CONSUMED,
        run_id="run-otel",
        run_number=1,
        experiment_id="run-otel:1",
        reason="approval consumed",
        approval_digest="c" * 64,
    )

    assert len(spans) == 1
    assert spans[0]["event.id"] == persisted.event_id
    assert spans[0]["autonomous.event.type"] == "approval.consumed"
