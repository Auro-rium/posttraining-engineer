"""Contract tests for the durable autonomous telemetry bridge."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import pytest

from app.autonomous.models import AutonomousRunState, AutonomousRunStatus, RunEventRecord, RunPhase
from app.autonomous.repository import InMemoryAutonomousRunRepository
from app.autonomous.telemetry import (
    CANONICAL_EVENT_TYPES,
    AutonomousEventType,
    DurableTelemetryBridge,
    DurableTelemetryError,
    validate_safe_metadata,
)
from app.observability import TelemetryRecorder


class RecordingRepository:
    def __init__(self) -> None:
        self.events: list[RunEventRecord] = []
        self.transitions: list[dict[str, Any]] = []
        self.transition_result = object()
        self.fail = False

    def append_event(
        self,
        run_id: str,
        *,
        event_type: str,
        reason: str,
        metadata: Mapping[str, str] | None = None,
    ) -> RunEventRecord:
        if self.fail:
            raise RuntimeError("durable store unavailable")
        event = RunEventRecord(
            run_id=run_id,
            sequence=len(self.events) + 1,
            event_type=event_type,
            to_status=AutonomousRunStatus.RUNNING,
            to_phase=RunPhase.BASELINE,
            reason=reason,
            metadata=dict(metadata or {}),
        )
        self.events.append(event)
        return event

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
    ) -> AutonomousRunState:
        if self.fail:
            raise RuntimeError("durable store unavailable")
        self.transitions.append(
            {
                "run_id": run_id,
                "expected_version": expected_version,
                "status": status,
                "phase": phase,
                "reason": reason,
                "event_type": event_type,
                "metadata": dict(metadata or {}),
            }
        )
        return cast(AutonomousRunState, self.transition_result)


class RecordingSink:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail = False

    def record(self, event_type: Any, **kwargs: Any) -> None:
        self.calls.append({"event_type": event_type, **kwargs})
        if self.fail:
            raise RuntimeError("optional sink unavailable")


def test_emit_persists_safe_event_before_forwarding_optional_telemetry() -> None:
    repository = RecordingRepository()
    sink = RecordingSink()
    bridge = DurableTelemetryBridge(repository, recorder=sink)

    persisted = bridge.emit(
        AutonomousEventType.PHASE_STARTED,
        run_id="run-1",
        run_number=1,
        experiment_id="exp-1",
        reason="phase started",
        phase="baseline",
        status="running",
        latency_ms=12.5,
        metadata={"reason_code": "baseline_start"},
    )

    assert persisted is repository.events[0]
    assert persisted.event_type == "phase.started"
    assert persisted.metadata["phase"] == "BASELINE"
    assert persisted.metadata["latency_ms"] == "12.5"
    assert persisted.metadata["run_number"] == "1"
    event_type = sink.calls[0]["event_type"]
    assert getattr(event_type, "value", event_type) == "phase.started"


def test_transition_uses_atomic_repository_transition_and_returns_durable_record() -> None:
    repository = RecordingRepository()
    sink = RecordingSink()
    bridge = DurableTelemetryBridge(repository, recorder=sink)

    result = bridge.transition(
        AutonomousEventType.RUN_COMPLETED,
        run_id="run-1",
        expected_version=3,
        status=AutonomousRunStatus.SUCCEEDED,
        phase=RunPhase.COMPLETED,
        reason="completed",
        run_number=1,
        experiment_id="exp-1",
        status_metadata="succeeded",
    )

    assert result is repository.transition_result
    assert repository.transitions[0]["expected_version"] == 3
    assert repository.transitions[0]["event_type"] == "run.completed"
    assert sink.calls[0]["status"] == "succeeded"


def test_durable_failure_is_authoritative_and_does_not_call_optional_sink() -> None:
    repository = RecordingRepository()
    repository.fail = True
    sink = RecordingSink()
    bridge = DurableTelemetryBridge(repository, recorder=sink)

    with pytest.raises(DurableTelemetryError, match="durable"):
        bridge.emit(
            AutonomousEventType.RUN_STARTED,
            run_id="run-1",
            run_number=1,
            experiment_id="exp-1",
            reason="started",
        )
    assert sink.calls == []


def test_optional_sink_failure_does_not_change_successful_durable_event() -> None:
    repository = RecordingRepository()
    sink = RecordingSink()
    sink.fail = True
    bridge = DurableTelemetryBridge(repository, recorder=sink)

    event = bridge.emit(
        AutonomousEventType.RUN_STARTED,
        run_id="run-1",
        run_number=1,
        experiment_id="exp-1",
        reason="started",
    )

    assert event is repository.events[0]
    assert len(repository.events) == 1


def test_durable_event_id_is_forwarded_to_optional_sink_metadata() -> None:
    repository = RecordingRepository()
    sink = RecordingSink()
    bridge = DurableTelemetryBridge(repository, recorder=sink)

    persisted = bridge.emit(
        AutonomousEventType.RUN_STARTED,
        run_id="run-1",
        run_number=1,
        experiment_id="exp-1",
        reason="started",
    )

    assert sink.calls[0]["attributes"]["event_id"] == persisted.event_id


def test_durable_event_id_survives_existing_recorder_sanitization() -> None:
    repository = RecordingRepository()
    exported: list[dict[str, object]] = []
    bridge = DurableTelemetryBridge(
        repository,
        recorder=TelemetryRecorder(exporter=exported.append, logger=None, tracer=None),
    )

    persisted = bridge.emit(
        AutonomousEventType.RUN_STARTED,
        run_id="run-1",
        run_number=1,
        experiment_id="exp-1",
        reason="started",
    )

    attributes = cast(dict[str, object], exported[0]["attributes"])
    assert attributes["event_id"] == persisted.event_id


def test_transition_forwards_the_repository_generated_event_id() -> None:
    repository = InMemoryAutonomousRunRepository()
    repository.create(
        AutonomousRunState(
            run_id="run-1",
            checkpoint_revision="a" * 40,
            benchmark_manifest_sha256="b" * 64,
        )
    )
    sink = RecordingSink()
    bridge = DurableTelemetryBridge(repository, recorder=sink)

    bridge.transition(
        AutonomousEventType.RUN_COMPLETED,
        run_id="run-1",
        expected_version=0,
        status=AutonomousRunStatus.SUCCEEDED,
        phase=RunPhase.COMPLETED,
        reason="completed",
        run_number=1,
        experiment_id="exp-1",
    )

    persisted = repository.list_events("run-1")[0]
    assert sink.calls[0]["attributes"]["event_id"] == persisted.event_id


def test_event_specific_correlation_fields_are_required() -> None:
    repository = RecordingRepository()
    bridge = DurableTelemetryBridge(repository)

    with pytest.raises(ValueError, match="phase"):
        bridge.emit(
            AutonomousEventType.PHASE_STARTED,
            run_id="run-1",
            run_number=1,
            experiment_id="exp-1",
            reason="started",
        )
    with pytest.raises(ValueError, match="job_id"):
        bridge.emit(
            AutonomousEventType.JOB_SUBMITTED,
            run_id="run-1",
            run_number=1,
            experiment_id="exp-1",
            reason="submitted",
        )
    with pytest.raises(ValueError, match="operation_key"):
        bridge.emit(
            AutonomousEventType.OPERATION_INTENT,
            run_id="run-1",
            run_number=1,
            experiment_id="exp-1",
            reason="requested",
        )


def test_explicit_repository_style_wrappers_keep_argument_order_unambiguous() -> None:
    repository = RecordingRepository()
    bridge = DurableTelemetryBridge(repository)

    emitted = bridge.append_event(
        "run-1",
        event_type=AutonomousEventType.RUN_STARTED,
        run_number=1,
        experiment_id="exp-1",
        reason="started",
    )
    assert emitted is repository.events[0]

    transitioned = bridge.transition_run(
        "run-1",
        event_type=AutonomousEventType.RUN_COMPLETED,
        expected_version=2,
        status=AutonomousRunStatus.SUCCEEDED,
        phase=RunPhase.COMPLETED,
        reason="completed",
        run_number=1,
        experiment_id="exp-1",
    )
    assert transitioned is repository.transition_result


def test_rejects_unstructured_reason_before_persistence() -> None:
    repository = RecordingRepository()
    bridge = DurableTelemetryBridge(repository)
    with pytest.raises(ValueError, match="reason"):
        bridge.emit(
            AutonomousEventType.RUN_STARTED,
            run_id="run-1",
            run_number=1,
            experiment_id="exp-1",
            reason="something entirely free form",
        )


def test_new_semantic_events_map_to_existing_observer_vocabulary() -> None:
    assert CANONICAL_EVENT_TYPES[AutonomousEventType.OPERATION_INTENT].value == "job.submitted"
    assert CANONICAL_EVENT_TYPES[AutonomousEventType.OPERATION_COMPLETED].value == "job.completed"


@pytest.mark.parametrize(
    "metadata",
    [
        {"prompt": "do not persist"},
        {"completion": "do not persist"},
        {"trajectory": "traj://private"},
        {"hidden_input": "sealed"},
        {"authorization": "Bearer abc"},
        {"unknown": "free-form"},
        {"status": "contains spaces"},
    ],
)
def test_rejects_sensitive_and_unknown_metadata(metadata: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        validate_safe_metadata(metadata)


def test_rejects_non_finite_measurements_before_persistence() -> None:
    repository = RecordingRepository()
    bridge = DurableTelemetryBridge(repository)
    with pytest.raises(ValueError, match="latency_ms"):
        bridge.emit(
            AutonomousEventType.PHASE_COMPLETED,
            run_id="run-1",
            run_number=1,
            experiment_id="exp-1",
            reason="phase completed",
            latency_ms=math.inf,
        )
    assert repository.events == []


def test_vocabulary_contains_control_plane_and_provider_lifecycle_events() -> None:
    values = {event.value for event in AutonomousEventType}
    assert {
        "run.started",
        "run.completed",
        "run.failed",
        "phase.started",
        "phase.completed",
        "phase.failed",
        "job.submitted",
        "job.completed",
        "job.failed",
        "promotion.decided",
        "cleanup.completed",
        "cleanup.failed",
        "approval.consumed",
        "operation.intent",
        "operation.submitted",
        "operation.completed",
        "operation.failed",
    } <= values
