"""Contract tests for the durable autonomous telemetry bridge."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import pytest

from app.autonomous.models import AutonomousRunState, AutonomousRunStatus, RunEventRecord, RunPhase
from app.autonomous.telemetry import (
    AutonomousEventType,
    DurableTelemetryBridge,
    DurableTelemetryError,
    validate_safe_metadata,
)


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
