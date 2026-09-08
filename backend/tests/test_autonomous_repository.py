from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.autonomous.models import (
    AutonomousRunState,
    AutonomousRunStatus,
    ExperimentRecord,
    RunEventRecord,
    RunOperation,
    RunOperationStatus,
    RunPhase,
)
from app.autonomous.repository import (
    ApprovalAlreadyConsumedError,
    ConcurrentUpdateError,
    InMemoryAutonomousRunRepository,
    LeaseConflictError,
    OperationAlreadyExistsError,
    RunAlreadyExistsError,
)


def make_run(run_id: str = "run-1") -> AutonomousRunState:
    return AutonomousRunState(
        run_id=run_id,
        model_id="google/functiongemma-270m-it",
        checkpoint_revision="a" * 40,
        benchmark_id="service-recovery-v1",
        benchmark_manifest_sha256="b" * 64,
        max_experiments=5,
        approved_budget_usd=25.0,
        status=AutonomousRunStatus.PREPARED,
        phase=RunPhase.PREPARED,
    )


def test_create_is_conditional_and_get_returns_defensive_copy() -> None:
    repository = InMemoryAutonomousRunRepository()
    created = repository.create(make_run())

    assert created.version == 0
    assert created.event_sequence == 0
    assert repository.get("run-1") == created

    with pytest.raises(RunAlreadyExistsError):
        repository.create(make_run())

    created.experiments.append(ExperimentRecord(experiment_number=1))  # type: ignore[attr-defined]
    assert repository.get("run-1") is not None
    assert repository.get("run-1").experiments == []  # type: ignore[union-attr]


def test_transition_is_optimistic_and_appends_atomic_ordered_event() -> None:
    repository = InMemoryAutonomousRunRepository()
    repository.create(make_run())

    transitioned = repository.transition(
        "run-1",
        expected_version=0,
        status=AutonomousRunStatus.QUEUED,
        phase=RunPhase.QUEUED,
        reason="approval consumed",
    )
    assert transitioned.version == 1
    assert transitioned.event_sequence == 1
    assert transitioned.status is AutonomousRunStatus.QUEUED

    with pytest.raises(ConcurrentUpdateError):
        repository.transition(
            "run-1",
            expected_version=0,
            status=AutonomousRunStatus.RUNNING,
            phase=RunPhase.BASELINE,
            reason="stale worker",
        )

    events = repository.list_events("run-1")
    assert len(events) == 1
    assert events[0].sequence == 1
    assert events[0].to_status is AutonomousRunStatus.QUEUED


def test_approval_digest_can_be_consumed_only_once_and_must_match_scope() -> None:
    repository = InMemoryAutonomousRunRepository()
    repository.create(make_run())

    consumed = repository.consume_approval("run-1", "c" * 64)
    assert consumed.approval_consumed is True
    assert consumed.approval_digest == "c" * 64

    with pytest.raises(ApprovalAlreadyConsumedError):
        repository.consume_approval("run-1", "c" * 64)
    with pytest.raises(ApprovalAlreadyConsumedError):
        repository.consume_approval("run-1", "d" * 64)


def test_lease_claim_renew_release_and_recovery_scan() -> None:
    repository = InMemoryAutonomousRunRepository()
    repository.create(make_run("queued"))
    repository.create(make_run("running"))
    repository.transition(
        "running",
        expected_version=0,
        status=AutonomousRunStatus.RUNNING,
        phase=RunPhase.BASELINE,
        reason="worker started",
    )
    now = datetime(2026, 9, 8, tzinfo=UTC)

    claimed = repository.claim_lease("queued", "worker-a", now=now, ttl_seconds=30)
    assert claimed.lease_owner == "worker-a"
    with pytest.raises(LeaseConflictError):
        repository.claim_lease("queued", "worker-b", now=now, ttl_seconds=30)
    renewed = repository.renew_lease("queued", "worker-a", now=now, ttl_seconds=60)
    assert renewed.lease_expires_at == now + timedelta(seconds=60)
    released = repository.release_lease("queued", "worker-a")
    assert released.lease_owner is None

    expired = repository.claim_lease("running", "worker-old", now=now, ttl_seconds=1)
    assert expired.lease_expires_at == now + timedelta(seconds=1)
    recoverable = repository.scan_recoverable(now=now + timedelta(seconds=2))
    assert [item.run_id for item in recoverable] == ["queued", "running"]


def test_operation_intent_is_idempotent_and_result_reconciles() -> None:
    repository = InMemoryAutonomousRunRepository()
    repository.create(make_run())
    operation = RunOperation(
        operation_key="run-1:1:training",
        run_id="run-1",
        experiment_number=1,
        phase=RunPhase.TRAINING,
        provider_name="sagemaker-job-1",
        status=RunOperationStatus.INTENT,
    )
    assert repository.put_operation_intent(operation) == operation
    assert repository.put_operation_intent(operation) == operation
    with pytest.raises(OperationAlreadyExistsError):
        repository.put_operation_intent(operation.model_copy(update={"provider_name": "other"}))

    result = repository.record_operation_result(
        "run-1", operation.operation_key, provider_id="arn:aws:sagemaker:job/1", status="SUCCEEDED"
    )
    assert result.provider_id is not None and result.provider_id.endswith("/1")
    assert repository.get_operation("run-1", operation.operation_key) == result


def test_event_and_history_reads_are_paginated() -> None:
    repository = InMemoryAutonomousRunRepository()
    repository.create(make_run())
    for index in range(3):
        repository.transition(
            "run-1",
            expected_version=index,
            status=AutonomousRunStatus.RUNNING,
            phase=RunPhase.BASELINE,
            reason=f"tick {index}",
        )
    events = repository.list_events("run-1", after_sequence=1, limit=1)
    assert [event.sequence for event in events.items] == [2]
    assert events.next_after == 2

    repository.add_experiment("run-1", ExperimentRecord(experiment_number=1))
    repository.add_experiment("run-1", ExperimentRecord(experiment_number=2))
    history = repository.list_experiments("run-1", offset=1, limit=1)
    assert [item.experiment_number for item in history.items] == [2]
    assert history.next_offset is None


def test_models_reject_invalid_scope_and_unknown_fields() -> None:
    with pytest.raises(ValueError):
        AutonomousRunState.model_validate(
            {**make_run().model_dump(), "checkpoint_revision": "mutable"}
        )
    with pytest.raises(ValueError):
        AutonomousRunState.model_validate({**make_run().model_dump(), "unexpected": True})
    with pytest.raises(ValueError):
        RunEventRecord(
            run_id="run-1",
            sequence=0,
            event_type="bad",
            to_status=AutonomousRunStatus.QUEUED,
            to_phase=RunPhase.QUEUED,
            reason="bad",
        )
