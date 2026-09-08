from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast

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
    DynamoDBAutonomousRunRepository,
    InMemoryAutonomousRunRepository,
    LeaseConflictError,
    OperationAlreadyExistsError,
    RunAlreadyExistsError,
    RunNotFoundError,
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

    with pytest.raises((TypeError, AttributeError)):
        created.experiments.append(ExperimentRecord(experiment_number=1))
    assert repository.get("run-1") is not None
    stored = repository.get("run-1")
    assert stored is not None
    assert tuple(stored.experiments) == ()


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
    assert (
        repository.record_operation_result(
            "run-1",
            operation.operation_key,
            provider_id="arn:aws:sagemaker:job/1",
            status=RunOperationStatus.SUCCEEDED,
        )
        == result
    )
    with pytest.raises(OperationAlreadyExistsError):
        repository.record_operation_result(
            "run-1",
            operation.operation_key,
            provider_id="arn:aws:sagemaker:job/2",
            status=RunOperationStatus.FAILED,
        )
    with pytest.raises(OperationAlreadyExistsError):
        repository.put_operation_intent(
            operation.model_copy(update={"experiment_number": 2, "phase": RunPhase.EVALUATION})
        )


def test_operation_retry_identity_uses_stable_request_digest_not_timestamps() -> None:
    repository = InMemoryAutonomousRunRepository()
    repository.create(make_run())
    first = RunOperation(
        operation_key="stable-key",
        run_id="run-1",
        experiment_number=1,
        phase=RunPhase.TRAINING,
        provider_name="job",
        request_digest="d" * 64,
    )
    second = first.model_copy(
        update={
            "created_at": first.created_at + timedelta(seconds=1),
            "updated_at": first.updated_at + timedelta(seconds=1),
        }
    )
    assert repository.put_operation_intent(first) == first
    assert repository.put_operation_intent(second) == first
    with pytest.raises(OperationAlreadyExistsError):
        repository.put_operation_intent(second.model_copy(update={"request_digest": "e" * 64}))


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


class StubDynamoTable:
    """Native resource-table stub with keyset pagination and captured writes."""

    name = "autonomous-runs"

    def __init__(self) -> None:
        self.items: list[dict[str, object]] = []
        self.puts: list[dict[str, object]] = []
        self.queries: list[dict[str, object]] = []
        self.page_size = 1

    def put_item(self, **kwargs: object) -> None:
        self.puts.append(kwargs)

    def get_item(self, **kwargs: object) -> dict[str, object]:
        key = kwargs["Key"]
        assert isinstance(key, dict)
        for item in self.items:
            if item["pk"] == key["pk"] and item["sk"] == key["sk"]:
                return {"Item": item}
        return {}

    def query(self, **kwargs: object) -> dict[str, object]:
        self.queries.append(kwargs)
        expression = str(kwargs["KeyConditionExpression"])
        values = cast(dict[str, str], kwargs["ExpressionAttributeValues"])
        prefix = values.get(":prefix")
        filtered = [item for item in self.items if item["pk"] == values[":pk"]]
        if "BETWEEN" in expression:
            start, end = values[":start"], values[":end"]
            filtered = [item for item in filtered if str(start) <= str(item["sk"]) <= str(end)]
        elif prefix is not None:
            filtered = [item for item in filtered if str(item["sk"]).startswith(str(prefix))]
        filtered.sort(key=lambda item: str(item["sk"]))
        cursor = kwargs.get("ExclusiveStartKey")
        if isinstance(cursor, dict):
            keys = [(item["pk"], item["sk"]) for item in filtered]
            try:
                filtered = filtered[keys.index((cursor["pk"], cursor["sk"])) + 1 :]
            except ValueError:
                pass
        limit = int(cast(int, kwargs.get("Limit", self.page_size)))
        selected = filtered[:limit]
        response: dict[str, object] = {"Items": selected}
        if len(filtered) > len(selected) and selected:
            response["LastEvaluatedKey"] = {"pk": selected[-1]["pk"], "sk": selected[-1]["sk"]}
        return response

    def scan(self, **kwargs: object) -> dict[str, object]:
        self.queries.append(kwargs)
        cursor = kwargs.get("ExclusiveStartKey")
        items = sorted(self.items, key=lambda item: (str(item["pk"]), str(item["sk"])))
        if isinstance(cursor, dict):
            keys = [(item["pk"], item["sk"]) for item in items]
            items = items[keys.index((cursor["pk"], cursor["sk"])) + 1 :]
        limit = int(cast(int, kwargs["Limit"]))
        selected = items[:limit]
        response: dict[str, object] = {"Items": selected}
        if len(items) > len(selected) and selected:
            response["LastEvaluatedKey"] = {"pk": selected[-1]["pk"], "sk": selected[-1]["sk"]}
        return response


class StubDynamoClient:
    def __init__(self) -> None:
        self.transactions: list[dict[str, object]] = []

    def transact_write_items(self, **kwargs: object) -> None:
        self.transactions.append(kwargs)


def seed_dynamo_table(table: StubDynamoTable) -> None:
    run = make_run()
    table.items.extend(
        [
            DynamoDBAutonomousRunRepository._item("STATE", run),
            DynamoDBAutonomousRunRepository._item(
                "EVENT#00000000000000000001",
                RunEventRecord(
                    run_id="run-1",
                    sequence=1,
                    event_type="x",
                    to_status=AutonomousRunStatus.QUEUED,
                    to_phase=RunPhase.QUEUED,
                    reason="queued",
                ),
            ),
            DynamoDBAutonomousRunRepository._item(
                "EXP#0001", ExperimentRecord(experiment_number=1), run_id="run-1"
            ),
            DynamoDBAutonomousRunRepository._item(
                "OP#run-1:1:training",
                RunOperation(
                    operation_key="run-1:1:training",
                    run_id="run-1",
                    experiment_number=1,
                    phase=RunPhase.TRAINING,
                    provider_name="job",
                ),
            ),
        ]
    )


def test_dynamo_reads_are_entity_bounded_and_keyset_paginated() -> None:
    table = StubDynamoTable()
    seed_dynamo_table(table)
    repository = DynamoDBAutonomousRunRepository(table=table, client=StubDynamoClient())

    events = repository.list_events("run-1", limit=1)
    assert [item.sequence for item in events.items] == [1]
    assert events.next_cursor is None
    assert "BETWEEN" in str(table.queries[0]["KeyConditionExpression"])

    experiments = repository.list_experiments("run-1", limit=1)
    assert [item.experiment_number for item in experiments.items] == [1]
    assert "BETWEEN" in str(table.queries[1]["KeyConditionExpression"])
    assert repository.get_operation("run-1", "run-1:1:training") is not None
    assert repository.get_operation("run-1") is not None


def test_dynamo_event_after_sequence_is_exclusive() -> None:
    table = StubDynamoTable()
    seed_dynamo_table(table)
    table.items.append(
        DynamoDBAutonomousRunRepository._item(
            "EVENT#00000000000000000002",
            RunEventRecord(
                run_id="run-1",
                sequence=2,
                event_type="x",
                to_status=AutonomousRunStatus.RUNNING,
                to_phase=RunPhase.BASELINE,
                reason="started",
            ),
        )
    )
    repository = DynamoDBAutonomousRunRepository(table=table, client=StubDynamoClient())
    page = repository.list_events("run-1", after_sequence=1, limit=1)
    assert [item.sequence for item in page.items] == [2]


def test_dynamo_lease_uses_native_resource_expression_values_and_derived_name() -> None:
    table = StubDynamoTable()
    table.items.append(DynamoDBAutonomousRunRepository._item("STATE", make_run()))
    repository = DynamoDBAutonomousRunRepository(table=table, client=StubDynamoClient())
    repository.claim_lease("run-1", "worker", now=datetime(2026, 9, 8, tzinfo=UTC))
    values = table.puts[-1]["ExpressionAttributeValues"]
    assert values == {":version": 0}


def test_dynamo_operation_intent_requires_existing_run_and_uses_transaction() -> None:
    table = StubDynamoTable()
    client = StubDynamoClient()
    repository = DynamoDBAutonomousRunRepository(table=table, client=client)
    operation = RunOperation(
        operation_key="orphan",
        run_id="missing",
        experiment_number=1,
        phase=RunPhase.TRAINING,
        provider_name="job",
    )
    with pytest.raises(RunNotFoundError):
        repository.put_operation_intent(operation)

    table.items.append(DynamoDBAutonomousRunRepository._item("STATE", make_run()))
    operation = operation.model_copy(update={"run_id": "run-1"})
    repository.put_operation_intent(operation)
    assert len(client.transactions) == 1
    transact_items = cast(list[dict[str, object]], client.transactions[-1]["TransactItems"])
    assert any("ConditionCheck" in item for item in transact_items)


def test_event_contract_rejects_raw_content_and_nested_contracts_are_immutable() -> None:
    with pytest.raises(ValueError):
        RunEventRecord(
            run_id="run-1",
            sequence=1,
            event_type="x",
            to_status=AutonomousRunStatus.QUEUED,
            to_phase=RunPhase.QUEUED,
            reason="raw prompt: secret words",
        )
    with pytest.raises(ValueError):
        RunEventRecord(
            run_id="run-1",
            sequence=1,
            event_type="x",
            to_status=AutonomousRunStatus.QUEUED,
            to_phase=RunPhase.QUEUED,
            reason="arbitrary private answer",
        )
    with pytest.raises(ValueError):
        RunEventRecord(
            run_id="run-1",
            sequence=1,
            event_type="x",
            to_status=AutonomousRunStatus.QUEUED,
            to_phase=RunPhase.QUEUED,
            reason="queued",
            metadata={"status": "arbitrary full answer"},
        )
    state = make_run()
    with pytest.raises(TypeError):
        state.metadata["raw"] = "trajectory text"


def test_recovery_scan_rejects_zero_limit_and_returns_cursor_for_physical_pages() -> None:
    table = StubDynamoTable()
    for index in range(3):
        table.items.append(DynamoDBAutonomousRunRepository._item("STATE", make_run(f"run-{index}")))
    repository = DynamoDBAutonomousRunRepository(table=table)
    with pytest.raises(ValueError):
        repository.scan_recoverable(limit=0)
