from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from app.autonomous.models import (
    AutonomousRunState,
    AutonomousRunStatus,
    ExperimentRecord,
    RunEventRecord,
    RunOperation,
    RunOperationStatus,
    RunPhase,
    validate_event_reason,
)
from app.autonomous.repository import (
    ApprovalAlreadyConsumedError,
    ConcurrentUpdateError,
    DynamoDBAutonomousRunRepository,
    IdempotencyKeyConflictError,
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


def make_approved_queued_run(run_id: str = "run-1") -> AutonomousRunState:
    state = make_run(run_id).model_copy(
        update={
            "status": AutonomousRunStatus.QUEUED,
            "phase": RunPhase.QUEUED,
            "approval_digest": "c" * 64,
            "approval_consumed": True,
        }
    )
    return AutonomousRunState.model_validate(state.model_dump(mode="python"))


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


def test_idempotency_claim_binds_request_and_persists_safe_response() -> None:
    repository = InMemoryAutonomousRunRepository()
    digest = "d" * 64

    pending, created = repository.claim_idempotency("prepare", "key-1", digest)
    replay, replay_created = repository.claim_idempotency("prepare", "key-1", digest)

    assert created is True
    assert replay_created is False
    assert pending.request_digest == digest
    assert pending.response is None
    assert replay == pending

    completed = repository.complete_idempotency(
        "prepare", "key-1", digest, {"run_id": "run-1", "status": "PREPARED"}
    )
    restored = repository.get_idempotency("prepare", "key-1")
    assert completed.response == {"run_id": "run-1", "status": "PREPARED"}
    assert restored == completed
    assert repository.claim_idempotency("prepare", "key-1", digest) == (completed, False)

    with pytest.raises(IdempotencyKeyConflictError):
        repository.claim_idempotency("prepare", "key-1", "e" * 64)


def test_idempotency_completion_requires_prior_matching_claim() -> None:
    repository = InMemoryAutonomousRunRepository()
    repository.claim_idempotency("start", "key-1", "a" * 64)

    with pytest.raises(IdempotencyKeyConflictError):
        repository.complete_idempotency("start", "key-1", "b" * 64, {"status": "QUEUED"})


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


def test_update_state_is_validated_optimistic_and_preserves_repository_fields() -> None:
    repository = InMemoryAutonomousRunRepository()
    repository.create(make_run())

    updated = repository.update_state(
        "run-1",
        expected_version=0,
        updates={"current_dataset_uri": "s3://artifacts/data.jsonl?versionId=v1"},
    )

    assert updated.current_dataset_uri == "s3://artifacts/data.jsonl?versionId=v1"
    assert updated.version == 1
    assert updated.event_sequence == 0
    with pytest.raises(ConcurrentUpdateError):
        repository.update_state("run-1", expected_version=0, updates={"safe_stop_requested": True})
    with pytest.raises(ValueError, match="repository-owned"):
        repository.update_state("run-1", expected_version=1, updates={"run_id": "other"})
    with pytest.raises(ValueError, match="unknown"):
        repository.update_state("run-1", expected_version=1, updates={"not_a_field": True})


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", AutonomousRunStatus.RUNNING),
        ("phase", RunPhase.TRAINING),
        ("approval_scope", {"max_cost_usd": 1.0}),
        ("benchmark_id", "other"),
        ("benchmark_suite", "other"),
        ("benchmark_version", "other"),
        ("benchmark_seed", 99),
        ("max_experiments", 1),
        ("approved_budget_usd", 1.0),
    ],
)
def test_update_state_rejects_lifecycle_and_approval_scope_mutations(
    field: str, value: object
) -> None:
    repository = InMemoryAutonomousRunRepository()
    repository.create(make_run())
    with pytest.raises(ValueError, match="state patch"):
        repository.update_state("run-1", expected_version=0, updates={field: value})


@pytest.mark.parametrize(
    "method,reason",
    [("request_cancel", "cancel requested"), ("request_safe_stop", "safe stop requested")],
)
def test_control_requests_are_atomic_and_append_lifecycle_event(method: str, reason: str) -> None:
    repository = InMemoryAutonomousRunRepository()
    repository.create(make_approved_queued_run())
    requested = getattr(repository, method)("run-1", expected_version=0)
    assert requested.version == 1
    assert requested.event_sequence == 1
    assert getattr(
        requested,
        "cancellation_requested" if method == "request_cancel" else "safe_stop_requested",
    )
    events = repository.list_events("run-1")
    assert len(events) == 1
    assert events[0].reason == reason


def test_control_request_is_optimistic() -> None:
    repository = InMemoryAutonomousRunRepository()
    repository.create(make_approved_queued_run())
    repository.request_cancel("run-1", expected_version=0)
    with pytest.raises(ConcurrentUpdateError):
        repository.request_safe_stop("run-1", expected_version=0)


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
    repository.create(make_approved_queued_run("queued"))
    repository.create(make_approved_queued_run("running"))
    repository.transition(
        "running", expected_version=0, status=AutonomousRunStatus.RUNNING,
        phase=RunPhase.BASELINE, reason="worker started"
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


def test_recovery_excludes_prepared_unapproved_runs_and_supports_cursor_pages() -> None:
    repository = InMemoryAutonomousRunRepository()
    repository.create(make_run("prepared"))
    for run_id in ("queued-a", "queued-b", "queued-c"):
        repository.create(make_approved_queued_run(run_id))
    first = repository.scan_recoverable(limit=2)
    assert [state.run_id for state in first] == ["queued-a", "queued-b"]
    assert first.next_cursor == {"run_id": "queued-b"}
    second = repository.scan_recoverable(limit=2, cursor=first.next_cursor)
    assert [state.run_id for state in second] == ["queued-c"]
    assert second.next_cursor is None


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


def test_operation_status_is_monotonic_and_same_status_updates_are_idempotent() -> None:
    repository = InMemoryAutonomousRunRepository()
    repository.create(make_run())
    operation = RunOperation(
        operation_key="monotonic",
        run_id="run-1",
        experiment_number=1,
        phase=RunPhase.TRAINING,
        provider_name="job",
    )
    repository.put_operation_intent(operation)
    running = repository.record_operation_result(
        "run-1", "monotonic", status=RunOperationStatus.RUNNING, provider_id="job-1"
    )
    assert running.status is RunOperationStatus.RUNNING
    with pytest.raises(OperationAlreadyExistsError):
        repository.record_operation_result(
            "run-1", "monotonic", status=RunOperationStatus.INTENT, provider_id="job-1"
        )
    succeeded = repository.record_operation_result(
        "run-1", "monotonic", status=RunOperationStatus.SUCCEEDED, provider_id="job-1",
        result={"job_name": "job-1", "refs": ["artifact://one"]},
    )
    assert repository.record_operation_result(
        "run-1", "monotonic", status=RunOperationStatus.SUCCEEDED, provider_id="job-1",
        result={"job_name": "job-1", "refs": ["artifact://one"]},
    ) == succeeded


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
            reason="queued",
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


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RunOperation(
            operation_key="unsafe", run_id="run-1", experiment_number=1,
            phase=RunPhase.TRAINING, provider_name="job", result={"raw_prompt": "secret"}
        ),
        lambda: ExperimentRecord(
            experiment_number=1, training_config={"objective": "DPO", "held_out": "task text"}
        ),
        lambda: AutonomousRunState.model_validate(
            {
                **make_run().model_dump(mode="python"),
                "current_hypothesis": {"completion": "raw response"},
            }
        ),
    ],
)
def test_durable_payloads_reject_raw_content_and_unsupported_training(factory: object) -> None:
    with pytest.raises(ValueError):
        cast(Any, factory)()


def test_durable_payloads_are_bounded_and_json_scalar_only() -> None:
    with pytest.raises(ValueError):
        RunOperation(
            operation_key="unsafe", run_id="run-1", experiment_number=1,
            phase=RunPhase.TRAINING, provider_name="job", result={"nested": object()}
        )
    with pytest.raises(ValueError):
        AutonomousRunState.model_validate(
            {
                **make_run().model_dump(mode="python"),
                "current_hypothesis": {"hypothesis": {"statement": "x" * 5000}},
            }
        )


def test_run_state_persists_complete_live_recovery_fields() -> None:
    state = make_run().model_copy(
        update={
            "base_checkpoint_uri": "s3://artifacts/base.tar.gz?versionId=v1",
            "base_checkpoint_sha256": "1" * 64,
            "champion_checkpoint_uri": "s3://artifacts/champion.tar.gz?versionId=v2",
            "champion_checkpoint_sha256": "2" * 64,
            "champion_score": 0.75,
            "benchmark_suite": "AgentGym/AgentEval",
            "benchmark_version": "agent-eval-v1",
            "benchmark_seed": 7,
            "current_hypothesis": {
                "hypothesis_id": "hyp-1",
                "evidence_ids": ["ev-1"],
            },
            "current_dataset_uri": "s3://artifacts/data.jsonl?versionId=v3",
            "current_dataset_sha256": "3" * 64,
            "current_training_job_id": "training-job-1",
            "current_evaluation_job_id": "evaluation-job-1",
            "current_candidate_uri": "s3://artifacts/candidate.tar.gz?versionId=v4",
            "current_candidate_sha256": "4" * 64,
            "approval_scope": {"max_experiments": 5, "max_cost_usd": 25.0},
            "approval_expires_at": datetime.now(UTC) + timedelta(minutes=5),
        }
    )

    restored = AutonomousRunState.model_validate(state.model_dump(mode="python"))
    assert restored.current_training_job_id == "training-job-1"
    assert restored.current_hypothesis == {
        "hypothesis_id": "hyp-1",
        "evidence_ids": ("ev-1",),
    }
    assert restored.approval_scope["max_experiments"] == 5
    with pytest.raises(TypeError):
        restored.current_hypothesis["hypothesis_id"] = "changed"


class StubDynamoTable:
    """Native resource-table stub with keyset pagination and captured writes."""

    name = "autonomous-runs"

    def __init__(self) -> None:
        self.items: list[dict[str, object]] = []
        self.puts: list[dict[str, object]] = []
        self.queries: list[dict[str, object]] = []
        self.get_item_calls: list[dict[str, object]] = []
        self.page_size = 1

    def put_item(self, **kwargs: object) -> None:
        self.puts.append(kwargs)

    def get_item(self, **kwargs: object) -> dict[str, object]:
        self.get_item_calls.append(kwargs)
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


def test_dynamo_idempotency_claim_and_completion_are_conditional_and_replayable() -> None:
    class ConditionalError(Exception):
        pass

    class IdempotencyTable:
        name = "autonomous-runs"

        def __init__(self) -> None:
            self.items: dict[tuple[str, str], dict[str, object]] = {}
            self.puts: list[dict[str, object]] = []

        def get_item(self, **kwargs: object) -> dict[str, object]:
            key = cast(dict[str, str], kwargs["Key"])
            item = self.items.get((key["pk"], key["sk"]))
            return {"Item": item} if item is not None else {}

        def put_item(self, **kwargs: object) -> None:
            self.puts.append(kwargs)
            item = cast(dict[str, object], kwargs["Item"])
            identity = (str(item["pk"]), str(item["sk"]))
            current = self.items.get(identity)
            condition = str(kwargs.get("ConditionExpression", ""))
            if condition == "attribute_not_exists(pk)" and current is not None:
                raise ConditionalError("conditional put failed")
            if condition.startswith("request_digest =") and (
                current is None
                or current.get("request_digest")
                != cast(dict[str, str], kwargs["ExpressionAttributeValues"])[":request_digest"]
                or current.get("state")
                != cast(dict[str, str], kwargs["ExpressionAttributeValues"])[":pending"]
            ):
                raise ConditionalError("conditional completion failed")
            self.items[identity] = item

    table = IdempotencyTable()
    repository = DynamoDBAutonomousRunRepository(table=table, client=StubDynamoClient())
    digest = "f" * 64

    first, created = repository.claim_idempotency("start", "key-1", digest)
    replay, replay_created = repository.claim_idempotency("start", "key-1", digest)
    assert created is True
    assert replay_created is False
    assert replay == first
    with pytest.raises(IdempotencyKeyConflictError):
        repository.claim_idempotency("start", "key-1", "e" * 64)

    completed = repository.complete_idempotency(
        "start", "key-1", digest, {"run_id": "run-1", "status": "QUEUED"}
    )
    assert repository.get_idempotency("start", "key-1") == completed
    conditions = [put["ConditionExpression"] for put in table.puts]
    assert conditions[0] == "attribute_not_exists(pk)"
    assert conditions[1] == "attribute_not_exists(pk)"
    assert conditions[-1] == "request_digest = :request_digest AND #state = :pending"


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
                    event_type="state.transitioned",
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
                event_type="state.transitioned",
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
    table.items.append(DynamoDBAutonomousRunRepository._item("STATE", make_approved_queued_run()))
    repository = DynamoDBAutonomousRunRepository(table=table, client=StubDynamoClient())
    repository.claim_lease("run-1", "worker", now=datetime(2026, 9, 8, tzinfo=UTC))
    values = table.puts[-1]["ExpressionAttributeValues"]
    assert values == {":version": 0}


def test_dynamo_update_state_uses_validated_conditional_put() -> None:
    table = StubDynamoTable()
    table.items.append(DynamoDBAutonomousRunRepository._item("STATE", make_run()))
    client = StubDynamoClient()
    repository = DynamoDBAutonomousRunRepository(table=table, client=client)

    updated = repository.update_state(
        "run-1", expected_version=0, updates={"cancellation_requested": True}
    )

    assert updated.cancellation_requested is True
    assert updated.version == 1
    assert len(client.transactions) == 1
    writes = cast(list[dict[str, object]], client.transactions[-1]["TransactItems"])
    assert "Put" in writes[0]
    put = cast(dict[str, object], writes[0]["Put"])
    assert put["ConditionExpression"] == "version = :version AND cancellation_requested = :false"


def test_dynamo_append_event_reads_exact_sequence_consistently() -> None:
    table = StubDynamoTable()
    run = make_run()
    table.items.append(DynamoDBAutonomousRunRepository._item("STATE", run))
    client = StubDynamoClient()
    repository = DynamoDBAutonomousRunRepository(table=table, client=client)
    event = RunEventRecord(
        run_id="run-1", sequence=1, event_type="state.transitioned", to_status=run.status,
        to_phase=run.phase, reason="queued"
    )
    table.items.append(DynamoDBAutonomousRunRepository._item("EVENT#00000000000000000001", event))
    returned = repository.append_event("run-1", event_type="state.transitioned", reason="queued")
    assert returned.sequence == 1
    assert table.queries == []
    call = cast(dict[str, object], table.get_item_calls[-1])
    key = cast(dict[str, object], call["Key"])
    assert key["sk"] == "EVENT#00000000000000000001"
    assert call["ConsistentRead"] is True


def test_dynamo_release_lease_maps_conditional_race_to_repository_error() -> None:
    table = StubDynamoTable()
    state = AutonomousRunState.model_validate(
        {
            **make_run().model_dump(mode="python"),
            "lease_owner": "worker-a",
            "lease_expires_at": datetime.now(UTC) + timedelta(minutes=1),
        }
    )
    table.items.append(DynamoDBAutonomousRunRepository._item("STATE", state))

    def fail(**kwargs: object) -> None:
        del kwargs
        raise type("ConditionalError", (Exception,), {})()

    table.put_item = fail  # type: ignore[method-assign]
    repository = DynamoDBAutonomousRunRepository(table=table)
    with pytest.raises(LeaseConflictError):
        repository.release_lease("run-1", "worker-a")


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
            event_type="state.transitioned",
            to_status=AutonomousRunStatus.QUEUED,
            to_phase=RunPhase.QUEUED,
            reason="raw prompt: secret words",
        )
    with pytest.raises(ValueError):
        RunEventRecord(
            run_id="run-1",
            sequence=1,
            event_type="state.transitioned",
            to_status=AutonomousRunStatus.QUEUED,
            to_phase=RunPhase.QUEUED,
            reason="arbitrary private answer",
        )
    with pytest.raises(ValueError):
        RunEventRecord(
            run_id="run-1",
            sequence=1,
            event_type="state.transitioned",
            to_status=AutonomousRunStatus.QUEUED,
            to_phase=RunPhase.QUEUED,
            reason="queued",
            metadata={"status": "arbitrary full answer"},
        )
    state = make_run()
    with pytest.raises(TypeError):
        state.metadata["raw"] = "trajectory text"


def test_event_reason_accepts_safe_lifecycle_phrases() -> None:
    for reason in (
        "provider request timed out",
        "safe stop requested",
        "run completed",
        "provider job failed",
    ):
        event = RunEventRecord(
            run_id="run-1",
            sequence=1,
            event_type="state.transitioned",
            to_status=AutonomousRunStatus.QUEUED,
            to_phase=RunPhase.QUEUED,
            reason=reason,
        )
        assert event.reason == reason


@pytest.mark.parametrize(
    "reason",
    [
        "provider request timed out",
        "safe stop requested",
        "baseline completed",
        "provider_request_timed_out",
        "SAFE_STOP_REQUESTED",
    ],
)
def test_event_reason_validator_accepts_exact_lifecycle_phrases_and_codes(reason: str) -> None:
    assert validate_event_reason(reason) == reason


@pytest.mark.parametrize(
    "reason",
    [
        "provider said customer Alice",
        "provider request timed out for customer Alice",
        "new arbitrary lifecycle reason",
        "PROVIDER_SAID_CUSTOMER_ALICE",
    ],
)
def test_event_reason_validator_rejects_free_form_or_unallowlisted_reasons(reason: str) -> None:
    with pytest.raises(ValueError, match="reason"):
        validate_event_reason(reason)


def test_event_metadata_requires_finite_numbers_and_allows_opaque_artifact_paths() -> None:
    event = RunEventRecord(
        run_id="run-1",
        sequence=1,
        event_type="state.transitioned",
        to_status=AutonomousRunStatus.QUEUED,
        to_phase=RunPhase.QUEUED,
        reason="queued",
        metadata={
            "artifact_id": "s3://bucket/trajectory-123",
            "cost_usd": "1.25",
            "latency_ms": "10",
        },
    )
    assert event.metadata["artifact_id"] == "s3://bucket/trajectory-123"
    for key, value in (("cost_usd", "NaN"), ("latency_ms", "Infinity")):
        with pytest.raises(ValueError):
            RunEventRecord(
                run_id="run-1",
                sequence=1,
                event_type="state.transitioned",
                to_status=AutonomousRunStatus.QUEUED,
                to_phase=RunPhase.QUEUED,
                reason="queued",
                metadata={key: value},
            )


def test_recovery_final_page_does_not_advertise_spurious_cursor() -> None:
    table = StubDynamoTable()
    table.items.append(DynamoDBAutonomousRunRepository._item("STATE", make_approved_queued_run()))
    repository = DynamoDBAutonomousRunRepository(table=table)
    page = repository.scan_recoverable(limit=1)
    assert len(page.items) == 1
    assert page.next_cursor is None


def test_recovery_scan_rejects_zero_limit_and_returns_cursor_for_physical_pages() -> None:
    table = StubDynamoTable()
    for index in range(3):
        table.items.append(
            DynamoDBAutonomousRunRepository._item(
                "STATE", make_approved_queued_run(f"run-{index}")
            )
        )
    repository = DynamoDBAutonomousRunRepository(table=table)
    with pytest.raises(ValueError):
        repository.scan_recoverable(limit=0)
