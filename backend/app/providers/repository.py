"""Durable DynamoDB run state and append-only event repository.

Run updates use an optimistic ``state_version`` condition.  A worker that
resumes with stale state cannot overwrite a newer transition, which is the
important correctness property for continuous post-training restarts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from app.posttraining.run_history import (
    MAX_RUNS,
    RunHistoryRecord,
    RunLimitExceeded,
)

__all__ = [
    "ConcurrentUpdateError",
    "DynamoDBRunRepository",
    "DynamoDBStateRepository",
    "OptionalDependencyError",
    "RepositoryError",
    "RunAlreadyExistsError",
    "RunEvent",
    "RunHistoryRecord",
    "RunLimitExceeded",
    "RunRecord",
    "RunRepository",
]


class OptionalDependencyError(RuntimeError):
    """Raised when an AWS-backed adapter is used without its optional SDK."""


class RepositoryError(RuntimeError):
    """Base class for durable repository failures."""


class ConcurrentUpdateError(RepositoryError):
    """The caller's expected state version is no longer current."""


class RunAlreadyExistsError(RepositoryError):
    """A run with the same id already exists."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(slots=True)
class RunRecord:
    run_id: str
    state_version: int = 0
    status: str = "running"
    phase: str = "initialized"
    data: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)


@dataclass(slots=True)
class RunEvent:
    run_id: str
    event_id: str = field(default_factory=lambda: uuid4().hex)
    event_type: str = "state.changed"
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)
    state_version: int | None = None


class RunRepository(Protocol):
    def create_run(self, run: RunRecord) -> RunRecord: ...

    def get_run(self, run_id: str) -> RunRecord | None: ...

    def update_run(
        self,
        run_id: str,
        *,
        expected_state_version: int,
        patch: Mapping[str, Any] | None = None,
        status: str | None = None,
        phase: str | None = None,
    ) -> RunRecord: ...

    def append_event(self, event: RunEvent) -> RunEvent: ...

    def list_events(self, run_id: str) -> list[RunEvent]: ...


class DynamoDBRunRepository:
    """DynamoDB implementation using a boto3 Table-like object.

    ``table`` is injectable for tests and for applications that already own a
    configured boto3 resource.  Without one, boto3 is imported lazily only on
    the first operation.
    """

    STATE_SK = "STATE"
    HISTORY_PK = "HISTORY"
    HISTORY_COUNTER_SK = "COUNTER"
    HISTORY_RUN_SK_PREFIX = "RUN#"

    def __init__(
        self,
        *,
        table_name: str | None = None,
        table: Any | None = None,
        region_name: str | None = None,
        resource: Any | None = None,
        client: Any | None = None,
    ) -> None:
        if table is None and not table_name:
            raise ValueError("table_name is required when table is not supplied")
        self.table_name = table_name
        self.region_name = region_name
        self._table = table
        self._resource = resource
        self._client = client

    def _table_or_create(self) -> Any:
        if self._table is not None:
            return self._table
        try:
            import boto3  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise OptionalDependencyError("Install boto3 to use DynamoDBRunRepository") from exc
        if self._resource is None:
            self._resource = boto3.resource("dynamodb", region_name=self.region_name)
        self._table = self._resource.Table(self.table_name)
        return self._table

    @staticmethod
    def _pk(run_id: str) -> str:
        if not run_id.strip():
            raise ValueError("run_id must not be empty")
        return f"RUN#{run_id}"

    @classmethod
    def _history_key(cls, run_id: str) -> dict[str, str]:
        if not run_id.strip():
            raise ValueError("run_id must not be empty")
        return {"pk": cls.HISTORY_PK, "sk": f"{cls.HISTORY_RUN_SK_PREFIX}{run_id}"}

    @classmethod
    def _history_counter_key(cls) -> dict[str, str]:
        return {"pk": cls.HISTORY_PK, "sk": cls.HISTORY_COUNTER_SK}

    def _transaction_client(self) -> Any:
        if self._client is not None:
            return self._client
        table = self._table_or_create()
        metadata = getattr(table, "meta", None)
        client = getattr(metadata, "client", None)
        if client is None and self._resource is not None:
            resource_metadata = getattr(self._resource, "meta", None)
            client = getattr(resource_metadata, "client", None)
        if client is None:
            raise RepositoryError(
                "DynamoDBRunRepository requires a DynamoDB client for history transactions"
            )
        self._client = client
        return client

    @staticmethod
    def _client_value(value: Any) -> dict[str, str | bool]:
        """Encode the scalar values used by the low-level DynamoDB client."""

        if isinstance(value, bool):
            return {"BOOL": value}
        if isinstance(value, int):
            return {"N": str(value)}
        if isinstance(value, float):
            return {"N": str(value)}
        return {"S": str(value)}

    @classmethod
    def _history_item(cls, record: RunHistoryRecord) -> dict[str, Any]:
        serialized = record.model_dump(mode="json")
        item: dict[str, Any] = {
            "pk": {"S": cls.HISTORY_PK},
            "sk": {"S": f"{cls.HISTORY_RUN_SK_PREFIX}{record.run_id}"},
            "entity": {"S": "run_history"},
            "run_id": {"S": record.run_id},
            "run_number": {"N": str(record.run_number)},
            "payload": {"S": _json(serialized)},
            "created_at": {"S": serialized["created_at"]},
            "updated_at": {"S": serialized["updated_at"]},
        }
        return item

    @staticmethod
    def _history_counter_count(item: Mapping[str, Any] | None) -> int:
        if not isinstance(item, Mapping):
            return 0
        value = item.get("run_count", 0)
        if isinstance(value, Mapping) and "N" in value:
            value = value["N"]
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _run_item(self, run: RunRecord) -> dict[str, Any]:
        return {
            "pk": self._pk(run.run_id),
            "sk": self.STATE_SK,
            "entity": "run",
            "run_id": run.run_id,
            "state_version": int(run.state_version),
            "status": run.status,
            "phase": run.phase,
            "payload": _json(run.data),
            "created_at": run.created_at,
            "updated_at": run.updated_at,
        }

    @staticmethod
    def _conditional_failure(exc: BaseException) -> bool:
        response = getattr(exc, "response", None)
        if isinstance(response, Mapping):
            error = response.get("Error")
            if (
                isinstance(error, Mapping)
                and error.get("Code") == "ConditionalCheckFailedException"
            ):
                return True
        return exc.__class__.__name__ in {"ConditionalCheckFailedException", "ConditionalError"}

    @staticmethod
    def _decode_run(item: Mapping[str, Any]) -> RunRecord:
        payload = item.get("payload", "{}")
        if isinstance(payload, str):
            parsed = json.loads(payload)
        else:
            parsed = dict(payload) if isinstance(payload, Mapping) else {}
        return RunRecord(
            run_id=str(item["run_id"]),
            state_version=int(item.get("state_version", 0)),
            status=str(item.get("status", "running")),
            phase=str(item.get("phase", "initialized")),
            data=dict(parsed),
            created_at=str(item.get("created_at", "")),
            updated_at=str(item.get("updated_at", "")),
        )

    @staticmethod
    def _decode_history(item: Mapping[str, Any]) -> RunHistoryRecord:
        payload = item.get("payload", "{}")
        if isinstance(payload, str):
            parsed: Any = json.loads(payload)
        elif isinstance(payload, Mapping):
            parsed = dict(payload)
        else:
            parsed = {}
        return RunHistoryRecord.model_validate(parsed)

    def _get_state_run(self, run_id: str) -> RunRecord | None:
        response = self._table_or_create().get_item(
            Key={"pk": self._pk(run_id), "sk": self.STATE_SK},
            ConsistentRead=True,
        )
        item = response.get("Item")
        return self._decode_run(item) if isinstance(item, Mapping) else None

    def _get_history_item(self, run_id: str) -> Mapping[str, Any] | None:
        response = self._table_or_create().get_item(
            Key=self._history_key(run_id),
            ConsistentRead=True,
        )
        item = response.get("Item")
        return item if isinstance(item, Mapping) else None

    def _get_history_counter(self) -> Mapping[str, Any] | None:
        response = self._table_or_create().get_item(
            Key=self._history_counter_key(),
            ConsistentRead=True,
        )
        item = response.get("Item")
        return item if isinstance(item, Mapping) else None

    @staticmethod
    def _transaction_failure_reason(exc: BaseException) -> str | None:
        response = getattr(exc, "response", None)
        if not isinstance(response, Mapping):
            return None
        reasons = response.get("CancellationReasons")
        if not isinstance(reasons, list):
            return None
        if reasons and isinstance(reasons[0], Mapping):
            if reasons[0].get("Code") == "ConditionalCheckFailed":
                return "limit"
        if len(reasons) > 1 and isinstance(reasons[1], Mapping):
            if reasons[1].get("Code") == "ConditionalCheckFailed":
                return "duplicate"
        return None

    def reserve_run(
        self, record: RunHistoryRecord, *, max_runs: int = MAX_RUNS
    ) -> RunHistoryRecord:
        """Atomically reserve one history slot and persist its immutable record.

        The counter update and record put share one DynamoDB transaction.  The
        counter condition is evaluated by DynamoDB, so two coordinators cannot
        both observe the final available slot and create a sixth run.
        """

        if not 1 <= max_runs <= MAX_RUNS:
            raise ValueError(f"max_runs must be between 1 and {MAX_RUNS}")
        if record.run_number > max_runs:
            raise RunLimitExceeded(f"maximum of {max_runs} runs reached")

        table_name = self.table_name or getattr(self._table_or_create(), "name", "")
        counter_key = self._history_counter_key()
        operations: list[dict[str, Any]] = [
            {
                "Update": {
                    "TableName": table_name,
                    "Key": {
                        "pk": self._client_value(counter_key["pk"]),
                        "sk": self._client_value(counter_key["sk"]),
                    },
                    "UpdateExpression": (
                        "SET #entity = :entity, #run_count = "
                        "if_not_exists(#run_count, :zero) + :one"
                    ),
                    "ConditionExpression": (
                        "attribute_not_exists(#run_count) OR #run_count < :max_runs"
                    ),
                    "ExpressionAttributeNames": {"#entity": "entity", "#run_count": "run_count"},
                    "ExpressionAttributeValues": {
                        ":entity": self._client_value("run_history_counter"),
                        ":zero": self._client_value(0),
                        ":one": self._client_value(1),
                        ":max_runs": self._client_value(max_runs),
                    },
                }
            },
            {
                "Put": {
                    "TableName": table_name,
                    "Item": self._history_item(record),
                    "ConditionExpression": (
                        "attribute_not_exists(#pk) AND attribute_not_exists(#sk)"
                    ),
                    "ExpressionAttributeNames": {"#pk": "pk", "#sk": "sk"},
                }
            },
        ]
        try:
            self._transaction_client().transact_write_items(TransactItems=operations)
        except Exception as exc:
            reason = self._transaction_failure_reason(exc)
            if reason == "limit":
                raise RunLimitExceeded(f"maximum of {max_runs} runs reached") from exc
            if reason == "duplicate":
                raise RunAlreadyExistsError(f"Run already exists: {record.run_id}") from exc

            # Some test doubles and older botocore versions omit cancellation
            # reasons.  These reads only classify the failed transaction; they
            # do not participate in the reservation itself.
            if self._get_history_item(record.run_id) is not None:
                raise RunAlreadyExistsError(f"Run already exists: {record.run_id}") from exc
            if self._history_counter_count(self._get_history_counter()) >= max_runs:
                raise RunLimitExceeded(f"maximum of {max_runs} runs reached") from exc
            raise
        return record

    def create_run(self, run: RunRecord) -> RunRecord:
        if not run.run_id.strip():
            raise ValueError("run_id must not be empty")
        try:
            self._table_or_create().put_item(
                Item=self._run_item(run),
                ConditionExpression="attribute_not_exists(#pk)",
                ExpressionAttributeNames={"#pk": "pk"},
            )
        except Exception as exc:
            if self._conditional_failure(exc):
                raise RunAlreadyExistsError(f"Run already exists: {run.run_id}") from exc
            raise
        return run

    def get_run(self, run_id: str) -> RunRecord | RunHistoryRecord | None:
        history_item = self._get_history_item(run_id)
        if history_item is not None:
            return self._decode_history(history_item)
        return self._get_state_run(run_id)

    def update_run(
        self,
        run_id: str,
        *,
        expected_state_version: int,
        patch: Mapping[str, Any] | None = None,
        status: str | None = None,
        phase: str | None = None,
    ) -> RunRecord:
        current = self._get_state_run(run_id)
        if current is None:
            raise RepositoryError(f"Run does not exist: {run_id}")
        if current.state_version != expected_state_version:
            raise ConcurrentUpdateError(
                f"Run {run_id} is at version {current.state_version}, "
                f"expected {expected_state_version}"
            )
        next_data = dict(current.data)
        next_data.update(patch or {})
        next_status = status if status is not None else current.status
        next_phase = phase if phase is not None else current.phase
        updated_at = _now()
        kwargs: dict[str, Any] = {
            "Key": {"pk": self._pk(run_id), "sk": self.STATE_SK},
            "UpdateExpression": (
                "SET #payload = :payload, #status = :status, #phase = :phase, "
                "#updated_at = :updated_at, #state_version = #state_version + :one"
            ),
            "ConditionExpression": "attribute_exists(#pk) AND #state_version = :expected",
            "ExpressionAttributeNames": {
                "#pk": "pk",
                "#payload": "payload",
                "#status": "status",
                "#phase": "phase",
                "#updated_at": "updated_at",
                "#state_version": "state_version",
            },
            "ExpressionAttributeValues": {
                ":payload": _json(next_data),
                ":status": next_status,
                ":phase": next_phase,
                ":updated_at": updated_at,
                ":expected": int(expected_state_version),
                ":one": 1,
            },
            "ReturnValues": "ALL_NEW",
        }
        try:
            response = self._table_or_create().update_item(**kwargs)
        except Exception as exc:
            if self._conditional_failure(exc):
                raise ConcurrentUpdateError(f"Stale state update for run {run_id}") from exc
            raise
        attributes = response.get("Attributes")
        if isinstance(attributes, Mapping):
            return self._decode_run(attributes)
        refreshed = self._get_state_run(run_id)
        if refreshed is None:  # pragma: no cover - a provider consistency failure
            raise RepositoryError(f"Run disappeared after update: {run_id}")
        return refreshed

    def append_event(self, event: RunEvent) -> RunEvent:
        if not event.run_id.strip() or not event.event_id.strip():
            raise ValueError("run_id and event_id must not be empty")
        item: dict[str, Any] = {
            "pk": self._pk(event.run_id),
            "sk": f"EVENT#{event.created_at}#{event.event_id}",
            "entity": "event",
            "run_id": event.run_id,
            "event_id": event.event_id,
            "event_type": event.event_type,
            "payload": _json(event.payload),
            "created_at": event.created_at,
        }
        if event.state_version is not None:
            item["state_version"] = int(event.state_version)
        try:
            self._table_or_create().put_item(
                Item=item,
                ConditionExpression="attribute_not_exists(#sk)",
                ExpressionAttributeNames={"#sk": "sk"},
            )
        except Exception as exc:
            if self._conditional_failure(exc):
                raise RepositoryError(f"Event already exists: {event.event_id}") from exc
            raise
        return event

    @staticmethod
    def _decode_event(item: Mapping[str, Any]) -> RunEvent:
        payload = item.get("payload", "{}")
        parsed = json.loads(payload) if isinstance(payload, str) else payload
        return RunEvent(
            run_id=str(item["run_id"]),
            event_id=str(item["event_id"]),
            event_type=str(item.get("event_type", "state.changed")),
            payload=dict(parsed) if isinstance(parsed, Mapping) else {},
            created_at=str(item.get("created_at", "")),
            state_version=(
                int(item["state_version"])
                if item.get("state_version") is not None
                else None
            ),
        )

    def list_events(self, run_id: str) -> list[RunEvent]:
        response = self._table_or_create().query(
            KeyConditionExpression="#pk = :pk AND begins_with(#sk, :prefix)",
            ExpressionAttributeNames={"#pk": "pk", "#sk": "sk"},
            ExpressionAttributeValues={":pk": self._pk(run_id), ":prefix": "EVENT#"},
            ScanIndexForward=True,
        )
        items = response.get("Items", [])
        return [
            self._decode_event(item)
            for item in items
            if isinstance(item, Mapping) and item.get("entity") == "event"
        ]

    def list_runs(self, *, limit: int = MAX_RUNS) -> list[RunHistoryRecord]:
        """Return the newest persisted history records in run-number order."""

        if not 1 <= limit <= MAX_RUNS:
            raise ValueError(f"limit must be between 1 and {MAX_RUNS}")
        table = self._table_or_create()
        items: list[Mapping[str, Any]] = []
        query_kwargs: dict[str, Any] = {
            "KeyConditionExpression": "#pk = :pk AND begins_with(#sk, :prefix)",
            "ExpressionAttributeNames": {"#pk": "pk", "#sk": "sk"},
            "ExpressionAttributeValues": {
                ":pk": self.HISTORY_PK,
                ":prefix": self.HISTORY_RUN_SK_PREFIX,
            },
            "ScanIndexForward": True,
        }
        while True:
            response = table.query(**query_kwargs)
            page = response.get("Items", [])
            if isinstance(page, list):
                items.extend(item for item in page if isinstance(item, Mapping))
            last_key = response.get("LastEvaluatedKey")
            if not isinstance(last_key, Mapping):
                break
            query_kwargs["ExclusiveStartKey"] = dict(last_key)

        records = [
            self._decode_history(item)
            for item in items
            if item.get("entity") == "run_history"
        ]
        records.sort(key=lambda record: (record.run_number, record.created_at, record.run_id))
        return records[-limit:]


# Stable descriptive alias for the durable state boundary.
DynamoDBStateRepository = DynamoDBRunRepository
