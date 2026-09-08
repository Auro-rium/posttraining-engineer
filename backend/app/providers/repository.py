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

    def __init__(
        self,
        *,
        table_name: str | None = None,
        table: Any | None = None,
        region_name: str | None = None,
        resource: Any | None = None,
    ) -> None:
        if table is None and not table_name:
            raise ValueError("table_name is required when table is not supplied")
        self.table_name = table_name
        self.region_name = region_name
        self._table = table
        self._resource = resource

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

    def get_run(self, run_id: str) -> RunRecord | None:
        response = self._table_or_create().get_item(
            Key={"pk": self._pk(run_id), "sk": self.STATE_SK},
            ConsistentRead=True,
        )
        item = response.get("Item")
        return self._decode_run(item) if isinstance(item, Mapping) else None

    def update_run(
        self,
        run_id: str,
        *,
        expected_state_version: int,
        patch: Mapping[str, Any] | None = None,
        status: str | None = None,
        phase: str | None = None,
    ) -> RunRecord:
        current = self.get_run(run_id)
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
        refreshed = self.get_run(run_id)
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


# Stable descriptive alias for the durable state boundary.
DynamoDBStateRepository = DynamoDBRunRepository
