"""In-memory and DynamoDB repositories for autonomous run state.

The in-memory adapter is used by contract tests and local development.  The
DynamoDB adapter keeps state, event, experiment, and operation records under a
single run partition and uses conditional writes/transactions for concurrency
and at-least-once provider reconciliation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from threading import RLock
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

from .models import (
    AutonomousRunState,
    AutonomousRunStatus,
    ExperimentRecord,
    RunEventRecord,
    RunOperation,
    RunOperationStatus,
    RunPhase,
    copy_for_storage,
    utc_now,
)

ModelT = TypeVar("ModelT")


class RepositoryError(RuntimeError):
    """Base error for durable run repository failures."""


class RunNotFoundError(RepositoryError):
    pass


class RunAlreadyExistsError(RepositoryError):
    pass


class ConcurrentUpdateError(RepositoryError):
    pass


class ApprovalAlreadyConsumedError(RepositoryError):
    pass


class LeaseConflictError(RepositoryError):
    pass


class OperationAlreadyExistsError(RepositoryError):
    pass


class OperationNotFoundError(RepositoryError):
    pass


class IdempotencyKeyConflictError(RepositoryError):
    """An idempotency key is already bound to another request or response."""


@dataclass(frozen=True)
class IdempotencyRecord:
    """Persisted request identity and a safe JSON response, if completed."""

    operation: str
    request_digest: str
    response: Mapping[str, Any] | None = None


class OptionalDependencyError(RepositoryError):
    pass


class Page[T]:
    """Small immutable-ish page value with list-compatible convenience methods."""

    def __init__(
        self,
        items: Sequence[T],
        next_after: int | None = None,
        next_offset: int | None = None,
        next_cursor: Mapping[str, Any] | None = None,
    ):
        self.items = list(items)
        self.next_after = next_after
        self.next_offset = next_offset
        self.next_cursor = dict(next_cursor) if next_cursor is not None else None

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> T:
        return self.items[index]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Page):
            return self.items == other.items
        if isinstance(other, list):
            return self.items == other
        return NotImplemented


EventPage = Page[RunEventRecord]
ExperimentPage = Page[ExperimentRecord]
StatePage = Page[AutonomousRunState]


@runtime_checkable
class AutonomousRunRepository(Protocol):
    def create(self, state: AutonomousRunState) -> AutonomousRunState: ...

    def claim_idempotency(
        self, operation: str, key: str, request_digest: str
    ) -> tuple[IdempotencyRecord, bool]: ...

    def get_idempotency(self, operation: str, key: str) -> IdempotencyRecord | None: ...

    def complete_idempotency(
        self,
        operation: str,
        key: str,
        request_digest: str,
        response: Mapping[str, Any],
    ) -> IdempotencyRecord: ...

    def get(self, run_id: str) -> AutonomousRunState | None: ...

    def update_state(
        self,
        run_id: str,
        *,
        expected_version: int,
        updates: Mapping[str, Any],
    ) -> AutonomousRunState: ...

    def request_cancel(
        self, run_id: str, *, expected_version: int
    ) -> AutonomousRunState: ...

    def request_safe_stop(
        self, run_id: str, *, expected_version: int
    ) -> AutonomousRunState: ...

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

    def append_event(
        self,
        run_id: str,
        *,
        event_type: str,
        reason: str,
        metadata: Mapping[str, str] | None = None,
    ) -> RunEventRecord: ...

    def consume_approval(self, run_id: str, approval_digest: str) -> AutonomousRunState: ...

    def claim_lease(
        self, run_id: str, owner: str, *, now: datetime | None = None, ttl_seconds: int = 60
    ) -> AutonomousRunState: ...

    def renew_lease(
        self, run_id: str, owner: str, *, now: datetime | None = None, ttl_seconds: int = 60
    ) -> AutonomousRunState: ...

    def release_lease(self, run_id: str, owner: str) -> AutonomousRunState: ...

    def scan_recoverable(
        self,
        *,
        now: datetime | None = None,
        limit: int = 100,
        cursor: Mapping[str, Any] | None = None,
    ) -> StatePage: ...

    def put_operation_intent(self, operation: RunOperation) -> RunOperation: ...

    def get_operation(
        self, run_id: str, operation_key: str | None = None
    ) -> RunOperation | None: ...

    def record_operation_result(
        self,
        run_id: str,
        operation_key: str,
        *,
        provider_id: str | None = None,
        status: RunOperationStatus | str,
        result: Mapping[str, Any] | None = None,
    ) -> RunOperation: ...

    def list_events(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 100,
        cursor: Mapping[str, Any] | None = None,
    ) -> EventPage: ...

    def add_experiment(self, run_id: str, experiment: ExperimentRecord) -> ExperimentRecord: ...

    def list_experiments(
        self,
        run_id: str,
        *,
        offset: int = 0,
        limit: int = 100,
        cursor: Mapping[str, Any] | None = None,
    ) -> ExperimentPage: ...


def _ensure_now(value: datetime | None) -> datetime:
    current = value or utc_now()
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("now must include a timezone")
    return current


def _validate_idempotency(operation: str, key: str, request_digest: str) -> None:
    if not isinstance(operation, str) or not operation or len(operation) > 300:
        raise ValueError("idempotency operation must be a non-empty bounded string")
    if not isinstance(key, str) or not key.strip() or len(key) > 200:
        raise ValueError("idempotency key must be a non-empty bounded string")
    if (
        not isinstance(request_digest, str)
        or len(request_digest) != 64
        or any(char not in "0123456789abcdef" for char in request_digest)
    ):
        raise ValueError("idempotency request digest must be a lowercase SHA-256")


def _safe_response(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("idempotency response must be a JSON object")
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        normalized = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("idempotency response must contain safe JSON values") from exc
    if not isinstance(normalized, dict):
        raise ValueError("idempotency response must be a JSON object")
    return normalized


def _ensure_ttl(ttl_seconds: int) -> int:
    if ttl_seconds <= 0:
        raise ValueError("lease ttl_seconds must be positive")
    return ttl_seconds


_OPERATION_STATUS_ORDER = {
    RunOperationStatus.INTENT: 0,
    RunOperationStatus.SUBMITTED: 1,
    RunOperationStatus.RUNNING: 2,
    RunOperationStatus.SUCCEEDED: 3,
    RunOperationStatus.FAILED: 3,
    RunOperationStatus.CANCELLED: 3,
}
_TERMINAL_OPERATION_STATUSES = frozenset(
    {RunOperationStatus.SUCCEEDED, RunOperationStatus.FAILED, RunOperationStatus.CANCELLED}
)
_RECOVERABLE_STATUSES = frozenset(
    {
        AutonomousRunStatus.QUEUED,
        AutonomousRunStatus.RUNNING,
        AutonomousRunStatus.CANCEL_REQUESTED,
        AutonomousRunStatus.SAFE_STOP_REQUESTED,
    }
)


def _recoverable(state: AutonomousRunState, when: datetime) -> bool:
    return (
        state.status in _RECOVERABLE_STATUSES
        and state.approval_consumed
        and not (
            state.lease_owner
            and state.lease_expires_at
            and state.lease_expires_at > when
        )
    )


def _operation_update(
    current: RunOperation,
    *,
    provider_id: str | None,
    status: RunOperationStatus,
    result: Mapping[str, Any] | None,
) -> RunOperation:
    """Apply one monotonic, idempotent provider callback to an intent."""

    normalized_result = dict(result or {})

    def canonical(value: Any) -> Any:
        if isinstance(value, Mapping):
            return tuple(sorted((str(key), canonical(item)) for key, item in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(canonical(item) for item in value)
        return value

    same_result = canonical(current.result) == canonical(normalized_result)
    if _OPERATION_STATUS_ORDER[status] < _OPERATION_STATUS_ORDER[current.status]:
        raise OperationAlreadyExistsError("operation status cannot move backwards")
    effective_provider = provider_id if provider_id is not None else current.provider_id
    if current.provider_id is not None and provider_id not in {None, current.provider_id}:
        raise OperationAlreadyExistsError("operation provider identity cannot change")
    if current.status in _TERMINAL_OPERATION_STATUSES:
        if (
            current.status is status
            and current.provider_id == effective_provider
            and same_result
        ):
            return current
        raise OperationAlreadyExistsError("terminal operation result cannot be changed")
    if (
        current.status is status
        and current.provider_id == effective_provider
        and same_result
    ):
        return current
    return RunOperation.model_validate(
        current.model_copy(
            update={
                "provider_id": effective_provider,
                "status": status,
                "result": normalized_result,
                "version": current.version + 1,
                "updated_at": utc_now(),
            }
        ).model_dump(mode="python")
    )


_PATCH_FORBIDDEN_FIELDS = frozenset(
    {
        "run_id",
        "model_id",
        "checkpoint_revision",
        "checkpoint_id",
        "base_checkpoint_uri",
        "base_checkpoint_sha256",
        "benchmark_manifest_sha256",
        "created_at",
        "version",
        "event_sequence",
        "approval_digest",
        "approval_consumed",
        "approval_consumed_at",
        "lease_owner",
        "lease_expires_at",
        "experiments",
        "updated_at",
        "status",
        "phase",
        "benchmark_id",
        "benchmark_suite",
        "benchmark_version",
        "benchmark_seed",
        "max_experiments",
        "approved_budget_usd",
        "approval_scope",
        "approval_expires_at",
    }
)


def _validated_state_updates(updates: Mapping[str, Any]) -> dict[str, Any]:
    """Reject identity, sequence, approval, lease, and history mutations."""

    normalized = dict(updates)
    unknown = set(normalized) - set(AutonomousRunState.model_fields)
    forbidden = set(normalized) & _PATCH_FORBIDDEN_FIELDS
    if unknown:
        raise ValueError(f"unknown autonomous state fields: {', '.join(sorted(unknown))}")
    if forbidden:
        raise ValueError(
            "state patch cannot mutate repository-owned fields: "
            + ", ".join(sorted(forbidden))
        )
    return normalized


class InMemoryAutonomousRunRepository:
    """Thread-safe repository with the same conditional semantics as DynamoDB."""

    def __init__(self) -> None:
        self._states: dict[str, AutonomousRunState] = {}
        self._events: dict[str, list[RunEventRecord]] = {}
        self._experiments: dict[str, list[ExperimentRecord]] = {}
        self._operations: dict[tuple[str, str], RunOperation] = {}
        self._idempotency: dict[tuple[str, str], IdempotencyRecord] = {}
        self._lock = RLock()

    def claim_idempotency(
        self, operation: str, key: str, request_digest: str
    ) -> tuple[IdempotencyRecord, bool]:
        _validate_idempotency(operation, key, request_digest)
        identity = (operation, key)
        with self._lock:
            current = self._idempotency.get(identity)
            if current is not None:
                if current.request_digest != request_digest:
                    raise IdempotencyKeyConflictError(
                        "idempotency key is bound to a different request"
                    )
                return _copy_idempotency(current), False
            created = IdempotencyRecord(operation=operation, request_digest=request_digest)
            self._idempotency[identity] = created
            return _copy_idempotency(created), True

    def get_idempotency(self, operation: str, key: str) -> IdempotencyRecord | None:
        if not operation or not key:
            raise ValueError("idempotency operation and key are required")
        with self._lock:
            current = self._idempotency.get((operation, key))
            return _copy_idempotency(current) if current is not None else None

    def complete_idempotency(
        self,
        operation: str,
        key: str,
        request_digest: str,
        response: Mapping[str, Any],
    ) -> IdempotencyRecord:
        _validate_idempotency(operation, key, request_digest)
        normalized = _safe_response(response)
        identity = (operation, key)
        with self._lock:
            current = self._idempotency.get(identity)
            if current is None or current.request_digest != request_digest:
                raise IdempotencyKeyConflictError("idempotency claim does not match completion")
            if current.response is not None:
                if _json(current.response) != _json(normalized):
                    raise IdempotencyKeyConflictError("idempotency response cannot be changed")
                return _copy_idempotency(current)
            completed = IdempotencyRecord(
                operation=operation,
                request_digest=request_digest,
                response=normalized,
            )
            self._idempotency[identity] = completed
            return _copy_idempotency(completed)

    def create(self, state: AutonomousRunState) -> AutonomousRunState:
        with self._lock:
            if state.run_id in self._states:
                raise RunAlreadyExistsError(f"run {state.run_id!r} already exists")
            self._states[state.run_id] = copy_for_storage(state)
            self._events[state.run_id] = []
            self._experiments[state.run_id] = []
            return copy_for_storage(state)

    def get(self, run_id: str) -> AutonomousRunState | None:
        with self._lock:
            state = self._states.get(run_id)
            return copy_for_storage(state) if state else None

    def update_state(
        self,
        run_id: str,
        *,
        expected_version: int,
        updates: Mapping[str, Any],
    ) -> AutonomousRunState:
        """Conditionally patch durable state without bypassing validation."""

        with self._lock:
            current = self._require(run_id)
            if current.version != expected_version:
                raise ConcurrentUpdateError(
                    f"run version is {current.version}; expected {expected_version}"
                )
            normalized = _validated_state_updates(updates)
            if set(normalized) <= {"cancellation_requested", "safe_stop_requested"}:
                if len(normalized) != 1 or normalized.get(next(iter(normalized))) is not True:
                    raise ValueError("control patches must request exactly one control flag")
                return self.request_cancel(
                    run_id, expected_version=expected_version
                ) if "cancellation_requested" in normalized else self.request_safe_stop(
                    run_id, expected_version=expected_version
                )
            next_state = AutonomousRunState.model_validate(
                current.model_copy(
                    update={
                        **normalized,
                        "version": current.version + 1,
                        "updated_at": utc_now(),
                    }
                ).model_dump(mode="python")
            )
            return self._store_state(next_state)

    def _request_control(
        self, run_id: str, *, expected_version: int, field: str, reason: str, event_type: str
    ) -> AutonomousRunState:
        with self._lock:
            current = self._require(run_id)
            if current.version != expected_version:
                raise ConcurrentUpdateError(
                    f"run version is {current.version}; expected {expected_version}"
                )
            if getattr(current, field):
                return copy_for_storage(current)
            when = utc_now()
            next_state = AutonomousRunState.model_validate(
                current.model_copy(
                    update={
                        field: True,
                        "version": current.version + 1,
                        "event_sequence": current.event_sequence + 1,
                        "updated_at": when,
                    }
                ).model_dump(mode="python")
            )
            event = RunEventRecord(
                run_id=run_id,
                sequence=next_state.event_sequence,
                event_type=event_type,
                from_status=current.status,
                to_status=current.status,
                from_phase=current.phase,
                to_phase=current.phase,
                reason=reason,
                occurred_at=when,
            )
            self._states[run_id] = copy_for_storage(next_state)
            self._events[run_id].append(copy_for_storage(event))
            return copy_for_storage(next_state)

    def request_cancel(self, run_id: str, *, expected_version: int) -> AutonomousRunState:
        return self._request_control(
            run_id,
            expected_version=expected_version,
            field="cancellation_requested",
            reason="cancel requested",
            event_type="run.cancel_requested",
        )

    def request_safe_stop(self, run_id: str, *, expected_version: int) -> AutonomousRunState:
        return self._request_control(
            run_id,
            expected_version=expected_version,
            field="safe_stop_requested",
            reason="safe stop requested",
            event_type="run.safe_stop_requested",
        )

    def _require(self, run_id: str) -> AutonomousRunState:
        state = self._states.get(run_id)
        if state is None:
            raise RunNotFoundError(f"run {run_id!r} was not found")
        return state

    def _store_state(self, state: AutonomousRunState) -> AutonomousRunState:
        self._states[state.run_id] = copy_for_storage(state)
        return copy_for_storage(state)

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
        with self._lock:
            current = self._require(run_id)
            if current.version != expected_version:
                raise ConcurrentUpdateError(
                    f"run version is {current.version}; expected {expected_version}"
                )
            occurred_at = utc_now()
            next_state = current.model_copy(
                update={
                    "status": status,
                    "phase": phase,
                    "version": current.version + 1,
                    "event_sequence": current.event_sequence + 1,
                    "updated_at": occurred_at,
                }
            )
            # model_copy(update=...) intentionally skips validation, so validate the snapshot.
            next_state = AutonomousRunState.model_validate(next_state.model_dump(mode="python"))
            event = RunEventRecord(
                run_id=run_id,
                sequence=next_state.event_sequence,
                event_type=event_type,
                from_status=current.status,
                to_status=status,
                from_phase=current.phase,
                to_phase=phase,
                reason=reason,
                metadata=dict(metadata or {}),
                occurred_at=occurred_at,
            )
            self._states[run_id] = copy_for_storage(next_state)
            self._events[run_id].append(copy_for_storage(event))
            return copy_for_storage(next_state)

    def consume_approval(self, run_id: str, approval_digest: str) -> AutonomousRunState:
        with self._lock:
            current = self._require(run_id)
            if current.approval_consumed:
                raise ApprovalAlreadyConsumedError("approval packet has already been consumed")
            if not approval_digest.strip():
                raise ValueError("approval_digest must not be blank")
            when = utc_now()
            next_state = current.model_copy(
                update={
                    "approval_digest": approval_digest,
                    "approval_consumed": True,
                    "approval_consumed_at": when,
                    "version": current.version + 1,
                    "event_sequence": current.event_sequence + 1,
                    "updated_at": when,
                }
            )
            next_state = AutonomousRunState.model_validate(next_state.model_dump(mode="python"))
            event = RunEventRecord(
                run_id=run_id,
                sequence=next_state.event_sequence,
                event_type="approval.consumed",
                from_status=current.status,
                to_status=current.status,
                from_phase=current.phase,
                to_phase=current.phase,
                reason="approval packet consumed",
                metadata={"approval_digest": approval_digest},
                occurred_at=when,
            )
            self._states[run_id] = copy_for_storage(next_state)
            self._events[run_id].append(copy_for_storage(event))
            return copy_for_storage(next_state)

    def append_event(
        self,
        run_id: str,
        *,
        event_type: str,
        reason: str,
        metadata: Mapping[str, str] | None = None,
    ) -> RunEventRecord:
        """Append a metadata-only event while atomically advancing its sequence."""

        with self._lock:
            current = self._require(run_id)
            when = utc_now()
            next_state = AutonomousRunState.model_validate(
                current.model_copy(
                    update={
                        "version": current.version + 1,
                        "event_sequence": current.event_sequence + 1,
                        "updated_at": when,
                    }
                ).model_dump(mode="python")
            )
            event = RunEventRecord(
                run_id=run_id,
                sequence=next_state.event_sequence,
                event_type=event_type,
                from_status=current.status,
                to_status=current.status,
                from_phase=current.phase,
                to_phase=current.phase,
                reason=reason,
                metadata=dict(metadata or {}),
                occurred_at=when,
            )
            self._states[run_id] = copy_for_storage(next_state)
            self._events[run_id].append(copy_for_storage(event))
            return copy_for_storage(event)

    def claim_lease(
        self, run_id: str, owner: str, *, now: datetime | None = None, ttl_seconds: int = 60
    ) -> AutonomousRunState:
        with self._lock:
            current = self._require(run_id)
            when = _ensure_now(now)
            ttl_seconds = _ensure_ttl(ttl_seconds)
            if not owner.strip():
                raise ValueError("lease owner must not be blank")
            if (
                current.lease_owner
                and current.lease_expires_at
                and current.lease_expires_at > when
            ):
                raise LeaseConflictError("run lease is held by a live worker")
            next_state = current.model_copy(
                update={
                    "lease_owner": owner,
                    "lease_expires_at": when + timedelta(seconds=ttl_seconds),
                    "version": current.version + 1,
                    "updated_at": when,
                }
            )
            return self._store_state(
                AutonomousRunState.model_validate(next_state.model_dump(mode="python"))
            )

    def renew_lease(
        self, run_id: str, owner: str, *, now: datetime | None = None, ttl_seconds: int = 60
    ) -> AutonomousRunState:
        with self._lock:
            current = self._require(run_id)
            when = _ensure_now(now)
            ttl_seconds = _ensure_ttl(ttl_seconds)
            if (
                current.lease_owner != owner
                or not current.lease_expires_at
                or current.lease_expires_at <= when
            ):
                raise LeaseConflictError("worker does not hold a live lease")
            next_state = current.model_copy(
                update={
                    "lease_expires_at": when + timedelta(seconds=ttl_seconds),
                    "version": current.version + 1,
                    "updated_at": when,
                }
            )
            return self._store_state(
                AutonomousRunState.model_validate(next_state.model_dump(mode="python"))
            )

    def release_lease(self, run_id: str, owner: str) -> AutonomousRunState:
        with self._lock:
            current = self._require(run_id)
            if current.lease_owner != owner:
                raise LeaseConflictError("worker does not hold the run lease")
            when = utc_now()
            next_state = current.model_copy(
                update={
                    "lease_owner": None,
                    "lease_expires_at": None,
                    "version": current.version + 1,
                    "updated_at": when,
                }
            )
            return self._store_state(
                AutonomousRunState.model_validate(next_state.model_dump(mode="python"))
            )

    def scan_recoverable(
        self,
        *,
        now: datetime | None = None,
        limit: int = 100,
        cursor: Mapping[str, Any] | None = None,
    ) -> StatePage:
        if limit < 1:
            raise ValueError("limit must be positive")
        when = _ensure_now(now)
        if cursor is not None and set(cursor) != {"run_id"}:
            raise ValueError("in-memory recovery cursor must contain run_id")
        after = str(cursor["run_id"]) if cursor is not None else None
        with self._lock:
            values = [
                copy_for_storage(state)
                for run_id, state in sorted(self._states.items())
                if (after is None or run_id > after) and _recoverable(state, when)
            ]
            selected = values[:limit]
            next_cursor = {"run_id": selected[-1].run_id} if len(values) > len(selected) else None
            return StatePage(selected, next_cursor=next_cursor)

    def put_operation_intent(self, operation: RunOperation) -> RunOperation:
        with self._lock:
            self._require(operation.run_id)
            key = (operation.run_id, operation.operation_key)
            existing = self._operations.get(key)
            if existing is not None:
                if (
                    existing.request_digest is not None
                    and existing.request_digest == operation.request_digest
                ):
                    return copy_for_storage(existing)
                if existing != operation:
                    raise OperationAlreadyExistsError(
                        "operation key is already bound to another intent"
                    )
                return copy_for_storage(existing)
            self._operations[key] = copy_for_storage(operation)
            return copy_for_storage(operation)

    def get_operation(self, run_id: str, operation_key: str | None = None) -> RunOperation | None:
        with self._lock:
            if operation_key is None:
                matches = [
                    item
                    for (stored_run_id, _), item in self._operations.items()
                    if stored_run_id == run_id
                ]
                return copy_for_storage(matches[0]) if len(matches) == 1 else None
            operation = self._operations.get((run_id, operation_key))
            return copy_for_storage(operation) if operation else None

    def record_operation_result(
        self,
        run_id: str,
        operation_key: str,
        *,
        provider_id: str | None = None,
        status: RunOperationStatus | str,
        result: Mapping[str, Any] | None = None,
    ) -> RunOperation:
        with self._lock:
            current = self._operations.get((run_id, operation_key))
            if current is None:
                raise OperationNotFoundError("operation intent does not exist")
            normalized_status = RunOperationStatus(status)
            updated = _operation_update(
                current,
                provider_id=provider_id,
                status=normalized_status,
                result=result,
            )
            self._operations[(run_id, operation_key)] = copy_for_storage(updated)
            return copy_for_storage(updated)

    def list_events(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 100,
        cursor: Mapping[str, Any] | None = None,
    ) -> EventPage:
        if limit < 1:
            raise ValueError("limit must be positive")
        if cursor is not None:
            raise ValueError("in-memory event reads do not accept a cursor")
        with self._lock:
            self._require(run_id)
            values = [event for event in self._events[run_id] if event.sequence > after_sequence]
            selected = values[: max(0, limit)]
            next_after = selected[-1].sequence if len(values) > len(selected) and selected else None
            return EventPage([copy_for_storage(item) for item in selected], next_after=next_after)

    def add_experiment(self, run_id: str, experiment: ExperimentRecord) -> ExperimentRecord:
        with self._lock:
            state = self._require(run_id)
            if experiment.experiment_number > state.max_experiments:
                raise RepositoryError("experiment exceeds approved maximum")
            if any(
                item.experiment_number == experiment.experiment_number
                for item in self._experiments[run_id]
            ):
                raise RepositoryError("experiment number already exists")
            stored = copy_for_storage(experiment)
            self._experiments[run_id].append(stored)
            self._experiments[run_id].sort(key=lambda item: item.experiment_number)
            when = utc_now()
            updated_state = state.model_copy(
                update={
                    "experiments": [*self._experiments[run_id]],
                    "version": state.version + 1,
                    "updated_at": when,
                }
            )
            self._states[run_id] = copy_for_storage(
                AutonomousRunState.model_validate(updated_state.model_dump(mode="python"))
            )
            return copy_for_storage(stored)

    def list_experiments(
        self,
        run_id: str,
        *,
        offset: int = 0,
        limit: int = 100,
        cursor: Mapping[str, Any] | None = None,
    ) -> ExperimentPage:
        if limit < 1:
            raise ValueError("limit must be positive")
        if cursor is not None:
            raise ValueError("in-memory experiment reads do not accept a cursor")
        with self._lock:
            self._require(run_id)
            values = self._experiments[run_id]
            selected = values[max(0, offset) : max(0, offset) + max(0, limit)]
            end = max(0, offset) + len(selected)
            next_offset = end if end < len(values) else None
            return ExperimentPage(
                [copy_for_storage(item) for item in selected], next_offset=next_offset
            )

    create_run = create
    get_run = get
    transition_run = transition
    list_recoverable = scan_recoverable
    reserve_operation = put_operation_intent
    complete_operation = record_operation_result
    list_history = list_experiments


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _copy_idempotency(record: IdempotencyRecord) -> IdempotencyRecord:
    response = _safe_response(record.response) if record.response is not None else None
    return IdempotencyRecord(
        operation=record.operation,
        request_digest=record.request_digest,
        response=response,
    )


class DynamoDBAutonomousRunRepository:
    """DynamoDB implementation with conditional writes and transaction boundaries."""

    STATE_SK = "STATE"

    def __init__(
        self,
        *,
        table_name: str | None = None,
        table: Any | None = None,
        client: Any | None = None,
        resource: Any | None = None,
        region_name: str | None = None,
    ) -> None:
        if table is None and not table_name:
            raise ValueError("table_name is required when table is not supplied")
        derived_name = getattr(table, "name", None) if table is not None else None
        self.table_name = table_name or derived_name
        if not self.table_name:
            raise ValueError("injected table must expose a name or table_name is required")
        self._table = table
        self._client = client
        self._resource = resource
        self.region_name = region_name

    def _table_or_create(self) -> Any:
        if self._table is not None:
            return self._table
        try:
            import boto3  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise OptionalDependencyError("boto3 is required for DynamoDB persistence") from exc
        if self._resource is None:
            self._resource = boto3.resource("dynamodb", region_name=self.region_name)
        self._table = self._resource.Table(self.table_name)
        return self._table

    def _client_or_create(self) -> Any:
        if self._client is not None:
            return self._client
        table = self._table_or_create()
        self._client = getattr(getattr(table, "meta", None), "client", None)
        if self._client is None:
            raise RepositoryError("DynamoDB transaction client is unavailable")
        return self._client

    def update_state(
        self,
        run_id: str,
        *,
        expected_version: int,
        updates: Mapping[str, Any],
    ) -> AutonomousRunState:
        """Persist a validated optimistic-CAS state patch."""

        current = self.get(run_id)
        if current is None:
            raise RunNotFoundError(f"run {run_id!r} was not found")
        if current.version != expected_version:
            raise ConcurrentUpdateError(
                f"run version is {current.version}; expected {expected_version}"
            )
        normalized = _validated_state_updates(updates)
        if set(normalized) <= {"cancellation_requested", "safe_stop_requested"}:
            if len(normalized) != 1 or normalized.get(next(iter(normalized))) is not True:
                raise ValueError("control patches must request exactly one control flag")
            return self.request_cancel(
                run_id, expected_version=expected_version
            ) if "cancellation_requested" in normalized else self.request_safe_stop(
                run_id, expected_version=expected_version
            )
        next_state = AutonomousRunState.model_validate(
            current.model_copy(
                update={
                    **normalized,
                    "version": current.version + 1,
                    "updated_at": utc_now(),
                }
            ).model_dump(mode="python")
        )
        try:
            self._table_or_create().put_item(
                Item=self._item(self.STATE_SK, next_state),
                ConditionExpression="version = :version",
                ExpressionAttributeValues={":version": current.version},
            )
        except Exception as exc:
            if self._conditional(exc):
                raise ConcurrentUpdateError("conditional state update failed") from exc
            raise
        return next_state

    def _request_control(
        self, run_id: str, *, expected_version: int, field: str, reason: str, event_type: str
    ) -> AutonomousRunState:
        current = self.get(run_id)
        if current is None:
            raise RunNotFoundError(f"run {run_id!r} was not found")
        if current.version != expected_version:
            raise ConcurrentUpdateError("stale run version")
        if getattr(current, field):
            return current
        when = utc_now()
        next_state = AutonomousRunState.model_validate(
            current.model_copy(
                update={
                    field: True,
                    "version": current.version + 1,
                    "event_sequence": current.event_sequence + 1,
                    "updated_at": when,
                }
            ).model_dump(mode="python")
        )
        event = RunEventRecord(
            run_id=run_id,
            sequence=next_state.event_sequence,
            event_type=event_type,
            from_status=current.status,
            to_status=current.status,
            from_phase=current.phase,
            to_phase=current.phase,
            reason=reason,
            occurred_at=when,
        )
        try:
            self._client_or_create().transact_write_items(
                TransactItems=[
                    {
                        "Put": {
                            "TableName": self.table_name,
                            "Item": self._ddb_item(self.STATE_SK, next_state),
                            "ConditionExpression": f"version = :version AND {field} = :false",
                            "ExpressionAttributeValues": {
                                ":version": {"N": str(current.version)},
                                ":false": {"BOOL": False},
                            },
                        }
                    },
                    {
                        "Put": {
                            "TableName": self.table_name,
                            "Item": self._ddb_item(f"EVENT#{event.sequence:020d}", event),
                            "ConditionExpression": "attribute_not_exists(pk)",
                        }
                    },
                ]
            )
        except Exception as exc:
            if self._conditional(exc):
                raise ConcurrentUpdateError("control request changed concurrently") from exc
            raise
        return next_state

    def request_cancel(self, run_id: str, *, expected_version: int) -> AutonomousRunState:
        return self._request_control(
            run_id,
            expected_version=expected_version,
            field="cancellation_requested",
            reason="cancel requested",
            event_type="run.cancel_requested",
        )

    def request_safe_stop(self, run_id: str, *, expected_version: int) -> AutonomousRunState:
        return self._request_control(
            run_id,
            expected_version=expected_version,
            field="safe_stop_requested",
            reason="safe stop requested",
            event_type="run.safe_stop_requested",
        )

    @staticmethod
    def _operation_sk(operation_key: str) -> str:
        return f"OP#{operation_key}"

    @staticmethod
    def _key(run_id: str, sort_key: str = "STATE") -> dict[str, str]:
        return {"pk": f"RUN#{run_id}", "sk": sort_key}

    @classmethod
    def _ddb_key(cls, run_id: str, sort_key: str = "STATE") -> dict[str, dict[str, str]]:
        return {key: cls._encode(value) for key, value in cls._key(run_id, sort_key).items()}

    @staticmethod
    def _encode(value: Any) -> dict[str, Any]:
        if value is None:
            return {"NULL": True}
        if isinstance(value, bool):
            return {"BOOL": value}
        if isinstance(value, (int, float)):
            return {"N": str(value)}
        if isinstance(value, Mapping):
            return {
                "M": {str(k): DynamoDBAutonomousRunRepository._encode(v) for k, v in value.items()}
            }
        if isinstance(value, (list, tuple)):
            return {"L": [DynamoDBAutonomousRunRepository._encode(item) for item in value]}
        return {"S": str(value)}

    @classmethod
    def _item(cls, sort_key: str, payload: Any, *, run_id: str | None = None) -> dict[str, Any]:
        data = payload.model_dump(mode="json")
        payload_run_id = getattr(payload, "run_id", None) or run_id
        if not isinstance(payload_run_id, str) or not payload_run_id:
            raise ValueError("run_id is required for persisted records")
        item: dict[str, Any] = {
            "entity": type(payload).__name__,
            "run_id": payload_run_id,
            "pk": f"RUN#{payload_run_id}",
            "sk": sort_key,
            "payload": _json(data),
        }
        if isinstance(payload, AutonomousRunState):
            item.update(
                {
                    "version": payload.version,
                    "event_sequence": payload.event_sequence,
                    "status": payload.status.value,
                    "phase": payload.phase.value,
                    "approval_consumed": payload.approval_consumed,
                }
            )
        elif isinstance(payload, RunOperation):
            item.update(
                {
                    "operation_key": payload.operation_key,
                    "experiment_number": payload.experiment_number,
                    "status": payload.status.value,
                    "version": payload.version,
                }
            )
        elif isinstance(payload, ExperimentRecord):
            item["experiment_number"] = payload.experiment_number
        return item

    @classmethod
    def _ddb_item(cls, sort_key: str, payload: Any, *, run_id: str | None = None) -> dict[str, Any]:
        """Encode a native resource item for the low-level DynamoDB client."""

        return {
            key: cls._encode(value)
            for key, value in cls._item(sort_key, payload, run_id=run_id).items()
        }

    @staticmethod
    def _decode(item: Mapping[str, Any], model: type[ModelT]) -> ModelT:
        payload = item.get("payload", "{}")
        if isinstance(payload, Mapping) and "S" in payload:
            payload = payload["S"]
        if not isinstance(payload, str):
            payload = "{}"
        return cast(ModelT, model.model_validate(json.loads(payload)))  # type: ignore[attr-defined]

    def create(self, state: AutonomousRunState) -> AutonomousRunState:
        try:
            self._table_or_create().put_item(
                Item=self._item(self.STATE_SK, state),
                ConditionExpression="attribute_not_exists(pk)",
            )
        except Exception as exc:
            if self._conditional(exc):
                raise RunAlreadyExistsError(f"run {state.run_id!r} already exists") from exc
            raise
        return copy_for_storage(state)

    def get(self, run_id: str) -> AutonomousRunState | None:
        response = self._table_or_create().get_item(Key=self._key(run_id), ConsistentRead=True)
        item = response.get("Item")
        return self._decode(item, AutonomousRunState) if isinstance(item, Mapping) else None

    def _conditional(self, exc: BaseException) -> bool:
        response = getattr(exc, "response", None)
        return bool(
            isinstance(response, Mapping)
            and isinstance(response.get("Error"), Mapping)
            and response["Error"].get("Code") == "ConditionalCheckFailedException"
        ) or exc.__class__.__name__ in {"ConditionalCheckFailedException", "ConditionalError"}

    @staticmethod
    def _idempotency_key(operation: str, key: str) -> dict[str, str]:
        identity = sha256(_json({"operation": operation, "key": key}).encode("utf-8")).hexdigest()
        return {"pk": f"IDEMP#{identity}", "sk": "RECORD"}

    @classmethod
    def _idempotency_item(
        cls,
        operation: str,
        request_digest: str,
        response: Mapping[str, Any] | None,
        *,
        key: str,
    ) -> dict[str, Any]:
        item = cls._idempotency_key(operation, key)
        state = "COMPLETED" if response is not None else "PENDING"
        item.update(
            {
                "entity": "LiveAPIIdempotency",
                "operation": operation,
                "request_digest": request_digest,
                "state": state,
                "payload": _json(
                    {
                        "operation": operation,
                        "request_digest": request_digest,
                        "state": state,
                        "response": dict(response) if response is not None else None,
                    }
                ),
            }
        )
        return item

    @staticmethod
    def _decode_idempotency(item: Mapping[str, Any]) -> IdempotencyRecord:
        payload = item.get("payload")
        if not isinstance(payload, str):
            raise RepositoryError("stored idempotency record has invalid payload")
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RepositoryError("stored idempotency record has invalid payload") from exc
        if not isinstance(decoded, Mapping):
            raise RepositoryError("stored idempotency record has invalid payload")
        operation = decoded.get("operation")
        request_digest = decoded.get("request_digest")
        state = decoded.get("state")
        response = decoded.get("response")
        if (
            not isinstance(operation, str)
            or not isinstance(request_digest, str)
            or state not in {"PENDING", "COMPLETED"}
        ):
            raise RepositoryError("stored idempotency record is malformed")
        if state == "PENDING":
            if response is not None:
                raise RepositoryError("pending idempotency record contains a response")
            normalized_response = None
        else:
            if not isinstance(response, Mapping):
                raise RepositoryError("completed idempotency record has no safe response")
            normalized_response = _safe_response(response)
        return IdempotencyRecord(
            operation=operation,
            request_digest=request_digest,
            response=normalized_response,
        )

    def get_idempotency(self, operation: str, key: str) -> IdempotencyRecord | None:
        if not operation or not key:
            raise ValueError("idempotency operation and key are required")
        response = self._table_or_create().get_item(
            Key=self._idempotency_key(operation, key), ConsistentRead=True
        )
        item = response.get("Item")
        if not isinstance(item, Mapping):
            return None
        record = self._decode_idempotency(item)
        if record.operation != operation:
            raise RepositoryError("stored idempotency identity does not match its key")
        return _copy_idempotency(record)

    def claim_idempotency(
        self, operation: str, key: str, request_digest: str
    ) -> tuple[IdempotencyRecord, bool]:
        _validate_idempotency(operation, key, request_digest)
        record = IdempotencyRecord(operation=operation, request_digest=request_digest)
        try:
            self._table_or_create().put_item(
                Item=self._idempotency_item(operation, request_digest, None, key=key),
                ConditionExpression="attribute_not_exists(pk)",
            )
        except Exception as exc:
            if not self._conditional(exc):
                raise
            existing = self.get_idempotency(operation, key)
            if existing is None:
                raise RepositoryError(
                    "idempotency claim raced but no durable record is readable"
                ) from exc
            if existing.request_digest != request_digest:
                raise IdempotencyKeyConflictError(
                    "idempotency key is bound to a different request"
                ) from exc
            return existing, False
        return record, True

    def complete_idempotency(
        self,
        operation: str,
        key: str,
        request_digest: str,
        response: Mapping[str, Any],
    ) -> IdempotencyRecord:
        _validate_idempotency(operation, key, request_digest)
        normalized = _safe_response(response)
        current = self.get_idempotency(operation, key)
        if current is None or current.request_digest != request_digest:
            raise IdempotencyKeyConflictError("idempotency claim does not match completion")
        if current.response is not None:
            if _json(current.response) != _json(normalized):
                raise IdempotencyKeyConflictError("idempotency response cannot be changed")
            return current
        try:
            self._table_or_create().put_item(
                Item=self._idempotency_item(operation, request_digest, normalized, key=key),
                ConditionExpression="request_digest = :request_digest AND #state = :pending",
                ExpressionAttributeNames={"#state": "state"},
                ExpressionAttributeValues={
                    ":request_digest": request_digest,
                    ":pending": "PENDING",
                },
            )
        except Exception as exc:
            if not self._conditional(exc):
                raise
            raced = self.get_idempotency(operation, key)
            if raced is not None and raced.request_digest == request_digest:
                if raced.response is not None and _json(raced.response) == _json(normalized):
                    return raced
            raise IdempotencyKeyConflictError(
                "idempotency completion changed concurrently"
            ) from exc
        return IdempotencyRecord(
            operation=operation,
            request_digest=request_digest,
            response=normalized,
        )

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
        current = self.get(run_id)
        if current is None:
            raise RunNotFoundError(f"run {run_id!r} was not found")
        if current.version != expected_version:
            raise ConcurrentUpdateError("stale run version")
        when = utc_now()
        next_state = AutonomousRunState.model_validate(
            current.model_copy(
                update={
                    "status": status,
                    "phase": phase,
                    "version": current.version + 1,
                    "event_sequence": current.event_sequence + 1,
                    "updated_at": when,
                }
            ).model_dump(mode="python")
        )
        event = RunEventRecord(
            run_id=run_id,
            sequence=next_state.event_sequence,
            event_type=event_type,
            from_status=current.status,
            to_status=status,
            from_phase=current.phase,
            to_phase=phase,
            reason=reason,
            metadata=dict(metadata or {}),
            occurred_at=when,
        )
        try:
            self._client_or_create().transact_write_items(
                TransactItems=[
                    {
                        "Put": {
                            "TableName": self.table_name,
                            "Item": self._ddb_item(self.STATE_SK, next_state),
                            "ConditionExpression": "version = :version",
                            "ExpressionAttributeValues": {":version": {"N": str(expected_version)}},
                        }
                    },
                    {
                        "Put": {
                            "TableName": self.table_name,
                            "Item": self._ddb_item(f"EVENT#{event.sequence:020d}", event),
                            "ConditionExpression": "attribute_not_exists(pk)",
                        }
                    },
                ]
            )
        except Exception as exc:
            if self._conditional(exc):
                raise ConcurrentUpdateError("conditional transition failed") from exc
            raise
        return next_state

    def append_event(
        self,
        run_id: str,
        *,
        event_type: str,
        reason: str,
        metadata: Mapping[str, str] | None = None,
    ) -> RunEventRecord:
        current = self.get(run_id)
        if current is None:
            raise RunNotFoundError(f"run {run_id!r} was not found")
        next_state = self.transition(
            run_id,
            expected_version=current.version,
            status=current.status,
            phase=current.phase,
            reason=reason,
            event_type=event_type,
            metadata=metadata,
        )
        response = self._table_or_create().get_item(
            Key=self._key(run_id, f"EVENT#{next_state.event_sequence:020d}"),
            ConsistentRead=True,
        )
        item = response.get("Item")
        if not isinstance(item, Mapping):
            raise RepositoryError("event append succeeded without a readable event")
        event = self._decode(item, RunEventRecord)
        if event.sequence != next_state.event_sequence:
            raise RepositoryError("event append returned an unexpected sequence")
        return event

    def consume_approval(self, run_id: str, approval_digest: str) -> AutonomousRunState:
        current = self.get(run_id)
        if current is None:
            raise RunNotFoundError(f"run {run_id!r} was not found")
        if current.approval_consumed:
            raise ApprovalAlreadyConsumedError("approval packet has already been consumed")
        when = utc_now()
        next_state = AutonomousRunState.model_validate(
            current.model_copy(
                update={
                    "approval_digest": approval_digest,
                    "approval_consumed": True,
                    "approval_consumed_at": when,
                    "version": current.version + 1,
                    "event_sequence": current.event_sequence + 1,
                    "updated_at": when,
                }
            ).model_dump(mode="python")
        )
        event = RunEventRecord(
            run_id=run_id,
            sequence=next_state.event_sequence,
            event_type="approval.consumed",
            from_status=current.status,
            to_status=current.status,
            from_phase=current.phase,
            to_phase=current.phase,
            reason="approval packet consumed",
            metadata={"approval_digest": approval_digest},
            occurred_at=when,
        )
        try:
            self._client_or_create().transact_write_items(
                TransactItems=[
                    {
                        "Put": {
                            "TableName": self.table_name,
                            "Item": self._ddb_item(self.STATE_SK, next_state),
                            "ConditionExpression": (
                                "version = :version AND approval_consumed = :false"
                            ),
                            "ExpressionAttributeValues": {
                                ":version": {"N": str(current.version)},
                                ":false": {"BOOL": False},
                            },
                        }
                    },
                    {
                        "Put": {
                            "TableName": self.table_name,
                            "Item": self._ddb_item(f"EVENT#{event.sequence:020d}", event),
                            "ConditionExpression": "attribute_not_exists(pk)",
                        }
                    },
                ]
            )
        except Exception as exc:
            if self._conditional(exc):
                raise ApprovalAlreadyConsumedError(
                    "approval packet was consumed concurrently"
                ) from exc
            raise
        return next_state

    # Lease mutations use conditional updates; transition-style events are not emitted for leases.
    def claim_lease(
        self, run_id: str, owner: str, *, now: datetime | None = None, ttl_seconds: int = 60
    ) -> AutonomousRunState:
        current = self.get(run_id)
        if current is None:
            raise RunNotFoundError(f"run {run_id!r} was not found")
        when = _ensure_now(now)
        ttl_seconds = _ensure_ttl(ttl_seconds)
        if (
            current.lease_owner
            and current.lease_expires_at
            and current.lease_expires_at > when
        ):
            raise LeaseConflictError("run lease is held by a live worker")
        next_state = AutonomousRunState.model_validate(
            current.model_copy(
                update={
                    "lease_owner": owner,
                    "lease_expires_at": when + timedelta(seconds=ttl_seconds),
                    "version": current.version + 1,
                    "updated_at": when,
                }
            ).model_dump(mode="python")
        )
        try:
            self._table_or_create().put_item(
                Item=self._item(self.STATE_SK, next_state),
                ConditionExpression="version = :version",
                ExpressionAttributeValues={":version": current.version},
            )
        except Exception as exc:
            if self._conditional(exc):
                raise LeaseConflictError("lease claim lost a race") from exc
            raise
        return next_state

    def renew_lease(
        self, run_id: str, owner: str, *, now: datetime | None = None, ttl_seconds: int = 60
    ) -> AutonomousRunState:
        current = self.get(run_id)
        if current is None:
            raise RunNotFoundError(f"run {run_id!r} was not found")
        when = _ensure_now(now)
        ttl_seconds = _ensure_ttl(ttl_seconds)
        if (
            current.lease_owner != owner
            or not current.lease_expires_at
            or current.lease_expires_at <= when
        ):
            raise LeaseConflictError("worker does not hold a live lease")
        next_state = AutonomousRunState.model_validate(
            current.model_copy(
                update={
                    "lease_expires_at": when + timedelta(seconds=ttl_seconds),
                    "version": current.version + 1,
                    "updated_at": when,
                }
            ).model_dump(mode="python")
        )
        try:
            self._table_or_create().put_item(
                Item=self._item(self.STATE_SK, next_state),
                ConditionExpression="version = :version",
                ExpressionAttributeValues={":version": current.version},
            )
        except Exception as exc:
            if self._conditional(exc):
                raise LeaseConflictError("lease renewal lost a race") from exc
            raise
        return next_state

    def release_lease(self, run_id: str, owner: str) -> AutonomousRunState:
        current = self.get(run_id)
        if current is None:
            raise RunNotFoundError(f"run {run_id!r} was not found")
        if current.lease_owner != owner:
            raise LeaseConflictError("worker does not hold the run lease")
        next_state = AutonomousRunState.model_validate(
            current.model_copy(
                update={
                    "lease_owner": None,
                    "lease_expires_at": None,
                    "version": current.version + 1,
                    "updated_at": utc_now(),
                }
            ).model_dump(mode="python")
        )
        try:
            self._table_or_create().put_item(
                Item=self._item(self.STATE_SK, next_state),
                ConditionExpression="version = :version",
                ExpressionAttributeValues={":version": current.version},
            )
        except Exception as exc:
            if self._conditional(exc):
                raise LeaseConflictError("lease release lost a race") from exc
            raise
        return next_state

    def scan_recoverable(
        self,
        *,
        now: datetime | None = None,
        limit: int = 100,
        cursor: Mapping[str, Any] | None = None,
    ) -> StatePage:
        if limit < 1:
            raise ValueError("limit must be positive")
        # Scan is intentionally metadata-only and bounded; production callers should add a GSI.
        when = _ensure_now(now)
        output: list[AutonomousRunState] = []
        next_cursor = dict(cursor) if cursor is not None else None
        while len(output) < limit:
            kwargs: dict[str, Any] = {"Limit": limit}
            if next_cursor is not None:
                kwargs["ExclusiveStartKey"] = next_cursor
            response = self._table_or_create().scan(**kwargs)
            items = response.get("Items", [])
            page_cursor: dict[str, Any] | None = None
            for index, item in enumerate(items):
                if item.get("sk") != self.STATE_SK:
                    continue
                state = self._decode(item, AutonomousRunState)
                if _recoverable(state, when):
                    output.append(state)
                    page_cursor = {"pk": item.get("pk"), "sk": item.get("sk")}
                    if len(output) >= limit:
                        has_more_in_page = index + 1 < len(items)
                        if not has_more_in_page and not isinstance(
                            response.get("LastEvaluatedKey"), Mapping
                        ):
                            page_cursor = None
                        break
            raw_cursor = response.get("LastEvaluatedKey")
            if len(output) >= limit and page_cursor is not None and (
                isinstance(raw_cursor, Mapping) or page_cursor is not None
            ):
                next_cursor = page_cursor
            else:
                next_cursor = dict(raw_cursor) if isinstance(raw_cursor, Mapping) else None
            if next_cursor is None:
                break
        return StatePage(output[:limit], next_cursor=next_cursor)

    def put_operation_intent(self, operation: RunOperation) -> RunOperation:
        if self.get(operation.run_id) is None:
            raise RunNotFoundError(f"run {operation.run_id!r} was not found")
        item = self._ddb_item(self._operation_sk(operation.operation_key), operation)
        try:
            self._client_or_create().transact_write_items(
                TransactItems=[
                    {
                        "ConditionCheck": {
                            "TableName": self.table_name,
                            "Key": self._ddb_key(operation.run_id, self.STATE_SK),
                            "ConditionExpression": "attribute_exists(pk)",
                        }
                    },
                    {
                        "Put": {
                            "TableName": self.table_name,
                            "Item": item,
                            "ConditionExpression": "attribute_not_exists(pk)",
                        }
                    },
                ]
            )
        except Exception as exc:
            if self._conditional(exc):
                existing = self.get_operation(operation.run_id, operation.operation_key)
                if (
                    existing is not None
                    and existing.request_digest is not None
                    and existing.request_digest == operation.request_digest
                ):
                    return existing
                if existing == operation:
                    return existing
                raise OperationAlreadyExistsError(
                    "operation key is already bound to another intent"
                ) from exc
            raise
        return operation

    def get_operation(self, run_id: str, operation_key: str | None = None) -> RunOperation | None:
        if operation_key is not None:
            response = self._table_or_create().get_item(
                Key=self._key(run_id, self._operation_sk(operation_key)), ConsistentRead=True
            )
            item = response.get("Item")
            return self._decode(item, RunOperation) if isinstance(item, Mapping) else None
        response = self._table_or_create().query(
            KeyConditionExpression="pk = :pk AND sk BETWEEN :start AND :end",
            ExpressionAttributeValues={
                ":pk": f"RUN#{run_id}",
                ":start": "OP#",
                ":end": "OP#\uffff",
            },
            Limit=2,
            ScanIndexForward=True,
        )
        items = response.get("Items", [])
        if len(items) != 1:
            return None
        item = items[0]
        return self._decode(item, RunOperation) if isinstance(item, Mapping) else None

    def record_operation_result(
        self,
        run_id: str,
        operation_key: str,
        *,
        provider_id: str | None = None,
        status: RunOperationStatus | str,
        result: Mapping[str, Any] | None = None,
    ) -> RunOperation:
        current = self.get_operation(run_id, operation_key)
        if current is None:
            raise OperationNotFoundError("operation intent does not exist")
        normalized_status = RunOperationStatus(status)
        updated = _operation_update(
            current,
            provider_id=provider_id,
            status=normalized_status,
            result=result,
        )
        if updated == current:
            return current
        try:
            self._table_or_create().put_item(
                Item=self._item(self._operation_sk(updated.operation_key), updated),
                ConditionExpression="version = :version",
                ExpressionAttributeValues={":version": current.version},
            )
        except Exception as exc:
            if self._conditional(exc):
                raise OperationAlreadyExistsError("operation result changed concurrently") from exc
            raise
        return updated

    def list_events(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 100,
        cursor: Mapping[str, Any] | None = None,
    ) -> EventPage:
        if limit < 1:
            raise ValueError("limit must be positive")
        expression_values: dict[str, Any] = {
            ":pk": f"RUN#{run_id}",
            ":start": f"EVENT#{after_sequence + 1:020d}",
            ":end": "EVENT#\uffff",
        }
        kwargs: dict[str, Any] = {
            "KeyConditionExpression": "pk = :pk AND sk BETWEEN :start AND :end",
            "ExpressionAttributeValues": expression_values,
            "Limit": limit,
            "ScanIndexForward": True,
        }
        if cursor is not None:
            kwargs["ExclusiveStartKey"] = dict(cursor)
        response = self._table_or_create().query(**kwargs)
        values = [self._decode(item, RunEventRecord) for item in response.get("Items", [])]
        return EventPage(
            values[:limit],
            next_after=values[-1].sequence if values and response.get("LastEvaluatedKey") else None,
            next_cursor=response.get("LastEvaluatedKey"),
        )

    def add_experiment(self, run_id: str, experiment: ExperimentRecord) -> ExperimentRecord:
        state = self.get(run_id)
        if state is None:
            raise RunNotFoundError(f"run {run_id!r} was not found")
        if experiment.experiment_number > state.max_experiments:
            raise RepositoryError("experiment exceeds approved maximum")
        if any(
            item.experiment_number == experiment.experiment_number for item in state.experiments
        ):
            raise RepositoryError("experiment number already exists")
        when = utc_now()
        next_state = AutonomousRunState.model_validate(
            state.model_copy(
                update={
                    "experiments": [*state.experiments, experiment],
                    "version": state.version + 1,
                    "updated_at": when,
                }
            ).model_dump(mode="python")
        )
        try:
            table_name = self.table_name
            if table_name is None:
                table_name = getattr(self._table_or_create(), "name", None)
            self._client_or_create().transact_write_items(
                TransactItems=[
                    {
                        "Put": {
                            "TableName": table_name,
                            "Item": self._ddb_item(
                                f"EXP#{experiment.experiment_number:04d}",
                                experiment,
                                run_id=run_id,
                            ),
                            "ConditionExpression": "attribute_not_exists(pk)",
                        }
                    },
                    {
                        "Put": {
                            "TableName": table_name,
                            "Item": self._ddb_item(self.STATE_SK, next_state),
                            "ConditionExpression": "version = :version",
                            "ExpressionAttributeValues": {":version": {"N": str(state.version)}},
                        }
                    },
                ]
            )
        except Exception as exc:
            if self._conditional(exc):
                raise ConcurrentUpdateError("experiment history changed concurrently") from exc
            raise
        return experiment

    def list_experiments(
        self,
        run_id: str,
        *,
        offset: int = 0,
        limit: int = 100,
        cursor: Mapping[str, Any] | None = None,
    ) -> ExperimentPage:
        if limit < 1:
            raise ValueError("limit must be positive")
        if offset < 0:
            raise ValueError("offset must not be negative")
        kwargs: dict[str, Any] = {
            "KeyConditionExpression": "pk = :pk AND sk BETWEEN :start AND :end",
            "ExpressionAttributeValues": {
                ":pk": f"RUN#{run_id}",
                ":start": "EXP#0000",
                ":end": "EXP#\uffff",
            },
            "ScanIndexForward": True,
            "Limit": offset + limit if cursor is None else limit,
        }
        if cursor is not None:
            kwargs["ExclusiveStartKey"] = dict(cursor)
        response = self._table_or_create().query(**kwargs)
        values = [self._decode(item, ExperimentRecord) for item in response.get("Items", [])]
        selected = values[offset : offset + limit] if cursor is None and offset else values[:limit]
        return ExperimentPage(
            selected,
            next_offset=(
                offset + len(selected)
                if cursor is None and offset + len(selected) < len(values)
                else None
            ),
            next_cursor=response.get("LastEvaluatedKey"),
        )

    create_run = create
    get_run = get
    transition_run = transition
    list_recoverable = scan_recoverable
    reserve_operation = put_operation_intent
    complete_operation = record_operation_result
    list_history = list_experiments


__all__ = [
    "ApprovalAlreadyConsumedError",
    "AutonomousRunRepository",
    "ConcurrentUpdateError",
    "DynamoDBAutonomousRunRepository",
    "DynamoDBRepository",
    "EventPage",
    "ExperimentPage",
    "InMemoryAutonomousRepository",
    "InMemoryAutonomousRunRepository",
    "LeaseConflictError",
    "OperationAlreadyExistsError",
    "OperationNotFoundError",
    "OptionalDependencyError",
    "Page",
    "RepositoryError",
    "RunAlreadyExistsError",
    "RunNotFoundError",
]

InMemoryAutonomousRepository = InMemoryAutonomousRunRepository
DynamoDBRepository = DynamoDBAutonomousRunRepository
