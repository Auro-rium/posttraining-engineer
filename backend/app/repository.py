"""Async run persistence with local and optional Firestore backends."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Protocol

from .models import RunEvent, RunState, utc_now


class RepositoryError(RuntimeError):
    """Base persistence error."""


class RunNotFoundError(RepositoryError):
    """Raised when a requested run does not exist."""


class VersionConflictError(RepositoryError):
    """Raised when optimistic concurrency detects a stale writer."""


class RunRepository(Protocol):
    async def create_run(self, run: RunState) -> RunState: ...

    async def get_run(self, run_id: str) -> RunState: ...

    async def save_run(self, run: RunState, *, expected_version: int) -> RunState: ...

    async def list_runs(self, *, limit: int = 100) -> list[RunState]: ...

    async def append_event(self, event: RunEvent) -> RunEvent: ...

    async def list_events(
        self, run_id: str, *, after_event_id: str | None = None, limit: int = 1000
    ) -> list[RunEvent]: ...


class InMemoryRunRepository:
    """Process-local repository suitable for tests and credential-free demos."""

    def __init__(self) -> None:
        self._runs: dict[str, RunState] = {}
        self._events: dict[str, list[RunEvent]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def create_run(self, run: RunState) -> RunState:
        async with self._lock:
            if run.run_id in self._runs:
                raise VersionConflictError(f"run already exists: {run.run_id}")
            stored = run.model_copy(deep=True)
            self._runs[run.run_id] = stored
            return stored.model_copy(deep=True)

    async def get_run(self, run_id: str) -> RunState:
        async with self._lock:
            try:
                return self._runs[run_id].model_copy(deep=True)
            except KeyError as exc:
                raise RunNotFoundError(run_id) from exc

    async def save_run(self, run: RunState, *, expected_version: int) -> RunState:
        async with self._lock:
            current = self._runs.get(run.run_id)
            if current is None:
                raise RunNotFoundError(run.run_id)
            if current.version != expected_version:
                raise VersionConflictError(
                    f"run {run.run_id} is version {current.version}, expected {expected_version}"
                )
            stored = run.model_copy(
                update={"version": expected_version + 1, "updated_at": utc_now()}, deep=True
            )
            self._runs[run.run_id] = stored
            return stored.model_copy(deep=True)

    async def list_runs(self, *, limit: int = 100) -> list[RunState]:
        if limit < 1:
            return []
        async with self._lock:
            runs = sorted(self._runs.values(), key=lambda item: item.created_at, reverse=True)
            return [item.model_copy(deep=True) for item in runs[:limit]]

    async def append_event(self, event: RunEvent) -> RunEvent:
        async with self._lock:
            if event.run_id not in self._runs:
                raise RunNotFoundError(event.run_id)
            if any(item.event_id == event.event_id for item in self._events[event.run_id]):
                return event.model_copy(deep=True)
            stored = event.model_copy(deep=True)
            self._events[event.run_id].append(stored)
            return stored.model_copy(deep=True)

    async def list_events(
        self, run_id: str, *, after_event_id: str | None = None, limit: int = 1000
    ) -> list[RunEvent]:
        if limit < 1:
            return []
        async with self._lock:
            if run_id not in self._runs:
                raise RunNotFoundError(run_id)
            events = self._events[run_id]
            start = 0
            if after_event_id is not None:
                for index, event in enumerate(events):
                    if event.event_id == after_event_id:
                        start = index + 1
                        break
            return [event.model_copy(deep=True) for event in events[start : start + limit]]


class FirestoreRunRepository:
    """Firestore implementation loaded only when explicitly instantiated.

    The official synchronous client is executed in worker threads so the public
    repository interface stays async and does not block the API event loop.
    """

    def __init__(
        self,
        *,
        project: str | None = None,
        database: str = "(default)",
        collection: str = "runs",
    ) -> None:
        try:
            from google.cloud import firestore
        except ImportError as exc:  # pragma: no cover - depends on optional cloud SDK
            raise RuntimeError(
                "Firestore support requires the google-cloud-firestore package"
            ) from exc
        self._firestore = firestore
        self._client = firestore.Client(project=project, database=database)
        self._collection = self._client.collection(collection)

    def _run_ref(self, run_id: str) -> Any:
        return self._collection.document(run_id)

    async def create_run(self, run: RunState) -> RunState:
        def create() -> None:
            self._run_ref(run.run_id).create(run.model_dump(mode="json"))

        try:
            await asyncio.to_thread(create)
        except Exception as exc:  # Google exception types remain optional
            if "AlreadyExists" in type(exc).__name__:
                raise VersionConflictError(f"run already exists: {run.run_id}") from exc
            raise
        return run.model_copy(deep=True)

    async def get_run(self, run_id: str) -> RunState:
        snapshot = await asyncio.to_thread(self._run_ref(run_id).get)
        if not snapshot.exists:
            raise RunNotFoundError(run_id)
        return RunState.model_validate(snapshot.to_dict())

    async def save_run(self, run: RunState, *, expected_version: int) -> RunState:
        transaction = self._client.transaction()
        stored = run.model_copy(
            update={"version": expected_version + 1, "updated_at": utc_now()}, deep=True
        )

        @self._firestore.transactional
        def update_in_transaction(txn: Any) -> None:
            ref = self._run_ref(run.run_id)
            snapshot = ref.get(transaction=txn)
            if not snapshot.exists:
                raise RunNotFoundError(run.run_id)
            actual_version = int(snapshot.get("version"))
            if actual_version != expected_version:
                raise VersionConflictError(
                    f"run {run.run_id} is version {actual_version}, expected {expected_version}"
                )
            txn.set(ref, stored.model_dump(mode="json"))

        await asyncio.to_thread(update_in_transaction, transaction)
        return stored

    async def list_runs(self, *, limit: int = 100) -> list[RunState]:
        if limit < 1:
            return []

        def stream() -> list[RunState]:
            query = self._collection.order_by("created_at", direction="DESCENDING").limit(limit)
            return [RunState.model_validate(snapshot.to_dict()) for snapshot in query.stream()]

        return await asyncio.to_thread(stream)

    async def append_event(self, event: RunEvent) -> RunEvent:
        snapshot = await asyncio.to_thread(self._run_ref(event.run_id).get)
        if not snapshot.exists:
            raise RunNotFoundError(event.run_id)

        def create_event() -> None:
            self._run_ref(event.run_id).collection("events").document(event.event_id).set(
                event.model_dump(mode="json")
            )

        await asyncio.to_thread(create_event)
        return event.model_copy(deep=True)

    async def list_events(
        self, run_id: str, *, after_event_id: str | None = None, limit: int = 1000
    ) -> list[RunEvent]:
        if limit < 1:
            return []
        if after_event_id is not None:
            # Event IDs are random, so resolve the cursor timestamp first.
            cursor = await asyncio.to_thread(
                self._run_ref(run_id).collection("events").document(after_event_id).get
            )
            if not cursor.exists:
                return []
            cursor_time = cursor.get("created_at")
        else:
            cursor_time = None

        def stream() -> list[RunEvent]:
            query = self._run_ref(run_id).collection("events").order_by("created_at")
            if cursor_time is not None:
                query = query.where("created_at", ">", cursor_time)
            return [RunEvent.model_validate(item.to_dict()) for item in query.limit(limit).stream()]

        return await asyncio.to_thread(stream)
