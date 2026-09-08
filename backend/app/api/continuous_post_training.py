"""Continuous post-training control-plane API contracts.

This module is deliberately an integration boundary.  It records traces,
cycle requests, human decisions, ordered events, and artifact metadata; it
does not run a training job or claim that an artifact exists merely because a
cycle was approved.  The default repository is process-local and attached to
the FastAPI application's state.  Production callers should inject a durable
implementation of :class:`PostTrainingRepository`.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Protocol
from uuid import UUID, uuid4

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

JsonObject = dict[str, Any]


def utc_now() -> datetime:
    """Return an aware UTC timestamp, kept as a function for test injection."""

    return datetime.now(UTC)


class TraceRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class CycleStatus(StrEnum):
    PENDING_APPROVAL = "pending_approval"
    QUEUED = "queued"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class EventType(StrEnum):
    CREATED = "cycle_created"
    APPROVED = "cycle_approved"
    REJECTED = "cycle_rejected"
    CANCEL_REQUESTED = "cycle_cancel_requested"
    CANCELLED = "cycle_cancelled"
    STATUS_CHANGED = "cycle_status_changed"
    ARTIFACT_RECORDED = "artifact_recorded"


class ContractModel(BaseModel):
    """Base model that makes the wire contract reject accidental fields."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class TraceMessage(ContractModel):
    role: TraceRole
    content: str = Field(min_length=1, max_length=100_000)
    name: str | None = Field(default=None, max_length=200)


class TraceCreate(ContractModel):
    source: str = Field(min_length=1, max_length=200)
    external_id: str | None = Field(default=None, max_length=500)
    occurred_at: datetime = Field(default_factory=utc_now)
    messages: list[TraceMessage] = Field(min_length=1, max_length=10_000)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_aware_timestamp(self) -> TraceCreate:
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone")
        return self


class TraceRead(TraceCreate):
    trace_id: UUID
    created_at: datetime


class CycleConfig(ContractModel):
    base_model: str = Field(min_length=1, max_length=500)
    recipe: str = Field(min_length=1, max_length=200)
    parameters: JsonObject = Field(default_factory=dict)


class CycleCreate(ContractModel):
    trace_ids: list[UUID] = Field(min_length=1, max_length=10_000)
    config: CycleConfig | None = None
    # ``configuration`` is accepted as a readable wire alias while ``config``
    # remains the canonical response field.
    configuration: CycleConfig | None = None

    @model_validator(mode="after")
    def normalize_config(self) -> CycleCreate:
        if self.config is None and self.configuration is None:
            raise ValueError("config is required")
        if (
            self.config is not None
            and self.configuration is not None
            and self.config != self.configuration
        ):
            raise ValueError("config and configuration must match when both are supplied")
        if len(set(self.trace_ids)) != len(self.trace_ids):
            raise ValueError("trace_ids must be unique")
        if self.config is None:
            self.config = self.configuration
        return self


class FailureInfo(ContractModel):
    code: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=2_000)


class CycleRead(ContractModel):
    cycle_id: UUID
    trace_ids: list[UUID]
    config: CycleConfig
    status: CycleStatus
    version: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime
    failure: FailureInfo | None = None


class DecisionRequest(ContractModel):
    expected_version: int | None = Field(default=None, ge=0)
    reason: str | None = Field(default=None, max_length=2_000)


class CycleEvent(ContractModel):
    event_id: UUID
    cycle_id: UUID
    sequence: int = Field(ge=1)
    event_type: EventType
    occurred_at: datetime
    previous_status: CycleStatus | None = None
    new_status: CycleStatus
    reason: str | None = None


class CycleEventsRead(ContractModel):
    cycle_id: UUID
    events: list[CycleEvent]
    next_after: int | None = None


class ArtifactRead(ContractModel):
    artifact_id: UUID
    cycle_id: UUID
    kind: str = Field(min_length=1, max_length=200)
    uri: str = Field(min_length=1, max_length=2_000)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    created_at: datetime
    metadata: JsonObject = Field(default_factory=dict)


class CycleArtifactsRead(ContractModel):
    cycle_id: UUID
    artifacts: list[ArtifactRead]


class RepositoryError(Exception):
    """Base for storage errors that should become stable API responses."""


class CycleNotFound(RepositoryError):
    pass


class VersionConflict(RepositoryError):
    pass


class InvalidCommand(RepositoryError):
    pass


class PostTrainingRepository(Protocol):
    """Storage contract used by :class:`CycleService`.

    A durable adapter should make ``compare_and_set_cycle`` and its event
    append one atomic operation.  Implementations must return defensive
    copies so callers cannot mutate stored state outside the repository.
    """

    async def create_trace(self, trace: TraceRead) -> TraceRead: ...

    async def get_trace(self, trace_id: UUID) -> TraceRead | None: ...

    async def create_cycle(self, cycle: CycleRead, event: CycleEvent) -> CycleRead: ...

    async def get_cycle(self, cycle_id: UUID) -> CycleRead | None: ...

    async def compare_and_set_cycle(
        self, cycle: CycleRead, expected_version: int, event: CycleEvent
    ) -> CycleRead: ...

    async def list_events(
        self, cycle_id: UUID, after_sequence: int = 0, limit: int = 500
    ) -> list[CycleEvent]: ...

    async def list_artifacts(self, cycle_id: UUID) -> list[ArtifactRead]: ...

    async def add_artifact(self, artifact: ArtifactRead, event: CycleEvent) -> ArtifactRead: ...


class PostTrainingService(Protocol):
    """Application service contract for dependency injection and fakes."""

    async def create_trace(self, request: TraceCreate) -> TraceRead: ...

    async def create_cycle(self, request: CycleCreate) -> CycleRead: ...

    async def get_cycle(self, cycle_id: UUID) -> CycleRead: ...

    async def list_events(
        self, cycle_id: UUID, after_sequence: int, limit: int
    ) -> CycleEventsRead: ...

    async def list_artifacts(self, cycle_id: UUID) -> CycleArtifactsRead: ...

    async def approve(self, cycle_id: UUID, request: DecisionRequest | None) -> CycleRead: ...

    async def reject(self, cycle_id: UUID, request: DecisionRequest | None) -> CycleRead: ...

    async def cancel(self, cycle_id: UUID, request: DecisionRequest | None) -> CycleRead: ...


class InMemoryPostTrainingRepository:
    """Small concurrency-safe repository for local tests and demonstrations."""

    def __init__(self) -> None:
        self._traces: dict[UUID, TraceRead] = {}
        self._cycles: dict[UUID, CycleRead] = {}
        self._events: dict[UUID, list[CycleEvent]] = {}
        self._artifacts: dict[UUID, list[ArtifactRead]] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _copy(model: ContractModel) -> Any:
        return model.model_copy(deep=True)

    async def create_trace(self, trace: TraceRead) -> TraceRead:
        async with self._lock:
            self._traces[trace.trace_id] = self._copy(trace)
            return self._copy(trace)

    async def get_trace(self, trace_id: UUID) -> TraceRead | None:
        async with self._lock:
            stored = self._traces.get(trace_id)
            return self._copy(stored) if stored is not None else None

    async def create_cycle(self, cycle: CycleRead, event: CycleEvent) -> CycleRead:
        async with self._lock:
            if cycle.cycle_id in self._cycles:
                raise InvalidCommand("cycle_id already exists")
            self._cycles[cycle.cycle_id] = self._copy(cycle)
            self._events[cycle.cycle_id] = [self._copy(event)]
            self._artifacts[cycle.cycle_id] = []
            return self._copy(cycle)

    async def get_cycle(self, cycle_id: UUID) -> CycleRead | None:
        async with self._lock:
            stored = self._cycles.get(cycle_id)
            return self._copy(stored) if stored is not None else None

    async def compare_and_set_cycle(
        self, cycle: CycleRead, expected_version: int, event: CycleEvent
    ) -> CycleRead:
        async with self._lock:
            current = self._cycles.get(cycle.cycle_id)
            if current is None:
                raise CycleNotFound("cycle not found")
            if current.version != expected_version:
                raise VersionConflict(
                    f"cycle version is {current.version}; expected {expected_version}"
                )
            self._cycles[cycle.cycle_id] = self._copy(cycle)
            self._events.setdefault(cycle.cycle_id, []).append(self._copy(event))
            return self._copy(cycle)

    async def list_events(
        self, cycle_id: UUID, after_sequence: int = 0, limit: int = 500
    ) -> list[CycleEvent]:
        async with self._lock:
            if cycle_id not in self._cycles:
                raise CycleNotFound("cycle not found")
            events = [
                event
                for event in self._events.get(cycle_id, [])
                if event.sequence > after_sequence
            ]
            return [self._copy(event) for event in events[:limit]]

    async def list_artifacts(self, cycle_id: UUID) -> list[ArtifactRead]:
        async with self._lock:
            if cycle_id not in self._cycles:
                raise CycleNotFound("cycle not found")
            return [self._copy(item) for item in self._artifacts.get(cycle_id, [])]

    async def add_artifact(self, artifact: ArtifactRead, event: CycleEvent) -> ArtifactRead:
        async with self._lock:
            if artifact.cycle_id not in self._cycles:
                raise CycleNotFound("cycle not found")
            self._artifacts.setdefault(artifact.cycle_id, []).append(self._copy(artifact))
            self._events.setdefault(artifact.cycle_id, []).append(self._copy(event))
            return self._copy(artifact)


# Short alias is useful in local test fixtures and keeps the implementation
# name discoverable without making ``InMemoryRepository`` the only contract.
InMemoryRepository = InMemoryPostTrainingRepository


class CycleService:
    """Validates commands and owns the cycle state machine."""

    def __init__(self, repository: PostTrainingRepository, *, clock: Any = utc_now) -> None:
        self.repository = repository
        self.clock = clock

    async def create_trace(self, request: TraceCreate) -> TraceRead:
        trace = TraceRead(
            trace_id=uuid4(),
            created_at=self.clock(),
            **request.model_dump(),
        )
        return await self.repository.create_trace(trace)

    async def create_cycle(self, request: CycleCreate) -> CycleRead:
        config = request.config
        if config is None:  # guarded by CycleCreate, retained for type checkers
            raise InvalidCommand("config is required")
        for trace_id in request.trace_ids:
            if await self.repository.get_trace(trace_id) is None:
                raise InvalidCommand(f"trace {trace_id} does not exist")
        now = self.clock()
        cycle = CycleRead(
            cycle_id=uuid4(),
            trace_ids=list(request.trace_ids),
            config=config,
            status=CycleStatus.PENDING_APPROVAL,
            version=0,
            created_at=now,
            updated_at=now,
        )
        event = CycleEvent(
            event_id=uuid4(),
            cycle_id=cycle.cycle_id,
            sequence=1,
            event_type=EventType.CREATED,
            occurred_at=now,
            new_status=cycle.status,
        )
        return await self.repository.create_cycle(cycle, event)

    async def get_cycle(self, cycle_id: UUID) -> CycleRead:
        cycle = await self.repository.get_cycle(cycle_id)
        if cycle is None:
            raise CycleNotFound("cycle not found")
        return cycle

    async def list_events(
        self, cycle_id: UUID, after_sequence: int = 0, limit: int = 500
    ) -> CycleEventsRead:
        events = await self.repository.list_events(cycle_id, after_sequence, limit)
        next_after = events[-1].sequence if len(events) == limit and events else None
        return CycleEventsRead(cycle_id=cycle_id, events=events, next_after=next_after)

    async def list_artifacts(self, cycle_id: UUID) -> CycleArtifactsRead:
        return CycleArtifactsRead(
            cycle_id=cycle_id,
            artifacts=await self.repository.list_artifacts(cycle_id),
        )

    async def approve(self, cycle_id: UUID, request: DecisionRequest | None = None) -> CycleRead:
        return await self._transition(
            cycle_id,
            request,
            allowed={CycleStatus.PENDING_APPROVAL},
            new_status=CycleStatus.QUEUED,
            event_type=EventType.APPROVED,
        )

    async def reject(self, cycle_id: UUID, request: DecisionRequest | None = None) -> CycleRead:
        if request is None or request.reason is None or not request.reason.strip():
            raise InvalidCommand("reason is required when rejecting a cycle")
        return await self._transition(
            cycle_id,
            request,
            allowed={CycleStatus.PENDING_APPROVAL},
            new_status=CycleStatus.REJECTED,
            event_type=EventType.REJECTED,
        )

    async def cancel(self, cycle_id: UUID, request: DecisionRequest | None = None) -> CycleRead:
        cycle = await self.get_cycle(cycle_id)
        if cycle.status == CycleStatus.RUNNING:
            return await self._transition(
                cycle_id,
                request,
                allowed={CycleStatus.RUNNING},
                new_status=CycleStatus.CANCEL_REQUESTED,
                event_type=EventType.CANCEL_REQUESTED,
            )
        return await self._transition(
            cycle_id,
            request,
            allowed={
                CycleStatus.PENDING_APPROVAL,
                CycleStatus.QUEUED,
                CycleStatus.CANCEL_REQUESTED,
            },
            new_status=CycleStatus.CANCELLED,
            event_type=EventType.CANCELLED,
        )

    async def _transition(
        self,
        cycle_id: UUID,
        request: DecisionRequest | None,
        *,
        allowed: set[CycleStatus],
        new_status: CycleStatus,
        event_type: EventType,
    ) -> CycleRead:
        current = await self.get_cycle(cycle_id)
        if current.status not in allowed:
            raise InvalidCommand(
                f"cannot transition cycle from {current.status.value} to {new_status.value}"
            )
        expected_version = (
            current.version
            if request is None or request.expected_version is None
            else request.expected_version
        )
        if expected_version != current.version:
            raise VersionConflict(
                f"cycle version is {current.version}; expected {expected_version}"
            )
        now = self.clock()
        updated = current.model_copy(
            update={"status": new_status, "version": current.version + 1, "updated_at": now}
        )
        event = CycleEvent(
            event_id=uuid4(),
            cycle_id=cycle_id,
            sequence=current.version + 2,
            event_type=event_type,
            occurred_at=now,
            previous_status=current.status,
            new_status=new_status,
            reason=request.reason if request is not None else None,
        )
        return await self.repository.compare_and_set_cycle(updated, expected_version, event)


async def get_repository(request: Request) -> PostTrainingRepository:
    """Resolve one repository per application, overridable in tests."""

    repository = getattr(request.app.state, "post_training_repository", None)
    if repository is None:
        repository = InMemoryPostTrainingRepository()
        request.app.state.post_training_repository = repository
    return repository


async def get_service(
    repository: Annotated[PostTrainingRepository, Depends(get_repository)],
) -> PostTrainingService:
    """Resolve the service through the repository dependency."""

    return CycleService(repository)


def install_post_training_api(app: Any, repository: PostTrainingRepository | None = None) -> Any:
    """Install this router and optionally provide an application repository.

    This helper is an integration hook for an application factory; importing
    it does not import or mutate ``app.main``.
    """

    if repository is not None:
        app.state.post_training_repository = repository
    app.include_router(router)
    return app


router = APIRouter(prefix="/api", tags=["continuous post-training"])
post_training_router = router


def _http_error(error: RepositoryError) -> HTTPException:
    if isinstance(error, CycleNotFound):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error))
    if isinstance(error, VersionConflict):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error))
    return HTTPException(status_code=422, detail=str(error))


@router.post("/traces", response_model=TraceRead, status_code=status.HTTP_201_CREATED)
async def create_trace(
    request: TraceCreate,
    service: Annotated[PostTrainingService, Depends(get_service)],
) -> TraceRead:
    try:
        return await service.create_trace(request)
    except RepositoryError as error:
        raise _http_error(error) from error


@router.post("/cycles", response_model=CycleRead, status_code=status.HTTP_201_CREATED)
async def create_cycle(
    request: CycleCreate,
    service: Annotated[PostTrainingService, Depends(get_service)],
) -> CycleRead:
    try:
        return await service.create_cycle(request)
    except RepositoryError as error:
        raise _http_error(error) from error


@router.get("/cycles/{cycle_id}/status", response_model=CycleRead)
@router.get("/cycles/{cycle_id}", response_model=CycleRead)
async def get_cycle(
    cycle_id: UUID,
    service: Annotated[PostTrainingService, Depends(get_service)],
) -> CycleRead:
    try:
        return await service.get_cycle(cycle_id)
    except RepositoryError as error:
        raise _http_error(error) from error


@router.get("/cycles/{cycle_id}/events", response_model=CycleEventsRead)
async def get_cycle_events(
    cycle_id: UUID,
    service: Annotated[PostTrainingService, Depends(get_service)],
    after: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 500,
) -> CycleEventsRead:
    try:
        return await service.list_events(cycle_id, after, limit)
    except RepositoryError as error:
        raise _http_error(error) from error


@router.get("/cycles/{cycle_id}/artifacts", response_model=CycleArtifactsRead)
async def get_cycle_artifacts(
    cycle_id: UUID,
    service: Annotated[PostTrainingService, Depends(get_service)],
) -> CycleArtifactsRead:
    try:
        return await service.list_artifacts(cycle_id)
    except RepositoryError as error:
        raise _http_error(error) from error


@router.post("/cycles/{cycle_id}/approve", response_model=CycleRead)
async def approve_cycle(
    cycle_id: UUID,
    service: Annotated[PostTrainingService, Depends(get_service)],
    request: Annotated[DecisionRequest | None, Body()] = None,
) -> CycleRead:
    try:
        return await service.approve(cycle_id, request)
    except RepositoryError as error:
        raise _http_error(error) from error


@router.post("/cycles/{cycle_id}/reject", response_model=CycleRead)
async def reject_cycle(
    cycle_id: UUID,
    service: Annotated[PostTrainingService, Depends(get_service)],
    request: Annotated[DecisionRequest | None, Body()] = None,
) -> CycleRead:
    try:
        return await service.reject(cycle_id, request)
    except RepositoryError as error:
        raise _http_error(error) from error


@router.post("/cycles/{cycle_id}/cancel", response_model=CycleRead)
async def cancel_cycle(
    cycle_id: UUID,
    service: Annotated[PostTrainingService, Depends(get_service)],
    request: Annotated[DecisionRequest | None, Body()] = None,
) -> CycleRead:
    try:
        return await service.cancel(cycle_id, request)
    except RepositoryError as error:
        raise _http_error(error) from error


__all__ = [
    "ArtifactRead",
    "CycleArtifactsRead",
    "CycleConfig",
    "CycleCreate",
    "CycleEvent",
    "CycleEventsRead",
    "CycleRead",
    "CycleService",
    "CycleStatus",
    "DecisionRequest",
    "EventType",
    "FailureInfo",
    "InMemoryPostTrainingRepository",
    "InMemoryRepository",
    "PostTrainingRepository",
    "PostTrainingService",
    "TraceCreate",
    "TraceMessage",
    "TraceRead",
    "TraceRole",
    "get_repository",
    "get_service",
    "install_post_training_api",
    "post_training_router",
    "router",
]
