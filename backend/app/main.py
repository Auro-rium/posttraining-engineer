"""ASGI bootstrap and the intentionally small coordinator HTTP API."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.agents import DecisionProvider
from app.cloud_provider import build_cloud_decision_provider
from app.demo import LocalDemoDecisionProvider, verify_demo
from app.models import (
    CheckpointManifest,
    EvaluationReport,
    EvidenceLabel,
    Experiment,
    RunEvent,
    RunPhase,
    RunState,
)
from app.orchestrator import (
    TERMINAL_PHASES,
    InvalidRunStateError,
    Orchestrator,
)
from app.repository import (
    FirestoreRunRepository,
    InMemoryRunRepository,
    RunNotFoundError,
    RunRepository,
    VersionConflictError,
)
from app.settings import Settings, get_settings
from app.telemetry import configure_cloud_trace, current_trace_id


class APIModel(BaseModel):
    """Strict base model for coordinator request/response envelopes."""

    model_config = ConfigDict(extra="forbid")


class CreateRunRequest(APIModel):
    """The only supported target/environment plus bounded demo inputs."""

    target_model: Literal["google/functiongemma-270m-it"] = "google/functiongemma-270m-it"
    environment: Literal["AgentGym/WebShop"] = "AgentGym/WebShop"
    max_experiments: Literal[1, 2] = 2
    baseline_success: float = Field(default=0.35, ge=0.0, le=1.0)
    baseline_regression_success: float = Field(default=1.0, ge=0.0, le=1.0)
    baseline_action_validity: float = Field(default=1.0, ge=0.0, le=1.0)


class RunCreated(APIModel):
    run_id: str
    phase: RunPhase
    evidence_label: EvidenceLabel


class CommandAccepted(APIModel):
    run_id: str
    command: Literal["auto"]
    status: Literal["accepted", "already_running"]


class ExperimentsResponse(APIModel):
    run_id: str
    experiments: list[Experiment]


class VerifyDemoRequest(APIModel):
    run_id: str = Field(min_length=1)


def _repository_for(settings: Settings) -> RunRepository:
    """Select durable cloud persistence only when cloud mode is explicit."""

    if settings.environment == "cloud":
        return FirestoreRunRepository(
            project=settings.google_cloud_project,
            database=settings.firestore_database,
        )
    return InMemoryRunRepository()


class RunController:
    """Serialize mutations per run while allowing cancellation between auto steps."""

    def __init__(self, repository: RunRepository, orchestrator: Orchestrator) -> None:
        self.repository = repository
        self.orchestrator = orchestrator
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._auto_tasks: dict[str, asyncio.Task[None]] = {}

    async def step(self, run_id: str) -> RunState:
        async with self._locks[run_id]:
            return await self.orchestrator.step(run_id)

    async def cancel(self, run_id: str) -> RunState:
        async with self._locks[run_id]:
            return await self.orchestrator.cancel(run_id)

    def start_auto(self, run_id: str) -> bool:
        task = self._auto_tasks.get(run_id)
        if task is not None and not task.done():
            return False
        self._auto_tasks[run_id] = asyncio.create_task(
            self._run_auto(run_id), name=f"auto:{run_id}"
        )
        return True

    async def _run_auto(self, run_id: str) -> None:
        try:
            for _ in range(16):
                async with self._locks[run_id]:
                    state = await self.orchestrator.step(run_id)
                if state.phase in TERMINAL_PHASES:
                    return
                await asyncio.sleep(0)
            raise InvalidRunStateError(
                "automatic run exceeded the bounded state-machine step count"
            )
        except Exception:
            # The orchestrator persists a sanitized FAILED event for specialist
            # failures. The task exception is consumed here so the event loop never
            # logs raw provider details.
            return

    async def stop(self) -> None:
        tasks = [task for task in self._auto_tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def _event_sse(event: RunEvent) -> str:
    body = event.model_dump_json()
    return f"id: {event.event_id}\nevent: {event.type}\ndata: {body}\n\n"


def create_app(
    *,
    settings: Settings | None = None,
    repository: RunRepository | None = None,
    provider: DecisionProvider | None = None,
) -> FastAPI:
    """Create a coordinator app with injectable boundaries for tests and cloud."""

    runtime = settings or get_settings()
    decision_provider = provider or (
        build_cloud_decision_provider(runtime)
        if runtime.environment == "cloud"
        else LocalDemoDecisionProvider()
    )
    run_repository = repository or _repository_for(runtime)
    orchestrator = Orchestrator(run_repository, decision_provider)
    controller = RunController(run_repository, orchestrator)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if runtime.otel_export_to_cloud:
            if not runtime.google_cloud_project:
                raise RuntimeError("Cloud Trace export requires GOOGLE_CLOUD_PROJECT")
            configure_cloud_trace(
                project_id=runtime.google_cloud_project,
                service_name=runtime.otel_service_name,
            )
        yield
        await controller.stop()

    api = FastAPI(title=runtime.app_name, version="0.1.0", lifespan=lifespan)
    api.state.settings = runtime
    api.state.repository = run_repository
    api.state.provider = decision_provider
    api.state.controller = controller
    api.add_middleware(
        CORSMiddleware,
        allow_origins=runtime.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Last-Event-ID"],
    )

    @api.exception_handler(RunNotFoundError)
    async def not_found_handler(_: Request, exc: RunNotFoundError) -> Response:
        return _error_response(status.HTTP_404_NOT_FOUND, "run_not_found", str(exc))

    @api.exception_handler(VersionConflictError)
    async def conflict_handler(_: Request, __: VersionConflictError) -> Response:
        return _error_response(
            status.HTTP_409_CONFLICT,
            "version_conflict",
            "run state changed concurrently; retry the command",
        )

    @api.exception_handler(InvalidRunStateError)
    async def invalid_state_handler(_: Request, exc: InvalidRunStateError) -> Response:
        return _error_response(status.HTTP_409_CONFLICT, "invalid_run_state", str(exc))

    @api.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "role": runtime.service_role, "mode": runtime.environment}

    @api.post(
        f"{runtime.api_prefix}/runs",
        response_model=RunCreated,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_run(command: CreateRunRequest) -> RunCreated:
        # Baseline numbers arrive in the command and are therefore explanatory
        # until the objective cloud evaluator measures both checkpoints.
        label = EvidenceLabel.EXPLANATION
        state = RunState(
            target_model=command.target_model,
            environment=command.environment,
            champion=CheckpointManifest(
                success=command.baseline_success,
                regression_success=command.baseline_regression_success,
                action_validity=command.baseline_action_validity,
                evidence_label=label,
            ),
            max_experiments=command.max_experiments,
        )
        stored = await run_repository.create_run(state)
        await run_repository.append_event(
            RunEvent(
                run_id=stored.run_id,
                type="run.created",
                phase=stored.phase,
                payload={"evidence_label": label.value},
                trace_id=current_trace_id(),
            )
        )
        return RunCreated(
            run_id=stored.run_id,
            phase=stored.phase,
            evidence_label=stored.champion.evidence_label,
        )

    @api.get(f"{runtime.api_prefix}/runs/{{run_id}}", response_model=RunState)
    async def get_run(run_id: str) -> RunState:
        return await run_repository.get_run(run_id)

    @api.get(
        f"{runtime.api_prefix}/runs/{{run_id}}/experiments",
        response_model=ExperimentsResponse,
    )
    async def get_experiments(run_id: str) -> ExperimentsResponse:
        state = await run_repository.get_run(run_id)
        return ExperimentsResponse(run_id=run_id, experiments=state.experiments)

    @api.post(f"{runtime.api_prefix}/runs/{{run_id}}/step", response_model=RunState)
    async def step_run(run_id: str) -> RunState:
        return await controller.step(run_id)

    @api.post(
        f"{runtime.api_prefix}/runs/{{run_id}}/auto",
        response_model=CommandAccepted,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def auto_run(run_id: str) -> CommandAccepted:
        state = await run_repository.get_run(run_id)
        if state.phase in TERMINAL_PHASES:
            raise HTTPException(status_code=409, detail="run is already terminal")
        started = controller.start_auto(run_id)
        return CommandAccepted(
            run_id=run_id,
            command="auto",
            status="accepted" if started else "already_running",
        )

    @api.post(f"{runtime.api_prefix}/runs/{{run_id}}/cancel", response_model=RunState)
    async def cancel_run(run_id: str) -> RunState:
        return await controller.cancel(run_id)

    @api.get(f"{runtime.api_prefix}/runs/{{run_id}}/events")
    async def stream_events(
        request: Request,
        run_id: str,
        last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    ) -> StreamingResponse:
        await run_repository.get_run(run_id)

        async def generate() -> AsyncIterator[str]:
            cursor = last_event_id
            terminal_idle_polls = 0
            while True:
                if await request.is_disconnected():
                    return
                events = await run_repository.list_events(run_id, after_event_id=cursor)
                if events:
                    terminal_idle_polls = 0
                    for event in events:
                        cursor = event.event_id
                        yield _event_sse(event)
                    continue
                state = await run_repository.get_run(run_id)
                if state.phase in TERMINAL_PHASES:
                    terminal_idle_polls += 1
                    if terminal_idle_polls >= 1:
                        return
                yield ": heartbeat\n\n"
                await asyncio.sleep(0.25)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @api.post(f"{runtime.api_prefix}/demo/verify", response_model=EvaluationReport)
    async def demo_verify(command: VerifyDemoRequest) -> EvaluationReport:
        state = await run_repository.get_run(command.run_id)
        try:
            return await verify_demo(decision_provider, state)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return api


def _error_response(status_code: int, code: str, message: str) -> Response:
    """Return a small sanitized error envelope without raw provider context."""

    return Response(
        status_code=status_code,
        media_type="application/json",
        content=json.dumps({"error": {"code": code, "message": message}}),
    )


def bootstrap_app() -> Any:
    """Dispatch one image to the coordinator API or an ADK A2A service."""

    runtime = get_settings()
    if runtime.service_role == "coordinator":
        return create_app(settings=runtime)
    from app.adk_service import create_a2a_app

    role_app = FastAPI(title=f"{runtime.app_name}-{runtime.service_role}")

    @role_app.get("/health")
    async def role_health() -> dict[str, str]:
        return {"status": "ok", "role": runtime.service_role, "mode": runtime.environment}

    role_app.mount("/", create_a2a_app())
    return role_app


app = bootstrap_app()
