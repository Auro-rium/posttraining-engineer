"""Authenticated HTTP boundary for the objective worker.

The service intentionally exposes only training/replay scope.  Hidden
validation data remains inside a sealed evaluator process and is never put in
an API response.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any, Protocol

from fastapi import Depends, FastAPI, Header, HTTPException, status

from .engine import ServiceRecoveryEngine
from .models import (
    BenchmarkRequest,
    BenchmarkResponse,
    CurationRequest,
    CurationResponse,
    TrajectoryReference,
)


class ObjectiveWorkerUnavailable(RuntimeError):
    """No real benchmark execution boundary was configured."""


class BenchmarkExecutionAdapter(Protocol):
    def execute_benchmark(
        self, request: BenchmarkRequest, engine: ServiceRecoveryEngine
    ) -> tuple[Any, ...]: ...


class ObjectiveService:
    """Application service backed by one deterministic objective engine."""

    def __init__(
        self,
        engine: ServiceRecoveryEngine,
        auth_token: str,
        execution_adapter: BenchmarkExecutionAdapter | Any | None = None,
    ) -> None:
        if not auth_token:
            raise ValueError("auth_token is required")
        self.engine = engine
        self.auth_token = auth_token
        self.execution_adapter = execution_adapter

    def benchmark(self, request: BenchmarkRequest) -> BenchmarkResponse:
        if self.execution_adapter is None:
            raise ObjectiveWorkerUnavailable("benchmark execution adapter is required")
        target = getattr(self.execution_adapter, "execute_benchmark", None)
        if target is None and callable(self.execution_adapter):
            target = self.execution_adapter
        if target is None:
            raise ObjectiveWorkerUnavailable("benchmark execution adapter is invalid")
        request_engine = ServiceRecoveryEngine(
            seed=self.engine.seed,
            sealed=self.engine.sealed,
        )
        try:
            try:
                raw_trajectories = target(request, request_engine)
            except TypeError:
                raw_trajectories = target(request)
        except Exception as exc:
            raise ObjectiveWorkerUnavailable(
                f"benchmark execution failed: {type(exc).__name__}"
            ) from exc
        if not isinstance(raw_trajectories, (tuple, list)):
            raise ObjectiveWorkerUnavailable("benchmark adapter must return trajectories")
        references: list[TrajectoryReference] = []
        successes = 0
        for trajectory in raw_trajectories:
            if not hasattr(trajectory, "trajectory_id") or not hasattr(trajectory, "split"):
                raise ObjectiveWorkerUnavailable(
                    "benchmark adapter returned an invalid trajectory"
                )
            if trajectory.split is not request.split:
                raise ObjectiveWorkerUnavailable(
                    "benchmark trajectory split does not match request"
                )
            successes += int(trajectory.success)
            references.append(
                TrajectoryReference(
                    trajectory_id=trajectory.trajectory_id,
                    task_id=trajectory.task_id,
                    split=trajectory.split,
                    verified=False,
                )
            )
        benchmark_id = (
            "benchmark-"
            + hashlib.sha256(
                f"{request.run_id}:{request.split.value}:"
                f"{','.join(item.trajectory_id for item in references)}".encode()
            ).hexdigest()[:24]
        )
        return BenchmarkResponse(
            benchmark_id=benchmark_id,
            run_id=request.run_id,
            split=request.split,
            total_tasks=len(references),
            successful_tasks=successes,
            success_rate=successes / len(references) if references else 0.0,
            trajectory_references=tuple(references),
        )

    def verify_curation(self, request: CurationRequest) -> CurationResponse:
        confirmed = []
        for trajectory in request.trajectories:
            try:
                result = self.engine.verify(trajectory)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            confirmed.append(result.trajectory)
        try:
            dataset = self.engine.build_dataset(
                tuple(confirmed), run_id=request.run_id, experiment_id=request.experiment_id
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return dataset


def _auth_dependency(expected_token: str) -> Callable[[str | None, str | None], None]:
    def require_auth(
        authorization: str | None = Header(default=None),
        x_objective_token: str | None = Header(default=None),
    ) -> None:
        bearer = (
            authorization.removeprefix("Bearer ").strip()
            if authorization and authorization.startswith("Bearer ")
            else None
        )
        if bearer != expected_token and x_objective_token != expected_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="objective worker authentication required",
            )

    return require_auth


def create_objective_app(
    engine: ServiceRecoveryEngine | None = None,
    *,
    auth_token: str,
    execution_adapter: BenchmarkExecutionAdapter | Any | None = None,
) -> FastAPI:
    """Create the isolated objective worker application for local or ECS use."""

    service = ObjectiveService(engine or ServiceRecoveryEngine(), auth_token, execution_adapter)
    app = FastAPI(title="service-recovery-objective-worker")
    auth = _auth_dependency(auth_token)

    @app.post("/v1/benchmark", response_model=BenchmarkResponse, dependencies=[Depends(auth)])
    def benchmark(request: BenchmarkRequest) -> BenchmarkResponse:
        try:
            return service.benchmark(request)
        except ObjectiveWorkerUnavailable as exc:
            raise HTTPException(status_code=503, detail=f"BLOCKED: {exc}") from exc

    @app.post("/v1/verify-curation", response_model=CurationResponse, dependencies=[Depends(auth)])
    def verify_curation(request: CurationRequest) -> CurationResponse:
        return service.verify_curation(request)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "healthy", "service": "objective-worker"}

    return app


create_app = create_objective_app
