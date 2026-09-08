"""Authenticated HTTP boundary for the objective worker.

The service intentionally exposes only training/replay scope.  Hidden
validation data remains inside a sealed evaluator process and is never put in
an API response.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from threading import RLock
from typing import Any, Protocol

from fastapi import Depends, FastAPI, Header, HTTPException, status

from .engine import ServiceRecoveryEngine
from .models import (
    BenchmarkExecutionResult,
    BenchmarkRequest,
    BenchmarkResponse,
    CurationRequest,
    CurationResponse,
    Trajectory,
    TrajectoryReference,
)


class ObjectiveWorkerUnavailable(RuntimeError):
    """No real benchmark execution boundary was configured."""


class BenchmarkExecutionAdapter(Protocol):
    def execute_benchmark(
        self, request: BenchmarkRequest, engine: ServiceRecoveryEngine
    ) -> BenchmarkExecutionResult: ...


class TrajectoryArtifactStore(Protocol):
    """Content-addressed reference store used by benchmark and curation."""

    def put(self, trajectory: Trajectory) -> TrajectoryReference: ...

    def get(self, trajectory_id: str) -> Trajectory | None: ...


class InMemoryTrajectoryArtifactStore:
    """Thread-safe local artifact registry for contract tests and development."""

    def __init__(self) -> None:
        self._items: dict[str, Trajectory] = {}
        self._lock = RLock()

    def put(self, trajectory: Trajectory) -> TrajectoryReference:
        if not trajectory.verified:
            raise ValueError("only verified trajectories may be stored")
        with self._lock:
            self._items[trajectory.trajectory_id] = trajectory.model_copy(deep=True)
        return TrajectoryReference(
            trajectory_id=trajectory.trajectory_id,
            task_id=trajectory.task_id,
            split=trajectory.split,
            verified=True,
        )

    def get(self, trajectory_id: str) -> Trajectory | None:
        with self._lock:
            trajectory = self._items.get(trajectory_id)
            return trajectory.model_copy(deep=True) if trajectory is not None else None


class ObjectiveService:
    """Application service backed by one deterministic objective engine."""

    def __init__(
        self,
        engine: ServiceRecoveryEngine,
        auth_token: str,
        execution_adapter: BenchmarkExecutionAdapter | Any | None = None,
        artifact_store: TrajectoryArtifactStore | None = None,
    ) -> None:
        if not auth_token:
            raise ValueError("auth_token is required")
        self.engine = engine
        self.auth_token = auth_token
        self.execution_adapter = execution_adapter
        self.artifact_store = artifact_store

    def benchmark(self, request: BenchmarkRequest) -> BenchmarkResponse:
        if self.execution_adapter is None:
            raise ObjectiveWorkerUnavailable("benchmark execution adapter is required")
        if self.artifact_store is None:
            raise ObjectiveWorkerUnavailable("trajectory artifact store is required")
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
                raw_result = target(request, request_engine)
            except TypeError:
                raw_result = target(request)
        except Exception as exc:
            raise ObjectiveWorkerUnavailable(
                f"benchmark execution failed: {type(exc).__name__}"
            ) from exc
        try:
            if not isinstance(raw_result, BenchmarkExecutionResult):
                raise ObjectiveWorkerUnavailable(
                    "benchmark adapter must return typed execution result"
                )
            raw_result = BenchmarkExecutionResult.model_validate(
                raw_result.model_dump(mode="python")
            )
            raw_trajectories = raw_result.trajectories
        except ObjectiveWorkerUnavailable:
            raise
        except Exception as exc:
            raise ObjectiveWorkerUnavailable(
                f"benchmark adapter returned malformed execution result: {type(exc).__name__}"
            ) from exc
        if len(raw_trajectories) != len(request.task_ids):
            raise ObjectiveWorkerUnavailable(
                "benchmark adapter returned incomplete task cardinality"
            )
        actual_task_ids = tuple(item.task_id for item in raw_trajectories)
        if actual_task_ids != request.task_ids:
            raise ObjectiveWorkerUnavailable("benchmark adapter task IDs do not match request")
        references: list[TrajectoryReference] = []
        successes = 0
        for trajectory in raw_trajectories:
            if trajectory.split is not request.split:
                raise ObjectiveWorkerUnavailable(
                    "benchmark trajectory split does not match request"
                )
            try:
                confirmed = self.engine.verify(trajectory).trajectory
                reference = self.artifact_store.put(confirmed)
            except Exception as exc:
                raise ObjectiveWorkerUnavailable(
                    f"benchmark trajectory failed verification or storage: {type(exc).__name__}"
                ) from exc
            successes += int(confirmed.success)
            references.append(reference)
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
        source_trajectories = list(request.trajectories)
        if request.trajectory_references:
            if self.artifact_store is None:
                raise HTTPException(status_code=422, detail="trajectory artifact store is required")
            for reference in request.trajectory_references:
                if not reference.verified:
                    raise HTTPException(
                        status_code=422, detail="trajectory reference is not verified"
                    )
                trajectory = self.artifact_store.get(reference.trajectory_id)
                if trajectory is None:
                    raise HTTPException(
                        status_code=422, detail="trajectory reference is not resolvable"
                    )
                if (
                    trajectory.task_id != reference.task_id
                    or trajectory.split is not reference.split
                    or not trajectory.verified
                ):
                    raise HTTPException(
                        status_code=422,
                        detail="trajectory reference metadata does not match stored artifact",
                    )
                source_trajectories.append(trajectory)
        confirmed = []
        for trajectory in source_trajectories:
            if trajectory.split is not request.split:
                raise HTTPException(
                    status_code=422, detail="trajectory split does not match replay scope"
                )
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
    artifact_store: TrajectoryArtifactStore | None = None,
) -> FastAPI:
    """Create the isolated objective worker application for local or ECS use."""

    service = ObjectiveService(
        engine or ServiceRecoveryEngine(), auth_token, execution_adapter, artifact_store
    )
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

    @app.get("/v1/auth-probe", dependencies=[Depends(auth)])
    def auth_probe() -> dict[str, str]:
        """Return a stable, metadata-only response after auth succeeds."""
        return {"status": "authenticated", "service": "objective-worker"}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "healthy", "service": "objective-worker"}

    return app


create_app = create_objective_app
