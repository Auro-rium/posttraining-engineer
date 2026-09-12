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

from .artifacts import (
    ObjectiveArtifactError,
    ObjectiveArtifactIntegrityError,
    ObjectiveArtifactNotFound,
)
from .engine import ServiceRecoveryEngine
from .execution import FunctionGemmaBenchmarkExecutionAdapter
from .models import (
    BenchmarkExecutionResult,
    BenchmarkRequest,
    BenchmarkResponse,
    CurationRequest,
    CurationResponse,
    Dataset,
    DatasetManifest,
    ObjectiveReadinessResponse,
    Trajectory,
    TrajectoryReference,
    deterministic_dataset_created_at,
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

    def resolve_trajectory_reference(self, reference: TrajectoryReference) -> Trajectory: ...

    def put_dataset(self, dataset: Dataset) -> DatasetManifest: ...


class InMemoryTrajectoryArtifactStore:
    """Thread-safe local artifact registry for contract tests and development."""

    def __init__(self) -> None:
        self._items: dict[str, Trajectory] = {}
        self._datasets: dict[str, Dataset] = {}
        self._lock = RLock()

    def put(self, trajectory: Trajectory) -> TrajectoryReference:
        if not trajectory.verified:
            raise ValueError("only verified trajectories may be stored")
        with self._lock:
            existing = self._items.get(trajectory.trajectory_id)
            if existing is not None and existing != trajectory:
                raise ValueError("trajectory ID is already bound to other bytes")
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

    def resolve_trajectory_reference(self, reference: TrajectoryReference) -> Trajectory:
        """Resolve against the stored identity, not caller metadata or flags."""

        if not reference.verified:
            raise ValueError("trajectory reference is not verified")
        if reference.split.value == "validation":
            raise ValueError("validation retrieval requires explicit evaluator scope")
        with self._lock:
            trajectory = self._items.get(reference.trajectory_id)
            if trajectory is None:
                raise KeyError(reference.trajectory_id)
            if (
                trajectory.task_id != reference.task_id
                or trajectory.split is not reference.split
                or not trajectory.verified
            ):
                raise ValueError("trajectory reference metadata does not match trusted index")
            return trajectory.model_copy(deep=True)

    def put_dataset(self, dataset: Dataset) -> DatasetManifest:
        """Persist the local test dataset with a stable immutable locator."""

        if not dataset.rows or any(
            not row.verifier_confirmed or row.source_type != "verified_replay"
            for row in dataset.rows
        ):
            raise ValueError("dataset rows must carry verifier-confirmed replay evidence")
        version = dataset.manifest.sha256[:16]
        manifest = dataset.manifest.model_copy(
            update={
                "s3_uri": (
                    f"s3://in-memory/objective/datasets/{dataset.manifest.dataset_id}.jsonl"
                    f"?versionId={version}"
                ),
                "created_at": deterministic_dataset_created_at(
                    dataset.manifest.dataset_id, dataset.manifest.sha256
                ),
            }
        )
        persisted = dataset.model_copy(update={"manifest": manifest}, deep=True)
        with self._lock:
            for row in persisted.rows:
                source = self._items.get(row.source_trajectory_id)
                if source is None or source.task_id != row.task_id or source.split is not row.split:
                    raise ValueError(
                        "dataset row source is not present in the trusted trajectory index"
                    )
            existing = self._datasets.get(dataset.manifest.dataset_id)
            if existing is not None and existing != persisted:
                raise ValueError("dataset ID is already bound to other bytes")
            self._datasets[dataset.manifest.dataset_id] = persisted
        return manifest


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

    def readiness(self) -> ObjectiveReadinessResponse:
        """Return metadata-only readiness without running a benchmark or model."""

        blockers: list[str] = []
        adapter_ready = False
        if type(self.execution_adapter) is not FunctionGemmaBenchmarkExecutionAdapter:
            blockers.append("functiongemma_adapter_unavailable")
        else:
            try:
                adapter_status = self.execution_adapter.readiness()
            except Exception:
                blockers.append("functiongemma_adapter_unavailable")
            else:
                adapter_ready = adapter_status.ready
                blockers.extend(adapter_status.blockers)

        store = self.artifact_store
        can_persist_trajectory = callable(getattr(store, "put", None))
        can_persist_dataset = callable(getattr(store, "put_dataset", None))
        can_resolve_references = callable(getattr(store, "resolve_trajectory_reference", None))
        benchmark_ready = adapter_ready and can_persist_trajectory
        curation_ready = (
            adapter_ready
            and can_persist_trajectory
            and can_persist_dataset
            and can_resolve_references
        )
        if not (can_persist_trajectory and can_persist_dataset and can_resolve_references):
            blockers.append("artifact_store_incomplete")

        capabilities = {
            "benchmark": benchmark_ready,
            "verify-curation": curation_ready,
        }
        status = "ready" if all(capabilities.values()) else "blocked"
        return ObjectiveReadinessResponse(
            status=status,
            capabilities=capabilities,
            blockers=tuple(dict.fromkeys(blockers)),
        )

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
        if self.artifact_store is None:
            raise HTTPException(
                status_code=503, detail="objective artifact persistence is required"
            )
        source_trajectories = list(request.trajectories)
        if request.trajectory_references:
            resolver = getattr(self.artifact_store, "resolve_trajectory_reference", None)
            if not callable(resolver):
                raise HTTPException(
                    status_code=503, detail="objective artifact reference resolver is unavailable"
                )
            for reference in request.trajectory_references:
                try:
                    # The persisted index decides whether this identity is
                    # authorized.  Caller-provided ``verified`` is only a
                    # claim that the resolver compares with that index.
                    trajectory = resolver(reference)
                except (
                    ObjectiveArtifactIntegrityError,
                    ObjectiveArtifactNotFound,
                    KeyError,
                    ValueError,
                ) as exc:
                    raise HTTPException(
                        status_code=422,
                        detail="trajectory reference is not trusted or resolvable",
                    ) from exc
                except ObjectiveArtifactError as exc:
                    raise HTTPException(
                        status_code=503, detail="objective artifact lookup is unavailable"
                    ) from exc
                except Exception as exc:
                    raise HTTPException(
                        status_code=503, detail="objective artifact lookup is unavailable"
                    ) from exc
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
                tuple(confirmed),
                run_id=request.run_id,
                experiment_id=request.experiment_id,
                scope=request.split,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        trajectory_persister = getattr(self.artifact_store, "put", None)
        if not callable(trajectory_persister):
            raise HTTPException(
                status_code=503, detail="objective trajectory persistence is unavailable"
            )
        for trajectory in confirmed:
            try:
                trajectory_persister(trajectory)
            except ObjectiveArtifactIntegrityError as exc:
                raise HTTPException(
                    status_code=422, detail="objective trajectory failed integrity checks"
                ) from exc
            except ObjectiveArtifactError as exc:
                raise HTTPException(
                    status_code=503, detail="objective trajectory persistence is unavailable"
                ) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=503, detail="objective trajectory persistence is unavailable"
                ) from exc
        persister = getattr(self.artifact_store, "put_dataset", None)
        if not callable(persister):
            raise HTTPException(
                status_code=503, detail="objective dataset persistence is unavailable"
            )
        try:
            persisted_manifest = persister(dataset)
            if not isinstance(persisted_manifest, DatasetManifest):
                raise TypeError("objective dataset persistence returned an invalid manifest")
            persisted = Dataset(
                manifest=persisted_manifest,
                rows=dataset.rows,
            )
        except ObjectiveArtifactIntegrityError as exc:
            raise HTTPException(
                status_code=422, detail="objective dataset failed integrity checks"
            ) from exc
        except ObjectiveArtifactError as exc:
            raise HTTPException(
                status_code=503, detail="objective dataset persistence is unavailable"
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=503,
                detail="objective dataset persistence returned invalid data",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail="objective dataset persistence is unavailable"
            ) from exc
        return persisted


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

    @app.get(
        "/v1/readiness",
        response_model=ObjectiveReadinessResponse,
        dependencies=[Depends(auth)],
    )
    def readiness() -> ObjectiveReadinessResponse:
        return service.readiness()

    @app.get("/v1/auth-probe", dependencies=[Depends(auth)])
    def auth_probe() -> dict[str, str]:
        """Return a stable, metadata-only response after auth succeeds."""
        return {"status": "authenticated", "service": "objective-worker"}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "healthy", "service": "objective-worker"}

    return app


create_app = create_objective_app
