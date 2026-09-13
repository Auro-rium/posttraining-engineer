"""Authenticated HTTP boundary for the objective worker.

The service intentionally exposes only training/replay scope.  Hidden
validation data remains inside a sealed evaluator process and is never put in
an API response.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Callable
from threading import RLock
from typing import Any, Protocol

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status

from app.posttraining.models import ArtifactKind, ArtifactReference
from app.posttraining.run_history import BenchmarkMetrics

from .artifacts import (
    ObjectiveArtifactError,
    ObjectiveArtifactIntegrityError,
    ObjectiveArtifactNotFound,
)
from .engine import ENGINE_VERSION, ServiceRecoveryEngine
from .execution import (
    FunctionGemmaBenchmarkExecutionAdapter,
    _objective_stage,
    objective_stage_trace,
)
from .models import (
    BenchmarkExecutionResult,
    BenchmarkRequest,
    BenchmarkResponse,
    CorrectionReplayOutcome,
    CorrectionReplayRequest,
    CorrectionReplayResponse,
    CurationRequest,
    CurationResponse,
    Dataset,
    DatasetManifest,
    EvidenceLabel,
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
        if not trajectory.verified or trajectory.verifier_success != trajectory.success:
            raise ValueError(
                "only verifier-confirmed replay trajectories with verifier outcome may be stored"
            )
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
                or trajectory.verifier_success != trajectory.success
            ):
                raise ValueError("trajectory reference metadata does not match trusted index")
            return trajectory.model_copy(deep=True)

    def put_dataset(self, dataset: Dataset) -> DatasetManifest:
        """Persist the local test dataset with a stable immutable locator."""

        if not dataset.rows or any(
            not row.verifier_confirmed
            or not row.verifier_success
            or row.source_type not in {"successful_replay", "repaired_replay"}
            for row in dataset.rows
        ):
            raise ValueError(
                "dataset rows must carry verifier-confirmed successful replay evidence"
            )
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
                if (
                    source is None
                    or source.task_id != row.task_id
                    or source.split is not row.split
                    or not source.verified
                    or source.success is not True
                    or source.verifier_success is not True
                    or source.repaired_from_trajectory_id != row.repaired_from_trajectory_id
                ):
                    raise ValueError(
                        "dataset row source is not a trusted successful trajectory"
                    )
                if row.repaired_from_trajectory_id is not None:
                    parent = self._items.get(row.repaired_from_trajectory_id)
                    if (
                        parent is None
                        or not parent.verified
                        or parent.success is not False
                        or parent.verifier_success is not False
                        or parent.task_id != row.task_id
                        or parent.split is not row.split
                    ):
                        raise ValueError("dataset repair lineage lacks a verified failed source")
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
        """Return static checks plus model-load/generation evidence from this process."""

        blockers: list[str] = []
        configuration_ready = False
        checkpoint_ready = False
        model_load_ready = False
        generation_ready = False
        if type(self.execution_adapter) is not FunctionGemmaBenchmarkExecutionAdapter:
            blockers.append("functiongemma_adapter_unavailable")
        else:
            configuration_ready = self.execution_adapter.configuration_ready
            try:
                adapter_status = self.execution_adapter.readiness()
            except Exception:
                blockers.append("functiongemma_adapter_unavailable")
            else:
                checkpoint_ready = adapter_status.checkpoint_verified
                blockers.extend(adapter_status.blockers)
            model_load_ready = self.execution_adapter.model_load_ready
            generation_ready = self.execution_adapter.generation_ready

        store = self.artifact_store
        can_persist_trajectory = callable(getattr(store, "put", None))
        can_persist_dataset = callable(getattr(store, "put_dataset", None))
        can_resolve_references = callable(getattr(store, "resolve_trajectory_reference", None))
        artifact_store_ready = (
            can_persist_trajectory and can_persist_dataset and can_resolve_references
        )
        execution_ready = all(
            (
                configuration_ready,
                checkpoint_ready,
                model_load_ready,
                generation_ready,
                artifact_store_ready,
            )
        )
        if not model_load_ready:
            blockers.append("model_load_not_verified")
        if not generation_ready:
            blockers.append("generation_not_verified")
        benchmark_ready = execution_ready and can_persist_trajectory
        curation_ready = (
            execution_ready
            and can_persist_trajectory
            and can_persist_dataset
            and can_resolve_references
        )
        if not artifact_store_ready:
            blockers.append("artifact_store_incomplete")

        capabilities = {
            "benchmark": benchmark_ready,
            "verify-curation": curation_ready,
        }
        status = "ready" if execution_ready and all(capabilities.values()) else "blocked"
        return ObjectiveReadinessResponse(
            status=status,
            configuration_ready=configuration_ready,
            checkpoint_ready=checkpoint_ready,
            model_load_ready=model_load_ready,
            generation_ready=generation_ready,
            artifact_store_ready=artifact_store_ready,
            execution_ready=execution_ready,
            capabilities=capabilities,
            blockers=tuple(dict.fromkeys(blockers)),
        )

    def benchmark(
        self, request: BenchmarkRequest, *, correlation_id: str | None = None
    ) -> BenchmarkResponse:
        """Execute one benchmark under an isolated, metadata-only trace context."""

        selected_id = correlation_id or uuid.uuid4().hex
        if re.fullmatch(r"[0-9a-f]{32}", selected_id) is None:
            selected_id = uuid.uuid4().hex
        policy = getattr(self.execution_adapter, "policy", None)
        revision = getattr(policy, "revision", None)
        if not isinstance(revision, str):
            revision = None
        with objective_stage_trace(selected_id, revision):
            with _objective_stage("BENCHMARK_COMPLETE"):
                return self._benchmark(request)

    def _benchmark(self, request: BenchmarkRequest) -> BenchmarkResponse:
        if self.execution_adapter is None:
            raise ObjectiveWorkerUnavailable("benchmark execution adapter is required")
        if self.artifact_store is None:
            raise ObjectiveWorkerUnavailable("trajectory artifact store is required")
        target = getattr(self.execution_adapter, "execute_benchmark", None)
        if target is None and callable(self.execution_adapter):
            target = self.execution_adapter
        if target is None:
            raise ObjectiveWorkerUnavailable("benchmark execution adapter is invalid")
        effective_seed = self.engine.seed if request.seed is None else request.seed
        request_engine = ServiceRecoveryEngine(seed=effective_seed, sealed=self.engine.sealed)
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
        expected_task_ids = request.execution_task_ids
        if len(raw_trajectories) != len(expected_task_ids):
            raise ObjectiveWorkerUnavailable(
                "benchmark adapter returned incomplete task cardinality"
            )
        actual_task_ids = tuple(item.task_id for item in raw_trajectories)
        if actual_task_ids != expected_task_ids:
            raise ObjectiveWorkerUnavailable("benchmark adapter task IDs do not match request")
        references: list[TrajectoryReference] = []
        successes = 0
        for trajectory in raw_trajectories:
            if trajectory.split is not request.split:
                raise ObjectiveWorkerUnavailable(
                    "benchmark trajectory split does not match request"
                )
            try:
                with _objective_stage("TRAJECTORY_VERIFY"):
                    confirmed = request_engine.verify(trajectory).trajectory
                with _objective_stage("S3_PERSIST"):
                    reference = self.artifact_store.put(confirmed)
            except Exception as exc:
                raise ObjectiveWorkerUnavailable(
                    f"benchmark trajectory failed verification or storage: {type(exc).__name__}"
                ) from exc
            successes += int(confirmed.success)
            references.append(reference)
        success_rate = successes / len(references)
        metrics = BenchmarkMetrics(
            aggregate=success_rate,
            per_environment={ENGINE_VERSION: success_rate},
        )
        benchmark_id = (
            "benchmark-"
            + hashlib.sha256(
                ":".join(
                    (
                        request.run_id,
                        request.split.value,
                        request.suite,
                        request.suite_version,
                        request.model_sha256,
                        str(effective_seed),
                        ",".join(item.trajectory_id for item in references),
                    )
                ).encode()
            ).hexdigest()[:24]
        )
        manifest = {
            "schema_version": "service-recovery-benchmark-manifest-v1",
            "benchmark_id": benchmark_id,
            "run_id": request.run_id,
            "suite": request.suite,
            "suite_version": request.suite_version,
            "model_id": request.model_uri,
            "model_sha256": request.model_sha256,
            "seed": effective_seed,
            "split": request.split.value,
            "objective_engine_version": ENGINE_VERSION,
            "num_episodes": len(references),
            "task_ids": list(expected_task_ids),
        }
        manifest_sha256 = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        report_artifact = None
        report_writer = getattr(self.artifact_store, "put_benchmark_report", None)
        if request.output_s3_uri is not None and callable(report_writer):
            report = {
                **manifest,
                "manifest_sha256": manifest_sha256,
                "metrics": metrics.model_dump(mode="json"),
                "successful_episodes": successes,
                "trajectory_references": [
                    item.model_dump(mode="json") for item in references
                ],
            }
            try:
                with _objective_stage("S3_PERSIST"):
                    stored_report = report_writer(
                        output_s3_uri=request.output_s3_uri,
                        report=report,
                        split=request.split,
                    )
            except Exception as exc:
                raise ObjectiveWorkerUnavailable(
                    "benchmark report persistence is unavailable"
                ) from exc
            report_artifact = ArtifactReference(
                artifact_id=f"benchmark-report-{stored_report.sha256[:24]}",
                kind=ArtifactKind.REPORT,
                uri=stored_report.version_ref,
                sha256=stored_report.sha256,
                size_bytes=stored_report.size_bytes,
                metadata={
                    "manifest_sha256": manifest_sha256,
                    "objective_engine_version": ENGINE_VERSION,
                    "suite": request.suite,
                    "suite_version": request.suite_version,
                },
            )
        verified = report_artifact is not None
        return BenchmarkResponse(
            benchmark_id=benchmark_id,
            run_id=request.run_id,
            suite=request.suite,
            suite_version=request.suite_version,
            model_id=request.model_uri,
            model_sha256=request.model_sha256,
            seed=effective_seed,
            split=request.split,
            metrics=metrics,
            trajectory_references=tuple(references),
            report_artifact=report_artifact,
            manifest_sha256=manifest_sha256,
            evidence_label=EvidenceLabel.LIVE if verified else EvidenceLabel.EXPLANATION,
            verified=verified,
        )

    def replay_corrections(self, request: CorrectionReplayRequest) -> CorrectionReplayResponse:
        """Replay curator proposals against their stored failed source trajectories."""

        if self.artifact_store is None:
            raise HTTPException(
                status_code=503, detail="objective trajectory persistence is required"
            )
        getter = getattr(self.artifact_store, "get", None)
        persister = getattr(self.artifact_store, "put", None)
        if not callable(getter) or not callable(persister):
            raise HTTPException(
                status_code=503, detail="objective trajectory storage is unavailable"
            )

        outcomes: list[CorrectionReplayOutcome] = []
        for proposal in request.proposals:
            try:
                source = getter(proposal.source_trajectory_id)
            except ObjectiveArtifactNotFound as exc:
                raise HTTPException(
                    status_code=422, detail="correction source is not a stored failure"
                ) from exc
            except ObjectiveArtifactIntegrityError as exc:
                raise HTTPException(
                    status_code=422, detail="correction source failed artifact integrity"
                ) from exc
            except ObjectiveArtifactError as exc:
                raise HTTPException(
                    status_code=503, detail="correction source lookup is unavailable"
                ) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=503, detail="correction source lookup is unavailable"
                ) from exc

            if (
                source is None
                or source.trajectory_id != proposal.source_trajectory_id
                or source.task_id != proposal.task_id
                or source.split is not proposal.split
                or source.split is not request.split
                or not source.verified
                or source.success is not False
                or source.verifier_success is not False
            ):
                raise HTTPException(
                    status_code=422,
                    detail="correction source must be a stored verifier-confirmed failure",
                )
            try:
                checked_source = self.engine.verify(source).trajectory
            except ValueError as exc:
                raise HTTPException(
                    status_code=422, detail="correction source failed deterministic replay"
                ) from exc
            if checked_source.verifier_success is not False:
                raise HTTPException(
                    status_code=422,
                    detail="correction source is not a verifier-confirmed failure",
                )

            try:
                candidate = self.engine.run_episode(
                    proposal.task_id,
                    proposal.actions,
                    split=proposal.split,
                    repaired_from_trajectory_id=proposal.source_trajectory_id,
                )
                replay = self.engine.verify(candidate)
            except (TypeError, ValueError, KeyError):
                outcomes.append(
                    CorrectionReplayOutcome(
                        proposal_id=proposal.proposal_id,
                        source_trajectory_id=proposal.source_trajectory_id,
                        task_id=proposal.task_id,
                        split=proposal.split,
                        status="REJECTED",
                        reason="invalid_correction",
                    )
                )
                continue
            if not replay.verified or not replay.replayed_success:
                outcomes.append(
                    CorrectionReplayOutcome(
                        proposal_id=proposal.proposal_id,
                        source_trajectory_id=proposal.source_trajectory_id,
                        task_id=proposal.task_id,
                        split=proposal.split,
                        status="REJECTED",
                        reason="replay_failed",
                    )
                )
                continue

            try:
                reference = persister(replay.trajectory)
            except ObjectiveArtifactIntegrityError as exc:
                raise HTTPException(
                    status_code=422, detail="passed correction failed artifact integrity"
                ) from exc
            except ObjectiveArtifactError as exc:
                raise HTTPException(
                    status_code=503, detail="passed correction persistence is unavailable"
                ) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=503, detail="passed correction persistence is unavailable"
                ) from exc
            outcomes.append(
                CorrectionReplayOutcome(
                    proposal_id=proposal.proposal_id,
                    source_trajectory_id=proposal.source_trajectory_id,
                    task_id=proposal.task_id,
                    split=proposal.split,
                    status="PASS",
                    reason="replay_passed",
                    trajectory_reference=reference,
                )
            )

        return CorrectionReplayResponse(
            run_id=request.run_id,
            experiment_id=request.experiment_id,
            split=request.split,
            outcomes=tuple(outcomes),
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
        for trajectory in confirmed:
            parent_id = trajectory.repaired_from_trajectory_id
            if parent_id is None:
                continue
            try:
                parent = self.artifact_store.get(parent_id)
                if (
                    parent is None
                    or not parent.verified
                    or parent.success is not False
                    or parent.verifier_success is not False
                    or parent.task_id != trajectory.task_id
                    or parent.split is not trajectory.split
                ):
                    raise HTTPException(
                        status_code=422,
                        detail="repair lineage requires a stored verifier-confirmed failed source",
                    )
                checked_parent = self.engine.verify(parent).trajectory
                if checked_parent.verifier_success is not False:
                    raise HTTPException(
                        status_code=422,
                        detail="repair lineage source did not replay as a failure",
                    )
            except HTTPException:
                raise
            except (
                ObjectiveArtifactIntegrityError,
                ObjectiveArtifactNotFound,
                KeyError,
                ValueError,
            ) as exc:
                raise HTTPException(
                    status_code=422,
                    detail="repair lineage source is not trusted or resolvable",
                ) from exc
            except ObjectiveArtifactError as exc:
                raise HTTPException(
                    status_code=503, detail="repair lineage lookup is unavailable"
                ) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=503, detail="repair lineage lookup is unavailable"
                ) from exc
        successful = tuple(
            trajectory
            for trajectory in confirmed
            if trajectory.success and trajectory.verifier_success is True
        )
        try:
            dataset = self.engine.build_dataset(
                successful,
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
        for trajectory in successful:
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
    def benchmark(request: BenchmarkRequest, response: Response) -> BenchmarkResponse:
        correlation_id = uuid.uuid4().hex
        response.headers["X-Objective-Correlation-ID"] = correlation_id
        try:
            return service.benchmark(request, correlation_id=correlation_id)
        except ObjectiveWorkerUnavailable as exc:
            raise HTTPException(
                status_code=503,
                detail="objective benchmark unavailable",
                headers={"X-Objective-Correlation-ID": correlation_id},
            ) from exc

    @app.post("/v1/verify-curation", response_model=CurationResponse, dependencies=[Depends(auth)])
    def verify_curation(request: CurationRequest) -> CurationResponse:
        return service.verify_curation(request)

    @app.post(
        "/v1/replay-corrections",
        response_model=CorrectionReplayResponse,
        dependencies=[Depends(auth)],
    )
    def replay_corrections(request: CorrectionReplayRequest) -> CorrectionReplayResponse:
        return service.replay_corrections(request)

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

    @app.get("/v1/health", dependencies=[Depends(auth)])
    def authenticated_health() -> dict[str, str]:
        """Return a metadata-only health response for coordinator callers."""
        return {"status": "healthy", "service": "objective-worker"}

    @app.get("/health")
    def health() -> dict[str, str]:
        """Private container/ALB health route; do not use as the public probe."""
        return {"status": "healthy", "service": "objective-worker"}

    return app


create_app = create_objective_app
