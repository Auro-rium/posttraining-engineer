"""Fail-closed cloud implementation of the specialist decision provider.

This module contains composition and transport code only. Gemini/ADK research
services and the execution service are reached through authenticated A2A calls;
no local demo fixtures are available on this path.
"""

from __future__ import annotations

import asyncio
import importlib
import time
from enum import StrEnum
from typing import Any, Protocol
from uuid import uuid4

import httpx
from pydantic import Field, model_validator

from app.agents import CuratedDataset, DecisionProvider
from app.cloud import VertexJobHandle, VertexTrainingLauncher
from app.models import (
    ArtifactKind,
    ArtifactRef,
    DatasetManifest,
    DatasetSplit,
    DomainModel,
    EvaluationReport,
    EvidenceLabel,
    Experiment,
    FailureCluster,
    Hypothesis,
    JobStatus,
    QLoRAConfig,
    RunState,
    SFTExample,
    TrainingResult,
    Trajectory,
)
from app.settings import Settings
from app.telemetry import async_telemetry_span, current_trace_id


class CloudProviderError(RuntimeError):
    """Raised when cloud evidence is missing, malformed, or unverifiable."""


class A2AOperation(StrEnum):
    ANALYZE_FAILURES = "analyze_failures"
    FORM_HYPOTHESIS = "form_hypothesis"
    CURATE_DATASET = "curate_dataset"
    DESIGN_TRAINING = "design_training"
    BENCHMARK = "benchmark"
    EVALUATE = "evaluate"
    TRAINING_EVIDENCE = "training_evidence"


class A2AOperationRequest(DomainModel):
    operation: A2AOperation
    run_id: str
    schema_version: str = "1.0"
    payload: dict[str, Any]


class A2AOperationResponse(DomainModel):
    operation: A2AOperation
    schema_version: str = "1.0"
    payload: dict[str, Any]


class IdentityTokenProvider(Protocol):
    async def token(self, audience: str) -> str: ...


class GoogleIdentityTokenProvider:
    """Mint Cloud Run ID tokens lazily through application-default credentials."""

    async def token(self, audience: str) -> str:
        def fetch() -> str:
            try:
                from google.auth.transport.requests import Request
                from google.oauth2.id_token import fetch_id_token
            except ImportError as exc:  # pragma: no cover - cloud extra only
                raise CloudProviderError("Google authentication dependencies are missing") from exc
            return str(fetch_id_token(Request(), audience))  # type: ignore[no-untyped-call]

        return await asyncio.to_thread(fetch)


class TypedA2ATransport(Protocol):
    async def request(
        self,
        *,
        operation: A2AOperation,
        run_id: str,
        payload: DomainModel,
        idempotency_key: str,
    ) -> dict[str, Any]: ...


class AuthenticatedHTTPA2ATransport:
    """Bounded JSON-RPC A2A transport authenticated with a Cloud Run ID token."""

    _RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

    def __init__(
        self,
        *,
        service_url: str,
        token_provider: IdentityTokenProvider,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not service_url.startswith("https://"):
            raise ValueError("cloud A2A service URLs must use HTTPS")
        if not 1 <= max_attempts <= 5:
            raise ValueError("max_attempts must be between one and five")
        self.service_url = service_url.rstrip("/") + "/"
        self.audience = service_url.rstrip("/")
        self.token_provider = token_provider
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self._client = client

    async def request(
        self,
        *,
        operation: A2AOperation,
        run_id: str,
        payload: DomainModel,
        idempotency_key: str,
    ) -> dict[str, Any]:
        async with async_telemetry_span(
            "a2a.request",
            attributes={
                "run_id": run_id,
                "operation": operation.value,
                "retry_count": self.max_attempts - 1,
            },
        ):
            return await self._request(
                operation=operation,
                run_id=run_id,
                payload=payload,
                idempotency_key=idempotency_key,
            )

    async def _request(
        self,
        *,
        operation: A2AOperation,
        run_id: str,
        payload: DomainModel,
        idempotency_key: str,
    ) -> dict[str, Any]:
        message_id = f"msg_{uuid4().hex}"
        request_data = A2AOperationRequest(
            operation=operation,
            run_id=run_id,
            payload=payload.model_dump(mode="json"),
        )
        params = _build_send_message_params(request_data, message_id=message_id)
        body = {
            "jsonrpc": "2.0",
            "id": message_id,
            "method": "SendMessage",
            "params": params,
        }
        trace_id = current_trace_id()
        token = await self.token_provider.token(self.audience)
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Idempotency-Key": idempotency_key,
        }
        if trace_id:
            headers["X-Cloud-Trace-Context"] = f"{trace_id}/0;o=1"
            headers["traceparent"] = f"00-{trace_id}-0000000000000001-01"

        client = self._client or httpx.AsyncClient(timeout=self.timeout_seconds)
        owns_client = self._client is None
        try:
            for attempt in range(1, self.max_attempts + 1):
                try:
                    response = await client.post(self.service_url, json=body, headers=headers)
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    if attempt == self.max_attempts:
                        raise CloudProviderError(
                            f"A2A {operation.value} transport failed after {attempt} attempts"
                        ) from exc
                    await asyncio.sleep(0.25 * 2 ** (attempt - 1))
                    continue
                if response.status_code in self._RETRYABLE_STATUS and attempt < self.max_attempts:
                    await asyncio.sleep(0.25 * 2 ** (attempt - 1))
                    continue
                if not 200 <= response.status_code < 300:
                    raise CloudProviderError(
                        f"A2A {operation.value} returned HTTP {response.status_code}"
                    )
                try:
                    response_body = response.json()
                except ValueError as exc:
                    raise CloudProviderError("A2A response is not valid JSON") from exc
                return _extract_a2a_payload(response_body, expected_operation=operation)
            raise CloudProviderError(f"A2A {operation.value} exhausted retry budget")
        finally:
            if owns_client:
                await client.aclose()


def _build_send_message_params(
    request: A2AOperationRequest, *, message_id: str
) -> dict[str, Any]:
    """Serialize the official A2A v1 ``SendMessageRequest`` protobuf JSON shape."""

    try:
        a2a_pb2 = importlib.import_module("a2a.types.a2a_pb2")
        json_format = importlib.import_module("google.protobuf.json_format")
    except ImportError as exc:  # pragma: no cover - cloud extra only
        raise CloudProviderError("A2A protocol dependencies are missing") from exc
    message = json_format.ParseDict(
        {
            "message": {
                "messageId": message_id,
                "role": "ROLE_USER",
                "parts": [{"data": request.model_dump(mode="json")}],
            }
        },
        a2a_pb2.SendMessageRequest(),
    )
    result = json_format.MessageToDict(message)
    if not isinstance(result, dict):  # pragma: no cover - protobuf contract guard
        raise CloudProviderError("could not serialize A2A SendMessageRequest")
    return result


def _extract_a2a_payload(body: Any, *, expected_operation: A2AOperation) -> dict[str, Any]:
    """Extract exactly one typed data artifact from an A2A JSON-RPC response."""

    if not isinstance(body, dict) or body.get("jsonrpc") != "2.0":
        raise CloudProviderError("malformed A2A JSON-RPC response")
    if body.get("error") is not None:
        error = body["error"]
        code = error.get("code", "unknown") if isinstance(error, dict) else "unknown"
        raise CloudProviderError(f"A2A operation failed with code {code}")
    result_json = body.get("result")
    if not isinstance(result_json, dict):
        raise CloudProviderError("A2A response is missing a result object")
    try:
        a2a_pb2 = importlib.import_module("a2a.types.a2a_pb2")
        json_format = importlib.import_module("google.protobuf.json_format")

        wire_response = json_format.ParseDict(result_json, a2a_pb2.SendMessageResponse())
    except Exception as exc:
        raise CloudProviderError("A2A result is not a valid SendMessageResponse") from exc
    payload_kind = wire_response.WhichOneof("payload")
    if payload_kind == "task":
        artifacts = list(wire_response.task.artifacts)
        if len(artifacts) != 1:
            raise CloudProviderError("A2A task response must contain exactly one artifact")
        parts = list(artifacts[0].parts)
    elif payload_kind == "message":
        parts = list(wire_response.message.parts)
    else:
        raise CloudProviderError("A2A response contains neither task nor message payload")
    if len(parts) != 1 or parts[0].WhichOneof("content") != "data":
        raise CloudProviderError("A2A response must contain exactly one typed data part")
    data = json_format.MessageToDict(parts[0].data)
    try:
        response = A2AOperationResponse.model_validate(data)
    except Exception as exc:
        raise CloudProviderError("A2A data part has an invalid response schema") from exc
    if response.operation is not expected_operation:
        raise CloudProviderError("A2A response operation does not match the request")
    return response.payload


class BenchmarkExecutionResult(DomainModel):
    trajectories: list[Trajectory] = Field(min_length=1)
    artifact: ArtifactRef
    evidence_label: EvidenceLabel
    provenance_complete: bool

    @model_validator(mode="after")
    def require_verifiable_train_evidence(self) -> BenchmarkExecutionResult:
        if self.evidence_label is EvidenceLabel.EXPLANATION or not self.provenance_complete:
            raise ValueError("benchmark evidence is not independently verifiable")
        if self.artifact.kind is not ArtifactKind.TRAJECTORIES:
            raise ValueError("benchmark must reference a trajectories artifact")
        if any(item.split is not DatasetSplit.TRAIN for item in self.trajectories):
            raise ValueError("research benchmark may contain train trajectories only")
        return self


class EvaluationExecutionResult(DomainModel):
    report: EvaluationReport

    @model_validator(mode="after")
    def require_verifiable_report(self) -> EvaluationExecutionResult:
        report = self.report
        if report.evidence_label is EvidenceLabel.EXPLANATION or not report.provenance_complete:
            raise ValueError("evaluation evidence is not independently verifiable")
        if report.artifact is None or report.artifact.kind is not ArtifactKind.EVALUATION_REPORT:
            raise ValueError("evaluation must reference a hashed report artifact")
        return self


class FailureAnalysisRequest(DomainModel):
    trajectories_artifact: ArtifactRef
    trajectory_ids: list[str] = Field(min_length=1)


class FailureAnalysisResponse(DomainModel):
    clusters: list[FailureCluster] = Field(min_length=1)


class HypothesisRequest(DomainModel):
    failure_clusters: list[FailureCluster] = Field(min_length=1)


class DatasetCurationRequest(DomainModel):
    hypothesis: Hypothesis
    trajectories_artifact: ArtifactRef


class DatasetCurationResponse(DomainModel):
    manifest: DatasetManifest
    examples: list[SFTExample] = Field(min_length=1)


class TrainingDesignRequest(DomainModel):
    hypothesis: Hypothesis
    dataset: DatasetManifest
    previous_configs: list[QLoRAConfig] = Field(default_factory=list)
    allowed_ranks: tuple[int, ...] = (8, 16, 32)
    allowed_learning_rates: tuple[float, ...] = (5e-5, 1e-4, 2e-4)
    allowed_epochs: tuple[int, ...] = (2, 3, 5)
    allowed_dropouts: tuple[float, ...] = (0.0, 0.05)


class BenchmarkRequest(DomainModel):
    target_model: str
    champion_model_uri: str
    environment: str
    split: DatasetSplit = DatasetSplit.TRAIN


class EvaluationRequest(DomainModel):
    experiment: Experiment
    champion_model_uri: str
    environment: str


class TrainingEvidenceRequest(DomainModel):
    vertex_resource_name: str
    output_uri: str


class ResearchOperations(Protocol):
    async def analyze_failures(
        self, run_id: str, request: FailureAnalysisRequest
    ) -> list[FailureCluster]: ...

    async def form_hypothesis(self, run_id: str, request: HypothesisRequest) -> Hypothesis: ...

    async def curate_dataset(
        self, run_id: str, request: DatasetCurationRequest
    ) -> CuratedDataset: ...

    async def design_training(
        self, run_id: str, request: TrainingDesignRequest
    ) -> QLoRAConfig: ...


class BenchmarkExecutor(Protocol):
    async def execute_benchmark(
        self, run_id: str, request: BenchmarkRequest
    ) -> BenchmarkExecutionResult: ...


class EvaluationExecutor(Protocol):
    async def execute_evaluation(
        self, run_id: str, request: EvaluationRequest
    ) -> EvaluationExecutionResult: ...


class TrainingEvidenceResolver(Protocol):
    async def resolve(self, run_id: str, request: TrainingEvidenceRequest) -> TrainingResult: ...


class TrainingLauncher(Protocol):
    def submit(
        self,
        *,
        run_id: str,
        experiment_id: str,
        dataset_uri: str,
        output_uri: str,
        qlora_args: dict[str, int | float | str],
    ) -> VertexJobHandle: ...

    def state(self, resource_name: str) -> str: ...


async def _typed_request[ResponseT: DomainModel](
    transport: TypedA2ATransport,
    *,
    operation: A2AOperation,
    run_id: str,
    request: DomainModel,
    response_type: type[ResponseT],
    idempotency_suffix: str,
) -> ResponseT:
    payload = await transport.request(
        operation=operation,
        run_id=run_id,
        payload=request,
        idempotency_key=f"{run_id}:{idempotency_suffix}",
    )
    try:
        return response_type.model_validate(payload)
    except Exception as exc:
        raise CloudProviderError(f"malformed {operation.value} response") from exc


class A2AResearchOperations:
    def __init__(self, transport: TypedA2ATransport) -> None:
        self.transport = transport

    async def analyze_failures(
        self, run_id: str, request: FailureAnalysisRequest
    ) -> list[FailureCluster]:
        response = await _typed_request(
            self.transport,
            operation=A2AOperation.ANALYZE_FAILURES,
            run_id=run_id,
            request=request,
            response_type=FailureAnalysisResponse,
            idempotency_suffix="analyze_failures",
        )
        return response.clusters

    async def form_hypothesis(self, run_id: str, request: HypothesisRequest) -> Hypothesis:
        return await _typed_request(
            self.transport,
            operation=A2AOperation.FORM_HYPOTHESIS,
            run_id=run_id,
            request=request,
            response_type=Hypothesis,
            idempotency_suffix="form_hypothesis",
        )

    async def curate_dataset(
        self, run_id: str, request: DatasetCurationRequest
    ) -> CuratedDataset:
        response = await _typed_request(
            self.transport,
            operation=A2AOperation.CURATE_DATASET,
            run_id=run_id,
            request=request,
            response_type=DatasetCurationResponse,
            idempotency_suffix="curate_dataset",
        )
        if response.manifest.example_count != len(response.examples):
            raise CloudProviderError("dataset manifest count does not match returned examples")
        return CuratedDataset(manifest=response.manifest, examples=tuple(response.examples))

    async def design_training(
        self, run_id: str, request: TrainingDesignRequest
    ) -> QLoRAConfig:
        return await _typed_request(
            self.transport,
            operation=A2AOperation.DESIGN_TRAINING,
            run_id=run_id,
            request=request,
            response_type=QLoRAConfig,
            idempotency_suffix=f"design_training:{len(request.previous_configs)}",
        )


class A2AExecutionOperations:
    def __init__(self, transport: TypedA2ATransport) -> None:
        self.transport = transport

    async def execute_benchmark(
        self, run_id: str, request: BenchmarkRequest
    ) -> BenchmarkExecutionResult:
        return await _typed_request(
            self.transport,
            operation=A2AOperation.BENCHMARK,
            run_id=run_id,
            request=request,
            response_type=BenchmarkExecutionResult,
            idempotency_suffix="benchmark",
        )

    async def execute_evaluation(
        self, run_id: str, request: EvaluationRequest
    ) -> EvaluationExecutionResult:
        return await _typed_request(
            self.transport,
            operation=A2AOperation.EVALUATE,
            run_id=run_id,
            request=request,
            response_type=EvaluationExecutionResult,
            idempotency_suffix=f"evaluate:{request.experiment.experiment_id}",
        )

    async def resolve(self, run_id: str, request: TrainingEvidenceRequest) -> TrainingResult:
        return await _typed_request(
            self.transport,
            operation=A2AOperation.TRAINING_EVIDENCE,
            run_id=run_id,
            request=request,
            response_type=TrainingResult,
            idempotency_suffix=f"training_evidence:{request.vertex_resource_name}",
        )


class CloudDecisionProvider(DecisionProvider):
    """Compose verifiable cloud operations behind the orchestration contract."""

    def __init__(
        self,
        *,
        research: ResearchOperations,
        benchmark_executor: BenchmarkExecutor,
        evaluation_executor: EvaluationExecutor,
        training_evidence: TrainingEvidenceResolver,
        launcher: TrainingLauncher,
        artifact_bucket: str,
        poll_interval_seconds: float = 5.0,
        training_timeout_seconds: float = 7200.0,
    ) -> None:
        self.research = research
        self.benchmark_executor = benchmark_executor
        self.evaluation_executor = evaluation_executor
        self.training_evidence = training_evidence
        self.launcher = launcher
        self.artifact_bucket = artifact_bucket
        self.poll_interval_seconds = poll_interval_seconds
        self.training_timeout_seconds = training_timeout_seconds
        self._benchmark_artifacts: dict[str, ArtifactRef] = {}
        self._benchmark_trajectory_ids: dict[str, list[str]] = {}

    async def benchmark(self, state: RunState) -> list[Trajectory]:
        async with async_telemetry_span(
            "cloud.benchmark", attributes={"run_id": state.run_id, "operation": "benchmark"}
        ):
            result = await self.benchmark_executor.execute_benchmark(
                state.run_id,
                BenchmarkRequest(
                    target_model=state.target_model,
                    champion_model_uri=state.champion.model_uri,
                    environment=state.environment,
                ),
            )
        if not isinstance(result, BenchmarkExecutionResult):
            raise CloudProviderError("benchmark executor returned the wrong result type")
        self._benchmark_artifacts[state.run_id] = result.artifact
        self._benchmark_trajectory_ids[state.run_id] = [
            item.trajectory_id for item in result.trajectories
        ]
        return result.trajectories

    def benchmark_evidence(self, run_id: str) -> tuple[ArtifactRef, list[str]]:
        """Return benchmark provenance for immediate durable RunState persistence."""

        artifact = self._benchmark_artifacts.get(run_id)
        trajectory_ids = self._benchmark_trajectory_ids.get(run_id)
        if artifact is None or not trajectory_ids:
            raise CloudProviderError("benchmark evidence is unavailable")
        return artifact, list(trajectory_ids)

    async def analyze_failures(
        self, state: RunState, trajectories: list[Trajectory]
    ) -> list[FailureCluster]:
        artifact = self._benchmark_artifacts.get(state.run_id) or state.benchmark_artifact
        if artifact is None:
            raise CloudProviderError("benchmark artifact provenance is unavailable")
        trajectory_ids = [item.trajectory_id for item in trajectories]
        if not trajectory_ids:
            trajectory_ids = state.benchmark_trajectory_ids
        if not trajectory_ids:
            raise CloudProviderError("benchmark trajectory provenance is unavailable")
        return await self.research.analyze_failures(
            state.run_id,
            FailureAnalysisRequest(
                trajectories_artifact=artifact,
                trajectory_ids=trajectory_ids,
            ),
        )

    async def form_hypothesis(self, state: RunState) -> Hypothesis:
        return await self.research.form_hypothesis(
            state.run_id, HypothesisRequest(failure_clusters=state.failure_clusters)
        )

    async def curate_dataset(self, state: RunState) -> CuratedDataset:
        if state.current_hypothesis is None:
            raise CloudProviderError("dataset curation requires a hypothesis")
        artifact = self._benchmark_artifacts.get(state.run_id) or state.benchmark_artifact
        if artifact is None:
            raise CloudProviderError("benchmark artifact provenance is unavailable")
        return await self.research.curate_dataset(
            state.run_id,
            DatasetCurationRequest(
                hypothesis=state.current_hypothesis,
                trajectories_artifact=artifact,
            ),
        )

    async def design_training(self, state: RunState) -> QLoRAConfig:
        if state.current_hypothesis is None or state.dataset is None:
            raise CloudProviderError("training design requires hypothesis and dataset")
        return await self.research.design_training(
            state.run_id,
            TrainingDesignRequest(
                hypothesis=state.current_hypothesis,
                dataset=state.dataset,
                previous_configs=[item.config for item in state.experiments],
            ),
        )

    async def launch_training(self, state: RunState, experiment: Experiment) -> TrainingResult:
        async with async_telemetry_span(
            "vertex.training",
            attributes={
                "run_id": state.run_id,
                "experiment_id": experiment.experiment_id,
                "operation": "qlora",
            },
        ) as span:
            dataset_uri = experiment.dataset.artifact.uri
            if not dataset_uri.startswith("gs://"):
                raise CloudProviderError("cloud training requires a GCS dataset artifact")
            output_uri = (
                f"gs://{self.artifact_bucket}/runs/{state.run_id}/"
                f"experiments/{experiment.experiment_id}"
            )
            handle = await asyncio.to_thread(
                self.launcher.submit,
                run_id=state.run_id,
                experiment_id=experiment.experiment_id,
                dataset_uri=dataset_uri,
                output_uri=output_uri,
                qlora_args=experiment.config.model_dump(mode="json"),
            )
            span.set_attribute("job_id", handle.resource_name)
            started = time.monotonic()
            while True:
                state_name = await asyncio.to_thread(self.launcher.state, handle.resource_name)
                normalized = state_name.upper()
                if normalized.endswith("SUCCEEDED"):
                    break
                if normalized.endswith(("FAILED", "CANCELLED", "EXPIRED")):
                    raise CloudProviderError(f"Vertex training ended in {normalized}")
                if not normalized.endswith(
                    ("SUBMITTED", "PENDING", "QUEUED", "RUNNING", "PAUSED", "UPDATING")
                ):
                    raise CloudProviderError(f"Vertex returned unknown training state {normalized}")
                if time.monotonic() - started >= self.training_timeout_seconds:
                    raise CloudProviderError("Vertex training exceeded its bounded timeout")
                await asyncio.sleep(self.poll_interval_seconds)
            result = await self.training_evidence.resolve(
                state.run_id,
                TrainingEvidenceRequest(
                    vertex_resource_name=handle.resource_name,
                    output_uri=output_uri,
                ),
            )
        if result.status is not JobStatus.SUCCEEDED or result.job_id != handle.resource_name:
            raise CloudProviderError("training evidence does not match the completed Vertex job")
        if result.checkpoint is None or result.checkpoint.kind is not ArtifactKind.CHECKPOINT:
            raise CloudProviderError("completed training is missing a hashed checkpoint")
        if result.logs is None or result.logs.kind is not ArtifactKind.TRAINING_LOG:
            raise CloudProviderError("completed training is missing hashed logs")
        return result

    async def evaluate(self, state: RunState, experiment: Experiment) -> EvaluationReport:
        async with async_telemetry_span(
            "cloud.evaluate",
            attributes={
                "run_id": state.run_id,
                "experiment_id": experiment.experiment_id,
                "operation": "heldout_evaluation",
            },
        ):
            result = await self.evaluation_executor.execute_evaluation(
                state.run_id,
                EvaluationRequest(
                    experiment=experiment,
                    champion_model_uri=state.champion.model_uri,
                    environment=state.environment,
                ),
            )
        if not isinstance(result, EvaluationExecutionResult):
            raise CloudProviderError("evaluation executor returned the wrong result type")
        return result.report


def build_cloud_decision_provider(
    settings: Settings,
    *,
    token_provider: IdentityTokenProvider | None = None,
    research_transport: TypedA2ATransport | None = None,
    execution_transport: TypedA2ATransport | None = None,
    launcher: TrainingLauncher | None = None,
) -> CloudDecisionProvider:
    """Build the deployable provider or fail before serving cloud traffic."""

    if settings.environment != "cloud":
        raise CloudProviderError("cloud provider requires ENVIRONMENT=cloud")
    if not settings.research_a2a_url or not settings.execution_a2a_url:
        raise CloudProviderError("RESEARCH_A2A_URL and EXECUTION_A2A_URL are required")
    if not settings.artifact_bucket:
        raise CloudProviderError("ARTIFACT_BUCKET is required")
    identity = token_provider or GoogleIdentityTokenProvider()
    research_wire = research_transport or AuthenticatedHTTPA2ATransport(
        service_url=settings.research_a2a_url,
        token_provider=identity,
        timeout_seconds=settings.a2a_timeout_seconds,
    )
    execution_wire = execution_transport or AuthenticatedHTTPA2ATransport(
        service_url=settings.execution_a2a_url,
        token_provider=identity,
        timeout_seconds=settings.a2a_timeout_seconds,
    )
    execution = A2AExecutionOperations(execution_wire)
    return CloudDecisionProvider(
        research=A2AResearchOperations(research_wire),
        benchmark_executor=execution,
        evaluation_executor=execution,
        training_evidence=execution,
        launcher=launcher or VertexTrainingLauncher(settings),
        artifact_bucket=settings.artifact_bucket,
        training_timeout_seconds=float(settings.compute_budget_minutes * 60),
    )
