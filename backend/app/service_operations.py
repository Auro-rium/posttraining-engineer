"""Service-side dispatch for typed A2A research and execution operations.

Gemini is used only for structured research judgments. Objective trajectories,
repair verification, training evidence, and evaluation metrics must come from a
separately configured evidence executor and are validated again at this boundary.
"""

from __future__ import annotations

import importlib
import json
from hashlib import sha256
from typing import Any, Protocol, TypeVar
from urllib.parse import urlsplit

import httpx
from pydantic import Field

from app.agents import validate_qlora_config
from app.cloud_provider import (
    A2AOperation,
    A2AOperationRequest,
    A2AOperationResponse,
    BenchmarkExecutionResult,
    BenchmarkRequest,
    DatasetCurationRequest,
    DatasetCurationResponse,
    EvaluationExecutionResult,
    EvaluationRequest,
    FailureAnalysisRequest,
    FailureAnalysisResponse,
    HypothesisRequest,
    IdentityTokenProvider,
    TrainingDesignRequest,
    TrainingEvidenceRequest,
)
from app.models import (
    Citation,
    DatasetSplit,
    DomainModel,
    Hypothesis,
    QLoRAConfig,
    ToolCall,
    TrainingResult,
    Trajectory,
)
from app.rag import KnowledgeDocument, LeakageSafeRAG
from app.telemetry import async_telemetry_span

ResponseT = TypeVar("ResponseT", bound=DomainModel)


class ServiceOperationError(RuntimeError):
    """Sanitized, coded error raised by a service operation boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class StructuredResearchGenerator(Protocol):
    """Generate one Pydantic-validated research artifact through ADK."""

    async def generate(
        self,
        *,
        operation: A2AOperation,
        instruction: str,
        payload: DomainModel,
        response_type: type[ResponseT],
    ) -> ResponseT: ...


class ResearchEvidenceLoader(Protocol):
    """Resolve a hashed trajectory artifact for grounded research prompts."""

    async def load_trajectories(self, request: FailureAnalysisRequest) -> list[Trajectory]: ...


class GroundedRetriever(Protocol):
    """Retrieve only previously validated, non-evaluation research chunks."""

    async def search(self, query: str, *, limit: int = 5) -> list[Citation]: ...


class ObjectiveEvidenceExecutor(Protocol):
    """External deterministic worker for all metrics and artifact-producing work."""

    async def benchmark(
        self, run_id: str, request: BenchmarkRequest
    ) -> BenchmarkExecutionResult: ...

    async def verify_curation(
        self, run_id: str, request: VerifiedCurationRequest
    ) -> DatasetCurationResponse: ...

    async def evaluate(
        self, run_id: str, request: EvaluationRequest
    ) -> EvaluationExecutionResult: ...

    async def training_evidence(
        self, run_id: str, request: TrainingEvidenceRequest
    ) -> TrainingResult: ...


class FailureAnalysisEvidence(DomainModel):
    trajectories_artifact_sha256: str
    trajectories: list[Trajectory] = Field(min_length=1, max_length=200)


class RepairProposal(DomainModel):
    source_trajectory_id: str
    source_step_index: int = Field(ge=0)
    target_action: ToolCall


class RepairProposalRequest(DomainModel):
    hypothesis: Hypothesis
    trajectories: list[Trajectory] = Field(min_length=1, max_length=200)


class RepairProposalResponse(DomainModel):
    proposals: list[RepairProposal] = Field(min_length=1, max_length=100)


class VerifiedCurationRequest(DomainModel):
    request: DatasetCurationRequest
    proposals: list[RepairProposal] = Field(min_length=1, max_length=100)


class HypothesisGroundedInput(DomainModel):
    request: HypothesisRequest
    retrieved_citations: list[Citation] = Field(min_length=1, max_length=10)


class ADKStructuredResearchGenerator:
    """Run a single-turn Gemini ADK agent with a Pydantic output schema."""

    def __init__(self, *, model: str) -> None:
        self.model = model

    async def generate(
        self,
        *,
        operation: A2AOperation,
        instruction: str,
        payload: DomainModel,
        response_type: type[ResponseT],
    ) -> ResponseT:
        try:
            from google.adk.agents import LlmAgent
            from google.adk.runners import InMemoryRunner
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - cloud dependency only
            raise ServiceOperationError("ADK_UNAVAILABLE", "Google ADK is not installed") from exc

        agent = LlmAgent(
            name=f"structured_{operation.value}",
            model=self.model,
            description=f"Produces a typed {operation.value} research artifact.",
            instruction=(
                instruction
                + " Return exactly the requested structured output. Do not invent evidence, "
                "metrics, verifier results, artifact URIs, or hashes."
            ),
            output_schema=response_type,
            mode="single_turn",
        )
        runner = InMemoryRunner(agent=agent, app_name=f"apte_{operation.value}")
        prompt = json.dumps(payload.model_dump(mode="json"), sort_keys=True)
        final_text: str | None = None
        async for event in runner.run_async(
            user_id="a2a-service",
            session_id=f"{operation.value}-single-turn",
            new_message=types.Content(role="user", parts=[types.Part.from_text(text=prompt)]),
        ):
            if not event.is_final_response() or event.content is None:
                continue
            text_parts = [part.text for part in (event.content.parts or []) if part.text]
            if text_parts:
                final_text = "".join(text_parts)
        if not final_text:
            raise ServiceOperationError("ADK_EMPTY_OUTPUT", "ADK returned no final artifact")
        try:
            return response_type.model_validate_json(final_text)
        except Exception as exc:
            raise ServiceOperationError(
                "ADK_INVALID_OUTPUT", "ADK output failed schema validation"
            ) from exc


class GCSResearchEvidenceLoader:
    """Load and integrity-check train trajectories from the configured GCS bucket."""

    def __init__(self, *, bucket: str, project: str | None = None) -> None:
        from app.artifacts import GCSArtifactStore

        self.store = GCSArtifactStore(bucket=bucket, project=project)

    async def load_trajectories(self, request: FailureAnalysisRequest) -> list[Trajectory]:
        try:
            raw = json.loads((await self.store.get_bytes(request.trajectories_artifact)).decode())
            values = raw.get("trajectories") if isinstance(raw, dict) else raw
            if not isinstance(values, list):
                raise ValueError("trajectory artifact must contain a list")
            trajectories = [Trajectory.model_validate(item) for item in values]
        except Exception as exc:
            raise ServiceOperationError(
                "TRAJECTORY_ARTIFACT_INVALID",
                "trajectory artifact could not be integrity-checked and decoded",
            ) from exc
        requested = set(request.trajectory_ids)
        if requested == {"*"}:
            selected = trajectories
        else:
            selected = [item for item in trajectories if item.trajectory_id in requested]
        if requested != {"*"} and {item.trajectory_id for item in selected} != requested:
            raise ServiceOperationError(
                "TRAJECTORY_EVIDENCE_MISSING", "requested trajectories are absent from artifact"
            )
        if not selected or len(selected) > 200:
            raise ServiceOperationError(
                "TRAJECTORY_EVIDENCE_BOUNDS",
                "research requires between one and 200 train trajectories",
            )
        if any(item.split is not DatasetSplit.TRAIN for item in selected):
            raise ServiceOperationError(
                "EVALUATION_LEAKAGE", "research service accepts train trajectories only"
            )
        return selected


class GCSGroundedRetriever:
    """Load a GCS JSON corpus once and enforce ``LeakageSafeRAG`` validation."""

    def __init__(
        self,
        *,
        corpus_uri: str,
        project: str | None = None,
        expected_sha256: str | None = None,
    ) -> None:
        parsed = urlsplit(corpus_uri)
        if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.strip("/"):
            raise ValueError("RAG_CORPUS_URI must be a gs:// bucket/object URI")
        if expected_sha256 is not None and (
            len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ValueError("RAG_CORPUS_SHA256 must be a lowercase SHA-256 digest")
        self.bucket = parsed.netloc
        self.object_name = parsed.path.lstrip("/")
        self.project = project
        self.expected_sha256 = expected_sha256
        self._index: LeakageSafeRAG | None = None

    async def _load(self) -> LeakageSafeRAG:
        if self._index is not None:
            return self._index
        try:
            storage = importlib.import_module("google.cloud.storage")
        except ImportError as exc:  # pragma: no cover - cloud dependency only
            raise ServiceOperationError("GCS_UNAVAILABLE", "GCS SDK is unavailable") from exc

        def download() -> bytes:
            return bytes(
                storage.Client(project=self.project)
                .bucket(self.bucket)
                .blob(self.object_name)
                .download_as_bytes()
            )

        try:
            import asyncio

            data = await asyncio.to_thread(download)
            if self.expected_sha256 and sha256(data).hexdigest() != self.expected_sha256:
                raise ValueError("RAG corpus hash mismatch")
            raw = json.loads(data.decode())
            values = raw.get("documents") if isinstance(raw, dict) else raw
            if not isinstance(values, list):
                raise ValueError("RAG corpus must contain a document list")
            documents = [KnowledgeDocument.model_validate(item) for item in values]
            index = LeakageSafeRAG()
            if index.ingest(documents) < 1:
                raise ValueError("RAG corpus contains no chunks")
        except Exception as exc:
            raise ServiceOperationError(
                "RAG_CORPUS_INVALID", "RAG corpus could not be validated and indexed"
            ) from exc
        self._index = index
        return index

    async def search(self, query: str, *, limit: int = 5) -> list[Citation]:
        index = await self._load()
        return index.search(query, limit=limit)


class RemoteObjectiveEvidenceExecutor:
    """Authenticated client for a sandboxed model/environment evidence worker."""

    def __init__(
        self,
        *,
        service_url: str,
        token_provider: IdentityTokenProvider,
        timeout_seconds: float = 3300.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not service_url.startswith("https://"):
            raise ValueError("objective executor URL must use HTTPS")
        self.service_url = service_url.rstrip("/")
        self.token_provider = token_provider
        self.timeout_seconds = timeout_seconds
        self.client = client

    async def _post(
        self,
        path: str,
        *,
        run_id: str,
        request: DomainModel,
        response_type: type[ResponseT],
    ) -> ResponseT:
        token = await self.token_provider.token(self.service_url)
        client = self.client or httpx.AsyncClient(timeout=self.timeout_seconds)
        owns_client = self.client is None
        try:
            response = await client.post(
                f"{self.service_url}{path}",
                headers={"Authorization": f"Bearer {token}"},
                json={"run_id": run_id, "payload": request.model_dump(mode="json")},
            )
            if not 200 <= response.status_code < 300:
                raise ServiceOperationError(
                    "OBJECTIVE_EXECUTOR_FAILED",
                    f"objective executor returned HTTP {response.status_code}",
                )
            return response_type.model_validate(response.json())
        except ServiceOperationError:
            raise
        except Exception as exc:
            raise ServiceOperationError(
                "OBJECTIVE_EVIDENCE_INVALID",
                "objective executor returned invalid evidence",
            ) from exc
        finally:
            if owns_client:
                await client.aclose()

    async def benchmark(self, run_id: str, request: BenchmarkRequest) -> BenchmarkExecutionResult:
        return await self._post(
            "/v1/benchmark",
            run_id=run_id,
            request=request,
            response_type=BenchmarkExecutionResult,
        )

    async def verify_curation(
        self, run_id: str, request: VerifiedCurationRequest
    ) -> DatasetCurationResponse:
        return await self._post(
            "/v1/verify-curation",
            run_id=run_id,
            request=request,
            response_type=DatasetCurationResponse,
        )

    async def evaluate(self, run_id: str, request: EvaluationRequest) -> EvaluationExecutionResult:
        return await self._post(
            "/v1/evaluate",
            run_id=run_id,
            request=request,
            response_type=EvaluationExecutionResult,
        )

    async def training_evidence(
        self, run_id: str, request: TrainingEvidenceRequest
    ) -> TrainingResult:
        return await self._post(
            "/v1/training-evidence",
            run_id=run_id,
            request=request,
            response_type=TrainingResult,
        )


class A2AOperationService:
    """Validate and dispatch one operation for one Cloud Run service role."""

    _RESEARCH_OPERATIONS = frozenset(
        {
            A2AOperation.ANALYZE_FAILURES,
            A2AOperation.FORM_HYPOTHESIS,
            A2AOperation.CURATE_DATASET,
            A2AOperation.DESIGN_TRAINING,
        }
    )
    _EXECUTION_OPERATIONS = frozenset(
        {
            A2AOperation.BENCHMARK,
            A2AOperation.EVALUATE,
            A2AOperation.TRAINING_EVIDENCE,
        }
    )

    def __init__(
        self,
        *,
        role: str,
        research_generator: StructuredResearchGenerator | None = None,
        evidence_loader: ResearchEvidenceLoader | None = None,
        retriever: GroundedRetriever | None = None,
        objective_executor: ObjectiveEvidenceExecutor | None = None,
    ) -> None:
        if role not in {"research", "execution"}:
            raise ValueError("A2A operation service role must be research or execution")
        if role == "research" and research_generator is None:
            raise ValueError("research service requires a structured ADK generator")
        if role == "research" and retriever is None:
            raise ValueError("research service requires a grounded retriever")
        if objective_executor is None:
            raise ValueError("service requires an objective evidence executor")
        self.role = role
        self.research_generator = research_generator
        self.evidence_loader = evidence_loader
        self.retriever = retriever
        self.objective_executor = objective_executor

    async def handle(self, request: A2AOperationRequest) -> A2AOperationResponse:
        allowed = (
            self._RESEARCH_OPERATIONS if self.role == "research" else self._EXECUTION_OPERATIONS
        )
        if request.operation not in allowed:
            raise ServiceOperationError(
                "OPERATION_NOT_ALLOWED", f"operation is not served by {self.role} service"
            )
        async with async_telemetry_span(
            "a2a.operation",
            attributes={"run_id": request.run_id, "operation": request.operation.value},
        ):
            if self.role == "research":
                payload = await self._handle_research(request)
            else:
                payload = await self._handle_execution(request)
        return A2AOperationResponse(operation=request.operation, payload=payload)

    async def _handle_research(self, envelope: A2AOperationRequest) -> dict[str, Any]:
        generator = self.research_generator
        if generator is None:  # defensive even though construction rejects it
            raise ServiceOperationError("ADK_UNAVAILABLE", "research generator is unavailable")

        if envelope.operation is A2AOperation.ANALYZE_FAILURES:
            failure_request = FailureAnalysisRequest.model_validate(envelope.payload)
            trajectories = await self._load_trajectories(failure_request)
            response = await generator.generate(
                operation=envelope.operation,
                instruction=(
                    "Cluster only recurring failures visible in the supplied train trajectories. "
                    "Every trajectory_id must come from the input."
                ),
                payload=FailureAnalysisEvidence(
                    trajectories_artifact_sha256=(failure_request.trajectories_artifact.sha256),
                    trajectories=trajectories,
                ),
                response_type=FailureAnalysisResponse,
            )
            allowed_ids = {item.trajectory_id for item in trajectories}
            if any(not set(cluster.trajectory_ids) <= allowed_ids for cluster in response.clusters):
                raise ServiceOperationError(
                    "UNGROUNDED_RESEARCH", "failure report referenced unknown trajectories"
                )
            return response.model_dump(mode="json")

        if envelope.operation is A2AOperation.FORM_HYPOTHESIS:
            hypothesis_request = HypothesisRequest.model_validate(envelope.payload)
            if self.retriever is None:
                raise ServiceOperationError(
                    "RAG_NOT_CONFIGURED", "grounded retrieval is unavailable"
                )
            query = " ".join(
                f"{cluster.label} {cluster.description}"
                for cluster in hypothesis_request.failure_clusters
            )
            async with async_telemetry_span(
                "rag.retrieve",
                attributes={"run_id": envelope.run_id, "operation": "form_hypothesis"},
            ) as span:
                citations = await self.retriever.search(query, limit=5)
                span.set_attribute("retrieval_count", len(citations))
            if not citations:
                raise ServiceOperationError(
                    "RAG_EVIDENCE_MISSING", "no grounded research evidence was retrieved"
                )
            hypothesis = await generator.generate(
                operation=envelope.operation,
                instruction=(
                    "Form one falsifiable hypothesis tied to exactly one supplied failure cluster. "
                    "Cite one or more supplied retrieval chunks and no other sources."
                ),
                payload=HypothesisGroundedInput(
                    request=hypothesis_request, retrieved_citations=citations
                ),
                response_type=Hypothesis,
            )
            cluster_ids = {item.cluster_id for item in hypothesis_request.failure_clusters}
            allowed_citations = {(item.document_id, item.chunk_id): item for item in citations}
            selected_keys = [(item.document_id, item.chunk_id) for item in hypothesis.citations]
            if (
                hypothesis.failure_cluster_id not in cluster_ids
                or not selected_keys
                or any(key not in allowed_citations for key in selected_keys)
            ):
                raise ServiceOperationError(
                    "UNGROUNDED_RESEARCH", "hypothesis contains unsupported evidence"
                )
            hypothesis = hypothesis.model_copy(
                update={"citations": [allowed_citations[key] for key in selected_keys]}
            )
            return hypothesis.model_dump(mode="json")

        if envelope.operation is A2AOperation.CURATE_DATASET:
            curation_request = DatasetCurationRequest.model_validate(envelope.payload)
            trajectories = await self._load_trajectories(
                FailureAnalysisRequest(
                    trajectories_artifact=curation_request.trajectories_artifact,
                    trajectory_ids=[curation_request.hypothesis.failure_cluster_id],
                ),
                allow_all=True,
            )
            proposals = await generator.generate(
                operation=envelope.operation,
                instruction=(
                    "Propose corrected FunctionGemma tool actions only. Do not claim verification. "
                    "Reference only supplied trajectory IDs and valid step indexes."
                ),
                payload=RepairProposalRequest(
                    hypothesis=curation_request.hypothesis, trajectories=trajectories
                ),
                response_type=RepairProposalResponse,
            )
            _validate_proposals(proposals.proposals, trajectories)
            verified = await self.objective_executor.verify_curation(
                envelope.run_id,
                VerifiedCurationRequest(request=curation_request, proposals=proposals.proposals),
            )
            source_ids = {item.trajectory_id for item in trajectories}
            if verified.manifest.example_count != len(verified.examples) or any(
                item.source_trajectory_id not in source_ids for item in verified.examples
            ):
                raise ServiceOperationError(
                    "CURATION_EVIDENCE_INVALID", "verified dataset does not match source evidence"
                )
            return verified.model_dump(mode="json")

        design_request = TrainingDesignRequest.model_validate(envelope.payload)
        config = await generator.generate(
            operation=envelope.operation,
            instruction=(
                "Select exactly one configuration from the allowed values and do not repeat a "
                "previous configuration."
            ),
            payload=design_request,
            response_type=QLoRAConfig,
        )
        validate_qlora_config(config)
        if any(config == previous for previous in design_request.previous_configs):
            raise ServiceOperationError(
                "DUPLICATE_EXPERIMENT", "ADK selected a previously used configuration"
            )
        return config.model_dump(mode="json")

    async def _handle_execution(self, envelope: A2AOperationRequest) -> dict[str, Any]:
        if envelope.operation is A2AOperation.BENCHMARK:
            benchmark_result = await self.objective_executor.benchmark(
                envelope.run_id, BenchmarkRequest.model_validate(envelope.payload)
            )
            return benchmark_result.model_dump(mode="json")
        if envelope.operation is A2AOperation.EVALUATE:
            evaluation_result = await self.objective_executor.evaluate(
                envelope.run_id, EvaluationRequest.model_validate(envelope.payload)
            )
            return evaluation_result.model_dump(mode="json")
        training_result = await self.objective_executor.training_evidence(
            envelope.run_id, TrainingEvidenceRequest.model_validate(envelope.payload)
        )
        return training_result.model_dump(mode="json")

    async def _load_trajectories(
        self, request: FailureAnalysisRequest, *, allow_all: bool = False
    ) -> list[Trajectory]:
        if self.evidence_loader is None:
            raise ServiceOperationError(
                "EVIDENCE_LOADER_NOT_CONFIGURED", "research evidence loader is unavailable"
            )
        if allow_all:
            # The curation wire contract supplies the artifact but not a list of
            # IDs. An empty sentinel asks the loader for the bounded train set.
            request = request.model_copy(update={"trajectory_ids": ["*"]})
        return await self.evidence_loader.load_trajectories(request)


def _validate_proposals(proposals: list[RepairProposal], trajectories: list[Trajectory]) -> None:
    by_id = {item.trajectory_id: item for item in trajectories}
    for proposal in proposals:
        trajectory = by_id.get(proposal.source_trajectory_id)
        if trajectory is None or proposal.source_step_index >= len(trajectory.steps):
            raise ServiceOperationError(
                "UNGROUNDED_REPAIR", "repair proposal references unknown trajectory evidence"
            )
