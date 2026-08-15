from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.adk_service import (
    _build_operation_service,
    _data_part,
    build_operation_agent_executor,
)
from app.cloud_provider import (
    A2AOperation,
    A2AOperationRequest,
    BenchmarkExecutionResult,
    BenchmarkRequest,
    DatasetCurationResponse,
    EvaluationExecutionResult,
    EvaluationRequest,
    FailureAnalysisRequest,
    HypothesisRequest,
    TrainingDesignRequest,
    TrainingEvidenceRequest,
)
from app.models import (
    ArtifactKind,
    ArtifactRef,
    Citation,
    DatasetManifest,
    DatasetSplit,
    EvidenceLabel,
    FailureCluster,
    Hypothesis,
    JobStatus,
    QLoRAConfig,
    SFTExample,
    ToolCall,
    TrainingResult,
    Trajectory,
    TrajectoryStep,
)
from app.service_operations import (
    A2AOperationService,
    RepairProposal,
    RepairProposalResponse,
    ServiceOperationError,
)
from app.settings import Settings


def artifact(kind: ArtifactKind, name: str) -> ArtifactRef:
    return ArtifactRef(
        kind=kind,
        uri=f"gs://test/{name}",
        sha256="a" * 64,
        size_bytes=10,
    )


def trajectory() -> Trajectory:
    return Trajectory(
        trajectory_id="trajectory-1",
        task_id="train-1",
        split=DatasetSplit.TRAIN,
        model_version="gemma-base",
        steps=[
            TrajectoryStep(
                index=0,
                observation="Find a blue shirt",
                action=ToolCall(name="search", arguments={"keywords": "shirt"}),
                reward=0.0,
            )
        ],
        reward=0.0,
        success=False,
    )


class FakeLoader:
    async def load_trajectories(self, request: FailureAnalysisRequest) -> list[Trajectory]:
        return [trajectory()]


class FakeGenerator:
    def __init__(self, *, unknown_trajectory: bool = False) -> None:
        self.unknown_trajectory = unknown_trajectory

    async def generate(
        self,
        *,
        operation: A2AOperation,
        instruction: str,
        payload: Any,
        response_type: Any,
    ) -> Any:
        if operation is A2AOperation.ANALYZE_FAILURES:
            return response_type(
                clusters=[
                    FailureCluster(
                        label="constraint loss",
                        description="search omitted a constraint",
                        trajectory_ids=["unknown" if self.unknown_trajectory else "trajectory-1"],
                        frequency=1,
                    )
                ]
            )
        if operation is A2AOperation.FORM_HYPOTHESIS:
            cluster = payload.request.failure_clusters[0]
            return response_type(
                failure_cluster_id=cluster.cluster_id,
                statement="constraint-preserving repairs improve success",
                expected_improvement="five percentage points",
                data_strategy="verified train-side repairs",
                citations=[payload.retrieved_citations[0]],
            )
        if operation is A2AOperation.CURATE_DATASET:
            return RepairProposalResponse(
                proposals=[
                    RepairProposal(
                        source_trajectory_id="trajectory-1",
                        source_step_index=0,
                        target_action=ToolCall(name="search", arguments={"keywords": "blue shirt"}),
                    )
                ]
            )
        return response_type(rank=8, learning_rate=1e-4, epochs=2, dropout=0.0)


class FakeObjectiveExecutor:
    def __init__(self) -> None:
        self.curation_calls = 0

    async def benchmark(self, run_id: str, request: BenchmarkRequest) -> BenchmarkExecutionResult:
        return BenchmarkExecutionResult(
            trajectories=[trajectory()],
            artifact=artifact(ArtifactKind.TRAJECTORIES, "trajectories.json"),
            evidence_label=EvidenceLabel.LIVE,
            provenance_complete=True,
        )

    async def verify_curation(self, run_id: str, request: Any) -> DatasetCurationResponse:
        self.curation_calls += 1
        return DatasetCurationResponse(
            manifest=DatasetManifest(
                artifact=artifact(ArtifactKind.DATASET, "dataset.jsonl"), example_count=1
            ),
            examples=[
                SFTExample(
                    source_trajectory_id="trajectory-1",
                    source_step_index=0,
                    observation="Find a blue shirt",
                    target_action=request.proposals[0].target_action,
                    verified=True,
                    verifier_reward_before=0.0,
                    verifier_reward_after=1.0,
                )
            ],
        )

    async def evaluate(self, run_id: str, request: EvaluationRequest) -> EvaluationExecutionResult:
        raise AssertionError("not used")

    async def training_evidence(
        self, run_id: str, request: TrainingEvidenceRequest
    ) -> TrainingResult:
        return TrainingResult(
            job_id=request.vertex_resource_name,
            status=JobStatus.SUCCEEDED,
            checkpoint=artifact(ArtifactKind.CHECKPOINT, "checkpoint"),
            logs=artifact(ArtifactKind.TRAINING_LOG, "logs.json"),
        )


class FakeRetriever:
    async def search(self, query: str, *, limit: int = 5) -> list[Citation]:
        return [
            Citation(
                document_id="qlora-doc",
                chunk_id="qlora-doc:0",
                title="QLoRA guidance",
                source_uri="gs://test/rag.json",
                excerpt="Use verified examples and bounded adapter configurations.",
                score=1.0,
            )
        ]


class EmptyRetriever:
    async def search(self, query: str, *, limit: int = 5) -> list[Citation]:
        return []


def research_service(*, generator: Any | None = None) -> A2AOperationService:
    return A2AOperationService(
        role="research",
        research_generator=generator or FakeGenerator(),
        evidence_loader=FakeLoader(),
        retriever=FakeRetriever(),
        objective_executor=FakeObjectiveExecutor(),
    )


async def test_research_analysis_is_grounded_and_typed() -> None:
    request = FailureAnalysisRequest(
        trajectories_artifact=artifact(ArtifactKind.TRAJECTORIES, "trajectories.json"),
        trajectory_ids=["trajectory-1"],
    )
    response = await research_service().handle(
        A2AOperationRequest(
            operation=A2AOperation.ANALYZE_FAILURES,
            run_id="run-1",
            payload=request.model_dump(mode="json"),
        )
    )

    assert response.operation is A2AOperation.ANALYZE_FAILURES
    assert response.payload["clusters"][0]["trajectory_ids"] == ["trajectory-1"]


async def test_research_rejects_gemini_reference_to_unknown_trajectory() -> None:
    request = FailureAnalysisRequest(
        trajectories_artifact=artifact(ArtifactKind.TRAJECTORIES, "trajectories.json"),
        trajectory_ids=["trajectory-1"],
    )
    with pytest.raises(ServiceOperationError, match="unknown trajectories"):
        await research_service(generator=FakeGenerator(unknown_trajectory=True)).handle(
            A2AOperationRequest(
                operation=A2AOperation.ANALYZE_FAILURES,
                run_id="run-1",
                payload=request.model_dump(mode="json"),
            )
        )


async def test_hypothesis_requires_nonempty_grounded_rag_citations() -> None:
    cluster = FailureCluster(
        label="constraint loss",
        description="search omitted a constraint",
        trajectory_ids=["trajectory-1"],
        frequency=1,
    )
    request = HypothesisRequest(failure_clusters=[cluster])
    response = await research_service().handle(
        A2AOperationRequest(
            operation=A2AOperation.FORM_HYPOTHESIS,
            run_id="run-1",
            payload=request.model_dump(mode="json"),
        )
    )

    assert response.payload["citations"][0]["chunk_id"] == "qlora-doc:0"

    service = A2AOperationService(
        role="research",
        research_generator=FakeGenerator(),
        evidence_loader=FakeLoader(),
        retriever=EmptyRetriever(),
        objective_executor=FakeObjectiveExecutor(),
    )
    with pytest.raises(ServiceOperationError, match="no grounded research evidence"):
        await service.handle(
            A2AOperationRequest(
                operation=A2AOperation.FORM_HYPOTHESIS,
                run_id="run-1",
                payload=request.model_dump(mode="json"),
            )
        )


async def test_curated_rows_come_from_objective_verifier() -> None:
    objective = FakeObjectiveExecutor()
    service = A2AOperationService(
        role="research",
        research_generator=FakeGenerator(),
        evidence_loader=FakeLoader(),
        retriever=FakeRetriever(),
        objective_executor=objective,
    )
    hypothesis = Hypothesis(
        failure_cluster_id="cluster-1",
        statement="repair searches",
        expected_improvement="five points",
        data_strategy="verified repairs",
    )
    from app.cloud_provider import DatasetCurationRequest

    request = DatasetCurationRequest(
        hypothesis=hypothesis,
        trajectories_artifact=artifact(ArtifactKind.TRAJECTORIES, "trajectories.json"),
    )
    response = await service.handle(
        A2AOperationRequest(
            operation=A2AOperation.CURATE_DATASET,
            run_id="run-1",
            payload=request.model_dump(mode="json"),
        )
    )

    assert objective.curation_calls == 1
    assert response.payload["examples"][0]["verified"] is True


async def test_duplicate_training_design_is_rejected() -> None:
    hypothesis = Hypothesis(
        failure_cluster_id="cluster-1",
        statement="repair searches",
        expected_improvement="five points",
        data_strategy="verified repairs",
    )
    config = QLoRAConfig(rank=8, learning_rate=1e-4, epochs=2, dropout=0.0)
    request = TrainingDesignRequest(
        hypothesis=hypothesis,
        dataset=DatasetManifest(
            artifact=artifact(ArtifactKind.DATASET, "dataset.jsonl"), example_count=1
        ),
        previous_configs=[config],
    )

    with pytest.raises(ServiceOperationError, match="previously used"):
        await research_service().handle(
            A2AOperationRequest(
                operation=A2AOperation.DESIGN_TRAINING,
                run_id="run-1",
                payload=request.model_dump(mode="json"),
            )
        )


async def test_execution_service_returns_objective_benchmark_only() -> None:
    service = A2AOperationService(role="execution", objective_executor=FakeObjectiveExecutor())
    request = BenchmarkRequest(
        target_model="google/functiongemma-270m-it",
        champion_model_uri="google/functiongemma-270m-it",
        environment="AgentGym/WebShop",
    )
    response = await service.handle(
        A2AOperationRequest(
            operation=A2AOperation.BENCHMARK,
            run_id="run-1",
            payload=request.model_dump(mode="json"),
        )
    )

    assert response.payload["provenance_complete"] is True
    assert response.payload["evidence_label"] == "LIVE"


def test_service_construction_fails_without_objective_executor() -> None:
    with pytest.raises(ValueError, match="OBJECTIVE_EXECUTION_URL"):
        _build_operation_service(Settings(service_role="execution", _env_file=None))


class RecordingEventQueue:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def enqueue_event(self, event: Any) -> None:
        self.events.append(event)


async def test_a2a_executor_emits_exactly_one_typed_data_artifact() -> None:
    from a2a.types import Message, Role, TaskArtifactUpdateEvent
    from google.protobuf.json_format import MessageToDict  # type: ignore[import-untyped]

    service = A2AOperationService(role="execution", objective_executor=FakeObjectiveExecutor())
    request = A2AOperationRequest(
        operation=A2AOperation.BENCHMARK,
        run_id="run-1",
        payload=BenchmarkRequest(
            target_model="google/functiongemma-270m-it",
            champion_model_uri="google/functiongemma-270m-it",
            environment="AgentGym/WebShop",
        ).model_dump(mode="json"),
    )
    context = SimpleNamespace(
        task_id="task-1",
        context_id="context-1",
        message=Message(
            message_id="message-1",
            role=Role.ROLE_USER,
            parts=[_data_part(request.model_dump(mode="json"))],
        ),
    )
    queue = RecordingEventQueue()

    await build_operation_agent_executor(service).execute(context, queue)

    artifacts = [event for event in queue.events if isinstance(event, TaskArtifactUpdateEvent)]
    assert len(artifacts) == 1
    assert len(artifacts[0].artifact.parts) == 1
    data = MessageToDict(artifacts[0].artifact.parts[0].data)
    assert data["operation"] == A2AOperation.BENCHMARK.value
    assert data["payload"]["provenance_complete"] is True
