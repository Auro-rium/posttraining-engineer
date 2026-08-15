from __future__ import annotations

import asyncio
import importlib
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from app.agents import CuratedDataset
from app.cloud import VertexJobHandle
from app.cloud_provider import (
    A2AOperation,
    A2AOperationResponse,
    AuthenticatedHTTPA2ATransport,
    BenchmarkExecutionResult,
    BenchmarkRequest,
    CloudDecisionProvider,
    CloudProviderError,
    DatasetCurationRequest,
    EvaluationExecutionResult,
    EvaluationRequest,
    FailureAnalysisRequest,
    HypothesisRequest,
    TrainingDesignRequest,
    TrainingEvidenceRequest,
    build_cloud_decision_provider,
)
from app.models import (
    ArtifactKind,
    ArtifactRef,
    DatasetManifest,
    DatasetSplit,
    EvaluationReport,
    EvidenceLabel,
    Experiment,
    FailureCluster,
    Hypothesis,
    JobStatus,
    QLoRAConfig,
    RunState,
    SFTExample,
    ToolCall,
    TrainingResult,
    Trajectory,
    TrajectoryStep,
)
from app.settings import Settings


def artifact(kind: ArtifactKind, name: str) -> ArtifactRef:
    return ArtifactRef(
        kind=kind,
        uri=f"gs://test-artifacts/{name}",
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
                observation="find a waterproof shoe",
                action=ToolCall(name="search", arguments={"keywords": "shoe"}),
            )
        ],
        reward=0.0,
        success=False,
    )


def hypothesis() -> Hypothesis:
    return Hypothesis(
        failure_cluster_id="cluster-1",
        statement="Preserve constraints.",
        expected_improvement="Improve held-out success.",
        data_strategy="Use verified train repairs.",
    )


def dataset() -> DatasetManifest:
    return DatasetManifest(
        artifact=artifact(ArtifactKind.DATASET, "dataset.jsonl"),
        example_count=1,
    )


class FakeResearch:
    async def analyze_failures(
        self, run_id: str, request: FailureAnalysisRequest
    ) -> list[FailureCluster]:
        return [
            FailureCluster(
                cluster_id="cluster-1",
                label="constraint_drop",
                description="constraint omitted",
                trajectory_ids=request.trajectory_ids,
                frequency=1,
            )
        ]

    async def form_hypothesis(self, run_id: str, request: HypothesisRequest) -> Hypothesis:
        return hypothesis()

    async def curate_dataset(
        self, run_id: str, request: DatasetCurationRequest
    ) -> CuratedDataset:
        example = SFTExample(
            source_trajectory_id="trajectory-1",
            source_step_index=0,
            observation="find a waterproof shoe",
            target_action=ToolCall(
                name="search", arguments={"keywords": "waterproof shoe"}
            ),
            verified=True,
            verifier_reward_before=0.0,
            verifier_reward_after=0.5,
        )
        return CuratedDataset(manifest=dataset(), examples=(example,))

    async def design_training(
        self, run_id: str, request: TrainingDesignRequest
    ) -> QLoRAConfig:
        return QLoRAConfig(rank=8, learning_rate=1e-4, epochs=2, dropout=0.0)


class FakeExecution:
    async def execute_benchmark(
        self, run_id: str, request: BenchmarkRequest
    ) -> BenchmarkExecutionResult:
        return BenchmarkExecutionResult(
            trajectories=[trajectory()],
            artifact=artifact(ArtifactKind.TRAJECTORIES, "trajectories.json"),
            evidence_label=EvidenceLabel.LIVE,
            provenance_complete=True,
        )

    async def execute_evaluation(
        self, run_id: str, request: EvaluationRequest
    ) -> EvaluationExecutionResult:
        return EvaluationExecutionResult(
            report=EvaluationReport(
                artifact=artifact(ArtifactKind.EVALUATION_REPORT, "evaluation.json"),
                task_count=20,
                champion_success=0.35,
                candidate_success=0.41,
                champion_regression_success=0.9,
                candidate_regression_success=0.89,
                champion_action_validity=0.99,
                candidate_action_validity=0.99,
                paired_improvement_positive=True,
                provenance_complete=True,
                evidence_label=EvidenceLabel.LIVE,
            )
        )

    async def resolve(self, run_id: str, request: TrainingEvidenceRequest) -> TrainingResult:
        return TrainingResult(
            job_id="projects/p/locations/l/customJobs/1",
            status=JobStatus.SUCCEEDED,
            checkpoint=artifact(ArtifactKind.CHECKPOINT, "checkpoint/manifest.json"),
            logs=artifact(ArtifactKind.TRAINING_LOG, "logs.json"),
            duration_seconds=10.0,
        )


class FakeLauncher:
    def __init__(self) -> None:
        self.states = iter(["JOB_STATE_RUNNING", "JOB_STATE_SUCCEEDED"])

    def submit(self, **_: Any) -> VertexJobHandle:
        return VertexJobHandle(
            resource_name="projects/p/locations/l/customJobs/1",
            display_name="test-job",
            state="SUBMITTED",
        )

    def state(self, resource_name: str) -> str:
        return next(self.states)


def test_cloud_provider_composes_verified_operations() -> None:
    async def scenario() -> None:
        execution = FakeExecution()
        provider = CloudDecisionProvider(
            research=FakeResearch(),
            benchmark_executor=execution,
            evaluation_executor=execution,
            training_evidence=execution,
            launcher=FakeLauncher(),
            artifact_bucket="test-artifacts",
            poll_interval_seconds=0.0,
            training_timeout_seconds=1.0,
        )
        state = RunState(run_id="RUN-1")
        trajectories = await provider.benchmark(state)
        clusters = await provider.analyze_failures(state, trajectories)
        benchmark_artifact, trajectory_ids = provider.benchmark_evidence(state.run_id)
        resumed_provider = CloudDecisionProvider(
            research=FakeResearch(),
            benchmark_executor=execution,
            evaluation_executor=execution,
            training_evidence=execution,
            launcher=FakeLauncher(),
            artifact_bucket="test-artifacts",
            poll_interval_seconds=0.0,
            training_timeout_seconds=1.0,
        )
        resumed_state = state.model_copy(
            update={
                "benchmark_artifact": benchmark_artifact,
                "benchmark_trajectory_ids": trajectory_ids,
            }
        )
        assert await resumed_provider.analyze_failures(resumed_state, []) == clusters
        state = state.model_copy(update={"failure_clusters": clusters})
        formed = await provider.form_hypothesis(state)
        state = state.model_copy(update={"current_hypothesis": formed})
        curated = await provider.curate_dataset(state)
        state = state.model_copy(update={"dataset": curated.manifest})
        config = await provider.design_training(state)
        experiment = Experiment(
            run_id=state.run_id,
            hypothesis=formed,
            config=config,
            dataset=curated.manifest,
        )
        trained = await provider.launch_training(state, experiment)
        evaluated = await provider.evaluate(state, experiment)
        assert trained.status is JobStatus.SUCCEEDED
        assert evaluated.evidence_label is EvidenceLabel.LIVE

    asyncio.run(scenario())


def test_cloud_evidence_models_reject_explanations() -> None:
    with pytest.raises(ValidationError, match="not independently verifiable"):
        BenchmarkExecutionResult(
            trajectories=[trajectory()],
            artifact=artifact(ArtifactKind.TRAJECTORIES, "trajectories.json"),
            evidence_label=EvidenceLabel.EXPLANATION,
            provenance_complete=True,
        )
    report = EvaluationReport(
        task_count=1,
        champion_success=0.3,
        candidate_success=0.4,
        champion_regression_success=1.0,
        candidate_regression_success=1.0,
        champion_action_validity=1.0,
        candidate_action_validity=1.0,
        paired_improvement_positive=True,
        provenance_complete=False,
        evidence_label=EvidenceLabel.EXPLANATION,
    )
    with pytest.raises(ValidationError, match="not independently verifiable"):
        EvaluationExecutionResult(report=report)


class FakeTokenProvider:
    async def token(self, audience: str) -> str:
        assert audience == "https://research.example"
        return "signed-id-token"


@pytest.mark.parametrize("response_shape", ["task", "message"])
def test_http_transport_sends_auth_idempotency_and_typed_a2a(
    response_shape: str,
) -> None:
    a2a_pb2 = importlib.import_module("a2a.types.a2a_pb2")
    json_format = importlib.import_module("google.protobuf.json_format")

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers["Authorization"]
        captured["idempotency"] = request.headers["Idempotency-Key"]
        request_body = __import__("json").loads(request.content)
        captured["method"] = request_body["method"]
        captured["role"] = request_body["params"]["message"]["role"]
        captured["part"] = request_body["params"]["message"]["parts"][0]
        response = A2AOperationResponse(
            operation=A2AOperation.FORM_HYPOTHESIS,
            payload=hypothesis().model_dump(mode="json"),
        )
        data_part = {"data": response.model_dump(mode="json")}
        if response_shape == "task":
            response_proto = json_format.ParseDict(
                {
                    "task": {
                        "id": "task-1",
                        "contextId": "RUN-1",
                        "artifacts": [
                            {"artifactId": "artifact-1", "parts": [data_part]}
                        ],
                    }
                },
                a2a_pb2.SendMessageResponse(),
            )
        else:
            response_proto = json_format.ParseDict(
                {
                    "message": {
                        "messageId": "response-1",
                        "role": "ROLE_AGENT",
                        "parts": [data_part],
                    }
                },
                a2a_pb2.SendMessageResponse(),
            )
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": "response-1",
                "result": json_format.MessageToDict(response_proto),
            },
        )

    async def scenario() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        transport = AuthenticatedHTTPA2ATransport(
            service_url="https://research.example",
            token_provider=FakeTokenProvider(),
            client=client,
        )
        response = await transport.request(
            operation=A2AOperation.FORM_HYPOTHESIS,
            run_id="RUN-1",
            payload=HypothesisRequest(
                failure_clusters=[
                    FailureCluster(
                        label="failure",
                        description="failure",
                        trajectory_ids=["trajectory-1"],
                        frequency=1,
                    )
                ]
            ),
            idempotency_key="RUN-1:research",
        )
        await client.aclose()
        assert Hypothesis.model_validate(response).statement == "Preserve constraints."

    asyncio.run(scenario())
    assert captured["authorization"] == "Bearer signed-id-token"
    assert captured["idempotency"] == "RUN-1:research"
    assert captured["method"] == "SendMessage"
    assert captured["role"] == "ROLE_USER"
    assert "kind" not in captured["part"]
    assert captured["part"]["data"]["operation"] == "form_hypothesis"
    assert captured["part"]["data"]["run_id"] == "RUN-1"


def test_factory_fails_closed_without_remote_a2a_urls() -> None:
    settings = Settings(
        environment="cloud",
        google_cloud_project="project",
        artifact_bucket="bucket",
        vertex_staging_bucket="gs://staging",
        training_container_uri="us-docker.pkg.dev/p/r/trainer:latest",
    )
    with pytest.raises(CloudProviderError, match="A2A_URL"):
        build_cloud_decision_provider(settings)


def test_cloud_runtime_timeouts_fit_cloud_run_request_ceiling() -> None:
    settings = Settings()
    assert settings.compute_budget_minutes == 55
    assert settings.a2a_timeout_seconds == 3300.0
    with pytest.raises(ValidationError):
        Settings(compute_budget_minutes=56)
    with pytest.raises(ValidationError):
        Settings(a2a_timeout_seconds=3501)
