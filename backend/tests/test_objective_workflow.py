from __future__ import annotations

from collections import deque

import pytest

from app.posttraining.models import ArtifactKind, ArtifactReference, EvidenceKind, EvidenceLabel
from app.posttraining.objective_workflow import (
    JobExecutionFailed,
    JobWaitPolicy,
    JobWaitTimeout,
    ObjectiveBenchmarkRequest,
    ObjectiveBenchmarkResult,
    canonical_manifest_sha256,
    execute_objective_benchmark,
    train_then_evaluate,
    wait_for_evaluation_job,
    wait_for_training_job,
)
from app.posttraining.run_history import BenchmarkMetrics
from app.providers.sagemaker import (
    EvaluationJobRequest,
    JobResult,
    JobStatus,
    TrainingJobRequest,
)


def _artifact(artifact_id: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=artifact_id,
        kind=ArtifactKind.REPORT,
        uri=f"s3://bucket/{artifact_id}",
        sha256="a" * 64,
    )


def _result(request: ObjectiveBenchmarkRequest) -> ObjectiveBenchmarkResult:
    return ObjectiveBenchmarkResult(
        benchmark_id="bench-001",
        run_id=request.run_id,
        suite=request.suite,
        suite_version=request.suite_version,
        model_id=request.model_uri,
        model_sha256=request.model_sha256,
        seed=request.seed,
        split=request.split,
        metrics=BenchmarkMetrics(aggregate=0.5, per_environment={"WebShop": 0.5}),
        report_artifact=_artifact("report-001"),
        manifest_sha256=canonical_manifest_sha256(
            {"suite": request.suite, "suite_version": request.suite_version, "seed": request.seed}
        ),
        evidence_label=EvidenceLabel.LIVE,
        verified=True,
    )


def test_objective_result_requires_real_artifact_for_live_evidence() -> None:
    with pytest.raises(ValueError, match="artifact reference"):
        ObjectiveBenchmarkResult(
            benchmark_id="bench-001",
            run_id="run-001",
            suite="AgentGym/WebShop",
            suite_version="2026-09-01",
            model_id="s3://bucket/base.tar.gz?versionId=base-v1",
            model_sha256="a" * 64,
            seed=7,
            split="baseline",
            metrics=BenchmarkMetrics(aggregate=0.5),
            manifest_sha256="a" * 64,
            evidence_label=EvidenceLabel.LIVE,
            verified=True,
        )


def test_objective_worker_result_is_provenance_checked() -> None:
    request = ObjectiveBenchmarkRequest(
        run_id="run-001",
        model_uri="s3://bucket/base.tar.gz?versionId=base-v1",
        model_sha256="a" * 64,
        suite="AgentGym/WebShop",
        suite_version="2026-09-01",
        seed=7,
        num_episodes=10,
        split="train",
        output_s3_uri="s3://bucket/runs/run-001",
    )

    class Worker:
        def execute_benchmark(
            self, received: ObjectiveBenchmarkRequest
        ) -> ObjectiveBenchmarkResult:
            assert received == request
            return _result(received)

    result = execute_objective_benchmark(Worker(), request)
    evidence = result.evidence(kind=EvidenceKind.BENCHMARK)
    assert result.metrics.aggregate == 0.5
    assert evidence.label is EvidenceLabel.LIVE
    assert evidence.artifact_ids == ("report-001",)

    mismatch = _result(request).model_copy(update={"model_sha256": "b" * 64})

    class BadWorker:
        def execute_benchmark(
            self, received: ObjectiveBenchmarkRequest
        ) -> ObjectiveBenchmarkResult:
            del received
            return mismatch

    with pytest.raises(ValueError, match="provenance"):
        execute_objective_benchmark(BadWorker(), request)


@pytest.mark.parametrize("split", ["baseline", "held_out", "hidden"])
def test_training_benchmark_request_rejects_non_training_splits(split: str) -> None:
    with pytest.raises(ValueError):
        ObjectiveBenchmarkRequest(
            run_id="run-001",
            model_uri="s3://bucket/base.tar.gz?versionId=base-v1",
            model_sha256="a" * 64,
            suite="AgentGym/WebShop",
            suite_version="2026-09-01",
            seed=7,
            num_episodes=10,
            split=split,
            output_s3_uri="s3://bucket/runs/run-001",
        )


class FakeProvider:
    def __init__(self, training: list[JobStatus], evaluation: list[JobStatus]) -> None:
        self.training = deque(training)
        self.evaluation = deque(evaluation)
        self.submitted: list[str] = []

    def submit_training(self, request: TrainingJobRequest) -> JobResult:
        self.submitted.append(f"train:{request.job_name}")
        return JobResult(request.job_name, "arn:train", JobStatus.SUBMITTED)

    def get_training_status(self, job_name: str) -> JobResult:
        return JobResult(job_name, "arn:train", self.training.popleft())

    def submit_evaluation(self, request: EvaluationJobRequest) -> JobResult:
        self.submitted.append(f"eval:{request.job_name}")
        return JobResult(request.job_name, "arn:eval", JobStatus.SUBMITTED)

    def get_evaluation_status(self, job_name: str) -> JobResult:
        return JobResult(job_name, "arn:eval", self.evaluation.popleft())


def _training_request() -> TrainingJobRequest:
    return TrainingJobRequest(
        job_name="train-run-001",
        role_arn="arn:role",
        image_uri="123.dkr.ecr/image:latest",
        input_s3_uri="s3://bucket/data",
        output_s3_uri="s3://bucket/model",
        instance_type="ml.g5.xlarge",
    )


def _evaluation_request() -> EvaluationJobRequest:
    return EvaluationJobRequest(
        job_name="eval-run-001",
        role_arn="arn:role",
        image_uri="123.dkr.ecr/eval:latest",
        input_s3_uri="s3://bucket/model",
        output_s3_uri="s3://bucket/eval",
        instance_type="ml.g5.xlarge",
    )


def test_train_then_evaluate_waits_for_training_before_submitting_evaluation() -> None:
    provider = FakeProvider(
        [JobStatus.IN_PROGRESS, JobStatus.COMPLETED], [JobStatus.COMPLETED]
    )
    sleeps: list[float] = []
    result = train_then_evaluate(
        provider,
        _training_request(),
        _evaluation_request(),
        policy=JobWaitPolicy(max_attempts=3, poll_interval_seconds=0.25),
        sleep=sleeps.append,
    )
    assert result.training.status is JobStatus.COMPLETED
    assert result.evaluation.status is JobStatus.COMPLETED
    assert provider.submitted == ["train:train-run-001", "eval:eval-run-001"]
    assert sleeps == [0.25]


def test_waiting_is_bounded_and_unknown_status_fails_closed() -> None:
    provider = FakeProvider([JobStatus.IN_PROGRESS, JobStatus.IN_PROGRESS], [])
    with pytest.raises(JobWaitTimeout, match="after 2 polls"):
        wait_for_training_job(
            provider,
            "train-run-001",
            policy=JobWaitPolicy(max_attempts=2, poll_interval_seconds=0),
            sleep=lambda _: None,
        )

    unknown = FakeProvider([JobStatus.UNKNOWN], [])
    with pytest.raises(JobExecutionFailed, match="UNKNOWN"):
        wait_for_training_job(
            unknown,
            "train-run-001",
            policy=JobWaitPolicy(max_attempts=1),
            sleep=lambda _: None,
        )


def test_failed_training_never_submits_evaluation() -> None:
    provider = FakeProvider([JobStatus.FAILED], [])
    with pytest.raises(JobExecutionFailed, match="ended in failed"):
        train_then_evaluate(
            provider,
            _training_request(),
            _evaluation_request(),
            policy=JobWaitPolicy(max_attempts=1),
            sleep=lambda _: None,
        )
    assert provider.submitted == ["train:train-run-001"]


def test_wait_evaluation_reaches_terminal_state() -> None:
    provider = FakeProvider([], [JobStatus.IN_PROGRESS, JobStatus.STOPPED])
    sleeps: list[float] = []
    result = wait_for_evaluation_job(
        provider,
        "eval-run-001",
        policy=JobWaitPolicy(max_attempts=2, poll_interval_seconds=1),
        sleep=sleeps.append,
    )
    assert result.status is JobStatus.STOPPED
    assert sleeps == [1]
