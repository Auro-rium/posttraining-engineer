from __future__ import annotations

from base64 import urlsafe_b64decode, urlsafe_b64encode
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from pydantic import ValidationError

from app.live_execution import (
    ApprovalPacket,
    AutonomousRunController,
    CheckStatus,
    GpuCapacityStatus,
    GpuQuotaStatus,
    LiveExecutionBlocked,
    LiveExecutionConfig,
    LiveExecutionFailed,
    ObjectiveWorkerClient,
    PreflightClassification,
    PreflightRunner,
    _decode_approval_token,
    config_from_environment,
    issue_approval_token,
)
from app.observability import TelemetryRecorder
from app.posttraining.models import ArtifactKind, ArtifactReference, EvidenceLabel
from app.posttraining.objective_workflow import ObjectiveBenchmarkRequest, ObjectiveBenchmarkResult
from app.posttraining.run_history import BenchmarkMetrics, RunHistoryRecord
from app.providers.sagemaker import JobResult, JobStatus


def _config(**overrides: object) -> LiveExecutionConfig:
    values: dict[str, object] = {
        "artifact_bucket": "demo-bucket",
        "dynamodb_table": "demo-history",
        "training_role_arn": "arn:aws:iam::123456789012:role/train",
        "training_image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/train:latest",
        "evaluation_image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/eval:latest",
        "objective_worker_url": "https://worker.example.com",
        "hf_repo_id": "google/functiongemma-270m-it",
        "hf_revision": "a" * 40,
        "training_input_s3_uri": "s3://demo-bucket/input/checkpoint",
        "evaluation_input_s3_uri": "s3://demo-bucket/input/held-out",
        "max_runtime_seconds": 3600,
    }
    values.update(overrides)
    return LiveExecutionConfig.model_validate(values)


def test_live_config_requires_immutable_hf_commit() -> None:
    with pytest.raises(ValidationError, match="immutable commit SHA"):
        _config(hf_revision="main")


def test_live_config_enforces_hard_cost_ceiling() -> None:
    with pytest.raises(ValidationError, match="exceeds"):
        _config(max_runtime_seconds=7200, training_hourly_cost_usd=2.0)


def test_objective_worker_requires_https() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        ObjectiveWorkerClient("http://worker.example.com")


def test_environment_config_fails_closed_when_required_inputs_missing() -> None:
    with pytest.raises(LiveExecutionBlocked, match="missing live configuration"):
        config_from_environment({})


def test_gpu_preflight_blocks_instance_outside_explicit_allowlist() -> None:
    config = _config(
        instance_type="ml.g5.2xlarge",
        gpu_instance_allowlist=("ml.g5.xlarge",),
        sagemaker_gpu_quota_code="L-0123456789abcdef0",
    )

    check = PreflightRunner(config)._check_gpu_readiness()

    assert check.status is CheckStatus.BLOCKED
    assert check.classification is PreflightClassification.BLOCKED_GPU_ALLOWLIST
    assert check.metadata["instance_type"] == "ml.g5.2xlarge"


class _ReadOnlyQuotaClient:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get_service_quota(self, **kwargs: str) -> dict[str, object]:
        self.calls.append(("get_service_quota", kwargs))
        return {"Quota": {"Value": self.value}}

    def __getattr__(self, name: str) -> object:
        if name.startswith(("put", "create", "delete", "update", "start", "stop")):
            raise AssertionError(f"preflight attempted mutating quota operation: {name}")
        raise AttributeError(name)


def test_gpu_preflight_blocks_insufficient_quota_without_mutating_aws() -> None:
    quota_client = _ReadOnlyQuotaClient(value=0)
    config = _config(sagemaker_gpu_quota_code="L-0123456789abcdef0")

    check = PreflightRunner(config, clients={"service-quotas": quota_client})._check_gpu_readiness()

    assert check.status is CheckStatus.BLOCKED
    assert check.classification is PreflightClassification.BLOCKED_GPU_QUOTA
    assert quota_client.calls == [
        (
            "get_service_quota",
            {"ServiceCode": "sagemaker", "QuotaCode": "L-0123456789abcdef0"},
        )
    ]


def test_gpu_preflight_reports_quota_coverage_as_non_mutating_readiness() -> None:
    quota_client = _ReadOnlyQuotaClient(value=1)
    config = _config(sagemaker_gpu_quota_code="L-0123456789abcdef0")

    runner = PreflightRunner(config, clients={"service-quotas": quota_client})
    check = runner._check_gpu_readiness()

    assert check.status is CheckStatus.PASSED
    assert runner._gpu_quota_status is GpuQuotaStatus.VERIFIED
    assert runner._gpu_capacity_status is GpuCapacityStatus.VERIFIED_BY_QUOTA
    assert quota_client.calls
    assert all(call[0] == "get_service_quota" for call in quota_client.calls)


def _approval_packet(**overrides: object) -> ApprovalPacket:
    values: dict[str, object] = {
        "run_id": "run-approval",
        "run_number": 1,
        "instance_type": "ml.g5.xlarge",
        "instance_count": 1,
        "volume_size_gb": 30,
        "max_runtime_seconds": 600,
        "estimated_cost_usd": 0.5,
        "manifest_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "issued_at": datetime.now(UTC),
        "expires_at": datetime.now(UTC) + timedelta(minutes=10),
    }
    values.update(overrides)
    return ApprovalPacket.model_validate(values)


def test_approval_token_is_bound_to_packet_and_rejects_tampering() -> None:
    packet = _approval_packet()
    token = issue_approval_token(packet, "demo-approval-secret")

    decoded = _decode_approval_token(token, "demo-approval-secret")
    assert decoded.digest == packet.digest

    other_packet = _approval_packet(run_number=2)
    assert other_packet.digest != packet.digest

    with pytest.raises(LiveExecutionBlocked, match="signature"):
        _decode_approval_token(token, "wrong-secret")

    version, payload, signature = token.split(".")
    tampered = ApprovalPacket.model_validate_json(
        urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
    ).model_copy(update={"run_number": 2})
    tampered_payload = urlsafe_b64encode(tampered.canonical_bytes()).decode("ascii").rstrip("=")
    with pytest.raises(LiveExecutionBlocked, match="signature"):
        _decode_approval_token(f"{version}.{tampered_payload}.{signature}", "demo-approval-secret")


def _controller(**overrides: object) -> AutonomousRunController:
    return AutonomousRunController(
        config=_config(**overrides),
        objective_worker=object(),
        provider=cast(Any, object()),
        artifact_store=cast(Any, object()),
        slots=cast(Any, object()),
    )


def _champion_record(*, kind: ArtifactKind = ArtifactKind.CHECKPOINT) -> RunHistoryRecord:
    artifact = ArtifactReference(
        artifact_id="champion-checkpoint",
        kind=kind,
        uri="s3://demo-bucket/champion/model.tar.gz?versionId=v42",
        sha256="c" * 64,
    )
    return RunHistoryRecord(
        run_id="run-1",
        run_number=1,
        candidate_artifact_id=artifact.artifact_id,
        artifact_refs=(artifact,),
    )


def test_promoted_champion_checkpoint_uri_and_digest_are_selected_from_artifact() -> None:
    controller = _controller(checkpoint_s3_uri="s3://demo-bucket/initial/model.tar.gz")
    champion = _champion_record()

    assert controller._checkpoint_uri(champion) == (
        "s3://demo-bucket/champion/model.tar.gz?versionId=v42"
    )
    assert controller._checkpoint_sha256(champion) == "c" * 64

    payload = controller._manifest_payload(
        run_id="run-2",
        run_number=2,
        parent_run_id="run-1",
        champion_run_id="run-1",
        checkpoint_sha256="c" * 64,
        checkpoint_uri=controller._checkpoint_uri(champion),
    )
    assert payload["checkpoint_s3_uri"] == (
        "s3://demo-bucket/champion/model.tar.gz?versionId=v42"
    )


def test_promoted_champion_must_reference_a_checkpoint_artifact() -> None:
    controller = _controller()
    with pytest.raises(LiveExecutionBlocked, match="not a checkpoint"):
        controller._checkpoint_uri(_champion_record(kind=ArtifactKind.REPORT))


def test_live_benchmark_rejects_result_with_different_manifest() -> None:
    class Worker:
        def execute_benchmark(
            self, request: ObjectiveBenchmarkRequest
        ) -> ObjectiveBenchmarkResult:
            received = request
            return ObjectiveBenchmarkResult(
                benchmark_id="benchmark-1",
                run_id=received.run_id,
                suite=received.suite,
                suite_version=received.suite_version,
                model_id=received.model_uri,
                seed=received.seed,
                split=received.split,
                metrics=BenchmarkMetrics(aggregate=0.5, per_environment={"web": 0.5}),
                report_artifact=ArtifactReference(
                    artifact_id="report-1",
                    kind=ArtifactKind.REPORT,
                    uri="s3://demo-bucket/report.json",
                    sha256="d" * 64,
                ),
                manifest_sha256="e" * 64,
                evidence_label=EvidenceLabel.LIVE,
                verified=True,
            )

    controller = _controller()
    controller.objective_worker = Worker()
    with pytest.raises(LiveExecutionFailed, match="manifest"):
        controller._benchmark(
            run_id="run-1",
            model_uri="s3://demo-bucket/base/model.tar.gz",
            split="baseline",
            episodes=2,
            output_s3_uri="s3://demo-bucket/run-1/baseline",
            manifest_sha256="f" * 64,
        )


def test_cleanup_telemetry_contains_provider_job_id_and_phase() -> None:
    events: list[dict[str, object]] = []

    class Provider:
        def stop_training(self, job_name: str) -> None:
            assert job_name == "train-1"

        def stop_evaluation(self, job_name: str) -> None:
            raise AssertionError("evaluation cleanup should not be called")

    controller = AutonomousRunController(
        config=_config(),
        objective_worker=object(),
        provider=cast(Any, Provider()),
        artifact_store=cast(Any, object()),
        slots=cast(Any, object()),
        telemetry=TelemetryRecorder(exporter=events.append, logger=None, tracer=None),
    )
    controller._cleanup(
        JobResult("train-1", "arn:train-1", JobStatus.IN_PROGRESS),
        None,
        run_id="run-1",
        run_number=1,
    )

    assert events[-1]["event_type"] == "cleanup.completed"
    assert events[-1]["phase"] == "training"
    assert events[-1]["job_id"] == "arn:train-1"
