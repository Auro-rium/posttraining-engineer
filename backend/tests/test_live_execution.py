from __future__ import annotations

from base64 import urlsafe_b64decode, urlsafe_b64encode
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from app.live_execution import (
    ApprovalPacket,
    CheckStatus,
    GpuCapacityStatus,
    GpuQuotaStatus,
    LiveExecutionBlocked,
    LiveExecutionConfig,
    ObjectiveWorkerClient,
    PreflightClassification,
    PreflightRunner,
    _decode_approval_token,
    config_from_environment,
    issue_approval_token,
)


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
