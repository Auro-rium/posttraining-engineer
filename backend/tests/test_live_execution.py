from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.live_execution import (
    LiveExecutionBlocked,
    LiveExecutionConfig,
    ObjectiveWorkerClient,
    config_from_environment,
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
