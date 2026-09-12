from __future__ import annotations

import hashlib
from email.message import Message
from io import BytesIO
from typing import Any
from urllib.error import HTTPError

import pytest

import app.live_execution as live_execution
from app.live_execution import (
    LiveExecutionBlocked,
    ObjectiveWorkerClient,
    PreflightRunner,
)


def _config(**overrides: object) -> Any:
    values: dict[str, object] = {
        "artifact_bucket": "demo-bucket",
        "dynamodb_table": "demo-history",
        "training_role_arn": "arn:aws:iam::123456789012:role/train",
        "training_image": (
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/train@sha256:"
            + "a" * 64
        ),
        "evaluation_image": (
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/eval@sha256:"
            + "b" * 64
        ),
        "objective_worker_url": "https://worker.example.com",
        "objective_worker_auth_token": "worker-secret",
        "hf_repo_id": "google/functiongemma-270m-it",
        "hf_revision": "c" * 40,
        "training_input_s3_uri": "s3://demo-bucket/post-training/inputs/training",
        "evaluation_input_s3_uri": "s3://demo-bucket/post-training/inputs/evaluation",
        "checkpoint_s3_uri": "s3://demo-bucket/checkpoints/base.tar.gz?versionId=v1",
        "checkpoint_sha256": "d" * 64,
        "sagemaker_gpu_quota_code": "L-0123456789abcdef0",
        "max_runtime_seconds": 3600,
    }
    values.update(overrides)
    from app.live_execution import LiveExecutionConfig

    return LiveExecutionConfig.model_validate(values)


class _BedrockCatalog:
    def get_foundation_model(self, **kwargs: object) -> dict[str, object]:
        assert kwargs == {"modelIdentifier": "nvidia.nemotron-super-3-120b"}
        return {"modelDetails": {"providerName": "NVIDIA"}}


class _BedrockRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def converse(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return {"output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}}}


def test_bedrock_preflight_requires_a_minimal_nemotron_runtime_probe() -> None:
    runtime = _BedrockRuntime()
    runner = PreflightRunner(
        _config(), clients={"bedrock": _BedrockCatalog(), "bedrock-runtime": runtime}
    )

    metadata = runner._check_bedrock_readiness()

    assert metadata["invocation"] == "verified"
    assert runtime.calls == [
        {
            "modelId": "nvidia.nemotron-super-3-120b",
            "messages": [
                {"role": "user", "content": [{"text": "Reply with OK."}]}
            ],
            "inferenceConfig": {"maxTokens": 1, "temperature": 0.0},
        }
    ]


def test_bedrock_client_factory_forces_sigv4(monkeypatch: pytest.MonkeyPatch) -> None:
    import boto3  # type: ignore[import-untyped]

    calls: list[dict[str, object]] = []

    def fake_client(name: str, **kwargs: object) -> object:
        calls.append({"name": name, **kwargs})
        return object()

    monkeypatch.setattr(boto3, "client", fake_client)
    runner = PreflightRunner(_config())

    runner._client("bedrock-runtime")

    assert calls[0]["name"] == "bedrock-runtime"
    client_config = calls[0]["config"]
    assert getattr(client_config, "signature_version", None) == "v4"


def test_objective_worker_health_requires_bearer_auth_without_exposing_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []

    def fake_http_json(url: str, **kwargs: object) -> dict[str, object]:
        requests.append({"url": url, **kwargs})
        if url.endswith("v1/auth-probe"):
            return {"status": "authenticated", "service": "objective-worker"}
        return {"status": "healthy"}

    monkeypatch.setattr(live_execution, "_http_json", fake_http_json)
    client = ObjectiveWorkerClient("https://worker.example.com", auth_token="worker-secret")

    assert client.health()["status"] == "healthy"
    assert requests[0]["headers"] == {"Authorization": "Bearer worker-secret"}
    assert "worker-secret" not in repr(client)


def test_objective_worker_without_auth_token_fails_closed() -> None:
    client = ObjectiveWorkerClient("https://worker.example.com")

    with pytest.raises(LiveExecutionBlocked, match="authentication"):
        client.health()


def test_preflight_proves_worker_token_with_protected_non_mutating_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []

    def fake_http_json(url: str, **kwargs: object) -> dict[str, object]:
        requests.append({"url": url, **kwargs})
        if url.endswith("v1/auth-probe"):
            return {"status": "authenticated", "service": "objective-worker"}
        if url.endswith("/health"):
            return {"status": "healthy"}
        if url.endswith("v1/readiness"):
            return {
                "status": "ready",
                "service": "objective-worker",
                "capabilities": {"benchmark": True, "verify-curation": True},
            }
        raise AssertionError(f"unexpected objective readiness request: {url}")

    monkeypatch.setattr(live_execution, "_http_json", fake_http_json)
    runner = live_execution.PreflightRunner(_config())

    assert runner._check_worker_readiness()["status"] == "ready"
    assert [request["url"].rsplit("/", 1)[-1] for request in requests] == [
        "health",
        "readiness",
        "auth-probe",
    ]
    assert all(
        request["headers"] == {"Authorization": "Bearer worker-secret"}
        for request in requests
    )
    assert requests[1]["method"] == "GET"
    assert "payload" not in requests[1]


def test_preflight_does_not_treat_schema_rejection_as_worker_execution_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_http_json(url: str, **kwargs: object) -> dict[str, object]:
        if url.endswith("/health"):
            return {"status": "healthy"}
        if url.endswith("v1/auth-probe"):
            return {"status": "authenticated", "service": "objective-worker"}
        if kwargs.get("method") == "POST":
            raise HTTPError(url, 422, "request validation", Message(), BytesIO(b"{}"))
        raise AssertionError(f"unexpected objective readiness request: {url}")

    monkeypatch.setattr(live_execution, "_http_json", fake_http_json)
    runner = PreflightRunner(_config())

    with pytest.raises(LiveExecutionBlocked, match="execution readiness"):
        runner._check_worker_readiness()


def test_preflight_rejects_objective_readiness_route_without_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_http_json(url: str, **kwargs: object) -> dict[str, object]:
        if url.endswith("/health"):
            return {"status": "healthy"}
        raise HTTPError(url, 401, "unauthorized", Message(), BytesIO(b"{}"))

    monkeypatch.setattr(live_execution, "_http_json", fake_http_json)
    runner = PreflightRunner(_config())

    with pytest.raises(LiveExecutionBlocked, match="execution readiness"):
        runner._check_worker_readiness()


def test_preflight_rejects_worker_token_when_protected_probe_returns_unauthorized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_http_json(url: str, **kwargs: object) -> dict[str, object]:
        del kwargs
        if url.endswith("v1/auth-probe"):
            return {"status": "unauthorized", "service": "objective-worker"}
        if url.endswith("/health"):
            return {"status": "healthy"}
        if url.endswith("v1/readiness"):
            return {
                "status": "ready",
                "service": "objective-worker",
                "capabilities": {"benchmark": True, "verify-curation": True},
            }
        raise AssertionError(f"unexpected objective readiness request: {url}")

    monkeypatch.setattr(live_execution, "_http_json", fake_http_json)
    runner = live_execution.PreflightRunner(_config())

    with pytest.raises(LiveExecutionBlocked, match="authentication"):
        runner._check_worker_readiness()


class _ReadOnlyS3:
    def __init__(self, *, checkpoint_digest: str, location: str | None = "us-east-1") -> None:
        self.checkpoint_digest = checkpoint_digest
        self.location = location
        self.calls: list[tuple[str, dict[str, object]]] = []

    def head_bucket(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("head_bucket", kwargs))
        return {}

    def get_bucket_location(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("get_bucket_location", kwargs))
        return {"LocationConstraint": self.location}

    def get_bucket_encryption(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("get_bucket_encryption", kwargs))
        return {
            "ServerSideEncryptionConfiguration": {
                "Rules": [
                    {"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}
                ]
            }
        }

    def get_bucket_versioning(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("get_bucket_versioning", kwargs))
        return {"Status": "Enabled"}

    def head_object(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("head_object", kwargs))
        return {"Metadata": {"sha256": self.checkpoint_digest}, "VersionId": "v1"}

    def __getattr__(self, name: str) -> object:
        if name.startswith(("put", "create", "delete", "update", "start", "stop")):
            raise AssertionError(f"preflight attempted mutating S3 operation: {name}")
        raise AttributeError(name)


def test_s3_preflight_requires_region_encryption_and_versioned_checkpoint() -> None:
    digest = hashlib.sha256(b"checkpoint").hexdigest()
    s3 = _ReadOnlyS3(checkpoint_digest=digest)
    config = _config(checkpoint_sha256=digest)
    runner = PreflightRunner(config, clients={"s3": s3})

    bucket = runner._check_s3_readiness()
    checkpoint = runner._check_checkpoint_readiness()

    assert bucket["region"] == "us-east-1"
    assert bucket["encryption"] == "AES256"
    assert checkpoint["version_id"] == "v1"
    assert not any(name == "head_bucket" for name, _ in s3.calls)
    assert (
        "head_object",
        {"Bucket": "demo-bucket", "Key": "checkpoints/base.tar.gz", "VersionId": "v1"},
    ) in s3.calls


class _ReadOnlyDynamoDB:
    def __init__(self, key_schema: list[dict[str, str]]) -> None:
        self.key_schema = key_schema

    def describe_table(self, **kwargs: object) -> dict[str, object]:
        assert kwargs == {"TableName": "demo-history"}
        return {
            "Table": {
                "TableStatus": "ACTIVE",
                "KeySchema": self.key_schema,
            }
        }


def test_dynamodb_preflight_requires_the_repository_key_schema() -> None:
    runner = PreflightRunner(
        _config(),
        clients={
            "dynamodb": _ReadOnlyDynamoDB(
                [{"AttributeName": "id", "KeyType": "HASH"}]
            )
        },
    )

    with pytest.raises(LiveExecutionBlocked, match="pk HASH and sk RANGE"):
        runner._check_dynamodb_readiness()


def test_dynamodb_preflight_accepts_the_repository_key_schema() -> None:
    runner = PreflightRunner(
        _config(),
        clients={
            "dynamodb": _ReadOnlyDynamoDB(
                [
                    {"AttributeName": "pk", "KeyType": "HASH"},
                    {"AttributeName": "sk", "KeyType": "RANGE"},
                ]
            )
        },
    )

    assert runner._check_dynamodb_readiness()["key_schema"] == "pk/sk"


class _ReadOnlyS3Inputs:
    def __init__(self, contents: list[dict[str, object]]) -> None:
        self.contents = contents
        self.calls: list[dict[str, object]] = []

    def list_objects_v2(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return {"Contents": self.contents, "KeyCount": len(self.contents)}


def test_sagemaker_input_preflight_requires_a_nonempty_prefix() -> None:
    client = _ReadOnlyS3Inputs([])
    runner = PreflightRunner(_config(), clients={"s3": client})

    with pytest.raises(LiveExecutionBlocked, match="training input prefix is empty"):
        runner._check_input_readiness("training")

    assert client.calls == [
        {
            "Bucket": "demo-bucket",
            "Prefix": "post-training/inputs/training/",
            "MaxKeys": 1,
        }
    ]


def test_sagemaker_input_preflight_rejects_inputs_outside_artifact_scope() -> None:
    config = _config(training_input_s3_uri="s3://other-bucket/post-training/inputs/training")
    runner = PreflightRunner(config, clients={"s3": _ReadOnlyS3Inputs([])})

    with pytest.raises(LiveExecutionBlocked, match="artifact bucket and prefix"):
        runner._check_input_readiness("training")


def test_checkpoint_preflight_rejects_unversioned_uri() -> None:
    config = _config(checkpoint_s3_uri="s3://demo-bucket/checkpoints/base.tar.gz")
    runner = PreflightRunner(config, clients={"s3": _ReadOnlyS3(checkpoint_digest="d" * 64)})

    with pytest.raises(LiveExecutionBlocked, match="versionId"):
        runner._check_checkpoint_readiness()


def test_checkpoint_preflight_rejects_a_different_bucket() -> None:
    config = _config(
        checkpoint_s3_uri="s3://other-bucket/checkpoints/base.tar.gz?versionId=v1"
    )
    runner = PreflightRunner(config, clients={"s3": _ReadOnlyS3(checkpoint_digest="d" * 64)})

    with pytest.raises(LiveExecutionBlocked, match="artifact bucket"):
        runner._check_checkpoint_readiness()


def test_s3_legacy_eu_location_is_eu_west_1() -> None:
    s3 = _ReadOnlyS3(checkpoint_digest="d" * 64, location="EU")
    config = _config(aws_region="eu-west-1")

    assert PreflightRunner(config, clients={"s3": s3})._check_s3_readiness()["region"] == (
        "eu-west-1"
    )


def test_live_config_serialization_excludes_objective_worker_token() -> None:
    config = _config()

    assert "worker-secret" not in repr(config)
    assert "objective_worker_auth_token" not in config.model_dump()
    assert "worker-secret" not in config.model_dump_json()


class _ReadOnlyIdentity:
    def get_caller_identity(self) -> dict[str, str]:
        return {"Account": "123456789012"}


class _ReadOnlyIam:
    def __init__(
        self,
        expected_role_name: str = "train",
        *,
        service_trust: str = "sagemaker.amazonaws.com",
    ) -> None:
        self.expected_role_name = expected_role_name
        self.service_trust = service_trust

    def get_role(self, **kwargs: object) -> dict[str, object]:
        assert kwargs == {"RoleName": self.expected_role_name}
        return {
            "Role": {
                "RoleName": "train",
                "AssumeRolePolicyDocument": {
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": self.service_trust},
                            "Action": "sts:AssumeRole",
                        }
                    ]
                },
            }
        }


class _ReadOnlyEcr:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def describe_images(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        image_id = kwargs["imageIds"]
        assert isinstance(image_id, list)
        digest = str(image_id[0]["imageDigest"])
        return {
            "imageDetails": [
                {
                    "imageDigest": digest,
                    "repositoryName": str(kwargs["repositoryName"]),
                    "registryId": "123456789012",
                }
            ]
        }

    def __getattr__(self, name: str) -> object:
        if name.startswith(("put", "create", "delete", "update", "start", "stop")):
            raise AssertionError(f"preflight attempted mutating ECR operation: {name}")
        raise AttributeError(name)


def test_ecr_preflight_requires_owned_digest_pinned_images() -> None:
    ecr = _ReadOnlyEcr()
    runner = PreflightRunner(
        _config(), clients={"ecr": ecr, "sts": _ReadOnlyIdentity(), "iam": _ReadOnlyIam()}
    )

    metadata = runner._check_sagemaker_readiness()

    assert metadata["images"] == "available"
    for call in ecr.calls:
        image_ids = call["imageIds"]
        assert isinstance(image_ids, list)
        assert isinstance(image_ids[0], dict)
        assert "imageDigest" in image_ids[0]


def test_ecr_preflight_rejects_tag_and_cross_region_images() -> None:
    config = _config(
        training_image="999999999999.dkr.ecr.eu-west-1.amazonaws.com/train:latest"
    )
    runner = PreflightRunner(
        config,
        clients={"ecr": _ReadOnlyEcr(), "sts": _ReadOnlyIdentity(), "iam": _ReadOnlyIam()},
    )

    with pytest.raises(LiveExecutionBlocked, match="digest-pinned"):
        runner._check_sagemaker_readiness()


def test_sagemaker_preflight_rejects_role_without_sagemaker_trust() -> None:
    runner = PreflightRunner(
        _config(),
        clients={
            "ecr": _ReadOnlyEcr(),
            "sts": _ReadOnlyIdentity(),
            "iam": _ReadOnlyIam(service_trust="ecs-tasks.amazonaws.com"),
        },
    )

    with pytest.raises(LiveExecutionBlocked, match="SageMaker service trust"):
        runner._check_sagemaker_readiness()


def test_sagemaker_preflight_preserves_iam_role_path() -> None:
    config = _config(training_role_arn="arn:aws:iam::123456789012:role/service/train")
    iam = _ReadOnlyIam(expected_role_name="service/train")
    runner = PreflightRunner(
        config,
        clients={"ecr": _ReadOnlyEcr(), "sts": _ReadOnlyIdentity(), "iam": iam},
    )

    runner._check_sagemaker_readiness()
