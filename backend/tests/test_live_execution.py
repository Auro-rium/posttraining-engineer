from __future__ import annotations

import hashlib
import io
import json
import tarfile
from base64 import urlsafe_b64decode, urlsafe_b64encode
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError

from app.live_execution import (
    ApprovalPacket,
    AutonomousRunController,
    CheckStatus,
    GpuCapacityStatus,
    GpuQuotaStatus,
    LiveEvaluationReader,
    LiveExecutionBlocked,
    LiveExecutionConfig,
    LiveExecutionFailed,
    LiveObjectiveAdapter,
    LiveRequestFactory,
    ObjectiveWorkerClient,
    PreflightClassification,
    PreflightRunner,
    _decode_approval_token,
    config_from_environment,
    create_autonomous_live_components,
    issue_approval_token,
)
from app.objective.engine import ServiceRecoveryEngine
from app.objective.models import (
    CurationRequest,
    ObjectiveSplit,
    ToolCall,
    TrajectoryReference,
)
from app.objective.service import InMemoryTrajectoryArtifactStore, ObjectiveService
from app.observability import TelemetryRecorder
from app.posttraining.models import ArtifactKind, ArtifactReference, EvidenceLabel
from app.posttraining.objective_workflow import ObjectiveBenchmarkRequest, ObjectiveBenchmarkResult
from app.posttraining.run_history import BenchmarkMetrics, RunHistoryRecord
from app.providers.artifacts import S3ArtifactStore
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


def test_live_timing_defaults_cover_the_full_five_experiment_window() -> None:
    config = _config(max_runtime_seconds=7200)

    assert config.approval_ttl_seconds == 24 * 60 * 60
    assert config.minimum_approval_ttl_seconds == 5 * 2 * 7200
    assert config.provider_max_polls == 241
    assert config.provider_poll_interval_seconds == 30.0


def test_live_config_rejects_approval_ttl_shorter_than_bounded_run() -> None:
    with pytest.raises(ValidationError, match="approval_ttl_seconds"):
        _config(max_runtime_seconds=7200, approval_ttl_seconds=71999)


def test_objective_worker_requires_https() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        ObjectiveWorkerClient("http://worker.example.com")


@pytest.mark.parametrize(
    ("base_url", "expected_url"),
    [
        (
            "https://worker.example.com",
            "https://worker.example.com/v1/benchmark",
        ),
        (
            "https://worker.example.com/",
            "https://worker.example.com/v1/benchmark",
        ),
        (
            "https://gateway.example.com/v1/",
            "https://gateway.example.com/v1/benchmark",
        ),
        (
            "https://proxy.example.com/objective/v1/",
            "https://proxy.example.com/objective/v1/benchmark",
        ),
    ],
)
def test_objective_worker_benchmark_uses_bounded_timeout_and_normalized_path(
    monkeypatch: pytest.MonkeyPatch,
    base_url: str,
    expected_url: str,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def request(url: str, **kwargs: Any) -> list[object]:
        calls.append((url, kwargs))
        return []

    monkeypatch.setattr("app.live_execution._http_json", request)
    worker = ObjectiveWorkerClient(base_url, auth_token="worker-secret")
    benchmark_request = ObjectiveBenchmarkRequest(
        run_id="run-1",
        model_uri="s3://bucket/model?versionId=v1",
        model_sha256="a" * 64,
        suite="AgentGym/AgentEval",
        suite_version="agent-eval-v1",
        seed=7,
        num_episodes=10,
        split="train",
        output_s3_uri="s3://bucket/output/run-1",
    )

    with pytest.raises(LiveExecutionFailed, match="non-object benchmark"):
        worker.execute_benchmark(benchmark_request)

    assert calls[0][0] == expected_url
    assert calls[0][1]["headers"] == {"Authorization": "Bearer worker-secret"}
    assert calls[0][1]["timeout"] == 600.0
    with pytest.raises(ValueError, match="600"):
        ObjectiveWorkerClient(
            "https://worker.example.com",
            auth_token="worker-secret",
            timeout_seconds=601,
        )


def test_live_request_factory_pins_worker_inputs_and_query_free_artifacts() -> None:
    config = _config(training_input_s3_uri=None)
    dataset_sha = "a" * 64
    base_sha = "b" * 64
    dataset = SimpleNamespace(
        dataset_id="dataset-1",
        uri=(
            f"s3://demo-bucket/datasets/run-1/1/{dataset_sha}/dataset.jsonl"
            "?versionId=dataset-v1"
        ),
        sha256=dataset_sha,
        artifact_id="dataset://dataset-1",
    )
    state = SimpleNamespace(
        run_id="run-1",
        benchmark_manifest_sha256="d" * 64,
        benchmark_version="agent-eval-v1",
        benchmark_seed=7,
        model_id="google/functiongemma-270m-it",
        checkpoint_revision="a" * 40,
        base_checkpoint_uri=f"s3://demo-bucket/checkpoints/{base_sha}.tar.gz?versionId=base-v1",
        base_checkpoint_sha256=base_sha,
        approval_scope={
            "instance_type": config.instance_type,
            "instance_count": config.instance_count,
            "volume_size_gb": config.volume_size_gb,
            "max_runtime_seconds": config.max_runtime_seconds,
        },
    )
    qlora = {
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
        "learning_rate": 0.0002,
        "epochs": 1,
        "sequence_length": 512,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
    factory = LiveRequestFactory(config)

    training = factory.training(
        state,
        experiment_number=1,
        dataset=dataset,
        config=SimpleNamespace(model_dump=lambda **_: qlora),
    )
    evaluation = factory.evaluation(
        state,
        experiment_number=1,
        candidate=SimpleNamespace(
            uri=f"s3://demo-bucket/checkpoints/{'c' * 64}.tar.gz?versionId=candidate-v1",
            sha256="c" * 64,
            artifact_id="checkpoint://" + "c" * 64,
        ),
    )

    assert training.input_s3_uri == f"s3://demo-bucket/datasets/run-1/1/{dataset_sha}"
    assert "?" not in training.input_s3_uri
    assert training.environment["RUN_ID"] == "run-1"
    assert training.environment["EXPERIMENT_ID"] == "run-1-1"
    assert training.environment["DATASET_ID"] == "dataset-1"
    assert training.environment["DATASET_SHA256"] == dataset_sha
    assert training.environment["APPROVED_DATASET_ARTIFACT_ID"] == "dataset://dataset-1"
    assert training.environment["BASE_MODEL_ID"] == config.target_model
    assert training.environment["BASE_MODEL_REVISION"] == config.hf_revision
    assert training.environment["BASE_MODEL_BUNDLE_SHA256"] == base_sha
    assert json.loads(training.environment["QLORA_CONFIG"]) == qlora
    assert evaluation.candidate_s3_uri == f"s3://demo-bucket/checkpoints/{'c' * 64}.tar.gz"
    assert evaluation.champion_s3_uri == f"s3://demo-bucket/checkpoints/{base_sha}.tar.gz"
    assert evaluation.base_model_s3_uri == f"s3://demo-bucket/checkpoints/{base_sha}.tar.gz"
    assert evaluation.sealed_s3_uri == config.evaluation_input_s3_uri
    assert evaluation.environment == {
        "RUN_ID": "run-1",
        "EXPERIMENT_ID": "run-1-1",
        "EVALUATION_MANIFEST_SHA256": "d" * 64,
        "EVALUATION_SUITE_VERSION": "agent-eval-v1",
        "OBJECTIVE_SEED": "7",
        "CANDIDATE_ARCHIVE_SHA256": "c" * 64,
        "CHAMPION_ARCHIVE_SHA256": base_sha,
        "CHAMPION_KIND": "base-model",
        "BASE_MODEL_ID": config.target_model,
        "BASE_MODEL_REVISION": config.hf_revision,
        "BASE_MODEL_BUNDLE_SHA256": base_sha,
    }

def test_live_request_factory_binds_dynamic_identity_outside_static_sealed_bundle() -> None:
    config = _config()
    factory = LiveRequestFactory(config)
    content_sha = "c" * 64
    candidate = SimpleNamespace(
        uri=f"s3://demo-bucket/checkpoints/{content_sha}.tar.gz?versionId=candidate-v1",
        sha256=content_sha,
    )
    requests = []
    for run_id, experiment_number in (("run-1", 1), ("run-2", 3)):
        state = SimpleNamespace(
            run_id=run_id,
            benchmark_manifest_sha256="d" * 64,
            benchmark_version="agent-eval-v1",
            benchmark_seed=7,
            base_checkpoint_uri=f"s3://demo-bucket/checkpoints/{'b' * 64}.tar.gz?versionId=base-v1",
            base_checkpoint_sha256="b" * 64,
            approval_scope={
                "instance_type": config.instance_type,
                "instance_count": config.instance_count,
                "volume_size_gb": config.volume_size_gb,
                "max_runtime_seconds": config.max_runtime_seconds,
            },
        )
        requests.append(
            factory.evaluation(state, experiment_number=experiment_number, candidate=candidate)
        )

    first, second = requests
    assert first.input_s3_uri == second.input_s3_uri == config.evaluation_input_s3_uri
    assert first.sealed_s3_uri == second.sealed_s3_uri == config.evaluation_input_s3_uri
    assert first.environment["EVALUATION_MANIFEST_SHA256"] == "d" * 64
    assert second.environment["EVALUATION_MANIFEST_SHA256"] == "d" * 64
    assert first.base_model_s3_uri == second.base_model_s3_uri == (
        f"s3://demo-bucket/checkpoints/{'b' * 64}.tar.gz"
    )
    assert first.champion_s3_uri == f"s3://demo-bucket/checkpoints/{'b' * 64}.tar.gz"
    assert first.environment["CHAMPION_KIND"] == "base-model"
    assert (first.environment["RUN_ID"], first.environment["EXPERIMENT_ID"]) == (
        "run-1",
        "run-1-1",
    )
    assert (second.environment["RUN_ID"], second.environment["EXPERIMENT_ID"]) == (
        "run-2",
        "run-2-3",
    )


def test_live_request_factory_uses_promoted_adapter_as_later_champion() -> None:
    config = _config()
    base_sha = "b" * 64
    champion_sha = "d" * 64
    state = SimpleNamespace(
        run_id="run-1",
        benchmark_manifest_sha256="e" * 64,
        benchmark_version="agent-eval-v1",
        benchmark_seed=7,
        base_checkpoint_uri=f"s3://demo-bucket/checkpoints/{base_sha}.tar.gz?versionId=base-v1",
        base_checkpoint_sha256=base_sha,
        champion_checkpoint_uri=(
            f"s3://demo-bucket/checkpoints/{champion_sha}.tar.gz?versionId=adapter-v2"
        ),
        champion_checkpoint_sha256=champion_sha,
        approval_scope={
            "instance_type": config.instance_type,
            "instance_count": config.instance_count,
            "volume_size_gb": config.volume_size_gb,
            "max_runtime_seconds": config.max_runtime_seconds,
        },
    )

    request = LiveRequestFactory(config).evaluation(
        state,
        experiment_number=2,
        candidate=SimpleNamespace(
            uri=f"s3://demo-bucket/checkpoints/{'c' * 64}.tar.gz?versionId=candidate-v2",
            sha256="c" * 64,
        ),
    )

    assert request.base_model_s3_uri == f"s3://demo-bucket/checkpoints/{base_sha}.tar.gz"
    assert request.champion_s3_uri == f"s3://demo-bucket/checkpoints/{champion_sha}.tar.gz"
    assert request.environment["BASE_MODEL_BUNDLE_SHA256"] == base_sha
    assert request.environment["CHAMPION_ARCHIVE_SHA256"] == champion_sha
    assert request.environment["CHAMPION_KIND"] == "qlora-adapter"


def test_live_request_factory_blocks_unversioned_or_mismatched_dataset() -> None:
    config = _config()
    state = SimpleNamespace(
        run_id="run-1",
        base_checkpoint_uri="s3://demo-bucket/base.tar.gz?versionId=base-v1",
        base_checkpoint_sha256="b" * 64,
        approval_scope={
            "instance_type": config.instance_type,
            "instance_count": config.instance_count,
            "volume_size_gb": config.volume_size_gb,
            "max_runtime_seconds": config.max_runtime_seconds,
        },
    )
    dataset = SimpleNamespace(
        dataset_id="dataset-1",
        uri=f"s3://demo-bucket/datasets/run-1/1/{'a' * 64}/dataset.jsonl",
        sha256="a" * 64,
        artifact_id="dataset://dataset-1",
    )
    qlora = {"model_dump": lambda **_: {}}

    with pytest.raises(LiveExecutionBlocked, match="versioned content-addressed"):
        LiveRequestFactory(config).training(
            state,
            experiment_number=1,
            dataset=dataset,
            config=SimpleNamespace(**qlora),
        )


def test_environment_config_fails_closed_when_required_inputs_missing() -> None:
    with pytest.raises(LiveExecutionBlocked, match="missing live configuration"):
        config_from_environment({})


def test_environment_config_uses_live_defaults_without_static_training_input() -> None:
    config = config_from_environment(
        {
            "S3_ARTIFACT_BUCKET": "demo-bucket",
            "DYNAMODB_TABLE_NAME": "demo-history",
            "SAGEMAKER_TRAINING_ROLE_ARN": "arn:aws:iam::123456789012:role/train",
            "SAGEMAKER_TRAINING_IMAGE_URI": (
                "123456789012.dkr.ecr.us-east-1.amazonaws.com/train:tag"
            ),
            "SAGEMAKER_EVALUATION_IMAGE_URI": (
                "123456789012.dkr.ecr.us-east-1.amazonaws.com/eval:tag"
            ),
            "OBJECTIVE_WORKER_URL": "https://worker.example.com",
            "HF_REPO_ID": "google/functiongemma-270m-it",
            "HF_REVISION": "a" * 40,
            "EVALUATION_INPUT_S3_URI": "s3://demo-bucket/eval",
            "SAGEMAKER_GPU_QUOTA_CODE": "L-01234567",
            "SAGEMAKER_PROCESSING_GPU_QUOTA_CODE": "L-89ABCDEF",
            "BASELINE_EPISODES": "1",
            "HELD_OUT_EPISODES": "2",
            "LIVE_BENCHMARK_MANIFEST_SHA256": "b" * 64,
        }
    )

    assert config.training_input_s3_uri is None
    assert config.approval_ttl_seconds == 86400
    assert config.objective_worker_timeout_seconds == 600
    assert config.provider_max_polls == 241
    assert config.sagemaker_gpu_quota_code == "L-01234567"
    assert config.sagemaker_processing_gpu_quota_code == "L-89ABCDEF"
    assert config.baseline_episodes == 1
    assert config.held_out_episodes == 2
    assert config.benchmark_manifest_sha256 == "b" * 64


def test_legacy_synchronous_controller_fails_closed_without_static_training_input() -> None:
    controller = _controller(training_input_s3_uri=None)

    with pytest.raises(
        LiveExecutionBlocked,
        match="legacy synchronous execution requires TRAINING_INPUT_S3_URI",
    ):
        controller.run_once(run_number=1, approval_token="")


def test_legacy_synchronous_controller_still_checks_static_training_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _controller()
    input_splits: list[str] = []

    class StopAtReadiness(RuntimeError):
        pass

    monkeypatch.setattr(
        AutonomousRunController,
        "_require_approval",
        lambda self, token, *, run_number: SimpleNamespace(digest="test"),
    )
    monkeypatch.setattr(
        PreflightRunner,
        "_check_input_readiness",
        lambda self, split: input_splits.append(split) or {"status": "available"},
    )
    monkeypatch.setattr(
        PreflightRunner,
        "run",
        lambda self: (_ for _ in ()).throw(StopAtReadiness()),
    )

    with pytest.raises(StopAtReadiness):
        controller.run_once(run_number=1, approval_token="valid-test-token")

    assert input_splits == ["training"]


def test_live_components_use_configured_poll_window_timeout_and_cost_bounds() -> None:
    from app.autonomous.repository import InMemoryAutonomousRunRepository

    class ReasoningModel:
        model_id = "nvidia.nemotron-super-3-120b"

        def invoke(self, prompt: str, *, agent_name: str, system_prompt: str) -> str:
            del prompt, agent_name, system_prompt
            return "{}"

    # A 1-hour maximum job runtime at the declared per-instance rates,
    # multiplied by two requested instances, gives $3 training / $2 evaluation
    # reservations. Across the unchanged five-run window, this remains $25.
    config = _config(
        max_runtime_seconds=3600,
        instance_count=2,
        objective_worker_timeout_seconds=420,
    )
    components = create_autonomous_live_components(
        config,
        repository=InMemoryAutonomousRunRepository(),
        model=ReasoningModel(),
        provider=object(),
        artifact_store=cast(S3ArtifactStore, object()),
    )

    assert components.supervisor.max_polls == 121
    assert components.supervisor.poll_interval_seconds == 30.0
    assert components.supervisor.objective.client.timeout_seconds == 420
    assert config.max_runs == 5
    assert config.max_cost_usd == 25.0
    assert config.estimated_worst_case_cost_usd == 25.0
    assert components.supervisor.phase_cost_upper_bounds_usd == {
        "training": 3.0,
        "evaluation": 2.0,
    }


def test_live_supervisor_cost_bounds_fail_closed_when_missing_or_invalid() -> None:
    from app.autonomous.repository import InMemoryAutonomousRunRepository
    from app.autonomous.supervisor import SupervisorBlocked

    class ReasoningModel:
        model_id = "nvidia.nemotron-super-3-120b"

        def invoke(self, prompt: str, *, agent_name: str, system_prompt: str) -> str:
            del prompt, agent_name, system_prompt
            return "{}"

    components = create_autonomous_live_components(
        _config(),
        repository=InMemoryAutonomousRunRepository(),
        model=ReasoningModel(),
        provider=object(),
        artifact_store=cast(S3ArtifactStore, object()),
    )

    for invalid_bounds in (
        {},
        {"training": 3.0},
        {"training": 0.0, "evaluation": 2.0},
        {"training": 3.0, "evaluation": float("nan")},
    ):
        components.supervisor.phase_cost_upper_bounds_usd = invalid_bounds
        with pytest.raises(SupervisorBlocked, match="cost upper bound"):
            components.supervisor._validate_cost_upper_bounds()

    zero_rate_components = create_autonomous_live_components(
        _config(training_hourly_cost_usd=0.0),
        repository=InMemoryAutonomousRunRepository(),
        model=ReasoningModel(),
        provider=object(),
        artifact_store=cast(S3ArtifactStore, object()),
    )
    assert zero_rate_components.supervisor.phase_cost_upper_bounds_usd == {
        "training": 0.0,
        "evaluation": 1.0,
    }
    with pytest.raises(SupervisorBlocked, match="cost upper bound is invalid for training"):
        zero_rate_components.supervisor._validate_cost_upper_bounds()


def test_preflight_requires_the_configured_approval_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    monkeypatch.delenv(config.approval_secret_env, raising=False)

    check = PreflightRunner(config)._check_approval_secret()

    assert check.status is CheckStatus.BLOCKED
    assert check.classification is PreflightClassification.BLOCKED_CONFIGURATION


def test_gpu_preflight_blocks_instance_outside_explicit_allowlist() -> None:
    config = _config(
        instance_type="ml.g5.2xlarge",
        gpu_instance_allowlist=("ml.g5.xlarge",),
    )

    check = PreflightRunner(config)._check_gpu_readiness()

    assert check.status is CheckStatus.BLOCKED
    assert check.classification is PreflightClassification.BLOCKED_GPU_ALLOWLIST
    assert check.metadata["instance_type"] == "ml.g5.2xlarge"


class _ReadOnlyQuotaClient:
    def __init__(self, value: float | dict[str, float]) -> None:
        self.values = value if isinstance(value, dict) else {}
        self.default_value = float(value) if isinstance(value, (float, int)) else 0.0
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get_service_quota(self, **kwargs: str) -> dict[str, object]:
        self.calls.append(("get_service_quota", kwargs))
        return {
            "Quota": {"Value": self.values.get(kwargs["QuotaCode"], self.default_value)}
        }

    def __getattr__(self, name: str) -> object:
        if name.startswith(("put", "create", "delete", "update", "start", "stop")):
            raise AssertionError(f"preflight attempted mutating quota operation: {name}")
        raise AttributeError(name)


def test_gpu_preflight_blocks_insufficient_quota_without_mutating_aws() -> None:
    quota_client = _ReadOnlyQuotaClient(value=0)
    config = _config(
        sagemaker_gpu_quota_code="L-01234567",
        sagemaker_processing_gpu_quota_code="L-89ABCDEF",
    )

    check = PreflightRunner(config, clients={"service-quotas": quota_client})._check_gpu_readiness()

    assert check.status is CheckStatus.BLOCKED
    assert check.classification is PreflightClassification.BLOCKED_GPU_QUOTA
    assert quota_client.calls == [
        (
            "get_service_quota",
            {"ServiceCode": "sagemaker", "QuotaCode": "L-01234567"},
        ),
        (
            "get_service_quota",
            {"ServiceCode": "sagemaker", "QuotaCode": "L-89ABCDEF"},
        ),
    ]


def test_gpu_preflight_blocks_when_processing_quota_is_zero_but_training_passes() -> None:
    quota_client = _ReadOnlyQuotaClient(
        value={"L-01234567": 1.0, "L-89ABCDEF": 0.0}
    )
    runner = PreflightRunner(
        _config(
            sagemaker_gpu_quota_code="L-01234567",
            sagemaker_processing_gpu_quota_code="L-89ABCDEF",
        ),
        clients={"service-quotas": quota_client},
    )

    check = runner._check_gpu_readiness()

    assert check.status is CheckStatus.BLOCKED
    assert check.classification is PreflightClassification.BLOCKED_GPU_QUOTA
    assert check.metadata["training_quota_value"] == "1.0"
    assert check.metadata["processing_quota_value"] == "0.0"
    assert check.metadata["insufficient_quota_types"] == "processing"
    assert runner._gpu_quota_status is GpuQuotaStatus.INSUFFICIENT


def test_gpu_preflight_blocks_if_either_quota_code_is_missing() -> None:
    quota_client = _ReadOnlyQuotaClient(value=1.0)
    runner = PreflightRunner(
        _config(
            sagemaker_gpu_quota_code="L-01234567",
            sagemaker_processing_gpu_quota_code=None,
        ),
        clients={"service-quotas": quota_client},
    )

    check = runner._check_gpu_readiness()

    assert check.status is CheckStatus.BLOCKED
    assert check.classification is PreflightClassification.BLOCKED_CONFIGURATION
    assert check.metadata["missing_quota_types"] == "processing"
    assert quota_client.calls == []


def test_gpu_preflight_reports_quota_coverage_as_non_mutating_readiness() -> None:
    quota_client = _ReadOnlyQuotaClient(
        value={"L-01234567": 1.0, "L-89ABCDEF": 1.0}
    )
    config = _config(
        sagemaker_gpu_quota_code="L-01234567",
        sagemaker_processing_gpu_quota_code="L-89ABCDEF",
    )

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
        "immutable_model_revision": "c" * 40,
        "max_experiments": 5,
        "max_cost_usd": 25.0,
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
    assert _approval_packet(max_experiments=4).digest != packet.digest
    assert _approval_packet(max_cost_usd=24.0).digest != packet.digest
    assert _approval_packet(immutable_model_revision="d" * 40).digest != packet.digest

    with pytest.raises(LiveExecutionBlocked, match="signature"):
        _decode_approval_token(token, "wrong-secret")

    version, payload, signature = token.split(".")
    tampered = ApprovalPacket.model_validate_json(
        urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
    ).model_copy(update={"run_number": 2})
    tampered_payload = urlsafe_b64encode(tampered.canonical_bytes()).decode("ascii").rstrip("=")
    with pytest.raises(LiveExecutionBlocked, match="signature"):
        _decode_approval_token(f"{version}.{tampered_payload}.{signature}", "demo-approval-secret")


def test_approval_packet_requires_aware_bounded_timestamps() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        _approval_packet(issued_at=datetime.now(), expires_at=datetime.now() + timedelta(minutes=5))
    with pytest.raises(ValidationError, match="future"):
        _approval_packet(
            issued_at=datetime.now(UTC) + timedelta(minutes=1),
            expires_at=datetime.now(UTC) + timedelta(minutes=6),
        )
    with pytest.raises(ValidationError, match="too long"):
        _approval_packet(
            issued_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(days=2),
        )

def _controller(**overrides: object) -> AutonomousRunController:
    return AutonomousRunController(
        config=_config(**overrides),
        objective_worker=object(),
        provider=cast(Any, object()),
        artifact_store=cast(Any, object()),
        slots=cast(Any, object()),
    )


class _LiveBody:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def read(self) -> bytes:
        return self.data


class _LiveArtifactS3:
    def __init__(self, source: bytes, *, source_digest: str | None = None) -> None:
        self.source = source
        self.source_digest = source_digest or hashlib.sha256(source).hexdigest()
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.retained: tuple[bytes, dict[str, str]] | None = None
        self.tamper_retained = False

    def head_object(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("head_object", kwargs))
        if str(kwargs["Key"]).startswith("post-training/checkpoints/"):
            if self.retained is None:
                raise KeyError("retained object not written")
            data, metadata = self.retained
            return {
                "Metadata": metadata,
                "ContentLength": len(data),
                "VersionId": "retained-v1",
                "ContentType": "application/gzip",
            }
        return {
            "Metadata": {"sha256": self.source_digest},
            "ContentLength": len(self.source),
            "VersionId": "source-v1",
            "ContentType": "application/gzip",
        }

    def get_object(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("get_object", kwargs))
        if str(kwargs["Key"]).startswith("post-training/checkpoints/"):
            if self.retained is None:
                raise KeyError("retained object not written")
            data, metadata = self.retained
            if self.tamper_retained:
                data = b"tampered retained bytes"
            return {
                "Body": _LiveBody(data),
                "Metadata": metadata,
                "ContentLength": len(data),
                "VersionId": "retained-v1",
            }
        return {
            "Body": _LiveBody(self.source),
            "Metadata": {"sha256": self.source_digest},
            "ContentLength": len(self.source),
            "VersionId": "source-v1",
        }

    def put_object(self, **kwargs: object) -> dict[str, str]:
        self.calls.append(("put_object", kwargs))
        self.retained = (
            bytes(cast(bytes, kwargs["Body"])),
            dict(cast(Mapping[str, str], kwargs["Metadata"])),
        )
        return {"VersionId": "retained-v1", "ETag": '"retained"'}


def _artifact_controller(client: _LiveArtifactS3) -> AutonomousRunController:
    return AutonomousRunController(
        config=_config(artifact_prefix="post-training"),
        objective_worker=object(),
        provider=cast(Any, object()),
        artifact_store=S3ArtifactStore("demo-bucket", client=client, prefix="post-training"),
        slots=cast(Any, object()),
    )


def _training_artifact_job(uri: str) -> JobResult:
    return JobResult(
        job_name="train-1",
        provider_job_id="arn:aws:sagemaker:us-east-1:123:training-job/train-1",
        status=JobStatus.COMPLETED,
        artifact_uri=uri,
    )


@pytest.mark.parametrize(
    "uri",
    [
        "s3://demo-bucket/post-training/run-1/model.tar.gz?versionId=source-v1&versionId=source-v2",
        "s3://demo-bucket/post-training/run-1/model.tar.gz?versionId=",
        "s3://demo-bucket/post-training/run-1/model.tar.gz?versionId=null",
    ],
)
def test_live_artifact_path_rejects_duplicate_blank_or_null_version_uri(
    uri: str,
) -> None:
    client = _LiveArtifactS3(b"checkpoint bytes")
    controller = _artifact_controller(client)

    with pytest.raises(LiveExecutionFailed, match=r"immutable.*versioned"):
        controller._artifact_from_job(_training_artifact_job(uri), ArtifactKind.CHECKPOINT)
    assert client.calls == []


def test_live_artifact_path_allows_scoped_versionless_output_at_canonicalization_boundary() -> None:
    data = b"checkpoint bytes"
    client = _LiveArtifactS3(data)
    controller = _artifact_controller(client)

    artifact = controller._artifact_from_job(
        _training_artifact_job("s3://demo-bucket/post-training/run-1/model.tar.gz"),
        ArtifactKind.CHECKPOINT,
    )

    assert artifact.uri.startswith("s3://demo-bucket/post-training/checkpoints/")
    assert "versionId=retained-v1" in artifact.uri


def test_live_artifact_path_downloads_exact_bytes_and_returns_retained_reference() -> None:
    data = b"checkpoint bytes"
    client = _LiveArtifactS3(data)
    controller = _artifact_controller(client)

    artifact = controller._artifact_from_job(
        _training_artifact_job(
            "s3://demo-bucket/post-training/run-1/model.tar.gz?versionId=source-v1"
        ),
        ArtifactKind.CHECKPOINT,
    )

    digest = hashlib.sha256(data).hexdigest()
    assert artifact.uri.startswith("s3://demo-bucket/post-training/checkpoints/")
    assert "versionId=retained-v1" in artifact.uri
    assert artifact.sha256 == digest
    assert artifact.size_bytes == len(data)
    assert any(name == "put_object" for name, _ in client.calls)
    get_calls = [kwargs for name, kwargs in client.calls if name == "get_object"]
    assert get_calls
    assert all(call["VersionId"] in {"source-v1", "retained-v1"} for call in get_calls)


def test_live_artifact_path_fails_on_retained_byte_hash_mismatch() -> None:
    client = _LiveArtifactS3(b"checkpoint bytes")
    client.tamper_retained = True
    controller = _artifact_controller(client)

    with pytest.raises(LiveExecutionFailed, match="integrity"):
        controller._artifact_from_job(
            _training_artifact_job(
                "s3://demo-bucket/post-training/run-1/model.tar.gz?versionId=source-v1"
            ),
            ArtifactKind.CHECKPOINT,
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
    assert payload["checkpoint_s3_uri"] == ("s3://demo-bucket/champion/model.tar.gz?versionId=v42")


def test_promoted_champion_must_reference_a_checkpoint_artifact() -> None:
    controller = _controller()
    with pytest.raises(LiveExecutionBlocked, match="not a checkpoint"):
        controller._checkpoint_uri(_champion_record(kind=ArtifactKind.REPORT))


def test_live_benchmark_rejects_result_with_different_manifest() -> None:
    class Worker:
        def execute_benchmark(self, request: ObjectiveBenchmarkRequest) -> ObjectiveBenchmarkResult:
            received = request
            return ObjectiveBenchmarkResult(
                benchmark_id="benchmark-1",
                run_id=received.run_id,
                suite=received.suite,
                suite_version=received.suite_version,
                model_id=received.model_uri,
                model_sha256=received.model_sha256,
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
            model_uri="s3://demo-bucket/base/model.tar.gz?versionId=base-v1",
            model_sha256="b" * 64,
            split="train",
            episodes=2,
            output_s3_uri="s3://demo-bucket/run-1/train",
            manifest_sha256="f" * 64,
        )


def test_live_training_benchmark_rejects_baseline_split_before_worker_call() -> None:
    class Worker:
        def execute_benchmark(self, request: ObjectiveBenchmarkRequest) -> ObjectiveBenchmarkResult:
            raise AssertionError(f"unsupported split reached worker: {request.split}")

    adapter = LiveObjectiveAdapter(Worker(), _config())
    state = SimpleNamespace(
        run_id="run-1",
        base_checkpoint_uri="s3://demo-bucket/base.tar.gz?versionId=base-v1",
    )

    with pytest.raises(LiveExecutionFailed, match="paired evaluator"):
        adapter.benchmark(state, split="baseline", experiment_number=0)


def test_live_supervisor_train_benchmark_uses_baseline_episode_count() -> None:
    class StopAfterRequest(RuntimeError):
        pass

    class Worker:
        request: ObjectiveBenchmarkRequest | None = None

        def execute_benchmark(self, request: ObjectiveBenchmarkRequest) -> ObjectiveBenchmarkResult:
            self.request = request
            raise StopAfterRequest("captured request")

    config = _config(baseline_episodes=10, held_out_episodes=15)
    worker = Worker()
    adapter = LiveObjectiveAdapter(worker, config)
    state = SimpleNamespace(
        run_id="run-1",
        base_checkpoint_uri="s3://demo-bucket/base.tar.gz?versionId=base-v1",
        base_checkpoint_sha256="b" * 64,
    )

    with pytest.raises(StopAfterRequest, match="captured request"):
        adapter.benchmark(state, split="train", experiment_number=0)

    assert worker.request is not None
    assert worker.request.num_episodes == 10


def test_live_supervisor_accepts_provider_unique_benchmark_evidence_id() -> None:
    class Worker:
        def execute_benchmark(self, request: ObjectiveBenchmarkRequest) -> ObjectiveBenchmarkResult:
            reference = TrajectoryReference(
                trajectory_id="train-trajectory-001",
                task_id="train-task-001",
                split=ObjectiveSplit.TRAIN,
                verified=True,
            )
            report = ArtifactReference(
                artifact_id="report-001",
                kind=ArtifactKind.REPORT,
                uri="s3://demo-bucket/report.json?versionId=report-v1",
                sha256="c" * 64,
            )
            return ObjectiveBenchmarkResult(
                benchmark_id="benchmark-7c883320d4dd",
                run_id=request.run_id,
                suite=request.suite,
                suite_version=request.suite_version,
                model_id=request.model_uri,
                model_sha256=request.model_sha256,
                seed=request.seed,
                split=request.split,
                metrics=BenchmarkMetrics(
                    aggregate=0.75,
                    per_environment={"service-recovery-v1": 0.75},
                ),
                trajectory_references=(reference,),
                report_artifact=report,
                manifest_sha256="d" * 64,
                evidence_label=EvidenceLabel.LIVE,
                verified=True,
            )

    adapter = LiveObjectiveAdapter(Worker(), _config())
    state = SimpleNamespace(
        run_id="run-1",
        base_checkpoint_uri="s3://demo-bucket/base.tar.gz?versionId=base-v1",
        base_checkpoint_sha256="b" * 64,
    )

    evidence = adapter.benchmark(state, split="train", experiment_number=0)

    assert evidence.evaluation.evidence.evidence_id == "benchmark-7c883320d4dd"
    assert evidence.evaluation.evidence.benchmark_id == "service-recovery-v1"
    assert evidence.evaluation.evidence.model_id == "google/functiongemma-270m-it"
    assert evidence.evaluation.run_number == 0
    assert evidence.artifact_ids == ("s3://demo-bucket/report.json?versionId=report-v1",)


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


def test_live_objective_handoff_preserves_references_across_adapter_restart() -> None:
    engine = ServiceRecoveryEngine(seed=7)
    task_id = "train-task-001"
    definition = engine._make_definition(task_id, ObjectiveSplit.TRAIN)
    if definition.failure_mode == "config_error":
        success_actions = [
            ToolCall(tool="read_config", arguments={}),
            ToolCall(
                tool="edit_config",
                arguments={"service": definition.service_name, "content": "fixed"},
            ),
            ToolCall(tool="run_healthcheck", arguments={"service": definition.service_name}),
        ]
    else:
        success_actions = [
            ToolCall(tool="restart_service", arguments={"service": definition.service_name}),
            ToolCall(tool="run_healthcheck", arguments={"service": definition.service_name}),
        ]
    trajectory = engine.verify(
        engine.run_episode(
            task_id,
            success_actions,
            split=ObjectiveSplit.TRAIN,
        )
    ).trajectory
    assert trajectory.success is True
    assert trajectory.verifier_success is True
    artifact_store = InMemoryTrajectoryArtifactStore()
    trusted_reference = artifact_store.put(trajectory)
    service = ObjectiveService(engine, "secret", artifact_store=artifact_store)
    expected_opaque = "trajectory://train/" + trajectory.trajectory_id + "/train-task-001/verified"

    class Worker:
        curation_references: tuple[Mapping[str, Any], ...] = ()

        def execute_benchmark(
            self, request: ObjectiveBenchmarkRequest
        ) -> ObjectiveBenchmarkResult:
            return ObjectiveBenchmarkResult(
                benchmark_id="service-recovery-v1",
                run_id=request.run_id,
                suite=request.suite,
                suite_version=request.suite_version,
                model_id=request.model_uri,
                model_sha256=request.model_sha256,
                seed=request.seed,
                split=request.split,
                metrics=BenchmarkMetrics(aggregate=0.0, per_environment={"api": 0.0}),
                trajectory_artifact=ArtifactReference(
                    artifact_id="trajectory-bundle-1",
                    kind=ArtifactKind.TRAJECTORY,
                    uri="s3://demo-bucket/trajectory-bundle.tar.gz?versionId=v1",
                    sha256="a" * 64,
                ),
                report_artifact=ArtifactReference(
                    artifact_id="benchmark-report-1",
                    kind=ArtifactKind.REPORT,
                    uri="s3://demo-bucket/benchmark-report.json?versionId=v1",
                    sha256="b" * 64,
                ),
                manifest_sha256="c" * 64,
                evidence_label=EvidenceLabel.LIVE,
                verified=True,
                trajectory_references=(trusted_reference,),
            )

        def verify_curation(
            self,
            *,
            run_id: str,
            experiment_id: str,
            trajectory_references: tuple[TrajectoryReference, ...],
        ) -> Mapping[str, Any]:
            serialized = tuple(
                reference.model_dump(mode="json") for reference in trajectory_references
            )
            self.curation_references = serialized
            request = CurationRequest(
                run_id=run_id,
                experiment_id=experiment_id,
                split=trajectory_references[0].split,
                trajectory_references=serialized,
            )
            return service.verify_curation(request).model_dump(mode="json")

    worker = Worker()
    state = SimpleNamespace(
        run_id="run-1",
        champion_checkpoint_uri=None,
        base_checkpoint_uri="s3://demo-bucket/base.tar.gz?versionId=base-v1",
        base_checkpoint_sha256="c" * 64,
        metadata={},
    )
    original = LiveObjectiveAdapter(worker, _config())

    benchmark = original.benchmark(state, split="train", experiment_number=1)

    assert benchmark.trajectory_refs == (expected_opaque,)
    restored_adapter = LiveObjectiveAdapter(worker, _config())
    dataset = restored_adapter.build_dataset(
        state,
        SimpleNamespace(selected_trajectory_refs=(expected_opaque,)),
        experiment_number=1,
    )
    assert worker.curation_references == (
        {
            "trajectory_id": trajectory.trajectory_id,
            "task_id": "train-task-001",
            "split": "train",
            "verified": True,
        },
    )
    assert dataset.dataset_id.startswith("dataset-")
    assert dataset.sha256


def test_objective_worker_curation_sends_actual_split_and_authenticated_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = TrajectoryReference(
        trajectory_id="traj-real-001",
        task_id="train-task-001",
        split=ObjectiveSplit.TRAIN,
        verified=True,
    )
    calls: list[tuple[str, dict[str, Any]]] = []

    def request(url: str, **kwargs: Any) -> Mapping[str, Any]:
        calls.append((url, kwargs))
        return {"manifest": {"dataset_id": "dataset-001"}}

    monkeypatch.setattr("app.live_execution._http_json", request)
    worker = ObjectiveWorkerClient("https://worker.example.com", auth_token="worker-secret")

    worker.verify_curation(
        run_id="run-1",
        experiment_id="run-1-1",
        trajectory_references=(reference,),
    )

    assert calls[0][0] == "https://worker.example.com/v1/verify-curation"
    assert calls[0][1]["headers"] == {"Authorization": "Bearer worker-secret"}
    assert calls[0][1]["payload"] == {
        "run_id": "run-1",
        "experiment_id": "run-1-1",
        "split": "train",
        "trajectory_references": [
            {
                "trajectory_id": "traj-real-001",
                "task_id": "train-task-001",
                "split": "train",
                "verified": True,
            }
        ],
    }


def _evaluation_archive(report: Mapping[str, Any]) -> bytes:
    payload = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        item = tarfile.TarInfo("evaluation.json")
        item.size = len(payload)
        archive.addfile(item, io.BytesIO(payload))
    return buffer.getvalue()


def _evaluation_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "evaluation-report-v1",
        "suite": "AgentGym/AgentEval",
        "suite_version": "agent-eval-v1",
        "evaluation_manifest_sha256": "a" * 64,
        "run_id": "run-1",
        "experiment_id": "run-1-1",
        "candidate_manifest_sha256": "b" * 64,
        "candidate_artifact_sha256": "d" * 64,
        "candidate_task_count": 2,
        "candidate_successful_tasks": 1,
        "candidate_success_rate": 0.5,
        "candidate_invalid_action_tasks": 0,
        "candidate_metrics": {
            "task_count": 2,
            "successful_tasks": 1,
            "success_rate": 0.5,
            "invalid_action_tasks": 0,
            "by_environment": {
                "api": {"task_count": 2, "successful_tasks": 1, "success_rate": 0.5}
            },
        },
        "champion_manifest_sha256": "c" * 64,
        "champion_artifact_sha256": "e" * 64,
        "champion_task_count": 2,
        "champion_successful_tasks": 0,
        "champion_success_rate": 0.0,
        "champion_invalid_action_tasks": 0,
        "champion_metrics": {
            "task_count": 2,
            "successful_tasks": 0,
            "success_rate": 0.0,
            "invalid_action_tasks": 0,
            "by_environment": {
                "api": {"task_count": 2, "successful_tasks": 0, "success_rate": 0.0}
            },
        },
        "candidate_task_successes": [True, False],
        "champion_task_successes": [False, False],
        "candidate_task_environments": ["api", "api"],
        "champion_task_environments": ["api", "api"],
        "regression_count": 0,
        "improvement_count": 1,
        "unchanged_count": 1,
        "regression_decision": "IMPROVED",
    }
    report["paired_outcomes_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "candidate": report["candidate_task_successes"],
                "champion": report["champion_task_successes"],
                "environments": report["candidate_task_environments"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    report["regression_evidence_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "candidate_manifest_sha256": report["candidate_manifest_sha256"],
                "champion_manifest_sha256": report["champion_manifest_sha256"],
                "candidate_artifact_sha256": report["candidate_artifact_sha256"],
                "champion_artifact_sha256": report["champion_artifact_sha256"],
                "task_count": 2,
                "regression_count": 0,
                "improvement_count": 1,
                "unchanged_count": 1,
                "candidate_task_successes": report["candidate_task_successes"],
                "champion_task_successes": report["champion_task_successes"],
                "candidate_task_environments": report["candidate_task_environments"],
                "champion_task_environments": report["champion_task_environments"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    report["report_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return report


def test_live_evaluation_reader_verifies_completed_sagemaker_report_artifact() -> None:
    from app.providers.artifacts import ArtifactRef

    report_bytes = _evaluation_archive(_evaluation_report())
    artifact_digest = hashlib.sha256(report_bytes).hexdigest()

    class Store:
        call: dict[str, Any]

        def __init__(self) -> None:
            self.call = {}

        def canonicalize_sagemaker_processing_output(
            self, uri: str, **kwargs: Any
        ) -> ArtifactRef:
            self.call = {"uri": uri, **kwargs}
            return ArtifactRef(
                bucket="demo-bucket",
                key="post-training/run-1/evaluations/" + artifact_digest + ".tar.gz",
                sha256=artifact_digest,
                size_bytes=len(report_bytes),
                version_id="retained-v1",
                content_type="application/gzip",
            )

        def get_bytes(self, reference: ArtifactRef) -> bytes:
            assert reference.version_id == "retained-v1"
            assert reference.sha256 == artifact_digest
            return report_bytes

    store = Store()
    state = SimpleNamespace(
        run_id="run-1",
        current_candidate_uri="s3://demo-bucket/post-training/run-1/checkpoints/candidate.tar.gz?versionId=checkpoint-v1",
        current_candidate_sha256="d" * 64,
        base_checkpoint_sha256="e" * 64,
        benchmark_id="service-recovery-v1",
        benchmark_manifest_sha256="a" * 64,
        benchmark_suite="AgentGym/AgentEval",
        benchmark_version="agent-eval-v1",
        benchmark_seed=7,
        model_id="google/functiongemma-270m-it",
    )
    job = JobResult(
        job_name="eval-run-1-1",
        provider_job_id="arn:aws:sagemaker:us-east-1:123:processing-job/eval-run-1-1",
        status=JobStatus.COMPLETED,
        artifact_uri="s3://demo-bucket/post-training/run-1/eval/1",
    )

    evidence = LiveEvaluationReader(store, _config()).read_evaluation(
        job, state=state, experiment_number=1
    )

    assert store.call["uri"] == job.artifact_uri
    assert store.call["allowed_source_prefix"] == "post-training/run-1/eval/1"
    assert evidence.evaluation.aggregate_score == 0.5
    assert evidence.evaluation.environment_scores == {"api": 0.5}
    assert evidence.evaluation.evidence.kind.value == "evaluation"
    assert evidence.champion_evaluation is not None
    assert evidence.champion_evaluation.aggregate_score == 0.0
    assert evidence.evaluation.champion_run_id == evidence.champion_evaluation.run_id
    assert evidence.artifact_ids == (f"evaluation-report://{artifact_digest}",)


def test_live_evaluation_reader_accepts_plain_evaluation_json_output() -> None:
    from app.providers.artifacts import ArtifactRef

    report_bytes = json.dumps(
        _evaluation_report(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    artifact_digest = hashlib.sha256(report_bytes).hexdigest()

    class Store:
        def canonicalize_sagemaker_processing_output(
            self, uri: str, **kwargs: Any
        ) -> ArtifactRef:
            assert uri == "s3://demo-bucket/post-training/run-1/eval/1"
            assert kwargs["allowed_source_prefix"] == "post-training/run-1/eval/1"
            return ArtifactRef(
                bucket="demo-bucket",
                key="post-training/run-1/evaluations/" + artifact_digest + ".json",
                sha256=artifact_digest,
                size_bytes=len(report_bytes),
                version_id="retained-v1",
                content_type="application/json",
            )

        def get_bytes(self, reference: ArtifactRef) -> bytes:
            assert reference.version_id == "retained-v1"
            assert reference.sha256 == artifact_digest
            return report_bytes

    state = SimpleNamespace(
        run_id="run-1",
        current_candidate_uri="s3://demo-bucket/post-training/run-1/checkpoints/candidate.tar.gz?versionId=checkpoint-v1",
        current_candidate_sha256="d" * 64,
        base_checkpoint_sha256="e" * 64,
        benchmark_id="service-recovery-v1",
        benchmark_manifest_sha256="a" * 64,
        benchmark_suite="AgentGym/AgentEval",
        benchmark_version="agent-eval-v1",
        benchmark_seed=7,
        model_id="google/functiongemma-270m-it",
    )
    job = JobResult(
        job_name="eval-run-1-1",
        provider_job_id="arn:aws:sagemaker:us-east-1:123:processing-job/eval-run-1-1",
        status=JobStatus.COMPLETED,
        artifact_uri="s3://demo-bucket/post-training/run-1/eval/1",
    )

    evidence = LiveEvaluationReader(Store(), _config()).read_evaluation(
        job, state=state, experiment_number=1
    )

    assert evidence.evaluation.aggregate_score == 0.5
    assert evidence.champion_evaluation is not None
    assert evidence.artifact_ids == (f"evaluation-report://{artifact_digest}",)


@pytest.mark.parametrize("mutation", ["report_hash", "run_id", "manifest", "champion", "paired"])
def test_live_evaluation_reader_rejects_unverified_or_wrong_scope_report(
    mutation: str,
) -> None:
    from app.providers.artifacts import ArtifactRef

    report = _evaluation_report()
    if mutation == "report_hash":
        report["report_sha256"] = "0" * 64
    elif mutation == "run_id":
        report["run_id"] = "another-run"
    elif mutation == "manifest":
        report["evaluation_manifest_sha256"] = "e" * 64
    elif mutation == "champion":
        report["champion_artifact_sha256"] = "f" * 64
    else:
        report["paired_outcomes_sha256"] = "0" * 64
    report_bytes = _evaluation_archive(report)
    artifact_digest = hashlib.sha256(report_bytes).hexdigest()

    class Store:
        def canonicalize_sagemaker_processing_output(
            self, uri: str, **kwargs: Any
        ) -> ArtifactRef:
            del uri, kwargs
            return ArtifactRef(
                bucket="demo-bucket",
                key="post-training/evaluation.tar.gz",
                sha256=artifact_digest,
                size_bytes=len(report_bytes),
                version_id="retained-v1",
            )

        def get_bytes(self, reference: ArtifactRef) -> bytes:
            del reference
            return report_bytes

    state = SimpleNamespace(
        run_id="run-1",
        current_candidate_uri="s3://demo-bucket/checkpoint.tar.gz?versionId=v1",
        current_candidate_sha256="d" * 64,
        base_checkpoint_sha256="e" * 64,
        benchmark_id="service-recovery-v1",
        benchmark_manifest_sha256="a" * 64,
        benchmark_suite="AgentGym/AgentEval",
        benchmark_version="agent-eval-v1",
        benchmark_seed=7,
        model_id="google/functiongemma-270m-it",
    )
    job = JobResult(
        job_name="eval-run-1-1",
        provider_job_id="arn:aws:sagemaker:us-east-1:123:processing-job/eval-run-1-1",
        status=JobStatus.COMPLETED,
        artifact_uri="s3://demo-bucket/post-training/run-1/eval/1",
    )

    with pytest.raises(LiveExecutionFailed):
        LiveEvaluationReader(Store(), _config()).read_evaluation(
            job, state=state, experiment_number=1
        )
