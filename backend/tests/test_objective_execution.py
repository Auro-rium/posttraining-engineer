from __future__ import annotations

import hashlib
import io
import json
import struct
import tarfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from app.main import _create_objective_application
from app.objective.engine import ServiceRecoveryEngine
from app.objective.execution import (
    FunctionGemmaBenchmarkExecutionAdapter,
    FunctionGemmaLocalPolicy,
    ObjectiveExecutionUnavailable,
    _DeferredLocalPolicy,
    _messages,
    _parse_function_call,
    _tool_schemas,
    build_benchmark_execution_adapter,
    checkpoint_snapshot_sha256,
)
from app.objective.models import (
    BenchmarkRequest,
    ObjectiveReadinessResponse,
    ObjectiveSplit,
    Task,
    ToolCall,
)

BASE_MODEL_URI = "s3://bucket/checkpoints/base.tar.gz?versionId=base-v1"
BASE_MODEL_SHA256 = "f" * 64
MODEL_REVISION = "a" * 40


def _champion_archive(tmp_path: Path, *, run_id: str = "run-champion") -> tuple[bytes, str]:
    source = tmp_path / "adapter-source"
    source.mkdir()
    qlora_config = {
        "rank": 8,
        "alpha": 16,
        "dropout": 0.0,
        "learning_rate": 1e-4,
        "epochs": 1,
        "sequence_length": 512,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
    (source / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "google/functiongemma-270m-it",
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "target_modules": qlora_config["target_modules"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (source / "adapter_model.safetensors").write_bytes(b"approved-lora-weights")
    (source / "training_metrics.json").write_text('{"train_loss":0.25}\n', encoding="utf-8")
    artifact_files = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(source.iterdir())
    ]
    artifact_sha256 = hashlib.sha256(
        json.dumps(artifact_files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest: dict[str, Any] = {
        "schema_version": "trainer-manifest-v1",
        "kind": "qlora-adapter",
        "run_id": run_id,
        "experiment_id": f"{run_id}-1",
        "dataset_id": "dataset-approved",
        "dataset_sha256": "d" * 64,
        "base_model_id": "google/functiongemma-270m-it",
        "base_model_revision": MODEL_REVISION,
        "artifact_id": f"checkpoint://{artifact_sha256}",
        "qlora_config": qlora_config,
        "artifact_files": artifact_files,
        "artifact_sha256": artifact_sha256,
        "training_metrics": {"train_loss": 0.25},
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (source / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for path in sorted(source.iterdir()):
            tar.add(path, arcname=path.name)
    payload = archive.getvalue()
    return payload, hashlib.sha256(payload).hexdigest()


class _RulePolicy:
    """Deterministic test double for the target-model policy boundary only."""

    def __init__(self) -> None:
        self.received: list[tuple[Task, tuple[dict[str, Any], ...]]] = []

    def __call__(self, task: Task, observations: tuple[dict[str, Any], ...]) -> ToolCall:
        self.received.append((task, observations))
        assert "failure_mode" not in task.model_dump()
        assert all(
            not {"reward", "done", "failure_mode"} & observation.keys()
            for observation in observations
        )
        service = task.service_name
        if not observations:
            return ToolCall(tool="read_config", arguments={"service": service})
        content = str(observations[-1].get("content", ""))
        if "broken_value_should_be_fixed" in content:
            return ToolCall(
                tool="edit_config",
                arguments={"service": service, "content": "setting2=value2"},
            )
        if "dependency=unavailable" in content or "healthcheck=failed" in content:
            return ToolCall(tool="restart_service", arguments={"service": service})
        return ToolCall(tool="run_healthcheck", arguments={"service": service})


def test_adapter_executes_model_actions_and_returns_full_verifier_confirmed_trajectories() -> None:
    policy = _RulePolicy()
    adapter = FunctionGemmaBenchmarkExecutionAdapter(
        policy,
        base_model_uri=BASE_MODEL_URI,
        base_model_sha256=BASE_MODEL_SHA256,
    )
    request = BenchmarkRequest(
        run_id="run-live-test",
        model_uri=BASE_MODEL_URI,
        model_sha256=BASE_MODEL_SHA256,
        split=ObjectiveSplit.TRAIN,
        task_ids=("train-001", "train-002"),
    )

    result = adapter.execute_benchmark(request, ServiceRecoveryEngine(seed=12))

    assert tuple(item.task_id for item in result.trajectories) == request.task_ids
    assert all(item.verified and item.done for item in result.trajectories)
    assert all(item.split is ObjectiveSplit.TRAIN for item in result.trajectories)
    assert all(item.steps for item in result.trajectories)
    assert all(step.reward in {0, 1} for item in result.trajectories for step in item.steps)
    assert any(observations for _, observations in policy.received)
    assert all(
        not {"reward", "done", "failure_mode"} & observation.keys()
        for _, observations in policy.received
        for observation in observations
    )


def test_adapter_fails_closed_when_policy_returns_an_untyped_action() -> None:
    class InvalidPolicy:
        def __call__(self, task: Task, observations: tuple[dict[str, Any], ...]) -> ToolCall:
            del task, observations
            return {"tool": "shell", "arguments": {}}  # type: ignore[return-value]

    adapter = FunctionGemmaBenchmarkExecutionAdapter(
        InvalidPolicy(),
        base_model_uri=BASE_MODEL_URI,
        base_model_sha256=BASE_MODEL_SHA256,
    )
    request = BenchmarkRequest(
        run_id="run-invalid",
        model_uri=BASE_MODEL_URI,
        model_sha256=BASE_MODEL_SHA256,
        task_ids=("train-001",),
    )

    with pytest.raises(ObjectiveExecutionUnavailable, match="invalid tool call"):
        adapter.execute_benchmark(request, ServiceRecoveryEngine(seed=1))


def test_environment_factory_does_not_select_a_mutable_or_simulated_policy() -> None:
    adapter = build_benchmark_execution_adapter(
        {
            "OBJECTIVE_MODEL_CHECKPOINT_DIR": "",
            "OBJECTIVE_MODEL_REVISION": "main",
            "OBJECTIVE_MODEL_SHA256": "",
        }
    )

    with pytest.raises(ObjectiveExecutionUnavailable, match="immutable local FunctionGemma"):
        adapter.execute_benchmark(
            BenchmarkRequest(
                run_id="run-blocked",
                model_uri=BASE_MODEL_URI,
                model_sha256=BASE_MODEL_SHA256,
                task_ids=("train-001",),
            ),
            ServiceRecoveryEngine(seed=1),
        )


@pytest.mark.parametrize(
    ("model_uri", "model_sha256"),
    [
        ("s3://bucket/checkpoints/candidate.tar.gz?versionId=candidate-v1", "e" * 64),
        (BASE_MODEL_URI, "e" * 64),
    ],
)
def test_adapter_rejects_non_base_or_unpinned_model_identity_before_inference(
    model_uri: str, model_sha256: str
) -> None:
    policy = _RulePolicy()
    adapter = FunctionGemmaBenchmarkExecutionAdapter(
        policy,
        base_model_uri=BASE_MODEL_URI,
        base_model_sha256=BASE_MODEL_SHA256,
    )

    with pytest.raises(ObjectiveExecutionUnavailable, match="cannot be materialized"):
        adapter.execute_benchmark(
            BenchmarkRequest(
                run_id="run-champion-v2",
                model_uri=model_uri,
                model_sha256=model_sha256,
                task_ids=("train-001",),
            ),
            ServiceRecoveryEngine(seed=1),
        )

    assert policy.received == []


def test_second_benchmark_uses_verified_champion_adapter_from_its_exact_s3_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.objective.execution as execution

    run_id = "run-champion"
    archive, archive_sha256 = _champion_archive(tmp_path, run_id=run_id)
    champion_uri = f"s3://bucket/checkpoints/{archive_sha256}.tar.gz?versionId=champion-v1"

    class _S3:
        def __init__(self) -> None:
            self.calls: list[dict[str, str]] = []

        def get_object(self, **kwargs: str) -> dict[str, Any]:
            self.calls.append(kwargs)
            return {
                "Body": io.BytesIO(archive),
                "VersionId": "champion-v1",
                "ContentLength": len(archive),
            }

    snapshot = _snapshot(tmp_path / "base-snapshot")
    from scripts.stage_functiongemma_checkpoint import validate_checkpoint_directory

    base_digest = checkpoint_snapshot_sha256(
        validate_checkpoint_directory(snapshot, revision=MODEL_REVISION)
    )
    base_policy = _DeferredLocalPolicy(snapshot, MODEL_REVISION, base_digest)
    base_rule = _RulePolicy()
    base_policy._policy = base_rule
    champion_rule = _RulePolicy()
    loaded: list[dict[str, Any]] = []

    def _load_with_adapter(
        cls: type[FunctionGemmaLocalPolicy],
        checkpoint_dir: str | Path,
        *,
        revision: str,
        expected_sha256: str,
        adapter_dir: str | Path | None = None,
    ) -> _RulePolicy:
        del cls
        loaded.append(
            {
                "checkpoint_dir": Path(checkpoint_dir),
                "revision": revision,
                "expected_sha256": expected_sha256,
                "adapter_dir": Path(adapter_dir) if adapter_dir is not None else None,
            }
        )
        return champion_rule

    monkeypatch.setattr(
        execution.FunctionGemmaLocalPolicy,
        "from_checkpoint",
        classmethod(_load_with_adapter),
    )
    s3 = _S3()
    adapter = FunctionGemmaBenchmarkExecutionAdapter(
        base_policy,
        base_model_uri=BASE_MODEL_URI,
        base_model_sha256=BASE_MODEL_SHA256,
        s3_client=s3,
    )

    base_request = BenchmarkRequest(
        run_id=run_id,
        model_uri=BASE_MODEL_URI,
        model_sha256=BASE_MODEL_SHA256,
        task_ids=("train-001",),
    )
    champion_request = BenchmarkRequest(
        run_id=run_id,
        model_uri=champion_uri,
        model_sha256=archive_sha256,
        task_ids=("train-002",),
    )
    engine = ServiceRecoveryEngine(seed=12)

    base_result = adapter.execute_benchmark(base_request, engine)
    champion_result = adapter.execute_benchmark(champion_request, engine)

    assert base_result.trajectories[0].verified
    assert champion_result.trajectories[0].verified
    assert len(base_rule.received) > 0
    assert len(champion_rule.received) > 0
    assert loaded == [
        {
            "checkpoint_dir": snapshot,
            "revision": MODEL_REVISION,
            "expected_sha256": base_digest,
            "adapter_dir": loaded[0]["adapter_dir"],
        }
    ]
    assert loaded[0]["adapter_dir"] is not None
    assert not loaded[0]["adapter_dir"].exists()
    assert s3.calls == [
        {
            "Bucket": "bucket",
            "Key": f"checkpoints/{archive_sha256}.tar.gz",
            "VersionId": "champion-v1",
        }
    ]


def test_champion_materialization_rejects_s3_version_mismatch(tmp_path: Path) -> None:
    archive, archive_sha256 = _champion_archive(tmp_path)

    class _S3:
        def get_object(self, **kwargs: str) -> dict[str, Any]:
            assert kwargs["VersionId"] == "champion-v1"
            return {"Body": io.BytesIO(archive), "VersionId": "other-version"}

    adapter = FunctionGemmaBenchmarkExecutionAdapter(
        _DeferredLocalPolicy(Path("/unused"), MODEL_REVISION, "e" * 64),
        base_model_uri=BASE_MODEL_URI,
        base_model_sha256=BASE_MODEL_SHA256,
        s3_client=_S3(),
    )
    with pytest.raises(ObjectiveExecutionUnavailable, match="requested champion object version"):
        adapter.execute_benchmark(
            BenchmarkRequest(
                run_id="run-champion",
                model_uri=(
                    f"s3://bucket/checkpoints/{archive_sha256}.tar.gz?versionId=champion-v1"
                ),
                model_sha256=archive_sha256,
                task_ids=("train-001",),
            ),
            ServiceRecoveryEngine(seed=1),
        )


def test_champion_materialization_rejects_archive_hash_mismatch(tmp_path: Path) -> None:
    archive, archive_sha256 = _champion_archive(tmp_path)

    class _S3:
        def get_object(self, **kwargs: str) -> dict[str, Any]:
            assert kwargs["VersionId"] == "champion-v1"
            tampered = archive + b"tampered"
            return {
                "Body": io.BytesIO(tampered),
                "VersionId": "champion-v1",
                "ContentLength": len(tampered),
            }

    adapter = FunctionGemmaBenchmarkExecutionAdapter(
        _DeferredLocalPolicy(Path("/unused"), MODEL_REVISION, "e" * 64),
        base_model_uri=BASE_MODEL_URI,
        base_model_sha256=BASE_MODEL_SHA256,
        s3_client=_S3(),
    )
    with pytest.raises(ObjectiveExecutionUnavailable, match="SHA-256 does not match"):
        adapter.execute_benchmark(
            BenchmarkRequest(
                run_id="run-champion",
                model_uri=(
                    f"s3://bucket/checkpoints/{archive_sha256}.tar.gz?versionId=champion-v1"
                ),
                model_sha256=archive_sha256,
                task_ids=("train-001",),
            ),
            ServiceRecoveryEngine(seed=1),
        )


def test_champion_materialization_rejects_cross_run_provenance(tmp_path: Path) -> None:
    archive, archive_sha256 = _champion_archive(tmp_path, run_id="another-run")

    class _S3:
        def get_object(self, **kwargs: str) -> dict[str, Any]:
            return {
                "Body": io.BytesIO(archive),
                "VersionId": kwargs["VersionId"],
                "ContentLength": len(archive),
            }

    adapter = FunctionGemmaBenchmarkExecutionAdapter(
        _DeferredLocalPolicy(Path("/unused"), MODEL_REVISION, "e" * 64),
        base_model_uri=BASE_MODEL_URI,
        base_model_sha256=BASE_MODEL_SHA256,
        s3_client=_S3(),
    )
    with pytest.raises(ObjectiveExecutionUnavailable, match="provenance does not match"):
        adapter.execute_benchmark(
            BenchmarkRequest(
                run_id="run-champion",
                model_uri=(
                    f"s3://bucket/checkpoints/{archive_sha256}.tar.gz?versionId=champion-v1"
                ),
                model_sha256=archive_sha256,
                task_ids=("train-001",),
            ),
            ServiceRecoveryEngine(seed=1),
        )


def test_benchmark_request_rejects_unversioned_model_uri() -> None:
    with pytest.raises(ValidationError, match="immutable S3 object version"):
        BenchmarkRequest(
            run_id="run-unpinned",
            model_uri="s3://bucket/checkpoints/base.tar.gz",
            model_sha256=BASE_MODEL_SHA256,
        )


@pytest.mark.parametrize(
    "output",
    [
        "before<start_function_call>call:read_config{service:<escape>api<escape>}"
        "<end_function_call>",
        "<start_function_call>call:read_config{service:<escape>api<escape>}"
        "<end_function_call>after",
        "<start_function_call>call:read_config{service:<escape>api<escape>}"
        "<end_function_call><start_function_call>call:restart_service{service:<escape>api<escape>}"
        "<end_function_call>",
        "<start_function_call>call:read_config{service:<escape>api<escape>,"
        "service:<escape>db<escape>}<end_function_call>",
        "<start_function_call>call:read_config{service:<escape>api<escape>,"
        "unexpected:<escape>value<escape>}<end_function_call>",
        "<start_function_call>call:read_config{service:api}<end_function_call>",
    ],
)
def test_functiongemma_protocol_rejects_extra_malformed_or_ambiguous_output(output: str) -> None:
    with pytest.raises(ObjectiveExecutionUnavailable):
        _parse_function_call(output)


def test_functiongemma_tool_schemas_are_closed_and_match_engine_arguments() -> None:
    schemas = _tool_schemas()
    by_name = {item["function"]["name"]: item["function"]["parameters"] for item in schemas}

    assert set(by_name) == {
        "get_logs",
        "inspect_service",
        "read_config",
        "edit_config",
        "restart_service",
        "run_healthcheck",
    }
    assert all(item["additionalProperties"] is False for item in by_name.values())
    assert all(
        item["required"] == ["service"] for name, item in by_name.items() if name != "edit_config"
    )
    assert by_name["edit_config"]["required"] == ["service", "content"]
    assert by_name["edit_config"]["properties"]["content"]["type"] == "string"


def test_functiongemma_prompt_explicitly_activates_function_calling_mode() -> None:
    task = ServiceRecoveryEngine(seed=7).reset(
        split=ObjectiveSplit.TRAIN,
        task_id="train-prompt-contract",
    )

    messages = _messages(task, ())

    assert messages[0]["role"] == "developer"
    assert (
        "You are a model that can do function calling with the following functions"
        in messages[0]["content"]
    )
    assert "one call at a time" in messages[0]["content"]


def test_function_parse_failure_telemetry_uses_safe_categorical_code(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    from app.objective.execution import _run_objective_stage, objective_stage_trace

    completion = "private-model-output-without-a-call"
    caplog.set_level(logging.INFO, logger="app.objective.execution")
    with objective_stage_trace("c" * 32, MODEL_REVISION):
        with pytest.raises(ObjectiveExecutionUnavailable):
            _run_objective_stage("FUNCTION_PARSE", lambda: _parse_function_call(completion))

    event = json.loads(caplog.records[-1].message)
    assert event["stage"] == "FUNCTION_PARSE"
    assert event["status"] == "failed"
    assert event["failure_code"] == "missing_function_call_frame"
    assert completion not in caplog.text


def test_objective_application_passes_explicit_model_identity_to_execution_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.main as main

    marker_store = object()
    marker_adapter = object()
    received: dict[str, Any] = {}

    class _StoreFactory:
        def __new__(cls, bucket: str, *, prefix: str) -> object:
            del cls
            received["store_config"] = (bucket, prefix)
            return marker_store

    def _build_adapter(config: dict[str, str]) -> object:
        received["model_config"] = config
        return marker_adapter

    def _create_app(engine: object, **kwargs: Any) -> object:
        received["app_config"] = (engine, kwargs)
        return marker_store

    monkeypatch.setattr(main, "S3TrajectoryArtifactStore", _StoreFactory)
    monkeypatch.setattr(main, "build_benchmark_execution_adapter", _build_adapter)
    monkeypatch.setattr(main, "create_objective_app", _create_app)

    result = main._create_objective_application(
        SimpleNamespace(
            s3_artifact_bucket="objective-artifacts",
            s3_artifact_prefix="objective-v2",
            objective_auth_token="secret",
            objective_model_checkpoint_dir="/opt/models/functiongemma",
            objective_model_revision="a" * 40,
            objective_model_sha256="b" * 64,
            objective_base_model_uri=BASE_MODEL_URI,
            objective_base_model_sha256=BASE_MODEL_SHA256,
        )
    )

    assert result is marker_store
    assert received["store_config"] == ("objective-artifacts", "objective-v2")
    assert received["model_config"] == {
        "AWS_REGION": "",
        "OBJECTIVE_MODEL_CHECKPOINT_DIR": "/opt/models/functiongemma",
        "OBJECTIVE_MODEL_REVISION": "a" * 40,
        "OBJECTIVE_MODEL_SHA256": "b" * 64,
        "OBJECTIVE_BASE_MODEL_URI": BASE_MODEL_URI,
        "OBJECTIVE_BASE_MODEL_SHA256": BASE_MODEL_SHA256,
    }
    engine, kwargs = received["app_config"]
    assert isinstance(engine, ServiceRecoveryEngine)
    assert kwargs["auth_token"] == "secret"
    assert kwargs["execution_adapter"] is marker_adapter
    assert kwargs["artifact_store"] is marker_store


def test_objective_application_wires_unconfigured_adapter_as_http_503() -> None:
    app = _create_objective_application(
        SimpleNamespace(
            s3_artifact_bucket="objective-artifacts",
            s3_artifact_prefix="objective",
            objective_auth_token="secret",
            objective_model_checkpoint_dir=None,
            objective_model_revision=None,
            objective_model_sha256=None,
            objective_base_model_uri=None,
            objective_base_model_sha256=None,
        )
    )
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post(
        "/v1/benchmark",
        json={
            "run_id": "blocked-run",
            "model_uri": BASE_MODEL_URI,
            "model_sha256": BASE_MODEL_SHA256,
            "task_ids": ["train-001"],
        },
        headers={"authorization": "Bearer secret"},
    )
    readiness = client.get(
        "/v1/readiness",
        headers={"authorization": "Bearer secret"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "objective benchmark unavailable"
    assert response.headers["x-objective-correlation-id"]
    assert readiness.status_code == 200
    assert readiness.json()["status"] == "blocked"
    assert readiness.json()["capabilities"] == {
        "benchmark": False,
        "verify-curation": False,
    }


def test_benchmark_stage_events_are_safe_and_cover_real_local_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging
    import sys

    from fastapi.testclient import TestClient

    from app.objective.service import InMemoryTrajectoryArtifactStore, create_objective_app
    from scripts.stage_functiongemma_checkpoint import validate_checkpoint_directory

    revision = "e" * 40
    snapshot = _snapshot(tmp_path / "snapshot")
    digest = checkpoint_snapshot_sha256(
        validate_checkpoint_directory(snapshot, revision=revision)
    )

    class _Tensor:
        shape = (1, 2)

        def to(self, device: object) -> _Tensor:
            del device
            return self

        def __getitem__(self, key: object) -> _Tensor:
            del key
            return self

    class _Processor:
        eos_token_id = 2

        def apply_chat_template(self, messages: object, **kwargs: Any) -> dict[str, _Tensor]:
            del messages, kwargs
            return {"input_ids": _Tensor()}

        def decode(self, completion: object, *, skip_special_tokens: bool) -> str:
            del completion
            assert skip_special_tokens is True
            return (
                "<start_function_call>call:get_logs{service:<escape>api<escape>}"
                "<end_function_call>"
            )

    class _Model:
        device = "cpu"

        def eval(self) -> None:
            return None

        def generate(self, **kwargs: Any) -> _Tensor:
            del kwargs
            return _Tensor()

    class _ProcessorLoader:
        @staticmethod
        def from_pretrained(path: str, **kwargs: Any) -> _Processor:
            assert path == str(snapshot.resolve())
            assert kwargs["revision"] == revision
            assert kwargs["local_files_only"] is True
            return _Processor()

    class _ModelLoader:
        @staticmethod
        def from_pretrained(path: str, **kwargs: Any) -> _Model:
            assert path == str(snapshot.resolve())
            assert kwargs["revision"] == revision
            assert kwargs["local_files_only"] is True
            return _Model()

    transformers = ModuleType("transformers")
    transformers.AutoProcessor = _ProcessorLoader  # type: ignore[attr-defined]
    transformers.AutoModelForCausalLM = _ModelLoader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    adapter = build_benchmark_execution_adapter(
        {
            "OBJECTIVE_MODEL_CHECKPOINT_DIR": str(snapshot),
            "OBJECTIVE_MODEL_REVISION": revision,
            "OBJECTIVE_MODEL_SHA256": digest,
            "OBJECTIVE_BASE_MODEL_URI": BASE_MODEL_URI,
            "OBJECTIVE_BASE_MODEL_SHA256": BASE_MODEL_SHA256,
        }
    )
    import app.objective.execution as execution

    monkeypatch.setattr(execution, "_local_inference_runtime_available", lambda: True)
    app = create_objective_app(
        auth_token="telemetry-secret",
        execution_adapter=adapter,
        artifact_store=InMemoryTrajectoryArtifactStore(),
    )
    client = TestClient(app)

    with caplog.at_level(logging.INFO, logger="app.objective.execution"):
        readiness_before = client.get(
            "/v1/readiness",
            headers={"authorization": "Bearer telemetry-secret"},
        )
        response = client.post(
            "/v1/benchmark",
            json={
                "run_id": "telemetry-run",
                "model_uri": BASE_MODEL_URI,
                "model_sha256": BASE_MODEL_SHA256,
                "task_ids": ["train-001"],
            },
            headers={"authorization": "Bearer telemetry-secret"},
        )
        readiness_after = client.get(
            "/v1/readiness",
            headers={"authorization": "Bearer telemetry-secret"},
        )

    assert response.status_code == 200
    assert readiness_before.json()["status"] == "blocked"
    assert readiness_before.json()["configuration_ready"] is True
    assert readiness_before.json()["checkpoint_ready"] is True
    assert readiness_before.json()["artifact_store_ready"] is True
    assert readiness_before.json()["model_load_ready"] is False
    assert readiness_before.json()["generation_ready"] is False
    assert readiness_before.json()["execution_ready"] is False
    assert readiness_after.json()["status"] == "ready"
    assert readiness_after.json()["model_load_ready"] is True
    assert readiness_after.json()["generation_ready"] is True
    assert readiness_after.json()["execution_ready"] is True
    events = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "app.objective.execution"
    ]
    required_stages = {
        "CHECKPOINT_RESOLVE",
        "PROCESSOR_LOAD",
        "MODEL_LOAD",
        "PROMPT_RENDER",
        "MODEL_GENERATE",
        "MODEL_DECODE",
        "FUNCTION_PARSE",
        "ENVIRONMENT_STEP",
        "TRAJECTORY_VERIFY",
        "S3_PERSIST",
        "BENCHMARK_COMPLETE",
    }
    assert required_stages <= {event["stage"] for event in events}
    correlation_ids = {event["correlation_id"] for event in events}
    assert correlation_ids == {response.headers["x-objective-correlation-id"]}
    allowed_fields = {
        "correlation_id",
        "stage",
        "exception_class",
        "duration_ms",
        "checkpoint_revision",
        "process_rss_bytes",
        "status",
    }
    assert all(set(event) <= allowed_fields for event in events)
    assert all(event["checkpoint_revision"] == revision for event in events)
    assert all(event["status"] in {"succeeded", "failed"} for event in events)
    assert all(
        isinstance(event["duration_ms"], (float, int))
        and event["duration_ms"] >= 0
        and event["duration_ms"] < float("inf")
        for event in events
    )
    assert all(
        event.get("process_rss_bytes") is None
        or isinstance(event["process_rss_bytes"], int)
        for event in events
    )
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "telemetry-secret" not in log_text
    assert "telemetry-run" not in log_text
    assert "<start_function_call>" not in log_text
    assert "BLOCKED:" not in response.text


def test_objective_stage_observer_failure_does_not_fail_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.objective.execution as execution

    def fail_sink(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("logging backend unavailable")

    monkeypatch.setattr(execution._OBJECTIVE_LOGGER, "info", fail_sink)

    with execution.objective_stage_trace("f" * 32, MODEL_REVISION):
        result = execution._run_objective_stage("MODEL_GENERATE", lambda: "generated")

    assert result == "generated"


def test_objective_stage_failure_logs_exception_class_without_exception_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    import app.objective.execution as execution

    with caplog.at_level(logging.INFO, logger="app.objective.execution"):
        with execution.objective_stage_trace("1" * 32, MODEL_REVISION):
            with pytest.raises(RuntimeError, match="private prompt sentinel"):
                execution._run_objective_stage(
                    "MODEL_GENERATE",
                    lambda: (_ for _ in ()).throw(RuntimeError("private prompt sentinel")),
                )

    events = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "app.objective.execution"
    ]
    assert len(events) == 1
    event = events[0]
    assert set(event) == {
        "checkpoint_revision",
        "correlation_id",
        "duration_ms",
        "exception_class",
        "process_rss_bytes",
        "stage",
        "status",
    }
    assert event["checkpoint_revision"] == MODEL_REVISION
    assert event["correlation_id"] == "1" * 32
    assert event["exception_class"] == "RuntimeError"
    assert event["stage"] == "MODEL_GENERATE"
    assert event["status"] == "failed"
    assert isinstance(event["duration_ms"], (float, int))
    assert event["duration_ms"] >= 0
    assert event["duration_ms"] < float("inf")
    assert "private prompt sentinel" not in caplog.text


def test_authenticated_readiness_blocks_without_a_complete_functiongemma_adapter() -> None:
    from app.objective.service import InMemoryTrajectoryArtifactStore, create_objective_app

    app = create_objective_app(
        auth_token="readiness-secret",
        artifact_store=InMemoryTrajectoryArtifactStore(),
    )
    from fastapi.testclient import TestClient

    client = TestClient(app)
    unauthenticated = client.get("/v1/readiness")
    response = client.get(
        "/v1/readiness",
        headers={"authorization": "Bearer readiness-secret"},
    )

    assert unauthenticated.status_code == 401
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "blocked"
    assert payload["service"] == "objective-worker"
    assert payload["configuration_ready"] is False
    assert payload["checkpoint_ready"] is False
    assert payload["model_load_ready"] is False
    assert payload["generation_ready"] is False
    assert payload["artifact_store_ready"] is True
    assert payload["execution_ready"] is False
    assert payload["capabilities"] == {"benchmark": False, "verify-curation": False}
    assert "functiongemma_adapter_unavailable" in payload["blockers"]
    assert "readiness-secret" not in response.text
    assert "hidden" not in response.text.lower()


def test_readiness_requires_verified_snapshot_and_runtime_without_loading_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient

    import app.objective.execution as execution
    from app.objective.service import InMemoryTrajectoryArtifactStore, create_objective_app

    revision = "c" * 40
    snapshot = _snapshot(tmp_path / "snapshot")
    from scripts.stage_functiongemma_checkpoint import validate_checkpoint_directory

    digest = checkpoint_snapshot_sha256(validate_checkpoint_directory(snapshot, revision=revision))
    adapter = build_benchmark_execution_adapter(
        {
            "OBJECTIVE_MODEL_CHECKPOINT_DIR": str(snapshot),
            "OBJECTIVE_MODEL_REVISION": revision,
            "OBJECTIVE_MODEL_SHA256": digest,
            "OBJECTIVE_BASE_MODEL_URI": BASE_MODEL_URI,
            "OBJECTIVE_BASE_MODEL_SHA256": BASE_MODEL_SHA256,
        }
    )
    assert isinstance(adapter, FunctionGemmaBenchmarkExecutionAdapter)
    monkeypatch.setattr(
        execution,
        "_local_inference_runtime_available",
        lambda: True,
        raising=False,
    )

    app = create_objective_app(
        auth_token="ready-secret",
        execution_adapter=adapter,
        artifact_store=InMemoryTrajectoryArtifactStore(),
    )
    response = TestClient(app).get(
        "/v1/readiness",
        headers={"authorization": "Bearer ready-secret"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "blocked",
        "service": "objective-worker",
        "model_id": "google/functiongemma-270m-it",
        "configuration_ready": True,
        "checkpoint_ready": True,
        "model_load_ready": False,
        "generation_ready": False,
        "artifact_store_ready": True,
        "execution_ready": False,
        "capabilities": {"benchmark": False, "verify-curation": False},
        "blockers": ["model_load_not_verified", "generation_not_verified"],
    }
    assert isinstance(adapter.policy, _DeferredLocalPolicy)
    assert adapter.policy._policy is None
    assert "ready-secret" not in response.text
    assert str(snapshot) not in response.text


def test_readiness_is_not_added_to_the_normal_coordinator_application() -> None:
    from fastapi import FastAPI

    from app.main import _select_application

    coordinator = FastAPI()
    selected = _select_application(SimpleNamespace(service_role="coordinator"), coordinator)

    assert selected is coordinator
    assert "/v1/readiness" not in {route.path for route in selected.routes}


def test_readiness_model_rejects_untrue_ready_claims_and_unscoped_fields() -> None:
    with pytest.raises(ValidationError, match="ready status"):
        ObjectiveReadinessResponse(
            status="ready",
            configuration_ready=True,
            checkpoint_ready=True,
            model_load_ready=False,
            generation_ready=False,
            artifact_store_ready=True,
            execution_ready=False,
            capabilities={"benchmark": True, "verify-curation": False},
            blockers=("checkpoint_unverified",),
        )

    with pytest.raises(ValidationError, match="extra_forbidden"):
        ObjectiveReadinessResponse(
            status="blocked",
            configuration_ready=False,
            checkpoint_ready=False,
            model_load_ready=False,
            generation_ready=False,
            artifact_store_ready=False,
            execution_ready=False,
            capabilities={"benchmark": False, "verify-curation": False},
            blockers=("functiongemma_adapter_unavailable",),
            auth_token="do-not-expose",
        )


def _snapshot(path: Path) -> Path:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"architectures": ["Gemma3ForCausalLM"], "model_type": "gemma3_text"}),
        encoding="utf-8",
    )
    (path / "tokenizer.json").write_text('{"version":1}', encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
    return path


def test_local_policy_loads_only_digest_pinned_checkpoint_without_hub_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision = "a" * 40
    snapshot = _snapshot(tmp_path / "snapshot")
    from scripts.stage_functiongemma_checkpoint import validate_checkpoint_directory

    digest = checkpoint_snapshot_sha256(validate_checkpoint_directory(snapshot, revision=revision))
    calls: list[tuple[str, dict[str, Any]]] = []

    class _Tensor:
        shape = (1, 2)

        def to(self, device: object) -> _Tensor:
            del device
            return self

        def __getitem__(self, key: object) -> _Tensor:
            del key
            return self

    class _Processor:
        eos_token_id = 2

        def apply_chat_template(self, messages: object, **kwargs: Any) -> dict[str, _Tensor]:
            del messages
            calls.append(("template", kwargs))
            return {"input_ids": _Tensor()}

        def decode(self, completion: object, *, skip_special_tokens: bool) -> str:
            del completion
            assert skip_special_tokens is True
            return (
                "<start_function_call>call:run_healthcheck{service:<escape>api<escape>}"
                "<end_function_call>"
            )

    class _Model:
        device = "cpu"

        def eval(self) -> None:
            calls.append(("eval", {}))

        def generate(self, **kwargs: Any) -> _Tensor:
            calls.append(("generate", kwargs))
            return _Tensor()

    class _ProcessorLoader:
        @staticmethod
        def from_pretrained(path: str, **kwargs: Any) -> Any:
            calls.append(("load", {"path": path, **kwargs}))
            return _Processor()

    class _ModelLoader:
        @staticmethod
        def from_pretrained(path: str, **kwargs: Any) -> Any:
            calls.append(("load", {"path": path, **kwargs}))
            return _Model()

    class _PeftModel:
        @staticmethod
        def from_pretrained(model: Any, path: str, **kwargs: Any) -> Any:
            calls.append(("adapter", {"path": path, **kwargs}))
            return model

    transformers = ModuleType("transformers")
    transformers.AutoProcessor = _ProcessorLoader  # type: ignore[attr-defined]
    transformers.AutoModelForCausalLM = _ModelLoader  # type: ignore[attr-defined]
    peft = ModuleType("peft")
    peft.PeftModel = _PeftModel  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "transformers", transformers)
    monkeypatch.setitem(__import__("sys").modules, "peft", peft)

    policy = FunctionGemmaLocalPolicy.from_checkpoint(
        snapshot,
        revision=revision,
        expected_sha256=digest,
        adapter_dir=snapshot,
    )
    task = Task(
        task_id="train-001",
        split=ObjectiveSplit.TRAIN,
        service_name="api",
        objective="restore the service",
        max_steps=10,
        engine_version="service-recovery-v1",
        seed=1,
    )
    action = policy(task, ())

    assert action == ToolCall(tool="run_healthcheck", arguments={"service": "api"})
    generation = next(item[1] for item in calls if item[0] == "generate")
    assert generation["pad_token_id"] == 2
    assert generation["max_new_tokens"] == 128
    assert generation["do_sample"] is False
    assert all(
        item[1].get("local_files_only") is True for item in calls if item[0] in {"load", "adapter"}
    )
    assert all(
        item[1].get("revision") == revision
        for item in calls
        if item[0] == "load" and item[1]["path"] == str(snapshot.resolve())
    )
    assert [item[0] for item in calls].count("adapter") == 1


def test_local_policy_rejects_missing_weights_before_loading_any_model(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(ObjectiveExecutionUnavailable, match="checkpoint is incomplete"):
        FunctionGemmaLocalPolicy.from_checkpoint(
            incomplete,
            revision="a" * 40,
            expected_sha256="b" * 64,
        )


def test_local_policy_rejects_digest_mismatch(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path / "snapshot")

    with pytest.raises(ObjectiveExecutionUnavailable, match="digest"):
        FunctionGemmaLocalPolicy.from_checkpoint(
            snapshot,
            revision="a" * 40,
            expected_sha256="b" * 64,
        )
