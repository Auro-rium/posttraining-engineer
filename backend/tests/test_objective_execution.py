from __future__ import annotations

import json
import struct
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


class _RulePolicy:
    """Deterministic test double for the target-model policy boundary only."""

    def __init__(self) -> None:
        self.received: list[tuple[Task, tuple[dict[str, Any], ...]]] = []

    def __call__(
        self, task: Task, observations: tuple[dict[str, Any], ...]
    ) -> ToolCall:
        self.received.append((task, observations))
        assert "failure_mode" not in task.model_dump()
        assert all(not {"reward", "done", "failure_mode"} & observation.keys()
                   for observation in observations)
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
    adapter = FunctionGemmaBenchmarkExecutionAdapter(policy)
    request = BenchmarkRequest(
        run_id="run-live-test",
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
        def __call__(
            self, task: Task, observations: tuple[dict[str, Any], ...]
        ) -> ToolCall:
            del task, observations
            return {"tool": "shell", "arguments": {}}  # type: ignore[return-value]

    adapter = FunctionGemmaBenchmarkExecutionAdapter(InvalidPolicy())
    request = BenchmarkRequest(run_id="run-invalid", task_ids=("train-001",))

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
            BenchmarkRequest(run_id="run-blocked", task_ids=("train-001",)),
            ServiceRecoveryEngine(seed=1),
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
        item["required"] == ["service"]
        for name, item in by_name.items()
        if name != "edit_config"
    )
    assert by_name["edit_config"]["required"] == ["service", "content"]
    assert by_name["edit_config"]["properties"]["content"]["type"] == "string"


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
        )
    )

    assert result is marker_store
    assert received["store_config"] == ("objective-artifacts", "objective-v2")
    assert received["model_config"] == {
        "OBJECTIVE_MODEL_CHECKPOINT_DIR": "/opt/models/functiongemma",
        "OBJECTIVE_MODEL_REVISION": "a" * 40,
        "OBJECTIVE_MODEL_SHA256": "b" * 64,
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
        )
    )
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post(
        "/v1/benchmark",
        json={"run_id": "blocked-run", "task_ids": ["train-001"]},
        headers={"authorization": "Bearer secret"},
    )
    readiness = client.get(
        "/v1/readiness",
        headers={"authorization": "Bearer secret"},
    )

    assert response.status_code == 503
    assert "BLOCKED" in response.json()["detail"]
    assert "ObjectiveExecutionUnavailable" in response.json()["detail"]
    assert readiness.status_code == 200
    assert readiness.json()["status"] == "blocked"
    assert readiness.json()["capabilities"] == {
        "benchmark": False,
        "verify-curation": False,
    }


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

    digest = checkpoint_snapshot_sha256(
        validate_checkpoint_directory(snapshot, revision=revision)
    )
    adapter = build_benchmark_execution_adapter(
        {
            "OBJECTIVE_MODEL_CHECKPOINT_DIR": str(snapshot),
            "OBJECTIVE_MODEL_REVISION": revision,
            "OBJECTIVE_MODEL_SHA256": digest,
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
        "status": "ready",
        "service": "objective-worker",
        "model_id": "google/functiongemma-270m-it",
        "capabilities": {"benchmark": True, "verify-curation": True},
        "blockers": [],
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
            capabilities={"benchmark": True, "verify-curation": False},
            blockers=("checkpoint_unverified",),
        )

    with pytest.raises(ValidationError, match="extra_forbidden"):
        ObjectiveReadinessResponse(
            status="blocked",
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

    digest = checkpoint_snapshot_sha256(
        validate_checkpoint_directory(snapshot, revision=revision)
    )
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
        def apply_chat_template(self, messages: object, **kwargs: Any) -> dict[str, _Tensor]:
            del messages
            calls.append(("template", kwargs))
            return {"input_ids": _Tensor()}

        def decode(self, completion: object, *, skip_special_tokens: bool) -> str:
            del completion
            assert skip_special_tokens is False
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

    transformers = ModuleType("transformers")
    transformers.AutoProcessor = _ProcessorLoader  # type: ignore[attr-defined]
    transformers.AutoModelForCausalLM = _ModelLoader  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "transformers", transformers)

    policy = FunctionGemmaLocalPolicy.from_checkpoint(
        snapshot, revision=revision, expected_sha256=digest
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
    assert all(item[1].get("local_files_only") is True for item in calls if item[0] == "load")
    assert all(
        item[1].get("revision") == revision
        for item in calls
        if item[0] == "load"
    )


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
