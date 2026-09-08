from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

from app.objective.engine import (
    ServiceRecoveryEngine,
    TrajectoryNotAdmissible,
    _SealedEvaluation,
)
from app.objective.models import (
    ALLOWED_TOOLS,
    BenchmarkExecutionResult,
    BenchmarkRequest,
    Dataset,
    ObjectiveSplit,
    ToolCall,
)
from app.objective.service import InMemoryTrajectoryArtifactStore, create_objective_app


def test_service_recovery_exposes_only_the_allowlisted_tools() -> None:
    engine = ServiceRecoveryEngine(seed=7)

    task = engine.reset(split=ObjectiveSplit.TRAIN, task_id="train-001")

    assert task.allowed_tools == ALLOWED_TOOLS
    assert set(task.allowed_tools) == {
        "get_logs",
        "inspect_service",
        "read_config",
        "edit_config",
        "restart_service",
        "run_healthcheck",
    }
    assert "failure_mode" not in task.model_dump()
    with pytest.raises(ValueError, match="not allowed"):
        engine.step("shell", {})


def test_same_seed_and_actions_produce_the_same_reward_and_trajectory() -> None:
    actions = [
        ToolCall(tool="inspect_service", arguments={"service": "api"}),
        ToolCall(tool="read_config", arguments={"service": "api"}),
    ]
    first = ServiceRecoveryEngine(seed=11).run_episode("train-001", actions)
    second = ServiceRecoveryEngine(seed=11).run_episode("train-001", actions)

    assert first == second
    assert [step.reward for step in first.steps] == [step.reward for step in second.steps]


def test_only_verifier_confirmed_trajectories_can_enter_a_dataset() -> None:
    engine = ServiceRecoveryEngine(seed=3)
    trajectory = engine.run_episode(
        "train-001",
        [ToolCall(tool="run_healthcheck", arguments={"service": "api"})],
        split=ObjectiveSplit.REPLAY,
    )
    unverified = trajectory.model_copy(update={"verified": False})

    with pytest.raises(TrajectoryNotAdmissible, match="verifier"):
        engine.build_dataset(
            [unverified], run_id="run-1", experiment_id="exp-1", scope=ObjectiveSplit.REPLAY
        )

    verified = engine.verify(trajectory)
    dataset = engine.build_dataset(
        [verified.trajectory],
        run_id="run-1",
        experiment_id="exp-1",
        scope=ObjectiveSplit.REPLAY,
    )
    assert dataset.manifest.row_count == 1
    assert dataset.manifest.s3_uri.endswith(f"{dataset.manifest.dataset_id}.jsonl")
    assert dataset.manifest.created_at.tzinfo is not None
    assert dataset.rows[0].verifier_confirmed is True
    assert dataset.rows[0].source_trajectory_id == trajectory.trajectory_id
    assert (
        dataset.manifest.sha256
        == hashlib.sha256(dataset.rows[0].canonical_json().encode()).hexdigest()
    )


def test_replay_rejects_mismatched_provenance() -> None:
    engine = ServiceRecoveryEngine(seed=5)
    trajectory = engine.run_episode(
        "train-001",
        [ToolCall(tool="inspect_service", arguments={"service": "api"})],
    )
    mismatched = trajectory.model_copy(update={"engine_version": "different"})

    with pytest.raises(ValueError, match="provenance"):
        engine.verify(mismatched)


def test_admission_rejects_a_forged_verified_zero_step_trajectory() -> None:
    engine = ServiceRecoveryEngine(seed=5)
    forged = engine.run_episode("replay-001", [], split=ObjectiveSplit.REPLAY).model_copy(
        update={"verified": True}
    )

    with pytest.raises(TrajectoryNotAdmissible, match="verifier"):
        engine.build_dataset(
            [forged], run_id="run-1", experiment_id="exp-1", scope=ObjectiveSplit.REPLAY
        )


def test_replay_scope_requires_every_trajectory_to_match_the_request_scope() -> None:
    engine = ServiceRecoveryEngine(seed=5)
    train = engine.run_episode(
        "train-001", [ToolCall(tool="get_logs", arguments={})], split=ObjectiveSplit.TRAIN
    )

    with pytest.raises(TrajectoryNotAdmissible, match="scope"):
        engine.build_dataset(
            [train.model_copy(update={"verified": True})],
            run_id="run-1",
            experiment_id="exp-1",
            scope=ObjectiveSplit.REPLAY,
        )


def test_sealed_engine_uses_a_private_non_serializing_hidden_path() -> None:
    result = ServiceRecoveryEngine(seed=2, sealed=True).run_episode(
        "hidden-001",
        [ToolCall(tool="run_healthcheck", arguments={})],
        split=ObjectiveSplit.HIDDEN,
    )

    assert isinstance(result, _SealedEvaluation)
    assert result.reward in {0, 1}
    assert not hasattr(result, "model_dump")


def test_reward_is_binary_and_failure_modes_drive_repair_behavior() -> None:
    engine = ServiceRecoveryEngine(seed=17)
    actions_by_mode = {
        "config_error": ToolCall(
            tool="edit_config", arguments={"service": "api", "content": "setting2=value2"}
        ),
        "dependency_failure": ToolCall(tool="restart_service", arguments={"service": "api"}),
        "healthcheck_failure": ToolCall(tool="restart_service", arguments={"service": "api"}),
    }
    covered: set[str] = set()
    for index in range(100):
        task_id = f"replay-{index:03d}"
        mode = engine._make_definition(task_id, ObjectiveSplit.REPLAY).failure_mode
        covered.add(mode)
        trajectory = engine.run_episode(
            task_id, [actions_by_mode[mode]], split=ObjectiveSplit.REPLAY
        )
        if trajectory.steps:
            assert all(step.reward in {0, 1} for step in trajectory.steps)
        if covered == set(actions_by_mode):
            break
    else:
        pytest.fail("deterministic fixture did not cover all failure modes")

    assert any(
        engine._make_definition(f"replay-{index:03d}", ObjectiveSplit.REPLAY).failure_mode
        == "config_error"
        for index in range(100)
    )
    assert any(
        engine._make_definition(f"replay-{index:03d}", ObjectiveSplit.REPLAY).failure_mode
        == "dependency_failure"
        for index in range(100)
    )
    assert any(
        engine._make_definition(f"replay-{index:03d}", ObjectiveSplit.REPLAY).failure_mode
        == "healthcheck_failure"
        for index in range(100)
    )


def test_dataset_contract_recomputes_digest() -> None:
    engine = ServiceRecoveryEngine(seed=3)
    trajectory = engine.verify(
        engine.run_episode(
            "replay-001",
            [ToolCall(tool="get_logs", arguments={})],
            split=ObjectiveSplit.REPLAY,
        )
    ).trajectory
    dataset = engine.build_dataset(
        [trajectory], run_id="run-1", experiment_id="exp-1", scope=ObjectiveSplit.REPLAY
    )
    forged_manifest = dataset.manifest.model_copy(update={"sha256": "a" * 64})
    with pytest.raises(ValueError, match="digest"):
        Dataset(manifest=forged_manifest, rows=dataset.rows)


def test_objective_service_requires_auth_and_rejects_hidden_split() -> None:
    client = TestClient(create_objective_app(ServiceRecoveryEngine(seed=1), auth_token="secret"))
    payload = {"run_id": "run-1", "split": "train", "task_ids": ["train-001"]}

    assert client.post("/v1/benchmark", json=payload).status_code == 401
    response = client.post(
        "/v1/benchmark",
        json={**payload, "split": "hidden"},
        headers={"x-objective-token": "secret"},
    )
    assert response.status_code == 422
    assert "hidden" in response.text

    response = client.post(
        "/v1/benchmark", json=payload, headers={"authorization": "Bearer secret"}
    )
    assert response.status_code == 503
    assert "adapter" in response.text


def test_benchmark_requires_an_injected_execution_adapter() -> None:
    class Adapter:
        def execute_benchmark(
            self, request: BenchmarkRequest, engine: ServiceRecoveryEngine
        ) -> BenchmarkExecutionResult:
            return BenchmarkExecutionResult(trajectories=tuple(
                engine.run_episode(
                    task_id,
                    [ToolCall(tool="run_healthcheck", arguments={})],
                    split=request.split,
                )
                for task_id in request.task_ids
            ))

    client = TestClient(
        create_objective_app(
            ServiceRecoveryEngine(seed=1), auth_token="secret", execution_adapter=Adapter()
            , artifact_store=InMemoryTrajectoryArtifactStore()
        )
    )
    response = client.post(
        "/v1/benchmark",
        json={"run_id": "run-1", "split": "train", "task_ids": ["train-001"]},
        headers={"authorization": "Bearer secret"},
    )
    assert response.status_code == 200
    assert response.json()["success_rate"] in {0, 1}
    assert "failure_mode" not in response.text


@pytest.mark.parametrize(
    "trajectories",
    [
        (),
        ("wrong-task",),
        ("train-001", "train-001"),
    ],
)
def test_benchmark_rejects_empty_incomplete_or_duplicate_adapter_output(
    trajectories: tuple[str, ...],
) -> None:
    class Adapter:
        def execute_benchmark(
            self, request: BenchmarkRequest, engine: ServiceRecoveryEngine
        ) -> BenchmarkExecutionResult:
            return BenchmarkExecutionResult(
                trajectories=tuple(
                    engine.run_episode(
                        task_id,
                        [ToolCall(tool="get_logs", arguments={})],
                        split=request.split,
                    )
                    for task_id in trajectories
                )
            )

    client = TestClient(
        create_objective_app(
            ServiceRecoveryEngine(seed=1),
            auth_token="secret",
            execution_adapter=Adapter(),
            artifact_store=InMemoryTrajectoryArtifactStore(),
        )
    )
    response = client.post(
        "/v1/benchmark",
        json={"run_id": "run-1", "split": "train", "task_ids": ["train-001"]},
        headers={"authorization": "Bearer secret"},
    )
    assert response.status_code == 503


def test_benchmark_references_are_replay_verified_and_resolvable_by_curation() -> None:
    class Adapter:
        def execute_benchmark(
            self, request: BenchmarkRequest, engine: ServiceRecoveryEngine
        ) -> BenchmarkExecutionResult:
            return BenchmarkExecutionResult(
                trajectories=tuple(
                    engine.run_episode(
                        task_id,
                        [ToolCall(tool="get_logs", arguments={})],
                        split=request.split,
                    )
                    for task_id in request.task_ids
                )
            )

    store = InMemoryTrajectoryArtifactStore()
    client = TestClient(
        create_objective_app(
            ServiceRecoveryEngine(seed=1),
            auth_token="secret",
            execution_adapter=Adapter(),
            artifact_store=store,
        )
    )
    headers = {"authorization": "Bearer secret"}
    benchmark = client.post(
        "/v1/benchmark",
        json={"run_id": "run-1", "split": "replay", "task_ids": ["replay-001"]},
        headers=headers,
    )
    assert benchmark.status_code == 200
    reference = benchmark.json()["trajectory_references"][0]
    assert reference["verified"] is True

    curated = client.post(
        "/v1/verify-curation",
        json={
            "run_id": "run-1",
            "experiment_id": "exp-1",
            "split": "replay",
            "trajectory_references": [reference],
        },
        headers=headers,
    )
    assert curated.status_code == 200
    assert curated.json()["manifest"]["row_count"] == 1


def test_curation_endpoint_returns_a_content_addressed_dataset_for_replay_scope() -> None:
    engine = ServiceRecoveryEngine(seed=9)
    trajectory = engine.run_episode(
        "train-001",
        [ToolCall(tool="inspect_service", arguments={"service": "api"})],
        split=ObjectiveSplit.REPLAY,
    )
    payload = {
        "run_id": "run-1",
        "experiment_id": "exp-1",
        "split": "replay",
        "trajectories": [trajectory.model_dump(mode="json")],
    }
    client = TestClient(create_objective_app(engine, auth_token="secret"))
    response = client.post(
        "/v1/verify-curation", json=payload, headers={"x-objective-token": "secret"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["manifest"]["row_count"] == 1
    assert len(body["manifest"]["sha256"]) == 64
    assert "failure_mode" not in response.text
