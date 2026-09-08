from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

from app.objective.engine import ServiceRecoveryEngine, TrajectoryNotAdmissible
from app.objective.models import ALLOWED_TOOLS, ObjectiveSplit, ToolCall
from app.objective.service import create_objective_app


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
    )
    unverified = trajectory.model_copy(update={"verified": False})

    with pytest.raises(TrajectoryNotAdmissible, match="verifier"):
        engine.build_dataset([unverified], run_id="run-1", experiment_id="exp-1")

    verified = engine.verify(trajectory)
    dataset = engine.build_dataset([verified.trajectory], run_id="run-1", experiment_id="exp-1")
    assert dataset.manifest.row_count == 1
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
    assert response.status_code == 200
    body = response.json()
    assert body["split"] == "train"
    assert "failure_mode" not in response.text
    assert "hidden" not in response.text


def test_curation_endpoint_returns_a_content_addressed_dataset_for_replay_scope() -> None:
    engine = ServiceRecoveryEngine(seed=9)
    trajectory = engine.run_episode(
        "train-001",
        [ToolCall(tool="inspect_service", arguments={"service": "api"})],
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
