from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app import live_execution
from app.live_execution import LiveExecutionConfig, LiveObjectiveAdapter, ObjectiveWorkerClient
from app.objective.engine import ServiceRecoveryEngine
from app.objective.models import (
    CorrectionProposal,
    CorrectionReplayOutcome,
    CorrectionReplayResponse,
    ObjectiveSplit,
    ToolCall,
    TrajectoryReference,
    encode_trajectory_reference,
)


def _config() -> LiveExecutionConfig:
    return LiveExecutionConfig(
        artifact_bucket="test-artifacts",
        dynamodb_table="runs",
        training_role_arn="arn:aws:iam::123456789012:role/train",
        training_image="123456789012.dkr.ecr.us-east-1.amazonaws.com/train:sha256-abc",
        evaluation_image="123456789012.dkr.ecr.us-east-1.amazonaws.com/eval:sha256-def",
        objective_worker_url="https://objective.example",
        hf_repo_id="google/functiongemma-270m-it",
        hf_revision="a" * 40,
        training_input_s3_uri="s3://artifacts/train",
        evaluation_input_s3_uri="s3://artifacts/eval",
    )


def _repair_trajectories() -> tuple[Any, Any, tuple[ToolCall, ...]]:
    engine = ServiceRecoveryEngine(seed=9)
    task_id = "replay-client-repair-001"
    failed = engine.verify(
        engine.run_episode(
            task_id,
            [ToolCall(tool="run_healthcheck", arguments={})] * 10,
            split=ObjectiveSplit.REPLAY,
        )
    ).trajectory
    definition = engine._make_definition(task_id, ObjectiveSplit.REPLAY)
    actions: tuple[ToolCall, ...]
    if definition.failure_mode == "config_error":
        actions = (
            ToolCall(
                tool="edit_config",
                arguments={"service": definition.service_name, "content": "fixed"},
            ),
        )
    else:
        actions = (
            ToolCall(tool="restart_service", arguments={"service": definition.service_name}),
            ToolCall(tool="run_healthcheck", arguments={"service": definition.service_name}),
        )
    repaired = engine.verify(
        engine.run_episode(
            task_id,
            actions,
            split=ObjectiveSplit.REPLAY,
            repaired_from_trajectory_id=failed.trajectory_id,
        )
    ).trajectory
    return failed, repaired, actions


def test_live_objective_adapter_replays_proposals_then_curation_uses_only_pass_refs() -> None:
    failed, repaired, actions = _repair_trajectories()
    source_reference = TrajectoryReference(
        trajectory_id=failed.trajectory_id,
        task_id=failed.task_id,
        split=failed.split,
        verified=True,
    )
    repaired_reference = TrajectoryReference(
        trajectory_id=repaired.trajectory_id,
        task_id=repaired.task_id,
        split=repaired.split,
        verified=True,
    )
    passing = CorrectionProposal(
        source_trajectory_id=failed.trajectory_id,
        task_id=failed.task_id,
        split=failed.split,
        actions=actions,
    )
    rejected = CorrectionProposal(
        source_trajectory_id=failed.trajectory_id,
        task_id=failed.task_id,
        split=failed.split,
        actions=tuple(ToolCall(tool="run_healthcheck", arguments={}) for _ in range(10)),
    )
    dataset = ServiceRecoveryEngine(seed=9).build_dataset(
        [repaired],
        run_id="run-client",
        experiment_id="run-client-1",
        scope=ObjectiveSplit.REPLAY,
    )

    class WorkerClient:
        def __init__(self) -> None:
            self.proposals: tuple[CorrectionProposal, ...] = ()
            self.curation_refs: tuple[TrajectoryReference, ...] = ()

        def replay_corrections(self, **kwargs: Any) -> CorrectionReplayResponse:
            self.proposals = tuple(kwargs["proposals"])
            return CorrectionReplayResponse(
                run_id=kwargs["run_id"],
                experiment_id=kwargs["experiment_id"],
                split=kwargs["split"],
                outcomes=(
                    CorrectionReplayOutcome(
                        proposal_id=passing.proposal_id,
                        source_trajectory_id=failed.trajectory_id,
                        task_id=failed.task_id,
                        split=failed.split,
                        status="PASS",
                        reason="replay_passed",
                        trajectory_reference=repaired_reference,
                    ),
                    CorrectionReplayOutcome(
                        proposal_id=rejected.proposal_id,
                        source_trajectory_id=failed.trajectory_id,
                        task_id=failed.task_id,
                        split=failed.split,
                        status="REJECTED",
                        reason="replay_failed",
                    ),
                ),
            )

        def verify_curation(self, **kwargs: Any) -> dict[str, Any]:
            self.curation_refs = tuple(kwargs["trajectory_references"])
            return dataset.model_dump(mode="json")

    client = WorkerClient()
    adapter = LiveObjectiveAdapter(client, _config())  # type: ignore[arg-type]
    plan = SimpleNamespace(
        selected_trajectory_refs=(encode_trajectory_reference(source_reference),),
        correction_proposals=(passing, rejected),
    )

    result = adapter.build_dataset(
        SimpleNamespace(run_id="run-client"), plan, experiment_number=1
    )

    assert result.dataset_id == dataset.manifest.dataset_id
    assert client.proposals == (passing, rejected)
    assert client.curation_refs == (source_reference, repaired_reference)


def test_objective_worker_client_posts_typed_replay_contract(monkeypatch: Any) -> None:
    failed, repaired, actions = _repair_trajectories()
    proposal = CorrectionProposal(
        source_trajectory_id=failed.trajectory_id,
        task_id=failed.task_id,
        split=failed.split,
        actions=actions,
    )
    reference = TrajectoryReference(
        trajectory_id=repaired.trajectory_id,
        task_id=repaired.task_id,
        split=repaired.split,
        verified=True,
    )
    response = CorrectionReplayResponse(
        run_id="run-http",
        experiment_id="exp-http",
        split=ObjectiveSplit.REPLAY,
        outcomes=(
            CorrectionReplayOutcome(
                proposal_id=proposal.proposal_id,
                source_trajectory_id=failed.trajectory_id,
                task_id=failed.task_id,
                split=ObjectiveSplit.REPLAY,
                status="PASS",
                reason="replay_passed",
                trajectory_reference=reference,
            ),
        ),
    )
    received: dict[str, Any] = {}

    def fake_http(url: str, **kwargs: Any) -> dict[str, Any]:
        received["url"] = url
        received.update(kwargs)
        return response.model_dump(mode="json")

    monkeypatch.setattr(live_execution, "_http_json", fake_http)
    client = ObjectiveWorkerClient(
        "https://objective.example", auth_token="secret", timeout_seconds=10
    )

    result = client.replay_corrections(
        run_id="run-http",
        experiment_id="exp-http",
        split=ObjectiveSplit.REPLAY,
        proposals=(proposal,),
    )

    assert result == response
    assert received["url"] == "https://objective.example/v1/replay-corrections"
    assert received["method"] == "POST"
    assert received["headers"] == {"Authorization": "Bearer secret"}
    assert received["payload"] == {
        "run_id": "run-http",
        "experiment_id": "exp-http",
        "split": "replay",
        "proposals": [
            {
                "source_trajectory_id": failed.trajectory_id,
                "task_id": failed.task_id,
                "split": "replay",
                "actions": [action.model_dump(mode="json") for action in actions],
            }
        ],
    }
