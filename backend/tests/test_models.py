from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from app.models import (
    AgentRole,
    DatasetSplit,
    EvaluationReport,
    EvidenceLabel,
    QLoRAConfig,
    RunEvent,
    RunPhase,
    RunState,
    SFTExample,
    ToolCall,
)
from app.repository import InMemoryRunRepository, VersionConflictError


def test_sft_example_requires_train_side_verified_improvement() -> None:
    values = {
        "source_trajectory_id": "trajectory-1",
        "source_step_index": 0,
        "observation": "shop observation",
        "target_action": ToolCall(name="search", arguments={"query": "blue shoes"}),
        "verified": True,
        "verifier_reward_before": 0.0,
        "verifier_reward_after": 0.5,
    }
    assert SFTExample(**values).source_split is DatasetSplit.TRAIN
    with pytest.raises(ValidationError, match="train split"):
        SFTExample(**values, source_split=DatasetSplit.HELDOUT)
    with pytest.raises(ValidationError, match="improve verifier reward"):
        SFTExample(**{**values, "verifier_reward_after": 0.0})


def test_qlora_search_space_is_enforced_by_the_contract() -> None:
    config = QLoRAConfig(rank=16, learning_rate=1e-4, epochs=3, dropout=0.05)
    assert config.max_sequence_length == 512
    with pytest.raises(ValidationError):
        QLoRAConfig(rank=64, learning_rate=1e-4, epochs=3, dropout=0.05)


def test_evaluation_deltas_are_derived_not_supplied() -> None:
    report = EvaluationReport(
        task_count=20,
        champion_success=0.35,
        candidate_success=0.41,
        champion_regression_success=0.8,
        candidate_regression_success=0.79,
        champion_action_validity=0.99,
        candidate_action_validity=0.99,
        paired_improvement_positive=True,
        provenance_complete=True,
        evidence_label=EvidenceLabel.LIVE,
    )
    assert report.success_delta == pytest.approx(0.06)
    assert report.regression_delta == pytest.approx(-0.01)


def test_run_budget_and_optimistic_repository_versions() -> None:
    async def scenario() -> None:
        repository = InMemoryRunRepository()
        created = await repository.create_run(RunState(run_id="RUN-1"))
        assert created.version == 0
        saved = await repository.save_run(
            created.model_copy(update={"phase": RunPhase.BENCHMARKING}),
            expected_version=0,
        )
        assert saved.version == 1
        with pytest.raises(VersionConflictError):
            await repository.save_run(created, expected_version=0)
        event = RunEvent(
            run_id=created.run_id,
            type="benchmark.started",
            phase=RunPhase.BENCHMARKING,
            agent=AgentRole.BENCHMARK_RUNNER,
        )
        await repository.append_event(event)
        assert (await repository.list_events(created.run_id))[0].event_id == event.event_id

    asyncio.run(scenario())
