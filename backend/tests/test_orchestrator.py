from __future__ import annotations

import pytest

from app.agents import AgentContractError, CuratedDataset, DataCurator
from app.demo import LocalDemoDecisionProvider
from app.models import (
    CheckpointManifest,
    DatasetSplit,
    EvaluationReport,
    EvidenceLabel,
    Experiment,
    FailureCluster,
    JobStatus,
    QLoRAConfig,
    RunPhase,
    RunState,
    SFTExample,
    TrainingResult,
)
from app.orchestrator import Orchestrator
from app.repository import InMemoryRunRepository


async def make_run(repository: InMemoryRunRepository, *, max_experiments: int = 2) -> RunState:
    return await repository.create_run(
        RunState(
            champion=CheckpointManifest(
                success=0.35,
                regression_success=0.90,
                action_validity=0.97,
            ),
            max_experiments=max_experiments,
        )
    )


async def test_step_runs_one_role_and_emits_one_event() -> None:
    repository = InMemoryRunRepository()
    state = await make_run(repository)
    orchestrator = Orchestrator(repository, LocalDemoDecisionProvider())

    advanced = await orchestrator.step(state.run_id)
    events = await repository.list_events(state.run_id)

    assert advanced.phase is RunPhase.ANALYZING
    assert advanced.version == 1
    assert len(events) == 1
    assert events[0].type == "benchmark.completed"
    assert events[0].payload == {"trajectory_count": 2}


async def test_auto_uses_at_most_two_candidates_and_rejects_demo_provenance() -> None:
    repository = InMemoryRunRepository()
    state = await make_run(repository)
    orchestrator = Orchestrator(repository, LocalDemoDecisionProvider())

    completed = await orchestrator.auto(state.run_id)
    events = await repository.list_events(state.run_id)

    assert completed.phase is RunPhase.COMPLETED
    assert completed.experiments_used == 2
    assert len(completed.experiments) == 2
    assert all(item.status.value == "REJECTED" for item in completed.experiments)
    assert completed.human_tuning_decisions == 0
    assert sum(event.type == "checkpoint.rejected" for event in events) == 2


class VerifiedProvider(LocalDemoDecisionProvider):
    async def evaluate(self, state: RunState, experiment: Experiment) -> EvaluationReport:
        report = await super().evaluate(state, experiment)
        return report.model_copy(
            update={
                "provenance_complete": True,
                "evidence_label": EvidenceLabel.PRIOR_VERIFIED_RUN,
            }
        )


async def test_qualifying_candidate_is_promoted_and_becomes_champion() -> None:
    repository = InMemoryRunRepository()
    state = await make_run(repository)
    orchestrator = Orchestrator(repository, VerifiedProvider())

    completed = await orchestrator.auto(state.run_id)

    assert completed.phase is RunPhase.COMPLETED
    assert completed.experiments_used == 1
    assert completed.experiments[0].promotion is not None
    assert completed.experiments[0].promotion.promoted is True
    assert completed.champion.version == completed.experiments[0].experiment_id
    assert completed.champion.success == pytest.approx(0.40)


async def test_cancel_is_immediate_and_idempotent() -> None:
    repository = InMemoryRunRepository()
    state = await make_run(repository)
    orchestrator = Orchestrator(repository, LocalDemoDecisionProvider())

    cancelled = await orchestrator.cancel(state.run_id)
    repeated = await orchestrator.cancel(state.run_id)

    assert cancelled.phase is RunPhase.CANCELLED
    assert repeated.phase is RunPhase.CANCELLED
    assert repeated.version == cancelled.version
    assert len(await repository.list_events(state.run_id)) == 1


class UnverifiedProvider(LocalDemoDecisionProvider):
    async def curate_dataset(self, state: RunState) -> CuratedDataset:
        curated = await super().curate_dataset(state)
        invalid = SFTExample.model_construct(
            source_trajectory_id="train-only",
            source_step_index=0,
            source_split=DatasetSplit.TRAIN,
            observation="observation",
            target_action=curated.examples[0].target_action,
            verified=False,
            verifier_reward_before=0.0,
            verifier_reward_after=0.5,
        )
        return CuratedDataset(manifest=curated.manifest, examples=(invalid,))


async def test_data_curator_rejects_unverified_repairs() -> None:
    state = RunState(
        failure_clusters=[
            FailureCluster(
                label="test",
                description="test failure",
                trajectory_ids=["train-only"],
                frequency=1,
            )
        ]
    )

    with pytest.raises(AgentContractError, match="unverified"):
        await DataCurator(UnverifiedProvider()).run(state)


class InvalidConfigProvider(LocalDemoDecisionProvider):
    async def design_training(self, state: RunState) -> QLoRAConfig:
        return QLoRAConfig.model_construct(
            # Intentionally bypasses validation to exercise the server-side guard.
            rank=4,  # type: ignore[arg-type]
            learning_rate=1e-4,
            epochs=2,
            dropout=0.0,
            max_sequence_length=512,
            effective_batch_size=32,
        )


async def test_invalid_training_config_fails_run() -> None:
    repository = InMemoryRunRepository()
    state = await make_run(repository)
    orchestrator = Orchestrator(repository, InvalidConfigProvider())

    for _ in range(4):
        state = await orchestrator.step(state.run_id)
    assert state.phase is RunPhase.DESIGNING

    with pytest.raises(AgentContractError, match="rank"):
        await orchestrator.step(state.run_id)

    failed = await repository.get_run(state.run_id)
    assert failed.phase is RunPhase.FAILED


class DuplicateConfigProvider(LocalDemoDecisionProvider):
    async def design_training(self, state: RunState) -> QLoRAConfig:
        return QLoRAConfig(rank=8, learning_rate=1e-4, epochs=2, dropout=0.0)


async def test_duplicate_training_config_is_rejected_on_second_attempt() -> None:
    repository = InMemoryRunRepository()
    state = await make_run(repository)
    orchestrator = Orchestrator(repository, DuplicateConfigProvider())

    for _ in range(10):
        state = await orchestrator.step(state.run_id)
    assert state.phase is RunPhase.DESIGNING

    with pytest.raises(AgentContractError, match="duplicate"):
        await orchestrator.step(state.run_id)

    failed = await repository.get_run(state.run_id)
    assert failed.phase is RunPhase.FAILED


class FailedTrainingProvider(LocalDemoDecisionProvider):
    async def launch_training(
        self, state: RunState, experiment: Experiment
    ) -> TrainingResult:
        return TrainingResult(job_id="failed-job", status=JobStatus.FAILED, error_code="OOM")


async def test_failed_training_job_fails_run() -> None:
    repository = InMemoryRunRepository()
    state = await make_run(repository)
    orchestrator = Orchestrator(repository, FailedTrainingProvider())

    for _ in range(5):
        state = await orchestrator.step(state.run_id)
    assert state.phase is RunPhase.TRAINING

    with pytest.raises(Exception, match="ended with status FAILED"):
        await orchestrator.step(state.run_id)

    failed = await repository.get_run(state.run_id)
    assert failed.phase is RunPhase.FAILED
