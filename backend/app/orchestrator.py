"""Deterministic state machine for the autonomous post-training research loop."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from app.agents import (
    BenchmarkRunner,
    ChampionManager,
    DataCurator,
    DecisionProvider,
    EvaluationAgent,
    FailureAnalyst,
    ResearchAgent,
    TrainingDesigner,
    TrainingExecutor,
)
from app.models import (
    AgentRole,
    CheckpointManifest,
    EvaluationReport,
    EvidenceLabel,
    Experiment,
    ExperimentStatus,
    JobStatus,
    PromotionDecision,
    RunEvent,
    RunPhase,
    RunState,
)
from app.telemetry import async_telemetry_span, current_trace_id

TERMINAL_PHASES = frozenset({RunPhase.COMPLETED, RunPhase.CANCELLED, RunPhase.FAILED})


class InvalidRunStateError(RuntimeError):
    """Raised when a persisted run cannot safely advance."""


class RunRepository(Protocol):
    """Minimum persistence interface required by the orchestration engine."""

    async def get_run(self, run_id: str) -> RunState: ...

    async def save_run(self, state: RunState, *, expected_version: int) -> RunState: ...

    async def append_event(self, event: RunEvent) -> RunEvent: ...


def _metric(report: EvaluationReport, *names: str) -> float:
    for name in names:
        value = getattr(report, name, None)
        if value is not None:
            return float(value)
    raise ValueError(f"evaluation report is missing required metric: {' or '.join(names)}")


def decide_promotion(
    evaluation: EvaluationReport, *, provenance_complete: bool | None = None
) -> PromotionDecision:
    """Apply the non-LLM champion gate and return every pass/fail reason.

    Success must improve by at least five percentage points, regression-canary
    performance may lose at most two points, action validity cannot decline, and
    the evidence chain must be complete.
    """

    champion_success = _metric(evaluation, "champion_success", "champion_success_rate")
    candidate_success = _metric(evaluation, "candidate_success", "candidate_success_rate")
    success_gain = candidate_success - champion_success

    regression_loss_value = getattr(evaluation, "regression_loss", None)
    if regression_loss_value is None:
        champion_regression = _metric(
            evaluation,
            "champion_regression_success",
            "champion_regression",
            "champion_regression_success_rate",
        )
        candidate_regression = _metric(
            evaluation,
            "candidate_regression_success",
            "candidate_regression",
            "candidate_regression_success_rate",
        )
        regression_loss = champion_regression - candidate_regression
    else:
        regression_loss = float(regression_loss_value)

    champion_validity = _metric(
        evaluation, "champion_validity", "champion_action_validity", "champion_valid_action_rate"
    )
    candidate_validity = _metric(
        evaluation, "candidate_validity", "candidate_action_validity", "candidate_valid_action_rate"
    )
    report_provenance = bool(getattr(evaluation, "provenance_complete", False))
    evidence_complete = report_provenance
    if provenance_complete is not None:
        evidence_complete = bool(provenance_complete) and report_provenance
    if getattr(evaluation, "evidence_label", None) is EvidenceLabel.EXPLANATION:
        evidence_complete = False

    checks = {
        "success gain is at least 5 percentage points": success_gain + 1e-12 >= 0.05,
        "regression loss is at most 2 percentage points": regression_loss <= 0.02 + 1e-12,
        "action validity does not decline": candidate_validity + 1e-12 >= champion_validity,
        "artifact provenance is complete": evidence_complete,
    }
    promoted = all(checks.values())
    reasons = [
        f"{'PASS' if passed else 'FAIL'}: {description}" for description, passed in checks.items()
    ]
    return PromotionDecision.model_validate(
        {
            "promoted": promoted,
            "reasons": reasons,
            "success_delta": success_gain,
            "regression_delta": -regression_loss,
        }
    )


class Orchestrator:
    """Advance one persisted run through the eight bounded specialist roles."""

    def __init__(self, repository: RunRepository, provider: DecisionProvider) -> None:
        self.repository = repository
        self.provider = provider
        self.benchmark_runner = BenchmarkRunner(provider)
        self.failure_analyst = FailureAnalyst(provider)
        self.research_agent = ResearchAgent(provider)
        self.data_curator = DataCurator(provider)
        self.training_designer = TrainingDesigner(provider)
        self.training_executor = TrainingExecutor(provider)
        self.evaluation_agent = EvaluationAgent(provider)
        self.champion_manager = ChampionManager()
        # Local trajectory objects are intentionally process-local. Cloud providers
        # persist their artifact URI and resolve it during failure analysis.
        self._trajectories: dict[str, list[Any]] = {}

    async def step(self, run_id: str) -> RunState:
        """Execute exactly one logical role and persist its output and event."""

        state = await self._require_run(run_id)
        if state.phase in TERMINAL_PHASES:
            return state
        if state.cancel_requested:
            return await self._set_terminal(state, RunPhase.CANCELLED, "run.cancelled")

        handlers: dict[RunPhase, Callable[[RunState], Awaitable[RunState]]] = {
            RunPhase.NOT_STARTED: self._benchmark,
            RunPhase.BENCHMARKING: self._benchmark,
            RunPhase.ANALYZING: self._analyze,
            RunPhase.RESEARCHING: self._research,
            RunPhase.CURATING: self._curate,
            RunPhase.DESIGNING: self._design,
            RunPhase.TRAINING: self._train,
            RunPhase.EVALUATING: self._evaluate,
            RunPhase.PROMOTING: self._promote,
        }
        handler = handlers.get(state.phase)
        if handler is None:
            raise InvalidRunStateError(f"no handler for run phase {state.phase}")
        roles = {
            RunPhase.NOT_STARTED: AgentRole.BENCHMARK_RUNNER,
            RunPhase.BENCHMARKING: AgentRole.BENCHMARK_RUNNER,
            RunPhase.ANALYZING: AgentRole.FAILURE_ANALYST,
            RunPhase.RESEARCHING: AgentRole.RESEARCH_AGENT,
            RunPhase.CURATING: AgentRole.DATA_CURATOR,
            RunPhase.DESIGNING: AgentRole.TRAINING_DESIGNER,
            RunPhase.TRAINING: AgentRole.TRAINING_EXECUTOR,
            RunPhase.EVALUATING: AgentRole.EVALUATION_AGENT,
            RunPhase.PROMOTING: AgentRole.CHAMPION_MANAGER,
        }
        role = roles[state.phase]
        try:
            async with async_telemetry_span(
                "agent.run",
                attributes={
                    "run_id": state.run_id,
                    "phase": state.phase,
                    "agent_name": role.value,
                },
            ):
                return await handler(state)
        except Exception as exc:
            failed = await self._save(state, phase=RunPhase.FAILED)
            await self._event(
                failed,
                "run.failed",
                None,
                {
                    "error_type": type(exc).__name__,
                    "message": "specialist operation failed",
                },
            )
            raise

    async def auto(self, run_id: str) -> RunState:
        """Advance until completion, cancellation, or failure with a hard safety bound."""

        state = await self._require_run(run_id)
        # One benchmark plus seven remaining roles for each of at most two candidates.
        for _ in range(16):
            if state.phase in TERMINAL_PHASES:
                return state
            state = await self.step(run_id)
        raise InvalidRunStateError("automatic run exceeded the bounded state-machine step count")

    async def cancel(self, run_id: str) -> RunState:
        """Idempotently cancel a run; a terminal completed run is never rewritten."""

        state = await self._require_run(run_id)
        if state.phase in TERMINAL_PHASES:
            return state
        cancelled = await self._save(state, phase=RunPhase.CANCELLED, cancel_requested=True)
        await self._event(cancelled, "run.cancelled", None, {})
        return cancelled

    async def _benchmark(self, state: RunState) -> RunState:
        trajectories = await self.benchmark_runner.run(state)
        self._trajectories[state.run_id] = trajectories
        persisted_evidence: dict[str, Any] = {}
        evidence_reader = getattr(self.provider, "benchmark_evidence", None)
        if callable(evidence_reader):
            artifact, trajectory_ids = evidence_reader(state.run_id)
            persisted_evidence = {
                "benchmark_artifact": artifact,
                "benchmark_trajectory_ids": trajectory_ids,
            }
        updated = await self._save(
            state,
            phase=RunPhase.ANALYZING,
            **persisted_evidence,
        )
        await self._event(
            updated,
            "benchmark.completed",
            AgentRole.BENCHMARK_RUNNER,
            {"trajectory_count": len(trajectories)},
        )
        return updated

    async def _analyze(self, state: RunState) -> RunState:
        trajectories = self._trajectories.get(state.run_id, [])
        clusters = await self.failure_analyst.run(state, trajectories)
        updated = await self._save(state, phase=RunPhase.RESEARCHING, failure_clusters=clusters)
        await self._event(
            updated,
            "failure_report.completed",
            AgentRole.FAILURE_ANALYST,
            {"cluster_count": len(clusters)},
        )
        return updated

    async def _research(self, state: RunState) -> RunState:
        hypothesis = await self.research_agent.run(state)
        updated = await self._save(
            state, phase=RunPhase.CURATING, current_hypothesis=hypothesis, dataset=None
        )
        await self._event(
            updated,
            "hypothesis.created",
            AgentRole.RESEARCH_AGENT,
            {"hypothesis_id": hypothesis.hypothesis_id},
        )
        return updated

    async def _curate(self, state: RunState) -> RunState:
        dataset = await self.data_curator.run(state)
        updated = await self._save(state, phase=RunPhase.DESIGNING, dataset=dataset)
        await self._event(
            updated,
            "dataset.verified",
            AgentRole.DATA_CURATOR,
            {"dataset_id": dataset.dataset_id, "example_count": dataset.example_count},
        )
        return updated

    async def _design(self, state: RunState) -> RunState:
        if state.experiments_used >= state.max_experiments:
            return await self._set_terminal(state, RunPhase.COMPLETED, "budget.exhausted")
        if state.current_hypothesis is None or state.dataset is None:
            raise InvalidRunStateError("training design requires a hypothesis and verified dataset")
        config = await self.training_designer.run(state)
        experiment = Experiment.model_validate(
            {
                "experiment_id": f"EXP-{uuid4().hex[:10]}",
                "run_id": state.run_id,
                "hypothesis": state.current_hypothesis,
                "config": config,
                "dataset": state.dataset,
                "status": ExperimentStatus.DESIGNED,
            }
        )
        updated = await self._save(
            state,
            phase=RunPhase.TRAINING,
            experiments=[*state.experiments, experiment],
            experiments_used=state.experiments_used + 1,
        )
        await self._event(
            updated,
            "experiment.designed",
            AgentRole.TRAINING_DESIGNER,
            {"experiment_id": experiment.experiment_id, "config": config.model_dump()},
        )
        return updated

    async def _train(self, state: RunState) -> RunState:
        experiment = self._current_experiment(state)
        result = await self.training_executor.run(state, experiment)
        if result.status is not JobStatus.SUCCEEDED:
            raise InvalidRunStateError(
                f"training job {result.job_id} ended with status {result.status.value}"
            )
        revised = experiment.model_copy(
            update={"status": ExperimentStatus.EVALUATING, "training_result": result}
        )
        updated = await self._save_replaced_experiment(state, revised, phase=RunPhase.EVALUATING)
        await self._event(
            updated,
            "training.completed",
            AgentRole.TRAINING_EXECUTOR,
            {"experiment_id": revised.experiment_id, "job_id": result.job_id},
        )
        return updated

    async def _evaluate(self, state: RunState) -> RunState:
        experiment = self._current_experiment(state)
        report = await self.evaluation_agent.run(state, experiment)
        revised = experiment.model_copy(update={"evaluation": report})
        updated = await self._save_replaced_experiment(state, revised, phase=RunPhase.PROMOTING)
        await self._event(
            updated,
            "evaluation.completed",
            AgentRole.EVALUATION_AGENT,
            {"experiment_id": revised.experiment_id},
        )
        return updated

    async def _promote(self, state: RunState) -> RunState:
        experiment = self._current_experiment(state)
        if experiment.evaluation is None:
            raise InvalidRunStateError("promotion requires an evaluation report")
        decision = self.champion_manager.run(
            experiment.evaluation,
            provenance_complete=_provenance_complete(experiment),
        )
        status = ExperimentStatus.PROMOTED if decision.promoted else ExperimentStatus.REJECTED
        revised = experiment.model_copy(update={"status": status, "promotion": decision})
        done = decision.promoted or state.experiments_used >= state.max_experiments
        next_phase = RunPhase.COMPLETED if done else RunPhase.RESEARCHING
        changes: dict[str, Any] = {"phase": next_phase}
        checkpoint = getattr(experiment.training_result, "checkpoint", None)
        if decision.promoted and checkpoint is not None:
            changes["champion"] = CheckpointManifest(
                version=experiment.experiment_id,
                model_uri=checkpoint.uri,
                success=experiment.evaluation.candidate_success,
                regression_success=experiment.evaluation.candidate_regression_success,
                action_validity=experiment.evaluation.candidate_action_validity,
                artifact=checkpoint,
                evidence_label=experiment.evaluation.evidence_label,
            )
        updated = await self._save_replaced_experiment(state, revised, **changes)
        await self._event(
            updated,
            "checkpoint.promoted" if decision.promoted else "checkpoint.rejected",
            AgentRole.CHAMPION_MANAGER,
            {"experiment_id": revised.experiment_id, "reasons": decision.reasons},
        )
        return updated

    async def _require_run(self, run_id: str) -> RunState:
        return await self.repository.get_run(run_id)

    async def _set_terminal(self, state: RunState, phase: RunPhase, event_type: str) -> RunState:
        updated = await self._save(state, phase=phase)
        await self._event(updated, event_type, None, {})
        return updated

    async def _event(
        self,
        state: RunState,
        event_type: str,
        agent: AgentRole | None,
        payload: dict[str, Any],
    ) -> None:
        event = RunEvent.model_validate(
            {
                "event_id": f"EVT-{uuid4().hex[:12]}",
                "run_id": state.run_id,
                "type": event_type,
                "phase": state.phase,
                "agent": agent,
                "payload": payload,
                "trace_id": current_trace_id(),
            }
        )
        await self.repository.append_event(event)

    async def _save(self, state: RunState, **changes: Any) -> RunState:
        candidate = state.model_copy(update={**changes, "updated_at": datetime.now(UTC)}, deep=True)
        return await self.repository.save_run(candidate, expected_version=state.version)

    async def _save_replaced_experiment(
        self, state: RunState, experiment: Experiment, **changes: Any
    ) -> RunState:
        experiments = [
            experiment if item.experiment_id == experiment.experiment_id else item
            for item in state.experiments
        ]
        return await self._save(state, experiments=experiments, **changes)

    @staticmethod
    def _current_experiment(state: RunState) -> Experiment:
        if not state.experiments:
            raise InvalidRunStateError("run has no current experiment")
        return state.experiments[-1]


def _provenance_complete(experiment: Experiment) -> bool:
    training = experiment.training_result
    evaluation = experiment.evaluation
    if training is None or evaluation is None:
        return False
    return bool(
        training.job_id
        and training.checkpoint
        and training.logs
        and experiment.dataset.artifact.sha256
        and evaluation.artifact
        and evaluation.provenance_complete
        and evaluation.evidence_label is not EvidenceLabel.EXPLANATION
    )
