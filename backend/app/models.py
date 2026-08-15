"""Typed domain contracts for autonomous post-training runs.

The models in this module deliberately contain no Google SDK dependencies.  They
are the wire and persistence contracts shared by the coordinator, the research
team, and the execution team.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp for persisted records."""

    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    """Create a readable, collision-resistant identifier."""

    return f"{prefix}_{uuid4().hex}"


class DomainModel(BaseModel):
    """Strict base model used by all public domain contracts."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RunPhase(StrEnum):
    NOT_STARTED = "NOT_STARTED"
    BENCHMARKING = "BENCHMARKING"
    ANALYZING = "ANALYZING"
    RESEARCHING = "RESEARCHING"
    CURATING = "CURATING"
    DESIGNING = "DESIGNING"
    TRAINING = "TRAINING"
    EVALUATING = "EVALUATING"
    PROMOTING = "PROMOTING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class AgentRole(StrEnum):
    BENCHMARK_RUNNER = "benchmark_runner"
    FAILURE_ANALYST = "failure_analyst"
    RESEARCH_AGENT = "research_agent"
    DATA_CURATOR = "data_curator"
    TRAINING_DESIGNER = "training_designer"
    TRAINING_EXECUTOR = "training_executor"
    EVALUATION_AGENT = "evaluation_agent"
    CHAMPION_MANAGER = "champion_manager"


class DatasetSplit(StrEnum):
    TRAIN = "train"
    HELDOUT = "heldout"
    REGRESSION = "regression"


class EvidenceLabel(StrEnum):
    LIVE = "LIVE"
    PRIOR_VERIFIED_RUN = "PRIOR_VERIFIED_RUN"
    EXPLANATION = "EXPLANATION"


class ArtifactKind(StrEnum):
    TRAJECTORIES = "trajectories"
    FAILURE_REPORT = "failure_report"
    DATASET = "dataset"
    CHECKPOINT = "checkpoint"
    TRAINING_LOG = "training_log"
    EVALUATION_REPORT = "evaluation_report"
    PROMOTION_DECISION = "promotion_decision"


class ArtifactRef(DomainModel):
    artifact_id: str = Field(default_factory=lambda: new_id("artifact"))
    kind: ArtifactKind
    uri: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(ge=0)
    content_type: str = "application/json"
    created_at: datetime = Field(default_factory=utc_now)


class ToolCall(DomainModel):
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class TrajectoryStep(DomainModel):
    index: int = Field(ge=0)
    observation: str = ""
    action: ToolCall | None = None
    reward: float | None = None
    valid: bool = True
    error_code: str | None = None


class Trajectory(DomainModel):
    trajectory_id: str = Field(default_factory=lambda: new_id("trajectory"))
    task_id: str = Field(min_length=1)
    split: DatasetSplit
    model_version: str = Field(min_length=1)
    steps: list[TrajectoryStep] = Field(default_factory=list)
    reward: float = Field(ge=0.0, le=1.0)
    success: bool
    created_at: datetime = Field(default_factory=utc_now)


class FailureCluster(DomainModel):
    cluster_id: str = Field(default_factory=lambda: new_id("cluster"))
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)
    trajectory_ids: list[str] = Field(min_length=1)
    frequency: int = Field(ge=1)


class Citation(DomainModel):
    document_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_uri: str | None = None
    excerpt: str = Field(min_length=1, max_length=1200)
    score: float = Field(ge=0.0)


class Hypothesis(DomainModel):
    hypothesis_id: str = Field(default_factory=lambda: new_id("hypothesis"))
    failure_cluster_id: str
    statement: str = Field(min_length=1)
    expected_improvement: str = Field(min_length=1)
    data_strategy: str = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)


class SFTExample(DomainModel):
    example_id: str = Field(default_factory=lambda: new_id("sft"))
    source_trajectory_id: str
    source_step_index: int = Field(ge=0)
    source_split: DatasetSplit = DatasetSplit.TRAIN
    observation: str
    target_action: ToolCall
    verified: bool
    verifier_reward_before: float = Field(ge=0.0, le=1.0)
    verifier_reward_after: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def require_verified_train_improvement(self) -> SFTExample:
        if self.source_split is not DatasetSplit.TRAIN:
            raise ValueError("SFT examples may only originate from the train split")
        if not self.verified:
            raise ValueError("unverified corrections cannot enter an SFT dataset")
        if self.verifier_reward_after <= self.verifier_reward_before:
            raise ValueError("a repaired example must improve verifier reward")
        return self


class DatasetManifest(DomainModel):
    dataset_id: str = Field(default_factory=lambda: new_id("dataset"))
    artifact: ArtifactRef
    example_count: int = Field(gt=0)
    source_splits: set[DatasetSplit] = Field(default_factory=lambda: {DatasetSplit.TRAIN})
    schema_version: str = "1.0"
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def reject_eval_data(self) -> DatasetManifest:
        if self.source_splits != {DatasetSplit.TRAIN}:
            raise ValueError("training datasets must contain train split data only")
        if self.artifact.kind is not ArtifactKind.DATASET:
            raise ValueError("dataset manifest must reference a dataset artifact")
        return self


class QLoRAConfig(DomainModel):
    rank: Literal[8, 16, 32]
    learning_rate: float
    epochs: Literal[2, 3, 5]
    dropout: float
    max_sequence_length: int = Field(default=512, ge=128, le=2048)
    effective_batch_size: int = Field(default=32, ge=1, le=128)

    @field_validator("learning_rate")
    @classmethod
    def allowed_learning_rate(cls, value: float) -> float:
        if value not in {5e-5, 1e-4, 2e-4}:
            raise ValueError("learning rate is outside the bounded QLoRA search space")
        return value

    @field_validator("dropout")
    @classmethod
    def allowed_dropout(cls, value: float) -> float:
        if value not in {0.0, 0.05}:
            raise ValueError("dropout is outside the bounded QLoRA search space")
        return value


class JobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TrainingResult(DomainModel):
    job_id: str
    status: JobStatus
    checkpoint: ArtifactRef | None = None
    logs: ArtifactRef | None = None
    duration_seconds: float | None = Field(default=None, ge=0.0)
    error_code: str | None = None

    @model_validator(mode="after")
    def successful_job_has_checkpoint(self) -> TrainingResult:
        if self.status is JobStatus.SUCCEEDED:
            if self.checkpoint is None or self.checkpoint.kind is not ArtifactKind.CHECKPOINT:
                raise ValueError("successful training must produce a checkpoint artifact")
        return self


class EvaluationReport(DomainModel):
    report_id: str = Field(default_factory=lambda: new_id("evaluation"))
    artifact: ArtifactRef | None = None
    task_count: int = Field(gt=0)
    champion_success: float = Field(ge=0.0, le=1.0)
    candidate_success: float = Field(ge=0.0, le=1.0)
    champion_regression_success: float = Field(ge=0.0, le=1.0)
    candidate_regression_success: float = Field(ge=0.0, le=1.0)
    champion_action_validity: float = Field(ge=0.0, le=1.0)
    candidate_action_validity: float = Field(ge=0.0, le=1.0)
    paired_improvement_positive: bool
    provenance_complete: bool
    evidence_label: EvidenceLabel

    @property
    def success_delta(self) -> float:
        return self.candidate_success - self.champion_success

    @property
    def regression_delta(self) -> float:
        return self.candidate_regression_success - self.champion_regression_success


class PromotionDecision(DomainModel):
    decision_id: str = Field(default_factory=lambda: new_id("promotion"))
    promoted: bool
    reasons: list[str] = Field(min_length=1)
    success_delta: float
    regression_delta: float
    decided_at: datetime = Field(default_factory=utc_now)


class CheckpointManifest(DomainModel):
    version: str = "gemma-base"
    model_uri: str = "google/functiongemma-270m-it"
    success: float = Field(default=0.0, ge=0.0, le=1.0)
    regression_success: float = Field(default=1.0, ge=0.0, le=1.0)
    action_validity: float = Field(default=1.0, ge=0.0, le=1.0)
    artifact: ArtifactRef | None = None
    evidence_label: EvidenceLabel = EvidenceLabel.EXPLANATION


class ExperimentStatus(StrEnum):
    DESIGNED = "DESIGNED"
    TRAINING = "TRAINING"
    EVALUATING = "EVALUATING"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class Experiment(DomainModel):
    experiment_id: str = Field(default_factory=lambda: new_id("experiment"))
    run_id: str
    hypothesis: Hypothesis
    config: QLoRAConfig
    dataset: DatasetManifest
    status: ExperimentStatus = ExperimentStatus.DESIGNED
    training_result: TrainingResult | None = None
    evaluation: EvaluationReport | None = None
    promotion: PromotionDecision | None = None
    created_at: datetime = Field(default_factory=utc_now)


class RunState(DomainModel):
    run_id: str = Field(default_factory=lambda: new_id("run"))
    target_model: str = "google/functiongemma-270m-it"
    environment: str = "AgentGym/WebShop"
    phase: RunPhase = RunPhase.NOT_STARTED
    champion: CheckpointManifest = Field(default_factory=CheckpointManifest)
    benchmark_artifact: ArtifactRef | None = None
    benchmark_trajectory_ids: list[str] = Field(default_factory=list)
    failure_clusters: list[FailureCluster] = Field(default_factory=list)
    current_hypothesis: Hypothesis | None = None
    dataset: DatasetManifest | None = None
    experiments: list[Experiment] = Field(default_factory=list)
    experiments_used: int = Field(default=0, ge=0, le=2)
    max_experiments: Literal[1, 2] = 2
    human_tuning_decisions: int = Field(default=0, ge=0)
    cancel_requested: bool = False
    version: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def enforce_experiment_budget(self) -> RunState:
        if self.experiments_used > self.max_experiments:
            raise ValueError("experiment budget exceeded")
        if len(self.experiments) > self.max_experiments:
            raise ValueError("too many candidate experiments")
        return self


class RunEvent(DomainModel):
    event_id: str = Field(default_factory=lambda: new_id("event"))
    run_id: str
    type: str = Field(min_length=1)
    phase: RunPhase
    agent: AgentRole | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
