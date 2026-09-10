"""Leakage-safe contracts for the service-recovery objective worker.

Only observations an agent could receive from the environment are represented
by the public models.  The engine keeps task failure definitions in private
state, and the sealed evaluator is the only caller allowed to open the hidden
split.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ALLOWED_TOOLS: tuple[str, ...] = (
    "get_logs",
    "inspect_service",
    "read_config",
    "edit_config",
    "restart_service",
    "run_healthcheck",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def deterministic_dataset_created_at(dataset_id: str, sha256: str) -> datetime:
    """Return restart-stable metadata for a content-addressed dataset."""

    digest = hashlib.sha256(f"{dataset_id}:{sha256}".encode()).hexdigest()
    return datetime(2026, 1, 1, tzinfo=UTC) + timedelta(
        seconds=int(digest[:12], 16) % (100 * 365 * 24 * 60 * 60)
    )


class ObjectiveSplit(StrEnum):
    TRAIN = "train"
    REPLAY = "replay"
    VALIDATION = "validation"
    HIDDEN = "hidden"


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class ToolCall(ContractModel):
    tool: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tool")
    @classmethod
    def validate_tool(cls, value: str) -> str:
        if value not in ALLOWED_TOOLS:
            raise ValueError(f"tool {value!r} is not allowed")
        return value


class Task(ContractModel):
    """Sanitized task metadata safe to send to a model or coordinator."""

    task_id: str = Field(min_length=1)
    split: ObjectiveSplit
    service_name: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    allowed_tools: tuple[str, ...] = ALLOWED_TOOLS
    max_steps: int = Field(default=10, ge=1)
    engine_version: str = Field(min_length=1)
    seed: int

    @field_validator("allowed_tools")
    @classmethod
    def validate_allowed_tools(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != ALLOWED_TOOLS:
            raise ValueError("task tools must equal the objective allow-list")
        return value

    @property
    def tools(self) -> tuple[str, ...]:
        """Compatibility spelling used by AgentGym-style callers."""

        return self.allowed_tools


class StepResult(ContractModel):
    call: ToolCall
    success: bool
    reward: float
    done: bool
    step: int = Field(ge=1)
    observation: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class Trajectory(ContractModel):
    trajectory_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    split: ObjectiveSplit
    engine_version: str = Field(min_length=1)
    seed: int
    steps: tuple[StepResult, ...] = Field(default_factory=tuple)
    total_reward: float
    success: bool
    done: bool
    verified: bool = False

    @model_validator(mode="after")
    def validate_steps(self) -> Trajectory:
        if any(step.step != index for index, step in enumerate(self.steps, start=1)):
            raise ValueError("trajectory steps must be ordered from one")
        if self.split is ObjectiveSplit.HIDDEN:
            raise ValueError("hidden trajectories cannot cross the objective boundary")
        return self


class BenchmarkExecutionResult(ContractModel):
    """Typed output returned by the real benchmark execution adapter."""

    trajectories: tuple[Trajectory, ...] = Field(default_factory=tuple)


class ReplayResult(ContractModel):
    trajectory: Trajectory
    verified: bool
    replayed_reward: float
    replayed_success: bool
    reason: str = Field(min_length=1)


class DatasetRow(ContractModel):
    source_trajectory_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    split: ObjectiveSplit
    messages: tuple[dict[str, Any], ...] = Field(min_length=1)
    failure_label: str = Field(min_length=1)
    verifier_confirmed: bool
    source_type: str = Field(min_length=1)

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


class DatasetManifest(ContractModel):
    dataset_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    row_count: int = Field(ge=0)
    sha256: str
    s3_uri: str = Field(min_length=1)
    created_at: datetime
    source_trajectory_ids: tuple[str, ...] = Field(default_factory=tuple)
    target_failure_classes: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("sha256 must be a lowercase 64-character digest")
        return value

    @field_validator("created_at")
    @classmethod
    def require_aware_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value


class Dataset(ContractModel):
    manifest: DatasetManifest
    rows: tuple[DatasetRow, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def row_count_matches(self) -> Dataset:
        if self.manifest.row_count != len(self.rows):
            raise ValueError("dataset manifest row_count does not match rows")
        payload = "\n".join(row.canonical_json() for row in self.rows)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        if digest != self.manifest.sha256:
            raise ValueError("dataset manifest digest does not match rows")
        return self


class BenchmarkRequest(ContractModel):
    run_id: str = Field(min_length=1)
    split: ObjectiveSplit = ObjectiveSplit.TRAIN
    task_ids: tuple[str, ...] = Field(default=("train-001",), min_length=1, max_length=100)

    @field_validator("split")
    @classmethod
    def train_scope_only(cls, value: ObjectiveSplit) -> ObjectiveSplit:
        if value not in {ObjectiveSplit.TRAIN, ObjectiveSplit.REPLAY}:
            raise ValueError(f"split {value.value!r} is outside the train/replay scope")
        return value

    @field_validator("task_ids")
    @classmethod
    def require_unique_task_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("task_ids must be unique")
        return value


class TrajectoryReference(ContractModel):
    trajectory_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    split: ObjectiveSplit
    verified: bool


class BenchmarkResponse(ContractModel):
    benchmark_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    split: ObjectiveSplit
    total_tasks: int = Field(ge=0)
    successful_tasks: int = Field(ge=0)
    success_rate: float = Field(ge=0, le=1)
    trajectory_references: tuple[TrajectoryReference, ...] = Field(default_factory=tuple)


class CurationRequest(ContractModel):
    run_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    split: ObjectiveSplit = ObjectiveSplit.REPLAY
    trajectories: tuple[Trajectory, ...] = Field(default_factory=tuple, max_length=100)
    trajectory_references: tuple[TrajectoryReference, ...] = Field(
        default_factory=tuple, max_length=100
    )

    @field_validator("split")
    @classmethod
    def replay_scope_only(cls, value: ObjectiveSplit) -> ObjectiveSplit:
        if value is not ObjectiveSplit.REPLAY:
            raise ValueError(f"split {value.value!r} is outside replay scope")
        return value

    @model_validator(mode="after")
    def require_trajectory_input(self) -> CurationRequest:
        if not self.trajectories and not self.trajectory_references:
            raise ValueError("trajectories or trajectory_references is required")
        return self


# The wire response is the dataset itself so workers can pass the manifest to
# the trainer without an extra envelope.  Keep a descriptive alias for callers
# that import a response contract.
CurationResponse = Dataset

# Descriptive aliases keep integrations from having to duplicate the wire
# contracts while preserving one canonical serialized shape.
Split = ObjectiveSplit
TaskSpec = Task
TrajectoryStep = StepResult
ReplayVerification = ReplayResult
SFTDataset = Dataset
