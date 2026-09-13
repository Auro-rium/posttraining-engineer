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
from typing import Any, Literal
from urllib.parse import parse_qs, unquote, urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.posttraining.models import ArtifactReference
from app.posttraining.models import EvidenceLabel as EvidenceLabel
from app.posttraining.run_history import BenchmarkMetrics

ALLOWED_TOOLS: tuple[str, ...] = (
    "get_logs",
    "inspect_service",
    "read_config",
    "edit_config",
    "restart_service",
    "run_healthcheck",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TRAJECTORY_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_TRAJECTORY_REFERENCE_HANDLE = re.compile(
    r"^trajectory://(train|replay|validation)/"
    r"([A-Za-z0-9][A-Za-z0-9_.:-]{0,127})/"
    r"([A-Za-z0-9][A-Za-z0-9_.:-]{0,127})/verified$"
)


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
    verifier_success: bool | None = None
    repaired_from_trajectory_id: str | None = None

    @model_validator(mode="after")
    def validate_steps(self) -> Trajectory:
        if any(step.step != index for index, step in enumerate(self.steps, start=1)):
            raise ValueError("trajectory steps must be ordered from one")
        if self.split is ObjectiveSplit.HIDDEN:
            raise ValueError("hidden trajectories cannot cross the objective boundary")
        if self.repaired_from_trajectory_id is not None:
            if (
                not _TRAJECTORY_IDENTIFIER.fullmatch(self.repaired_from_trajectory_id)
                or self.repaired_from_trajectory_id.lower() == "unknown"
            ):
                raise ValueError("repair source must be a path-safe trajectory identifier")
            if self.repaired_from_trajectory_id == self.trajectory_id:
                raise ValueError("a trajectory cannot repair itself")
        return self


class BenchmarkExecutionResult(ContractModel):
    """Typed output returned by the real benchmark execution adapter."""

    trajectories: tuple[Trajectory, ...] = Field(default_factory=tuple)


class ObjectiveWorkerCapabilities(ContractModel):
    benchmark: bool
    verify_curation: bool = Field(alias="verify-curation")


class ObjectiveReadinessResponse(ContractModel):
    """Authenticated, non-invasive execution readiness attestation."""

    status: Literal["ready", "blocked"]
    service: Literal["objective-worker"] = "objective-worker"
    model_id: Literal["google/functiongemma-270m-it"] = "google/functiongemma-270m-it"
    configuration_ready: bool
    checkpoint_ready: bool
    model_load_ready: bool
    generation_ready: bool
    artifact_store_ready: bool
    execution_ready: bool
    capabilities: ObjectiveWorkerCapabilities
    blockers: tuple[
        Literal[
            "functiongemma_adapter_unavailable",
            "checkpoint_unverified",
            "inference_runtime_unavailable",
            "artifact_store_incomplete",
            "model_identity_unavailable",
            "model_load_not_verified",
            "generation_not_verified",
        ],
        ...,
    ] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def status_matches_capabilities(self) -> ObjectiveReadinessResponse:
        observed_execution_ready = all(
            (
                self.configuration_ready,
                self.checkpoint_ready,
                self.model_load_ready,
                self.generation_ready,
                self.artifact_store_ready,
            )
        )
        if self.execution_ready != observed_execution_ready:
            raise ValueError("execution readiness must match its component attestations")
        capabilities_ready = self.capabilities.benchmark and self.capabilities.verify_curation
        if self.status == "ready" and (
            not capabilities_ready or self.blockers or not self.execution_ready
        ):
            raise ValueError("ready status requires both capabilities and no blockers")
        if self.status == "blocked" and (
            capabilities_ready and not self.blockers and self.execution_ready
        ):
            raise ValueError("blocked status requires a missing capability or blocker")
        if (not capabilities_ready or not self.execution_ready) and not self.blockers:
            raise ValueError("blocked readiness requires a blocker")
        return self


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
    verifier_success: bool
    repaired_from_trajectory_id: str | None = None
    source_type: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_successful_verifier_target(self) -> DatasetRow:
        if not self.verifier_confirmed or not self.verifier_success:
            raise ValueError("SFT rows require verifier-confirmed successful outcomes")
        if (self.source_type == "repaired_replay") != (
            self.repaired_from_trajectory_id is not None
        ):
            raise ValueError("repair lineage must match the dataset row source type")
        if self.repaired_from_trajectory_id is not None and (
            not _TRAJECTORY_IDENTIFIER.fullmatch(self.repaired_from_trajectory_id)
            or self.repaired_from_trajectory_id.lower() == "unknown"
        ):
            raise ValueError("repair source must be a path-safe trajectory identifier")
        return self

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
    """Wire contract accepted from the autonomous coordinator.

    ``task_ids`` remains an internal/legacy test hook. Live callers specify an
    episode count; the worker derives stable task IDs from the immutable run
    scope so a retry measures the same tasks.
    """

    run_id: str = Field(min_length=1)
    model_uri: str = Field(min_length=1)
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    suite: str = Field(default="service-recovery", min_length=1)
    suite_version: str = Field(default="service-recovery-v1", min_length=1)
    seed: int | None = None
    num_episodes: int | None = Field(default=None, ge=1, le=100)
    split: ObjectiveSplit = ObjectiveSplit.TRAIN
    task_ids: tuple[str, ...] | None = Field(default=None, min_length=1, max_length=100)
    output_s3_uri: str | None = None

    @field_validator("model_uri")
    @classmethod
    def require_immutable_model_uri(cls, value: str) -> str:
        parsed = urlparse(value)
        versions = parse_qs(parsed.query).get("versionId", [])
        if (
            parsed.scheme != "s3"
            or not parsed.netloc
            or not parsed.path
            or len(versions) != 1
            or not versions[0]
            or versions[0].strip().lower() == "null"
        ):
            raise ValueError("model_uri must be an immutable S3 object version")
        return value

    @field_validator("split")
    @classmethod
    def train_scope_only(cls, value: ObjectiveSplit) -> ObjectiveSplit:
        if value not in {ObjectiveSplit.TRAIN, ObjectiveSplit.REPLAY}:
            raise ValueError(f"split {value.value!r} is outside the train/replay scope")
        return value

    @field_validator("output_s3_uri")
    @classmethod
    def require_s3_output_scope(cls, value: str | None) -> str | None:
        if value is None:
            return value
        parsed = urlparse(value)
        if (
            parsed.scheme != "s3"
            or not parsed.netloc
            or not parsed.path.strip("/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("output_s3_uri must be an S3 prefix without query or fragment")
        path = unquote(parsed.path.lstrip("/"))
        if any(not item or item in {".", ".."} for item in path.split("/")):
            raise ValueError("output_s3_uri contains an unsafe path")
        return value

    @field_validator("task_ids")
    @classmethod
    def require_unique_task_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value is not None and len(set(value)) != len(value):
            raise ValueError("task_ids must be unique")
        return value

    @model_validator(mode="after")
    def resolve_episode_scope(self) -> BenchmarkRequest:
        if self.task_ids is None and self.num_episodes is None:
            raise ValueError("num_episodes is required")
        if self.task_ids is not None:
            if self.num_episodes is not None and len(self.task_ids) != self.num_episodes:
                raise ValueError("task_ids count must match num_episodes")
            object.__setattr__(self, "num_episodes", len(self.task_ids))
            return self
        if self.seed is None:
            raise ValueError("seed is required when task_ids are derived from num_episodes")

        digest = hashlib.sha256(
            f"{self.run_id}:{self.suite}:{self.suite_version}:{self.seed}:{self.split.value}".encode()
        ).hexdigest()[:16]
        object.__setattr__(
            self,
            "task_ids",
            tuple(
                f"{self.split.value}-{digest}-{index:03d}"
                for index in range(1, int(self.num_episodes or 0) + 1)
            ),
        )
        return self

    @property
    def execution_task_ids(self) -> tuple[str, ...]:
        """Stable task IDs consumed by the execution adapter."""

        return self.task_ids or ()


class TrajectoryReference(ContractModel):
    trajectory_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    split: ObjectiveSplit
    verified: bool

    @field_validator("trajectory_id", "task_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if not _TRAJECTORY_IDENTIFIER.fullmatch(value) or value.lower() == "unknown":
            raise ValueError("trajectory reference identifiers must be path-safe")
        return value

    @model_validator(mode="after")
    def public_reference_only(self) -> TrajectoryReference:
        if self.split is ObjectiveSplit.HIDDEN:
            raise ValueError("hidden trajectory references cannot cross the objective boundary")
        return self


class CorrectionProposal(ContractModel):
    """Untrusted action proposal tied to one public failed trajectory."""

    source_trajectory_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    split: ObjectiveSplit
    actions: tuple[ToolCall, ...] = Field(min_length=1, max_length=10)

    @field_validator("source_trajectory_id")
    @classmethod
    def require_stored_trajectory_identifier(cls, value: str) -> str:
        if not _TRAJECTORY_IDENTIFIER.fullmatch(value) or value.lower() == "unknown":
            raise ValueError("correction source must be a path-safe trajectory identifier")
        return value

    @field_validator("split")
    @classmethod
    def correction_scope_only(cls, value: ObjectiveSplit) -> ObjectiveSplit:
        if value not in {ObjectiveSplit.TRAIN, ObjectiveSplit.REPLAY}:
            raise ValueError("correction proposal scope is limited to train and replay")
        return value

    @model_validator(mode="after")
    def action_arguments_are_allowlisted(self) -> CorrectionProposal:
        allowed_arguments = {
            "get_logs": {"service"},
            "inspect_service": {"service"},
            "read_config": {"service"},
            "edit_config": {"service", "content"},
            "restart_service": {"service"},
            "run_healthcheck": {"service"},
        }
        for action in self.actions:
            if set(action.arguments).difference(allowed_arguments[action.tool]):
                raise ValueError("correction action contains unrecognized arguments")
            if any(not isinstance(value, str) for value in action.arguments.values()):
                raise ValueError("correction action arguments must be strings")
        return self

    @property
    def proposal_id(self) -> str:
        """Stable, content-derived ID used only to correlate a replay response."""

        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return "correction-" + hashlib.sha256(payload.encode()).hexdigest()[:24]


class CorrectionReplayRequest(ContractModel):
    run_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    split: ObjectiveSplit
    proposals: tuple[CorrectionProposal, ...] = Field(min_length=1, max_length=100)

    @field_validator("split")
    @classmethod
    def replay_scope_only(cls, value: ObjectiveSplit) -> ObjectiveSplit:
        if value not in {ObjectiveSplit.TRAIN, ObjectiveSplit.REPLAY}:
            raise ValueError("correction replay scope is limited to train and replay")
        return value

    @model_validator(mode="after")
    def proposals_match_scope_and_are_unique(self) -> CorrectionReplayRequest:
        if any(proposal.split is not self.split for proposal in self.proposals):
            raise ValueError("correction proposal split does not match replay scope")
        if len({proposal.proposal_id for proposal in self.proposals}) != len(self.proposals):
            raise ValueError("correction proposals must be unique")
        return self


class CorrectionReplayOutcome(ContractModel):
    proposal_id: str = Field(min_length=1)
    source_trajectory_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    split: ObjectiveSplit
    status: Literal["PASS", "REJECTED"]
    reason: Literal["replay_passed", "replay_failed", "invalid_correction"]
    trajectory_reference: TrajectoryReference | None = None

    @field_validator("split")
    @classmethod
    def outcome_scope_only(cls, value: ObjectiveSplit) -> ObjectiveSplit:
        if value not in {ObjectiveSplit.TRAIN, ObjectiveSplit.REPLAY}:
            raise ValueError("correction replay outcome scope is limited to train and replay")
        return value

    @model_validator(mode="after")
    def pass_requires_verified_reference(self) -> CorrectionReplayOutcome:
        reference = self.trajectory_reference
        if self.status == "PASS":
            if self.reason != "replay_passed" or reference is None:
                raise ValueError("passed correction requires successful replay provenance")
            if (
                not reference.verified
                or reference.trajectory_id == self.source_trajectory_id
                or reference.task_id != self.task_id
                or reference.split is not self.split
            ):
                raise ValueError("passed correction reference does not match replay provenance")
        elif self.reason == "replay_passed" or reference is not None:
            raise ValueError("rejected correction cannot expose an admitted trajectory")
        return self


class CorrectionReplayResponse(ContractModel):
    run_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    split: ObjectiveSplit
    outcomes: tuple[CorrectionReplayOutcome, ...]

    @field_validator("split")
    @classmethod
    def response_scope_only(cls, value: ObjectiveSplit) -> ObjectiveSplit:
        if value not in {ObjectiveSplit.TRAIN, ObjectiveSplit.REPLAY}:
            raise ValueError("correction replay response scope is limited to train and replay")
        return value

    @model_validator(mode="after")
    def outcomes_match_scope(self) -> CorrectionReplayResponse:
        if any(outcome.split is not self.split for outcome in self.outcomes):
            raise ValueError("correction replay outcome split does not match request")
        if len({outcome.proposal_id for outcome in self.outcomes}) != len(self.outcomes):
            raise ValueError("correction replay outcomes must have unique proposal IDs")
        return self


def encode_trajectory_reference(reference: TrajectoryReference) -> str:
    """Encode actual safe trajectory provenance in a restart-stable handoff ID."""

    if not reference.verified:
        raise ValueError("only verifier-confirmed trajectory references may be handed off")
    if reference.split is ObjectiveSplit.HIDDEN:
        raise ValueError("hidden trajectory references cannot cross the objective boundary")
    return (
        f"trajectory://{reference.split.value}/{reference.trajectory_id}/"
        f"{reference.task_id}/verified"
    )


def decode_trajectory_reference(value: str) -> TrajectoryReference:
    """Decode only the canonical, public reference format persisted by the supervisor."""

    if not isinstance(value, str):
        raise ValueError("trajectory handoff reference must be a string")
    match = _TRAJECTORY_REFERENCE_HANDLE.fullmatch(value)
    if match is None:
        raise ValueError("trajectory handoff reference is malformed or not verifier-confirmed")
    split, trajectory_id, task_id = match.groups()
    reference = TrajectoryReference(
        trajectory_id=trajectory_id,
        task_id=task_id,
        split=ObjectiveSplit(split),
        verified=True,
    )
    if encode_trajectory_reference(reference) != value:
        raise ValueError("trajectory handoff reference is not canonical")
    return reference


class BenchmarkResponse(ContractModel):
    """Typed result wire-compatible with the coordinator's evidence model."""

    benchmark_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    suite: str = Field(min_length=1)
    suite_version: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    split: ObjectiveSplit
    metrics: BenchmarkMetrics
    trajectory_artifact: ArtifactReference | None = None
    trajectory_references: tuple[TrajectoryReference, ...] = Field(default_factory=tuple)
    report_artifact: ArtifactReference | None = None
    manifest_sha256: str | None = None
    evidence_label: EvidenceLabel = EvidenceLabel.EXPLANATION
    verified: bool = False

    @field_validator("manifest_sha256")
    @classmethod
    def validate_manifest_sha256(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256.fullmatch(value):
            raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_verified_artifact(self) -> BenchmarkResponse:
        if self.evidence_label in {EvidenceLabel.LIVE, EvidenceLabel.PRIOR_VERIFIED_RUN}:
            if not self.verified:
                raise ValueError("verified benchmark evidence must set verified=True")
            if self.manifest_sha256 is None:
                raise ValueError("verified benchmark evidence requires manifest_sha256")
            if self.trajectory_artifact is None and self.report_artifact is None:
                raise ValueError("verified benchmark evidence requires an artifact reference")
        if any(not reference.verified for reference in self.trajectory_references):
            raise ValueError("benchmark trajectory references must be verifier-confirmed")
        if any(reference.split is not self.split for reference in self.trajectory_references):
            raise ValueError("benchmark trajectory reference split does not match response")
        return self


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
    def curation_scope_only(cls, value: ObjectiveSplit) -> ObjectiveSplit:
        if value not in {ObjectiveSplit.TRAIN, ObjectiveSplit.REPLAY}:
            raise ValueError(f"split {value.value!r} is outside the train/replay curation scope")
        return value

    @model_validator(mode="after")
    def require_trajectory_input(self) -> CurationRequest:
        if not self.trajectories and not self.trajectory_references:
            raise ValueError("trajectories or trajectory_references is required")
        if any(reference.split is not self.split for reference in self.trajectory_references):
            raise ValueError("trajectory reference split does not match curation scope")
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
