"""Typed records used by the deterministic post-training control plane.

The models in this module deliberately contain references and measurements, not
model-generated explanations.  An artifact is immutable once recorded and an
evidence record carries enough provenance for a gate to reject an ungrounded
candidate.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ArtifactKind(StrEnum):
    """Kinds of immutable objects that can participate in a cycle."""

    CHECKPOINT = "checkpoint"
    MODEL = "model"
    DATASET = "dataset"
    TRAJECTORY = "trajectory"
    REPORT = "report"
    CONFIGURATION = "configuration"


class EvidenceLabel(StrEnum):
    """Provenance labels shown to operators and judges."""

    LIVE = "LIVE"
    PRIOR_VERIFIED_RUN = "PRIOR_VERIFIED_RUN"
    EXPLANATION = "EXPLANATION"


class EvidenceKind(StrEnum):
    """Objective measurement or verification represented by an evidence record."""

    BENCHMARK = "benchmark"
    EVALUATION = "evaluation"
    REPLAY = "replay"
    TRAINING = "training"
    PROMOTION = "promotion"


class CycleState(StrEnum):
    """Lifecycle states for one candidate post-training cycle."""

    CREATED = "created"
    BENCHMARKED = "benchmarked"
    EVALUATED = "evaluated"
    APPROVED = "approved"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class Artifact(BaseModel):
    """Content-addressed immutable artifact reference.

    ``sha256`` is intentionally required.  A URI without an integrity digest
    cannot be used as promotion evidence because it could point at mutable
    storage.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    artifact_id: str = Field(min_length=1)
    kind: ArtifactKind
    uri: str = Field(min_length=1)
    sha256: str
    size_bytes: int | None = Field(default=None, ge=0)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("sha256 must be a lowercase 64-character hexadecimal digest")
        return value


class Evidence(BaseModel):
    """A typed, provenance-bearing objective result.

    Metrics are scalar values so promotion logic can remain deterministic and
    independent of any LLM output format.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    evidence_id: str = Field(min_length=1)
    kind: EvidenceKind
    label: EvidenceLabel
    artifact_ids: tuple[str, ...] = Field(default_factory=tuple)
    metrics: dict[str, float] = Field(default_factory=dict)
    verified: bool = False
    benchmark_id: str = Field(min_length=1)
    suite: str = Field(min_length=1)
    suite_version: str = Field(min_length=1)
    manifest_sha256: str | None = None
    seed: int | None = None
    model_id: str | None = None

    @field_validator("artifact_ids")
    @classmethod
    def validate_artifact_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("artifact_ids cannot contain blank references")
        return value

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: dict[str, float]) -> dict[str, float]:
        for name, metric in value.items():
            if not name.strip():
                raise ValueError("metric names cannot be blank")
            if not isinstance(metric, (int, float)):
                raise ValueError(f"metric {name!r} must be numeric")
        return value

    @field_validator("manifest_sha256")
    @classmethod
    def validate_manifest_sha256(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256.fullmatch(value):
            raise ValueError("manifest_sha256 must be a lowercase 64-character hexadecimal digest")
        return value

    @model_validator(mode="after")
    def validate_provenance(self) -> Evidence:
        if self.label in (EvidenceLabel.LIVE, EvidenceLabel.PRIOR_VERIFIED_RUN):
            if not self.verified:
                raise ValueError("live and prior-run evidence must be marked verified")
            if not self.artifact_ids:
                raise ValueError("verified evidence must reference at least one artifact")
            if not self.manifest_sha256:
                raise ValueError("verified evidence must include a manifest sha256")
        return self


class CycleEvent(BaseModel):
    """Append-only state transition record."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    event_id: str | None = None
    from_state: CycleState
    to_state: CycleState
    reason: str = Field(min_length=1)
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PostTrainingCycle(BaseModel):
    """State and immutable references for a continuous post-training cycle."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    cycle_id: str = Field(min_length=1)
    state: CycleState = CycleState.CREATED
    champion: Artifact | None = None
    candidate: Artifact | None = None
    evidence: tuple[Evidence, ...] = Field(default_factory=tuple)
    events: tuple[CycleEvent, ...] = Field(default_factory=tuple)
    champion_score: float | None = None
    candidate_score: float | None = None
    gate_passed: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# Descriptive aliases keep the public API convenient without duplicating models.
ArtifactRef = Artifact
ArtifactReference = Artifact
EvidenceRecord = Evidence
EvidenceType = EvidenceKind
ArtifactType = ArtifactKind
Cycle = PostTrainingCycle
CycleStatus = CycleState
CycleTransition = CycleEvent
