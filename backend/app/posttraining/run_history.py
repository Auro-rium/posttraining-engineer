"""Bounded, repository-agnostic history for comparable post-training runs.

This module contains only typed records and the registry boundary.  It does not
know about AWS, an API framework, graph rendering, or a particular persistence
implementation.  A repository implementing :class:`RunHistoryRepository` is
responsible for making ``reserve_run`` atomic across coordinator instances.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from typing import Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from .models import ArtifactReference, Evidence, EvidenceKind, EvidenceLabel

MAX_RUNS = 5
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class FrozenDict(dict[str, object]):
    """A dict-compatible mapping that rejects every mutation operation.

    Pydantic's ``frozen`` model option protects model attributes, but does not
    recursively freeze ordinary dictionaries.  Keeping this as a dict
    subclass preserves existing ``model_dump`` and JSON serialization while
    making nested history snapshots safe to retain and share.
    """

    def _immutable(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("mapping is immutable")

    __setitem__ = _immutable  # type: ignore[assignment]
    __delitem__ = _immutable  # type: ignore[assignment]
    clear = _immutable  # type: ignore[assignment]
    pop = _immutable  # type: ignore[assignment]
    popitem = _immutable  # type: ignore[assignment]
    setdefault = _immutable  # type: ignore[assignment]
    update = _immutable  # type: ignore[assignment]
    __ior__ = _immutable  # type: ignore[assignment]


def _freeze_mapping(value: Mapping[str, object]) -> FrozenDict:
    return FrozenDict(value)


def _freeze_artifact(value: ArtifactReference) -> ArtifactReference:
    """Copy an artifact before freezing its nested metadata mapping."""

    copied = value.model_copy(deep=True)
    object.__setattr__(copied, "metadata", _freeze_mapping(dict(copied.metadata)))
    return copied


def _freeze_evidence(value: Evidence) -> Evidence:
    """Copy evidence before freezing its objective metrics mapping."""

    return _FrozenEvidence.model_validate(value.model_dump(mode="python"))


class _FrozenEvidence(Evidence):
    """Evidence model compatible with :class:`Evidence` but immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    @field_validator("metrics", mode="after")
    @classmethod
    def freeze_metrics(cls, value: dict[str, float]) -> dict[str, float]:
        return _freeze_mapping(value)  # type: ignore[return-value]


class RunStatus(StrEnum):
    """Lifecycle status persisted for one top-level optimization run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class RunDecision(StrEnum):
    """Decision produced by the deterministic promotion gate."""

    PROMOTE = "PROMOTE"
    REJECT = "REJECT"


class RunLimitExceeded(RuntimeError):
    """The repository rejected a reservation because the run cap was reached."""


class BenchmarkMetrics(BaseModel):
    """Aggregate and per-environment objective metrics for one evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    aggregate: float
    per_environment: dict[str, float] = Field(default_factory=dict)

    @field_validator("aggregate")
    @classmethod
    def validate_aggregate(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("aggregate must be finite")
        return value

    @field_validator("per_environment")
    @classmethod
    def validate_per_environment(cls, value: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for name, metric in value.items():
            if not name.strip():
                raise ValueError("environment names cannot be blank")
            if not isfinite(metric):
                raise ValueError(f"metric for {name!r} must be finite")
            normalized[name] = metric
        return _freeze_mapping(normalized)  # type: ignore[return-value]


class RunHistoryRecord(BaseModel):
    """Immutable history record for one candidate-versus-champion run."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    run_id: str = Field(min_length=1)
    run_number: int = Field(ge=1, le=MAX_RUNS)
    parent_run_id: str | None = None
    champion_run_id: str | None = None
    champion_artifact_id: str | None = None
    candidate_artifact_id: str | None = None
    benchmark_id: str | None = None
    suite: str | None = None
    suite_version: str | None = None
    seed: int | None = None
    model_id: str | None = None
    status: RunStatus = RunStatus.PENDING
    decision: RunDecision | None = None
    manifest_sha256: str | None = None
    baseline_metrics: BenchmarkMetrics | None = None
    candidate_metrics: BenchmarkMetrics | None = None
    artifact_refs: tuple[ArtifactReference, ...] = Field(default_factory=tuple)
    evidence: tuple[Evidence, ...] = Field(default_factory=tuple)
    decision_reasons: tuple[str, ...] = Field(default_factory=tuple)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("run_id must not be blank")
        return value

    @field_validator("parent_run_id", "champion_run_id")
    @classmethod
    def validate_link_id(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("run links must not be blank")
        return value

    @field_validator("manifest_sha256")
    @classmethod
    def validate_manifest_sha256(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256.fullmatch(value):
            raise ValueError("manifest_sha256 must be a lowercase 64-character digest")
        return value

    @field_validator("benchmark_id", "suite", "suite_version", "model_id")
    @classmethod
    def validate_optional_provenance(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("provenance values must not be blank")
        return value

    @field_validator("artifact_refs", mode="after")
    @classmethod
    def freeze_artifact_refs(
        cls, value: tuple[ArtifactReference, ...]
    ) -> tuple[ArtifactReference, ...]:
        return tuple(_freeze_artifact(item) for item in value)

    @field_validator("evidence", mode="after")
    @classmethod
    def freeze_evidence(cls, value: tuple[Evidence, ...]) -> tuple[Evidence, ...]:
        return tuple(_freeze_evidence(item) for item in value)

    @model_validator(mode="after")
    def validate_terminal_record(self) -> RunHistoryRecord:
        if self.parent_run_id == self.run_id or self.champion_run_id == self.run_id:
            raise ValueError("a run cannot link to itself")

        if self.run_number == 1 and (self.parent_run_id or self.champion_run_id):
            raise ValueError("run 1 cannot link to a prior run")
        if self.run_number > 1 and (self.parent_run_id is None or self.champion_run_id is None):
            raise ValueError("runs after run 1 require parent_run_id and champion_run_id")

        if self.status is RunStatus.COMPLETED and self.decision is not RunDecision.PROMOTE:
            raise ValueError("status and decision must agree: completed requires PROMOTE")
        if self.status is RunStatus.REJECTED and self.decision is not RunDecision.REJECT:
            raise ValueError("status and decision must agree: rejected requires REJECT")
        if (
            self.status not in {RunStatus.COMPLETED, RunStatus.REJECTED}
            and self.decision is not None
        ):
            raise ValueError("status and decision must agree: non-terminal runs have no decision")
        if self.status in {RunStatus.COMPLETED, RunStatus.REJECTED} and self.decision is None:
            raise ValueError("status and decision must agree: terminal runs require a decision")

        has_decision = self.decision is not None
        requires_evidence = has_decision or self.status in {
            RunStatus.COMPLETED,
            RunStatus.REJECTED,
        }
        if requires_evidence and self.manifest_sha256 is None:
            raise ValueError("manifest_sha256 is required for a terminal decision or completed run")
        if has_decision and (
            self.baseline_metrics is None or self.candidate_metrics is None
        ):
            raise ValueError("baseline_metrics and candidate_metrics are required for a decision")
        if requires_evidence:
            if not self.artifact_refs or self.candidate_artifact_id is None:
                raise ValueError(
                    "verified evidence requires artifact references and a candidate artifact"
                )
            artifact_ids = {item.artifact_id for item in self.artifact_refs}
            if self.candidate_artifact_id not in artifact_ids:
                raise ValueError("candidate_artifact_id must have an artifact reference")
            if any(
                value is None
                for value in (
                    self.benchmark_id,
                    self.suite,
                    self.suite_version,
                    self.seed,
                    self.model_id,
                )
            ):
                raise ValueError(
                    "benchmark_id, suite, suite_version, seed, and model_id are required "
                    "for a terminal decision"
                )
            if not self.evidence:
                raise ValueError("verified evidence is required for a terminal decision")
            for evidence in self.evidence:
                if (
                    evidence.kind is not EvidenceKind.EVALUATION
                    or evidence.label not in {EvidenceLabel.LIVE, EvidenceLabel.PRIOR_VERIFIED_RUN}
                    or not evidence.verified
                    or not evidence.artifact_ids
                ):
                    raise ValueError(
                        "terminal decisions require verified evidence with artifact references"
                    )
                if any(item not in artifact_ids for item in evidence.artifact_ids):
                    raise ValueError(
                        "evidence artifact references must be present in artifact_refs"
                    )
                if (
                    evidence.benchmark_id != self.benchmark_id
                    or evidence.suite != self.suite
                    or evidence.suite_version != self.suite_version
                    or evidence.seed != self.seed
                    or evidence.model_id != self.model_id
                    or evidence.manifest_sha256 != self.manifest_sha256
                ):
                    raise ValueError("evidence provenance must match the terminal run provenance")
        return self


class RunComparisonRow(BaseModel):
    """Flattened run record used directly by API and graph consumers."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    run_id: str
    run_number: int = Field(ge=1, le=MAX_RUNS)
    parent_run_id: str | None = None
    champion_run_id: str | None = None
    status: RunStatus
    decision: RunDecision | None = None
    manifest_sha256: str | None = None
    benchmark_id: str | None = None
    suite: str | None = None
    suite_version: str | None = None
    seed: int | None = None
    model_id: str | None = None
    champion_artifact_id: str | None = None
    candidate_artifact_id: str | None = None
    baseline_aggregate: float | None = None
    candidate_aggregate: float | None = None
    absolute_delta: float | None = None
    relative_improvement: float | None = None
    baseline_per_environment: dict[str, float] = Field(default_factory=dict)
    candidate_per_environment: dict[str, float] = Field(default_factory=dict)
    artifact_refs: tuple[ArtifactReference, ...] = Field(default_factory=tuple)
    evidence: tuple[Evidence, ...] = Field(default_factory=tuple)
    decision_reasons: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("baseline_per_environment", "candidate_per_environment", mode="after")
    @classmethod
    def freeze_metric_mappings(cls, value: dict[str, float]) -> dict[str, float]:
        return _freeze_mapping(dict(value))  # type: ignore[return-value]

    @field_validator("artifact_refs", mode="after")
    @classmethod
    def freeze_artifact_mappings(
        cls, value: tuple[ArtifactReference, ...]
    ) -> tuple[ArtifactReference, ...]:
        return tuple(_freeze_artifact(item) for item in value)

    @field_validator("evidence", mode="after")
    @classmethod
    def freeze_evidence_mappings(cls, value: tuple[Evidence, ...]) -> tuple[Evidence, ...]:
        return tuple(_freeze_evidence(item) for item in value)

    @classmethod
    def from_record(cls, record: RunHistoryRecord) -> RunComparisonRow:
        baseline = record.baseline_metrics
        candidate = record.candidate_metrics
        baseline_aggregate = baseline.aggregate if baseline else None
        candidate_aggregate = candidate.aggregate if candidate else None
        delta = (
            candidate_aggregate - baseline_aggregate
            if baseline_aggregate is not None and candidate_aggregate is not None
            else None
        )
        relative = (
            delta / abs(baseline_aggregate)
            if delta is not None and baseline_aggregate is not None and baseline_aggregate != 0
            else None
        )
        return cls(
            run_id=record.run_id,
            run_number=record.run_number,
            parent_run_id=record.parent_run_id,
            champion_run_id=record.champion_run_id,
            status=record.status,
            decision=record.decision,
            manifest_sha256=record.manifest_sha256,
            benchmark_id=record.benchmark_id,
            suite=record.suite,
            suite_version=record.suite_version,
            seed=record.seed,
            model_id=record.model_id,
            champion_artifact_id=record.champion_artifact_id,
            candidate_artifact_id=record.candidate_artifact_id,
            baseline_aggregate=baseline_aggregate,
            candidate_aggregate=candidate_aggregate,
            absolute_delta=delta,
            relative_improvement=relative,
            baseline_per_environment=baseline.per_environment if baseline else {},
            candidate_per_environment=candidate.per_environment if candidate else {},
            artifact_refs=record.artifact_refs,
            evidence=record.evidence,
            decision_reasons=record.decision_reasons,
        )


class ComparisonDTO(BaseModel):
    """Ordered, bounded comparison snapshot for an API response or graph."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    comparison_id: str = Field(min_length=1)
    run_ids: tuple[str, ...] = Field(max_length=MAX_RUNS)
    rows: tuple[RunComparisonRow, ...] = Field(max_length=MAX_RUNS)
    environment_names: tuple[str, ...] = Field(default_factory=tuple)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field(return_type=int)  # type: ignore[prop-decorator]
    @property
    def run_count(self) -> int:
        return len(self.rows)

    @model_validator(mode="after")
    def validate_rows(self) -> ComparisonDTO:
        row_ids = tuple(row.run_id for row in self.rows)
        if row_ids != self.run_ids:
            raise ValueError("run_ids must match rows in order")
        environments = tuple(
            sorted(
                {
                    *(
                        name
                        for row in self.rows
                        for name in row.baseline_per_environment
                    ),
                    *(
                        name
                        for row in self.rows
                        for name in row.candidate_per_environment
                    ),
                }
            )
        )
        if self.environment_names and self.environment_names != environments:
            raise ValueError("environment_names must match row metric environments")
        if not self.environment_names:
            object.__setattr__(self, "environment_names", environments)
        return self


@runtime_checkable
class RunHistoryRepository(Protocol):
    """Persistence contract for an atomic five-run registry.

    ``reserve_run`` is the single write boundary for the history.  An
    implementation MUST atomically validate all of these conditions in its
    backing store and commit the record and reservation together:

    * the current reservation count is below ``max_runs`` (at most five);
    * ``record.run_id`` is new; and
    * ``record.run_number`` is the next sequence number.

    A failed conditional transaction must leave both the counter and record
    unchanged and raise :class:`RunLimitExceeded` for the cap or a repository
    duplicate error for an existing id.  The registry performs read-only link
    diagnostics before this call, but never replaces this atomic reservation
    with a read-count-then-write sequence.
    """

    def reserve_run(
        self, record: RunHistoryRecord, *, max_runs: int = MAX_RUNS
    ) -> RunHistoryRecord: ...

    def get_history_run(self, run_id: str) -> RunHistoryRecord | None: ...

    def list_history_runs(self, *, limit: int = MAX_RUNS) -> Sequence[RunHistoryRecord]: ...


class RunRegistry:
    """Repository-agnostic façade for registering and comparing at most five runs."""

    def __init__(self, repository: RunHistoryRepository, *, max_runs: int = MAX_RUNS) -> None:
        if not 1 <= max_runs <= MAX_RUNS:
            raise ValueError(f"max_runs must be between 1 and {MAX_RUNS}")
        self._repository = repository
        self._max_runs = max_runs

    @property
    def max_runs(self) -> int:
        return self._max_runs

    def register(self, record: RunHistoryRecord) -> RunHistoryRecord:
        """Atomically reserve and persist one run record through the repository."""

        if record.run_number > self._max_runs:
            raise RunLimitExceeded(f"maximum of {self._max_runs} runs reached")
        existing = tuple(self._repository.list_history_runs(limit=self._max_runs))
        if len(existing) >= self._max_runs:
            raise RunLimitExceeded(f"maximum of {self._max_runs} runs reached")
        expected_number = max((item.run_number for item in existing), default=0) + 1
        if record.run_number != expected_number:
            raise ValueError(
                "run_number must be next sequential number "
                f"{expected_number}, got {record.run_number}"
            )
        if record.run_number > 1:
            assert record.parent_run_id is not None
            assert record.champion_run_id is not None
            parent = self._repository.get_history_run(record.parent_run_id)
            if parent is None:
                raise ValueError(f"parent_run_id {record.parent_run_id!r} does not exist")
            if parent.run_number != record.run_number - 1:
                raise ValueError("parent_run_id must reference the previous run")
            champion = self._repository.get_history_run(record.champion_run_id)
            if champion is None:
                raise ValueError(f"champion_run_id {record.champion_run_id!r} does not exist")
            if champion.run_number >= record.run_number:
                raise ValueError("champion_run_id must reference an earlier run")
        return self._repository.reserve_run(record, max_runs=self._max_runs)

    def get(self, run_id: str) -> RunHistoryRecord:
        record = self._repository.get_history_run(run_id)
        if record is None:
            raise KeyError(run_id)
        return record

    def list_runs(self, *, limit: int | None = None) -> tuple[RunHistoryRecord, ...]:
        requested = self._max_runs if limit is None else limit
        if not 1 <= requested <= self._max_runs:
            raise ValueError(f"limit must be between 1 and {self._max_runs}")
        records = self._repository.list_history_runs(limit=requested)
        return tuple(sorted(records, key=lambda record: record.run_number))

    def compare(self, run_ids: Iterable[str] | None = None) -> ComparisonDTO:
        """Return ordered flattened metrics and provenance for selected runs."""

        if run_ids is None:
            records = self.list_runs()
        else:
            requested = tuple(run_ids)
            if len(requested) > self._max_runs:
                raise ValueError(f"a comparison can contain at most {self._max_runs} runs")
            if len(set(requested)) != len(requested):
                raise ValueError("comparison run_ids must be unique")
            records = tuple(self.get(run_id) for run_id in requested)
            records = tuple(sorted(records, key=lambda record: record.run_number))

        rows = tuple(RunComparisonRow.from_record(record) for record in records)
        return ComparisonDTO(
            comparison_id=f"comparison-{uuid4().hex}",
            run_ids=tuple(row.run_id for row in rows),
            rows=rows,
        )


# Descriptive aliases keep the contract easy to discover for integrations.
RunMetrics = BenchmarkMetrics
RunComparison = ComparisonDTO
RunHistory = RunHistoryRecord
RunRegistryContract = RunHistoryRepository


__all__ = [
    "MAX_RUNS",
    "BenchmarkMetrics",
    "ComparisonDTO",
    "RunComparison",
    "RunComparisonRow",
    "RunDecision",
    "RunHistory",
    "RunHistoryRecord",
    "RunHistoryRepository",
    "RunLimitExceeded",
    "RunMetrics",
    "RunRegistry",
    "RunRegistryContract",
    "RunStatus",
]
