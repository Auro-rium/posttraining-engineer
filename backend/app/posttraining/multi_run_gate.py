"""Pure promotion gate for a bounded sequence of post-training runs.

This module owns no storage, clocks, random state, or AWS clients.  It compares
two immutable evaluation records and returns a replayable decision.  A caller
can therefore persist the input records and reproduce the exact decision later.
"""

from __future__ import annotations

from enum import StrEnum
from math import isfinite

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .gate import PromotionGate, PromotionGateConfig
from .models import Evidence, EvidenceLabel

MAX_RUNS = 5
_VERIFIED_LABELS = frozenset({EvidenceLabel.LIVE, EvidenceLabel.PRIOR_VERIFIED_RUN})


class RunPromotionDecision(StrEnum):
    """Terminal decision emitted by the multi-run gate."""

    PROMOTE = "PROMOTE"
    REJECT = "REJECT"


class MultiRunPromotionGateConfig(BaseModel):
    """Fixed policy for a bounded run sequence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_relative_improvement: float = Field(default=0.10, ge=0)
    max_absolute_regression: float = Field(default=0.05, ge=0)
    max_runs: int = Field(default=MAX_RUNS, ge=1, le=MAX_RUNS)

    @field_validator("min_relative_improvement", "max_absolute_regression")
    @classmethod
    def require_finite_threshold(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("gate thresholds must be finite")
        return value


class MultiRunEvaluation(BaseModel):
    """Objective score and provenance for one numbered candidate/champion run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    run_number: int = Field(ge=0)
    champion_run_id: str | None = Field(default=None, min_length=1)
    aggregate_score: float
    environment_scores: dict[str, float] = Field(default_factory=dict)
    evidence: Evidence

    @field_validator("aggregate_score")
    @classmethod
    def require_finite_aggregate(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("aggregate_score must be finite")
        return value

    @field_validator("environment_scores")
    @classmethod
    def require_finite_environment_scores(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not name.strip() for name in value):
            raise ValueError("environment names cannot be blank")
        if any(not isfinite(score) for score in value.values()):
            raise ValueError("environment scores must be finite")
        return value


class EnvironmentComparison(BaseModel):
    """Deterministic comparison for one benchmark environment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    environment: str = Field(min_length=1)
    champion_score: float | None = None
    candidate_score: float | None = None
    delta: float | None = None
    relative_change: float | None = None
    regression: float | None = None
    passed: bool


class MultiRunPromotionResult(BaseModel):
    """Complete, serializable result of one multi-run promotion decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: RunPromotionDecision
    passed: bool
    champion_run_id: str
    candidate_run_id: str
    champion_run_number: int
    candidate_run_number: int
    champion_score: float
    candidate_score: float
    aggregate_delta: float
    aggregate_improvement: float | None
    max_environment_regression: float | None
    improvement_passed: bool
    regression_passed: bool
    provenance_passed: bool
    run_sequence_passed: bool
    environments: tuple[EnvironmentComparison, ...] = ()
    reasons: tuple[str, ...] = ()


class MultiRunPromotionGate:
    """Evaluate a candidate against the current champion without side effects."""

    def __init__(self, config: MultiRunPromotionGateConfig | None = None) -> None:
        self.config = config or MultiRunPromotionGateConfig()

    def evaluate(
        self,
        champion: MultiRunEvaluation,
        candidate: MultiRunEvaluation,
    ) -> MultiRunPromotionResult:
        """Return the same decision for the same two evaluation records."""

        aggregate_gate = PromotionGate(
            PromotionGateConfig(
                min_relative_improvement=self.config.min_relative_improvement,
                # Environment-level checks own regression enforcement below.
                max_absolute_regression=0,
                require_verified_evidence=False,
            )
        )
        aggregate_result = aggregate_gate.evaluate(
            champion.aggregate_score,
            candidate.aggregate_score,
            regression_score=0,
        )

        run_sequence_passed = (
            candidate.run_id != champion.run_id
            and candidate.run_number == champion.run_number + 1
            and 1 <= candidate.run_number <= self.config.max_runs
        )
        provenance_passed = self._compatible_evidence(candidate.evidence, champion.evidence)

        environments = self._compare_environments(champion, candidate)
        regression_passed = bool(environments) and all(item.passed for item in environments)
        max_regression = (
            max((item.regression or 0.0) for item in environments) if environments else None
        )

        reasons: list[str] = []
        if not aggregate_result.improvement_passed:
            reasons.append("minimum aggregate improvement gate failed")
        if not regression_passed:
            reasons.append("maximum per-environment regression gate failed")
        if not provenance_passed:
            reasons.append(
                "verified LIVE or PRIOR_VERIFIED_RUN evidence with matching suite, version, "
                "and manifest is required"
            )
        if not run_sequence_passed:
            reasons.append(
                "run number must be the next sequential run and cannot exceed "
                f"maximum of {self.config.max_runs}"
            )

        passed = (
            aggregate_result.improvement_passed
            and regression_passed
            and provenance_passed
            and run_sequence_passed
        )
        return MultiRunPromotionResult(
            decision=RunPromotionDecision.PROMOTE if passed else RunPromotionDecision.REJECT,
            passed=passed,
            champion_run_id=champion.run_id,
            candidate_run_id=candidate.run_id,
            champion_run_number=champion.run_number,
            candidate_run_number=candidate.run_number,
            champion_score=champion.aggregate_score,
            candidate_score=candidate.aggregate_score,
            aggregate_delta=aggregate_result.absolute_delta,
            aggregate_improvement=aggregate_result.relative_improvement,
            max_environment_regression=max_regression,
            improvement_passed=aggregate_result.improvement_passed,
            regression_passed=regression_passed,
            provenance_passed=provenance_passed,
            run_sequence_passed=run_sequence_passed,
            environments=environments,
            reasons=tuple(reasons),
        )

    __call__ = evaluate

    def _compare_environments(
        self, champion: MultiRunEvaluation, candidate: MultiRunEvaluation
    ) -> tuple[EnvironmentComparison, ...]:
        names = tuple(sorted(set(champion.environment_scores) | set(candidate.environment_scores)))
        comparisons: list[EnvironmentComparison] = []
        for name in names:
            champion_score = champion.environment_scores.get(name)
            candidate_score = candidate.environment_scores.get(name)
            if champion_score is None or candidate_score is None:
                comparisons.append(
                    EnvironmentComparison(
                        environment=name,
                        champion_score=champion_score,
                        candidate_score=candidate_score,
                        passed=False,
                    )
                )
                continue

            delta = candidate_score - champion_score
            regression = max(0.0, -delta)
            relative_change = None if champion_score == 0 else delta / abs(champion_score)
            comparisons.append(
                EnvironmentComparison(
                    environment=name,
                    champion_score=champion_score,
                    candidate_score=candidate_score,
                    delta=delta,
                    relative_change=relative_change,
                    regression=regression,
                    passed=regression <= self.config.max_absolute_regression,
                )
            )
        return tuple(comparisons)

    @staticmethod
    def _compatible_evidence(candidate: Evidence, champion: Evidence) -> bool:
        return (
            candidate.verified
            and champion.verified
            and candidate.label in _VERIFIED_LABELS
            and champion.label in _VERIFIED_LABELS
            and bool(candidate.artifact_ids)
            and bool(champion.artifact_ids)
            and candidate.manifest_sha256 is not None
            and candidate.manifest_sha256 == champion.manifest_sha256
            and candidate.suite == champion.suite
            and candidate.suite_version == champion.suite_version
        )


def evaluate_multi_run_promotion(
    champion: MultiRunEvaluation,
    candidate: MultiRunEvaluation,
    *,
    config: MultiRunPromotionGateConfig | None = None,
) -> MultiRunPromotionResult:
    """Functional convenience wrapper for :class:`MultiRunPromotionGate`."""

    return MultiRunPromotionGate(config).evaluate(champion, candidate)


MultiRunPromotionGateResult = MultiRunPromotionResult
