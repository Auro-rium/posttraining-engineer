"""Fail-closed deterministic promotion gate."""

from __future__ import annotations

from enum import StrEnum
from math import isfinite

from pydantic import BaseModel, ConfigDict, Field

from .models import Evidence, EvidenceLabel


class PromotionDecision(StrEnum):
    PROMOTE = "PROMOTE"
    REJECT = "REJECT"


class PromotionGateConfig(BaseModel):
    """Thresholds and provenance policy for a promotion decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_relative_improvement: float = Field(default=0.10, ge=0)
    max_absolute_regression: float = Field(default=0.05, ge=0)
    require_verified_evidence: bool = True


class PromotionResult(BaseModel):
    """Complete, replayable output of the promotion gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: PromotionDecision
    passed: bool
    champion_score: float
    candidate_score: float
    absolute_delta: float
    relative_improvement: float | None
    regression: float
    improvement_passed: bool
    regression_passed: bool
    provenance_passed: bool
    reasons: tuple[str, ...] = ()


class PromotionGate:
    """Apply promotion thresholds without model or wall-clock input.

    ``regression_score`` is an optional separately measured absolute regression.
    When omitted, regression is derived from the candidate/champion score delta.
    """

    def __init__(self, config: PromotionGateConfig | None = None) -> None:
        self.config = config or PromotionGateConfig()

    def evaluate(
        self,
        champion_score: float,
        candidate_score: float,
        *,
        regression_score: float | None = None,
        candidate_evidence: Evidence | None = None,
        champion_evidence: Evidence | None = None,
    ) -> PromotionResult:
        values = (champion_score, candidate_score)
        if not all(isfinite(value) for value in values):
            raise ValueError("champion_score and candidate_score must be finite")
        if regression_score is not None:
            if not isfinite(regression_score) or regression_score < 0:
                raise ValueError("regression_score must be a finite non-negative number")

        delta = candidate_score - champion_score
        relative = None if champion_score == 0 else delta / abs(champion_score)
        improvement_passed = (
            candidate_score > 0
            if champion_score == 0
            else relative is not None and relative >= self.config.min_relative_improvement
        )
        if regression_score is None:
            regression = max(0.0, -delta)
        else:
            regression = regression_score
        regression_passed = regression <= self.config.max_absolute_regression

        provenance_passed = True
        reasons: list[str] = []
        if self.config.require_verified_evidence:
            provenance_passed = self._compatible_evidence(candidate_evidence, champion_evidence)
            if not provenance_passed:
                reasons.append("verified compatible candidate and champion evidence is required")
        if not improvement_passed:
            reasons.append("minimum relative improvement gate failed")
        if not regression_passed:
            reasons.append("maximum absolute regression gate failed")

        passed = improvement_passed and regression_passed and provenance_passed
        return PromotionResult(
            decision=PromotionDecision.PROMOTE if passed else PromotionDecision.REJECT,
            passed=passed,
            champion_score=champion_score,
            candidate_score=candidate_score,
            absolute_delta=delta,
            relative_improvement=relative,
            regression=regression,
            improvement_passed=improvement_passed,
            regression_passed=regression_passed,
            provenance_passed=provenance_passed,
            reasons=tuple(reasons),
        )

    __call__ = evaluate

    @staticmethod
    def _compatible_evidence(candidate: Evidence | None, champion: Evidence | None) -> bool:
        if candidate is None or champion is None:
            return False
        if not candidate.verified or not champion.verified:
            return False
        if candidate.label not in (EvidenceLabel.LIVE, EvidenceLabel.PRIOR_VERIFIED_RUN):
            return False
        if champion.label not in (EvidenceLabel.LIVE, EvidenceLabel.PRIOR_VERIFIED_RUN):
            return False
        return (
            candidate.suite == champion.suite
            and candidate.suite_version == champion.suite_version
            and candidate.manifest_sha256 == champion.manifest_sha256
            and (candidate.seed is None or champion.seed is None or candidate.seed == champion.seed)
        )


def evaluate_promotion(
    champion_score: float,
    candidate_score: float,
    *,
    config: PromotionGateConfig | None = None,
    regression_score: float | None = None,
    candidate_evidence: Evidence | None = None,
    champion_evidence: Evidence | None = None,
) -> PromotionResult:
    """Functional convenience wrapper around :class:`PromotionGate`."""

    return PromotionGate(config).evaluate(
        champion_score,
        candidate_score,
        regression_score=regression_score,
        candidate_evidence=candidate_evidence,
        champion_evidence=champion_evidence,
    )


PromotionGateResult = PromotionResult
