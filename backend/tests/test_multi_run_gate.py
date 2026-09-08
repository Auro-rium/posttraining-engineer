"""Tests for the bounded, deterministic multi-run promotion gate."""

from __future__ import annotations

import pytest

from app.posttraining.models import Evidence, EvidenceKind, EvidenceLabel
from app.posttraining.multi_run_gate import (
    MAX_RUNS,
    MultiRunEvaluation,
    MultiRunPromotionGate,
    MultiRunPromotionGateConfig,
    RunPromotionDecision,
)

MANIFEST = "a" * 64


def _evidence(
    evidence_id: str,
    *,
    label: EvidenceLabel = EvidenceLabel.LIVE,
    verified: bool = True,
    suite: str = "agentgym",
    suite_version: str = "2026-09-01",
    manifest: str | None = MANIFEST,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        kind=EvidenceKind.EVALUATION,
        label=label,
        artifact_ids=(f"artifact-{evidence_id}",),
        metrics={"aggregate": 0.0},
        verified=verified,
        benchmark_id="agentgym-held-out",
        suite=suite,
        suite_version=suite_version,
        manifest_sha256=manifest,
        seed=17,
    )


def _evaluation(
    run_id: str,
    run_number: int,
    aggregate: float,
    environments: dict[str, float],
    *,
    evidence: Evidence | None = None,
    champion_run_id: str | None = None,
) -> MultiRunEvaluation:
    return MultiRunEvaluation(
        run_id=run_id,
        run_number=run_number,
        champion_run_id=champion_run_id,
        aggregate_score=aggregate,
        environment_scores=environments,
        evidence=evidence or _evidence(f"{run_id}-evaluation"),
    )


def test_promotes_ten_percent_improvement_with_no_environment_regression() -> None:
    champion = _evaluation("run-000", 0, 0.50, {"babyai": 0.50, "webshop": 0.50})
    candidate = _evaluation(
        "run-001",
        1,
        0.55,
        {"babyai": 0.50, "webshop": 0.53},
        champion_run_id=champion.run_id,
    )

    result = MultiRunPromotionGate().evaluate(champion, candidate)

    assert result.decision is RunPromotionDecision.PROMOTE
    assert result.aggregate_improvement == pytest.approx(0.10)
    assert result.run_sequence_passed is True
    assert result.regression_passed is True


def test_rejects_candidate_below_minimum_aggregate_improvement() -> None:
    champion = _evaluation("run-000", 0, 0.50, {"babyai": 0.50})
    candidate = _evaluation("run-001", 1, 0.549, {"babyai": 0.50}, champion_run_id="run-000")

    result = MultiRunPromotionGate().evaluate(champion, candidate)

    assert result.decision is RunPromotionDecision.REJECT
    assert result.improvement_passed is False
    assert "minimum aggregate improvement" in result.reasons[0]


def test_rejects_environment_regression_even_when_aggregate_improves() -> None:
    champion = _evaluation("run-000", 0, 0.50, {"babyai": 0.80, "webshop": 0.20})
    candidate = _evaluation(
        "run-001",
        1,
        0.60,
        {"babyai": 0.74, "webshop": 0.46},
        champion_run_id="run-000",
    )

    result = MultiRunPromotionGate().evaluate(champion, candidate)

    assert result.decision is RunPromotionDecision.REJECT
    assert result.improvement_passed is True
    assert result.regression_passed is False
    assert result.environments[0].environment == "babyai"
    assert result.environments[0].regression == pytest.approx(0.06)


def test_rejects_unverified_or_incompatible_evidence() -> None:
    champion = _evaluation("run-000", 0, 0.50, {"babyai": 0.50})
    candidate = _evaluation(
        "run-001",
        1,
        0.56,
        {"babyai": 0.56},
        champion_run_id="run-000",
        evidence=_evidence("candidate", label=EvidenceLabel.EXPLANATION),
    )

    result = MultiRunPromotionGate().evaluate(champion, candidate)

    assert result.decision is RunPromotionDecision.REJECT
    assert result.provenance_passed is False

    incompatible = _evaluation(
        "run-001",
        1,
        0.56,
        {"babyai": 0.56},
        champion_run_id="run-000",
        evidence=_evidence("candidate", suite_version="different"),
    )
    mismatch_result = MultiRunPromotionGate().evaluate(champion, incompatible)
    assert mismatch_result.provenance_passed is False


def test_enforces_sequential_run_numbers_and_hard_five_run_limit() -> None:
    champion = _evaluation("run-004", 4, 0.50, {"babyai": 0.50})
    sixth = _evaluation("run-006", MAX_RUNS + 1, 0.60, {"babyai": 0.60}, champion_run_id="run-004")

    result = MultiRunPromotionGate().evaluate(champion, sixth)

    assert result.decision is RunPromotionDecision.REJECT
    assert result.run_sequence_passed is False
    assert "maximum of 5" in " ".join(result.reasons)

    skipped = _evaluation("run-003", 3, 0.60, {"babyai": 0.60}, champion_run_id="run-004")
    skipped_result = MultiRunPromotionGate().evaluate(champion, skipped)
    assert skipped_result.run_sequence_passed is False


def test_result_is_reproducible_and_environment_order_is_canonical() -> None:
    champion = _evaluation("run-000", 0, 0.50, {"webshop": 0.50, "babyai": 0.50})
    candidate = _evaluation(
        "run-001",
        1,
        0.60,
        {"webshop": 0.60, "babyai": 0.50},
        champion_run_id="run-000",
    )
    gate = MultiRunPromotionGate(
        MultiRunPromotionGateConfig(min_relative_improvement=0.20)
    )

    first = gate.evaluate(champion, candidate)
    second = gate.evaluate(champion, candidate)

    assert first == second
    assert tuple(item.environment for item in first.environments) == ("babyai", "webshop")
