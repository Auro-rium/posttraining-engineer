from __future__ import annotations

import pytest

from app.models import EvaluationReport, EvidenceLabel
from app.orchestrator import decide_promotion


def report(**overrides: object) -> EvaluationReport:
    values: dict[str, object] = {
        "task_count": 100,
        "champion_success": 0.35,
        "candidate_success": 0.40,
        "champion_regression_success": 0.90,
        "candidate_regression_success": 0.88,
        "champion_action_validity": 0.97,
        "candidate_action_validity": 0.97,
        "paired_improvement_positive": True,
        "provenance_complete": True,
        "evidence_label": EvidenceLabel.LIVE,
    }
    values.update(overrides)
    return EvaluationReport.model_validate(values)


def test_promotes_at_exact_thresholds() -> None:
    decision = decide_promotion(report(), provenance_complete=True)

    assert decision.promoted is True
    assert decision.success_delta == pytest.approx(0.05)
    assert decision.regression_delta == pytest.approx(-0.02)
    assert all(reason.startswith("PASS") for reason in decision.reasons)


@pytest.mark.parametrize(
    ("overrides", "failed_reason"),
    [
        ({"candidate_success": 0.399}, "success gain"),
        ({"candidate_regression_success": 0.879}, "regression loss"),
        ({"candidate_action_validity": 0.969}, "action validity"),
        ({"provenance_complete": False}, "provenance"),
    ],
)
def test_rejects_each_failed_gate(overrides: dict[str, object], failed_reason: str) -> None:
    decision = decide_promotion(report(**overrides), provenance_complete=True)

    assert decision.promoted is False
    assert any(reason.startswith("FAIL") and failed_reason in reason for reason in decision.reasons)


def test_external_provenance_check_cannot_be_overridden_by_report() -> None:
    decision = decide_promotion(report(provenance_complete=True), provenance_complete=False)

    assert decision.promoted is False
    assert any("FAIL" in reason and "provenance" in reason for reason in decision.reasons)
