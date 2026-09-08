from __future__ import annotations

import pytest

from app.comparison_graph import RunMetricPoint, build_comparison, render_comparison_svg


def point(run_number: int, *, baseline: float, candidate: float, decision: str = "REJECT") -> RunMetricPoint:
    return RunMetricPoint(
        run_id=f"run-{run_number:03d}",
        run_number=run_number,
        baseline_score=baseline,
        candidate_score=candidate,
        decision=decision,
        per_environment={"webshop": candidate, "wordle": candidate - 0.05},
    )


def test_build_comparison_sorts_runs_and_computes_deltas() -> None:
    comparison = build_comparison(
        [point(2, baseline=0.4, candidate=0.5), point(1, baseline=0.3, candidate=0.35)]
    )

    assert [item["run_id"] for item in comparison["runs"]] == ["run-001", "run-002"]
    assert comparison["runs"][0]["delta"] == pytest.approx(0.05)
    assert comparison["runs"][1]["relative_improvement"] == pytest.approx(0.25)
    assert comparison["environments"]["webshop"] == [0.35, 0.5]


def test_build_comparison_rejects_duplicate_or_out_of_range_runs() -> None:
    with pytest.raises(ValueError, match="unique"):
        build_comparison([point(1, baseline=0.3, candidate=0.35), point(1, baseline=0.3, candidate=0.4)])
    with pytest.raises(ValueError, match="1 and 5"):
        build_comparison([point(6, baseline=0.3, candidate=0.35)])


def test_render_comparison_svg_contains_both_series_and_run_ids() -> None:
    comparison = build_comparison(
        [point(1, baseline=0.3, candidate=0.35), point(2, baseline=0.35, candidate=0.45, decision="PROMOTE")]
    )

    svg = render_comparison_svg(comparison)

    assert svg.startswith("<svg ")
    assert "Baseline" in svg
    assert "Candidate" in svg
    assert "run-001" in svg
    assert "run-002" in svg
