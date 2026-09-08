from __future__ import annotations

import pytest

from app.comparison_graph import RunMetricPoint, build_comparison, render_comparison_svg


def point(
    run_number: int, *, baseline: float, candidate: float, decision: str = "REJECT"
) -> RunMetricPoint:
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
        build_comparison(
            [point(1, baseline=0.3, candidate=0.35), point(1, baseline=0.3, candidate=0.4)]
        )
    with pytest.raises(ValueError, match="1 and 5"):
        build_comparison([point(6, baseline=0.3, candidate=0.35)])


def test_render_comparison_svg_contains_both_series_and_run_ids() -> None:
    comparison = build_comparison(
        [
            point(1, baseline=0.3, candidate=0.35),
            point(2, baseline=0.35, candidate=0.45, decision="PROMOTE"),
        ]
    )

    svg = render_comparison_svg(comparison)

    assert svg.startswith("<svg ")
    assert "Baseline" in svg
    assert "Candidate" in svg
    assert "run-001" in svg
    assert "run-002" in svg


def test_build_comparison_canonicalizes_environment_mappings() -> None:
    first = RunMetricPoint(
        run_id="run-001",
        run_number=1,
        baseline_score=0.3,
        candidate_score=0.35,
        decision="REJECT",
        per_environment={"wordle": 0.3, "webshop": 0.35},
    )

    comparison = build_comparison([first])

    assert list(comparison["runs"][0]["per_environment"]) == ["webshop", "wordle"]
    assert list(comparison["environments"]) == ["webshop", "wordle"]


def test_render_comparison_svg_contains_environment_series() -> None:
    comparison = build_comparison(
        [point(1, baseline=0.3, candidate=0.35), point(2, baseline=0.35, candidate=0.45)]
    )

    svg = render_comparison_svg(comparison, width=500)

    assert 'data-series="environment:webshop"' in svg
    assert 'data-series="environment:wordle"' in svg
    assert "Environment: webshop" in svg
    assert "Environment: wordle" in svg


@pytest.mark.parametrize(
    ("comparison", "message"),
    [
        ({"runs": []}, "at least one run"),
        ({"runs": [dict(baseline_score=0.1, candidate_score=0.2)] * 6}, "at most 5 runs"),
        ({"run_count": 2, "runs": [dict(baseline_score=0.1, candidate_score=0.2)]}, "run_count"),
        (
            {
                "run_count": 1,
                "runs": [
                    dict(run_id="run-001", baseline_score=float("nan"), candidate_score=0.2)
                ],
            },
            "finite",
        ),
        (
            {
                "run_count": 1,
                "runs": [
                    dict(run_id="run-001", baseline_score=0.1, candidate_score=float("inf"))
                ],
            },
            "finite",
        ),
    ],
)
def test_render_comparison_svg_rejects_invalid_input(
    comparison: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        render_comparison_svg(comparison)


def test_render_comparison_svg_legend_stays_within_narrow_width() -> None:
    comparison = build_comparison(
        [
            RunMetricPoint(
                run_id="run-001",
                run_number=1,
                baseline_score=0.3,
                candidate_score=0.35,
                decision="PROMOTE",
                per_environment={"babyai": 0.31, "movie": 0.32, "webshop": 0.35, "wordle": 0.3},
            )
        ]
    )

    svg = render_comparison_svg(comparison, width=320)

    assert 'x1="292"' not in svg
    assert 'x2="' in svg
