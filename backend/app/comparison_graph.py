"""Pure run-comparison data and SVG rendering for the post-training demo.

The renderer deliberately has no plotting or AWS dependency.  A coordinator
can persist the returned comparison payload and upload the SVG as an immutable
artifact without changing the measured values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from html import escape
from math import isfinite
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class RunMetricPoint:
    """One completed run's baseline/candidate measurements."""

    run_id: str
    run_number: int
    baseline_score: float
    candidate_score: float
    decision: str
    per_environment: dict[str, float]

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if not 1 <= self.run_number <= 5:
            raise ValueError("run_number must be between 1 and 5")
        if not self.decision.strip():
            raise ValueError("decision must not be empty")
        if not all(isfinite(value) for value in (self.baseline_score, self.candidate_score)):
            raise ValueError("scores must be finite")
        if any(not name.strip() for name in self.per_environment):
            raise ValueError("environment names must not be empty")
        if not all(isfinite(value) for value in self.per_environment.values()):
            raise ValueError("environment scores must be finite")


def _relative_improvement(baseline: float, candidate: float) -> float | None:
    if baseline == 0:
        return None
    return (candidate - baseline) / abs(baseline)


def build_comparison(points: Iterable[RunMetricPoint]) -> dict[str, Any]:
    """Return stable JSON-ready comparison data for at most five runs."""

    ordered = sorted(points, key=lambda item: item.run_number)
    if not ordered:
        raise ValueError("at least one run is required")
    if len(ordered) > 5:
        raise ValueError("comparison supports at most 5 runs")
    run_numbers = [item.run_number for item in ordered]
    run_ids = [item.run_id for item in ordered]
    if len(set(run_numbers)) != len(run_numbers) or len(set(run_ids)) != len(run_ids):
        raise ValueError("run numbers and run IDs must be unique")

    runs: list[dict[str, Any]] = []
    environments = sorted({name for item in ordered for name in item.per_environment})
    environment_series: dict[str, list[float | None]] = {name: [] for name in environments}
    for item in ordered:
        delta = item.candidate_score - item.baseline_score
        runs.append(
            {
                **asdict(item),
                "delta": delta,
                "relative_improvement": _relative_improvement(
                    item.baseline_score, item.candidate_score
                ),
            }
        )
        for name in environments:
            environment_series[name].append(item.per_environment.get(name))
    return {
        "run_count": len(runs),
        "runs": runs,
        "environments": environment_series,
    }


def render_comparison_svg(comparison: dict[str, Any], *, width: int = 760, height: int = 440) -> str:
    """Render aggregate baseline/candidate series as a self-contained SVG."""

    runs = comparison.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("comparison must contain at least one run")
    if width < 320 or height < 240:
        raise ValueError("graph dimensions are too small")

    values = [float(value) for run in runs for value in (run["baseline_score"], run["candidate_score"])]
    low, high = min(0.0, min(values)), max(1.0, max(values))
    left, right, top, bottom = 76, width - 28, 42, height - 72
    chart_width, chart_height = right - left, bottom - top

    def x(index: int) -> float:
        return left if len(runs) == 1 else left + chart_width * index / (len(runs) - 1)

    def y(value: float) -> float:
        return bottom - ((value - low) / (high - low)) * chart_height

    def points_for(key: str) -> str:
        return " ".join(f"{x(i):.1f},{y(float(run[key])):.1f}" for i, run in enumerate(runs))

    labels = []
    for i, run in enumerate(runs):
        labels.append(
            f'<text x="{x(i):.1f}" y="{bottom + 28}" text-anchor="middle" '
            f'font-size="11" fill="#475569">{escape(str(run["run_id"]))}</text>'
        )
        decision = escape(str(run.get("decision", "UNKNOWN")))
        labels.append(
            f'<text x="{x(i):.1f}" y="{bottom + 44}" text-anchor="middle" '
            f'font-size="10" fill="#64748b">{decision}</text>'
        )

    grid = []
    for tick in range(5):
        value = low + (high - low) * tick / 4
        ypos = y(value)
        grid.append(f'<line x1="{left}" y1="{ypos:.1f}" x2="{right}" y2="{ypos:.1f}" stroke="#e2e8f0"/>')
        grid.append(
            f'<text x="{left - 10}" y="{ypos + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#64748b">{value:.2f}</text>'
        )

    return "".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<text x="28" y="24" font-family="sans-serif" font-size="16" font-weight="600" '
            'fill="#0f172a">Post-training run comparison</text>',
            *grid,
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#94a3b8"/>',
            f'<polyline points="{points_for("baseline_score")}" fill="none" stroke="#64748b" stroke-width="3"/>',
            f'<polyline points="{points_for("candidate_score")}" fill="none" stroke="#2563eb" stroke-width="3"/>',
            *labels,
            '<line x1="520" y1="23" x2="545" y2="23" stroke="#64748b" stroke-width="3"/>',
            '<text x="551" y="27" font-family="sans-serif" font-size="11" fill="#334155">Baseline</text>',
            '<line x1="620" y1="23" x2="645" y2="23" stroke="#2563eb" stroke-width="3"/>',
            '<text x="651" y="27" font-family="sans-serif" font-size="11" fill="#334155">Candidate</text>',
            '</svg>',
        ]
    )
