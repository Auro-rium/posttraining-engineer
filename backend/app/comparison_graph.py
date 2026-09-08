"""Pure run-comparison data and SVG rendering for the post-training demo.

The renderer deliberately has no plotting or AWS dependency.  A coordinator
can persist the returned comparison payload and upload the SVG as an immutable
artifact without changing the measured values.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from html import escape
from math import isfinite
from typing import Any


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
        runs[-1]["per_environment"] = {
            name: item.per_environment[name] for name in sorted(item.per_environment)
        }
        for name in environments:
            environment_series[name].append(item.per_environment.get(name))
    return {
        "run_count": len(runs),
        "runs": runs,
        "environments": environment_series,
    }


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite numeric")
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    return numeric


def _normalise_comparison(
    comparison: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[float | None]]]:
    if not isinstance(comparison, Mapping):
        raise ValueError("comparison must be a mapping")
    runs = comparison.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("comparison must contain at least one run")
    if len(runs) > 5:
        raise ValueError("comparison supports at most 5 runs")
    run_count = comparison.get("run_count")
    if isinstance(run_count, bool) or not isinstance(run_count, int) or run_count != len(runs):
        raise ValueError("run_count must match the number of runs")

    normalised: list[dict[str, Any]] = []
    names_from_runs: set[str] = set()
    for index, raw_run in enumerate(runs):
        if not isinstance(raw_run, Mapping):
            raise ValueError(f"run {index} must be a mapping")
        run_id = raw_run.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(f"run {index} run_id must not be empty")
        run = dict(raw_run)
        run["baseline_score"] = _finite_number(
            raw_run.get("baseline_score"), f"run {index} baseline_score"
        )
        run["candidate_score"] = _finite_number(
            raw_run.get("candidate_score"), f"run {index} candidate_score"
        )
        if "run_number" in raw_run:
            run_number = raw_run["run_number"]
            if (
                isinstance(run_number, bool)
                or not isinstance(run_number, int)
                or not 1 <= run_number <= 5
            ):
                raise ValueError(f"run {index} run_number must be between 1 and 5")
        for field in ("delta", "relative_improvement"):
            if field in raw_run and raw_run[field] is not None:
                run[field] = _finite_number(raw_run[field], f"run {index} {field}")

        raw_environments = raw_run.get("per_environment", {})
        if not isinstance(raw_environments, Mapping):
            raise ValueError(f"run {index} per_environment must be a mapping")
        environments: dict[str, float] = {}
        raw_names = list(raw_environments)
        if any(not isinstance(name, str) or not name.strip() for name in raw_names):
            raise ValueError(f"run {index} environment names must not be empty")
        for name in sorted(raw_names):
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"run {index} environment names must not be empty")
            environments[name] = _finite_number(
                raw_environments[name], f"run {index} environment {name}"
            )
        names_from_runs.update(environments)
        run["per_environment"] = environments
        normalised.append(run)

    run_numbers = [run["run_number"] for run in normalised if "run_number" in run]
    if len(run_numbers) != len(set(run_numbers)):
        raise ValueError("run numbers must be unique")
    run_ids = [run["run_id"] for run in normalised]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("run IDs must be unique")

    if "environments" not in comparison:
        return normalised, {
            name: [run["per_environment"].get(name) for run in normalised]
            for name in sorted(names_from_runs)
        }
    supplied_series = comparison["environments"]
    if not isinstance(supplied_series, Mapping):
        raise ValueError("environments must be a mapping")
    supplied_names = list(supplied_series)
    if any(not isinstance(name, str) or not name.strip() for name in supplied_names):
        raise ValueError("environment names must not be empty")
    if set(supplied_series) != names_from_runs:
        raise ValueError("environments must match per_environment names")
    environment_series: dict[str, list[float | None]] = {}
    for name in sorted(supplied_series):
        series = supplied_series[name]
        if not isinstance(series, list) or len(series) != len(normalised):
            raise ValueError(f"environment {name} series must match run_count")
        checked: list[float | None] = []
        for index, value in enumerate(series):
            checked.append(
                None
                if value is None
                else _finite_number(value, f"environment {name} value {index}")
            )
        environment_series[name] = checked
    return normalised, environment_series


def render_comparison_svg(
    comparison: dict[str, Any], *, width: int = 760, height: int = 440
) -> str:
    """Render aggregate and per-environment series as a self-contained SVG."""

    runs, environment_series = _normalise_comparison(comparison)
    if (
        isinstance(width, bool)
        or not isinstance(width, (int, float))
        or not isfinite(float(width))
        or isinstance(height, bool)
        or not isinstance(height, (int, float))
        or not isfinite(float(height))
        or width < 320
        or height < 240
    ):
        raise ValueError("graph dimensions are too small")

    values = [
        value
        for run in runs
        for value in (run["baseline_score"], run["candidate_score"])
    ]
    values.extend(
        value for series in environment_series.values() for value in series if value is not None
    )
    low, high = min(0.0, min(values)), max(1.0, max(values))
    left, right, bottom = 76, width - 28, height - 72

    colors = ("#d97706", "#059669", "#7c3aed", "#db2777", "#0891b2")
    legend_items = [
        ("Baseline", "#64748b", "solid"),
        ("Candidate", "#2563eb", "solid"),
    ] + [
        (f"Environment: {name}", colors[index % len(colors)], "dashed")
        for index, name in enumerate(environment_series)
    ]
    legend_width = right - left
    legend_rows: list[list[tuple[str, str, str]]] = [[]]
    row_width = 0
    for item in legend_items:
        item_width = 30 + len(item[0]) * 6
        if legend_rows[-1] and row_width + item_width > legend_width:
            legend_rows.append([])
            row_width = 0
        legend_rows[-1].append(item)
        row_width += item_width
    top = 40 + len(legend_rows) * 18 + 8
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
        grid.append(
            f'<line x1="{left}" y1="{ypos:.1f}" x2="{right}" y2="{ypos:.1f}" '
            'stroke="#e2e8f0"/>'
        )
        grid.append(
            f'<text x="{left - 10}" y="{ypos + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#64748b">{value:.2f}</text>'
        )

    environment_lines = []
    for index, (name, series) in enumerate(environment_series.items()):
        color = colors[index % len(colors)]
        segments: list[list[str]] = [[]]
        for run_index, value in enumerate(series):
            if value is None:
                if segments[-1]:
                    segments.append([])
                continue
            segments[-1].append(f"{x(run_index):.1f},{y(value):.1f}")
        for segment in segments:
            if segment:
                points = " ".join(segment)
                environment_lines.append(
                    f'<polyline data-series="environment:{escape(name)}" points="{points}" '
                    f'fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="5 4"/>'
                )

    legend = []
    for row_index, row in enumerate(legend_rows):
        cursor = left
        for label, color, style in row:
            dash = ' stroke-dasharray="5 4"' if style == "dashed" else ""
            legend.append(
                f'<line x1="{cursor}" y1="{40 + row_index * 18}" '
                f'x2="{cursor + 20}" y2="{40 + row_index * 18}" stroke="{color}" '
                f'stroke-width="3"{dash}/>'
            )
            legend.append(
                f'<text x="{cursor + 26}" y="{44 + row_index * 18}" '
                f'font-family="sans-serif" font-size="11" fill="#334155">{escape(label)}</text>'
            )
            cursor += 30 + len(label) * 6

    return "".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<text x="28" y="24" font-family="sans-serif" font-size="16" font-weight="600" '
            'fill="#0f172a">Post-training run comparison</text>',
            *legend,
            *grid,
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#94a3b8"/>',
            f'<polyline data-series="aggregate:baseline" points="{points_for("baseline_score")}" '
            'fill="none" stroke="#64748b" stroke-width="3"/>',
            f'<polyline data-series="aggregate:candidate" points="{points_for("candidate_score")}" '
            'fill="none" stroke="#2563eb" stroke-width="3"/>',
            *environment_lines,
            *labels,
            '</svg>',
        ]
    )
