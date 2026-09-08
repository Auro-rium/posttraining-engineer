"""Run-history comparison and graph endpoints.

The router consumes a repository-backed registry.  It never invents metrics;
incomplete records are rejected before a graph is rendered.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, status

from app.comparison_graph import RunMetricPoint, build_comparison, render_comparison_svg
from app.posttraining.run_history import MAX_RUNS, ComparisonDTO


class ComparisonRegistry(Protocol):
    def compare(self, run_ids: Sequence[str] | None = None) -> ComparisonDTO: ...


router = APIRouter(tags=["run-comparison"])


def get_comparison_registry(request: Request) -> ComparisonRegistry:
    registry = getattr(request.app.state, "run_registry", None)
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="run comparison registry is not configured",
        )
    return registry


def _comparison(requested_ids: list[str] | None, registry: ComparisonRegistry) -> ComparisonDTO:
    if requested_ids is not None:
        if not requested_ids or len(requested_ids) > MAX_RUNS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"run_ids must contain between 1 and {MAX_RUNS} IDs",
            )
        if len(set(requested_ids)) != len(requested_ids):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="run_ids must be unique",
            )
    try:
        return registry.compare(requested_ids)
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


def _graph_payload(comparison: ComparisonDTO) -> dict[str, object]:
    points: list[RunMetricPoint] = []
    for row in comparison.rows:
        if row.baseline_aggregate is None or row.candidate_aggregate is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"run {row.run_id} has incomplete metrics for graphing",
            )
        points.append(
            RunMetricPoint(
                run_id=row.run_id,
                run_number=row.run_number,
                baseline_score=row.baseline_aggregate,
                candidate_score=row.candidate_aggregate,
                decision=row.decision.value if row.decision else row.status.value,
                per_environment=row.candidate_per_environment,
            )
        )
    try:
        chart_data = build_comparison(points)
    except ValueError as exc:
        # An empty or malformed history is a client-visible readiness issue,
        # not an unhandled server exception.  Keep the API fail-closed so the
        # dashboard can render the error state and retry after a run exists.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return {
        "comparison": comparison.model_dump(mode="json"),
        "chart_data": chart_data,
        "svg": render_comparison_svg(chart_data),
    }


@router.get("/api/runs/compare", response_model=ComparisonDTO)
async def compare_runs(
    run_ids: list[str] | None = Query(default=None),  # noqa: B008
    registry: ComparisonRegistry = Depends(get_comparison_registry),  # noqa: B008
) -> ComparisonDTO:
    """Return up to five ordered run records for comparison."""

    return _comparison(run_ids, registry)


@router.get("/api/runs/graph")
async def run_graph(
    run_ids: list[str] | None = Query(default=None),  # noqa: B008
    registry: ComparisonRegistry = Depends(get_comparison_registry),  # noqa: B008
) -> dict[str, object]:
    """Return chart data and a self-contained SVG for the selected runs."""

    return _graph_payload(_comparison(run_ids, registry))


def install_run_comparison_api(app: FastAPI, registry: ComparisonRegistry | None = None) -> None:
    """Install comparison routes and optionally attach a registry."""

    if registry is not None:
        app.state.run_registry = registry
    app.include_router(router)


__all__ = ["ComparisonRegistry", "get_comparison_registry", "install_run_comparison_api", "router"]
