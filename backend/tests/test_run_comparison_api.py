from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.run_comparison import install_run_comparison_api
from app.posttraining.run_history import (
    ComparisonDTO,
    RunComparisonRow,
    RunDecision,
    RunStatus,
)


def comparison() -> ComparisonDTO:
    rows = (
        RunComparisonRow(
            run_id="run-001",
            run_number=1,
            status=RunStatus.REJECTED,
            decision=RunDecision.REJECT,
            baseline_aggregate=0.40,
            candidate_aggregate=0.42,
            absolute_delta=0.02,
            relative_improvement=0.05,
            baseline_per_environment={"webshop": 0.40},
            candidate_per_environment={"webshop": 0.42},
        ),
        RunComparisonRow(
            run_id="run-002",
            run_number=2,
            status=RunStatus.COMPLETED,
            decision=RunDecision.PROMOTE,
            baseline_aggregate=0.42,
            candidate_aggregate=0.50,
            absolute_delta=0.08,
            relative_improvement=0.190476,
            baseline_per_environment={"webshop": 0.42},
            candidate_per_environment={"webshop": 0.50},
        ),
    )
    return ComparisonDTO(
        comparison_id="comparison-test",
        run_ids=("run-001", "run-002"),
        rows=rows,
        created_at=datetime(2026, 9, 8, tzinfo=UTC),
    )


class FakeRegistry:
    def compare(self, run_ids: list[str] | None = None) -> ComparisonDTO:
        result = comparison()
        if run_ids is None:
            return result
        selected = tuple(run_ids)
        return result.model_copy(
            update={
                "run_ids": selected,
                "rows": tuple(row for row in result.rows if row.run_id in selected),
            }
        )


@pytest.mark.anyio
async def test_comparison_endpoint_returns_ordered_rows() -> None:
    app = FastAPI()
    install_run_comparison_api(app, FakeRegistry())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/runs/compare")

    assert response.status_code == 200
    body = response.json()
    assert body["run_ids"] == ["run-001", "run-002"]
    assert body["rows"][1]["decision"] == "PROMOTE"


@pytest.mark.anyio
async def test_graph_endpoint_returns_svg_and_comparison_data() -> None:
    app = FastAPI()
    install_run_comparison_api(app, FakeRegistry())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/runs/graph")

    assert response.status_code == 200
    body = response.json()
    assert body["comparison"]["run_count"] == 2
    assert body["svg"].startswith("<svg ")


@pytest.mark.anyio
async def test_comparison_endpoint_rejects_more_than_five_requested_runs() -> None:
    app = FastAPI()
    install_run_comparison_api(app, FakeRegistry())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            "/api/runs/compare",
            params=[("run_ids", f"run-{number:03d}") for number in range(1, 7)],
        )

    assert response.status_code == 422
