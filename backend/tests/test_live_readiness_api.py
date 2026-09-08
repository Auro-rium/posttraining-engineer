from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.live_readiness import install_live_readiness_api
from app.live_execution import (
    CheckResult,
    CheckStatus,
    GpuCapacityStatus,
    GpuQuotaStatus,
    PreflightClassification,
    PreflightReport,
    PreflightStatus,
)


class FakePreflight:
    config = SimpleNamespace(target_model="google/functiongemma-270m-it")

    def run(self) -> PreflightReport:
        return PreflightReport(
            status=PreflightStatus.BLOCKED,
            checked_at=datetime.now(UTC),
            region="us-east-1",
            estimated_worst_case_cost_usd=7.5,
            checks=(
                CheckResult(
                    name="gpu_quota",
                    status=CheckStatus.BLOCKED,
                    detail="secret provider error text must not leave this boundary",
                    classification=PreflightClassification.BLOCKED_GPU_QUOTA,
                ),
            ),
            classification=PreflightClassification.BLOCKED_GPU_QUOTA,
            blocked_classifications=(PreflightClassification.BLOCKED_GPU_QUOTA,),
            gpu_instance_type="ml.g5.xlarge",
            gpu_instance_allowlist=("ml.g5.xlarge",),
            gpu_quota_status=GpuQuotaStatus.INSUFFICIENT,
            gpu_capacity_status=GpuCapacityStatus.UNAVAILABLE,
        )


@pytest.mark.anyio
async def test_live_readiness_exposes_gpu_and_approval_metadata_without_details() -> None:
    from fastapi import FastAPI

    app = FastAPI()
    install_live_readiness_api(app)
    app.state.live_preflight_runner = FakePreflight()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/live/readiness")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "BLOCKED"
    assert body["gpu"]["quota_status"] == "INSUFFICIENT"
    assert body["gpu"]["capacity_status"] == "UNAVAILABLE"
    assert body["approval"]["required"] is True
    assert body["approval"]["packet_sha256"] is None
    assert "secret provider error" not in response.text


@pytest.mark.anyio
async def test_live_readiness_is_blocked_when_live_configuration_is_absent() -> None:
    from fastapi import FastAPI

    app = FastAPI()
    install_live_readiness_api(app)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/live/readiness")

    assert response.status_code == 200
    assert response.json()["status"] == "BLOCKED"
    assert response.json()["approval"]["required"] is True
