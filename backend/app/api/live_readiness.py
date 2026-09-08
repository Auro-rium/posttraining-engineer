"""Read-only live-run readiness metadata for the execution console.

The route intentionally reports a blocked readiness state as a successful HTTP
response.  Readiness is an operator decision surface, not a server error, and
the UI must be able to render why an AWS run is waiting without receiving
provider exception text, credentials, prompts, or approval tokens.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict

from app.live_execution import (
    NEMOTRON_MODEL_ID,
    GpuCapacityStatus,
    GpuQuotaStatus,
    LiveExecutionBlocked,
    PreflightClassification,
    PreflightReport,
    PreflightRunner,
    PreflightStatus,
    config_from_environment,
)


class GpuReadinessPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_type: str | None = None
    allowlist: tuple[str, ...] = ()
    quota_status: GpuQuotaStatus = GpuQuotaStatus.UNKNOWN
    capacity_status: GpuCapacityStatus = GpuCapacityStatus.UNKNOWN


class ApprovalReadinessPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required: bool = True
    binding: str = "HMAC-SHA256 approval token bound to an immutable packet"
    packet_sha256: str | None = None


class LiveReadinessPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: PreflightStatus
    classification: PreflightClassification
    checked_at: str | None = None
    region: str
    reasoning_model: str = NEMOTRON_MODEL_ID
    target_model: str | None = None
    estimated_worst_case_cost_usd: float | None = None
    gpu: GpuReadinessPayload
    approval: ApprovalReadinessPayload
    checks: tuple[dict[str, Any], ...] = ()


router = APIRouter(tags=["live-readiness"])


def _blocked_payload(reasoning_model: str = NEMOTRON_MODEL_ID) -> LiveReadinessPayload:
    return LiveReadinessPayload(
        status=PreflightStatus.BLOCKED,
        classification=PreflightClassification.BLOCKED_CONFIGURATION,
        region="unknown",
        reasoning_model=reasoning_model,
        gpu=GpuReadinessPayload(),
        approval=ApprovalReadinessPayload(),
    )


def _report_payload(report: PreflightReport, *, target_model: str) -> LiveReadinessPayload:
    # Keep check metadata restricted to the typed readiness contract.  In
    # particular, do not forward arbitrary provider details or configured URIs.
    checks = tuple(
        {
            "name": item.name,
            "status": item.status.value,
            "classification": item.classification.value if item.classification else None,
        }
        for item in report.checks
    )
    return LiveReadinessPayload(
        status=report.status,
        classification=report.classification,
        checked_at=report.checked_at.isoformat(),
        region=report.region,
        target_model=target_model,
        estimated_worst_case_cost_usd=report.estimated_worst_case_cost_usd,
        gpu=GpuReadinessPayload(
            instance_type=report.gpu_instance_type or None,
            allowlist=report.gpu_instance_allowlist,
            quota_status=report.gpu_quota_status,
            capacity_status=report.gpu_capacity_status,
        ),
        approval=ApprovalReadinessPayload(),
        checks=checks,
    )


@router.get("/api/live/readiness", response_model=LiveReadinessPayload)
def live_readiness(request: Request) -> LiveReadinessPayload:
    """Run a read-only preflight and return safe UI-facing metadata."""

    # Tests/deployments may inject a preflight runner without changing the
    # route contract.  Normal operation constructs the runner from env only;
    # it never creates resources or submits a provider job.
    runner = getattr(request.app.state, "live_preflight_runner", None)
    if runner is None:
        try:
            config = config_from_environment()
            runner = PreflightRunner(config)
        except (LiveExecutionBlocked, ValueError):
            return _blocked_payload()
        except Exception:
            return _blocked_payload()

    try:
        report = runner.run()
        target_model = getattr(getattr(runner, "config", None), "target_model", "")
        return _report_payload(report, target_model=target_model or "unknown")
    except Exception:
        return _blocked_payload()


def install_live_readiness_api(app: Any) -> None:
    """Install the read-only readiness route."""

    app.include_router(router)


__all__ = ["LiveReadinessPayload", "install_live_readiness_api", "router"]
