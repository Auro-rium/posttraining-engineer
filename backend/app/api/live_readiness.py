"""Read-only live-run readiness metadata for the execution console.

The route intentionally reports a blocked readiness state as a successful HTTP
response.  Readiness is an operator decision surface, not a server error, and
the UI must be able to render why an AWS run is waiting without receiving
provider exception text, credentials, prompts, or approval tokens.
"""

from __future__ import annotations

import os
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

_SAFE_BLOCKER_REASONS = {
    "live_configuration": "Required live configuration is missing or invalid.",
    "aws_identity": "The AWS identity could not be verified.",
    "bedrock_model_access": "Required reasoning-model access is unavailable.",
    "huggingface_pinned_revision": "The pinned Hugging Face model revision is unavailable.",
    "s3_artifact_bucket": "The artifact bucket is missing or does not meet required settings.",
    "sagemaker_training_input": (
        "The SageMaker training input prefix is missing, empty, or outside the artifact scope."
    ),
    "sagemaker_evaluation_input": (
        "The SageMaker evaluation input prefix is missing, empty, or outside the artifact scope."
    ),
    "pinned_checkpoint_artifact": (
        "The pinned checkpoint is missing or its immutable digest could not be verified."
    ),
    "dynamodb_history_table": (
        "The DynamoDB history table is inactive or has an unsupported key schema."
    ),
    "sagemaker_role_and_images": (
        "The SageMaker role trust or pinned worker images could not be verified."
    ),
    "gpu_allowlist": "The requested GPU instance is not on the configured allowlist.",
    "gpu_quota": "Configured GPU quota is insufficient or could not be verified.",
    "objective_worker": (
        "The objective worker is unavailable, unauthenticated, or not execution-ready."
    ),
    "approval_secret": "The one-run approval secret is not configured.",
    "cost_ceiling": "The worst-case SageMaker estimate exceeds the configured cost ceiling.",
}


def _blocked_payload(reasoning_model: str = NEMOTRON_MODEL_ID) -> LiveReadinessPayload:
    return LiveReadinessPayload(
        status=PreflightStatus.BLOCKED,
        classification=PreflightClassification.BLOCKED_CONFIGURATION,
        region="unknown",
        reasoning_model=reasoning_model,
        gpu=GpuReadinessPayload(),
        approval=ApprovalReadinessPayload(),
        checks=(
            {
                "name": "live_configuration",
                "status": "BLOCKED",
                "classification": PreflightClassification.BLOCKED_CONFIGURATION.value,
                "reason": _SAFE_BLOCKER_REASONS["live_configuration"],
            },
        ),
    )


def _report_payload(report: PreflightReport, *, target_model: str) -> LiveReadinessPayload:
    # Keep check metadata restricted to the typed readiness contract.  In
    # particular, do not forward arbitrary provider details or configured URIs.
    checks = tuple(
        {
            "name": item.name,
            "status": item.status.value,
            "classification": item.classification.value if item.classification else None,
            **(
                {"reason": _SAFE_BLOCKER_REASONS.get(item.name, "This readiness check is blocked.")}
                if item.status.value == "BLOCKED"
                else {}
            ),
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


def _approval_secret_configured(request: Request, runner: Any) -> bool:
    injected = getattr(request.app.state, "live_approval_secret", None)
    if isinstance(injected, str) and injected:
        return True
    config = getattr(runner, "config", None)
    name = getattr(config, "approval_secret_env", "LIVE_APPROVAL_SECRET")
    return bool(os.getenv(name, ""))


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
        payload = _report_payload(report, target_model=target_model or "unknown")
        # A successful provider/configuration probe is insufficient to launch
        # work unless the one-run HMAC approval secret is available too.
        if payload.status is PreflightStatus.READY and not _approval_secret_configured(
            request, runner
        ):
            return payload.model_copy(
                update={
                    "status": PreflightStatus.BLOCKED,
                    "classification": PreflightClassification.BLOCKED_CONFIGURATION,
                }
            )
        return payload
    except Exception:
        return _blocked_payload()


def install_live_readiness_api(app: Any) -> None:
    """Install the read-only readiness route."""

    app.include_router(router)


__all__ = ["LiveReadinessPayload", "install_live_readiness_api", "router"]
