"""Fail-closed HTTP control plane for autonomous live runs.

The router is deliberately dependency-injected.  Importing it never creates
an AWS client, starts a provider job, or installs a local execution fallback.
Applications must provide the durable repository, supervisor and dispatcher
on ``app.state`` before the mutating endpoints can proceed.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, cast
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from fastapi import APIRouter, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from app.autonomous.models import AutonomousRunState, AutonomousRunStatus, RunPhase
from app.autonomous.repository import (
    ApprovalAlreadyConsumedError,
    ConcurrentUpdateError,
    IdempotencyKeyConflictError,
    RepositoryError,
    RunAlreadyExistsError,
    RunNotFoundError,
)
from app.live_execution import (
    SAGEMAKER_PHASES_PER_EXPERIMENT,
    ApprovalPacket,
    LiveExecutionBlocked,
    _decode_approval_token,
    config_from_environment,
)

logger = logging.getLogger(__name__)
DISPATCH_RECOVERY_INTERVAL_SECONDS = 30.0


class LiveAPIError(RuntimeError):
    """A configured live control-plane dependency is unavailable."""


class PrepareRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    run_id: str | None = Field(default=None, min_length=1, max_length=200)
    run_number: int = Field(default=1, ge=1, le=5)
    model_id: str = "google/functiongemma-270m-it"
    checkpoint_revision: str | None = None
    checkpoint_s3_uri: str | None = None
    checkpoint_sha256: str | None = None
    benchmark_manifest_sha256: str | None = None
    benchmark_id: str = "service-recovery-v1"
    benchmark_suite: str = "AgentGym/AgentEval"
    benchmark_version: str = "agent-eval-v1"
    seed: int = 7
    max_experiments: int = Field(default=5, ge=1, le=5)
    max_cost_usd: float = Field(default=25.0, gt=0, le=25.0)
    instance_type: str | None = None
    instance_count: int | None = Field(default=None, ge=1)
    volume_size_gb: int | None = Field(default=None, ge=1)
    max_runtime_seconds: int | None = Field(default=None, ge=1)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    expires_at: datetime | None = None


class StartRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    run_id: str | None = Field(default=None, min_length=1, max_length=200)
    approval_token: str | None = None


class RunCommandResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: str
    phase: str
    version: int
    idempotency_key: str | None = None
    stop_reason: str | None = None


router = APIRouter(prefix="/api/live", tags=["autonomous-live"])


def _state_payload(state: AutonomousRunState) -> dict[str, Any]:
    """Serialize only the typed metadata state; never expose prompts/content."""

    return state.model_dump(mode="json")


def _repo(request: Request) -> Any:
    repository = getattr(request.app.state, "live_repository", None)
    if repository is None:
        raise LiveAPIError("durable live repository is not configured")
    return repository


def _dispatcher(request: Request) -> Any:
    dispatcher = getattr(request.app.state, "live_dispatcher", None)
    if dispatcher is None or not callable(getattr(dispatcher, "dispatch_once", None)):
        raise LiveAPIError("live dispatcher is not configured")
    return dispatcher


def _config(request: Request) -> Any:
    configured = getattr(request.app.state, "live_config", None)
    if configured is not None:
        return configured
    try:
        return config_from_environment()
    except Exception as exc:
        raise LiveAPIError("live configuration is blocked") from exc


def _approval_secret(request: Request) -> str:
    value = getattr(request.app.state, "live_approval_secret", None)
    if value is None:
        config = _config(request)
        env_name = getattr(config, "approval_secret_env", "LIVE_APPROVAL_SECRET")
        value = os.getenv(env_name, "")
    if not isinstance(value, str) or not value:
        raise LiveAPIError("live approval secret is not configured")
    return value


async def _call(value: Any, *args: Any, **kwargs: Any) -> Any:
    if callable(value):
        value = value(*args, **kwargs)
    if inspect.isawaitable(value):
        return await value
    return value


async def _preflight(request: Request) -> None:
    runner = getattr(request.app.state, "live_preflight_runner", None)
    if runner is None:
        runner = getattr(request.app.state, "live_preflight", None)
    if runner is None:
        config = _config(request)
        from app.live_execution import PreflightRunner

        runner = PreflightRunner(config)
    result = await _call(runner.run if hasattr(runner, "run") else runner)
    if not bool(getattr(result, "ready", False)):
        raise LiveAPIError("live preflight is BLOCKED")


def _begin_idempotency(
    request: Request,
    operation: str,
    key: str,
    request_value: Mapping[str, Any],
    *,
    allow_pending_recovery: bool = False,
) -> tuple[str, Mapping[str, Any] | None, bool]:
    """Atomically reserve a key for a canonical request or return its safe replay."""

    try:
        canonical = json.dumps(
            {"operation": operation, "request": request_value},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="request cannot be canonicalized") from exc
    request_digest = sha256(canonical.encode("utf-8")).hexdigest()
    try:
        repository = _repo(request)
        claim = getattr(repository, "claim_idempotency", None)
        if not callable(claim):
            raise LiveAPIError("durable idempotency support is not configured")
        record, created = claim(operation, key, request_digest)
    except IdempotencyKeyConflictError as exc:
        raise HTTPException(
            status_code=409, detail="Idempotency-Key was used for a different request"
        ) from exc
    except (LiveAPIError, RepositoryError) as exc:
        raise HTTPException(status_code=424, detail="durable idempotency is unavailable") from exc
    except Exception as exc:
        raise HTTPException(status_code=424, detail="durable idempotency is unavailable") from exc
    response = getattr(record, "response", None)
    if response is not None:
        if not isinstance(response, Mapping):
            raise HTTPException(status_code=424, detail="durable idempotency record is invalid")
        return request_digest, dict(response), False
    pending = not created
    if pending and not allow_pending_recovery:
        raise HTTPException(status_code=409, detail="request outcome is still pending")
    return request_digest, None, pending


def _complete_idempotency(
    request: Request,
    operation: str,
    key: str,
    request_digest: str,
    response: Mapping[str, Any],
) -> None:
    """Persist the exact client-safe response; storage failures fail the request closed."""

    try:
        repository = _repo(request)
        complete = getattr(repository, "complete_idempotency", None)
        if not callable(complete):
            raise LiveAPIError("durable idempotency completion is not configured")
        complete(operation, key, request_digest, response)
    except IdempotencyKeyConflictError as exc:
        raise HTTPException(status_code=409, detail="idempotency record changed") from exc
    except (LiveAPIError, RepositoryError) as exc:
        raise HTTPException(
            status_code=424, detail="durable idempotency response is unavailable"
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=424, detail="durable idempotency response is unavailable"
        ) from exc


def _require_idempotency(key: str | None) -> str:
    if key is None or not key.strip() or len(key) > 200:
        raise HTTPException(status_code=400, detail="Idempotency-Key header is required")
    return key.strip()


def _response_for(state: AutonomousRunState, key: str | None = None) -> RunCommandResponse:
    return RunCommandResponse(
        run_id=state.run_id,
        status=state.status.value,
        phase=state.phase.value,
        version=state.version,
        idempotency_key=key,
        stop_reason=state.stop_reason,
    )


def _scope_value(request: PrepareRunRequest, config: Any, name: str, default: Any = None) -> Any:
    value = getattr(request, name)
    if value is not None:
        return value
    return getattr(config, name, default)


def _packet_and_state(
    request: PrepareRunRequest,
    config: Any,
    *,
    run_id: str | None = None,
    issued_at: datetime | None = None,
) -> tuple[ApprovalPacket, AutonomousRunState]:
    run_id = run_id or request.run_id or f"run-{uuid4().hex}"
    revision = _scope_value(
        request, config, "checkpoint_revision", getattr(config, "hf_revision", "")
    )
    checkpoint_uri = _scope_value(request, config, "checkpoint_s3_uri")
    checkpoint_sha = _scope_value(request, config, "checkpoint_sha256")
    manifest_sha = _scope_value(request, config, "benchmark_manifest_sha256")
    configured_model = getattr(config, "target_model", request.model_id)
    configured_max_experiments = int(
        getattr(config, "max_runs", getattr(config, "max_experiments", 5))
    )
    configured_max_cost = float(getattr(config, "max_cost_usd", 25.0))
    if request.model_id != configured_model:
        raise LiveAPIError("requested model is outside configured live scope")
    if request.max_experiments > configured_max_experiments:
        raise LiveAPIError("requested experiment bound exceeds configured live scope")
    if request.max_cost_usd > configured_max_cost:
        raise LiveAPIError("requested budget exceeds configured live scope")
    if not isinstance(revision, str) or len(revision) != 40:
        raise LiveAPIError("immutable checkpoint revision is required")
    if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64:
        raise LiveAPIError("immutable checkpoint digest is required")
    if not isinstance(checkpoint_uri, str) or not checkpoint_uri.startswith("s3://"):
        raise LiveAPIError("versioned checkpoint S3 URI is required")
    parsed_checkpoint = urlparse(checkpoint_uri)
    if not parsed_checkpoint.netloc or not parsed_checkpoint.path.strip("/"):
        raise LiveAPIError("versioned checkpoint S3 URI is required")
    if not parse_qs(parsed_checkpoint.query).get("versionId", [""])[0]:
        raise LiveAPIError("checkpoint S3 URI must pin an object version")
    configured_values = {
        "instance_type": getattr(config, "instance_type", ""),
        "instance_count": getattr(config, "instance_count", 1),
        "volume_size_gb": getattr(config, "volume_size_gb", 30),
        "max_runtime_seconds": getattr(config, "max_runtime_seconds", 3600),
        "estimated_cost_usd": getattr(config, "estimated_run_cost_usd", 0.0),
    }
    configured_benchmark_id = getattr(config, "benchmark_id", request.benchmark_id)
    if request.benchmark_id != configured_benchmark_id:
        raise LiveAPIError("requested benchmark is outside configured live scope")
    for field_name, configured in configured_values.items():
        requested = getattr(request, field_name)
        if requested is not None and requested != configured:
            raise LiveAPIError(f"requested {field_name} is outside configured live scope")
    scope = {
        "run_id": run_id,
        "run_number": request.run_number,
        "model_id": request.model_id,
        "checkpoint_revision": revision.lower(),
        "checkpoint_s3_uri": checkpoint_uri or "",
        "checkpoint_sha256": checkpoint_sha,
        "benchmark_id": request.benchmark_id,
        "benchmark_suite": request.benchmark_suite,
        "benchmark_version": request.benchmark_version,
        "seed": request.seed,
        "max_experiments": request.max_experiments,
        "max_cost_usd": request.max_cost_usd,
        "instance_type": configured_values["instance_type"],
        "instance_count": configured_values["instance_count"],
        "volume_size_gb": configured_values["volume_size_gb"],
        "max_runtime_seconds": configured_values["max_runtime_seconds"],
        "estimated_cost_usd": configured_values["estimated_cost_usd"],
        "baseline_episodes": int(getattr(config, "baseline_episodes", 10)),
        "held_out_episodes": int(getattr(config, "held_out_episodes", 15)),
    }
    manifest_sha = manifest_sha or sha256(str(sorted(scope.items())).encode("utf-8")).hexdigest()
    now = issued_at or datetime.now(UTC)
    ttl_seconds = int(getattr(config, "approval_ttl_seconds", 86400))
    minimum_ttl_seconds = (
        request.max_experiments
        * SAGEMAKER_PHASES_PER_EXPERIMENT
        * int(configured_values["max_runtime_seconds"])
    )
    if ttl_seconds < minimum_ttl_seconds:
        raise LiveAPIError(
            "configured approval window is shorter than the bounded experiment runtime"
        )
    if request.expires_at is not None:
        if request.expires_at.tzinfo is None or request.expires_at.utcoffset() is None:
            raise LiveAPIError("approval expiry must include a timezone")
        expires = request.expires_at.astimezone(UTC)
    else:
        expires = now + timedelta(seconds=ttl_seconds)
    if expires <= now:
        raise LiveAPIError("approval expiry must be in the future")
    if expires - now > timedelta(seconds=ttl_seconds):
        raise LiveAPIError("approval expiry exceeds configured approval window")
    if expires - now < timedelta(seconds=minimum_ttl_seconds):
        raise LiveAPIError(
            "approval expiry is shorter than the bounded training/evaluation window"
        )
    packet = ApprovalPacket(
        run_id=run_id,
        run_number=request.run_number,
        instance_type=str(_scope_value(request, config, "instance_type", "")),
        instance_count=int(_scope_value(request, config, "instance_count", 1)),
        volume_size_gb=int(_scope_value(request, config, "volume_size_gb", 30)),
        max_runtime_seconds=int(_scope_value(request, config, "max_runtime_seconds", 3600)),
        estimated_cost_usd=float(
            _scope_value(
                request,
                config,
                "estimated_cost_usd",
                getattr(config, "estimated_run_cost_usd", 0.0),
            )
        ),
        immutable_model_revision=revision,
        max_experiments=request.max_experiments,
        max_cost_usd=request.max_cost_usd,
        target_model=request.model_id,
        benchmark_id=request.benchmark_id,
        objective_suite=request.benchmark_suite,
        objective_suite_version=request.benchmark_version,
        seed=request.seed,
        baseline_episodes=int(getattr(config, "baseline_episodes", 10)),
        held_out_episodes=int(getattr(config, "held_out_episodes", 15)),
        checkpoint_s3_uri=checkpoint_uri,
        manifest_sha256=manifest_sha,
        checkpoint_sha256=checkpoint_sha,
        issued_at=now,
        expires_at=expires,
    )
    state = AutonomousRunState(
        run_id=run_id,
        model_id=request.model_id,
        checkpoint_revision=revision,
        checkpoint_id=checkpoint_uri,
        base_checkpoint_uri=checkpoint_uri,
        base_checkpoint_sha256=checkpoint_sha,
        benchmark_id=request.benchmark_id,
        benchmark_manifest_sha256=manifest_sha,
        benchmark_suite=request.benchmark_suite,
        benchmark_version=request.benchmark_version,
        benchmark_seed=request.seed,
        max_experiments=request.max_experiments,
        approved_budget_usd=request.max_cost_usd,
        approval_scope=scope | {"packet_sha256": packet.digest},
        approval_expires_at=expires,
        metadata={"checkpoint_uri": checkpoint_uri or ""},
        created_at=now,
    )
    return packet, state


def _validate_prepared_packet(
    packet: ApprovalPacket, state: AutonomousRunState, config: Any
) -> None:
    """Ensure the signed envelope, durable snapshot, and deployment agree."""

    scope = state.approval_scope
    expected_scope = {
        "run_id": state.run_id,
        "run_number": scope.get("run_number"),
        "model_id": state.model_id,
        "checkpoint_revision": state.checkpoint_revision,
        "checkpoint_s3_uri": state.base_checkpoint_uri,
        "checkpoint_sha256": state.base_checkpoint_sha256,
        "benchmark_id": state.benchmark_id,
        "benchmark_suite": state.benchmark_suite,
        "benchmark_version": state.benchmark_version,
        "seed": state.benchmark_seed,
        "max_experiments": state.max_experiments,
        "max_cost_usd": state.approved_budget_usd,
        "instance_type": scope.get("instance_type"),
        "instance_count": scope.get("instance_count"),
        "volume_size_gb": scope.get("volume_size_gb"),
        "max_runtime_seconds": scope.get("max_runtime_seconds"),
        "estimated_cost_usd": scope.get("estimated_cost_usd"),
        "baseline_episodes": scope.get("baseline_episodes"),
        "held_out_episodes": scope.get("held_out_episodes"),
    }
    packet_values = {
        "run_id": packet.run_id,
        "run_number": packet.run_number,
        "model_id": packet.target_model,
        "checkpoint_revision": packet.immutable_model_revision,
        "checkpoint_s3_uri": packet.checkpoint_s3_uri,
        "checkpoint_sha256": packet.checkpoint_sha256,
        "benchmark_id": packet.benchmark_id,
        "benchmark_suite": packet.objective_suite,
        "benchmark_version": packet.objective_suite_version,
        "seed": packet.seed,
        "max_experiments": packet.max_experiments,
        "max_cost_usd": packet.max_cost_usd,
        "instance_type": packet.instance_type,
        "instance_count": packet.instance_count,
        "volume_size_gb": packet.volume_size_gb,
        "max_runtime_seconds": packet.max_runtime_seconds,
        "estimated_cost_usd": packet.estimated_cost_usd,
        "baseline_episodes": packet.baseline_episodes,
        "held_out_episodes": packet.held_out_episodes,
    }
    if packet_values != expected_scope:
        raise LiveExecutionBlocked("approval token does not match the durable approved scope")
    if scope.get("packet_sha256") != packet.digest:
        raise LiveExecutionBlocked("approval token does not match prepared scope")
    expected_model = getattr(config, "target_model", packet.target_model)
    if packet.target_model != expected_model:
        raise LiveExecutionBlocked("approved model is outside configured live scope")
    config_scope = {
        "checkpoint_revision": getattr(config, "hf_revision", packet.immutable_model_revision),
        "checkpoint_s3_uri": getattr(config, "checkpoint_s3_uri", packet.checkpoint_s3_uri),
        "checkpoint_sha256": getattr(config, "checkpoint_sha256", packet.checkpoint_sha256),
        "benchmark_id": getattr(config, "benchmark_id", packet.benchmark_id),
        "benchmark_suite": getattr(config, "objective_suite", packet.objective_suite),
        "benchmark_version": getattr(
            config, "objective_suite_version", packet.objective_suite_version
        ),
        "seed": getattr(config, "seed", packet.seed),
        "max_experiments": getattr(
            config, "max_runs", getattr(config, "max_experiments", packet.max_experiments)
        ),
        "max_cost_usd": getattr(config, "max_cost_usd", packet.max_cost_usd),
        "instance_type": getattr(config, "instance_type", packet.instance_type),
        "instance_count": getattr(config, "instance_count", packet.instance_count),
        "volume_size_gb": getattr(config, "volume_size_gb", packet.volume_size_gb),
        "max_runtime_seconds": getattr(
            config, "max_runtime_seconds", packet.max_runtime_seconds
        ),
        "estimated_cost_usd": getattr(
            config, "estimated_run_cost_usd", packet.estimated_cost_usd
        ),
        "baseline_episodes": getattr(config, "baseline_episodes", packet.baseline_episodes),
        "held_out_episodes": getattr(config, "held_out_episodes", packet.held_out_episodes),
    }
    for field_name, expected in config_scope.items():
        approved = packet_values[field_name]
        # These values are deployment-wide ceilings. A signed run may narrow
        # them, but must never expand them; the exact packet-to-state comparison
        # above still prevents changing the narrower approved values at start.
        if field_name in {"max_experiments", "max_cost_usd"}:
            in_scope = (
                isinstance(approved, (int, float))
                and not isinstance(approved, bool)
                and isinstance(expected, (int, float))
                and not isinstance(expected, bool)
                and approved <= expected
            )
        else:
            in_scope = approved == expected
        if not in_scope:
            raise LiveExecutionBlocked(f"approved {field_name} is outside configured live scope")


async def _schedule_dispatch(request: Request) -> None:
    operation = _dispatcher(request).dispatch_once
    task = asyncio.create_task(_call(operation), name="autonomous-live-dispatch")
    tasks = getattr(request.app.state, "live_tasks", None)
    if tasks is None:
        tasks = set()
        request.app.state.live_tasks = tasks
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    await asyncio.sleep(0)


def _live_task_set(app: Any) -> set[asyncio.Task[Any]]:
    tasks = getattr(app.state, "live_tasks", None)
    if tasks is None:
        tasks = set()
        app.state.live_tasks = tasks
    return tasks


async def _periodic_dispatch_recovery(app: Any) -> None:
    """Recover durable queued/expired runs after startup and on a short cadence."""

    dispatcher = getattr(app.state, "live_dispatcher", None)
    recover = getattr(dispatcher, "recover_incomplete_runs", None)
    if not callable(recover):
        return
    interval = float(
        getattr(
            app.state,
            "live_recovery_interval_seconds",
            DISPATCH_RECOVERY_INTERVAL_SECONDS,
        )
    )
    if interval <= 0:
        raise ValueError("live recovery interval must be positive")
    while True:
        try:
            await _call(recover)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Keep recovery alive, but do not put provider exception details or
            # payloads in logs. Durable state and telemetry remain authoritative.
            logger.warning("periodic autonomous dispatcher recovery failed")
        await asyncio.sleep(interval)


@router.post("/runs/prepare", status_code=status.HTTP_201_CREATED)
async def prepare_run(
    request: Request,
    body: PrepareRunRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, Any]:
    key = _require_idempotency(idempotency_key)
    operation = "prepare"
    request_digest, cached, pending = _begin_idempotency(
        request,
        operation,
        key,
        body.model_dump(mode="json"),
        allow_pending_recovery=True,
    )
    if cached is not None:
        return cast(dict[str, Any], dict(cached))
    try:
        await _preflight(request)
        config = _config(request)
        run_id = body.run_id or f"run-{sha256(f'{key}:{request_digest}'.encode()).hexdigest()[:32]}"
        packet, state = _packet_and_state(body, config, run_id=run_id)
        repository = _repo(request)
        try:
            repository.create(state)
        except RunAlreadyExistsError as exc:
            if not pending:
                raise
            existing = repository.get(run_id)
            if (
                existing is None
                or existing.status is not AutonomousRunStatus.PREPARED
                or existing.approval_consumed
            ):
                raise
            recovered_packet, recovered_state = _packet_and_state(
                body, config, run_id=run_id, issued_at=existing.created_at
            )
            if (
                recovered_packet.digest != existing.approval_scope.get("packet_sha256")
                or recovered_state.approval_scope != existing.approval_scope
                or recovered_packet.expires_at <= datetime.now(UTC)
            ):
                raise LiveExecutionBlocked(
                    "pending prepare outcome cannot be recovered safely"
                ) from exc
            packet, state = recovered_packet, existing
    except RunAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail="run already exists") from exc
    except (LiveAPIError, LiveExecutionBlocked, ValueError) as exc:
        raise HTTPException(status_code=424, detail=str(exc)) from exc
    payload = {
        "run_id": state.run_id,
        "run_number": body.run_number,
        "status": state.status.value,
        "approval_packet": packet.model_dump(mode="json"),
        "packet_sha256": packet.digest,
    }
    _complete_idempotency(request, operation, key, request_digest, payload)
    return payload


async def _recover_pending_start(
    request: Request,
    repository: Any,
    run_id: str,
    key: str,
    request_digest: str,
    packet: ApprovalPacket,
    state: AutonomousRunState,
) -> RunCommandResponse:
    """Complete a lost start response only when durable state proves its approval."""

    if not state.approval_consumed or state.approval_digest != packet.digest:
        raise HTTPException(status_code=409, detail="approval was consumed by another request")
    if state.status is AutonomousRunStatus.PREPARED:
        try:
            state = repository.transition(
                run_id,
                expected_version=state.version,
                status=AutonomousRunStatus.QUEUED,
                phase=RunPhase.QUEUED,
                reason="run queued",
                event_type="run.queued",
            )
        except ConcurrentUpdateError as exc:
            latest = repository.get(run_id)
            if latest is None or latest.status is AutonomousRunStatus.PREPARED:
                raise HTTPException(
                    status_code=409, detail="run changed concurrently; retry"
                ) from exc
            state = latest
    response = _response_for(state, key)
    _complete_idempotency(
        request, "start", key, request_digest, response.model_dump(mode="json")
    )
    if state.status in {
        AutonomousRunStatus.QUEUED,
        AutonomousRunStatus.RUNNING,
        AutonomousRunStatus.CANCEL_REQUESTED,
        AutonomousRunStatus.SAFE_STOP_REQUESTED,
    }:
        await _schedule_dispatch(request)
    return response


async def _start_run(
    request: Request, run_id: str, body: StartRunRequest, key: str
) -> RunCommandResponse:
    token = body.approval_token or request.headers.get("X-Approval-Token")
    operation = "start"
    request_digest, cached, pending = _begin_idempotency(
        request,
        operation,
        key,
        {"run_id": run_id, "approval_token": token},
        allow_pending_recovery=True,
    )
    if cached is not None:
        return RunCommandResponse.model_validate(cached)
    try:
        repository = _repo(request)
        _dispatcher(request)
    except LiveAPIError as exc:
        raise HTTPException(status_code=424, detail=str(exc)) from exc
    state = repository.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="run not found")
    packet: ApprovalPacket | None = None
    if pending and state.approval_consumed:
        try:
            packet = _decode_approval_token(token or "", _approval_secret(request))
        except (LiveAPIError, LiveExecutionBlocked, ValueError) as exc:
            raise HTTPException(status_code=424, detail=str(exc)) from exc
        if packet.run_id != run_id or state.approval_digest != packet.digest:
            raise HTTPException(status_code=409, detail="approval was consumed by another request")
        return await _recover_pending_start(
            request, repository, run_id, key, request_digest, packet, state
        )
    if state.status is not AutonomousRunStatus.PREPARED:
        raise HTTPException(status_code=409, detail="run is not prepared")
    try:
        await _preflight(request)
        packet = _decode_approval_token(token or "", _approval_secret(request))
        if packet.run_id != run_id:
            raise LiveExecutionBlocked("approval token run does not match request")
        _validate_prepared_packet(packet, state, _config(request))
        state = repository.consume_approval(run_id, packet.digest)
        state = repository.transition(
            run_id,
            expected_version=state.version,
            status=AutonomousRunStatus.QUEUED,
            phase=RunPhase.QUEUED,
            reason="run queued",
            event_type="run.queued",
        )
        response = _response_for(state, key)
        _complete_idempotency(
            request, operation, key, request_digest, response.model_dump(mode="json")
        )
        await _schedule_dispatch(request)
        return response
    except ApprovalAlreadyConsumedError as exc:
        if pending and packet is not None:
            current = repository.get(run_id)
            if (
                current is not None
                and current.approval_consumed
                and current.approval_digest == packet.digest
            ):
                return await _recover_pending_start(
                    request, repository, run_id, key, request_digest, packet, current
                )
        raise HTTPException(
            status_code=409, detail="approval packet has already been consumed"
        ) from exc
    except ConcurrentUpdateError as exc:
        raise HTTPException(status_code=409, detail="run changed concurrently; retry") from exc
    except (LiveAPIError, LiveExecutionBlocked, ValueError) as exc:
        raise HTTPException(status_code=424, detail=str(exc)) from exc


@router.post("/runs", status_code=status.HTTP_202_ACCEPTED)
async def start_run_root(
    request: Request,
    body: StartRunRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> RunCommandResponse:
    key = _require_idempotency(idempotency_key)
    run_id = body.run_id
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required")
    return await _start_run(request, run_id, body, key)


@router.post("/runs/{run_id}/start", status_code=status.HTTP_202_ACCEPTED)
async def start_run(
    run_id: str,
    request: Request,
    body: StartRunRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> RunCommandResponse:
    key = _require_idempotency(idempotency_key)
    return await _start_run(request, run_id, body, key)


@router.get("/runs/{run_id}")
async def get_run(run_id: str, request: Request) -> dict[str, Any]:
    try:
        state = _repo(request).get(run_id)
    except LiveAPIError as exc:
        raise HTTPException(status_code=424, detail=str(exc)) from exc
    if state is None:
        raise HTTPException(status_code=404, detail="run not found")
    return _state_payload(state)


@router.get("/runs/{run_id}/events")
async def get_events(
    run_id: str, request: Request, after: int = 0, limit: int = 100
) -> dict[str, Any]:
    try:
        page = _repo(request).list_events(run_id, after_sequence=after, limit=min(limit, 1000))
    except LiveAPIError as exc:
        raise HTTPException(status_code=424, detail=str(exc)) from exc
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc
    return {"items": [item.model_dump(mode="json") for item in page], "next_after": page.next_after}


@router.get("/runs/{run_id}/experiments")
async def get_experiments(
    run_id: str, request: Request, offset: int = 0, limit: int = 100
) -> dict[str, Any]:
    try:
        page = _repo(request).list_experiments(
            run_id, offset=max(0, offset), limit=min(limit, 1000)
        )
    except LiveAPIError as exc:
        raise HTTPException(status_code=424, detail=str(exc)) from exc
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc
    return {
        "items": [item.model_dump(mode="json") for item in page],
        "next_offset": page.next_offset,
    }


@router.get("/runs/{run_id}/artifacts")
async def get_artifacts(run_id: str, request: Request) -> dict[str, Any]:
    try:
        state = _repo(request).get(run_id)
    except LiveAPIError as exc:
        raise HTTPException(status_code=424, detail=str(exc)) from exc
    if state is None:
        raise HTTPException(status_code=404, detail="run not found")
    ids = (*state.baseline_artifact_ids, *state.champion_artifact_ids)
    return {"items": [{"artifact_id": artifact_id} for artifact_id in dict.fromkeys(ids)]}


async def _command(
    request: Request,
    run_id: str,
    *,
    cancel: bool,
    idempotency_key: str | None,
) -> RunCommandResponse:
    key = _require_idempotency(idempotency_key)
    operation_scope = "control:" + run_id
    control = "cancel" if cancel else "safe-stop"
    request_digest, cached, _ = _begin_idempotency(
        request,
        operation_scope,
        key,
        {"run_id": run_id, "control": control},
        allow_pending_recovery=True,
    )
    if cached is not None:
        return RunCommandResponse.model_validate(cached)
    try:
        repository = _repo(request)
    except LiveAPIError as exc:
        raise HTTPException(status_code=424, detail=str(exc)) from exc
    state = repository.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="run not found")
    operation_name = "request_cancel" if cancel else "request_safe_stop"
    operation = getattr(repository, operation_name, None)
    if not callable(operation):
        raise HTTPException(status_code=501, detail="durable control mutation is unavailable")
    try:
        state = operation(run_id, expected_version=state.version)
    except ConcurrentUpdateError as exc:
        raise HTTPException(status_code=409, detail="run changed concurrently; retry") from exc
    response = _response_for(state, key)
    _complete_idempotency(
        request, operation_scope, key, request_digest, response.model_dump(mode="json")
    )
    return response


@router.post("/runs/{run_id}/cancel", status_code=status.HTTP_202_ACCEPTED)
async def cancel_run(
    run_id: str,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> RunCommandResponse:
    return await _command(request, run_id, cancel=True, idempotency_key=idempotency_key)


@router.post("/runs/{run_id}/safe-stop", status_code=status.HTTP_202_ACCEPTED)
async def safe_stop_run(
    run_id: str,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> RunCommandResponse:
    return await _command(request, run_id, cancel=False, idempotency_key=idempotency_key)


@router.post("/runs/{run_id}/safe_stop", status_code=status.HTTP_202_ACCEPTED)
async def safe_stop_run_legacy(
    run_id: str,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> RunCommandResponse:
    return await _command(request, run_id, cancel=False, idempotency_key=idempotency_key)


router.add_api_route("/runs/{run_id}/status", get_run, methods=["GET"], name="get_run_status")
router.add_api_route("/runs/{run_id}/state", get_run, methods=["GET"], name="get_run_state")


def install_autonomous_live_api(app: Any) -> None:
    """Install routes and lifecycle hooks without constructing dependencies."""

    app.include_router(router)

    @app.on_event("startup")  # type: ignore[untyped-decorator]
    async def _recover_live_runs() -> None:
        dispatcher = getattr(app.state, "live_dispatcher", None)
        if dispatcher is not None and callable(
            getattr(dispatcher, "recover_incomplete_runs", None)
        ):
            tasks = _live_task_set(app)
            if getattr(app.state, "live_recovery_task", None) is None:
                task = asyncio.create_task(
                    _periodic_dispatch_recovery(app),
                    name="autonomous-live-periodic-recovery",
                )
                app.state.live_recovery_task = task
                tasks.add(task)
                task.add_done_callback(tasks.discard)
                # Start the first scan promptly without blocking ASGI startup
                # on a long-running recovered experiment.
                await asyncio.sleep(0)

    @app.on_event("shutdown")  # type: ignore[untyped-decorator]
    async def _shutdown_live_runs() -> None:
        tasks = list(getattr(app.state, "live_tasks", set()))
        app.state.live_recovery_task = None
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        dispatcher = getattr(app.state, "live_dispatcher", None)
        shutdown = getattr(dispatcher, "shutdown", None) if dispatcher is not None else None
        if not callable(shutdown) and dispatcher is not None:
            shutdown = getattr(dispatcher, "close", None)
        if callable(shutdown):
            await _call(shutdown)


__all__ = [
    "PrepareRunRequest",
    "RunCommandResponse",
    "StartRunRequest",
    "install_autonomous_live_api",
    "router",
]
