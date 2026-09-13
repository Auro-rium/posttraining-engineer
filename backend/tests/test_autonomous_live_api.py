from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.autonomous_live import install_autonomous_live_api
from app.autonomous.models import AutonomousRunState, AutonomousRunStatus, RunPhase
from app.autonomous.repository import InMemoryAutonomousRunRepository, RepositoryError
from app.live_execution import (
    ApprovalPacket,
    _decode_approval_token,
    issue_approval_token,
)

REVISION = "a" * 40
MANIFEST = "b" * 64
CHECKPOINT = "c" * 64


def _state(run_id: str = "run-1", **updates: object) -> AutonomousRunState:
    values: dict[str, object] = dict(
        run_id=run_id,
        checkpoint_revision=REVISION,
        benchmark_manifest_sha256=MANIFEST,
        base_checkpoint_sha256=CHECKPOINT,
        base_checkpoint_uri="s3://bucket/checkpoint?versionId=v1",
    )
    values.update(updates)
    return AutonomousRunState(**values)


def _app(repository: InMemoryAutonomousRunRepository | None = None) -> FastAPI:
    app = FastAPI()
    app.state.live_repository = repository or InMemoryAutonomousRunRepository()
    app.state.live_approval_secret = "test-secret"
    app.state.live_config = SimpleNamespace(
        target_model="google/functiongemma-270m-it",
        objective_suite="AgentGym/AgentEval",
        objective_suite_version="agent-eval-v1",
        seed=7,
        max_runs=5,
        max_experiments=5,
        max_cost_usd=25.0,
        instance_type="ml.g5.xlarge",
        instance_count=1,
        volume_size_gb=30,
        max_runtime_seconds=3600,
        approval_ttl_seconds=86400,
        estimated_run_cost_usd=1.0,
        checkpoint_s3_uri="s3://bucket/checkpoint?versionId=v1",
        checkpoint_sha256=CHECKPOINT,
        hf_revision=REVISION,
    )
    app.state.live_preflight = lambda: SimpleNamespace(ready=True)
    app.state.live_supervisor = SimpleNamespace()
    app.state.live_dispatcher = SimpleNamespace(dispatch_once=lambda: [])
    install_autonomous_live_api(app)
    return app


def _token(run_id: str = "run-1") -> str:
    packet = ApprovalPacket(
        run_id=run_id,
        run_number=1,
        instance_type="ml.g5.xlarge",
        instance_count=1,
        volume_size_gb=30,
        max_runtime_seconds=3600,
        estimated_cost_usd=1.0,
        immutable_model_revision=REVISION,
        max_experiments=5,
        max_cost_usd=25.0,
        target_model="google/functiongemma-270m-it",
        objective_suite="AgentGym/AgentEval",
        objective_suite_version="agent-eval-v1",
        seed=7,
        benchmark_id="service-recovery-v1",
        baseline_episodes=10,
        held_out_episodes=15,
        checkpoint_s3_uri="s3://bucket/checkpoint?versionId=v1",
        manifest_sha256=MANIFEST,
        checkpoint_sha256=CHECKPOINT,
        issued_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    return issue_approval_token(packet, "test-secret")


def _approval_scope(packet: ApprovalPacket) -> dict[str, object]:
    return {
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
        "packet_sha256": packet.digest,
    }


@pytest.mark.anyio
async def test_live_api_start_consumes_approval_and_never_requires_step() -> None:
    app = _app()
    repository = app.state.live_repository
    prepared = _state()
    token = _token()
    packet = _decode_approval_token(token, "test-secret")
    repository.create(prepared.model_copy(update={"approval_scope": _approval_scope(packet)}))
    dispatch_calls: list[str] = []

    async def dispatch_once() -> list[str]:
        dispatch_calls.append("dispatch")
        return ["run-1"]

    app.state.live_dispatcher.dispatch_once = dispatch_once
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/live/runs/run-1/start",
            headers={"Idempotency-Key": "idem-1"},
            json={"approval_token": token},
        )
        duplicate = await client.post(
            "/api/live/runs/run-1/start",
            headers={"Idempotency-Key": "idem-2"},
            json={"approval_token": token},
        )

    assert response.status_code == 202
    assert response.json()["status"] == "QUEUED"
    assert duplicate.status_code == 409
    assert repository.get("run-1").approval_consumed is True
    assert dispatch_calls == ["dispatch"]


@pytest.mark.anyio
async def test_start_rejects_tampered_hmac_without_consuming_approval() -> None:
    app = _app()
    repository = app.state.live_repository
    token = _token()
    packet = _decode_approval_token(token, "test-secret")
    repository.create(_state(approval_scope=_approval_scope(packet)))
    tampered = token[:-1] + ("0" if token[-1] != "0" else "1")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/live/runs/run-1/start",
            headers={"Idempotency-Key": "tampered-approval"},
            json={"approval_token": tampered},
        )

    assert response.status_code == 424
    state = repository.get("run-1")
    assert state is not None
    assert state.status is AutonomousRunStatus.PREPARED
    assert state.approval_consumed is False


@pytest.mark.anyio
async def test_start_replays_persisted_response_before_rechecking_consumed_approval() -> None:
    app = _app()
    repository = app.state.live_repository
    token = _token()
    packet = _decode_approval_token(token, "test-secret")
    repository.create(_state(approval_scope=_approval_scope(packet)))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        first = await client.post(
            "/api/live/runs/run-1/start",
            headers={"Idempotency-Key": "start-replay"},
            json={"approval_token": token},
        )
        replay = await client.post(
            "/api/live/runs/run-1/start",
            headers={"Idempotency-Key": "start-replay"},
            json={"approval_token": token},
        )
        changed = await client.post(
            "/api/live/runs/run-1/start",
            headers={"Idempotency-Key": "start-replay"},
            json={"approval_token": "different-token"},
        )

    assert first.status_code == 202
    assert replay.status_code == 202
    assert replay.json() == first.json()
    assert changed.status_code == 409
    assert repository.get("run-1").version == 2


@pytest.mark.anyio
async def test_start_recovers_pending_key_after_approval_was_consumed_and_run_queued() -> None:
    app = _app()
    repository = app.state.live_repository
    token = _token()
    packet = _decode_approval_token(token, "test-secret")
    repository.create(_state(approval_scope=_approval_scope(packet)))
    original_complete = repository.complete_idempotency
    fail_first_completion = True

    def complete_once_then_fail(
        operation: str, key: str, request_digest: str, response: object
    ) -> object:
        nonlocal fail_first_completion
        if operation == "start" and fail_first_completion:
            fail_first_completion = False
            raise RepositoryError("simulated interruption before idempotency completion")
        return original_complete(operation, key, request_digest, response)  # type: ignore[arg-type]

    repository.complete_idempotency = complete_once_then_fail
    dispatch_calls: list[str] = []

    async def dispatch_once() -> list[str]:
        dispatch_calls.append("dispatch")
        return ["run-1"]

    app.state.live_dispatcher.dispatch_once = dispatch_once
    headers = {"Idempotency-Key": "recover-start"}
    body = {"approval_token": token}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        interrupted = await client.post("/api/live/runs/run-1/start", headers=headers, json=body)
        recovered = await client.post("/api/live/runs/run-1/start", headers=headers, json=body)

    assert interrupted.status_code == 424
    assert recovered.status_code == 202, recovered.text
    assert recovered.json() == {
        "run_id": "run-1",
        "status": "QUEUED",
        "phase": "QUEUED",
        "version": 2,
        "idempotency_key": "recover-start",
        "stop_reason": None,
    }
    assert repository.get("run-1").approval_consumed is True
    assert [event.event_type for event in repository.list_events("run-1")] == [
        "approval.consumed",
        "run.queued",
    ]
    assert dispatch_calls == ["dispatch"]


@pytest.mark.anyio
async def test_start_recovers_pending_claim_after_approval_consumption() -> None:
    app = _app()
    repository = app.state.live_repository
    token = _token()
    packet = _decode_approval_token(token, "test-secret")
    repository.create(_state(approval_scope=_approval_scope(packet)))
    original_transition = repository.transition
    fail_first_queue_transition = True

    def transition_once_then_fail(*args: object, **kwargs: object) -> AutonomousRunState:
        nonlocal fail_first_queue_transition
        if kwargs.get("event_type") == "run.queued" and fail_first_queue_transition:
            fail_first_queue_transition = False
            raise RepositoryError("simulated interruption after approval consumption")
        return original_transition(*args, **kwargs)

    repository.transition = transition_once_then_fail
    headers = {"Idempotency-Key": "recover-consume-only"}
    body = {"approval_token": token}
    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        interrupted = await client.post("/api/live/runs/run-1/start", headers=headers, json=body)
        recovered = await client.post("/api/live/runs/run-1/start", headers=headers, json=body)

    assert interrupted.status_code == 500
    assert recovered.status_code == 202, recovered.text
    state = repository.get("run-1")
    assert state is not None
    assert state.status is AutonomousRunStatus.QUEUED
    assert state.version == 2
    assert [event.event_type for event in repository.list_events("run-1")] == [
        "approval.consumed",
        "run.queued",
    ]


@pytest.mark.anyio
async def test_prepare_returns_canonical_packet_for_external_signing() -> None:
    app = _app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        prepared = await client.post(
            "/api/live/runs/prepare",
            headers={"Idempotency-Key": "prepare-1"},
            json={
                "run_id": "run-prepare",
                "checkpoint_revision": REVISION,
                "checkpoint_sha256": CHECKPOINT,
                "checkpoint_s3_uri": "s3://bucket/checkpoint?versionId=v1",
                "benchmark_manifest_sha256": MANIFEST,
            },
        )

    assert prepared.status_code == 201
    body = prepared.json()
    packet = ApprovalPacket.model_validate(body["approval_packet"])
    assert packet.digest == body["packet_sha256"]
    assert packet.expires_at - packet.issued_at == timedelta(days=1)
    assert app.state.live_repository.get("run-prepare").status is AutonomousRunStatus.PREPARED


@pytest.mark.anyio
async def test_prepare_rejects_approval_expiry_shorter_than_full_bounded_run() -> None:
    app = _app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        prepared = await client.post(
            "/api/live/runs/prepare",
            headers={"Idempotency-Key": "prepare-short-approval"},
            json={
                "run_id": "run-short-approval",
                "checkpoint_revision": REVISION,
                "checkpoint_sha256": CHECKPOINT,
                "checkpoint_s3_uri": "s3://bucket/checkpoint?versionId=v1",
                "benchmark_manifest_sha256": MANIFEST,
                "expires_at": (datetime.now(UTC) + timedelta(hours=5)).isoformat(),
            },
        )

    assert prepared.status_code == 424
    assert "shorter than the bounded" in prepared.json()["detail"]


@pytest.mark.anyio
async def test_prepare_replays_pending_claim_after_run_was_created() -> None:
    app = _app()
    repository = app.state.live_repository
    original_complete = repository.complete_idempotency
    fail_first_completion = True
    attempted_responses: list[dict[str, object]] = []

    def complete_once_then_fail(
        operation: str, key: str, request_digest: str, response: Mapping[str, object]
    ) -> object:
        nonlocal fail_first_completion
        if operation == "prepare" and fail_first_completion:
            fail_first_completion = False
            attempted_responses.append(dict(response))
            raise RepositoryError("simulated interruption before idempotency completion")
        return original_complete(operation, key, request_digest, response)

    repository.complete_idempotency = complete_once_then_fail
    headers = {"Idempotency-Key": "recover-prepare"}
    body = {
        "checkpoint_revision": REVISION,
        "checkpoint_sha256": CHECKPOINT,
        "checkpoint_s3_uri": "s3://bucket/checkpoint?versionId=v1",
        "benchmark_manifest_sha256": MANIFEST,
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        interrupted = await client.post("/api/live/runs/prepare", headers=headers, json=body)
        recovered = await client.post("/api/live/runs/prepare", headers=headers, json=body)

    assert interrupted.status_code == 424
    assert recovered.status_code == 201, recovered.text
    assert attempted_responses == [recovered.json()]
    assert app.state.live_repository.get(recovered.json()["run_id"]) is not None


@pytest.mark.anyio
async def test_start_accepts_approval_bounds_narrower_than_deployment_caps() -> None:
    app = _app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        prepared = await client.post(
            "/api/live/runs/prepare",
            headers={"Idempotency-Key": "prepare-narrow"},
            json={
                "run_id": "run-narrow",
                "max_experiments": 3,
                "max_cost_usd": 10,
                "checkpoint_revision": REVISION,
                "checkpoint_sha256": CHECKPOINT,
                "checkpoint_s3_uri": "s3://bucket/checkpoint?versionId=v1",
                "benchmark_manifest_sha256": MANIFEST,
            },
        )
        token = issue_approval_token(
            ApprovalPacket.model_validate(prepared.json()["approval_packet"]), "test-secret"
        )
        started = await client.post(
            "/api/live/runs/run-narrow/start",
            headers={"Idempotency-Key": "start-narrow"},
            json={"approval_token": token},
        )

    assert prepared.status_code == 201, prepared.text
    assert started.status_code == 202, started.text
    assert app.state.live_repository.get("run-narrow").approved_budget_usd == 10


@pytest.mark.anyio
async def test_prepare_replays_same_request_and_rejects_key_reuse_for_changed_request() -> None:
    app = _app()
    body = {
        "run_id": "run-idempotent",
        "checkpoint_revision": REVISION,
        "checkpoint_sha256": CHECKPOINT,
        "checkpoint_s3_uri": "s3://bucket/checkpoint?versionId=v1",
        "benchmark_manifest_sha256": MANIFEST,
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        first = await client.post(
            "/api/live/runs/prepare", headers={"Idempotency-Key": "prepare-replay"}, json=body
        )
        replay = await client.post(
            "/api/live/runs/prepare", headers={"Idempotency-Key": "prepare-replay"}, json=body
        )
        changed = await client.post(
            "/api/live/runs/prepare",
            headers={"Idempotency-Key": "prepare-replay"},
            json={**body, "seed": 8},
        )

    assert first.status_code == 201
    assert replay.status_code == 201
    assert replay.json() == first.json()
    assert changed.status_code == 409
    assert app.state.live_repository.get("run-idempotent") is not None


@pytest.mark.anyio
async def test_live_mutation_fails_closed_without_durable_idempotency_support() -> None:
    app = _app()
    app.state.live_repository = object()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/live/runs/prepare",
            headers={"Idempotency-Key": "no-store"},
            json={"run_id": "must-not-be-created"},
        )

    assert response.status_code == 424
    assert "idempotency" in response.text.lower()


@pytest.mark.anyio
async def test_control_idempotency_key_cannot_switch_from_cancel_to_safe_stop() -> None:
    repository = InMemoryAutonomousRunRepository()
    app = _app(repository)
    repository.create(_state(status=AutonomousRunStatus.QUEUED, phase=RunPhase.QUEUED))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        cancelled = await client.post(
            "/api/live/runs/run-1/cancel", headers={"Idempotency-Key": "control-reuse"}
        )
        changed = await client.post(
            "/api/live/runs/run-1/safe-stop", headers={"Idempotency-Key": "control-reuse"}
        )

    assert cancelled.status_code == 202, cancelled.text
    assert changed.status_code == 409
    state = repository.get("run-1")
    assert state is not None
    assert state.cancellation_requested is True
    assert state.safe_stop_requested is False


@pytest.mark.anyio
async def test_control_retries_pending_key_without_duplicate_control_event() -> None:
    repository = InMemoryAutonomousRunRepository()
    app = _app(repository)
    repository.create(_state(status=AutonomousRunStatus.QUEUED, phase=RunPhase.QUEUED))
    original_complete = repository.complete_idempotency
    fail_first_completion = True

    def complete_once_then_fail(
        operation: str, key: str, request_digest: str, response: Mapping[str, object]
    ) -> object:
        nonlocal fail_first_completion
        if operation == "control:run-1" and fail_first_completion:
            fail_first_completion = False
            raise RepositoryError("simulated interruption after control mutation")
        return original_complete(operation, key, request_digest, response)

    repository.complete_idempotency = complete_once_then_fail
    headers = {"Idempotency-Key": "recover-control"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        interrupted = await client.post("/api/live/runs/run-1/cancel", headers=headers)
        recovered = await client.post("/api/live/runs/run-1/cancel", headers=headers)

    assert interrupted.status_code == 424
    assert recovered.status_code == 202, recovered.text
    state = repository.get("run-1")
    assert state is not None and state.cancellation_requested is True
    assert [event.event_type for event in repository.list_events("run-1")] == [
        "run.cancel_requested"
    ]


@pytest.mark.anyio
async def test_main_live_mutation_is_blocked_without_real_adapters() -> None:
    app = FastAPI()
    install_autonomous_live_api(app)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/live/runs",
            headers={"Idempotency-Key": "blocked-1"},
            json={"run_id": "run-1", "approval_token": "opaque"},
        )

    assert response.status_code == 424
    assert "fallback" not in response.text.lower()


@pytest.mark.anyio
async def test_live_api_exposes_state_events_experiments_and_artifacts() -> None:
    app = _app()
    repository = app.state.live_repository
    repository.create(
        _state(
            status=AutonomousRunStatus.QUEUED,
            phase=RunPhase.QUEUED,
            baseline_artifact_ids=("baseline-artifact",),
            champion_artifact_ids=("champion-artifact", "baseline-artifact"),
        )
    )
    repository.request_safe_stop("run-1", expected_version=0)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        state = await client.get("/api/live/runs/run-1")
        events = await client.get("/api/live/runs/run-1/events")
        experiments = await client.get("/api/live/runs/run-1/experiments")
        artifacts = await client.get("/api/live/runs/run-1/artifacts")

    assert state.status_code == 200
    assert state.json()["run_id"] == "run-1"
    assert events.status_code == 200
    assert events.json()["items"][0]["event_type"] == "run.safe_stop_requested"
    assert experiments.status_code == 200
    assert artifacts.status_code == 200
    assert artifacts.json()["items"] == [
        {"artifact_id": "baseline-artifact"},
        {"artifact_id": "champion-artifact"},
    ]


@pytest.mark.anyio
async def test_live_api_control_calls_are_idempotent_and_metadata_only() -> None:
    repository = InMemoryAutonomousRunRepository()
    app = _app(repository)
    repository.create(_state(status=AutonomousRunStatus.QUEUED, phase=RunPhase.QUEUED))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        cancel = await client.post(
            "/api/live/runs/run-1/cancel", headers={"Idempotency-Key": "cancel-1"}
        )
        repeat = await client.post(
            "/api/live/runs/run-1/cancel", headers={"Idempotency-Key": "cancel-1"}
        )

    assert cancel.status_code == 202
    assert repeat.status_code == 202
    assert repeat.json() == cancel.json()
    assert repository.get("run-1").cancellation_requested is True
    assert "prompt" not in repeat.text.lower()


@pytest.mark.anyio
async def test_safe_stop_replays_same_response_and_rejects_cancel_key_reuse() -> None:
    repository = InMemoryAutonomousRunRepository()
    app = _app(repository)
    repository.create(_state(status=AutonomousRunStatus.QUEUED, phase=RunPhase.QUEUED))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        first = await client.post(
            "/api/live/runs/run-1/safe-stop", headers={"Idempotency-Key": "stop-replay"}
        )
        replay = await client.post(
            "/api/live/runs/run-1/safe-stop", headers={"Idempotency-Key": "stop-replay"}
        )
        switched = await client.post(
            "/api/live/runs/run-1/cancel", headers={"Idempotency-Key": "stop-replay"}
        )

    assert first.status_code == 202
    assert replay.status_code == 202
    assert replay.json() == first.json()
    assert switched.status_code == 409
    state = repository.get("run-1")
    assert state is not None
    assert state.safe_stop_requested is True
    assert state.cancellation_requested is False


@pytest.mark.anyio
async def test_lifespan_runs_nonblocking_periodic_recovery_and_stops_it() -> None:
    repository = InMemoryAutonomousRunRepository()
    repository.create(_state(status=AutonomousRunStatus.QUEUED, phase=RunPhase.QUEUED))
    calls: list[str] = []
    first_recovery_started = asyncio.Event()
    release_first_recovery = asyncio.Event()
    second_recovery_started = asyncio.Event()

    class Dispatcher:
        async def recover_incomplete_runs(self) -> list[str]:
            calls.append("recover")
            if len(calls) == 1:
                first_recovery_started.set()
                await release_first_recovery.wait()
            elif len(calls) == 2:
                second_recovery_started.set()
            return ["run-1"]

        async def shutdown(self) -> None:
            calls.append("shutdown")

    app = _app(repository)
    app.state.live_dispatcher = Dispatcher()
    app.state.live_recovery_interval_seconds = 0.01
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(first_recovery_started.wait(), timeout=1)
        # Startup must complete while the initial durable scan is still active.
        assert calls == ["recover"]
        release_first_recovery.set()
        await asyncio.wait_for(second_recovery_started.wait(), timeout=1)
    assert calls.count("recover") >= 2
    assert calls[-1] == "shutdown"
