from __future__ import annotations

from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.state import OptimizationRun
from app.main import (
    _create_application_orchestrator,
    _create_run_registry,
    _record_phase_telemetry,
    _select_application,
    app,
)
from app.observability import EventType, TelemetryRecorder
from app.providers.repository import DynamoDBRunRepository
from app.runtime_config import RuntimeConfig


def test_local_registry_is_process_local_and_bounded() -> None:
    registry = _create_run_registry(SimpleNamespace(app_mode="local"))

    assert registry.list_runs() == ()
    assert registry.max_runs == 5


def test_aws_registry_constructs_without_provisioning_resources() -> None:
    registry = _create_run_registry(
        SimpleNamespace(
            app_mode="aws",
            dynamodb_table_name="existing-post-training-runs",
            aws_region="us-east-1",
        )
    )

    repository = registry._repository
    assert isinstance(repository, DynamoDBRunRepository)
    assert repository.table_name == "existing-post-training-runs"


def test_application_orchestrator_receives_configured_model() -> None:
    configured_model = "nvidia.nemotron-super-3-120b"
    orchestrator = _create_application_orchestrator(
        SimpleNamespace(app_mode="local", aws_region="us-east-1", strands_model=configured_model)
    )

    assert orchestrator.model_id == configured_model
    for agent_name in (
        "benchmark_agent",
        "failure_analyst_agent",
        "research_agent",
        "data_curator_agent",
        "training_designer_agent",
        "training_executor_agent",
        "eval_agent",
        "champion_manager_agent",
    ):
        model = getattr(orchestrator, agent_name).agent.model
        assert getattr(model, "config", {}).get("model_id") == configured_model


@pytest.mark.anyio
async def test_main_exposes_comparison_and_telemetry_readiness() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        health = await client.get("/health")
        comparison = await client.get("/api/runs/compare")

    assert health.status_code == 200
    assert health.json()["components"]["telemetry"] == "ready"
    assert health.json()["components"]["run_history"] == "ready"
    assert health.json()["reasoning_model"] == "nvidia.nemotron-super-3-120b"
    assert comparison.status_code == 200
    assert comparison.json()["run_count"] == 0
    assert type(app.state.telemetry).__name__ == "TelemetryRecorder"


@pytest.mark.anyio
async def test_objective_role_selects_only_authenticated_objective_routes() -> None:
    config = RuntimeConfig(
        _env_file=None,
        app_mode="aws",
        service_role="objective",
        s3_artifact_bucket="objective-artifacts",
        s3_artifact_prefix="objective",
        objective_auth_token="objective-secret",
    )
    objective_app = _select_application(config, app)
    paths = {getattr(route, "path", "") for route in objective_app.routes}

    assert {"/health", "/v1/auth-probe", "/v1/benchmark", "/v1/verify-curation"} <= paths
    assert "/api/runs" not in paths

    async with AsyncClient(
        transport=ASGITransport(app=objective_app), base_url="http://test"
    ) as client:
        health = await client.get("/health")
        unauthorized = await client.get("/v1/auth-probe")
        authenticated = await client.get(
            "/v1/auth-probe", headers={"Authorization": "Bearer objective-secret"}
        )
        benchmark = await client.post(
            "/v1/benchmark",
            json={"run_id": "readiness-probe", "split": "invalid"},
            headers={"Authorization": "Bearer objective-secret"},
        )
        curation = await client.post(
            "/v1/verify-curation",
            json={
                "run_id": "readiness-probe",
                "experiment_id": "readiness-probe",
                "split": "invalid",
            },
            headers={"Authorization": "Bearer objective-secret"},
        )

    assert health.status_code == 200
    assert unauthorized.status_code == 401
    assert authenticated.status_code == 200
    # Invalid split probes establish route+auth capability without running or
    # exposing any objective tasks.
    assert benchmark.status_code == 422
    assert curation.status_code == 422


def test_phase_telemetry_covers_jobs_and_promotion_without_model_content() -> None:
    run = OptimizationRun(
        runId="run-test",
        targetModel="google/functiongemma-270m-it",
        baseCheckpoint="hf://google/functiongemma-270m-it",
        environment="agentgym-service-recovery",
        objective="maximize task success rate",
        budget={},
    )
    events: list[dict[str, object]] = []
    previous_recorder = app.state.telemetry
    previous_numbers = app.state.run_numbers
    app.state.telemetry = TelemetryRecorder(exporter=events.append, logger=None)
    app.state.run_numbers = {run.runId: 3}
    try:
        _record_phase_telemetry(run, "execute_training", None, started=True)
        _record_phase_telemetry(
            run,
            "execute_training",
            {"status": "completed", "output": "job completed"},
        )
        _record_phase_telemetry(
            run,
            "promote_decision",
            {"status": "completed", "output": '{"promotion_decision":"REJECT"}'},
        )
    finally:
        app.state.telemetry = previous_recorder
        app.state.run_numbers = previous_numbers

    event_types = [item["event_type"] for item in events]
    assert event_types == [
        EventType.PHASE_STARTED.value,
        EventType.JOB_SUBMITTED.value,
        EventType.PHASE_COMPLETED.value,
        EventType.JOB_COMPLETED.value,
        EventType.PHASE_COMPLETED.value,
        EventType.PROMOTION_DECIDED.value,
    ]
    assert all("output" not in item for item in events)
    assert all(item["run_number"] == 3 for item in events)
