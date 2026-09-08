"""Focused contract tests for the isolated continuous post-training API."""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.continuous_post_training import (
    CycleConfig,
    CycleCreate,
    CycleService,
    CycleStatus,
    DecisionRequest,
    InMemoryPostTrainingRepository,
    TraceCreate,
    TraceMessage,
    TraceRole,
    router,
)


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def create_cycle(client: TestClient) -> str:
    trace_response = client.post(
        "/api/traces",
        json={
            "source": "local-test",
            "messages": [{"role": "user", "content": "repair this"}],
        },
    )
    assert trace_response.status_code == 201
    trace_id = trace_response.json()["trace_id"]
    cycle_response = client.post(
        "/api/cycles",
        json={
            "trace_ids": [trace_id],
            "config": {"base_model": "functiongemma", "recipe": "qlora"},
        },
    )
    assert cycle_response.status_code == 201
    return cycle_response.json()["cycle_id"]


def test_trace_cycle_decision_status_events_and_artifacts(client: TestClient) -> None:
    cycle_id = create_cycle(client)

    status_response = client.get(f"/api/cycles/{cycle_id}/status")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == CycleStatus.PENDING_APPROVAL

    events_response = client.get(f"/api/cycles/{cycle_id}/events")
    assert events_response.status_code == 200
    assert [event["sequence"] for event in events_response.json()["events"]] == [1]

    approve_response = client.post(
        f"/api/cycles/{cycle_id}/approve", json={"expected_version": 0}
    )
    assert approve_response.status_code == 200
    assert approve_response.json()["status"] == CycleStatus.QUEUED
    assert approve_response.json()["version"] == 1

    stale_response = client.post(
        f"/api/cycles/{cycle_id}/cancel", json={"expected_version": 0}
    )
    assert stale_response.status_code == 409

    cancel_response = client.post(
        f"/api/cycles/{cycle_id}/cancel", json={"expected_version": 1, "reason": "stop"}
    )
    assert cancel_response.status_code == 200
    assert cancel_response.json()["status"] == CycleStatus.CANCELLED

    tail_response = client.get(f"/api/cycles/{cycle_id}/events?after=1")
    assert [event["sequence"] for event in tail_response.json()["events"]] == [2, 3]
    assert client.get(f"/api/cycles/{cycle_id}/artifacts").json()["artifacts"] == []


def test_reject_requires_reason_and_unknown_trace_is_rejected(client: TestClient) -> None:
    trace_id = uuid4()
    cycle_response = client.post(
        "/api/cycles",
        json={
            "trace_ids": [str(trace_id)],
            "config": {"base_model": "functiongemma", "recipe": "qlora"},
        },
    )
    assert cycle_response.status_code == 422

    cycle_id = create_cycle(client)
    assert client.post(f"/api/cycles/{cycle_id}/reject").status_code == 422
    rejected = client.post(
        f"/api/cycles/{cycle_id}/reject", json={"reason": "held-out policy"}
    )
    assert rejected.status_code == 200
    assert rejected.json()["status"] == CycleStatus.REJECTED
    assert client.post(f"/api/cycles/{cycle_id}/approve").status_code == 422


def test_app_state_repositories_are_isolated() -> None:
    first = FastAPI()
    second = FastAPI()
    first.include_router(router)
    second.include_router(router)
    first_client = TestClient(first)
    second_client = TestClient(second)
    cycle_id = create_cycle(first_client)
    assert second_client.get(f"/api/cycles/{cycle_id}").status_code == 404


@pytest.mark.anyio
async def test_repository_compare_and_set_has_one_winner() -> None:
    repository = InMemoryPostTrainingRepository()
    service = CycleService(repository)
    trace = await service.create_trace(
        TraceCreate(
            source="concurrency",
            messages=[TraceMessage(role=TraceRole.USER, content="x")],
        )
    )
    cycle = await service.create_cycle(
        CycleCreate(
            trace_ids=[UUID(str(trace.trace_id))],
            config=CycleConfig(base_model="m", recipe="r"),
        )
    )

    results = await asyncio.gather(
        service.approve(cycle.cycle_id, DecisionRequest(expected_version=0)),
        service.approve(cycle.cycle_id, DecisionRequest(expected_version=0)),
        return_exceptions=True,
    )
    assert sum(not isinstance(result, Exception) for result in results) == 1
    assert sum(isinstance(result, Exception) for result in results) == 1
    events = await repository.list_events(cycle.cycle_id)
    assert [event.sequence for event in events] == [1, 2]
