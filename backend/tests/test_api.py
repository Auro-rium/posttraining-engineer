from __future__ import annotations

import time
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from app.cloud_provider import CloudProviderError
from app.main import create_app
from app.models import EvidenceLabel
from app.settings import Settings


def _client() -> TestClient:
    return TestClient(create_app(settings=Settings(environment="test")))


def _create_run(client: TestClient) -> str:
    response = client.post("/api/runs", json={})
    assert response.status_code == 201
    body = response.json()
    assert body["phase"] == "NOT_STARTED"
    assert body["evidence_label"] == EvidenceLabel.EXPLANATION
    return str(body["run_id"])


def _wait_for_terminal(client: TestClient, run_id: str) -> dict[str, Any]:
    for _ in range(200):
        body = client.get(f"/api/runs/{run_id}").json()
        if body["phase"] in {"COMPLETED", "FAILED", "CANCELLED"}:
            return cast(dict[str, Any], body)
        time.sleep(0.005)
    raise AssertionError("background auto run did not become terminal")


def test_create_get_step_and_experiments() -> None:
    with _client() as client:
        assert client.get("/health").json() == {
            "status": "ok",
            "role": "coordinator",
            "mode": "test",
        }
        run_id = _create_run(client)

        state = client.get(f"/api/runs/{run_id}").json()
        assert state["champion"]["success"] == 0.35
        assert state["champion"]["evidence_label"] == "EXPLANATION"

        stepped = client.post(f"/api/runs/{run_id}/step")
        assert stepped.status_code == 200
        assert stepped.json()["phase"] == "ANALYZING"
        assert client.get(f"/api/runs/{run_id}/experiments").json()["experiments"] == []


def test_auto_sse_resume_and_demo_verify() -> None:
    with _client() as client:
        run_id = _create_run(client)
        accepted = client.post(f"/api/runs/{run_id}/auto")
        assert accepted.status_code == 202
        assert accepted.json()["status"] == "accepted"

        state = _wait_for_terminal(client, run_id)
        assert state["phase"] == "COMPLETED"
        assert state["experiments_used"] == 2
        assert all(item["status"] == "REJECTED" for item in state["experiments"])

        stream = client.get(f"/api/runs/{run_id}/events")
        assert stream.status_code == 200
        assert "event: run.created" in stream.text
        assert stream.text.count("event: checkpoint.rejected") == 2
        first_id = next(
            line.removeprefix("id: ")
            for line in stream.text.splitlines()
            if line.startswith("id: ")
        )
        resumed = client.get(f"/api/runs/{run_id}/events", headers={"Last-Event-ID": first_id})
        assert "event: run.created" not in resumed.text
        assert "event: benchmark.completed" in resumed.text

        verification = client.post("/api/demo/verify", json={"run_id": run_id})
        assert verification.status_code == 200
        assert verification.json()["evidence_label"] == "EXPLANATION"
        assert verification.json()["provenance_complete"] is False


def test_cancel_missing_run_and_scope_validation() -> None:
    with _client() as client:
        run_id = _create_run(client)
        cancelled = client.post(f"/api/runs/{run_id}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["phase"] == "CANCELLED"
        assert client.post(f"/api/runs/{run_id}/auto").status_code == 409

        missing = client.get("/api/runs/not-present")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "run_not_found"

        invalid = client.post("/api/runs", json={"target_model": "some-other-model"})
        assert invalid.status_code == 422


def test_cloud_mode_cannot_fall_back_to_explanatory_provider() -> None:
    settings = Settings(
        environment="cloud",
        google_cloud_project="test-project",
        artifact_bucket="test-artifacts",
        vertex_staging_bucket="gs://test-staging",
        training_container_uri="us-docker.pkg.dev/test/trainer:sha",
    )
    with pytest.raises(CloudProviderError, match="A2A_URL"):
        create_app(settings=settings)
