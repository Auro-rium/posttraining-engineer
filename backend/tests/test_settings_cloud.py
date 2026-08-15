from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.cloud import CloudIntegrationError, VertexTrainingLauncher, safe_cloud_metadata
from app.settings import Settings


def test_local_settings_are_runnable_without_credentials() -> None:
    settings = Settings(_env_file=None)

    assert settings.environment == "local"
    assert settings.max_candidates == 2
    assert settings.target_model == "google/functiongemma-270m-it"


def test_cloud_settings_fail_closed_without_coordinates() -> None:
    with pytest.raises(ValidationError, match="cloud mode requires"):
        Settings(environment="cloud", _env_file=None)


def test_telemetry_content_capture_cannot_be_enabled() -> None:
    with pytest.raises(ValidationError, match="OTEL_CAPTURE_CONTENT"):
        Settings(otel_capture_content=True, _env_file=None)


def test_vertex_launcher_rejects_local_mode() -> None:
    with pytest.raises(CloudIntegrationError, match="cloud mode"):
        VertexTrainingLauncher(Settings(_env_file=None))


def test_cloud_metadata_is_reduced_to_scalars() -> None:
    assert safe_cloud_metadata({"state": "RUNNING"}) == "{'state': 'RUNNING'}"
    assert safe_cloud_metadata(7) == 7


def test_cloud_research_service_requires_rag_corpus() -> None:
    with pytest.raises(ValidationError, match="RAG_CORPUS_URI"):
        Settings(
            environment="cloud",
            service_role="research",
            google_cloud_project="project",
            artifact_bucket="artifacts",
            vertex_staging_bucket="gs://staging",
            training_container_uri="registry/trainer:sha",
            objective_execution_url="https://objective.run.app",
            _env_file=None,
        )

    settings = Settings(
        environment="cloud",
        service_role="research",
        google_cloud_project="project",
        artifact_bucket="artifacts",
        vertex_staging_bucket="gs://staging",
        training_container_uri="registry/trainer:sha",
        objective_execution_url="https://objective.run.app",
        rag_corpus_uri="gs://artifacts/rag/corpus.json",
        rag_corpus_sha256="a" * 64,
        _env_file=None,
    )
    assert settings.a2a_timeout_seconds == 3300.0
