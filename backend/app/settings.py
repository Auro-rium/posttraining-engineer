"""Runtime configuration with an explicit local-demo versus cloud boundary."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated service configuration.

    Local mode deliberately uses deterministic adapters. Cloud mode fails fast when
    the Google Cloud coordinates required for verifiable evidence are absent.
    """

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "autonomous-post-training-engineer"
    environment: Literal["local", "test", "cloud"] = "local"
    service_role: Literal["coordinator", "research", "execution"] = "coordinator"
    evidence_label: Literal["LIVE", "PRIOR_VERIFIED_RUN", "EXPLANATION"] = "EXPLANATION"
    api_prefix: str = "/api"
    allowed_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    gemini_model: str = "gemini-3.5-flash"
    target_model: str = "google/functiongemma-270m-it"
    environment_name: str = "agentgym-webshop"
    max_candidates: int = Field(default=2, ge=1, le=2)
    compute_budget_minutes: int = Field(default=55, ge=1, le=55)
    a2a_timeout_seconds: float = Field(default=3300.0, ge=1.0, le=3500.0)

    google_cloud_project: str | None = None
    google_cloud_location: str = "us-central1"
    firestore_database: str = "(default)"
    artifact_bucket: str | None = None
    vertex_staging_bucket: str | None = None
    training_container_uri: str | None = None
    hf_secret_id: str | None = None
    research_a2a_url: str | None = None
    execution_a2a_url: str | None = None
    objective_execution_url: str | None = None
    rag_corpus_uri: str | None = None
    rag_corpus_sha256: str | None = None
    public_service_url: str | None = None
    agentgym_base_url: str | None = None
    local_artifact_dir: str = "/tmp/autonomous-post-training-artifacts"

    otel_service_name: str = "post-training-coordinator"
    otel_export_to_cloud: bool = False
    otel_capture_content: bool = False

    @model_validator(mode="after")
    def validate_cloud_settings(self) -> "Settings":
        if self.otel_capture_content:
            raise ValueError(
                "OTEL_CAPTURE_CONTENT must remain false to protect prompts and eval data"
            )
        if self.environment == "cloud":
            missing = [
                name
                for name, value in {
                    "GOOGLE_CLOUD_PROJECT": self.google_cloud_project,
                    "ARTIFACT_BUCKET": self.artifact_bucket,
                    "VERTEX_STAGING_BUCKET": self.vertex_staging_bucket,
                    "TRAINING_CONTAINER_URI": self.training_container_uri,
                }.items()
                if not value
            ]
            if missing:
                raise ValueError(f"cloud mode requires: {', '.join(missing)}")
            if self.service_role in {"research", "execution"} and not self.objective_execution_url:
                raise ValueError(
                    "cloud research/execution services require OBJECTIVE_EXECUTION_URL"
                )
            if self.service_role == "research" and not self.rag_corpus_uri:
                raise ValueError("cloud research service requires RAG_CORPUS_URI")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide immutable settings instance."""

    return Settings()
