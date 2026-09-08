"""Validated runtime configuration for local and AWS deployments."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.agents.prompt_contract import NEMOTRON_MODEL_ID
from app.providers.bedrock import SIGV4_AUTH_MODE, BedrockAuthMode


class RuntimeConfig(BaseSettings):
    """Configuration shared by the HTTP process and AWS adapters."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_mode: Literal["local", "aws"] = "local"
    service_role: Literal["coordinator", "research", "execution"] = "coordinator"
    aws_region: str = "us-east-1"
    # Keep Bedrock authentication explicit.  SigV4 uses the configured AWS IAM
    # credential chain and is not affected by a stale AWS_BEARER_TOKEN_BEDROCK
    # environment variable.
    bedrock_auth_mode: BedrockAuthMode = SIGV4_AUTH_MODE
    # Reasoning is intentionally pinned: changing this model invalidates prompt
    # provenance and makes comparisons between autonomous runs ambiguous.
    strands_model: str = Field(default=NEMOTRON_MODEL_ID, min_length=1)
    target_model: str = "google/functiongemma-270m-it"
    # The hackathon workflow admits at most five top-level candidate runs.  A
    # hard upper bound here prevents an environment variable or deployment
    # override from silently widening the deterministic run budget.
    max_experiments: int = Field(default=5, ge=1, le=5)
    max_cost_usd: float = Field(default=25.0, ge=0, le=25.0)
    max_training_time_min: int = Field(default=120, ge=1)

    # Objective provenance is part of every comparable evaluation.  Keep the
    # suite and version in configuration so a coordinator cannot accidentally
    # compare metrics from different benchmark contracts.
    objective_suite: str = Field(default="AgentGym/AgentEval", min_length=1)
    objective_suite_version: str = Field(default="agent-eval-v1", min_length=1)

    # Telemetry is metadata-only and can be disabled for an explicitly
    # credential-free local run.  The endpoint is optional because the default
    # logger exporter works without an OTLP collector.
    telemetry_enabled: bool = True
    telemetry_exporter: Literal["logging", "otlp", "none"] = "logging"
    telemetry_otlp_endpoint: str | None = None

    s3_artifact_bucket: str | None = None
    s3_artifact_prefix: str = "post-training"
    dynamodb_table_name: str | None = None
    sagemaker_training_role_arn: str | None = None
    sagemaker_training_image_uri: str | None = None
    sagemaker_evaluation_image_uri: str | None = None
    agentcore_enabled: bool = False

    @model_validator(mode="after")
    def validate_aws_requirements(self) -> RuntimeConfig:
        if self.strands_model != NEMOTRON_MODEL_ID:
            raise ValueError(
                "strands_model is pinned to NVIDIA Nemotron Super 3 120B: "
                f"{NEMOTRON_MODEL_ID}"
            )
        if self.telemetry_exporter == "otlp" and not self.telemetry_otlp_endpoint:
            raise ValueError(
                "telemetry_otlp_endpoint is required when telemetry_exporter=otlp"
            )
        if self.app_mode == "aws":
            required = {
                "s3_artifact_bucket": self.s3_artifact_bucket,
                "dynamodb_table_name": self.dynamodb_table_name,
                "sagemaker_training_role_arn": self.sagemaker_training_role_arn,
                "sagemaker_training_image_uri": self.sagemaker_training_image_uri,
                "sagemaker_evaluation_image_uri": self.sagemaker_evaluation_image_uri,
            }
            missing = sorted(name for name, value in required.items() if not value)
            if missing:
                raise ValueError(
                    "AWS deployment is missing required configuration: " + ", ".join(missing)
                )
        return self


def get_runtime_config() -> RuntimeConfig:
    """Construct configuration at process startup."""

    return RuntimeConfig()
