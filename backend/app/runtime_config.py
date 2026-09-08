"""Validated runtime configuration for local and AWS deployments."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    strands_model: str = "nvidia.nemotron-super-3-120b"
    target_model: str = "google/functiongemma-270m-it"
    max_experiments: int = Field(default=2, ge=1, le=2)
    max_cost_usd: float = Field(default=50.0, ge=0)
    max_training_time_min: int = Field(default=120, ge=1)

    s3_artifact_bucket: str | None = None
    s3_artifact_prefix: str = "post-training"
    dynamodb_table_name: str | None = None
    sagemaker_training_role_arn: str | None = None
    sagemaker_training_image_uri: str | None = None
    sagemaker_evaluation_image_uri: str | None = None
    agentcore_enabled: bool = False

    @model_validator(mode="after")
    def validate_aws_requirements(self) -> RuntimeConfig:
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
