"""Credential-free interfaces and optional AWS-backed provider adapters.

The package deliberately does not import boto3 or Strands at module import time.
This keeps local schema/orchestration tests usable without AWS credentials (or
without the optional SDKs installed).
"""

from .artifacts import ArtifactRef, ArtifactStore, S3ArtifactProvider, S3ArtifactStore
from .bedrock import (
    SIGV4_AUTH_MODE,
    BedrockAuthMode,
    BedrockModelProvider,
    BedrockStrandsModel,
)
from .repository import (
    ConcurrentUpdateError,
    DynamoDBRunRepository,
    DynamoDBStateRepository,
    RunEvent,
    RunRecord,
    RunRepository,
)
from .sagemaker import (
    EvaluationJobRequest,
    EvaluationProvider,
    JobStatus,
    SageMakerEvaluationProvider,
    SageMakerProvider,
    SageMakerTrainingProvider,
    TrainingJobRequest,
    TrainingProvider,
)

__all__ = [
    "SIGV4_AUTH_MODE",
    "ArtifactRef",
    "ArtifactStore",
    "BedrockAuthMode",
    "BedrockModelProvider",
    "BedrockStrandsModel",
    "ConcurrentUpdateError",
    "DynamoDBRunRepository",
    "DynamoDBStateRepository",
    "EvaluationJobRequest",
    "EvaluationProvider",
    "JobStatus",
    "RunEvent",
    "RunRecord",
    "RunRepository",
    "S3ArtifactProvider",
    "S3ArtifactStore",
    "SageMakerEvaluationProvider",
    "SageMakerProvider",
    "SageMakerTrainingProvider",
    "TrainingJobRequest",
    "TrainingProvider",
]
