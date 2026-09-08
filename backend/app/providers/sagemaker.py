"""SageMaker training and evaluation provider interfaces.

Training uses a SageMaker training job; evaluation uses a processing job with
the same explicit input/output artifact contract.  The client is injected or
created lazily, so importing this module cannot require AWS credentials.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class OptionalDependencyError(RuntimeError):
    """Raised when a live SageMaker client is requested without boto3."""


class JobStatus(StrEnum):
    SUBMITTED = "submitted"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class TrainingJobRequest:
    job_name: str
    role_arn: str
    image_uri: str
    input_s3_uri: str
    output_s3_uri: str
    instance_type: str
    instance_count: int = 1
    volume_size_gb: int = 30
    max_runtime_seconds: int = 3600
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    tags: list[dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class EvaluationJobRequest:
    job_name: str
    role_arn: str
    image_uri: str
    input_s3_uri: str
    output_s3_uri: str
    instance_type: str
    model_s3_uri: str | None = None
    instance_count: int = 1
    volume_size_gb: int = 30
    max_runtime_seconds: int = 3600
    environment: dict[str, str] = field(default_factory=dict)
    command: list[str] | None = None
    tags: list[dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class JobResult:
    job_name: str
    provider_job_id: str | None
    status: JobStatus
    artifact_uri: str | None = None
    failure_reason: str | None = None
    raw_response: Mapping[str, Any] = field(default_factory=dict)


class TrainingProvider(Protocol):
    def submit_training(self, request: TrainingJobRequest) -> JobResult: ...

    def get_training_status(self, job_name: str) -> JobResult: ...

    def stop_training(self, job_name: str) -> None: ...


class EvaluationProvider(Protocol):
    def submit_evaluation(self, request: EvaluationJobRequest) -> JobResult: ...

    def get_evaluation_status(self, job_name: str) -> JobResult: ...

    def stop_evaluation(self, job_name: str) -> None: ...


def _require(value: str, name: str) -> str:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _status(value: object) -> JobStatus:
    normalized = str(value or "").strip().lower()
    if normalized in {"submitted", "pending", "starting"}:
        return JobStatus.SUBMITTED
    if normalized in {"inprogress", "in_progress", "stopping", "downloading", "uploading"}:
        return JobStatus.IN_PROGRESS
    if normalized in {"completed", "succeeded", "success"}:
        return JobStatus.COMPLETED
    if normalized in {"failed", "failure"}:
        return JobStatus.FAILED
    if normalized in {"stopped", "stopping"}:
        return JobStatus.STOPPED
    return JobStatus.UNKNOWN


class SageMakerProvider(TrainingProvider, EvaluationProvider):
    """Concrete SageMaker adapter for training and evaluation jobs."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        region_name: str | None = None,
    ) -> None:
        self._client = client
        self.region_name = region_name

    def _client_or_create(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import boto3  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise OptionalDependencyError("Install boto3 to use SageMakerProvider") from exc
        self._client = boto3.client("sagemaker", region_name=self.region_name)
        return self._client

    @staticmethod
    def _validate_training(request: TrainingJobRequest) -> None:
        for value, name in (
            (request.job_name, "job_name"),
            (request.role_arn, "role_arn"),
            (request.image_uri, "image_uri"),
            (request.input_s3_uri, "input_s3_uri"),
            (request.output_s3_uri, "output_s3_uri"),
            (request.instance_type, "instance_type"),
        ):
            _require(value, name)
        if (
            request.instance_count < 1
            or request.volume_size_gb < 1
            or request.max_runtime_seconds < 1
        ):
            raise ValueError("SageMaker resource sizes and runtime must be positive")

    @staticmethod
    def _validate_evaluation(request: EvaluationJobRequest) -> None:
        for value, name in (
            (request.job_name, "job_name"),
            (request.role_arn, "role_arn"),
            (request.image_uri, "image_uri"),
            (request.input_s3_uri, "input_s3_uri"),
            (request.output_s3_uri, "output_s3_uri"),
            (request.instance_type, "instance_type"),
        ):
            _require(value, name)
        if request.model_s3_uri is not None:
            _require(request.model_s3_uri, "model_s3_uri")
        if (
            request.instance_count < 1
            or request.volume_size_gb < 1
            or request.max_runtime_seconds < 1
        ):
            raise ValueError("SageMaker resource sizes and runtime must be positive")

    def submit_training(self, request: TrainingJobRequest) -> JobResult:
        self._validate_training(request)
        response = self._client_or_create().create_training_job(
            TrainingJobName=request.job_name,
            RoleArn=request.role_arn,
            AlgorithmSpecification={
                "TrainingImage": request.image_uri,
                "TrainingInputMode": "File",
            },
            InputDataConfig=[
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": request.input_s3_uri,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                }
            ],
            OutputDataConfig={"S3OutputPath": request.output_s3_uri},
            ResourceConfig={
                "InstanceType": request.instance_type,
                "InstanceCount": request.instance_count,
                "VolumeSizeInGB": request.volume_size_gb,
            },
            StoppingCondition={"MaxRuntimeInSeconds": request.max_runtime_seconds},
            HyperParameters={str(k): str(v) for k, v in request.hyperparameters.items()},
            Environment=request.environment,
            Tags=request.tags,
        )
        return JobResult(
            job_name=request.job_name,
            provider_job_id=(
                str(response["TrainingJobArn"])
                if response.get("TrainingJobArn")
                else request.job_name
            ),
            status=JobStatus.SUBMITTED,
            raw_response=response,
        )

    def get_training_status(self, job_name: str) -> JobResult:
        _require(job_name, "job_name")
        response = self._client_or_create().describe_training_job(TrainingJobName=job_name)
        model_artifacts = response.get("ModelArtifacts")
        artifact_uri = (
            model_artifacts.get("S3ModelArtifacts")
            if isinstance(model_artifacts, Mapping)
            else None
        )
        return JobResult(
            job_name=job_name,
            provider_job_id=(
                str(response["TrainingJobArn"])
                if response.get("TrainingJobArn")
                else job_name
            ),
            status=_status(response.get("TrainingJobStatus")),
            artifact_uri=str(artifact_uri) if artifact_uri else None,
            failure_reason=(
                str(response["FailureReason"])
                if response.get("FailureReason")
                else None
            ),
            raw_response=response,
        )

    def stop_training(self, job_name: str) -> None:
        self._client_or_create().stop_training_job(TrainingJobName=_require(job_name, "job_name"))

    def submit_training_job(self, request: TrainingJobRequest) -> JobResult:
        """Compatibility alias with the AWS API's job-oriented naming."""
        return self.submit_training(request)

    def get_training_job(self, job_name: str) -> JobResult:
        return self.get_training_status(job_name)

    def submit_evaluation(self, request: EvaluationJobRequest) -> JobResult:
        self._validate_evaluation(request)
        app_spec: dict[str, Any] = {"ImageUri": request.image_uri}
        if request.command:
            app_spec["ContainerEntrypoint"] = request.command
        processing_inputs = [
            {
                "InputName": "evaluation",
                "S3Input": {
                    "S3Uri": request.input_s3_uri,
                    "LocalPath": "/opt/ml/processing/input",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3CompressionType": "None",
                },
            }
        ]
        if request.model_s3_uri:
            processing_inputs.append(
                {
                    "InputName": "model",
                    "S3Input": {
                        "S3Uri": request.model_s3_uri,
                        "LocalPath": "/opt/ml/processing/model",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                        "S3CompressionType": "None",
                    },
                }
            )
        response = self._client_or_create().create_processing_job(
            ProcessingJobName=request.job_name,
            RoleArn=request.role_arn,
            AppSpecification=app_spec,
            ProcessingInputs=processing_inputs,
            ProcessingOutputConfig={
                "Outputs": [
                    {
                        "OutputName": "evaluation",
                        "S3Output": {
                            "S3Uri": request.output_s3_uri,
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob",
                        },
                    }
                ]
            },
            ProcessingResources={
                "ClusterConfig": {
                    "InstanceType": request.instance_type,
                    "InstanceCount": request.instance_count,
                    "VolumeSizeInGB": request.volume_size_gb,
                }
            },
            StoppingCondition={"MaxRuntimeInSeconds": request.max_runtime_seconds},
            Environment=request.environment,
            Tags=request.tags,
        )
        return JobResult(
            job_name=request.job_name,
            provider_job_id=(
                str(response["ProcessingJobArn"])
                if response.get("ProcessingJobArn")
                else request.job_name
            ),
            status=JobStatus.SUBMITTED,
            raw_response=response,
        )

    def get_evaluation_status(self, job_name: str) -> JobResult:
        _require(job_name, "job_name")
        response = self._client_or_create().describe_processing_job(ProcessingJobName=job_name)
        outputs = response.get("ProcessingOutputConfig")
        artifact_uri: str | None = None
        if isinstance(outputs, Mapping):
            listed = outputs.get("Outputs", [])
            if listed and isinstance(listed[0], Mapping):
                s3_output = listed[0].get("S3Output")
                if isinstance(s3_output, Mapping) and s3_output.get("S3Uri"):
                    artifact_uri = str(s3_output["S3Uri"])
        return JobResult(
            job_name=job_name,
            provider_job_id=(
                str(response["ProcessingJobArn"])
                if response.get("ProcessingJobArn")
                else job_name
            ),
            status=_status(response.get("ProcessingJobStatus")),
            artifact_uri=artifact_uri,
            failure_reason=(
                str(response["FailureReason"])
                if response.get("FailureReason")
                else None
            ),
            raw_response=response,
        )

    def stop_evaluation(self, job_name: str) -> None:
        self._client_or_create().stop_processing_job(
            ProcessingJobName=_require(job_name, "job_name")
        )

    def submit_evaluation_job(self, request: EvaluationJobRequest) -> JobResult:
        return self.submit_evaluation(request)

    def get_evaluation_job(self, job_name: str) -> JobResult:
        return self.get_evaluation_status(job_name)


SageMakerTrainingProvider = SageMakerProvider
SageMakerEvaluationProvider = SageMakerProvider
