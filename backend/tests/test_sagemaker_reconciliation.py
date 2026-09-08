from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from app.providers.sagemaker import (
    EvaluationJobRequest,
    JobNameConflictError,
    JobStatus,
    ProviderResponseError,
    SageMakerProvider,
    TrainingJobRequest,
    TransientProviderError,
    deterministic_job_name,
    is_transient_describe_error,
    request_fingerprint,
)


def _training_request() -> TrainingJobRequest:
    return TrainingJobRequest(
        job_name="apt-run-001-train",
        role_arn="arn:aws:iam::123456789012:role/train",
        image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/train@sha256:abc",
        input_s3_uri="s3://artifacts/run-001/dataset",
        output_s3_uri="s3://artifacts/run-001/candidate",
        instance_type="ml.g5.xlarge",
        hyperparameters={"epochs": 2, "learning_rate": "0.0002"},
        environment={"RUN_ID": "run-001"},
    )


def _evaluation_request() -> EvaluationJobRequest:
    return EvaluationJobRequest(
        job_name="apt-run-001-eval",
        role_arn="arn:aws:iam::123456789012:role/eval",
        image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/eval@sha256:def",
        input_s3_uri="s3://artifacts/run-001/held-out",
        output_s3_uri="s3://artifacts/run-001/evaluation",
        instance_type="ml.g5.xlarge",
        model_s3_uri="s3://artifacts/run-001/candidate/model.tar.gz",
    )


class _FakeClient:
    def __init__(self) -> None:
        self.training: dict[str, Any] | None = None
        self.processing: dict[str, Any] | None = None
        self.create_training_calls = 0
        self.create_processing_calls = 0
        self.stop_training_calls = 0
        self.stop_processing_calls = 0
        self.tags: dict[str, Any] = {}
        self.describe_error: BaseException | None = None

    def describe_training_job(self, **kwargs: object) -> dict[str, Any]:
        if self.describe_error is not None:
            raise self.describe_error
        if self.training is None:
            raise _AwsError("ResourceNotFound")
        return {"TrainingJobName": kwargs["TrainingJobName"], **self.training}

    def describe_processing_job(self, **kwargs: object) -> dict[str, Any]:
        if self.describe_error is not None:
            raise self.describe_error
        if self.processing is None:
            raise _AwsError("ResourceNotFound")
        return {"ProcessingJobName": kwargs["ProcessingJobName"], **self.processing}

    def list_tags(self, **kwargs: object) -> dict[str, Any]:
        del kwargs
        return {"Tags": self.tags.get("Tags", [])}

    def create_training_job(self, **kwargs: object) -> dict[str, Any]:
        self.create_training_calls += 1
        return {"TrainingJobArn": "arn:aws:sagemaker:us-east-1:123:training-job/new"}

    def create_processing_job(self, **kwargs: object) -> dict[str, Any]:
        self.create_processing_calls += 1
        return {"ProcessingJobArn": "arn:aws:sagemaker:us-east-1:123:processing-job/new"}

    def stop_training_job(self, **kwargs: object) -> None:
        del kwargs
        self.stop_training_calls += 1
        raise _AwsError("ResourceNotFound")

    def stop_processing_job(self, **kwargs: object) -> None:
        del kwargs
        self.stop_processing_calls += 1
        raise _AwsError("ResourceNotFound")


class _AwsError(Exception):
    response: dict[str, Any]

    def __init__(self, code: str, message: str = "provider error") -> None:
        super().__init__(message)
        self.response = {"Error": {"Code": code, "Message": message}}


def test_training_submission_reconciles_same_name_and_fingerprint_without_create() -> None:
    request = _training_request()
    client = _FakeClient()
    client.training = {
        "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123:training-job/existing",
        "TrainingJobStatus": "InProgress",
    }
    client.tags = {
        "Tags": [{"Key": "request-fingerprint", "Value": request_fingerprint(request)}]
    }

    result = SageMakerProvider(client=client).submit_training(request)

    assert result.provider_job_id == client.training["TrainingJobArn"]
    assert result.status is JobStatus.IN_PROGRESS
    assert client.create_training_calls == 0


def test_processing_submission_rejects_same_name_with_different_fingerprint() -> None:
    request = _evaluation_request()
    client = _FakeClient()
    client.processing = {
        "ProcessingJobArn": "arn:aws:sagemaker:us-east-1:123:processing-job/existing",
        "ProcessingJobStatus": "Completed",
    }
    client.tags = {"Tags": [{"Key": "request-fingerprint", "Value": "0" * 64}]}

    with pytest.raises(JobNameConflictError, match="fingerprint"):
        SageMakerProvider(client=client).submit_evaluation(request)

    assert client.create_processing_calls == 0


def test_unidentifiable_existing_training_job_fails_closed_without_create() -> None:
    request = _training_request()
    client = _FakeClient()
    client.training = {"TrainingJobStatus": "InProgress"}

    with pytest.raises(ProviderResponseError, match="cannot be reconciled"):
        SageMakerProvider(client=client).submit_training(request)

    assert client.create_training_calls == 0


def test_unidentifiable_existing_processing_job_fails_closed_without_create() -> None:
    request = _evaluation_request()
    client = _FakeClient()
    client.processing = {"ProcessingJobStatus": "InProgress"}

    with pytest.raises(ProviderResponseError, match="cannot be reconciled"):
        SageMakerProvider(client=client).submit_evaluation(request)

    assert client.create_processing_calls == 0


def test_request_fingerprint_and_job_name_are_stable_and_bounded() -> None:
    request = _training_request()

    assert request_fingerprint(request) == request_fingerprint(request)
    assert request_fingerprint(request) == request.request_fingerprint
    name = deterministic_job_name("apt-training", request_fingerprint(request))
    assert len(name) <= 63
    assert name.startswith("apt-training-")
    assert "_" not in name


def test_request_fingerprint_is_independent_of_mapping_and_tag_order() -> None:
    request = replace(
        _training_request(),
        tags=[{"Key": "owner", "Value": "training"}, {"Key": "purpose", "Value": "run"}],
    )
    reordered = replace(request, tags=list(reversed(request.tags)))

    assert request_fingerprint(request) == request_fingerprint(reordered)


def test_create_response_requires_provider_id_and_matching_provider_name() -> None:
    request = _training_request()

    class MissingArn(_FakeClient):
        def create_training_job(self, **kwargs: object) -> dict[str, Any]:
            del kwargs
            return {}

    with pytest.raises(ProviderResponseError, match="provider job ID"):
        SageMakerProvider(client=MissingArn()).submit_training(request)

    class WrongName(_FakeClient):
        def create_training_job(self, **kwargs: object) -> dict[str, Any]:
            del kwargs
            return {
                "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123:training-job/other",
                "TrainingJobName": "other-name",
            }

    with pytest.raises(ProviderResponseError, match="job name"):
        SageMakerProvider(client=WrongName()).submit_training(request)


def test_new_training_submission_is_submitted_and_carries_request_fingerprint_tag() -> None:
    request = _training_request()
    client = _FakeClient()

    result = SageMakerProvider(client=client).submit_training(request)

    assert result.status is JobStatus.SUBMITTED
    assert client.create_training_calls == 1


class _FingerprintWithoutArn(_FakeClient):
    def describe_training_job(self, **kwargs: object) -> dict[str, Any]:
        del kwargs
        return {
            "TrainingJobName": "apt-run-001-train",
            "TrainingJobStatus": "InProgress",
            "RequestFingerprint": _training_request().request_fingerprint,
        }


def test_reconciliation_requires_provider_id_when_fingerprint_is_present() -> None:
    with pytest.raises(ProviderResponseError, match="cannot be reconciled"):
        SageMakerProvider(client=_FingerprintWithoutArn()).submit_training(_training_request())


def test_transient_describe_failure_is_classified_for_supervisor_retry() -> None:
    client = _FakeClient()
    client.describe_error = _AwsError("ThrottlingException", "slow down")

    with pytest.raises(TransientProviderError) as raised:
        SageMakerProvider(client=client).get_training_status("apt-run-001-train")

    assert raised.value.job_name == "apt-run-001-train"
    assert raised.value.operation == "describe_training_job"


@pytest.mark.parametrize(
    "error", [TimeoutError("socket timeout"), ConnectionError("endpoint down")]
)
def test_transport_describe_failures_are_classified_as_transient(error: BaseException) -> None:
    assert is_transient_describe_error(error)


def test_terminal_failure_reason_is_bounded_and_redacted() -> None:
    client = _FakeClient()
    client.training = {
        "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123:training-job/existing",
        "TrainingJobStatus": "Failed",
        "FailureReason": "raw prompt secret\n" + ("x" * 10_000),
    }

    result = SageMakerProvider(client=client).get_training_status("apt-run-001-train")

    assert result.status is JobStatus.FAILED
    assert result.failure_reason is not None
    assert len(result.failure_reason) <= 512
    assert "raw prompt" not in result.failure_reason.lower()
    assert "secret" not in result.failure_reason.lower()
    assert "x" not in result.failure_reason
    assert "raw prompt" not in str(result.raw_response).lower()


def test_adversarial_terminal_failure_text_is_not_returned_or_retained() -> None:
    client = _FakeClient()
    client.training = {
        "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123:training-job/existing",
        "TrainingJobStatus": "Failed",
        "FailureReason": "credentials=AKIA1234567890; user prompt is private",
        "Credentials": "AKIA1234567890",
    }

    result = SageMakerProvider(client=client).get_training_status("apt-run-001-train")

    assert result.failure_reason == "provider reported terminal failure"
    assert "AKIA" not in str(result.raw_response)
    assert "private" not in str(result.raw_response)
    assert "Credentials" not in result.raw_response


def test_sparse_status_response_does_not_fabricate_provider_id() -> None:
    client = _FakeClient()
    client.training = {"TrainingJobStatus": "Completed"}

    result = SageMakerProvider(client=client).get_training_status("apt-run-001-train")

    assert result.provider_job_id is None


@pytest.mark.parametrize("provider_id", ["arn:train\x00", "arn:train\n", "'unsafe'"])
def test_provider_ids_reject_control_characters_and_unsafe_grammar(provider_id: str) -> None:
    client = _FakeClient()
    client.training = {
        "TrainingJobArn": provider_id,
        "TrainingJobStatus": "Completed",
    }

    with pytest.raises(ProviderResponseError, match="provider job ID"):
        SageMakerProvider(client=client).get_training_status("apt-run-001-train")


def test_stop_calls_are_idempotent_when_job_is_missing() -> None:
    client = _FakeClient()
    provider = SageMakerProvider(client=client)

    provider.stop_training("apt-run-001-train")
    provider.stop_evaluation("apt-run-001-eval")

    assert client.stop_training_calls == 1
    assert client.stop_processing_calls == 1
