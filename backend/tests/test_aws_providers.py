from __future__ import annotations

import hashlib
import json
from typing import ClassVar

import pytest

from app.providers.artifacts import ArtifactRef, S3ArtifactStore
from app.providers.bedrock import BedrockStrandsModel
from app.providers.repository import (
    ConcurrentUpdateError,
    DynamoDBRunRepository,
    RunEvent,
    RunRecord,
)
from app.providers.sagemaker import (
    EvaluationJobRequest,
    SageMakerProvider,
    TrainingJobRequest,
)


class FakeBody:
    def __init__(self, value: bytes) -> None:
        self.value = value

    def read(self) -> bytes:
        return self.value


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str, str | None], tuple[bytes, dict[str, str]]] = {}

    def put_object(self, **kwargs: object) -> dict[str, str]:
        version_id = "v-1"
        key = (str(kwargs["Bucket"]), str(kwargs["Key"]), version_id)
        self.objects[key] = (bytes(kwargs["Body"]), dict(kwargs["Metadata"]))
        return {"VersionId": version_id, "ETag": '"etag"'}

    def get_object(self, **kwargs: object) -> dict[str, object]:
        version_id = str(kwargs.get("VersionId", "v-1"))
        value, metadata = self.objects[(str(kwargs["Bucket"]), str(kwargs["Key"]), version_id)]
        return {"Body": FakeBody(value), "Metadata": metadata, "VersionId": version_id}


def test_s3_artifact_refs_include_content_hash_and_version() -> None:
    client = FakeS3()
    store = S3ArtifactStore("artifacts", client=client)

    ref = store.put_bytes("runs/r1/trace.jsonl", b"hello", content_type="application/jsonl")

    assert ref == ArtifactRef(
        bucket="artifacts",
        key="runs/r1/trace.jsonl",
        sha256=hashlib.sha256(b"hello").hexdigest(),
        size_bytes=5,
        version_id="v-1",
        content_type="application/jsonl",
        etag='"etag"',
    )
    assert store.get_bytes(ref) == b"hello"
    assert ref.uri == "s3://artifacts/runs/r1/trace.jsonl"
    assert ref.version_ref.endswith("?versionId=v-1")


class FakeTable:
    def __init__(self) -> None:
        self.items: dict[tuple[str, str], dict[str, object]] = {}
        self.calls: list[dict[str, object]] = []

    def put_item(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        item = kwargs["Item"]
        key = (str(item["pk"]), str(item["sk"]))
        if key in self.items and "ConditionExpression" in kwargs:
            raise ConditionalError()
        self.items[key] = dict(item)
        return {}

    def get_item(self, **kwargs: object) -> dict[str, object]:
        item = self.items.get((str(kwargs["Key"]["pk"]), str(kwargs["Key"]["sk"])))
        return {"Item": item} if item is not None else {}

    def update_item(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        key = (str(kwargs["Key"]["pk"]), str(kwargs["Key"]["sk"]))
        item = self.items[key]
        expected = int(kwargs["ExpressionAttributeValues"][":expected"])
        if int(item["state_version"]) != expected:
            raise ConditionalError()
        item["payload"] = kwargs["ExpressionAttributeValues"][":payload"]
        item["status"] = kwargs["ExpressionAttributeValues"][":status"]
        item["phase"] = kwargs["ExpressionAttributeValues"][":phase"]
        item["updated_at"] = kwargs["ExpressionAttributeValues"][":updated_at"]
        item["state_version"] = int(item["state_version"]) + 1
        return {"Attributes": dict(item)}

    def query(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        pk = str(kwargs["ExpressionAttributeValues"][":pk"])
        return {
            "Items": [
                item for (item_pk, _), item in sorted(self.items.items()) if item_pk == pk
            ]
        }


class ConditionalError(Exception):
    response: ClassVar[dict[str, dict[str, str]]] = {
        "Error": {"Code": "ConditionalCheckFailedException"}
    }


def test_dynamodb_updates_require_expected_state_version() -> None:
    table = FakeTable()
    repository = DynamoDBRunRepository(table=table)
    repository.create_run(RunRecord(run_id="r1", data={"candidate": "a"}))

    updated = repository.update_run("r1", expected_state_version=0, patch={"candidate": "b"})
    assert updated.state_version == 1
    assert updated.data == {"candidate": "b"}
    assert "#state_version = #state_version + :one" in str(table.calls[-1]["UpdateExpression"])
    assert "#state_version = :expected" in str(table.calls[-1]["ConditionExpression"])

    with pytest.raises(ConcurrentUpdateError):
        repository.update_run("r1", expected_state_version=0, patch={"candidate": "c"})


def test_dynamodb_events_are_append_only_and_json_encoded() -> None:
    table = FakeTable()
    repository = DynamoDBRunRepository(table=table)
    repository.create_run(RunRecord(run_id="r1"))
    repository.append_event(
        RunEvent(run_id="r1", event_id="e1", event_type="phase.started", payload={"n": 1})
    )

    events = repository.list_events("r1")
    assert events[0].event_id == "e1"
    assert events[0].payload == {"n": 1}
    event_item = next(item for item in table.items.values() if item.get("event_id") == "e1")
    assert json.loads(str(event_item["payload"])) == {"n": 1}


def test_bedrock_wrapper_defers_model_construction() -> None:
    constructed: list[dict[str, object]] = []

    def factory(**kwargs: object) -> object:
        constructed.append(kwargs)
        return object()

    wrapper = BedrockStrandsModel(
        "amazon.nova-lite-v1:0", region_name="us-east-1", model_factory=factory
    )
    assert constructed == []
    wrapper.create_model()
    wrapper.create_model()
    assert constructed == [{"model_id": "amazon.nova-lite-v1:0", "region_name": "us-east-1"}]


def test_sagemaker_training_and_evaluation_requests_map_to_native_calls() -> None:
    class FakeSageMaker:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def create_training_job(self, **kwargs: object) -> dict[str, object]:
            self.calls.append(("training", kwargs))
            return {"TrainingJobArn": "arn:train"}

        def create_processing_job(self, **kwargs: object) -> dict[str, object]:
            self.calls.append(("evaluation", kwargs))
            return {"ProcessingJobArn": "arn:eval"}

        def describe_training_job(self, **kwargs: object) -> dict[str, object]:
            return {
                "TrainingJobName": kwargs["TrainingJobName"],
                "TrainingJobStatus": "Completed",
            }

        def describe_processing_job(self, **kwargs: object) -> dict[str, object]:
            return {
                "ProcessingJobName": kwargs["ProcessingJobName"],
                "ProcessingJobStatus": "Completed",
            }

    client = FakeSageMaker()
    provider = SageMakerProvider(client=client)
    training = provider.submit_training(
        TrainingJobRequest(
            job_name="train-1",
            role_arn="arn:role",
            image_uri="123.dkr.ecr/image:latest",
            input_s3_uri="s3://bucket/data",
            output_s3_uri="s3://bucket/output",
            instance_type="ml.g5.xlarge",
            hyperparameters={"epochs": 1},
        )
    )
    evaluation = provider.submit_evaluation(
        EvaluationJobRequest(
            job_name="eval-1",
            role_arn="arn:role",
            image_uri="123.dkr.ecr/eval:latest",
            input_s3_uri="s3://bucket/candidate",
            output_s3_uri="s3://bucket/eval-output",
            instance_type="ml.g5.xlarge",
        )
    )

    assert training.provider_job_id == "arn:train"
    assert evaluation.provider_job_id == "arn:eval"
    assert provider.get_training_status("train-1").status == "completed"
    assert provider.get_evaluation_status("eval-1").status == "completed"
    assert client.calls[0][1]["HyperParameters"] == {"epochs": "1"}
