from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import ClassVar, cast

import pytest

from app.posttraining.models import Artifact, ArtifactKind, Evidence, EvidenceKind, EvidenceLabel
from app.posttraining.run_history import RunDecision, RunHistoryRecord, RunStatus
from app.providers.artifacts import ArtifactRef, S3ArtifactStore
from app.providers.bedrock import BedrockStrandsModel
from app.providers.repository import (
    ConcurrentUpdateError,
    DynamoDBRunRepository,
    RunAlreadyExistsError,
    RunEvent,
    RunLimitExceeded,
    RunRecord,
    RunSequenceError,
)
from app.providers.sagemaker import (
    EvaluationJobRequest,
    ProviderResponseError,
    SageMakerProvider,
    TrainingJobRequest,
)
from workers.evaluator.evaluate import parse_evaluation_inputs
from workers.trainer.train import parse_training_inputs


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

    def head_object(self, **kwargs: object) -> dict[str, object]:
        version_id = str(kwargs["VersionId"])
        data, metadata = self.objects[
            (str(kwargs["Bucket"]), str(kwargs["Key"]), version_id)
        ]
        return {
            "VersionId": version_id,
            "ContentLength": len(data),
            "Metadata": metadata,
            "ContentType": "application/jsonl",
            "ETag": '"etag"',
        }

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


class FakeTransactionClient:
    def __init__(self, table: FakeTransactionalTable) -> None:
        self.table = table
        self.calls: list[dict[str, object]] = []

    def transact_write_items(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        operations = kwargs["TransactItems"]
        assert isinstance(operations, list)
        update = operations[0]["Update"]
        put = operations[1]["Put"]
        assert isinstance(update, dict)
        assert isinstance(put, dict)
        counter_key = (
            str(update["Key"]["pk"]["S"]),
            str(update["Key"]["sk"]["S"]),
        )
        history_item = {
            key: self._decode(value)
            for key, value in put["Item"].items()
        }
        assert isinstance(history_item, dict)
        history_key = (str(history_item["pk"]), str(history_item["sk"]))
        counter = self.table.items.get(counter_key)
        count = int(counter.get("run_count", 0)) if counter else 0
        last_number = int(counter.get("last_run_number", 0)) if counter else 0
        maximum = int(update["ExpressionAttributeValues"][":max_runs"]["N"])
        if count >= maximum:
            raise TransactionLimitError()
        expected = int(update["ExpressionAttributeValues"][":expected"]["N"])
        next_number = int(update["ExpressionAttributeValues"][":next"]["N"])
        if last_number != expected:
            raise TransactionSequenceError()
        if history_key in self.table.items:
            raise TransactionDuplicateError()
        self.table.items[counter_key] = {
            "pk": counter_key[0],
            "sk": counter_key[1],
            "entity": "run_history_counter",
            "run_count": count + 1,
            "last_run_number": next_number,
        }
        self.table.items[history_key] = dict(history_item)
        return {}

    @staticmethod
    def _decode(value: object) -> object:
        assert isinstance(value, dict)
        if "S" in value:
            return value["S"]
        if "N" in value:
            return int(value["N"])
        return value


class FakeTransactionalTable(FakeTable):
    def __init__(self) -> None:
        super().__init__()
        self.meta = type("Meta", (), {"client": FakeTransactionClient(self)})()


class TransactionLimitError(Exception):
    response: ClassVar[dict[str, object]] = {
        "Error": {"Code": "TransactionCanceledException"},
        "CancellationReasons": [{"Code": "ConditionalCheckFailed"}, {}],
    }


class TransactionSequenceError(Exception):
    response: ClassVar[dict[str, object]] = {
        "Error": {"Code": "TransactionCanceledException"},
        "CancellationReasons": [{"Code": "ConditionalCheckFailed"}, {}],
    }


class TransactionDuplicateError(Exception):
    response: ClassVar[dict[str, object]] = {
        "Error": {"Code": "TransactionCanceledException"},
        "CancellationReasons": [{"Code": "None"}, {"Code": "ConditionalCheckFailed"}],
    }


def history_record(number: int, run_id: str | None = None) -> RunHistoryRecord:
    candidate_id = f"candidate-{number:03d}"
    return RunHistoryRecord(
        run_id=run_id or f"run-{number:03d}",
        run_number=number,
        status=RunStatus.COMPLETED,
        decision=RunDecision.PROMOTE,
        manifest_sha256="a" * 64,
        champion_run_id="champion" if number > 1 else None,
        parent_run_id=f"run-{number - 1:03d}" if number > 1 else None,
        champion_artifact_id="champion-checkpoint",
            candidate_artifact_id=candidate_id,
            benchmark_id="agentgym-held-out",
            suite="AgentGym",
            suite_version="v1",
            seed=7,
            model_id="google/functiongemma-270m-it",
        baseline_metrics={"aggregate": 0.40, "per_environment": {"WebShop": 0.40}},
        candidate_metrics={"aggregate": 0.50, "per_environment": {"WebShop": 0.50}},
        artifact_refs=(
                Artifact(
                    artifact_id=candidate_id,
                kind=ArtifactKind.CHECKPOINT,
                uri=f"s3://artifacts/candidate-{number:03d}",
                sha256="b" * 64,
                ),
            ),
            evidence=(
                Evidence(
                    evidence_id=f"evaluation-{number:03d}",
                    kind=EvidenceKind.EVALUATION,
                    label=EvidenceLabel.LIVE,
                    artifact_ids=(candidate_id,),
                    metrics={"aggregate": 0.50, "WebShop": 0.50},
                    verified=True,
                    benchmark_id="agentgym-held-out",
                    suite="AgentGym",
                    suite_version="v1",
                    manifest_sha256="a" * 64,
                    seed=7,
                    model_id="google/functiongemma-270m-it",
                ),
            ),
        )


def test_dynamodb_history_round_trips_links_metrics_manifest_and_artifacts() -> None:
    table = FakeTransactionalTable()
    repository = DynamoDBRunRepository(table=table)
    expected = history_record(1)

    assert repository.reserve_run(expected) == expected
    assert repository.get_history_run(expected.run_id) == expected
    assert repository.list_history_runs() == [expected]
    item = table.items[("HISTORY", "RUN#run-001")]
    payload = json.loads(str(item["payload"]))
    assert payload["champion_run_id"] is None
    assert payload["manifest_sha256"] == "a" * 64
    assert payload["baseline_metrics"]["per_environment"] == {"WebShop": 0.4}
    assert payload["artifact_refs"][0]["sha256"] == "b" * 64


def test_dynamodb_history_reservation_enforces_cap_and_duplicate_ids_atomically() -> None:
    table = FakeTransactionalTable()
    repository = DynamoDBRunRepository(table=table)
    for number in range(1, 6):
        repository.reserve_run(history_record(number))

    with pytest.raises(RunLimitExceeded, match="maximum of 5"):
        repository.reserve_run(history_record(5, "run-overflow"))

    duplicate_table = FakeTransactionalTable()
    duplicate_repository = DynamoDBRunRepository(table=duplicate_table)
    duplicate_repository.reserve_run(history_record(1))
    with pytest.raises(RunAlreadyExistsError, match="run-001"):
        duplicate_repository.reserve_run(history_record(1))

    transaction = table.meta.client.calls[0]
    assert len(transaction["TransactItems"]) == 2
    assert "#run_count < :max_runs" in str(
        transaction["TransactItems"][0]["Update"]["ConditionExpression"]
    )
    assert "#last_run_number" in str(transaction["TransactItems"][0]["Update"])


def test_dynamodb_history_requires_the_next_sequence_number_atomically() -> None:
    table = FakeTransactionalTable()
    repository = DynamoDBRunRepository(table=table)

    with pytest.raises(RunSequenceError, match="expected run 1"):
        repository.reserve_run(history_record(2, "run-gap"))

    repository.reserve_run(history_record(1))
    with pytest.raises(RunSequenceError, match="expected run 2"):
        repository.reserve_run(history_record(3, "run-gap-2"))

    counter = table.items[("HISTORY", "COUNTER")]
    assert counter["run_count"] == 1
    assert counter["last_run_number"] == 1


def test_dynamodb_updates_require_expected_state_version() -> None:
    table = FakeTable()
    repository = DynamoDBRunRepository(table=table)
    created = repository.create_run(RunRecord(run_id="r1", data={"candidate": "a"}))
    assert repository.get_run("r1") == created
    assert repository.get_history_run("r1") is None

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


def test_bedrock_sigv4_uses_explicit_session_even_with_bearer_env(monkeypatch) -> None:
    """A stale bearer token must not be selected by the live model path."""

    import boto3

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "intentionally-invalid")
    explicit_session = object()
    monkeypatch.setattr(boto3, "Session", lambda *, region_name: explicit_session)
    constructed: list[dict[str, object]] = []

    def factory(**kwargs: object) -> object:
        constructed.append(kwargs)
        return object()

    wrapper = BedrockStrandsModel(
        "nvidia.nemotron-super-3-120b",
        region_name="us-east-1",
    )
    # Keep the test offline while exercising the production session-creation
    # branch (which is skipped when a custom factory is injected).
    monkeypatch.setattr(wrapper, "_factory", lambda: factory)
    wrapper.create_model()

    assert len(constructed) == 1
    assert constructed[0]["model_id"] == "nvidia.nemotron-super-3-120b"
    assert constructed[0]["boto_session"] is explicit_session
    client_config = constructed[0]["boto_client_config"]
    assert getattr(client_config, "signature_version", None) == "v4"
    # The wrapper neither reads nor mutates the bearer token; auth is owned by
    # the explicit boto session (and therefore the IAM/SigV4 credential chain).
    import os

    assert os.environ["AWS_BEARER_TOKEN_BEDROCK"] == "intentionally-invalid"


def test_bedrock_sigv4_mode_rejects_unknown_auth_mode() -> None:
    import pytest

    with pytest.raises(ValueError, match="SigV4"):
        BedrockStrandsModel("nvidia.nemotron-super-3-120b", auth_mode="bearer")  # type: ignore[arg-type]


def test_sigv4_boto_session_does_not_use_bearer_environment_token(monkeypatch) -> None:
    """The SDK client signer remains SigV4 when a bearer token is present."""

    import boto3

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "intentionally-invalid")
    session = boto3.Session(
        aws_access_key_id="test-access-key",
        aws_secret_access_key="test-secret-key",
        region_name="us-east-1",
    )
    client = session.client("bedrock-runtime")

    assert client._request_signer._credentials.method == "explicit"
    assert client._request_signer._credentials.access_key == "test-access-key"


def test_sagemaker_training_and_evaluation_requests_map_to_native_calls() -> None:
    class ResourceNotFoundError(Exception):
        def __init__(self, job_name: object) -> None:
            super().__init__(f"job {job_name!r} was not found")
            self.response: dict[str, object] = {
                "Error": {
                    "Code": "ResourceNotFoundException",
                    "Message": f"job {job_name!r} was not found",
                }
            }

    class FakeSageMaker:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []
            self.training: dict[str, object] | None = None
            self.processing: dict[str, object] | None = None
            self.tags: dict[str, list[dict[str, str]]] = {}

        def create_training_job(self, **kwargs: object) -> dict[str, object]:
            self.calls.append(("training", kwargs))
            arn = "arn:aws:sagemaker:us-east-1:123456789012:training-job/train-1"
            self.training = {
                "TrainingJobName": kwargs["TrainingJobName"],
                "TrainingJobArn": arn,
                "TrainingJobStatus": "Completed",
            }
            self.tags[arn] = cast(list[dict[str, str]], kwargs.get("Tags", []))
            return {"TrainingJobArn": arn}

        def create_processing_job(self, **kwargs: object) -> dict[str, object]:
            self.calls.append(("evaluation", kwargs))
            arn = "arn:aws:sagemaker:us-east-1:123456789012:processing-job/eval-1"
            self.processing = {
                "ProcessingJobName": kwargs["ProcessingJobName"],
                "ProcessingJobArn": arn,
                "ProcessingJobStatus": "Completed",
            }
            self.tags[arn] = cast(list[dict[str, str]], kwargs.get("Tags", []))
            return {"ProcessingJobArn": arn}

        def describe_training_job(self, **kwargs: object) -> dict[str, object]:
            if self.training is None:
                raise ResourceNotFoundError(kwargs["TrainingJobName"])
            return dict(self.training)

        def describe_processing_job(self, **kwargs: object) -> dict[str, object]:
            if self.processing is None:
                raise ResourceNotFoundError(kwargs["ProcessingJobName"])
            return dict(self.processing)

        def list_tags(self, **kwargs: object) -> dict[str, object]:
            arn = str(kwargs["ResourceArn"])
            return {"Tags": self.tags[arn]}

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

    assert (
        training.provider_job_id
        == "arn:aws:sagemaker:us-east-1:123456789012:training-job/train-1"
    )
    assert (
        evaluation.provider_job_id
        == "arn:aws:sagemaker:us-east-1:123456789012:processing-job/eval-1"
    )
    assert provider.get_training_status("train-1").status == "completed"
    assert provider.get_evaluation_status("eval-1").status == "completed"
    assert client.calls[0][1]["HyperParameters"] == {"epochs": "1"}
    training_tags = cast(list[dict[str, str]], client.calls[0][1]["Tags"])
    assert training_tags[-1]["Key"] == "request-fingerprint"


def test_sagemaker_payloads_feed_strict_trainer_and_evaluator_parsers(
    tmp_path: Path,
) -> None:
    """Native SageMaker channel/env payloads must boot both audited workers."""

    class _NotFound(Exception):
        def __init__(self, name: object) -> None:
            super().__init__(str(name))
            self.response = {
                "Error": {"Code": "ResourceNotFoundException", "Message": "not found"}
            }

    class _Client:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def describe_training_job(self, **kwargs: object) -> dict[str, object]:
            raise _NotFound(kwargs["TrainingJobName"])

        def describe_processing_job(self, **kwargs: object) -> dict[str, object]:
            raise _NotFound(kwargs["ProcessingJobName"])

        def create_training_job(self, **kwargs: object) -> dict[str, object]:
            self.calls.append(("training", kwargs))
            return {"TrainingJobArn": "arn:aws:sagemaker:us-east-1:123:training-job/train"}

        def create_processing_job(self, **kwargs: object) -> dict[str, object]:
            self.calls.append(("evaluation", kwargs))
            return {
                "ProcessingJobArn": "arn:aws:sagemaker:us-east-1:123:processing-job/eval"
            }

    client = _Client()
    provider = SageMakerProvider(client=client)
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    model_dir = tmp_path / "model"
    training_environment = {
        "RUN_ID": "run-1",
        "EXPERIMENT_ID": "exp-1",
        "DATASET_ID": "dataset-1",
        "DATASET_SHA256": "a" * 64,
        "BASE_MODEL_ID": "google/functiongemma-270m-it",
        "BASE_MODEL_REVISION": "b" * 40,
        "SM_MODEL_DIR": str(model_dir),
    }
    provider.submit_training(
        TrainingJobRequest(
            job_name="train-contract",
            role_arn="arn:role",
            image_uri="123.dkr.ecr/train@sha256:" + "c" * 64,
            input_s3_uri="s3://bucket/dataset/run-1?versionId=dataset-v1",
            output_s3_uri="s3://bucket/output/run-1",
            instance_type="ml.g5.xlarge",
            environment=training_environment,
        )
    )
    training_payload = client.calls[0][1]
    assert training_payload["InputDataConfig"][0]["ChannelName"] == "train"
    parser_environment = dict(training_payload["Environment"])
    parser_environment["SM_CHANNEL_TRAIN"] = str(train_dir)
    parsed_training = parse_training_inputs(parser_environment)
    assert parsed_training.dataset_id == "dataset-1"
    assert parsed_training.base_model_revision == "b" * 40

    candidate_uri = "s3://bucket/checkpoints/candidate.tar.gz?versionId=candidate-v1"
    champion_uri = "s3://bucket/checkpoints/champion.tar.gz?versionId=champion-v1"
    sealed_uri = "s3://bucket/evaluation/sealed?versionId=sealed-v1"
    sealed_dir = tmp_path / "sealed"
    sealed_dir.mkdir()
    evaluation_environment = {
        "RUN_ID": "run-1",
        "EXPERIMENT_ID": "exp-1",
        "EVALUATION_MANIFEST_SHA256": "d" * 64,
        "EVALUATION_SUITE_VERSION": "agent-eval-v1",
        "OBJECTIVE_SEED": "7",
        "SM_OUTPUT_DATA_DIR": str(tmp_path / "evaluation-output"),
    }
    provider.submit_evaluation(
        EvaluationJobRequest(
            job_name="eval-contract",
            role_arn="arn:role",
            image_uri="123.dkr.ecr/eval@sha256:" + "e" * 64,
            input_s3_uri=sealed_uri,
            sealed_s3_uri=sealed_uri,
            candidate_s3_uri=candidate_uri,
            champion_s3_uri=champion_uri,
            output_s3_uri="s3://bucket/evaluation/output",
            instance_type="ml.g5.xlarge",
            environment=evaluation_environment,
        )
    )
    evaluation_payload = client.calls[1][1]
    input_names = {
        item["InputName"]: item for item in evaluation_payload["ProcessingInputs"]
    }
    assert set(input_names) == {"candidate", "champion", "sealed"}
    assert input_names["candidate"]["S3Input"]["S3Uri"] == candidate_uri
    assert input_names["champion"]["S3Input"]["S3Uri"] == champion_uri
    assert input_names["sealed"]["S3Input"]["S3Uri"] == sealed_uri
    entrypoint = evaluation_payload["AppSpecification"]["ContainerEntrypoint"]
    assert entrypoint[:2] == ["python", "-c"]
    assert "extractall" in entrypoint[2]
    assert "evaluate.py" in entrypoint[2]
    evaluator_environment = dict(evaluation_payload["Environment"])
    for name, item in input_names.items():
        native_path = Path(item["S3Input"]["LocalPath"])
        assert native_path == Path(f"/opt/ml/processing/input/{name}")
        parser_path = tmp_path / name
        parser_path.mkdir(parents=True, exist_ok=True)
        evaluator_environment[f"SM_CHANNEL_{name.upper()}"] = str(parser_path)
    parsed_evaluation = parse_evaluation_inputs(evaluator_environment)
    assert parsed_evaluation.candidate_dir == (tmp_path / "candidate").resolve()
    assert parsed_evaluation.champion_dir == (tmp_path / "champion").resolve()
    assert parsed_evaluation.sealed_dir == (tmp_path / "sealed").resolve()


def test_sagemaker_terminal_result_preserves_actual_cost_and_billable_time() -> None:
    class _Client:
        def describe_training_job(self, **kwargs: object) -> dict[str, object]:
            return {
                "TrainingJobName": kwargs["TrainingJobName"],
                "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123:training-job/train",
                "TrainingJobStatus": "Completed",
                "ActualCostUsd": 1.25,
                "BillableTimeInSeconds": 600,
            }

    result = SageMakerProvider(client=_Client()).get_training_status("train-cost")

    assert result.raw_response["actual_cost_usd"] == pytest.approx(1.25)
    assert result.raw_response["BillableTimeInSeconds"] == 600


def test_sagemaker_rejects_empty_s3_artifact_uri() -> None:
    class _Client:
        def describe_training_job(self, **kwargs: object) -> dict[str, object]:
            return {
                "TrainingJobName": kwargs["TrainingJobName"],
                "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123:training-job/train",
                "TrainingJobStatus": "Completed",
                "ModelArtifacts": {"S3ModelArtifacts": "s3://"},
            }

    with pytest.raises(ProviderResponseError, match="invalid artifact URI"):
        SageMakerProvider(client=_Client()).get_training_status("train-empty-artifact")


def test_sagemaker_rejects_empty_processing_output_uri() -> None:
    class _Client:
        def describe_processing_job(self, **kwargs: object) -> dict[str, object]:
            return {
                "ProcessingJobName": kwargs["ProcessingJobName"],
                "ProcessingJobArn": "arn:aws:sagemaker:us-east-1:123:processing-job/eval",
                "ProcessingJobStatus": "Completed",
                "ProcessingOutputConfig": {
                    "Outputs": [{"S3Output": {"S3Uri": "s3://"}}]
                },
            }

    with pytest.raises(ProviderResponseError, match="invalid artifact URI"):
        SageMakerProvider(client=_Client()).get_evaluation_status("eval-empty-artifact")
