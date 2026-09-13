from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.live_execution import LiveExecutionConfig, LiveRequestFactory
from app.providers.sagemaker import SageMakerProvider, TrainingJobRequest
from scripts.stage_functiongemma_checkpoint import build_deterministic_bundle
from workers.trainer.train import (
    BASE_MODEL_ID,
    TrainingWorkerError,
    _load_base_model_locally,
    parse_training_inputs,
)


def _base_model_bundle(tmp_path: Path, revision: str = "a" * 40) -> tuple[Path, str]:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"architectures": ["Gemma3ForCausalLM"], "model_type": "gemma3_text"})
    )
    (checkpoint / "tokenizer.json").write_text('{"version":1}')
    (checkpoint / "tokenizer_config.json").write_text("{}")
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode()
    (checkpoint / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"\x00\x00\x00\x00"
    )
    bundle = build_deterministic_bundle(checkpoint, revision=revision)
    channel = tmp_path / "base-model-channel"
    channel.mkdir()
    (channel / f"{bundle.sha256}.tar.gz").write_bytes(bundle.data)
    return channel, bundle.sha256


def _training_environment(tmp_path: Path, *, bundle_sha256: str) -> dict[str, str]:
    train = tmp_path / "train"
    train.mkdir(exist_ok=True)
    return {
        "SM_CHANNEL_TRAIN": str(train),
        "SM_MODEL_DIR": str(tmp_path / "model"),
        "RUN_ID": "run-1",
        "EXPERIMENT_ID": "run-1-1",
        "DATASET_ID": "dataset-1",
        "DATASET_SHA256": "b" * 64,
        "APPROVED_DATASET_ARTIFACT_ID": "dataset://dataset-1",
        "BASE_MODEL_ID": BASE_MODEL_ID,
        "BASE_MODEL_REVISION": "a" * 40,
        "BASE_MODEL_BUNDLE_SHA256": bundle_sha256,
    }


def test_trainer_requires_materialized_base_model_channel(tmp_path: Path) -> None:
    with pytest.raises(TrainingWorkerError, match="SM_CHANNEL_BASE_MODEL"):
        parse_training_inputs(_training_environment(tmp_path, bundle_sha256="c" * 64))


def test_trainer_extracts_only_the_pinned_functiongemma_bundle(tmp_path: Path) -> None:
    channel, bundle_sha256 = _base_model_bundle(tmp_path)
    environment = _training_environment(tmp_path, bundle_sha256=bundle_sha256)
    environment["SM_CHANNEL_BASE_MODEL"] = str(channel)

    inputs = parse_training_inputs(environment)

    assert inputs.base_model_dir.is_dir()
    assert (inputs.base_model_dir / "model.safetensors").is_file()
    assert inputs.base_model_dir != inputs.model_dir
    assert inputs.base_model_revision == "a" * 40


def test_trainer_rejects_base_model_channel_digest_mismatch(tmp_path: Path) -> None:
    channel, _ = _base_model_bundle(tmp_path)
    environment = _training_environment(tmp_path, bundle_sha256="c" * 64)
    environment["SM_CHANNEL_BASE_MODEL"] = str(channel)

    with pytest.raises(TrainingWorkerError, match="content-addressed"):
        parse_training_inputs(environment)


def test_trainer_rejects_unexpected_files_in_base_model_channel(tmp_path: Path) -> None:
    channel, bundle_sha256 = _base_model_bundle(tmp_path)
    (channel / "unexpected.json").write_text("{}")
    environment = _training_environment(tmp_path, bundle_sha256=bundle_sha256)
    environment["SM_CHANNEL_BASE_MODEL"] = str(channel)

    with pytest.raises(TrainingWorkerError, match="content-addressed"):
        parse_training_inputs(environment)


def test_trainer_model_loaders_use_only_the_materialized_checkpoint(tmp_path: Path) -> None:
    channel, bundle_sha256 = _base_model_bundle(tmp_path)
    environment = _training_environment(tmp_path, bundle_sha256=bundle_sha256)
    environment["SM_CHANNEL_BASE_MODEL"] = str(channel)
    inputs = parse_training_inputs(environment)
    calls: dict[str, tuple[str, dict[str, Any]]] = {}

    class ProcessorLoader:
        @classmethod
        def from_pretrained(cls, name: str, **kwargs: Any) -> object:
            calls["processor"] = (name, kwargs)
            return object()

    class ModelLoader:
        @classmethod
        def from_pretrained(cls, name: str, **kwargs: Any) -> object:
            calls["model"] = (name, kwargs)
            return object()

    _load_base_model_locally(
        inputs,
        processor_loader=ProcessorLoader,
        model_loader=ModelLoader,
        quantization_config=object(),
        device_map="auto",
    )

    expected_path = str(inputs.base_model_dir)
    assert calls["processor"][0] == expected_path
    assert calls["processor"][1]["local_files_only"] is True
    assert calls["model"][0] == expected_path
    assert calls["model"][1]["local_files_only"] is True
    assert "revision" not in calls["model"][1]


def test_request_factory_uses_versioned_base_checkpoint_as_training_channel() -> None:
    config = LiveExecutionConfig.model_validate(
        {
            "aws_region": "us-east-1",
            "artifact_bucket": "demo-bucket",
            "dynamodb_table": "demo-history",
            "training_role_arn": "arn:aws:iam::123456789012:role/train",
            "training_image": (
                "123456789012.dkr.ecr.us-east-1.amazonaws.com/train@sha256:" + "c" * 64
            ),
            "evaluation_image": (
                "123456789012.dkr.ecr.us-east-1.amazonaws.com/eval@sha256:" + "d" * 64
            ),
            "objective_worker_url": "https://worker.example.com",
            "hf_repo_id": BASE_MODEL_ID,
            "training_input_s3_uri": "s3://demo-bucket/train",
            "evaluation_input_s3_uri": "s3://demo-bucket/eval",
            "hf_revision": "a" * 40,
        }
    )
    digest = "e" * 64
    state = SimpleNamespace(
        run_id="run-1",
        model_id=BASE_MODEL_ID,
        checkpoint_revision="a" * 40,
        base_checkpoint_uri=f"s3://demo-bucket/checkpoints/{digest}.tar.gz?versionId=base-v1",
        base_checkpoint_sha256=digest,
        approval_scope={
            "instance_type": config.instance_type,
            "instance_count": config.instance_count,
            "volume_size_gb": config.volume_size_gb,
            "max_runtime_seconds": config.max_runtime_seconds,
        },
    )
    dataset = SimpleNamespace(
        dataset_id="dataset-1",
        artifact_id="dataset://dataset-1",
        sha256="f" * 64,
        uri=f"s3://demo-bucket/datasets/run-1/1/{'f' * 64}/dataset.jsonl?versionId=dataset-v1",
    )
    qlora = {
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
        "learning_rate": 0.0002,
        "epochs": 1,
        "sequence_length": 512,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }

    request = LiveRequestFactory(config).training(
        state,
        experiment_number=1,
        dataset=dataset,
        config=SimpleNamespace(model_dump=lambda **_: qlora),
    )

    assert request.base_model_s3_uri == f"s3://demo-bucket/checkpoints/{digest}.tar.gz"
    assert request.environment["BASE_MODEL_BUNDLE_SHA256"] == digest


def test_sagemaker_training_request_has_base_model_and_parent_adapter_channels() -> None:
    class FakeClient:
        calls: list[dict[str, Any]]

        def __init__(self) -> None:
            self.calls = []

        def describe_training_job(self, **_: object) -> dict[str, object]:
            raise _NotFound()

        def list_tags(self, **_: object) -> dict[str, object]:
            return {"Tags": []}

        def create_training_job(self, **kwargs: Any) -> dict[str, object]:
            self.calls.append(kwargs)
            return {"TrainingJobArn": "arn:aws:sagemaker:us-east-1:123:training-job/train"}

    class _NotFound(Exception):
        response: dict[str, Any]

        def __init__(self) -> None:
            self.response = {
                "Error": {"Code": "ResourceNotFoundException", "Message": "not found"}
            }

    dataset_sha = "a" * 64
    base_sha = "b" * 64
    parent_sha = "c" * 64
    client = FakeClient()
    request = TrainingJobRequest(
        job_name="train-base-model",
        role_arn="arn:aws:iam::123456789012:role/train",
        image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/train@sha256:" + "d" * 64,
        input_s3_uri=f"s3://demo-bucket/train/{dataset_sha}",
        base_model_s3_uri=f"s3://demo-bucket/base/{base_sha}.tar.gz",
        parent_adapter_s3_uri=f"s3://demo-bucket/parent/{parent_sha}.tar.gz",
        output_s3_uri="s3://demo-bucket/output",
        instance_type="ml.g5.xlarge",
        environment={
            "RUN_ID": "run-1",
            "EXPERIMENT_ID": "run-1-2",
            "DATASET_ID": "dataset-1",
            "DATASET_SHA256": dataset_sha,
            "APPROVED_DATASET_ARTIFACT_ID": "dataset://dataset-1",
            "BASE_MODEL_ID": BASE_MODEL_ID,
            "BASE_MODEL_REVISION": "e" * 40,
            "BASE_MODEL_BUNDLE_SHA256": base_sha,
            "QLORA_CONFIG": "{}",
            "APPROVED_PARENT_ARTIFACT_ID": "checkpoint://" + "f" * 64,
            "APPROVED_PARENT_MANIFEST_SHA256": "1" * 64,
            "APPROVED_PARENT_ARTIFACT_SHA256": "2" * 64,
            "APPROVED_PARENT_ARCHIVE_SHA256": parent_sha,
        },
    )

    SageMakerProvider(client=client).submit_training(request)

    channels = {
        item["ChannelName"]: item["DataSource"]["S3DataSource"]["S3Uri"]
        for item in client.calls[0]["InputDataConfig"]
    }
    assert channels == {
        "train": f"s3://demo-bucket/train/{dataset_sha}",
        "base_model": f"s3://demo-bucket/base/{base_sha}.tar.gz",
        "parent_adapter": f"s3://demo-bucket/parent/{parent_sha}.tar.gz",
    }
