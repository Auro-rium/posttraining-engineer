from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from app.objective.engine import ServiceRecoveryEngine
from app.objective.models import DatasetManifest, DatasetRow, ObjectiveSplit
from workers.evaluator.evaluate import (
    EvaluationWorkerError,
    parse_evaluation_inputs,
    verify_checkpoint_artifact,
)
from workers.trainer.train import (
    TrainingWorkerError,
    parse_training_inputs,
    write_training_manifest,
)


def _dataset_fixture(root: Path) -> Path:
    train = root / "train"
    train.mkdir()
    row = DatasetRow(
        source_trajectory_id="traj-1",
        task_id="replay-1",
        split=ObjectiveSplit.REPLAY,
        messages=({"role": "tool", "name": "get_logs", "arguments": {}},),
        failure_label="service_recovery",
        verifier_confirmed=True,
        source_type="verified_replay",
    )
    payload = row.canonical_json()
    digest = hashlib.sha256(payload.encode()).hexdigest()
    manifest = DatasetManifest(
        dataset_id="dataset-1",
        run_id="run-1",
        experiment_id="exp-1",
        row_count=1,
        sha256=digest,
        s3_uri="s3://bucket/dataset-1.jsonl",
        created_at=datetime.now(UTC),
        source_trajectory_ids=(row.source_trajectory_id,),
        target_failure_classes=("service_recovery",),
    )
    (train / "dataset.json").write_text(
        json.dumps(
            {
                "manifest": manifest.model_dump(mode="json"),
                "rows": [row.model_dump(mode="json")],
            }
        )
        + "\n"
    )
    return train


def _dataset_digest(train: Path) -> str:
    value = json.loads((train / "dataset.json").read_text())["manifest"]["sha256"]
    return cast(str, value)


def test_training_parser_requires_train_channel_and_rejects_sealed_channels(tmp_path: Path) -> None:
    with pytest.raises(TrainingWorkerError, match="SM_CHANNEL_TRAIN"):
        parse_training_inputs(
            {"RUN_ID": "run-1", "EXPERIMENT_ID": "exp-1", "DATASET_SHA256": "a" * 64},
            {},
        )

    train = _dataset_fixture(tmp_path)
    with pytest.raises(TrainingWorkerError, match=r"sealed|hidden|evaluation"):
        parse_training_inputs(
            {
                "SM_CHANNEL_TRAIN": str(train),
                "SM_CHANNEL_HIDDEN": str(tmp_path / "hidden"),
                "RUN_ID": "run-1",
                "EXPERIMENT_ID": "exp-1",
                "DATASET_ID": "dataset-1",
                "DATASET_SHA256": "a" * 64,
                "BASE_MODEL_ID": "google/functiongemma-270m-it",
                "BASE_MODEL_REVISION": "a" * 40,
                "SM_MODEL_DIR": str(tmp_path / "model"),
            },
            {"train": train},
        )


def test_evaluator_parser_is_sealed_only_and_requires_pinned_manifest(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    env = {
        "RUN_ID": "run-1",
        "EXPERIMENT_ID": "exp-1",
        "EVALUATION_MANIFEST_SHA256": "b" * 64,
        "EVALUATION_SUITE_VERSION": "agent-eval-v1",
        "OBJECTIVE_SEED": "7",
        "SM_OUTPUT_DATA_DIR": str(tmp_path / "output"),
    }
    parsed = parse_evaluation_inputs(env, {"candidate": candidate, "sealed": sealed})
    assert parsed.evaluation_manifest_sha256 == "b" * 64

    with pytest.raises(EvaluationWorkerError, match=r"sealed|hidden"):
        parse_evaluation_inputs(env, {"candidate": candidate, "train": tmp_path / "train"})


def test_training_manifest_is_deterministic_and_refuses_empty_output(tmp_path: Path) -> None:
    train = _dataset_fixture(tmp_path)
    dataset_sha256 = _dataset_digest(train)
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "adapter_model.safetensors").write_bytes(b"real-adapter")
    manifest_path = write_training_manifest(
        train,
        output_dir=tmp_path / "model",
        run_id="run-1",
        experiment_id="exp-1",
        dataset_id="dataset-1",
        dataset_sha256=dataset_sha256,
        base_model_id="google/functiongemma-270m-it",
        base_model_revision="a" * 40,
        qlora_config={
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "learning_rate": 0.0002,
            "epochs": 1,
            "sequence_length": 512,
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
    )
    payload = json.loads(manifest_path.read_text())
    assert len(payload["artifact_sha256"]) == 64
    assert payload["artifact_files"]
    assert payload["manifest_sha256"] == hashlib.sha256(
        json.dumps(
            {k: v for k, v in payload.items() if k != "manifest_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    with pytest.raises(TrainingWorkerError, match=r"artifact|output"):
        write_training_manifest(
            train,
            output_dir=tmp_path / "empty-model",
            run_id="run-1",
            experiment_id="exp-1",
            dataset_id="dataset-1",
            dataset_sha256=dataset_sha256,
            base_model_id="google/functiongemma-270m-it",
            base_model_revision="a" * 40,
            qlora_config={},
        )


def test_training_manifest_binds_to_the_source_dataset_digest(tmp_path: Path) -> None:
    train = _dataset_fixture(tmp_path)
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "adapter_model.safetensors").write_bytes(b"real-adapter")
    with pytest.raises(TrainingWorkerError, match="dataset"):
        write_training_manifest(
            train,
            output_dir=tmp_path / "model",
            run_id="run-1",
            experiment_id="exp-1",
            dataset_id="dataset-1",
            dataset_sha256="a" * 64,
            base_model_id="google/functiongemma-270m-it",
            base_model_revision="a" * 40,
            qlora_config={
                "rank": 8,
                "alpha": 16,
                "dropout": 0.05,
                "learning_rate": 0.0002,
                "epochs": 1,
                "sequence_length": 512,
                "batch_size": 1,
                "gradient_accumulation_steps": 4,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
        )


def test_training_manifest_rejects_qlora_config_outside_fixed_search_space(tmp_path: Path) -> None:
    train = _dataset_fixture(tmp_path)
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "adapter_model.safetensors").write_bytes(b"real-adapter")
    with pytest.raises(TrainingWorkerError, match="QLORA"):
        write_training_manifest(
            train,
            output_dir=tmp_path / "model",
            run_id="run-1",
            experiment_id="exp-1",
            dataset_id="dataset-1",
            dataset_sha256=_dataset_digest(train),
            base_model_id="google/functiongemma-270m-it",
            base_model_revision="a" * 40,
            qlora_config={"rank": 7},
        )


def test_evaluator_refuses_to_report_absent_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(EvaluationWorkerError, match=r"checkpoint|artifact"):
        verify_checkpoint_artifact(tmp_path / "missing")


def test_evaluator_rejects_training_channel_from_environment(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    with pytest.raises(EvaluationWorkerError, match=r"train|sealed"):
        parse_evaluation_inputs(
            {
                "SM_CHANNEL_TRAIN": str(tmp_path / "train"),
                "RUN_ID": "run-1",
                "EXPERIMENT_ID": "exp-1",
                "EVALUATION_MANIFEST_SHA256": "b" * 64,
                "EVALUATION_SUITE_VERSION": "agent-eval-v1",
                "OBJECTIVE_SEED": "7",
                "SM_OUTPUT_DATA_DIR": str(tmp_path / "output"),
            },
            {"candidate": candidate, "sealed": sealed},
        )


def test_sealed_engine_does_not_serialize_hidden_task_details() -> None:
    result = ServiceRecoveryEngine(seed=7, sealed=True).run_episode(
        "hidden-1", [], split=ObjectiveSplit.HIDDEN
    )
    assert not hasattr(result, "model_dump")
