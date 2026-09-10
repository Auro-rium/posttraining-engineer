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
    EvaluationMetrics,
    EvaluationWorkerError,
    InvalidModelAction,
    _decode_actions,
    build_evaluation_report,
    parse_evaluation_inputs,
    render_action_prompt,
    run_evaluation,
    verify_checkpoint_artifact,
)
from workers.trainer.train import (
    BASE_MODEL_ID,
    TrainingWorkerError,
    format_sft_example,
    load_training_dataset,
    parse_training_inputs,
    verify_parent_adapter,
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


def _qlora_config() -> dict[str, object]:
    return {
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


def test_worker_images_copy_shared_contracts_and_runtime_dependencies() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    trainer_dockerfile = (backend_root / "workers/trainer/Dockerfile").read_text()
    evaluator_dockerfile = (backend_root / "workers/evaluator/Dockerfile").read_text()
    assert "COPY app" in trainer_dockerfile
    assert "COPY app" in evaluator_dockerfile
    assert "COPY prompts" in trainer_dockerfile
    assert "COPY prompts" in evaluator_dockerfile
    assert "pydantic" in (backend_root / "workers/trainer/requirements.txt").read_text()
    assert "pydantic" in (backend_root / "workers/evaluator/requirements.txt").read_text()


def test_parent_adapter_manifest_is_verified_and_carried_into_lineage(tmp_path: Path) -> None:
    train = _dataset_fixture(tmp_path)
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "adapter_model.safetensors").write_bytes(b"parent")
    parent_manifest = write_training_manifest(
        train,
        output_dir=parent,
        run_id="run-1",
        experiment_id="exp-1",
        dataset_id="dataset-1",
        dataset_sha256=_dataset_digest(train),
        base_model_id=BASE_MODEL_ID,
        base_model_revision="a" * 40,
        qlora_config=_qlora_config(),
    )
    parent_metadata = verify_parent_adapter(parent)
    parent_payload = json.loads(parent_manifest.read_text())
    assert parent_metadata["manifest_sha256"] == parent_payload["manifest_sha256"]

    output = tmp_path / "candidate"
    output.mkdir()
    (output / "adapter_model.safetensors").write_bytes(b"candidate")
    manifest_path = write_training_manifest(
        train,
        output_dir=output,
        run_id="run-1",
        experiment_id="exp-1",
        dataset_id="dataset-1",
        dataset_sha256=_dataset_digest(train),
        base_model_id=BASE_MODEL_ID,
        base_model_revision="a" * 40,
        qlora_config=_qlora_config(),
        parent_manifest=parent_metadata,
    )
    payload = json.loads(manifest_path.read_text())
    assert payload["parent_adapter"]["manifest_sha256"] == parent_metadata["manifest_sha256"]

    with pytest.raises(TrainingWorkerError, match=r"parent|checksum"):
        verify_parent_adapter(tmp_path / "missing-parent")


def test_training_and_evaluation_share_action_prompt_serialization() -> None:
    row = {
        "task_id": "replay-1",
        "messages": (
            {
                "role": "tool",
                "name": "run_healthcheck",
                "arguments": {"service": "api"},
                "observation": {"service": "api", "healthy": False},
            },
        ),
    }
    example = format_sft_example(row)
    assert example["completion"] == (
        "<start_function_call>call:run_healthcheck{service:<escape>api<escape>}"
        "<end_function_call>"
    )
    prompt_payload = json.loads(example["prompt"])
    assert prompt_payload["task_id"] == "replay-1"
    assert prompt_payload["messages"][0]["role"] == "developer"
    assert prompt_payload["tools"][0]["type"] == "function"
    task = ServiceRecoveryEngine(seed=7, sealed=True).reset(
        split=ObjectiveSplit.HIDDEN, task_id="hidden-1"
    )
    assert json.loads(render_action_prompt(task))["objective"] == task.objective


def test_evaluator_report_contains_explicit_paired_regression_evidence(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    inputs = parse_evaluation_inputs(
        {
            "RUN_ID": "run-1",
            "EXPERIMENT_ID": "exp-1",
            "EVALUATION_MANIFEST_SHA256": "b" * 64,
            "EVALUATION_SUITE_VERSION": "agent-eval-v1",
            "OBJECTIVE_SEED": "7",
            "SM_OUTPUT_DATA_DIR": str(tmp_path / "output"),
        },
        {"candidate": candidate, "sealed": sealed},
    )
    payload = build_evaluation_report(
        inputs,
        candidate_metrics=EvaluationMetrics(2, 1, (True, False)),
        candidate_manifest={"manifest_sha256": "c" * 64, "artifact_sha256": "d" * 64},
        champion_metrics=EvaluationMetrics(2, 2, (True, True)),
        champion_manifest={"manifest_sha256": "e" * 64, "artifact_sha256": "f" * 64},
    )
    assert payload["regression_count"] == 1
    assert payload["regression_decision"] == "REGRESSED"
    assert len(payload["regression_evidence_sha256"]) == 64


def test_sealed_manifest_binds_task_bytes_and_rejects_duplicates(tmp_path: Path) -> None:
    train = _dataset_fixture(tmp_path)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "adapter_model.safetensors").write_bytes(b"candidate")
    write_training_manifest(
        train,
        output_dir=candidate,
        run_id="run-1",
        experiment_id="exp-1",
        dataset_id="dataset-1",
        dataset_sha256=_dataset_digest(train),
        base_model_id=BASE_MODEL_ID,
        base_model_revision="a" * 40,
        qlora_config=_qlora_config(),
    )
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    task_bytes = b'{"tasks":["hidden-1"]}\n'
    (sealed / "tasks.json").write_bytes(task_bytes)
    unsigned = {
        "run_id": "run-1",
        "experiment_id": "exp-1",
        "objective_seed": 7,
        "suite": "AgentGym/AgentEval",
        "suite_version": "agent-eval-v1",
        "task_bundle_sha256": hashlib.sha256(task_bytes).hexdigest(),
        "task_count": 1,
    }
    manifest_digest = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (sealed / "manifest.json").write_text(
        json.dumps({**unsigned, "manifest_sha256": manifest_digest}, sort_keys=True) + "\n"
    )
    inputs = parse_evaluation_inputs(
        {
            "RUN_ID": "run-1",
            "EXPERIMENT_ID": "exp-1",
            "EVALUATION_MANIFEST_SHA256": manifest_digest,
            "EVALUATION_SUITE_VERSION": "agent-eval-v1",
            "OBJECTIVE_SEED": "7",
            "SM_OUTPUT_DATA_DIR": str(tmp_path / "output"),
        },
        {"candidate": candidate, "sealed": sealed},
    )
    report_path = run_evaluation(inputs, policy=lambda task: ())
    report = json.loads(report_path.read_text())
    assert len(report["report_sha256"]) == 64
    (sealed / "tasks.json").write_bytes(b'{"tasks":["hidden-2"]}\n')
    with pytest.raises(EvaluationWorkerError, match="task bundle"):
        run_evaluation(inputs, policy=lambda task: ())


def test_training_admission_rejects_untrusted_source_type(tmp_path: Path) -> None:
    train = _dataset_fixture(tmp_path)
    raw = json.loads((train / "dataset.json").read_text())
    raw["rows"][0]["source_type"] = "untrusted"
    raw["manifest"]["sha256"] = hashlib.sha256(
        DatasetRow.model_validate(raw["rows"][0]).canonical_json().encode()
    ).hexdigest()
    (train / "dataset.json").write_text(json.dumps(raw))
    inputs = parse_training_inputs(
        {
            "RUN_ID": "run-1",
            "EXPERIMENT_ID": "exp-1",
            "DATASET_ID": "dataset-1",
            "DATASET_SHA256": raw["manifest"]["sha256"],
            "BASE_MODEL_ID": BASE_MODEL_ID,
            "BASE_MODEL_REVISION": "a" * 40,
            "SM_MODEL_DIR": str(tmp_path / "model"),
        },
        {"train": train},
    )
    with pytest.raises(TrainingWorkerError, match=r"source|verified"):
        load_training_dataset(inputs)


def test_artifact_id_is_bound_to_verified_content_digest(tmp_path: Path) -> None:
    train = _dataset_fixture(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    manifest_path = write_training_manifest(
        train,
        output_dir=checkpoint,
        run_id="run-1",
        experiment_id="exp-1",
        dataset_id="dataset-1",
        dataset_sha256=_dataset_digest(train),
        base_model_id=BASE_MODEL_ID,
        base_model_revision="a" * 40,
        qlora_config=_qlora_config(),
    )
    payload = json.loads(manifest_path.read_text())
    payload["artifact_id"] = "checkpoint://approved-but-wrong"
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in payload.items() if key != "manifest_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(TrainingWorkerError, match="content-bound"):
        verify_parent_adapter(checkpoint)


def test_evaluator_rejects_unlisted_duplicate_and_symlink_files(tmp_path: Path) -> None:
    train = _dataset_fixture(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    manifest_path = write_training_manifest(
        train,
        output_dir=checkpoint,
        run_id="run-1",
        experiment_id="exp-1",
        dataset_id="dataset-1",
        dataset_sha256=_dataset_digest(train),
        base_model_id=BASE_MODEL_ID,
        base_model_revision="a" * 40,
        qlora_config=_qlora_config(),
    )
    (checkpoint / "unlisted.bin").write_bytes(b"unlisted")
    with pytest.raises(EvaluationWorkerError, match="complete"):
        verify_checkpoint_artifact(checkpoint)
    (checkpoint / "unlisted.bin").unlink()
    (checkpoint / "link.bin").symlink_to(checkpoint / "adapter_model.safetensors")
    with pytest.raises(EvaluationWorkerError, match="symlink"):
        verify_checkpoint_artifact(checkpoint)
    (checkpoint / "link.bin").unlink()
    payload = json.loads(manifest_path.read_text())
    payload["artifact_files"].append(payload["artifact_files"][0])
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in payload.items() if key != "manifest_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(EvaluationWorkerError, match="duplicate"):
        verify_checkpoint_artifact(checkpoint)


def test_functiongemma_special_call_parser_rejects_unknown_tools() -> None:
    calls = _decode_actions(
        "<start_function_call>call:get_logs{service:<escape>api<escape>}"
        "<end_function_call>"
    )
    assert calls[0].tool == "get_logs"
    assert calls[0].arguments == {"service": "api"}
    with pytest.raises(InvalidModelAction, match="unknown"):
        _decode_actions(
            "<start_function_call>call:unknown_tool{}<end_function_call>"
        )


def test_evaluation_metrics_reject_impossible_aggregate() -> None:
    with pytest.raises(EvaluationWorkerError, match="exceed"):
        EvaluationMetrics(task_count=1, successful_tasks=2)


def test_evaluator_binds_checkpoint_run_and_experiment_before_scoring(tmp_path: Path) -> None:
    train = _dataset_fixture(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    write_training_manifest(
        train,
        output_dir=checkpoint,
        run_id="run-1",
        experiment_id="exp-1",
        dataset_id="dataset-1",
        dataset_sha256=_dataset_digest(train),
        base_model_id=BASE_MODEL_ID,
        base_model_revision="a" * 40,
        qlora_config=_qlora_config(),
    )
    with pytest.raises(EvaluationWorkerError, match="run identity"):
        verify_checkpoint_artifact(checkpoint, run_id="other-run", experiment_id="exp-1")
    with pytest.raises(EvaluationWorkerError, match="experiment identity"):
        verify_checkpoint_artifact(checkpoint, run_id="run-1", experiment_id="other-exp")


def test_evaluator_binds_sealed_manifest_seed_and_run_identity(tmp_path: Path) -> None:
    train = _dataset_fixture(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    write_training_manifest(
        train,
        output_dir=checkpoint,
        run_id="run-1",
        experiment_id="exp-1",
        dataset_id="dataset-1",
        dataset_sha256=_dataset_digest(train),
        base_model_id=BASE_MODEL_ID,
        base_model_revision="a" * 40,
        qlora_config=_qlora_config(),
    )
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    task_bytes = b'{"tasks":["hidden-1"]}\n'
    (sealed / "tasks.json").write_bytes(task_bytes)
    unsigned = {
        "run_id": "run-1",
        "experiment_id": "exp-1",
        "objective_seed": 7,
        "suite": "AgentGym/AgentEval",
        "suite_version": "agent-eval-v1",
        "task_bundle_sha256": hashlib.sha256(task_bytes).hexdigest(),
        "task_count": 1,
    }
    digest = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (sealed / "manifest.json").write_text(
        json.dumps({**unsigned, "manifest_sha256": digest})
    )
    inputs = parse_evaluation_inputs(
        {
            "RUN_ID": "run-1",
            "EXPERIMENT_ID": "exp-1",
            "EVALUATION_MANIFEST_SHA256": digest,
            "EVALUATION_SUITE_VERSION": "agent-eval-v1",
            "OBJECTIVE_SEED": "8",
            "SM_OUTPUT_DATA_DIR": str(tmp_path / "output"),
        },
        {"candidate": checkpoint, "sealed": sealed},
    )
    with pytest.raises(EvaluationWorkerError, match="seed"):
        run_evaluation(inputs, policy=lambda task: ())


def test_real_model_decoder_rejects_unmarked_json_protocol() -> None:
    with pytest.raises(InvalidModelAction, match="marker"):
        _decode_actions('[{"tool":"get_logs","arguments":{}}]')
