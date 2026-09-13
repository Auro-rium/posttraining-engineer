from __future__ import annotations

import hashlib
import io
import json
import struct
import sys
import tarfile
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from app.objective.engine import ServiceRecoveryEngine
from app.objective.models import DatasetManifest, DatasetRow, ObjectiveSplit
from scripts.stage_functiongemma_checkpoint import build_deterministic_bundle
from workers.evaluator.evaluate import (
    EvaluationMetrics,
    EvaluationWorkerError,
    InvalidModelAction,
    _decode_actions,
    _model_policy,
    _sealed_manifest,
    build_evaluation_report,
    parse_evaluation_inputs,
    render_action_prompt,
    run_evaluation,
    verify_checkpoint_artifact,
)
from workers.trainer.train import (
    BASE_MODEL_ID,
    TrainingInputs,
    TrainingWorkerError,
    format_sft_example,
    load_training_dataset,
    parse_training_inputs,
    run_training,
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
        verifier_success=True,
        repaired_from_trajectory_id=None,
        source_type="successful_replay",
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


def _base_model_channel(root: Path) -> tuple[Path, str]:
    snapshot = root / "base-model-snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        json.dumps({"architectures": ["Gemma3ForCausalLM"], "model_type": "gemma3_text"})
    )
    (snapshot / "tokenizer.json").write_text('{"version":1}')
    (snapshot / "tokenizer_config.json").write_text("{}")
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode()
    (snapshot / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"\x00" * 4
    )
    bundle = build_deterministic_bundle(snapshot, revision="a" * 40)
    channel = root / "base-model-channel"
    channel.mkdir()
    (channel / f"{bundle.sha256}.tar.gz").write_bytes(bundle.data)
    return channel, bundle.sha256


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


def _write_adapter_artifacts(output_dir: Path, weights: bytes = b"adapter") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": BASE_MODEL_ID,
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
            sort_keys=True,
        )
        + "\n"
    )
    (output_dir / "adapter_model.safetensors").write_bytes(weights)
    (output_dir / "training_metrics.json").write_text('{"train_loss":0.5}\n')


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
    _write_adapter_artifacts(tmp_path / "model", b"real-adapter")
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
    _write_adapter_artifacts(tmp_path / "model", b"real-adapter")
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


def test_training_manifest_rejects_output_without_a_real_peft_adapter(tmp_path: Path) -> None:
    train = _dataset_fixture(tmp_path)
    output = tmp_path / "model"
    output.mkdir()
    (output / "training_metrics.json").write_text('{"train_loss":0.5}\n')

    with pytest.raises(TrainingWorkerError, match="adapter"):
        write_training_manifest(
            train,
            output_dir=output,
            run_id="run-1",
            experiment_id="exp-1",
            dataset_id="dataset-1",
            dataset_sha256=_dataset_digest(train),
            base_model_id=BASE_MODEL_ID,
            base_model_revision="a" * 40,
            qlora_config=_qlora_config(),
            training_metrics={"train_loss": 0.5},
        )


def test_training_refuses_to_reuse_a_preexisting_model_output(tmp_path: Path) -> None:
    train = _dataset_fixture(tmp_path)
    base_model_channel, base_model_sha256 = _base_model_channel(tmp_path)
    model_dir = tmp_path / "model"
    _write_adapter_artifacts(model_dir, b"stale-adapter")
    inputs = parse_training_inputs(
        {
            "SM_CHANNEL_TRAIN": str(train),
            "SM_CHANNEL_BASE_MODEL": str(base_model_channel),
            "SM_MODEL_DIR": str(model_dir),
            "RUN_ID": "run-1",
            "EXPERIMENT_ID": "exp-1",
            "DATASET_ID": "dataset-1",
            "DATASET_SHA256": _dataset_digest(train),
            "APPROVED_DATASET_ARTIFACT_ID": "dataset://dataset-1",
            "BASE_MODEL_ID": BASE_MODEL_ID,
            "BASE_MODEL_REVISION": "a" * 40,
            "BASE_MODEL_BUNDLE_SHA256": base_model_sha256,
            "QLORA_CONFIG": json.dumps(_qlora_config()),
        }
    )

    with pytest.raises(TrainingWorkerError, match="empty"):
        run_training(inputs)


def test_training_manifest_rejects_qlora_config_outside_fixed_search_space(tmp_path: Path) -> None:
    train = _dataset_fixture(tmp_path)
    _write_adapter_artifacts(tmp_path / "model", b"real-adapter")
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
    _write_adapter_artifacts(parent, b"parent")
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
    _write_adapter_artifacts(output, b"candidate")
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


def test_trainer_extracts_and_verifies_approved_parent_channel(tmp_path: Path) -> None:
    train = _dataset_fixture(tmp_path)
    base_model_channel, base_model_sha256 = _base_model_channel(tmp_path)
    parent = tmp_path / "parent"
    parent.mkdir()
    _write_adapter_artifacts(parent, b"approved parent")
    manifest_path = write_training_manifest(
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
    parent_manifest = json.loads(manifest_path.read_text())
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
        for path in parent.iterdir():
            bundle.add(path, arcname=path.name)
    archive_bytes = archive.getvalue()
    archive_sha = hashlib.sha256(archive_bytes).hexdigest()
    channel = tmp_path / "parent-channel"
    channel.mkdir()
    (channel / f"{archive_sha}.tar.gz").write_bytes(archive_bytes)

    inputs = parse_training_inputs(
        {
            "SM_CHANNEL_TRAIN": str(train),
            "SM_CHANNEL_BASE_MODEL": str(base_model_channel),
            "SM_CHANNEL_PARENT_ADAPTER": str(channel),
            "SM_MODEL_DIR": str(tmp_path / "model"),
            "RUN_ID": "run-1",
            "EXPERIMENT_ID": "run-1-2",
            "DATASET_ID": "dataset-1",
            "DATASET_SHA256": _dataset_digest(train),
            "APPROVED_DATASET_ARTIFACT_ID": "dataset://dataset-1",
            "BASE_MODEL_ID": BASE_MODEL_ID,
            "BASE_MODEL_REVISION": "a" * 40,
            "BASE_MODEL_BUNDLE_SHA256": base_model_sha256,
            "QLORA_CONFIG": json.dumps(_qlora_config()),
            "APPROVED_PARENT_ARTIFACT_ID": parent_manifest["artifact_id"],
            "APPROVED_PARENT_MANIFEST_SHA256": parent_manifest["manifest_sha256"],
            "APPROVED_PARENT_ARTIFACT_SHA256": parent_manifest["artifact_sha256"],
            "APPROVED_PARENT_ARCHIVE_SHA256": archive_sha,
        }
    )

    assert inputs.parent_adapter_dir is not None
    assert not inputs.parent_adapter_dir.is_relative_to(inputs.model_dir)
    assert list(inputs.model_dir.iterdir()) == []
    assert verify_parent_adapter(inputs.parent_adapter_dir)["manifest_sha256"] == parent_manifest[
        "manifest_sha256"
    ]


def test_evaluator_extracts_candidate_and_champion_archives(tmp_path: Path) -> None:
    train = _dataset_fixture(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_adapter_artifacts(checkpoint, b"candidate")
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
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
        for path in checkpoint.iterdir():
            bundle.add(path, arcname=path.name)
    payload = archive.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    candidate_channel = tmp_path / "candidate-channel"
    champion_channel = tmp_path / "champion-channel"
    sealed_channel = tmp_path / "sealed-channel"
    for channel in (candidate_channel, champion_channel, sealed_channel):
        channel.mkdir()
    for channel in (candidate_channel, champion_channel):
        (channel / f"{digest}.tar.gz").write_bytes(payload)

    inputs = parse_evaluation_inputs(
        {
            "RUN_ID": "run-1",
            "EXPERIMENT_ID": "exp-1",
            "EVALUATION_MANIFEST_SHA256": "b" * 64,
            "EVALUATION_SUITE_VERSION": "agent-eval-v1",
            "OBJECTIVE_SEED": "7",
            "CANDIDATE_ARCHIVE_SHA256": digest,
            "CHAMPION_ARCHIVE_SHA256": digest,
            "SM_OUTPUT_DATA_DIR": str(tmp_path / "evaluation-output"),
        },
        {
            "candidate": candidate_channel,
            "champion": champion_channel,
            "sealed": sealed_channel,
        },
    )

    assert verify_checkpoint_artifact(
        inputs.candidate_dir, run_id="run-1", experiment_id="exp-1"
    )["kind"] == "qlora-adapter"
    assert inputs.champion_dir is not None
    assert (inputs.champion_dir / "manifest.json").is_file()


def test_evaluator_extracts_content_addressed_base_model_channel(tmp_path: Path) -> None:
    base_snapshot = tmp_path / "base-snapshot"
    base_snapshot.mkdir()
    (base_snapshot / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma3_text",
                "architectures": ["Gemma3ForCausalLM"],
            }
        )
    )
    (base_snapshot / "model.safetensors").write_bytes(b"immutable-base-weights")
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
        for path in base_snapshot.iterdir():
            bundle.add(path, arcname=path.name)
    archive_bytes = archive.getvalue()
    archive_sha = hashlib.sha256(archive_bytes).hexdigest()
    base_channel = tmp_path / "base-model-channel"
    base_channel.mkdir()
    (base_channel / f"{archive_sha}.tar.gz").write_bytes(archive_bytes)
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
            "BASE_MODEL_ID": BASE_MODEL_ID,
            "BASE_MODEL_REVISION": "a" * 40,
            "BASE_MODEL_BUNDLE_SHA256": archive_sha,
            "EVALUATION_BASE_MODEL_DIR": str(base_channel),
            "SM_OUTPUT_DATA_DIR": str(tmp_path / "output"),
        },
        {"candidate": candidate, "sealed": sealed},
    )

    assert inputs.base_model_dir is not None
    assert (inputs.base_model_dir / "model.safetensors").read_bytes() == b"immutable-base-weights"
    assert inputs.base_model_archive_sha256 == archive_sha
    assert inputs.base_model_revision == "a" * 40


def test_evaluator_base_model_policy_loads_only_local_and_skips_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    class _Model:
        device = "cpu"

        def eval(self) -> _Model:
            return self

    class _Processor:
        pass

    class _AutoProcessor:
        @staticmethod
        def from_pretrained(path: Path, **kwargs: object) -> _Processor:
            calls.append(("processor", str(path), dict(kwargs)))
            return _Processor()

    class _AutoModel:
        @staticmethod
        def from_pretrained(path: Path, **kwargs: object) -> _Model:
            calls.append(("model", str(path), dict(kwargs)))
            return _Model()

    transformer_module = types.ModuleType("transformers")
    transformer_module.AutoProcessor = _AutoProcessor
    transformer_module.AutoModelForCausalLM = _AutoModel
    peft_module = types.ModuleType("peft")

    class _UnexpectedPeft:
        @staticmethod
        def from_pretrained(*_: object, **__: object) -> _Model:
            raise AssertionError("base-model champion must not load a LoRA adapter")

    peft_module.PeftModel = _UnexpectedPeft
    monkeypatch.setitem(sys.modules, "transformers", transformer_module)
    monkeypatch.setitem(sys.modules, "peft", peft_module)
    base_dir = tmp_path / "immutable-base"
    base_dir.mkdir()
    (base_dir / "config.json").write_text(
        json.dumps({"model_type": "gemma3_text", "architectures": ["Gemma3ForCausalLM"]})
    )
    (base_dir / "model.safetensors").write_bytes(b"weights")

    _model_policy(
        base_dir,
        {"base_model_id": BASE_MODEL_ID, "base_model_revision": "a" * 40},
        base_model_dir=base_dir,
        use_adapter=False,
    )

    assert [(kind, path) for kind, path, _ in calls] == [
        ("processor", str(base_dir)),
        ("model", str(base_dir)),
    ]
    assert all(call[2].get("local_files_only") is True for call in calls)


def test_evaluator_adapter_policy_applies_lora_to_local_base_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    class _Model:
        device = "cpu"

        def eval(self) -> _Model:
            return self

    class _Processor:
        pass

    class _AutoProcessor:
        @staticmethod
        def from_pretrained(path: Path, **kwargs: object) -> _Processor:
            calls.append(("processor", str(path), dict(kwargs)))
            return _Processor()

    class _AutoModel:
        @staticmethod
        def from_pretrained(path: Path, **kwargs: object) -> _Model:
            calls.append(("base", str(path), dict(kwargs)))
            return _Model()

    class _PeftModel:
        @staticmethod
        def from_pretrained(model: _Model, path: Path, **kwargs: object) -> _Model:
            calls.append(("adapter", str(path), dict(kwargs)))
            return model

    transformer_module = types.ModuleType("transformers")
    transformer_module.AutoProcessor = _AutoProcessor
    transformer_module.AutoModelForCausalLM = _AutoModel
    peft_module = types.ModuleType("peft")
    peft_module.PeftModel = _PeftModel
    monkeypatch.setitem(sys.modules, "transformers", transformer_module)
    monkeypatch.setitem(sys.modules, "peft", peft_module)
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    adapter_dir = tmp_path / "candidate-adapter"
    adapter_dir.mkdir()

    _model_policy(
        adapter_dir,
        {"base_model_id": BASE_MODEL_ID, "base_model_revision": "a" * 40},
        base_model_dir=base_dir,
        use_adapter=True,
    )

    assert [(kind, path) for kind, path, _ in calls] == [
        ("processor", str(base_dir)),
        ("base", str(base_dir)),
        ("adapter", str(adapter_dir)),
    ]
    assert all(call[2].get("local_files_only") is True for call in calls)


def test_first_evaluation_accepts_base_model_champion_and_adapter_candidate(
    tmp_path: Path,
) -> None:
    train = tmp_path / "training-input"
    train.mkdir()
    dataset_sha = "c" * 64
    (train / "dataset.json").write_text(
        json.dumps(
            {
                "manifest": {
                    "dataset_id": "dataset-1",
                    "run_id": "run-1",
                    "experiment_id": "exp-1",
                    "sha256": dataset_sha,
                }
            }
        )
    )
    candidate = tmp_path / "candidate"
    _write_adapter_artifacts(candidate, b"candidate-adapter")
    write_training_manifest(
        train,
        output_dir=candidate,
        run_id="run-1",
        experiment_id="exp-1",
        dataset_id="dataset-1",
        dataset_sha256=dataset_sha,
        base_model_id=BASE_MODEL_ID,
        base_model_revision="a" * 40,
        qlora_config=_qlora_config(),
    )

    base_snapshot = tmp_path / "base-snapshot"
    base_snapshot.mkdir()
    (base_snapshot / "config.json").write_text(
        json.dumps({"model_type": "gemma3_text", "architectures": ["Gemma3ForCausalLM"]})
    )
    (base_snapshot / "model.safetensors").write_bytes(b"immutable-base-weights")
    base_archive = io.BytesIO()
    with tarfile.open(fileobj=base_archive, mode="w:gz") as bundle:
        for path in base_snapshot.iterdir():
            bundle.add(path, arcname=path.name)
    base_archive_bytes = base_archive.getvalue()
    base_archive_sha = hashlib.sha256(base_archive_bytes).hexdigest()
    base_model_channel = tmp_path / "base-model-channel"
    champion_channel = tmp_path / "champion-channel"
    sealed = tmp_path / "sealed"
    for directory in (base_model_channel, champion_channel, sealed):
        directory.mkdir()
    for directory in (base_model_channel, champion_channel):
        (directory / f"{base_archive_sha}.tar.gz").write_bytes(base_archive_bytes)

    task_bytes = b'{"tasks":["hidden-1"]}\n'
    (sealed / "tasks.json").write_bytes(task_bytes)
    unsigned = {
        "objective_seed": 7,
        "suite": "AgentGym/AgentEval",
        "suite_version": "agent-eval-v1",
        "task_bundle_sha256": hashlib.sha256(task_bytes).hexdigest(),
        "task_count": 1,
    }
    evaluation_digest = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (sealed / "manifest.json").write_text(
        json.dumps({**unsigned, "manifest_sha256": evaluation_digest}, sort_keys=True)
    )
    inputs = parse_evaluation_inputs(
        {
            "RUN_ID": "run-1",
            "EXPERIMENT_ID": "exp-1",
            "EVALUATION_MANIFEST_SHA256": evaluation_digest,
            "EVALUATION_SUITE_VERSION": "agent-eval-v1",
            "OBJECTIVE_SEED": "7",
            "BASE_MODEL_ID": BASE_MODEL_ID,
            "BASE_MODEL_REVISION": "a" * 40,
            "BASE_MODEL_ARCHIVE_SHA256": base_archive_sha,
            "CHAMPION_ARCHIVE_SHA256": base_archive_sha,
            "CHAMPION_KIND": "base-model",
            "SM_OUTPUT_DATA_DIR": str(tmp_path / "output"),
        },
        {
            "candidate": candidate,
            "base_model": base_model_channel,
            "champion": champion_channel,
            "sealed": sealed,
        },
    )

    report_path = run_evaluation(inputs, policy=lambda task: ())
    report = json.loads(report_path.read_text())
    assert report["candidate_kind"] == "qlora-adapter"
    assert report["champion_kind"] == "base-model"
    assert report["champion_artifact_sha256"] == base_archive_sha

    assert inputs.base_model_dir is not None
    with pytest.raises(EvaluationWorkerError, match=r"manifest\.json"):
        verify_checkpoint_artifact(inputs.base_model_dir)


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
    _write_adapter_artifacts(candidate, b"candidate")
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
    assert report["run_id"] == "run-1"
    assert report["experiment_id"] == "exp-1"
    (sealed / "tasks.json").write_bytes(b'{"tasks":["hidden-2"]}\n')
    with pytest.raises(EvaluationWorkerError, match="task bundle"):
        run_evaluation(inputs, policy=lambda task: ())


@pytest.mark.parametrize("identity_key", ["run_id", "experiment_id"])
def test_static_sealed_manifest_rejects_per_run_identity(identity_key: str, tmp_path: Path) -> None:
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    task_bytes = b'{"tasks":["hidden-1"]}\n'
    (sealed / "tasks.json").write_bytes(task_bytes)
    unsigned = {
        "objective_seed": 7,
        "suite": "AgentGym/AgentEval",
        "suite_version": "agent-eval-v1",
        "task_bundle_sha256": hashlib.sha256(task_bytes).hexdigest(),
        "task_count": 1,
        identity_key: "run-1" if identity_key == "run_id" else "run-1-1",
    }
    digest = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (sealed / "manifest.json").write_text(json.dumps({**unsigned, "manifest_sha256": digest}))
    inputs = parse_evaluation_inputs(
        {
            "RUN_ID": "run-1",
            "EXPERIMENT_ID": "run-1-1",
            "EVALUATION_MANIFEST_SHA256": digest,
            "EVALUATION_SUITE_VERSION": "agent-eval-v1",
            "OBJECTIVE_SEED": "7",
            "SM_OUTPUT_DATA_DIR": str(tmp_path / "output"),
        },
        {"candidate": tmp_path, "sealed": sealed},
    )

    with pytest.raises(EvaluationWorkerError, match=r"static sealed manifest.*run identity"):
        _sealed_manifest(inputs)


def test_static_sealed_manifest_is_reused_across_run_scopes(tmp_path: Path) -> None:
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    task_bytes = b'{"tasks":["hidden-1"]}\n'
    (sealed / "tasks.json").write_bytes(task_bytes)
    unsigned = {
        "objective_seed": 7,
        "suite": "AgentGym/AgentEval",
        "suite_version": "agent-eval-v1",
        "task_bundle_sha256": hashlib.sha256(task_bytes).hexdigest(),
        "task_count": 1,
    }
    digest = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (sealed / "manifest.json").write_text(json.dumps({**unsigned, "manifest_sha256": digest}))

    for run_id, experiment_id in (("run-1", "run-1-1"), ("run-2", "run-2-3")):
        inputs = parse_evaluation_inputs(
            {
                "RUN_ID": run_id,
                "EXPERIMENT_ID": experiment_id,
                "EVALUATION_MANIFEST_SHA256": digest,
                "EVALUATION_SUITE_VERSION": "agent-eval-v1",
                "OBJECTIVE_SEED": "7",
                "SM_OUTPUT_DATA_DIR": str(tmp_path / "output" / experiment_id),
            },
            {"candidate": tmp_path, "sealed": sealed},
        )
        manifest, task_ids = _sealed_manifest(inputs)

        assert manifest["manifest_sha256"] == digest
        assert task_ids == ["hidden-1"]


def test_training_admission_rejects_untrusted_source_type(tmp_path: Path) -> None:
    train = _dataset_fixture(tmp_path)
    raw = json.loads((train / "dataset.json").read_text())
    raw["rows"][0]["source_type"] = "untrusted"
    raw["manifest"]["sha256"] = hashlib.sha256(
        DatasetRow.model_validate(raw["rows"][0]).canonical_json().encode()
    ).hexdigest()
    (train / "dataset.json").write_text(json.dumps(raw))
    inputs = TrainingInputs(
        train_dir=train,
        model_dir=tmp_path / "model",
        base_model_dir=tmp_path / "base-model",
        run_id="run-1",
        experiment_id="exp-1",
        dataset_id="dataset-1",
        dataset_sha256=raw["manifest"]["sha256"],
        base_model_id=BASE_MODEL_ID,
        base_model_revision="a" * 40,
        dataset_artifact_id="dataset://dataset-1",
    )
    with pytest.raises(TrainingWorkerError, match=r"source|verified"):
        load_training_dataset(inputs)


def test_artifact_id_is_bound_to_verified_content_digest(tmp_path: Path) -> None:
    train = _dataset_fixture(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_adapter_artifacts(checkpoint, b"adapter")
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


def test_evaluator_rejects_self_consistent_manifest_without_adapter_files(tmp_path: Path) -> None:
    train = _dataset_fixture(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    _write_adapter_artifacts(checkpoint)
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
    (checkpoint / "adapter_config.json").unlink()
    (checkpoint / "adapter_model.safetensors").unlink()
    payload = json.loads(manifest_path.read_text())
    payload["artifact_files"] = [
        entry for entry in payload["artifact_files"] if entry["path"] == "training_metrics.json"
    ]
    payload["artifact_sha256"] = hashlib.sha256(
        json.dumps(payload["artifact_files"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload["artifact_id"] = f"checkpoint://{payload['artifact_sha256']}"
    payload.pop("manifest_sha256")
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(payload, sort_keys=True) + "\n")

    with pytest.raises(EvaluationWorkerError, match="adapter_config"):
        verify_checkpoint_artifact(checkpoint)


def test_evaluator_rejects_unlisted_duplicate_and_symlink_files(tmp_path: Path) -> None:
    train = _dataset_fixture(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_adapter_artifacts(checkpoint, b"adapter")
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
    _write_adapter_artifacts(checkpoint, b"adapter")
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


def test_evaluator_binds_sealed_manifest_identity(tmp_path: Path) -> None:
    train = _dataset_fixture(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_adapter_artifacts(checkpoint, b"adapter")
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
        "objective_seed": 7,
        "suite": "AgentGym/AgentEval",
        "suite_version": "agent-eval-v1",
        "task_bundle_sha256": hashlib.sha256(task_bytes).hexdigest(),
        "task_count": 1,
    }
    digest = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (sealed / "manifest.json").write_text(json.dumps({**unsigned, "manifest_sha256": digest}))
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
