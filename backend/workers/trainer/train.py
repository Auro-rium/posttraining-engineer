"""SageMaker entrypoint for real FunctionGemma SFT+QLoRA training.

The module deliberately keeps contract validation and checksum helpers free of
Transformers, PEFT, TRL, and bitsandbytes imports.  Those libraries are loaded
only by :func:`run_training`, so the coordinator can validate a job contract in
a small Python environment.  A missing model, dataset, or output artifact is a
hard failure; this worker never emits a fabricated checkpoint manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

BASE_MODEL_ID = "google/functiongemma-270m-it"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")
_ALLOWED_SPLITS = frozenset({"train", "replay"})
_SEALED_NAMES = frozenset({"hidden", "sealed", "validation", "eval", "evaluation", "test"})


class TrainingWorkerError(ValueError):
    """The training job input or output violates the worker contract."""


class TrainingArtifactError(TrainingWorkerError):
    """A checkpoint artifact was absent or could not be verified."""


@dataclass(frozen=True, slots=True)
class TrainingInputs:
    train_dir: Path
    model_dir: Path
    run_id: str
    experiment_id: str
    dataset_id: str
    dataset_sha256: str
    base_model_id: str
    base_model_revision: str
    parent_adapter_dir: Path | None = None
    qlora_config: Mapping[str, Any] | None = None


def _required(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise TrainingWorkerError(f"{name} is required")
    return value


def _digest(value: str, name: str) -> str:
    if not _SHA256.fullmatch(value):
        raise TrainingWorkerError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _path(value: str, name: str, *, directory: bool = True) -> Path:
    candidate = Path(value).expanduser()
    if directory and (not candidate.exists() or not candidate.is_dir()):
        raise TrainingWorkerError(f"{name} must be an existing directory")
    if not directory and (not candidate.exists() or not candidate.is_file()):
        raise TrainingWorkerError(f"{name} must be an existing file")
    return candidate.resolve()


def _normalise_channels(channels: Mapping[str, str | Path]) -> dict[str, Path]:
    normalised: dict[str, Path] = {}
    for name, value in channels.items():
        key = str(name).strip().lower()
        if key in _SEALED_NAMES or any(part in _SEALED_NAMES for part in key.split("_")):
            raise TrainingWorkerError(f"trainer cannot consume sealed/evaluation channel {name!r}")
        if key != "train":
            raise TrainingWorkerError(
                f"unsupported trainer channel {name!r}; only train is allowed"
            )
        candidate = Path(value).expanduser()
        if not candidate.exists() or not candidate.is_dir():
            raise TrainingWorkerError("train channel must be an existing directory")
        normalised[key] = candidate.resolve()
    return normalised


def parse_training_inputs(
    env: Mapping[str, str] | None = None,
    channels: Mapping[str, str | Path] | None = None,
) -> TrainingInputs:
    """Parse the strict SageMaker training channel/environment contract."""

    values = dict(os.environ if env is None else env)
    channel_values = dict(channels or {})
    for env_name, env_value in values.items():
        if env_name.startswith("SM_CHANNEL_") and env_value.strip():
            channel_name = env_name.removeprefix("SM_CHANNEL_").lower()
            if channel_name != "train":
                _normalise_channels({channel_name: env_value})
    train_env = values.get("SM_CHANNEL_TRAIN", "").strip()
    if not train_env and "train" not in channel_values:
        raise TrainingWorkerError("SM_CHANNEL_TRAIN is required")
    if train_env:
        channel_values.setdefault("train", train_env)
    channel_paths = _normalise_channels(channel_values)
    train_dir = channel_paths["train"]
    model_dir = Path(_required(values, "SM_MODEL_DIR")).expanduser().resolve()
    if model_dir.exists() and not model_dir.is_dir():
        raise TrainingWorkerError("SM_MODEL_DIR must be a directory")
    model_dir.mkdir(parents=True, exist_ok=True)
    revision = _required(values, "BASE_MODEL_REVISION")
    if not _REVISION.fullmatch(revision):
        raise TrainingWorkerError("BASE_MODEL_REVISION must be a 40-character immutable revision")
    model_id = _required(values, "BASE_MODEL_ID")
    if model_id != BASE_MODEL_ID:
        raise TrainingWorkerError(f"BASE_MODEL_ID must equal {BASE_MODEL_ID!r}")
    parent = values.get("PARENT_ADAPTER_DIR", "").strip()
    parent_path = _path(parent, "PARENT_ADAPTER_DIR") if parent else None
    config_text = values.get("QLORA_CONFIG", "").strip()
    config: Mapping[str, Any] | None = None
    if config_text:
        try:
            decoded = json.loads(config_text)
        except json.JSONDecodeError as exc:
            raise TrainingWorkerError("QLORA_CONFIG must be valid JSON") from exc
        if not isinstance(decoded, dict):
            raise TrainingWorkerError("QLORA_CONFIG must be a JSON object")
        config = decoded
    return TrainingInputs(
        train_dir=train_dir,
        model_dir=model_dir,
        run_id=_required(values, "RUN_ID"),
        experiment_id=_required(values, "EXPERIMENT_ID"),
        dataset_id=_required(values, "DATASET_ID"),
        dataset_sha256=_digest(_required(values, "DATASET_SHA256"), "DATASET_SHA256"),
        base_model_id=model_id,
        base_model_revision=revision,
        parent_adapter_dir=parent_path,
        qlora_config=config,
    )


def file_sha256(path: Path) -> str:
    """Hash one regular file without loading it all into memory."""

    if not path.is_file():
        raise TrainingArtifactError(f"artifact file is absent: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_files(output_dir: Path) -> list[dict[str, Any]]:
    if not output_dir.exists() or not output_dir.is_dir():
        raise TrainingArtifactError(f"training output directory is absent: {output_dir}")
    files = sorted(
        item
        for item in output_dir.rglob("*")
        if item.is_file() and item.name != "manifest.json"
    )
    if not files:
        raise TrainingArtifactError("training produced no checkpoint artifacts")
    return [
        {
            "path": item.relative_to(output_dir).as_posix(),
            "size_bytes": item.stat().st_size,
            "sha256": file_sha256(item),
        }
        for item in files
    ]


def _artifact_digest(files: list[dict[str, Any]]) -> str:
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _manifest_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def format_sft_example(row: Mapping[str, Any]) -> dict[str, str]:
    """Format one objective row as the prompt/completion policy contract."""

    task_id = row.get("task_id")
    messages = row.get("messages")
    if not isinstance(task_id, str) or not task_id.strip():
        raise TrainingWorkerError("SFT row task_id is required")
    if not isinstance(messages, (list, tuple)) or not messages:
        raise TrainingWorkerError("SFT row messages are required")
    observations: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    service = ""
    for message in messages:
        if not isinstance(message, Mapping):
            raise TrainingWorkerError("SFT row messages must be mappings")
        name = message.get("name")
        arguments = message.get("arguments", {})
        observation = message.get("observation", {})
        if not isinstance(name, str) or not name.strip() or not isinstance(arguments, Mapping):
            raise TrainingWorkerError("SFT row contains an invalid tool call")
        if not isinstance(observation, Mapping):
            raise TrainingWorkerError("SFT row contains an invalid observation")
        if not service and isinstance(observation.get("service"), str):
            service = observation["service"]
        observations.append(dict(observation))
        actions.append({"tool": name, "arguments": dict(arguments)})
    prompt = json.dumps(
        {
            "task_id": task_id,
            "objective": "restore the service and pass its health check",
            "service": service,
            "observations": observations,
            "output": "JSON array of {tool, arguments} objects",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    completion = json.dumps(actions, sort_keys=True, separators=(",", ":"))
    return {"prompt": prompt, "completion": completion}


def _verify_manifest_directory(directory: Path, *, label: str) -> dict[str, Any]:
    if not directory.exists() or not directory.is_dir():
        raise TrainingWorkerError(f"{label} directory is absent")
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise TrainingWorkerError(f"{label} manifest.json is absent")
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingWorkerError(f"{label} manifest is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise TrainingWorkerError(f"{label} manifest must be a JSON object")
    digest = payload.get("manifest_sha256")
    artifact_digest = payload.get("artifact_sha256")
    files = payload.get("artifact_files")
    if (
        payload.get("kind") != "qlora-adapter"
        or not isinstance(digest, str)
        or not _SHA256.fullmatch(digest)
        or not isinstance(artifact_digest, str)
        or not _SHA256.fullmatch(artifact_digest)
        or not isinstance(files, list)
        or not files
    ):
        raise TrainingWorkerError(f"{label} manifest is incomplete")
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if _manifest_digest(unsigned) != digest:
        raise TrainingWorkerError(f"{label} manifest checksum does not match content")
    actual_files = _artifact_files(directory)
    if actual_files != files:
        raise TrainingWorkerError(f"{label} artifact file list does not match content")
    if _artifact_digest(actual_files) != artifact_digest:
        raise TrainingWorkerError(f"{label} artifact checksum does not match content")
    return payload


def verify_parent_adapter(parent_adapter_dir: Path) -> dict[str, Any]:
    """Verify an immutable parent adapter manifest before it can be loaded."""

    return _verify_manifest_directory(parent_adapter_dir, label="parent adapter")


def write_training_manifest(
    dataset_dir: Path,
    *,
    output_dir: Path,
    run_id: str,
    experiment_id: str,
    dataset_id: str,
    dataset_sha256: str,
    base_model_id: str,
    base_model_revision: str,
    qlora_config: Mapping[str, Any],
    parent_manifest: Mapping[str, Any] | None = None,
    training_metrics: Mapping[str, Any] | None = None,
) -> Path:
    """Write a deterministic, content-addressed checkpoint manifest.

    ``dataset_dir`` is retained as a required input to ensure callers have a
    real source dataset.  The manifest itself includes only its identity and
    the produced output file checksums; it never copies rows or prompts.
    """

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise TrainingWorkerError("dataset input directory is absent")
    if not dataset_id.strip() or not run_id.strip() or not experiment_id.strip():
        raise TrainingWorkerError("run, experiment, and dataset IDs are required")
    _digest(dataset_sha256, "dataset_sha256")
    if base_model_id != BASE_MODEL_ID:
        raise TrainingWorkerError(f"base_model_id must equal {BASE_MODEL_ID!r}")
    if not _REVISION.fullmatch(base_model_revision):
        raise TrainingWorkerError("base_model_revision must be a 40-character immutable revision")
    dataset_candidates = (dataset_dir / "dataset.json", dataset_dir / "manifest.json")
    dataset_manifest_path = next(
        (item for item in dataset_candidates if item.is_file()), None
    )
    if dataset_manifest_path is None:
        raise TrainingWorkerError("dataset manifest is absent")
    try:
        source_payload = json.loads(dataset_manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingWorkerError("dataset manifest is not valid JSON") from exc
    source_manifest = source_payload.get("manifest", source_payload)
    if not isinstance(source_manifest, dict):
        raise TrainingWorkerError("dataset manifest is not a JSON object")
    if (
        source_manifest.get("dataset_id") != dataset_id
        or source_manifest.get("sha256") != dataset_sha256
        or source_manifest.get("run_id") != run_id
        or source_manifest.get("experiment_id") != experiment_id
    ):
        raise TrainingWorkerError("dataset manifest does not match training input")
    files = _artifact_files(output_dir)
    config = _validate_qlora(qlora_config)
    parent: dict[str, Any] | None = None
    if parent_manifest is not None:
        parent = {
            "artifact_id": str(
                parent_manifest.get(
                    "artifact_id", f"checkpoint://{parent_manifest.get('manifest_sha256', '')[:24]}"
                )
            ),
            "manifest_sha256": parent_manifest.get("manifest_sha256"),
            "artifact_sha256": parent_manifest.get("artifact_sha256"),
            "base_model_revision": parent_manifest.get("base_model_revision"),
            "qlora_config": parent_manifest.get("qlora_config"),
        }
        if not isinstance(parent["manifest_sha256"], str) or not _SHA256.fullmatch(
            parent["manifest_sha256"]
        ):
            raise TrainingWorkerError("parent adapter manifest digest is invalid")
        if not isinstance(parent["artifact_sha256"], str) or not _SHA256.fullmatch(
            parent["artifact_sha256"]
        ):
            raise TrainingWorkerError("parent adapter artifact digest is invalid")
        if not isinstance(parent["base_model_revision"], str) or not _REVISION.fullmatch(
            parent["base_model_revision"]
        ):
            raise TrainingWorkerError("parent adapter base revision is invalid")
        if not isinstance(parent["qlora_config"], Mapping):
            raise TrainingWorkerError("parent adapter QLoRA config is absent")
        if parent["base_model_revision"] != base_model_revision:
            raise TrainingWorkerError("parent adapter base revision does not match training input")
        if parent["qlora_config"] != config:
            raise TrainingWorkerError("parent adapter QLoRA config is incompatible")
    payload: dict[str, Any] = {
        "schema_version": "trainer-manifest-v1",
        "kind": "qlora-adapter",
        "run_id": run_id,
        "experiment_id": experiment_id,
        "dataset_id": dataset_id,
        "dataset_sha256": dataset_sha256,
        "base_model_id": base_model_id,
        "base_model_revision": base_model_revision,
        "qlora_config": json.loads(json.dumps(config, sort_keys=True)),
        "artifact_files": files,
        "artifact_sha256": _artifact_digest(files),
    }
    if parent is not None:
        payload["parent_adapter"] = parent
    if training_metrics is not None:
        metrics = json.loads(json.dumps(dict(training_metrics), sort_keys=True))
        if not isinstance(metrics, dict) or not metrics:
            raise TrainingWorkerError("training metrics must be a non-empty mapping")
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            for value in metrics.values()
        ):
            raise TrainingWorkerError("training metrics must contain numeric values")
        payload["training_metrics"] = metrics
    payload["manifest_sha256"] = _manifest_digest(payload)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return manifest_path


def load_training_dataset(inputs: TrainingInputs) -> Any:
    """Load and validate the objective Dataset without importing ML libraries."""

    from app.objective.models import Dataset

    candidates = (inputs.train_dir / "dataset.json", inputs.train_dir / "dataset.jsonl")
    source = next((item for item in candidates if item.is_file()), None)
    if source is None:
        raise TrainingWorkerError(
            "training dataset artifact dataset.json or dataset.jsonl is absent"
        )
    try:
        if source.name == "dataset.json":
            dataset = Dataset.model_validate_json(source.read_text())
        else:
            from app.objective.models import DatasetManifest, DatasetRow

            manifest_path = inputs.train_dir / "manifest.json"
            if not manifest_path.is_file():
                raise TrainingWorkerError("dataset.jsonl requires manifest.json")
            rows = tuple(
                DatasetRow.model_validate_json(line)
                for line in source.read_text().splitlines()
                if line.strip()
            )
            dataset = Dataset(
                manifest=DatasetManifest.model_validate_json(manifest_path.read_text()), rows=rows
            )
    except TrainingWorkerError:
        raise
    except Exception as exc:
        raise TrainingWorkerError("training dataset failed contract validation") from exc
    if (
        dataset.manifest.dataset_id != inputs.dataset_id
        or dataset.manifest.sha256 != inputs.dataset_sha256
        or dataset.manifest.run_id != inputs.run_id
        or dataset.manifest.experiment_id != inputs.experiment_id
    ):
        raise TrainingWorkerError("training dataset manifest does not match job inputs")
    if not dataset.rows:
        raise TrainingWorkerError("training dataset must contain at least one row")
    source_ids = tuple(row.source_trajectory_id for row in dataset.rows)
    if (
        len(set(source_ids)) != len(source_ids)
        or tuple(dataset.manifest.source_trajectory_ids) != source_ids
    ):
        raise TrainingWorkerError("training dataset source trajectory references are not canonical")
    if not dataset.manifest.target_failure_classes:
        raise TrainingWorkerError("training dataset has no declared failure classes")
    if any(
        row.split.value not in _ALLOWED_SPLITS
        or not row.verifier_confirmed
        or row.source_type != "verified_replay"
        or row.failure_label not in dataset.manifest.target_failure_classes
        or any(
            marker in row.task_id.lower() or marker in row.source_trajectory_id.lower()
            for marker in ("hidden", "sealed", "validation", "eval", "test")
        )
        for row in dataset.rows
    ):
        raise TrainingWorkerError("training dataset contains unverified, untrusted, or sealed rows")
    return dataset


def _validate_qlora(config: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from app.autonomous.agents import validate_qlora_config

        return validate_qlora_config(config).model_dump(mode="json")
    except Exception as exc:
        raise TrainingWorkerError("QLORA_CONFIG is outside the fixed search space") from exc


def run_training(inputs: TrainingInputs) -> Path:
    """Run actual Transformers/PEFT QLoRA training and return its manifest."""

    dataset = load_training_dataset(inputs)
    config = _validate_qlora(inputs.qlora_config or {})
    parent_manifest: dict[str, Any] | None = None
    if inputs.parent_adapter_dir is not None:
        parent_manifest = verify_parent_adapter(inputs.parent_adapter_dir)
        if parent_manifest.get("base_model_revision") != inputs.base_model_revision:
            raise TrainingWorkerError("parent adapter base revision does not match training input")
        if parent_manifest.get("qlora_config") != config:
            raise TrainingWorkerError("parent adapter QLoRA config is incompatible")
    try:
        # Heavy dependencies are intentionally local to the real execution path.
        import torch  # type: ignore[import-not-found]
        from peft import (  # type: ignore[import-not-found]
            LoraConfig,
            PeftModel,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise TrainingWorkerError(
            "Transformers/PEFT training dependencies are unavailable"
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            inputs.base_model_id, revision=inputs.base_model_revision, trust_remote_code=False
        )
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            inputs.base_model_id,
            revision=inputs.base_model_revision,
            quantization_config=quantization,
            device_map="auto",
            trust_remote_code=False,
        )
        model = prepare_model_for_kbit_training(model)
        if inputs.parent_adapter_dir is not None:
            model = PeftModel.from_pretrained(
                model, str(inputs.parent_adapter_dir), is_trainable=True
            )
        else:
            model = get_peft_model(
                model,
                LoraConfig(
                    r=config["rank"],
                    lora_alpha=config["alpha"],
                    lora_dropout=config["dropout"],
                    target_modules=list(config["target_modules"]),
                    task_type="CAUSAL_LM",
                ),
        )

        def tokenize(row: Any) -> dict[str, Any]:
            example = format_sft_example(row)
            rendered = f"{example['prompt']}\n{example['completion']}"
            return cast(
                dict[str, Any],
                tokenizer(
                    rendered,
                    truncation=True,
                    max_length=config["sequence_length"],
                ),
            )

        rows = [
            {"task_id": row.task_id, "messages": row.messages}
            for row in dataset.rows
        ]
        tokenized = [tokenize(row) for row in rows]
        args = TrainingArguments(
            output_dir=str(inputs.model_dir),
            num_train_epochs=config["epochs"],
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            learning_rate=config["learning_rate"],
            max_steps=-1,
            report_to=[],
            save_strategy="no",
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )
        training_result = trainer.train()
        raw_metrics = getattr(training_result, "metrics", None)
        if not isinstance(raw_metrics, Mapping) or not raw_metrics:
            raise TrainingWorkerError("training did not return metrics")
        metrics: dict[str, Any] = {}
        for name, value in raw_metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            metrics[str(name)] = value
        if not metrics:
            raise TrainingWorkerError("training did not return numeric metrics")
        (inputs.model_dir / "training_metrics.json").write_text(
            json.dumps(metrics, sort_keys=True, separators=(",", ":")) + "\n"
        )
        model.save_pretrained(inputs.model_dir)
        tokenizer.save_pretrained(inputs.model_dir)
    except Exception as exc:
        raise TrainingWorkerError(f"real QLoRA training failed: {type(exc).__name__}") from exc
    return write_training_manifest(
        inputs.train_dir,
        output_dir=inputs.model_dir,
        run_id=inputs.run_id,
        experiment_id=inputs.experiment_id,
        dataset_id=inputs.dataset_id,
        dataset_sha256=inputs.dataset_sha256,
        base_model_id=inputs.base_model_id,
        base_model_revision=inputs.base_model_revision,
        qlora_config=config,
        parent_manifest=parent_manifest,
        training_metrics=metrics,
    )


def main() -> None:
    inputs = parse_training_inputs()
    manifest = run_training(inputs)
    # This line executes only after the checkpoint and manifest are verified.
    print(json.dumps(json.loads(manifest.read_text()), sort_keys=True))


if __name__ == "__main__":
    main()


# Descriptive aliases keep SageMaker orchestration adapters decoupled from the
# CLI spelling while preserving one implementation of each contract.
parse_training_channels = parse_training_inputs
create_training_manifest = write_training_manifest
train = run_training
