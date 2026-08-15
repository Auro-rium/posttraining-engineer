"""Vertex-compatible FunctionGemma QLoRA training worker.

The worker consumes verifier-approved JSONL rows, writes a PEFT adapter and an
immutable manifest, and uploads both to the requested GCS prefix. It never
receives held-out evaluation data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

ALLOWED_RANKS = {8, 16, 32}
ALLOWED_LEARNING_RATES = {5e-5, 1e-4, 2e-4}
ALLOWED_EPOCHS = {2, 3, 5}
ALLOWED_DROPOUTS = {0.0, 0.05}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--dataset-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--model-id", default="google/functiongemma-270m-it")
    parser.add_argument("--hf-secret-id")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--effective-batch-size", type=int, default=32)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    allowed = {
        "rank": (args.rank, ALLOWED_RANKS),
        "learning_rate": (args.learning_rate, ALLOWED_LEARNING_RATES),
        "epochs": (args.epochs, ALLOWED_EPOCHS),
        "dropout": (args.dropout, ALLOWED_DROPOUTS),
    }
    invalid = [name for name, (value, values) in allowed.items() if value not in values]
    if invalid:
        raise ValueError(f"configuration outside approved bounds: {', '.join(invalid)}")
    if args.max_seq_length != 512 or args.effective_batch_size != 32:
        raise ValueError("sequence length and effective batch size are fixed for the hackathon")


def split_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"expected gs:// URI, got {uri!r}")
    bucket, _, prefix = uri[5:].partition("/")
    if not bucket or not prefix:
        raise ValueError(f"GCS URI requires bucket and object prefix: {uri!r}")
    return bucket, prefix.rstrip("/")


def download_dataset(uri: str, destination: Path) -> None:
    from google.cloud import storage

    bucket_name, blob_name = split_gcs_uri(uri)
    storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(destination)


def upload_directory(source: Path, output_uri: str) -> None:
    from google.cloud import storage

    bucket_name, prefix = split_gcs_uri(output_uri)
    bucket = storage.Client().bucket(bucket_name)
    for path in source.rglob("*"):
        if path.is_file():
            relative = path.relative_to(source).as_posix()
            bucket.blob(f"{prefix}/{relative}").upload_from_filename(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(package: str) -> str:
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:  # pragma: no cover - provenance best effort only
        return "unknown"


def read_hf_token(secret_id: str | None) -> str | None:
    """Resolve optional model access at runtime without placing it in job args."""

    token = os.environ.get("HF_TOKEN")
    if token or not secret_id:
        return token
    from google.cloud import secretmanager

    project = os.environ.get("CLOUD_ML_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError("Vertex did not expose a project ID for Secret Manager access")
    name = f"projects/{project}/secrets/{secret_id}/versions/latest"
    response = secretmanager.SecretManagerServiceClient().access_secret_version(
        request={"name": name}
    )
    return response.payload.data.decode("utf-8")


def train(args: argparse.Namespace, workdir: Path) -> dict[str, Any]:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    dataset_path = workdir / "verified-sft.jsonl"
    output_path = workdir / "adapter"
    download_dataset(args.dataset_uri, dataset_path)

    token = read_hf_token(args.hf_secret_id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=token)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=token,
        device_map="auto",
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
    )
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    def render(row: dict[str, Any]) -> dict[str, str]:
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError("each verified SFT row requires a messages list")
        tools = row.get("tools")
        return {
            "text": tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    rendered = dataset.map(render, remove_columns=dataset.column_names)
    micro_batch_size = 4
    gradient_accumulation = args.effective_batch_size // micro_batch_size
    training_config = SFTConfig(
        output_dir=str(output_path),
        dataset_text_field="text",
        max_length=args.max_seq_length,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        seed=17,
    )
    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=rendered,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    result = trainer.train()
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    manifest = {
        "schema_version": "1.0",
        "run_id": args.run_id,
        "experiment_id": args.experiment_id,
        "base_model": args.model_id,
        "dataset_uri": args.dataset_uri,
        "dataset_sha256": sha256_file(dataset_path),
        "output_uri": args.output_uri,
        "configuration": {
            "rank": args.rank,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "dropout": args.dropout,
            "max_seq_length": args.max_seq_length,
            "effective_batch_size": args.effective_batch_size,
            "seed": 17,
        },
        "train_metrics": result.metrics,
        "runtime": {
            "python": platform.python_version(),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "trl": package_version("trl"),
            "peft": package_version("peft"),
            "cuda": torch.version.cuda,
        },
    }
    manifest_path = output_path / "training-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    upload_directory(output_path, args.output_uri)
    return manifest


def main() -> None:
    args = parse_args()
    validate_args(args)
    # Vertex's AIP_MODEL_DIR may be a gs:// URI. Training uses an explicit local
    # scratch directory and uploads the final adapter through the storage SDK.
    workdir = Path(os.environ.get("TRAIN_WORKDIR", "/tmp/post-training-worker"))
    workdir.mkdir(parents=True, exist_ok=True)
    manifest = train(args, workdir)
    print(
        json.dumps(
            {
                "event": "training_complete",
                "run_id": manifest["run_id"],
                "experiment_id": manifest["experiment_id"],
                "output_uri": manifest["output_uri"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
