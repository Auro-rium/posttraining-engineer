from __future__ import annotations

import argparse

import pytest

from training.train import split_gcs_uri, validate_args


def config(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "rank": 16,
        "learning_rate": 1e-4,
        "epochs": 3,
        "dropout": 0.05,
        "max_seq_length": 512,
        "effective_batch_size": 32,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_training_config_accepts_only_whitelisted_values() -> None:
    validate_args(config())

    with pytest.raises(ValueError, match="rank"):
        validate_args(config(rank=64))


def test_fixed_training_dimensions_cannot_drift() -> None:
    with pytest.raises(ValueError, match="fixed"):
        validate_args(config(max_seq_length=1024))


def test_split_gcs_uri_requires_bucket_and_object() -> None:
    assert split_gcs_uri("gs://artifacts/run/dataset.jsonl") == (
        "artifacts",
        "run/dataset.jsonl",
    )
    with pytest.raises(ValueError):
        split_gcs_uri("/tmp/dataset.jsonl")
