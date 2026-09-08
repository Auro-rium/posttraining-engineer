from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.stage_functiongemma_checkpoint import (
    TARGET_MODEL_ID,
    CheckpointStagingError,
    build_deterministic_bundle,
    stage_checkpoint,
    validate_checkpoint_directory,
)

REVISION = "a" * 40


def _checkpoint(path: Path, *, include_weights: bool = True) -> Path:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"model_type": "gemma3_text"}, sort_keys=True), encoding="utf-8"
    )
    (path / "tokenizer.json").write_text("{\"version\": 1}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    if include_weights:
        (path / "model.safetensors").write_bytes(b"weights")
    return path


def test_revision_is_required_to_be_an_immutable_commit(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")

    with pytest.raises(CheckpointStagingError, match=r"immutable.*40-character"):
        validate_checkpoint_directory(checkpoint, revision="main")


def test_missing_required_checkpoint_file_fails_closed(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint", include_weights=False)

    with pytest.raises(CheckpointStagingError, match="weight"):
        validate_checkpoint_directory(checkpoint, revision=REVISION)


def test_gated_incomplete_and_cache_lock_inputs_are_rejected(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    (checkpoint / ".gated").write_text("true", encoding="utf-8")
    with pytest.raises(CheckpointStagingError, match="gated"):
        validate_checkpoint_directory(checkpoint, revision=REVISION)

    (checkpoint / ".gated").unlink()
    (checkpoint / "download.lock").write_text("", encoding="utf-8")
    with pytest.raises(CheckpointStagingError, match="lock"):
        validate_checkpoint_directory(checkpoint, revision=REVISION)

    (checkpoint / "download.lock").unlink()
    (checkpoint / "model.safetensors.incomplete").write_bytes(b"partial")
    with pytest.raises(CheckpointStagingError, match="incomplete"):
        validate_checkpoint_directory(checkpoint, revision=REVISION)


def test_bundle_bytes_and_hash_are_deterministic(tmp_path: Path) -> None:
    first = _checkpoint(tmp_path / "first")
    second = _checkpoint(tmp_path / "second")
    # Different filesystem mtimes and directory names must not affect identity.
    bundle_a = build_deterministic_bundle(first, revision=REVISION)
    bundle_b = build_deterministic_bundle(second, revision=REVISION)

    assert bundle_a.data == bundle_b.data
    assert bundle_a.sha256 == bundle_b.sha256
    assert bundle_a.sha256 == hashlib.sha256(bundle_a.data).hexdigest()
    assert bundle_a.model_id == TARGET_MODEL_ID


class _VersionedS3:
    def __init__(self) -> None:
        self.put_calls: list[dict[str, object]] = []

    def get_bucket_versioning(self, **kwargs: object) -> dict[str, str]:
        assert kwargs == {"Bucket": "artifacts"}
        return {"Status": "Enabled"}

    def put_object(self, **kwargs: object) -> dict[str, str]:
        self.put_calls.append(kwargs)
        return {"VersionId": "version-17", "ETag": '"etag"'}


def test_stage_upload_is_content_addressed_and_has_version_metadata(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    client = _VersionedS3()

    staged = stage_checkpoint(
        checkpoint,
        bucket="artifacts",
        revision=REVISION,
        s3_client=client,
    )

    call = client.put_calls[0]
    metadata = call["Metadata"]
    assert isinstance(metadata, dict)
    assert call["Bucket"] == "artifacts"
    assert str(call["Key"]).endswith(f"/{REVISION}/{staged.sha256}.tar.gz")
    assert metadata["sha256"] == staged.sha256
    assert metadata["hf-revision"] == REVISION
    assert metadata["model-id"] == TARGET_MODEL_ID
    assert staged.version_id == "version-17"
    assert staged.version_ref.endswith("?versionId=version-17")


def test_staging_rejects_unversioned_bucket_without_upload(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")

    class Unversioned(_VersionedS3):
        def get_bucket_versioning(self, **kwargs: object) -> dict[str, str]:
            return {"Status": "Suspended"}

    client = Unversioned()
    with pytest.raises(CheckpointStagingError, match="versioning"):
        stage_checkpoint(checkpoint, bucket="artifacts", revision=REVISION, s3_client=client)
    assert client.put_calls == []
