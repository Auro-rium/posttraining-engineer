from __future__ import annotations

import base64
import builtins
import hashlib
import io
import json
import struct
from pathlib import Path

import pytest

import scripts.stage_functiongemma_checkpoint as staging
from scripts.stage_functiongemma_checkpoint import (
    TARGET_MODEL_ID,
    CheckpointStagingError,
    build_deterministic_bundle,
    main,
    stage_checkpoint,
    validate_checkpoint_directory,
)

REVISION = "a" * 40


def _checkpoint(path: Path, *, include_weights: bool = True) -> Path:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {"architectures": ["Gemma3ForCausalLM"], "model_type": "gemma3_text"},
            sort_keys=True,
        ),
    )
    (path / "tokenizer.json").write_text("{\"version\": 1}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    if include_weights:
        (path / "model.safetensors").write_bytes(_safetensors("weight"))
    return path


def _safetensors(*tensor_names: str) -> bytes:
    header = {
        name: {"dtype": "F32", "shape": [1], "data_offsets": [index * 4, (index + 1) * 4]}
        for index, name in enumerate(tensor_names)
    }
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded + b"\x00" * (4 * len(tensor_names))


def test_revision_is_required_to_be_an_immutable_commit(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")

    with pytest.raises(CheckpointStagingError, match=r"immutable.*40-character"):
        validate_checkpoint_directory(checkpoint, revision="main")


def test_only_functiongemma_target_model_can_be_staged(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")

    with pytest.raises(CheckpointStagingError, match=r"only.*target model"):
        validate_checkpoint_directory(
            checkpoint,
            revision=REVISION,
            model_id="google/gemma-3-4b-it",
        )


@pytest.mark.parametrize(
    "config",
    [
        {"architectures": ["Gemma3ForCausalLM"], "model_type": "other"},
        {"architectures": ["OtherForCausalLM"], "model_type": "gemma3_text"},
        {"model_type": "gemma3_text"},
    ],
)
def test_exact_functiongemma_config_identity_is_required(
    tmp_path: Path, config: dict[str, object]
) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(CheckpointStagingError, match=r"FunctionGemma|architecture|model_type"):
        validate_checkpoint_directory(checkpoint, revision=REVISION)


def test_malformed_safetensors_weight_is_rejected(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    (checkpoint / "model.safetensors").write_bytes(b"not-a-safetensors-file")

    with pytest.raises(CheckpointStagingError, match="safetensors"):
        validate_checkpoint_directory(checkpoint, revision=REVISION)


def test_pytorch_weight_format_is_rejected_closed(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint", include_weights=False)
    (checkpoint / "pytorch_model.bin").write_bytes(b"not-a-pytorch-archive")

    with pytest.raises(CheckpointStagingError, match="PyTorch"):
        validate_checkpoint_directory(checkpoint, revision=REVISION)


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


@pytest.mark.parametrize(
    ("filename", "flag", "value"),
    [
        ("config.json", "private", "true"),
        ("tokenizer.json", "access_restricted", "1"),
        ("tokenizer_config.json", "gated", "yes"),
        ("metadata.json", "private", "on"),
    ],
)
def test_truthy_restricted_json_flags_are_rejected(
    tmp_path: Path, filename: str, flag: str, value: str
) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    (checkpoint / filename).write_text(json.dumps({flag: value}), encoding="utf-8")

    with pytest.raises(CheckpointStagingError, match="gated/private/restricted"):
        validate_checkpoint_directory(checkpoint, revision=REVISION)


def test_nested_cache_refs_locks_and_symlink_roots_are_rejected(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    (checkpoint / "cache" / "refs").mkdir(parents=True)
    (checkpoint / "cache" / "refs" / "main").write_text(REVISION, encoding="utf-8")
    with pytest.raises(CheckpointStagingError, match="mutable cache"):
        validate_checkpoint_directory(checkpoint, revision=REVISION)

    (checkpoint / "cache").rename(checkpoint / "cache-removed")
    (checkpoint / ".locks").mkdir()
    with pytest.raises(CheckpointStagingError, match="lock"):
        validate_checkpoint_directory(checkpoint, revision=REVISION)

    symlink_root = tmp_path / "checkpoint-link"
    symlink_root.symlink_to(checkpoint, target_is_directory=True)
    with pytest.raises(CheckpointStagingError, match=r"root.*symlink"):
        validate_checkpoint_directory(symlink_root, revision=REVISION)

    descendant = _checkpoint(tmp_path / "descendant")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (descendant / "linked.bin").symlink_to(outside)
    with pytest.raises(CheckpointStagingError, match="unsupported symlink"):
        validate_checkpoint_directory(descendant, revision=REVISION)


@pytest.mark.parametrize("filename", ["config.json", "tokenizer.json", "tokenizer_config.json"])
def test_required_json_files_must_be_objects(tmp_path: Path, filename: str) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    (checkpoint / filename).write_text("not-json", encoding="utf-8")

    with pytest.raises(CheckpointStagingError, match="JSON"):
        validate_checkpoint_directory(checkpoint, revision=REVISION)


@pytest.mark.parametrize("mapped_value", ["tokenizer.json", 17])
def test_indexed_weights_require_supported_shard_map(
    tmp_path: Path, mapped_value: object
) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    (checkpoint / "model.safetensors").unlink()
    (checkpoint / "model-00001-of-00001.safetensors").write_bytes(b"shard")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": mapped_value}}), encoding="utf-8"
    )

    with pytest.raises(
        CheckpointStagingError, match=r"(invalid shard name|non-string shard)"
    ):
        validate_checkpoint_directory(checkpoint, revision=REVISION)


def _indexed_checkpoint(path: Path) -> Path:
    checkpoint = _checkpoint(path)
    (checkpoint / "model.safetensors").unlink()
    (checkpoint / "model-00001-of-00002.safetensors").write_bytes(_safetensors("layer.0"))
    (checkpoint / "model-00002-of-00002.safetensors").write_bytes(_safetensors("layer.1"))
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    return checkpoint


def test_valid_indexed_shards_are_accepted(tmp_path: Path) -> None:
    checkpoint = _indexed_checkpoint(tmp_path / "checkpoint")

    files = validate_checkpoint_directory(checkpoint, revision=REVISION)

    assert {item.path for item in files} >= {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
    }


def test_indexed_shards_missing_from_disk_are_rejected(tmp_path: Path) -> None:
    checkpoint = _indexed_checkpoint(tmp_path / "checkpoint")
    (checkpoint / "model-00002-of-00002.safetensors").unlink()

    with pytest.raises(CheckpointStagingError, match="missing weight shard"):
        validate_checkpoint_directory(checkpoint, revision=REVISION)


def test_sharded_safetensors_require_an_index(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    (checkpoint / "model.safetensors").unlink()
    (checkpoint / "model-00001-of-00002.safetensors").write_bytes(b"shard-1")
    (checkpoint / "model-00002-of-00002.safetensors").write_bytes(b"shard-2")

    with pytest.raises(CheckpointStagingError, match="weight index"):
        validate_checkpoint_directory(checkpoint, revision=REVISION)


def test_indexed_safetensors_require_a_complete_numbered_set(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    (checkpoint / "model.safetensors").unlink()
    (checkpoint / "model-00001-of-00003.safetensors").write_bytes(_safetensors("layer.0"))
    (checkpoint / "model-00003-of-00003.safetensors").write_bytes(_safetensors("layer.2"))
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": "model-00001-of-00003.safetensors",
                    "layer.2": "model-00003-of-00003.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CheckpointStagingError, match=r"complete|missing"):
        validate_checkpoint_directory(checkpoint, revision=REVISION)


def test_index_tensor_map_must_match_safetensors_headers(tmp_path: Path) -> None:
    checkpoint = _indexed_checkpoint(tmp_path / "checkpoint")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "wrong": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CheckpointStagingError, match=r"tensor|header"):
        validate_checkpoint_directory(checkpoint, revision=REVISION)


def test_bundle_uses_the_same_bytes_as_validation_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    original = staging._read_regular_file
    original_weight = (checkpoint / "model.safetensors").read_bytes()
    changed_weight = _safetensors("changed")

    def mutate_after_read(path: Path) -> bytes:
        data = original(path)
        if path.name == "model.safetensors":
            path.write_bytes(changed_weight)
        return data

    monkeypatch.setattr(staging, "_read_regular_file", mutate_after_read)
    bundle = build_deterministic_bundle(checkpoint, revision=REVISION)

    manifest_weight = next(item for item in bundle.files if item.path == "model.safetensors")
    assert manifest_weight.sha256 == hashlib.sha256(original_weight).hexdigest()
    with staging.tarfile.open(
        fileobj=io.BytesIO(staging.gzip.decompress(bundle.data)), mode="r:"
    ) as archive:
        member = archive.extractfile("model.safetensors")
        assert member is not None
        assert member.read() == original_weight


@pytest.mark.parametrize("filename", ["metadata.json", "report.json", "manifest.json"])
def test_optional_hf_metadata_and_reports_must_be_object_json(
    tmp_path: Path, filename: str
) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    (checkpoint / filename).write_text("[]", encoding="utf-8")

    with pytest.raises(CheckpointStagingError, match=r"JSON.*object"):
        validate_checkpoint_directory(checkpoint, revision=REVISION)


def test_unavailable_hf_revision_metadata_is_rejected(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    (checkpoint / "metadata.json").write_text(
        json.dumps({"sha": REVISION, "disabled": True}), encoding="utf-8"
    )

    with pytest.raises(CheckpointStagingError, match=r"unavailable|restricted"):
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

    def get_bucket_encryption(self, **kwargs: object) -> dict[str, object]:
        assert kwargs == {"Bucket": "artifacts"}
        return {
            "ServerSideEncryptionConfiguration": {
                "Rules": [
                    {"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}
                ]
            }
        }

    def put_object(self, **kwargs: object) -> dict[str, object]:
        self.put_calls.append(kwargs)
        body = kwargs["Body"]
        assert isinstance(body, bytes)
        return {
            "VersionId": "version-17",
            "ChecksumSHA256": base64.b64encode(hashlib.sha256(body).digest()).decode("ascii"),
            "ETag": '"etag"',
        }


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
    assert metadata["s3-encryption"] == "AES256"
    assert metadata["checksum-sha256"] == staged.sha256
    assert "ChecksumSHA256" in call
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


def test_staging_rejects_missing_bucket_encryption_without_upload(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")

    class Unencrypted(_VersionedS3):
        def get_bucket_encryption(self, **kwargs: object) -> dict[str, object]:
            return {}

    client = Unencrypted()
    with pytest.raises(CheckpointStagingError, match="encryption"):
        stage_checkpoint(checkpoint, bucket="artifacts", revision=REVISION, s3_client=client)
    assert client.put_calls == []


def test_staging_rejects_missing_checksum_provenance(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")

    class NoChecksum(_VersionedS3):
        def put_object(self, **kwargs: object) -> dict[str, object]:
            self.put_calls.append(kwargs)
            return {"VersionId": "version-17"}

    with pytest.raises(CheckpointStagingError, match="checksum"):
        stage_checkpoint(
            checkpoint,
            bucket="artifacts",
            revision=REVISION,
            s3_client=NoChecksum(),
        )


def test_staging_rejects_mismatched_checksum_provenance(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")

    class WrongChecksum(_VersionedS3):
        def put_object(self, **kwargs: object) -> dict[str, object]:
            self.put_calls.append(kwargs)
            return {"VersionId": "version-17", "ChecksumSHA256": "wrong"}

    with pytest.raises(CheckpointStagingError, match="matching checksum"):
        stage_checkpoint(
            checkpoint,
            bucket="artifacts",
            revision=REVISION,
            s3_client=WrongChecksum(),
        )


@pytest.mark.parametrize("version_id", [None, "", "null", "NULL", 17])
def test_staging_rejects_invalid_s3_version_ids(tmp_path: Path, version_id: object) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")

    class InvalidVersion(_VersionedS3):
        def put_object(self, **kwargs: object) -> dict[str, object]:
            self.put_calls.append(kwargs)
            return {"VersionId": version_id}

    with pytest.raises(CheckpointStagingError, match="version id"):
        stage_checkpoint(
            checkpoint,
            bucket="artifacts",
            revision=REVISION,
            s3_client=InvalidVersion(),
        )


def test_dry_run_does_not_import_or_construct_boto3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    real_import = builtins.__import__

    def reject_boto3(name: str, *args: object, **kwargs: object) -> object:
        if name == "boto3":
            raise AssertionError("dry-run attempted to import boto3")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_boto3)
    assert (
        main(
            [
                "--checkpoint-dir",
                str(checkpoint),
                "--revision",
                REVISION,
                "--dry-run",
            ]
        )
        == 0
    )
