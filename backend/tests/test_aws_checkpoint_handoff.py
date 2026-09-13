from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from aws_checkpoint_handoff import (  # noqa: E402
    CheckpointHandoffError,
    stage_from_huggingface,
)

REVISION = "a" * 40


def test_handoff_stages_snapshot_without_huggingface_local_metadata(
    tmp_path: Path,
) -> None:
    class _VersionedEncryptedS3:
        uploaded: dict[str, object] | None = None

        def get_bucket_versioning(self, **_: object) -> dict[str, str]:
            return {"Status": "Enabled"}

        def get_bucket_encryption(self, **_: object) -> dict[str, object]:
            return {
                "ServerSideEncryptionConfiguration": {
                    "Rules": [
                        {
                            "ApplyServerSideEncryptionByDefault": {
                                "SSEAlgorithm": "aws:kms"
                            }
                        }
                    ]
                }
            }

        def put_object(self, **kwargs: object) -> dict[str, str]:
            self.uploaded = kwargs
            return {
                "VersionId": "version-123",
                "ChecksumSHA256": str(kwargs["ChecksumSHA256"]),
            }

    def download(**kwargs: object) -> str:
        checkpoint = Path(str(kwargs["local_dir"]))
        (checkpoint / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["Gemma3ForCausalLM"],
                    "model_type": "gemma3_text",
                }
            ),
            encoding="utf-8",
        )
        (checkpoint / "tokenizer.json").write_text('{"version":1}', encoding="utf-8")
        (checkpoint / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        header = json.dumps(
            {
                "weight": {
                    "data_offsets": [0, 4],
                    "dtype": "F32",
                    "shape": [1],
                }
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        (checkpoint / "model.safetensors").write_bytes(
            struct.pack("<Q", len(header)) + header + b"\x00" * 4
        )
        metadata = checkpoint / ".cache" / "huggingface" / "download"
        metadata.mkdir(parents=True)
        (metadata.parent / ".gitignore").write_text("*", encoding="utf-8")
        (metadata / "config.json.metadata").write_text("hub metadata", encoding="utf-8")
        (metadata / "config.json.lock").write_text("", encoding="utf-8")
        nested_metadata = metadata / "nested" / "tokenizer.json.metadata"
        nested_metadata.parent.mkdir()
        nested_metadata.write_text("nested hub metadata", encoding="utf-8")
        (nested_metadata.parent / "tokenizer.json.lock").write_text("", encoding="utf-8")
        return str(checkpoint)

    s3_client = _VersionedEncryptedS3()
    result = stage_from_huggingface(
        bucket="artifact-bucket",
        revision=REVISION,
        prefix="post-training/checkpoints",
        region="us-east-1",
        token="test-token",
        snapshot_download=download,
        s3_client=s3_client,
    )

    assert result["status"] == "STAGED"
    assert result["version_ref"] == (
        "s3://artifact-bucket/post-training/checkpoints/functiongemma-270m-it/"
        f"{REVISION}/{result['sha256']}.tar.gz?versionId=version-123"
    )
    assert result["file_count"] == 4
    assert s3_client.uploaded is not None


def test_handoff_fails_closed_on_unexpected_local_cache_content() -> None:
    def download(**kwargs: object) -> str:
        cache = Path(str(kwargs["local_dir"])) / ".cache"
        cache.mkdir()
        (cache / "untrusted-content").write_text("not Hub metadata", encoding="utf-8")
        return str(kwargs["local_dir"])

    with pytest.raises(CheckpointHandoffError, match="unexpected Hugging Face local metadata"):
        stage_from_huggingface(
            bucket="artifact-bucket",
            revision=REVISION,
            prefix="post-training/checkpoints",
            region="us-east-1",
            token="test-token",
            snapshot_download=download,
            s3_client=object(),
        )


def test_handoff_downloads_explicit_revision_then_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def download(**kwargs: object) -> str:
        calls["download"] = kwargs
        return str(kwargs["local_dir"])

    class _Staged:
        def to_dict(self) -> dict[str, object]:
            return {"status": "STAGED", "version_ref": "s3://bucket/key?versionId=v1"}

    def stage(directory: str, **kwargs: object) -> _Staged:
        calls["stage"] = {"directory": directory, **kwargs}
        return _Staged()

    monkeypatch.setattr("aws_checkpoint_handoff.stage_checkpoint", stage)
    result = stage_from_huggingface(
        bucket="bucket",
        revision=REVISION,
        prefix="post-training/checkpoints",
        region="us-east-1",
        token="secret",
        snapshot_download=download,
    )

    assert result["status"] == "STAGED"
    assert calls["download"] == {
        "repo_id": "google/functiongemma-270m-it",
        "revision": REVISION,
        "local_dir": calls["stage"]["directory"],  # type: ignore[index]
        "token": "secret",
    }


def test_handoff_requires_token() -> None:
    with pytest.raises(CheckpointHandoffError, match="HF_TOKEN"):
        stage_from_huggingface(
            bucket="bucket",
            revision=REVISION,
            prefix="post-training/checkpoints",
            region=None,
            token=None,
        )
