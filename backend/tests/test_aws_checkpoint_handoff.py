from __future__ import annotations

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
        revision="a" * 40,
        prefix="post-training/checkpoints",
        region="us-east-1",
        token="secret",
        snapshot_download=download,
    )

    assert result["status"] == "STAGED"
    assert calls["download"] == {
        "repo_id": "google/functiongemma-270m-it",
        "revision": "a" * 40,
        "local_dir": calls["stage"]["directory"],  # type: ignore[index]
        "token": "secret",
    }


def test_handoff_requires_token() -> None:
    with pytest.raises(CheckpointHandoffError, match="HF_TOKEN"):
        stage_from_huggingface(
            bucket="bucket",
            revision="a" * 40,
            prefix="post-training/checkpoints",
            region=None,
            token=None,
        )
