from __future__ import annotations

import hashlib
import io
import json
import struct
from pathlib import Path
from typing import Any

import pytest

from scripts import start_backend
from scripts.bootstrap_objective_checkpoint import (
    ObjectiveCheckpointBootstrapError,
    bootstrap_checkpoint,
)
from scripts.stage_functiongemma_checkpoint import build_deterministic_bundle


def _snapshot(path: Path) -> Path:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"architectures": ["Gemma3ForCausalLM"], "model_type": "gemma3_text"}),
        encoding="utf-8",
    )
    (path / "tokenizer.json").write_text('{"version":1}', encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    (path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"\0" * 4
    )
    return path


class _S3:
    def __init__(self, body: bytes, *, version_id: str = "base-v1") -> None:
        self.body = body
        self.version_id = version_id
        self.calls: list[dict[str, str]] = []

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"Body": io.BytesIO(self.body), "VersionId": self.version_id}


def test_bootstrap_downloads_exact_s3_version_and_attests_extracted_snapshot(
    tmp_path: Path,
) -> None:
    revision = "a" * 40
    bundle = build_deterministic_bundle(_snapshot(tmp_path / "source"), revision=revision)
    s3 = _S3(bundle.data)
    environment = {
        "OBJECTIVE_MODEL_CHECKPOINT_DIR": str(tmp_path / "models" / "functiongemma"),
        "OBJECTIVE_MODEL_REVISION": revision,
        "OBJECTIVE_BASE_MODEL_URI": (
            "s3://artifact-bucket/post-training/checkpoints/base.tar.gz?versionId=base-v1"
        ),
        "OBJECTIVE_BASE_MODEL_SHA256": hashlib.sha256(bundle.data).hexdigest(),
    }

    snapshot_digest = bootstrap_checkpoint(environment, s3_client=s3)

    assert s3.calls == [
        {
            "Bucket": "artifact-bucket",
            "Key": "post-training/checkpoints/base.tar.gz",
            "VersionId": "base-v1",
        }
    ]
    installed = Path(environment["OBJECTIVE_MODEL_CHECKPOINT_DIR"])
    assert (installed / "config.json").is_file()
    assert environment["OBJECTIVE_MODEL_SHA256"] == snapshot_digest
    assert len(snapshot_digest) == 64


def test_bootstrap_rejects_bundle_digest_mismatch_without_installing(tmp_path: Path) -> None:
    revision = "b" * 40
    bundle = build_deterministic_bundle(_snapshot(tmp_path / "source"), revision=revision)
    s3 = _S3(bundle.data + b"tamper")
    target = tmp_path / "models" / "functiongemma"
    environment = {
        "OBJECTIVE_MODEL_CHECKPOINT_DIR": str(target),
        "OBJECTIVE_MODEL_REVISION": revision,
        "OBJECTIVE_BASE_MODEL_URI": "s3://artifact-bucket/base.tar.gz?versionId=base-v1",
        "OBJECTIVE_BASE_MODEL_SHA256": hashlib.sha256(bundle.data).hexdigest(),
    }

    with pytest.raises(ObjectiveCheckpointBootstrapError, match="bundle SHA-256"):
        bootstrap_checkpoint(environment, s3_client=s3)

    assert not target.exists()


def test_bootstrap_rejects_unversioned_base_checkpoint_uri_before_s3_read(tmp_path: Path) -> None:
    s3 = _S3(b"unused")
    environment = {
        "OBJECTIVE_MODEL_CHECKPOINT_DIR": str(tmp_path / "model"),
        "OBJECTIVE_MODEL_REVISION": "c" * 40,
        "OBJECTIVE_BASE_MODEL_URI": "s3://artifact-bucket/base.tar.gz",
        "OBJECTIVE_BASE_MODEL_SHA256": "d" * 64,
    }

    with pytest.raises(ObjectiveCheckpointBootstrapError, match="versionId"):
        bootstrap_checkpoint(environment, s3_client=s3)

    assert s3.calls == []


def test_objective_container_bootstraps_before_execing_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExecCalled(Exception):
        pass

    events: list[tuple[str, list[str] | None]] = []

    def fake_execvp(executable: str, args: list[str]) -> None:
        events.append((executable, list(args)))
        raise ExecCalled

    monkeypatch.setenv("SERVICE_ROLE", "objective")
    monkeypatch.setenv("PORT", "8123")
    monkeypatch.setattr(
        start_backend,
        "bootstrap_checkpoint",
        lambda: events.append(("bootstrap", None)),
    )
    monkeypatch.setattr(
        start_backend,
        "execvp",
        fake_execvp,
    )

    with pytest.raises(ExecCalled):
        start_backend.main()

    assert events == [
        ("bootstrap", None),
        (
            "uvicorn",
            ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8123"],
        ),
    ]


def test_checkpoint_bootstrap_failure_never_starts_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SERVICE_ROLE", "objective")
    monkeypatch.setattr(
        start_backend,
        "bootstrap_checkpoint",
        lambda: (_ for _ in ()).throw(ObjectiveCheckpointBootstrapError("blocked")),
    )
    monkeypatch.setattr(
        start_backend,
        "execvp",
        lambda *_args: pytest.fail("Uvicorn must not start without the model"),
    )

    with pytest.raises(ObjectiveCheckpointBootstrapError, match="blocked"):
        start_backend.main()


def test_coordinator_startup_skips_objective_checkpoint_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExecCalled(Exception):
        pass

    monkeypatch.setenv("SERVICE_ROLE", "coordinator")
    monkeypatch.setattr(
        start_backend,
        "bootstrap_checkpoint",
        lambda: pytest.fail("coordinator must not download the objective model"),
    )

    def fake_execvp(executable: str, args: list[str]) -> None:
        assert executable == "uvicorn"
        assert args[1] == "app.main:app"
        raise ExecCalled

    monkeypatch.setattr(start_backend, "execvp", fake_execvp)
    with pytest.raises(ExecCalled):
        start_backend.main()
