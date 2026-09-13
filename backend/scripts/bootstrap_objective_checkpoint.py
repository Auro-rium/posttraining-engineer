"""Materialize the exact approved FunctionGemma base bundle before Uvicorn."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
from collections.abc import Mapping, MutableMapping
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qs, urlparse

from scripts.stage_functiongemma_checkpoint import (
    TARGET_MODEL_ID,
    CheckpointStagingError,
    validate_checkpoint_directory,
)

_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ObjectiveCheckpointBootstrapError(RuntimeError):
    """The objective base checkpoint was not safely materialized."""


def _s3_version(value: str) -> tuple[str, str, str]:
    parsed = urlparse(value)
    versions = parse_qs(parsed.query, keep_blank_values=True).get("versionId", [])
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.strip("/")
        or len(versions) != 1
        or not versions[0].strip()
        or versions[0].strip().lower() == "null"
        or parsed.fragment
    ):
        raise ObjectiveCheckpointBootstrapError(
            "OBJECTIVE_BASE_MODEL_URI must pin an S3 object version with versionId"
        )
    return parsed.netloc, parsed.path.lstrip("/"), versions[0].strip()


def _extract_snapshot(archive_path: Path, destination: Path) -> None:
    destination.mkdir(mode=0o700)
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                name = PurePosixPath(member.name)
                if (
                    name.is_absolute()
                    or not name.parts
                    or any(part in {"", ".", ".."} for part in name.parts)
                    or not member.isfile()
                ):
                    raise ObjectiveCheckpointBootstrapError(
                        "checkpoint bundle contains an unsafe or non-regular file"
                    )
                output = destination.joinpath(*name.parts)
                output.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ObjectiveCheckpointBootstrapError(
                        "checkpoint bundle contains an unreadable file"
                    )
                with source, output.open("xb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                output.chmod(0o600)
    except ObjectiveCheckpointBootstrapError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise ObjectiveCheckpointBootstrapError(
            "staged checkpoint bundle is not a readable gzip tar archive"
        ) from exc


def bootstrap_checkpoint(
    environment: MutableMapping[str, str] | Mapping[str, str] | None = None,
    *,
    s3_client: Any | None = None,
) -> str:
    """Download one immutable bundle, validate it, then expose its snapshot digest.

    ``OBJECTIVE_BASE_MODEL_SHA256`` is the exact S3 bundle checksum. The
    existing ``OBJECTIVE_MODEL_SHA256`` remains the validator's digest over the
    extracted snapshot's sorted file identities and is populated after the
    bundle has been checked.
    """

    values = os.environ if environment is None else environment
    checkpoint_dir = values.get("OBJECTIVE_MODEL_CHECKPOINT_DIR", "").strip()
    revision = values.get("OBJECTIVE_MODEL_REVISION", "").strip()
    model_uri = values.get("OBJECTIVE_BASE_MODEL_URI", "").strip()
    bundle_sha256 = values.get("OBJECTIVE_BASE_MODEL_SHA256", "").strip()
    if not checkpoint_dir or not revision or not model_uri or not bundle_sha256:
        raise ObjectiveCheckpointBootstrapError(
            "objective checkpoint directory, revision, immutable S3 URI, and bundle SHA-256 "
            "are required"
        )
    if not _REVISION_RE.fullmatch(revision):
        raise ObjectiveCheckpointBootstrapError(
            "OBJECTIVE_MODEL_REVISION must be an immutable 40-character commit SHA"
        )
    if not _SHA256_RE.fullmatch(bundle_sha256):
        raise ObjectiveCheckpointBootstrapError(
            "OBJECTIVE_BASE_MODEL_SHA256 must be a lowercase SHA-256 digest"
        )
    bucket, key, version_id = _s3_version(model_uri)
    target = Path(checkpoint_dir).expanduser()
    if target.is_symlink():
        raise ObjectiveCheckpointBootstrapError(
            "objective checkpoint destination must not be a symlink"
        )
    if target.exists():
        raise ObjectiveCheckpointBootstrapError(
            "objective checkpoint destination already exists; refusing to replace it"
        )
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise ObjectiveCheckpointBootstrapError(
            "objective checkpoint parent directory must not be a symlink"
        )
    if s3_client is None:
        try:
            import boto3  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - dependency is present in deployment
            raise ObjectiveCheckpointBootstrapError(
                "boto3 is required to read the staged checkpoint"
            ) from exc
        s3_client = boto3.client("s3", region_name=values.get("AWS_REGION"))

    try:
        response = s3_client.get_object(Bucket=bucket, Key=key, VersionId=version_id)
        if response.get("VersionId") != version_id:
            raise ObjectiveCheckpointBootstrapError(
                "S3 did not return the requested checkpoint object version"
            )
        body = response.get("Body")
        if body is None or not callable(getattr(body, "read", None)):
            raise ObjectiveCheckpointBootstrapError("S3 checkpoint object body is unreadable")
    except ObjectiveCheckpointBootstrapError:
        raise
    except Exception as exc:
        raise ObjectiveCheckpointBootstrapError(
            "could not read the configured checkpoint object version from S3"
        ) from exc

    with tempfile.TemporaryDirectory(prefix="objective-checkpoint-", dir=parent) as temporary:
        scratch = Path(temporary)
        archive_path = scratch / "checkpoint.tar.gz"
        digest = hashlib.sha256()
        try:
            with body, archive_path.open("xb") as archive_file:
                while chunk := body.read(1024 * 1024):
                    digest.update(chunk)
                    archive_file.write(chunk)
        except Exception as exc:
            raise ObjectiveCheckpointBootstrapError(
                "could not stream the immutable checkpoint bundle from S3"
            ) from exc
        if digest.hexdigest() != bundle_sha256:
            raise ObjectiveCheckpointBootstrapError(
                "downloaded checkpoint bundle SHA-256 does not match the pinned digest"
            )

        staged_snapshot = scratch / "snapshot"
        _extract_snapshot(archive_path, staged_snapshot)
        try:
            files = validate_checkpoint_directory(
                staged_snapshot,
                revision=revision,
                model_id=TARGET_MODEL_ID,
            )
        except (OSError, CheckpointStagingError) as exc:
            raise ObjectiveCheckpointBootstrapError(
                "staged checkpoint does not match the pinned FunctionGemma revision"
            ) from exc
        snapshot_digest = hashlib.sha256(
            json.dumps(
                [
                    {"path": item.path, "sha256": item.sha256, "size_bytes": item.size_bytes}
                    for item in sorted(files, key=lambda item: item.path)
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        configured_snapshot_sha256 = values.get("OBJECTIVE_MODEL_SHA256", "").strip()
        if configured_snapshot_sha256 and configured_snapshot_sha256 != snapshot_digest:
            raise ObjectiveCheckpointBootstrapError(
                "extracted checkpoint snapshot SHA-256 does not match its configured digest"
            )
        try:
            os.replace(staged_snapshot, target)
        except OSError as exc:
            raise ObjectiveCheckpointBootstrapError(
                "validated FunctionGemma checkpoint could not be installed"
            ) from exc

    if isinstance(values, MutableMapping):
        values["OBJECTIVE_MODEL_SHA256"] = snapshot_digest
    return snapshot_digest


def main() -> int:
    try:
        bootstrap_checkpoint()
    except ObjectiveCheckpointBootstrapError as exc:
        raise SystemExit(f"objective checkpoint bootstrap blocked: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
