"""Validate and stage an immutable FunctionGemma checkpoint bundle.

The command only accepts an already materialized local checkpoint.  It never
downloads from Hugging Face and it refuses snapshots with mutable refs, gate
markers, partial files, or cache locks.  The uploaded object is a deterministic
gzip tarball whose key contains its SHA-256 digest.  S3 bucket versioning is
checked before the first mutating call and the returned version id is part of
the result that callers should persist.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

TARGET_MODEL_ID = "google/functiongemma-270m-it"
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WEIGHT_NAMES = (
    "model.safetensors",
    "pytorch_model.bin",
)
_GATE_MARKERS = frozenset(
    {".gated", "gated", "gated.json", "access_request.json", "access_denied"}
)
_RESTRICTED_FLAGS = frozenset(
    {"gated", "is_gated", "private", "is_private", "access_restricted", "access_denied"}
)


class CheckpointStagingError(ValueError):
    """Raised when a checkpoint cannot be proven safe and complete to stage."""


@dataclass(frozen=True, slots=True)
class CheckpointFile:
    """Digest and size of one regular checkpoint file."""

    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class DeterministicBundle:
    """A deterministic checkpoint archive and its content identity."""

    data: bytes
    sha256: str
    model_id: str
    revision: str
    files: tuple[CheckpointFile, ...]

    @property
    def size_bytes(self) -> int:
        return len(self.data)

    @property
    def bundle_sha256(self) -> str:
        return self.sha256

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": "functiongemma-checkpoint-v1",
            "model_id": self.model_id,
            "hf_revision": self.revision,
            "bundle_sha256": self.sha256,
            "bundle_size_bytes": self.size_bytes,
            "files": [item.to_dict() for item in self.files],
        }


@dataclass(frozen=True, slots=True)
class StagedCheckpoint:
    """Immutable S3 identity returned after a successful staging upload."""

    model_id: str
    revision: str
    sha256: str
    size_bytes: int
    file_count: int
    bucket: str
    key: str
    version_id: str
    metadata: dict[str, str]

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"

    @property
    def version_ref(self) -> str:
        return f"{self.uri}?versionId={self.version_id}"

    @property
    def bundle_sha256(self) -> str:
        return self.sha256

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "STAGED",
            "model_id": self.model_id,
            "hf_revision": self.revision,
            "sha256": self.sha256,
            "bundle_sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "file_count": self.file_count,
            "bucket": self.bucket,
            "key": self.key,
            "uri": self.uri,
            "version_id": self.version_id,
            "version_ref": self.version_ref,
            "metadata": dict(self.metadata),
        }


def validate_immutable_revision(revision: str) -> str:
    """Return a canonical revision or reject mutable/non-SHA references."""

    if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision):
        raise CheckpointStagingError(
            "hf revision must be an immutable lowercase 40-character commit SHA"
        )
    return revision


def _relative_files(root: Path) -> list[Path]:
    if root.is_symlink():
        raise CheckpointStagingError(f"checkpoint root must not be a symlink: {root}")
    if not root.exists() or not root.is_dir():
        raise CheckpointStagingError(f"checkpoint directory does not exist: {root}")
    paths: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        parts = tuple(part.lower() for part in relative.parts)
        name = path.name.lower()
        if path.is_symlink():
            raise CheckpointStagingError(f"checkpoint contains unsupported symlink: {relative}")
        if name in _GATE_MARKERS or "gated" in name:
            raise CheckpointStagingError(f"checkpoint is gated: {relative}")
        if (
            name.endswith((".lock", ".incomplete", ".part"))
            or name.startswith("lock")
            or "incomplete" in name
        ):
            raise CheckpointStagingError(
                f"checkpoint contains cache lock/partial input: {relative}"
            )
        if "refs" in parts:
            raise CheckpointStagingError(f"checkpoint contains mutable cache reference: {relative}")
        if any(part in {".locks", "locks"} or part.startswith(".lock") for part in parts):
            raise CheckpointStagingError(f"checkpoint contains cache lock directory: {relative}")
        if path.is_file():
            paths.append(path)
        elif not path.is_dir():
            raise CheckpointStagingError(f"checkpoint contains unsupported file type: {relative}")
    return paths


def _truthy_restricted_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value) if isinstance(value, (int, float)) else False


def _contains_restricted_flag(value: object) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = str(key).strip().lower().replace("-", "_")
            if normalized_key in _RESTRICTED_FLAGS and _truthy_restricted_flag(item):
                return True
            if _contains_restricted_flag(item):
                return True
    elif isinstance(value, list):
        return any(_contains_restricted_flag(item) for item in value)
    return False


def _reject_gated_metadata(paths: list[Path], root: Path) -> None:
    for path in paths:
        if path.suffix.lower() != ".json":
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CheckpointStagingError(
                f"invalid checkpoint JSON metadata: {path.relative_to(root)}"
            ) from exc
        if _contains_restricted_flag(value):
            raise CheckpointStagingError(
                f"checkpoint metadata is gated/private/restricted: {path.relative_to(root)}"
            )


def _validate_required_files(paths: list[Path], root: Path) -> None:
    names = {path.relative_to(root).as_posix() for path in paths}
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    missing = sorted(required - names)
    if missing:
        raise CheckpointStagingError(
            "checkpoint is incomplete; missing required file(s): " + ", ".join(missing)
        )

    for filename in sorted(required):
        path = root / filename
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CheckpointStagingError(
                f"required checkpoint JSON is invalid: {filename}"
            ) from exc
        if not isinstance(value, dict):
            raise CheckpointStagingError(f"required checkpoint JSON must be an object: {filename}")

    weights = [
        path
        for path in paths
        if path.name in _WEIGHT_NAMES
        or (path.name.startswith("model-") and path.suffix == ".safetensors")
        or (path.name.startswith("pytorch_model-") and path.suffix == ".bin")
    ]
    if not weights:
        raise CheckpointStagingError(
            "checkpoint is incomplete; missing model weight file "
            "(model.safetensors or pytorch_model.bin)"
        )
    if any(path.stat().st_size == 0 for path in weights):
        raise CheckpointStagingError("checkpoint is incomplete; model weight file is empty")

    index_paths = [
        path
        for path in paths
        if path.name in {"model.safetensors.index.json", "pytorch_model.bin.index.json"}
    ]
    for index_path in index_paths:
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CheckpointStagingError(
                f"invalid weight index: {index_path.relative_to(root)}"
            ) from exc
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise CheckpointStagingError(
                f"weight index has no weight_map: {index_path.relative_to(root)}"
            )
        expected_suffix = ".safetensors" if index_path.name.startswith("model.") else ".bin"
        mapped_shards: set[str] = set()
        for value in weight_map.values():
            if not isinstance(value, str) or not value:
                raise CheckpointStagingError(
                    f"weight index contains a non-string shard: {index_path.relative_to(root)}"
                )
            shard = PurePosixPath(value)
            if (
                shard.name != value
                or not value.endswith(expected_suffix)
                or not (
                    value.startswith("model-")
                    if expected_suffix == ".safetensors"
                    else value.startswith("pytorch_model-")
                )
            ):
                raise CheckpointStagingError(
                    f"weight index contains an invalid shard name: {value}"
                )
            mapped_shards.add(value)
        available_shards = {
            path.name
            for path in paths
            if (
                path.name.startswith("model-")
                and path.name.endswith(".safetensors")
                if expected_suffix == ".safetensors"
                else path.name.startswith("pytorch_model-")
                and path.name.endswith(".bin")
            )
        }
        missing_shards = sorted(mapped_shards - names)
        if missing_shards:
            raise CheckpointStagingError(
                "checkpoint is incomplete; missing weight shard(s): " + ", ".join(missing_shards)
            )
        if available_shards != mapped_shards:
            raise CheckpointStagingError(
                "weight index does not match available model shards: "
                f"expected {sorted(mapped_shards)}, found {sorted(available_shards)}"
            )


def validate_checkpoint_directory(
    checkpoint_dir: str | Path,
    *,
    revision: str,
    model_id: str = TARGET_MODEL_ID,
) -> tuple[CheckpointFile, ...]:
    """Validate a local checkpoint and return sorted per-file digests."""

    if model_id != TARGET_MODEL_ID:
        raise CheckpointStagingError(
            f"only the pinned target model {TARGET_MODEL_ID!r} may be staged"
        )
    validate_immutable_revision(revision)
    root = Path(checkpoint_dir).expanduser()
    paths = _relative_files(root)
    _reject_gated_metadata(paths, root)
    _validate_required_files(paths, root)
    if not paths:
        raise CheckpointStagingError("checkpoint directory is empty")
    result: list[CheckpointFile] = []
    for path in paths:
        data = path.read_bytes()
        result.append(
            CheckpointFile(
                path=path.relative_to(root).as_posix(),
                sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
            )
        )
    return tuple(result)


# Compatibility alias for callers that use the shorter name.
validate_checkpoint = validate_checkpoint_directory


def build_deterministic_bundle(
    checkpoint_dir: str | Path,
    *,
    revision: str,
    model_id: str = TARGET_MODEL_ID,
) -> DeterministicBundle:
    """Build reproducible gzip/tar bytes from a validated local checkpoint."""

    root = Path(checkpoint_dir).expanduser()
    files = validate_checkpoint_directory(root, revision=revision, model_id=model_id)
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for item in files:
            data = (root / PurePosixPath(item.path)).read_bytes()
            info = tarfile.TarInfo(item.path)
            info.size = len(data)
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.mtime = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(data))
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", filename="", mtime=0) as output:
        output.write(tar_bytes.getvalue())
    data = compressed.getvalue()
    return DeterministicBundle(
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        model_id=model_id,
        revision=revision,
        files=files,
    )


def _safe_prefix(prefix: str) -> str:
    clean = prefix.strip("/")
    if not clean or any(part in {".", ".."} for part in clean.split("/")):
        raise CheckpointStagingError("S3 prefix must be a safe non-empty path")
    return clean


def stage_checkpoint(
    checkpoint_dir: str | Path,
    *,
    bucket: str,
    revision: str,
    model_id: str = TARGET_MODEL_ID,
    prefix: str = "post-training/checkpoints",
    s3_client: Any | None = None,
    region: str | None = None,
) -> StagedCheckpoint:
    """Validate, deterministically bundle, and upload one versioned object."""

    if not isinstance(bucket, str) or not bucket.strip():
        raise CheckpointStagingError("S3 bucket is required")
    bundle = build_deterministic_bundle(checkpoint_dir, revision=revision, model_id=model_id)
    key = (
        f"{_safe_prefix(prefix)}/{model_id.rsplit('/', 1)[-1]}/"
        f"{revision}/{bundle.sha256}.tar.gz"
    )
    if s3_client is None:
        try:
            import boto3  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - dependency is present in deployment
            raise CheckpointStagingError("boto3 is required for S3 staging") from exc
        s3_client = boto3.client("s3", region_name=region)

    try:
        versioning = s3_client.get_bucket_versioning(Bucket=bucket)
    except Exception as exc:
        raise CheckpointStagingError("could not verify S3 bucket versioning") from exc
    if not isinstance(versioning, dict) or versioning.get("Status") != "Enabled":
        raise CheckpointStagingError("S3 bucket versioning must be Enabled before staging")

    metadata = {
        "sha256": bundle.sha256,
        "bundle-sha256": bundle.sha256,
        "model-id": bundle.model_id,
        "hf-revision": bundle.revision,
        "file-count": str(len(bundle.files)),
        "bundle-size-bytes": str(bundle.size_bytes),
        "s3-versioning": "Enabled",
    }
    try:
        response = s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=bundle.data,
            ContentType="application/gzip",
            Metadata=metadata,
        )
    except Exception as exc:
        raise CheckpointStagingError(
            "S3 checkpoint upload failed; artifact was not verified"
        ) from exc
    version_id = response.get("VersionId") if isinstance(response, dict) else None
    if (
        not isinstance(version_id, str)
        or not version_id.strip()
        or version_id.strip().lower() == "null"
    ):
        raise CheckpointStagingError("S3 upload did not return a version id")
    version_id = version_id.strip()
    return StagedCheckpoint(
        model_id=bundle.model_id,
        revision=bundle.revision,
        sha256=bundle.sha256,
        size_bytes=bundle.size_bytes,
        file_count=len(bundle.files),
        bucket=bucket,
        key=key,
        version_id=str(version_id),
        metadata=metadata,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True, help="lowercase 40-character HF commit SHA")
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--prefix", default="post-training/checkpoints")
    parser.add_argument("--region", default=None)
    parser.add_argument("--model-id", default=TARGET_MODEL_ID)
    parser.add_argument(
        "--dry-run", action="store_true", help="validate and hash without uploading"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        bundle = build_deterministic_bundle(
            args.checkpoint_dir, revision=args.revision, model_id=args.model_id
        )
        if args.dry_run:
            print(json.dumps({"status": "VALIDATED", **bundle.manifest()}, sort_keys=True))
            return 0
        if not args.bucket:
            raise CheckpointStagingError("--bucket is required unless --dry-run is used")
        result = stage_checkpoint(
            args.checkpoint_dir,
            bucket=args.bucket,
            revision=args.revision,
            model_id=args.model_id,
            prefix=args.prefix,
            region=args.region,
        )
        print(json.dumps(result.to_dict(), sort_keys=True))
        return 0
    except CheckpointStagingError as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke tests
    raise SystemExit(main())
