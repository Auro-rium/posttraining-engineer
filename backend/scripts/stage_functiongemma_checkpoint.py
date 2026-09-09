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
import base64
import gzip
import hashlib
import io
import json
import os
import re
import struct
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

TARGET_MODEL_ID = "google/functiongemma-270m-it"
EXPECTED_MODEL_TYPE = "gemma3_text"
EXPECTED_ARCHITECTURES = ("Gemma3ForCausalLM",)
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WEIGHT_NAMES = (
    "model.safetensors",
    "pytorch_model.bin",
)
_GATE_MARKERS = frozenset(
    {".gated", "gated", "gated.json", "access_request.json", "access_denied"}
)
_CACHE_COMPONENTS = frozenset(
    {".cache", "cache", "refs", "snapshots", "blobs", ".git", ".hg", ".svn"}
)
_RESTRICTED_FLAGS = frozenset(
    {
        "gated",
        "is_gated",
        "private",
        "is_private",
        "access_restricted",
        "access_denied",
        "disabled",
        "unavailable",
    }
)
_SHARD_RE = {
    ".safetensors": re.compile(r"^model-(\d{5})-of-(\d{5})\.safetensors$"),
    ".bin": re.compile(r"^pytorch_model-(\d{5})-of-(\d{5})\.bin$"),
}
_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "C64": 8,
    "C128": 16,
}
_MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024


class CheckpointStagingError(ValueError):
    """Raised when a checkpoint cannot be proven safe and complete to stage."""


def _parse_shard_name(name: str, suffix: str) -> tuple[int, int] | None:
    match = _SHARD_RE[suffix].fullmatch(name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


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
class _CheckpointSnapshot:
    """One read-only byte snapshot used for all validation and archive writes."""

    files: tuple[CheckpointFile, ...]
    contents: Mapping[str, bytes]


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
    encryption: str
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
            "encryption": self.encryption,
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
        if any(part in _CACHE_COMPONENTS for part in parts):
            if "refs" in parts:
                raise CheckpointStagingError(
                    f"checkpoint contains mutable cache reference: {relative}"
                )
            raise CheckpointStagingError(f"checkpoint contains mutable cache artifact: {relative}")
        if any(part in {".locks", "locks"} or part.startswith(".lock") for part in parts):
            raise CheckpointStagingError(f"checkpoint contains cache lock directory: {relative}")
        if path.is_file():
            paths.append(path)
        elif not path.is_dir():
            raise CheckpointStagingError(f"checkpoint contains unsupported file type: {relative}")
    return paths


def _read_regular_file(path: Path) -> bytes:
    """Read one regular file without following a symlink introduced in a race."""

    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise CheckpointStagingError(f"could not read checkpoint file: {path}") from exc
    try:
        with os.fdopen(fd, "rb") as stream:
            return stream.read()
    except OSError as exc:
        raise CheckpointStagingError(f"could not read checkpoint file: {path}") from exc


def _truthy_restricted_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "n", "off", "none", "null"}
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


def _parse_json(data: bytes, relative: str) -> dict[str, object]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointStagingError(f"invalid checkpoint JSON metadata: {relative}") from exc
    if not isinstance(value, dict):
        raise CheckpointStagingError(
            f"checkpoint JSON metadata/report must be an object: {relative}"
        )
    return value


def _reject_gated_metadata(
    contents: Mapping[str, bytes], *, revision: str, model_id: str
) -> None:
    for relative, data in contents.items():
        if not relative.lower().endswith(".json"):
            continue
        value = _parse_json(data, relative)
        if _contains_restricted_flag(value):
            raise CheckpointStagingError(
                f"checkpoint metadata is gated/private/restricted: {relative}"
            )
        _validate_hf_metadata_identity(
            value,
            relative,
            revision=revision,
            model_id=model_id,
        )


def _validate_hf_metadata_identity(
    value: Mapping[str, object],
    relative: str,
    *,
    revision: str,
    model_id: str,
) -> None:
    """Validate identity/status fields when a Hub metadata report is present."""

    for field in ("siblings", "files"):
        observed = value.get(field)
        if observed is not None and not isinstance(observed, list):
            raise CheckpointStagingError(f"HF metadata has an invalid {field} shape: {relative}")
    for field in ("status", "state"):
        observed = value.get(field)
        if observed is not None and not isinstance(observed, str):
            raise CheckpointStagingError(f"HF metadata has an invalid {field} shape: {relative}")

    for field in ("sha", "revision", "commit_sha", "hf_revision"):
        observed = value.get(field)
        if observed is not None:
            if not isinstance(observed, str) or not _REVISION_RE.fullmatch(observed):
                raise CheckpointStagingError(f"HF metadata has an invalid revision: {relative}")
            if observed != revision:
                raise CheckpointStagingError(
                    f"HF metadata revision does not match requested commit: {relative}"
                )
    for field in ("model_id", "repo_id", "id"):
        observed = value.get(field)
        if observed is not None and observed != model_id:
            raise CheckpointStagingError(f"HF metadata model does not match target: {relative}")
    for field in ("status", "state"):
        observed = value.get(field)
        if isinstance(observed, str) and observed.strip().lower() in {
            "unavailable",
            "disabled",
            "gated",
            "private",
            "error",
        }:
            raise CheckpointStagingError(f"HF revision is unavailable: {relative}")


def _validate_required_files_legacy(paths: list[Path], root: Path) -> None:
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
        if filename == "config.json" and not isinstance(value.get("model_type"), str):
            raise CheckpointStagingError(f"required checkpoint JSON has no model_type: {filename}")
        if filename == "tokenizer.json" and not isinstance(
            value.get("version"), (str, int, float)
        ):
            raise CheckpointStagingError(f"required checkpoint JSON has no version: {filename}")

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
    nested_weights = [path for path in weights if path.parent != root]
    if nested_weights:
        raise CheckpointStagingError(
            "model weight files must be at the checkpoint root: "
            + ", ".join(path.relative_to(root).as_posix() for path in nested_weights)
        )

    safetensor_shards = {
        path.name for path in paths if _SHARD_RE[".safetensors"].fullmatch(path.name)
    }
    bin_shards = {path.name for path in paths if _SHARD_RE[".bin"].fullmatch(path.name)}
    unsupported_shards = {
        path.name
        for path in paths
        if (path.name.startswith("model-") and path.suffix == ".safetensors")
        or (path.name.startswith("pytorch_model-") and path.suffix == ".bin")
    } - safetensor_shards - bin_shards
    if unsupported_shards:
        raise CheckpointStagingError(
            "checkpoint contains an invalid weight shard name: "
            + ", ".join(sorted(unsupported_shards))
        )

    index_paths = [
        path
        for path in paths
        if path.name in {"model.safetensors.index.json", "pytorch_model.bin.index.json"}
    ]
    nested_indexes = [path for path in index_paths if path.parent != root]
    if nested_indexes:
        raise CheckpointStagingError(
            "weight indexes must be at the checkpoint root: "
            + ", ".join(path.relative_to(root).as_posix() for path in nested_indexes)
        )
    index_names = {path.name for path in index_paths}
    if safetensor_shards and "model.safetensors.index.json" not in index_names:
        raise CheckpointStagingError("sharded safetensors require a weight index")
    if bin_shards and "pytorch_model.bin.index.json" not in index_names:
        raise CheckpointStagingError("sharded pytorch weights require a weight index")
    if safetensor_shards and bin_shards:
        raise CheckpointStagingError("checkpoint mixes safetensors and pytorch weight shards")

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
        index_metadata = index.get("metadata") if isinstance(index, dict) else None
        if index_metadata is not None and not isinstance(index_metadata, dict):
            raise CheckpointStagingError(
                f"weight index metadata has an invalid shape: {index_path.relative_to(root)}"
            )
        if isinstance(index_metadata, dict) and "total_size" in index_metadata:
            total_size = index_metadata["total_size"]
            if (
                isinstance(total_size, bool)
                or not isinstance(total_size, (int, float))
                or total_size < 0
            ):
                raise CheckpointStagingError(
                    f"weight index metadata has an invalid total_size: "
                    f"{index_path.relative_to(root)}"
                )
        expected_suffix = ".safetensors" if index_path.name.startswith("model.") else ".bin"
        expected_shards = safetensor_shards if expected_suffix == ".safetensors" else bin_shards
        mapped_shards: set[str] = set()
        parsed_shards: list[tuple[int, int]] = []
        for parameter, value in weight_map.items():
            if not isinstance(parameter, str) or not parameter:
                raise CheckpointStagingError(
                    "weight index contains an invalid parameter name: "
                    f"{index_path.relative_to(root)}"
                )
            if not isinstance(value, str) or not value:
                raise CheckpointStagingError(
                    f"weight index contains a non-string shard: {index_path.relative_to(root)}"
                )
            shard = PurePosixPath(value)
            shard_parts = _parse_shard_name(value, expected_suffix)
            if shard.name != value or shard_parts is None:
                raise CheckpointStagingError(
                    f"weight index contains an invalid shard name: {value}"
                )
            shard_number, declared_total = shard_parts
            if shard_number < 1 or declared_total < 1:
                raise CheckpointStagingError(
                    f"weight index contains an invalid shard name: {value}"
                )
            if shard_number > declared_total:
                raise CheckpointStagingError(
                    f"weight index contains an invalid shard name: {value}"
                )
            mapped_shards.add(value)
            parsed_shards.append((shard_number, declared_total))
        available_shards = expected_shards
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
        shard_numbers = {number for number, _ in parsed_shards}
        declared_totals = {total for _, total in parsed_shards}
        expected_total = next(iter(declared_totals))
        if len(declared_totals) != 1 or expected_total != len(mapped_shards):
            raise CheckpointStagingError(
                f"weight index has an incomplete shard set: {index_path.relative_to(root)}"
            )
        if shard_numbers != set(range(1, expected_total + 1)):
            raise CheckpointStagingError(
                f"weight index has an incomplete shard set: {index_path.relative_to(root)}"
            )
        for shard_name in mapped_shards:
            if (root / shard_name).stat().st_size == 0:
                raise CheckpointStagingError(f"model weight file is empty: {shard_name}")


def _parse_safetensors_header(data: bytes, relative: str) -> frozenset[str]:
    if len(data) < 8:
        raise CheckpointStagingError(f"invalid safetensors header: {relative}")
    (header_size,) = struct.unpack("<Q", data[:8])
    if header_size > _MAX_SAFETENSORS_HEADER_BYTES or header_size > len(data) - 8:
        raise CheckpointStagingError(f"invalid safetensors header: {relative}")
    try:
        header_value = json.loads(data[8 : 8 + header_size].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointStagingError(f"invalid safetensors header: {relative}") from exc
    if not isinstance(header_value, dict):
        raise CheckpointStagingError(f"invalid safetensors header: {relative}")
    metadata = header_value.get("__metadata__")
    if metadata is not None and (
        not isinstance(metadata, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items()
        )
    ):
        raise CheckpointStagingError(f"invalid safetensors metadata: {relative}")
    payload_size = len(data) - 8 - header_size
    ranges: list[tuple[int, int]] = []
    tensor_names: set[str] = set()
    for tensor_name, descriptor in header_value.items():
        if tensor_name == "__metadata__":
            continue
        if not isinstance(tensor_name, str) or not tensor_name or not isinstance(descriptor, dict):
            raise CheckpointStagingError(f"invalid safetensors tensor header: {relative}")
        dtype = descriptor.get("dtype")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        element_size = _SAFETENSORS_DTYPE_BYTES.get(dtype) if isinstance(dtype, str) else None
        if element_size is None or not isinstance(shape, list) or not isinstance(offsets, list):
            raise CheckpointStagingError(f"invalid safetensors tensor header: {relative}")
        if any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape):
            raise CheckpointStagingError(f"invalid safetensors tensor shape: {relative}")
        if len(offsets) != 2 or any(
            isinstance(offset, bool) or not isinstance(offset, int) for offset in offsets
        ):
            raise CheckpointStagingError(f"invalid safetensors tensor offsets: {relative}")
        start, end = offsets
        if start < 0 or end < start or end > payload_size:
            raise CheckpointStagingError(f"invalid safetensors tensor offsets: {relative}")
        expected_size = element_size
        for dimension in shape:
            expected_size *= dimension
        if end - start != expected_size:
            raise CheckpointStagingError(f"safetensors tensor size mismatch: {relative}")
        ranges.append((start, end))
        tensor_names.add(tensor_name)
    if not tensor_names:
        raise CheckpointStagingError(f"safetensors file has no tensors: {relative}")
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise CheckpointStagingError(f"safetensors tensor ranges are incomplete: {relative}")
        cursor = end
    if cursor != payload_size:
        raise CheckpointStagingError(f"safetensors tensor ranges are incomplete: {relative}")
    return frozenset(tensor_names)


def _validate_required_files(contents: Mapping[str, bytes]) -> None:
    names = set(contents)
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    missing = sorted(required - names)
    if missing:
        raise CheckpointStagingError(
            "checkpoint is incomplete; missing required file(s): " + ", ".join(missing)
        )
    required_json = {filename: _parse_json(contents[filename], filename) for filename in required}
    config = required_json["config.json"]
    if config.get("model_type") != EXPECTED_MODEL_TYPE:
        raise CheckpointStagingError("config.json is not the exact FunctionGemma model_type")
    if config.get("architectures") != list(EXPECTED_ARCHITECTURES):
        raise CheckpointStagingError("config.json has an unexpected FunctionGemma architecture")
    if "_name_or_path" in config and config["_name_or_path"] not in {"", TARGET_MODEL_ID}:
        raise CheckpointStagingError("config.json has an unexpected model identity")
    if not isinstance(required_json["tokenizer.json"].get("version"), (str, int, float)):
        raise CheckpointStagingError("required checkpoint JSON has no version: tokenizer.json")

    safetensor_shards = {
        name for name in names if _parse_shard_name(name, ".safetensors") is not None
    }
    invalid_shards = {
        name for name in names if name.startswith("model-") and name.endswith(".safetensors")
    } - safetensor_shards
    if invalid_shards:
        raise CheckpointStagingError(
            "checkpoint contains an invalid weight shard name: " + ", ".join(sorted(invalid_shards))
        )
    pytorch_weights = {
        name
        for name in names
        if name == "pytorch_model.bin"
        or (name.startswith("pytorch_model-") and name.endswith(".bin"))
    }
    if pytorch_weights:
        raise CheckpointStagingError("PyTorch weight format is unsupported; use safetensors")
    if "model.safetensors" not in names and not safetensor_shards:
        raise CheckpointStagingError(
            "checkpoint is incomplete; missing model weight file model.safetensors"
        )
    if "model.safetensors" in names and safetensor_shards:
        raise CheckpointStagingError("checkpoint mixes unsharded and sharded safetensors")

    index_name = "model.safetensors.index.json"
    if "model.safetensors" in names:
        if index_name in names:
            raise CheckpointStagingError("unsharded safetensors must not have a weight index")
        _parse_safetensors_header(contents["model.safetensors"], "model.safetensors")
        return
    if not safetensor_shards or index_name not in names:
        raise CheckpointStagingError("sharded safetensors require a weight index")

    index = _parse_json(contents[index_name], index_name)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise CheckpointStagingError(f"weight index has no weight_map: {index_name}")
    index_metadata = index.get("metadata")
    if index_metadata is not None and not isinstance(index_metadata, dict):
        raise CheckpointStagingError(f"weight index metadata has an invalid shape: {index_name}")
    mapped_shards: set[str] = set()
    mapped_tensors: dict[str, str] = {}
    shard_tensors: dict[str, frozenset[str]] = {}
    parsed_shards: list[tuple[int, int]] = []
    for parameter, value in weight_map.items():
        if not isinstance(parameter, str) or not parameter:
            raise CheckpointStagingError(
                f"weight index contains an invalid parameter name: {index_name}"
            )
        if not isinstance(value, str) or not value:
            raise CheckpointStagingError(f"weight index contains a non-string shard: {index_name}")
        shard = PurePosixPath(value)
        shard_parts = _parse_shard_name(value, ".safetensors")
        if shard.name != value or shard_parts is None:
            raise CheckpointStagingError(f"weight index contains an invalid shard name: {value}")
        shard_number, declared_total = shard_parts
        if shard_number > declared_total:
            raise CheckpointStagingError(f"weight index contains an invalid shard name: {value}")
        if value not in contents:
            raise CheckpointStagingError(
                f"checkpoint is incomplete; missing weight shard(s): {value}"
            )
        if value not in shard_tensors:
            shard_tensors[value] = _parse_safetensors_header(contents[value], value)
        if parameter in mapped_tensors:
            raise CheckpointStagingError(f"weight index contains a duplicate tensor: {parameter}")
        mapped_tensors[parameter] = value
        mapped_shards.add(value)
        parsed_shards.append((shard_number, declared_total))
    if mapped_shards != safetensor_shards:
        raise CheckpointStagingError("weight index does not match available model shards")
    declared_totals = {total for _, total in parsed_shards}
    expected_total = next(iter(declared_totals))
    if len(declared_totals) != 1 or expected_total != len(mapped_shards):
        raise CheckpointStagingError(f"weight index has an incomplete shard set: {index_name}")
    if {number for number, _ in parsed_shards} != set(range(1, expected_total + 1)):
        raise CheckpointStagingError(f"weight index has an incomplete shard set: {index_name}")
    all_tensors = set().union(*shard_tensors.values())
    if all_tensors != set(mapped_tensors):
        raise CheckpointStagingError("weight index tensor map does not match safetensors headers")
    for tensor_name, shard_name in mapped_tensors.items():
        if tensor_name not in shard_tensors[shard_name]:
            raise CheckpointStagingError(
                f"weight index tensor is absent from shard header: {tensor_name}"
            )


def _checkpoint_snapshot(
    checkpoint_dir: str | Path, *, revision: str, model_id: str
) -> _CheckpointSnapshot:
    if model_id != TARGET_MODEL_ID:
        raise CheckpointStagingError(
            f"only the pinned target model {TARGET_MODEL_ID!r} may be staged"
        )
    validate_immutable_revision(revision)
    root = Path(checkpoint_dir).expanduser()
    paths = _relative_files(root)
    contents = {
        path.relative_to(root).as_posix(): _read_regular_file(path) for path in paths
    }
    if not contents:
        raise CheckpointStagingError("checkpoint directory is empty")
    _reject_gated_metadata(contents, revision=revision, model_id=model_id)
    _validate_required_files(contents)
    files = tuple(
        CheckpointFile(path=name, sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
        for name, data in contents.items()
    )
    return _CheckpointSnapshot(files=files, contents=contents)


def validate_checkpoint_directory(
    checkpoint_dir: str | Path,
    *,
    revision: str,
    model_id: str = TARGET_MODEL_ID,
) -> tuple[CheckpointFile, ...]:
    """Validate a local checkpoint and return sorted per-file digests."""
    return _checkpoint_snapshot(
        checkpoint_dir, revision=revision, model_id=model_id
    ).files


# Compatibility alias for callers that use the shorter name.
validate_checkpoint = validate_checkpoint_directory


def build_deterministic_bundle(
    checkpoint_dir: str | Path,
    *,
    revision: str,
    model_id: str = TARGET_MODEL_ID,
) -> DeterministicBundle:
    """Build reproducible gzip/tar bytes from a validated local checkpoint."""

    snapshot = _checkpoint_snapshot(checkpoint_dir, revision=revision, model_id=model_id)
    files = snapshot.files
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for item in files:
            data = snapshot.contents[item.path]
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


def _required_bucket_encryption(s3_client: Any, bucket: str) -> str:
    """Return the configured SSE algorithm or fail before any upload."""

    try:
        response = s3_client.get_bucket_encryption(Bucket=bucket)
    except Exception as exc:
        raise CheckpointStagingError("could not verify S3 bucket encryption") from exc
    if not isinstance(response, Mapping):
        raise CheckpointStagingError("S3 bucket encryption report has an invalid shape")
    configuration = response.get("ServerSideEncryptionConfiguration")
    if not isinstance(configuration, Mapping):
        raise CheckpointStagingError("S3 bucket encryption is missing its configuration")
    rules = configuration.get("Rules")
    if not isinstance(rules, list):
        raise CheckpointStagingError("S3 bucket encryption report has no rules")
    algorithms: set[str] = set()
    for rule in rules:
        if not isinstance(rule, Mapping):
            continue
        default = rule.get("ApplyServerSideEncryptionByDefault")
        if isinstance(default, Mapping):
            algorithm = default.get("SSEAlgorithm")
            if isinstance(algorithm, str):
                algorithms.add(algorithm)
    supported = algorithms.intersection({"AES256", "aws:kms"})
    if not supported:
        raise CheckpointStagingError("S3 bucket encryption must be SSE-S3 or SSE-KMS")
    return sorted(supported)[0]


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
    encryption = _required_bucket_encryption(s3_client, bucket)
    checksum_b64 = base64.b64encode(bytes.fromhex(bundle.sha256)).decode("ascii")

    metadata = {
        "sha256": bundle.sha256,
        "bundle-sha256": bundle.sha256,
        "model-id": bundle.model_id,
        "hf-revision": bundle.revision,
        "file-count": str(len(bundle.files)),
        "bundle-size-bytes": str(bundle.size_bytes),
        "s3-versioning": "Enabled",
        "s3-encryption": encryption,
        "checksum-algorithm": "SHA256",
        "checksum-sha256": bundle.sha256,
        "checksum-sha256-base64": checksum_b64,
    }
    try:
        response = s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=bundle.data,
            ContentType="application/gzip",
            ServerSideEncryption=encryption,
            ChecksumSHA256=checksum_b64,
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
    returned_checksum = response.get("ChecksumSHA256") if isinstance(response, dict) else None
    if not isinstance(returned_checksum, str) or returned_checksum != checksum_b64:
        raise CheckpointStagingError("S3 upload did not return matching checksum provenance")
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
        encryption=encryption,
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
