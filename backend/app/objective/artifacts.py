"""Immutable objective-worker artifacts.

This module is deliberately separate from the objective HTTP service.  It is
the durable boundary used by a real objective worker to retain verified
trajectories and SFT datasets.  The public trajectory reference contains only
safe task metadata; the S3 location, version, digest, and internal index are
never sent to a model.

The implementation uses content-addressed object keys and requires a concrete
S3 ``VersionId`` for every write and read.  ``boto3`` is imported lazily so
unit tests can inject a small client without making AWS calls.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from threading import RLock
from typing import Any, cast
from urllib.parse import unquote

from app.providers.artifacts import ArtifactRef

from .models import (
    Dataset,
    DatasetManifest,
    DatasetRow,
    ObjectiveSplit,
    Trajectory,
    TrajectoryReference,
    deterministic_dataset_created_at,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_PUBLIC_SPLITS = frozenset({ObjectiveSplit.TRAIN, ObjectiveSplit.REPLAY, ObjectiveSplit.VALIDATION})
_PUBLIC_SPLIT_ORDER = (
    ObjectiveSplit.TRAIN,
    ObjectiveSplit.REPLAY,
    ObjectiveSplit.VALIDATION,
)
_CURATION_SPLITS = frozenset({ObjectiveSplit.TRAIN, ObjectiveSplit.REPLAY})


class ObjectiveArtifactError(RuntimeError):
    """Base error for an unavailable or untrusted objective artifact."""


class ObjectiveArtifactIntegrityError(ObjectiveArtifactError, ValueError):
    """Raised when S3 bytes or provenance cannot be verified."""


class ObjectiveArtifactNotFound(ObjectiveArtifactError):
    """Raised for an explicit artifact read that does not exist."""


@dataclass(frozen=True, slots=True)
class StoredTrajectory:
    """Internal immutable identity for a stored public trajectory."""

    reference: TrajectoryReference
    artifact: ArtifactRef


@dataclass(frozen=True, slots=True)
class StoredDataset:
    """Internal immutable identity for a stored training dataset."""

    dataset: Dataset
    artifact: ArtifactRef

    @property
    def manifest(self) -> DatasetManifest:
        return self.dataset.manifest


def _validate_bucket(value: object) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", value) is None
    ):
        raise ValueError("bucket must be a valid DNS-compatible S3 bucket name")
    if ".." in value or re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", value):
        raise ValueError("bucket must not be an IPv4 address or contain adjacent dots")
    return value


def _validate_path(value: object, *, name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not value and allow_empty:
        return value
    if not value:
        raise ValueError(f"{name} must not be empty")
    decoded = unquote(value)
    parts = decoded.split("/")
    if any(
        not part
        or part in {".", ".."}
        or any(ord(char) < 0x20 or ord(char) == 0x7F for char in part)
        for part in parts
    ):
        raise ValueError(f"{name} contains an empty, dot, or control segment")
    return decoded


def _validate_identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a bounded path-safe identifier")
    return value


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _required_version(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip().lower() == "null":
        raise ObjectiveArtifactIntegrityError(f"{context} requires an immutable S3 VersionId")
    return value.strip()


def _required_sha(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ObjectiveArtifactIntegrityError(f"{context} requires a lowercase SHA-256 digest")
    return value


def _required_size(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ObjectiveArtifactIntegrityError(f"{context} requires a non-negative byte size")
    return value


def _is_missing(exc: BaseException) -> bool:
    """Recognize only an actual missing-object response, never auth failures."""

    if isinstance(exc, (KeyError, FileNotFoundError)):
        return True
    response = getattr(exc, "response", None)
    if isinstance(response, Mapping):
        error = response.get("Error")
        if isinstance(error, Mapping):
            code = str(error.get("Code", ""))
            return code in {"404", "NoSuchKey", "NotFound", "NoSuchVersion"}
    return False


def _is_precondition_failed(exc: BaseException) -> bool:
    """Recognize S3's conditional-write conflict without hiding other errors."""

    response = getattr(exc, "response", None)
    if isinstance(response, Mapping):
        error = response.get("Error")
        if isinstance(error, Mapping) and str(error.get("Code", "")) in {
            "409",
            "412",
            "PreconditionFailed",
        }:
            return True
    return False


def _stable_created_at(dataset: Dataset) -> datetime:
    """Derive restart-stable metadata from the immutable dataset identity."""
    return deterministic_dataset_created_at(
        dataset.manifest.dataset_id, dataset.manifest.sha256
    )


class S3ObjectiveArtifactStore:
    """Persist verified public trajectories and curation-safe datasets in S3.

    The class intentionally exposes the same ``put``/``get`` shape as the
    objective service's ``TrajectoryArtifactStore`` protocol.  ``put`` and
    ``get`` cover only train/replay data.  Validation trajectories can be
    stored and retrieved only by explicitly selecting the validation split;
    hidden data has no public storage or retrieval method.
    """

    def __init__(
        self, bucket: str, *, client: Any | None = None, prefix: str = "objective"
    ) -> None:
        self.bucket = _validate_bucket(bucket)
        self.prefix = _validate_path(prefix, name="artifact prefix", allow_empty=True)
        self._client = client
        self._write_lock = RLock()

    @staticmethod
    def sha256(data: bytes) -> str:
        return _digest(data)

    def _client_or_create(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import boto3  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ObjectiveArtifactError("Install boto3 to use S3 objective artifacts") from exc
        self._client = boto3.client("s3")
        return self._client

    def _key(self, *parts: str) -> str:
        clean = [_validate_path(part, name="artifact key segment") for part in parts]
        key = "/".join(part.strip("/") for part in clean)
        return f"{self.prefix}/{key}" if self.prefix else key

    def _assert_key(self, key: str) -> None:
        clean = _validate_path(key, name="artifact key")
        if self.prefix and not (clean == self.prefix or clean.startswith(f"{self.prefix}/")):
            raise ObjectiveArtifactIntegrityError("artifact key is outside the configured prefix")

    @staticmethod
    def _metadata(response: Mapping[str, Any]) -> Mapping[str, Any]:
        value = response.get("Metadata")
        return value if isinstance(value, Mapping) else {}

    @staticmethod
    def _metadata_value(metadata: Mapping[str, Any], name: str) -> object:
        for key, value in metadata.items():
            if str(key).lower() == name.lower():
                return value
        return None

    def _head(self, key: str, *, version_id: str | None = None) -> dict[str, Any]:
        self._assert_key(key)
        kwargs: dict[str, Any] = {"Bucket": self.bucket, "Key": key}
        if version_id is not None:
            kwargs["VersionId"] = _required_version(version_id, context="artifact read")
        try:
            response = self._client_or_create().head_object(**kwargs)
        except Exception as exc:
            if _is_missing(exc):
                raise ObjectiveArtifactNotFound(f"artifact does not exist: {key}") from exc
            raise ObjectiveArtifactError(f"S3 head failed for objective artifact: {key}") from exc
        result = dict(response)
        observed_version = _required_version(result.get("VersionId"), context="S3 head")
        if version_id is not None and observed_version != version_id:
            raise ObjectiveArtifactIntegrityError(
                "S3 head VersionId does not match requested version"
            )
        return result

    def _ref_from_head(self, key: str, head: Mapping[str, Any], *, context: str) -> ArtifactRef:
        metadata = self._metadata(head)
        digest = _required_sha(self._metadata_value(metadata, "sha256"), context=context)
        size = _required_size(head.get("ContentLength"), context=context)
        version = _required_version(head.get("VersionId"), context=context)
        return ArtifactRef(
            bucket=self.bucket,
            key=key,
            sha256=digest,
            size_bytes=size,
            version_id=version,
            content_type=(str(head["ContentType"]) if head.get("ContentType") else None),
            etag=(str(head["ETag"]) if head.get("ETag") else None),
        )

    def _read_bytes(self, ref: ArtifactRef, *, context: str) -> bytes:
        if ref.bucket != self.bucket:
            raise ObjectiveArtifactIntegrityError("artifact is outside the configured bucket")
        self._assert_key(ref.key)
        version = _required_version(ref.version_id, context=context)
        expected_sha = _required_sha(ref.sha256, context=context)
        expected_size = _required_size(ref.size_bytes, context=context)
        head = self._head(ref.key, version_id=version)
        observed = self._ref_from_head(ref.key, head, context=context)
        if observed.sha256 != expected_sha or observed.size_bytes != expected_size:
            raise ObjectiveArtifactIntegrityError("S3 metadata does not match expected artifact")
        try:
            response = dict(
                self._client_or_create().get_object(
                    Bucket=self.bucket, Key=ref.key, VersionId=version
                )
            )
        except Exception as exc:
            if _is_missing(exc):
                raise ObjectiveArtifactNotFound(f"artifact does not exist: {ref.key}") from exc
            raise ObjectiveArtifactError(
                f"S3 download failed for objective artifact: {ref.key}"
            ) from exc
        response_version = _required_version(response.get("VersionId"), context="S3 download")
        if response_version != version:
            raise ObjectiveArtifactIntegrityError(
                "S3 download VersionId does not match requested version"
            )
        body = response.get("Body")
        if body is None:
            raise ObjectiveArtifactIntegrityError("S3 download did not contain a body")
        data = body.read() if hasattr(body, "read") else body
        if not isinstance(data, bytes):
            data = bytes(data)
        if len(data) != expected_size or _digest(data) != expected_sha:
            raise ObjectiveArtifactIntegrityError(
                "downloaded bytes do not match artifact digest or size"
            )
        return data

    def _put_bytes(
        self, key: str, data: bytes, *, kind: str, split: ObjectiveSplit | None
    ) -> ArtifactRef:
        if not isinstance(data, bytes):
            raise TypeError("objective artifact data must be bytes")
        self._assert_key(key)
        digest = _digest(data)
        metadata = {
            "sha256": digest,
            "size-bytes": str(len(data)),
            "artifact-kind": kind,
        }
        if split is not None:
            metadata["split"] = split.value

        def existing_or_raise() -> ArtifactRef | None:
            try:
                head = self._head(key)
            except ObjectiveArtifactNotFound:
                return None
            existing = self._ref_from_head(key, head, context="existing artifact")
            if existing.sha256 != digest or existing.size_bytes != len(data):
                raise ObjectiveArtifactIntegrityError(
                    "content-addressed key is already bound to other bytes"
                )
            self._read_bytes(existing, context="existing artifact")
            return existing

        # The local lock makes the injected test client thread-safe.  The S3
        # conditional write below provides the same reconcile-on-conflict
        # behavior across independently scaled objective workers.
        with self._write_lock:
            existing = existing_or_raise()
            if existing is not None:
                return existing
            try:
                response = dict(
                    self._client_or_create().put_object(
                        Bucket=self.bucket,
                        Key=key,
                        Body=data,
                        ContentType="application/json",
                        Metadata=metadata,
                        IfNoneMatch="*",
                    )
                )
            except Exception as exc:
                if _is_precondition_failed(exc):
                    reconciled = existing_or_raise()
                    if reconciled is not None:
                        return reconciled
                raise ObjectiveArtifactError(
                    f"S3 upload failed for objective artifact: {key}"
                ) from exc
            version = _required_version(response.get("VersionId"), context="S3 upload")
            ref = ArtifactRef(
                bucket=self.bucket,
                key=key,
                sha256=digest,
                size_bytes=len(data),
                version_id=version,
                content_type="application/json",
                etag=(str(response["ETag"]) if response.get("ETag") else None),
            )
            # Verify the provider committed exactly what was returned before the
            # reference can cross the objective-worker boundary.
            self._read_bytes(ref, context="uploaded artifact")
            return ref

    def _write_index(self, key: str, payload: Mapping[str, Any]) -> ArtifactRef:
        return self._put_bytes(key, _canonical_json(payload), kind="index", split=None)

    def _read_index(self, key: str) -> dict[str, Any]:
        head = self._head(key)
        ref = self._ref_from_head(key, head, context="artifact index")
        try:
            value = json.loads(self._read_bytes(ref, context="artifact index").decode("utf-8"))
            if not isinstance(value, dict):
                raise TypeError("artifact index must be a JSON object")
            return cast(dict[str, Any], value)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise ObjectiveArtifactIntegrityError("artifact index is not valid JSON") from exc

    def put_trajectory(self, trajectory: Trajectory) -> TrajectoryReference:
        if trajectory.split is ObjectiveSplit.HIDDEN:
            raise ObjectiveArtifactIntegrityError(
                "hidden trajectories cannot cross the public artifact boundary"
            )
        if trajectory.split not in _PUBLIC_SPLITS:
            raise ObjectiveArtifactIntegrityError("unsupported trajectory split")
        if not trajectory.verified:
            raise ObjectiveArtifactIntegrityError(
                "only verifier-confirmed trajectories may be stored"
            )
        trajectory_id = _validate_identifier(trajectory.trajectory_id, name="trajectory_id")
        payload = _canonical_json(trajectory.model_dump(mode="json"))
        digest = _digest(payload)
        key = self._key("trajectories", trajectory.split.value, digest[:2], f"{digest}.json")
        artifact = self._put_bytes(key, payload, kind="trajectory", split=trajectory.split)
        index_key = self._key("trajectory-index", trajectory.split.value, f"{trajectory_id}.json")
        index = {
            "trajectory_id": trajectory_id,
            "task_id": trajectory.task_id,
            "split": trajectory.split.value,
            "verified": True,
            "artifact": artifact.to_dict(),
        }
        self._write_index(index_key, index)
        return TrajectoryReference(
            trajectory_id=trajectory_id,
            task_id=trajectory.task_id,
            split=trajectory.split,
            verified=True,
        )

    put = put_trajectory

    def _index_for(self, trajectory_id: str, *, split: ObjectiveSplit) -> dict[str, Any] | None:
        key = self._key("trajectory-index", split.value, f"{trajectory_id}.json")
        try:
            return self._read_index(key)
        except ObjectiveArtifactNotFound:
            return None

    def get_trajectory(
        self,
        trajectory_id: str,
        *,
        split: ObjectiveSplit | None = None,
        allow_validation: bool = False,
    ) -> Trajectory | None:
        trajectory_id = _validate_identifier(trajectory_id, name="trajectory_id")
        if split is ObjectiveSplit.HIDDEN:
            raise ObjectiveArtifactIntegrityError("hidden trajectories cannot be retrieved")
        splits = (split,) if split is not None else (ObjectiveSplit.TRAIN, ObjectiveSplit.REPLAY)
        if ObjectiveSplit.VALIDATION in splits and not allow_validation:
            raise ObjectiveArtifactIntegrityError(
                "validation retrieval requires explicit evaluator scope"
            )
        for candidate_split in splits:
            if candidate_split not in _PUBLIC_SPLITS:
                raise ObjectiveArtifactIntegrityError("unsupported trajectory split")
            index = self._index_for(trajectory_id, split=candidate_split)
            if index is None:
                continue
            if (
                index.get("trajectory_id") != trajectory_id
                or index.get("split") != candidate_split.value
            ):
                raise ObjectiveArtifactIntegrityError(
                    "trajectory index provenance does not match request"
                )
            if index.get("verified") is not True:
                raise ObjectiveArtifactIntegrityError("trajectory index is not verifier-confirmed")
            raw_ref = index.get("artifact")
            if not isinstance(raw_ref, Mapping):
                raise ObjectiveArtifactIntegrityError(
                    "trajectory index lacks an artifact reference"
                )
            try:
                ref = ArtifactRef.from_dict(raw_ref)
                value = json.loads(self._read_bytes(ref, context="trajectory").decode("utf-8"))
                trajectory = Trajectory.model_validate(value)
            except ObjectiveArtifactError:
                raise
            except (
                KeyError,
                TypeError,
                ValueError,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as exc:
                raise ObjectiveArtifactIntegrityError("stored trajectory is malformed") from exc
            if (
                trajectory.trajectory_id != trajectory_id
                or trajectory.split is not candidate_split
                or not trajectory.verified
            ):
                raise ObjectiveArtifactIntegrityError(
                    "stored trajectory provenance does not match index"
                )
            return trajectory
        return None

    get = get_trajectory

    def resolve_trajectory_reference(
        self, reference: TrajectoryReference, *, allow_validation: bool = False
    ) -> Trajectory:
        """Resolve using the durable index, never caller flags as authority.

        The reference's split and ``verified`` bit are claims to compare with
        persisted index metadata.  They do not grant validation access; that
        requires an explicit evaluator-only argument at the storage boundary.
        """

        if not reference.verified:
            raise ObjectiveArtifactIntegrityError("trajectory reference is not verified")
        if reference.split is ObjectiveSplit.HIDDEN:
            raise ObjectiveArtifactIntegrityError("hidden trajectories cannot be resolved")
        trajectory_id = _validate_identifier(reference.trajectory_id, name="trajectory_id")
        trusted_index: dict[str, Any] | None = None
        trusted_split: ObjectiveSplit | None = None
        for candidate_split in _PUBLIC_SPLIT_ORDER:
            index = self._index_for(trajectory_id, split=candidate_split)
            if index is None:
                continue
            raw_split = index.get("split")
            if not isinstance(raw_split, str):
                raise ObjectiveArtifactIntegrityError("trajectory index contains an invalid split")
            try:
                indexed_split = ObjectiveSplit(raw_split)
            except ValueError as exc:
                raise ObjectiveArtifactIntegrityError(
                    "trajectory index contains an invalid split"
                ) from exc
            if indexed_split is not candidate_split:
                raise ObjectiveArtifactIntegrityError(
                    "trajectory index provenance does not match its key"
                )
            trusted_index = index
            trusted_split = indexed_split
            break
        if trusted_index is None or trusted_split is None:
            raise ObjectiveArtifactIntegrityError("trajectory reference is not resolvable")
        if trusted_split is ObjectiveSplit.VALIDATION and not allow_validation:
            raise ObjectiveArtifactIntegrityError(
                "validation retrieval requires explicit evaluator scope"
            )
        if (
            trusted_index.get("trajectory_id") != trajectory_id
            or trusted_index.get("task_id") != reference.task_id
            or trusted_split is not reference.split
            or trusted_index.get("verified") is not True
        ):
            raise ObjectiveArtifactIntegrityError(
                "trajectory reference metadata does not match trusted index"
            )
        trajectory = self.get_trajectory(
            trajectory_id, split=trusted_split, allow_validation=allow_validation
        )
        if trajectory is None:
            raise ObjectiveArtifactIntegrityError("trajectory reference is not resolvable")
        return trajectory

    def put_dataset(self, dataset: Dataset) -> DatasetManifest:
        if not dataset.rows:
            raise ObjectiveArtifactIntegrityError("dataset requires at least one row")
        row_splits = {row.split for row in dataset.rows}
        if len(row_splits) != 1:
            raise ObjectiveArtifactIntegrityError("dataset rows must use one curation split")
        for row in dataset.rows:
            if row.split not in _CURATION_SPLITS:
                raise ObjectiveArtifactIntegrityError(
                    "validation and hidden rows cannot enter training datasets"
                )
            if not row.verifier_confirmed or row.source_type != "verified_replay":
                raise ObjectiveArtifactIntegrityError(
                    "dataset rows must carry verifier-confirmed replay evidence"
                )
        if (
            tuple(row.source_trajectory_id for row in dataset.rows)
            != dataset.manifest.source_trajectory_ids
        ):
            raise ObjectiveArtifactIntegrityError("dataset manifest provenance does not match rows")
        for row in dataset.rows:
            source = self.get_trajectory(row.source_trajectory_id, split=row.split)
            if source is None or source.task_id != row.task_id or not source.verified:
                raise ObjectiveArtifactIntegrityError(
                    "dataset row source is not present in the trusted trajectory index"
                )
        dataset_id = _validate_identifier(dataset.manifest.dataset_id, name="dataset_id")
        run_id = _validate_identifier(dataset.manifest.run_id, name="run_id")
        experiment_id = _validate_identifier(dataset.manifest.experiment_id, name="experiment_id")
        payload_digest = _required_sha(dataset.manifest.sha256, context="dataset manifest")
        expected_dataset_id = (
            "dataset-"
            + hashlib.sha256(f"{run_id}:{experiment_id}:{payload_digest}".encode()).hexdigest()[:24]
        )
        if dataset_id != expected_dataset_id:
            raise ObjectiveArtifactIntegrityError(
                "dataset ID is not deterministically bound to its run, experiment, and content"
            )
        jsonl_payload = "\n".join(row.canonical_json() for row in dataset.rows).encode("utf-8")
        if _digest(jsonl_payload) != payload_digest:
            raise ObjectiveArtifactIntegrityError("dataset rows do not match manifest digest")
        key = self._key("datasets", run_id, experiment_id, f"{payload_digest}.jsonl")
        artifact = self._put_bytes(key, jsonl_payload, kind="dataset-jsonl", split=None)
        payload_manifest = dataset.manifest.model_copy(
            update={
                "created_at": _stable_created_at(dataset),
                "s3_uri": artifact.version_ref,
            }
        )
        manifest_key = self._key(
            "datasets", run_id, experiment_id, f"{payload_digest}.manifest.json"
        )
        manifest_artifact = self._put_bytes(
            manifest_key,
            _canonical_json(payload_manifest.model_dump(mode="json")),
            kind="dataset-manifest",
            split=None,
        )
        index_key = self._key("dataset-index", f"{dataset_id}.json")
        self._write_index(
            index_key,
            {
                "dataset_id": dataset_id,
                "run_id": run_id,
                "experiment_id": experiment_id,
                "dataset_artifact": artifact.to_dict(),
                "manifest_artifact": manifest_artifact.to_dict(),
            },
        )
        # The returned URI is the exact immutable version that a trainer may
        # consume; the serialized object remains deterministic and content-addressed.
        return payload_manifest.model_copy(update={"s3_uri": artifact.version_ref})

    def get_dataset(self, dataset_id: str) -> Dataset | None:
        dataset_id = _validate_identifier(dataset_id, name="dataset_id")
        key = self._key("dataset-index", f"{dataset_id}.json")
        try:
            index = self._read_index(key)
        except ObjectiveArtifactNotFound:
            return None
        if index.get("dataset_id") != dataset_id:
            raise ObjectiveArtifactIntegrityError("dataset index provenance does not match request")
        raw_dataset_ref = index.get("dataset_artifact")
        raw_manifest_ref = index.get("manifest_artifact")
        if not isinstance(raw_dataset_ref, Mapping) or not isinstance(raw_manifest_ref, Mapping):
            raise ObjectiveArtifactIntegrityError(
                "dataset index lacks immutable artifact references"
            )
        try:
            dataset_ref = ArtifactRef.from_dict(raw_dataset_ref)
            manifest_ref = ArtifactRef.from_dict(raw_manifest_ref)
            manifest_value = json.loads(
                self._read_bytes(manifest_ref, context="dataset manifest").decode("utf-8")
            )
            if not isinstance(manifest_value, Mapping):
                raise TypeError("dataset manifest must be an object")
            manifest = DatasetManifest.model_validate(manifest_value)
            row_lines = self._read_bytes(dataset_ref, context="dataset JSONL").decode(
                "utf-8"
            ).split("\n")
            rows = tuple(DatasetRow.model_validate(json.loads(line)) for line in row_lines if line)
            dataset = Dataset(manifest=manifest, rows=rows)
        except ObjectiveArtifactError:
            raise
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ObjectiveArtifactIntegrityError("stored dataset is malformed") from exc
        if dataset.manifest.dataset_id != dataset_id:
            raise ObjectiveArtifactIntegrityError("stored dataset ID does not match index")
        if dataset.manifest.s3_uri != dataset_ref.version_ref:
            raise ObjectiveArtifactIntegrityError(
                "stored dataset manifest does not point to its immutable JSONL artifact"
            )
        if index.get("run_id") != dataset.manifest.run_id or index.get(
            "experiment_id"
        ) != dataset.manifest.experiment_id:
            raise ObjectiveArtifactIntegrityError("dataset index scope does not match manifest")
        if any(row.split not in _CURATION_SPLITS for row in dataset.rows):
            raise ObjectiveArtifactIntegrityError(
                "stored dataset crosses the sealed evaluation boundary"
            )
        return dataset

    def get_dataset_for_curation(self, dataset_id: str) -> Dataset | None:
        """Curation-facing alias that can never retrieve validation/hidden data."""

        return self.get_dataset(dataset_id)


# Short aliases make dependency injection readable without weakening the
# explicit S3 implementation name in operational code.
ObjectiveArtifactStore = S3ObjectiveArtifactStore
S3TrajectoryArtifactStore = S3ObjectiveArtifactStore
ObjectiveTrajectoryArtifactStore = S3ObjectiveArtifactStore
