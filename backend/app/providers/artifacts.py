"""S3-backed immutable artifact storage.

The adapter stores the SHA-256 digest as object metadata and returns the S3
version id.  A caller can therefore persist the returned :class:`ArtifactRef`
with a run record and later read exactly the bytes that were used by a phase.
No AWS SDK is imported until an operation needs a default client.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from re import fullmatch
from typing import Any, Protocol
from urllib.parse import parse_qs, quote, unquote, urlparse


class OptionalDependencyError(RuntimeError):
    """Raised when an AWS-backed adapter is used without its optional SDK."""


class ArtifactIntegrityError(ValueError):
    """Raised when an artifact cannot be proven immutable and content-addressed."""


def _validate_s3_path(value: object, *, name: str, allow_empty: bool = False) -> str:
    """Validate a slash-delimited S3 key/prefix without path traversal forms."""

    if not isinstance(value, str):
        raise ArtifactIntegrityError(f"{name} path must be a string")
    if not value and allow_empty:
        return value
    if not value:
        raise ArtifactIntegrityError(f"{name} path must not be empty")
    segments = value.split("/")
    if any(
        not segment
        or segment in {".", ".."}
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in segment)
        for segment in segments
    ):
        raise ArtifactIntegrityError(
            f"{name} path contains an empty, dot, or control segment"
        )
    return value


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Content and location identity for one immutable S3 object version."""

    bucket: str
    key: str
    sha256: str
    size_bytes: int
    version_id: str | None = None
    content_type: str | None = None
    etag: str | None = None

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"

    @property
    def version_ref(self) -> str:
        """Return a stable, serializable S3 reference including version id."""
        if self.version_id:
            return f"{self.uri}?versionId={quote(self.version_id, safe='')}"
        return self.uri

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["uri"] = self.uri
        result["version_ref"] = self.version_ref
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactRef:
        return cls(
            bucket=str(value["bucket"]),
            key=str(value["key"]),
            sha256=str(value["sha256"]),
            size_bytes=int(value["size_bytes"]),
            version_id=(str(value["version_id"]) if value.get("version_id") else None),
            content_type=(str(value["content_type"]) if value.get("content_type") else None),
            etag=(str(value["etag"]) if value.get("etag") else None),
        )

    @classmethod
    def from_uri(
        cls,
        value: str,
        *,
        sha256: str,
        size_bytes: int = 0,
        content_type: str | None = None,
    ) -> ArtifactRef:
        parsed = urlparse(value)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
            raise ValueError(f"Not an S3 URI: {value!r}")
        version_values = parse_qs(parsed.query).get("versionId", [])
        return cls(
            bucket=parsed.netloc,
            key=parsed.path.lstrip("/"),
            sha256=sha256,
            size_bytes=size_bytes,
            version_id=version_values[0] if version_values else None,
            content_type=content_type,
        )

    @classmethod
    def from_live_uri(
        cls,
        value: str,
        *,
        sha256: str,
        size_bytes: int,
        content_type: str | None = None,
    ) -> ArtifactRef:
        """Parse an S3 reference for live evidence with exactly one version id."""

        parsed = urlparse(value)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
            raise ValueError(f"Not an S3 URI: {value!r}")
        version_values = parse_qs(parsed.query, keep_blank_values=True).get("versionId", [])
        if len(version_values) != 1 or not version_values[0] or version_values[0].lower() == "null":
            raise ValueError("S3 live URI must contain one immutable VersionId")
        key = _validate_s3_path(unquote(parsed.path.lstrip("/")), name="artifact key")
        return cls(
            bucket=parsed.netloc,
            key=key,
            sha256=sha256,
            size_bytes=size_bytes,
            version_id=version_values[0],
            content_type=content_type,
        )


class ArtifactStore(Protocol):
    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> ArtifactRef: ...

    def get_bytes(self, ref: ArtifactRef) -> bytes: ...

    def put_json(
        self,
        key: str,
        value: Any,
        *,
        metadata: Mapping[str, str] | None = None,
    ) -> ArtifactRef: ...


class S3ArtifactStore:
    """Write and read hash-checked artifacts in an S3 bucket.

    ``client`` is intentionally a duck-typed parameter.  Tests can provide a
    small fake and production code can provide a boto3 S3 client or session
    wrapper without making boto3 a hard import-time dependency.
    """

    def __init__(self, bucket: str, *, client: Any | None = None, prefix: str = "") -> None:
        if not bucket.strip():
            raise ValueError("bucket must not be empty")
        self.bucket = bucket
        self.prefix = _validate_s3_path(prefix, name="artifact prefix", allow_empty=True)
        self._client = client

    @staticmethod
    def sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _client_or_create(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import boto3  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise OptionalDependencyError("Install boto3 to use S3ArtifactStore") from exc
        self._client = boto3.client("s3")
        return self._client

    def _key(self, key: str) -> str:
        clean = _validate_s3_path(key, name="artifact key")
        return f"{self.prefix}/{clean}" if self.prefix else clean

    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> ArtifactRef:
        if not isinstance(data, bytes):
            raise TypeError("artifact data must be bytes")
        digest = self.sha256(data)
        object_metadata = {str(k).lower(): str(v) for k, v in (metadata or {}).items()}
        object_metadata["sha256"] = digest
        kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": self._key(key),
            "Body": data,
            "Metadata": object_metadata,
        }
        if content_type:
            kwargs["ContentType"] = content_type
        response = self._client_or_create().put_object(**kwargs)
        version_id = self._required_version(
            response.get("VersionId"), context="uploaded artifact"
        )
        uploaded = ArtifactRef(
            bucket=self.bucket,
            key=kwargs["Key"],
            sha256=digest,
            size_bytes=len(data),
            version_id=version_id,
            content_type=content_type,
            etag=(str(response["ETag"]) if response.get("ETag") else None),
        )
        return self.verify_immutable(
            uploaded,
            expected_sha256=digest,
            expected_size_bytes=len(data),
        )

    def put_json(
        self,
        key: str,
        value: Any,
        *,
        metadata: Mapping[str, str] | None = None,
    ) -> ArtifactRef:
        data = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return self.put_bytes(key, data, content_type="application/json", metadata=metadata)

    def get_bytes(self, ref: ArtifactRef) -> bytes:
        kwargs: dict[str, Any] = {"Bucket": ref.bucket, "Key": ref.key}
        if ref.version_id:
            kwargs["VersionId"] = ref.version_id
        response = self._client_or_create().get_object(**kwargs)
        body = response["Body"]
        data = body.read() if hasattr(body, "read") else bytes(body)
        actual = self.sha256(data)
        if actual != ref.sha256:
            raise ArtifactIntegrityError(
                f"Artifact hash mismatch for {ref.version_ref}: expected {ref.sha256}, got {actual}"
            )
        return data

    @staticmethod
    def _required_version(value: object, *, context: str = "artifact") -> str:
        if not isinstance(value, str) or not value.strip() or value.strip().lower() == "null":
            raise ArtifactIntegrityError(f"{context} requires an immutable VersionId")
        return value.strip()

    @staticmethod
    def _required_digest(value: object, *, context: str = "artifact") -> str:
        if not isinstance(value, str) or fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ArtifactIntegrityError(f"{context} requires a lowercase SHA-256 digest")
        return value

    @staticmethod
    def _required_size(value: object, *, context: str = "artifact") -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ArtifactIntegrityError(f"{context} requires a non-negative expected size")
        return value

    @staticmethod
    def _metadata(response: Mapping[str, Any]) -> Mapping[str, Any]:
        metadata = response.get("Metadata")
        return metadata if isinstance(metadata, Mapping) else {}

    @staticmethod
    def _metadata_value(metadata: Mapping[str, Any], name: str) -> object:
        for key, value in metadata.items():
            if str(key).lower() == name.lower():
                return value
        return None

    @staticmethod
    def _key_in_prefix(key: str, prefix: str) -> bool:
        clean_prefix = _validate_s3_path(prefix, name="allowed prefix", allow_empty=True)
        clean_key = _validate_s3_path(key, name="artifact key")
        return (
            not clean_prefix
            or clean_key == clean_prefix
            or clean_key.startswith(f"{clean_prefix}/")
        )

    def _exact_object(self, ref: ArtifactRef) -> bytes:
        version_id = self._required_version(ref.version_id)
        response = self._client_or_create().get_object(
            Bucket=ref.bucket,
            Key=ref.key,
            VersionId=version_id,
        )
        response_mapping = dict(response)
        response_version = self._required_version(
            response_mapping.get("VersionId"), context="downloaded artifact"
        )
        if response_version != version_id:
            raise ArtifactIntegrityError(
                "downloaded artifact VersionId does not match the requested immutable version"
            )
        body = response_mapping.get("Body")
        if body is None:
            raise ArtifactIntegrityError("S3 object response did not contain a body")
        data = body.read() if hasattr(body, "read") else bytes(body)
        if not isinstance(data, bytes):
            data = bytes(data)
        return data

    def verify_immutable(
        self,
        ref: ArtifactRef,
        *,
        expected_sha256: str | None = None,
        expected_size_bytes: int | None = None,
        allowed_bucket: str | None = None,
        allowed_prefix: str | None = None,
    ) -> ArtifactRef:
        """Verify one exact S3 object version before it enters live evidence.

        Every expected property is checked against both S3 metadata and the
        downloaded bytes.  The version and location checks happen before any
        S3 operation so an untrusted reference cannot be used to probe another
        bucket or mutable current object.
        """

        version_id = self._required_version(ref.version_id)
        digest = self._required_digest(
            ref.sha256 if expected_sha256 is None else expected_sha256
        )
        size_bytes = self._required_size(
            ref.size_bytes if expected_size_bytes is None else expected_size_bytes
        )
        bucket = self.bucket if allowed_bucket is None else allowed_bucket
        if not isinstance(bucket, str) or not bucket.strip() or ref.bucket != bucket:
            raise ArtifactIntegrityError("artifact is outside the allowed bucket")
        prefix = self.prefix if allowed_prefix is None else allowed_prefix
        if not isinstance(prefix, str) or not self._key_in_prefix(ref.key, prefix):
            raise ArtifactIntegrityError("artifact is outside the allowed prefix")

        head = dict(
            self._client_or_create().head_object(
                Bucket=ref.bucket,
                Key=ref.key,
                VersionId=version_id,
            )
        )
        head_version = self._required_version(head.get("VersionId"), context="S3 head")
        if head_version != version_id:
            raise ArtifactIntegrityError("S3 head VersionId does not match the requested version")
        metadata = self._metadata(head)
        metadata_digest = self._metadata_value(metadata, "sha256")
        if metadata_digest != digest:
            raise ArtifactIntegrityError("S3 metadata SHA-256 does not match the expected digest")
        observed_size = self._required_size(
            head.get("ContentLength"), context="S3 head ContentLength"
        )
        if observed_size != size_bytes:
            raise ArtifactIntegrityError("S3 metadata size does not match the expected size")

        data = self._exact_object(ref)
        if len(data) != size_bytes:
            raise ArtifactIntegrityError("downloaded bytes size does not match the expected size")
        actual_digest = self.sha256(data)
        if actual_digest != digest:
            raise ArtifactIntegrityError(
                "downloaded bytes SHA-256 does not match the expected digest"
            )
        return ArtifactRef(
            bucket=ref.bucket,
            key=ref.key,
            sha256=digest,
            size_bytes=size_bytes,
            version_id=version_id,
            content_type=(
                str(head.get("ContentType")) if head.get("ContentType") else ref.content_type
            ),
            etag=(str(head.get("ETag")) if head.get("ETag") else ref.etag),
        )

    # Keep a short name for supervisor/provider callers while retaining the
    # explicit name for code review and audit logs.
    verify = verify_immutable

    def canonicalize_sagemaker_output(
        self,
        output_uri: str,
        *,
        retained_prefix: str,
        allowed_source_bucket: str,
        allowed_source_prefix: str,
        expected_sha256: str | None = None,
        expected_size_bytes: int | None = None,
    ) -> ArtifactRef:
        """Copy a SageMaker output archive into a retained content-addressed object.

        SageMaker output paths are mutable and their user metadata is not used
        as the checkpoint identity.  The current version is pinned, the bytes
        are downloaded, and a digest-derived key is uploaded to this store.
        The retained upload must itself be versioned and is verified end-to-end.
        """

        parsed = urlparse(output_uri)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
            raise ArtifactIntegrityError(f"Not an S3 output archive URI: {output_uri!r}")
        source_key = _validate_s3_path(
            unquote(parsed.path.lstrip("/")), name="SageMaker output key"
        )
        if not isinstance(allowed_source_bucket, str) or not allowed_source_bucket.strip():
            raise ArtifactIntegrityError("allowed source bucket must not be empty")
        if parsed.netloc != allowed_source_bucket:
            raise ArtifactIntegrityError("SageMaker output is outside the allowed source bucket")
        allowed_source_prefix = _validate_s3_path(
            allowed_source_prefix, name="allowed source prefix"
        )
        if not self._key_in_prefix(source_key, allowed_source_prefix):
            raise ArtifactIntegrityError("SageMaker output is outside the allowed source prefix")
        clean_prefix = _validate_s3_path(retained_prefix, name="retained prefix")
        source_versions = parse_qs(parsed.query, keep_blank_values=True).get("versionId", [])
        if len(source_versions) > 1 or (source_versions and not source_versions[0]):
            raise ArtifactIntegrityError("SageMaker output URI has an invalid VersionId")
        source_version = source_versions[0] if source_versions else None
        head_kwargs: dict[str, Any] = {"Bucket": parsed.netloc, "Key": source_key}
        if source_version:
            head_kwargs["VersionId"] = self._required_version(source_version, context="output")
        head = dict(self._client_or_create().head_object(**head_kwargs))
        resolved_version = self._required_version(head.get("VersionId"), context="SageMaker output")
        if source_version and resolved_version != source_version:
            raise ArtifactIntegrityError(
                "S3 head VersionId does not match the requested source VersionId"
            )
        source_ref = ArtifactRef(
            bucket=parsed.netloc,
            key=source_key,
            sha256="0" * 64,
            size_bytes=self._required_size(
                head.get("ContentLength"), context="SageMaker output ContentLength"
            ),
            version_id=resolved_version,
        )
        data = self._exact_object(source_ref)
        digest = self.sha256(data)
        if expected_sha256 is not None and digest != self._required_digest(expected_sha256):
            raise ArtifactIntegrityError(
                "downloaded output archive SHA-256 does not match expected digest"
            )
        if expected_size_bytes is not None and len(data) != self._required_size(
            expected_size_bytes
        ):
            raise ArtifactIntegrityError(
                "downloaded output archive size does not match expected size"
            )
        retained_key = f"{clean_prefix}/{digest}.tar.gz"
        try:
            retained = self.put_bytes(
                retained_key,
                data,
                content_type="application/gzip",
                metadata={
                    "source-uri": output_uri,
                    "source-version-id": resolved_version,
                    "sha256": digest,
                },
            )
        except ArtifactIntegrityError as exc:
            raise ArtifactIntegrityError(f"retained artifact upload failed: {exc}") from exc
        self._required_version(retained.version_id, context="retained artifact")
        return self.verify_immutable(
            retained,
            expected_sha256=digest,
            expected_size_bytes=len(data),
        )

    # Descriptive aliases for supervisor code that calls this operation
    # retention or output canonicalization.
    retain_sagemaker_output = canonicalize_sagemaker_output
    canonicalize_output_archive = canonicalize_sagemaker_output

    def get_json(self, ref: ArtifactRef) -> Any:
        return json.loads(self.get_bytes(ref).decode("utf-8"))

    def head(self, ref: ArtifactRef) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"Bucket": ref.bucket, "Key": ref.key}
        if ref.version_id:
            kwargs["VersionId"] = ref.version_id
        return dict(self._client_or_create().head_object(**kwargs))


# Descriptive alias for callers that call all cloud integrations providers.
S3ArtifactProvider = S3ArtifactStore
