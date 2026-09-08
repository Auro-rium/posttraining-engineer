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
from typing import Any, Protocol
from urllib.parse import parse_qs, urlparse


class OptionalDependencyError(RuntimeError):
    """Raised when an AWS-backed adapter is used without its optional SDK."""


class ArtifactIntegrityError(ValueError):
    """Raised when downloaded bytes do not match their recorded digest."""


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
            return f"{self.uri}?versionId={self.version_id}"
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
        self.prefix = prefix.strip("/")
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
        clean = key.strip("/")
        if not clean:
            raise ValueError("artifact key must not be empty")
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
        return ArtifactRef(
            bucket=self.bucket,
            key=kwargs["Key"],
            sha256=digest,
            size_bytes=len(data),
            version_id=(str(response["VersionId"]) if response.get("VersionId") else None),
            content_type=content_type,
            etag=(str(response["ETag"]) if response.get("ETag") else None),
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

    def get_json(self, ref: ArtifactRef) -> Any:
        return json.loads(self.get_bytes(ref).decode("utf-8"))

    def head(self, ref: ArtifactRef) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"Bucket": ref.bucket, "Key": ref.key}
        if ref.version_id:
            kwargs["VersionId"] = ref.version_id
        return dict(self._client_or_create().head_object(**kwargs))


# Descriptive alias for callers that call all cloud integrations providers.
S3ArtifactProvider = S3ArtifactStore
