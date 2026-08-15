"""Content-addressed artifact storage for local runs and Google Cloud Storage."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from .models import ArtifactKind, ArtifactRef


class ArtifactStore(Protocol):
    async def put_bytes(
        self, *, key: str, data: bytes, kind: ArtifactKind, content_type: str
    ) -> ArtifactRef: ...

    async def get_bytes(self, ref: ArtifactRef) -> bytes: ...

    async def exists(self, ref: ArtifactRef) -> bool: ...


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_relative_key(key: str) -> Path:
    candidate = PurePosixPath(key)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ValueError("artifact key must be a non-empty relative path")
    return Path(*candidate.parts)


def _file_uri_path(uri: str) -> Path:
    """Resolve a local artifact URI outside async functions."""

    return Path(uri.removeprefix("file://")).resolve()


class LocalArtifactStore:
    """Filesystem store for development; writes are content-verified on read."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    async def put_bytes(
        self,
        *,
        key: str,
        data: bytes,
        kind: ArtifactKind,
        content_type: str = "application/octet-stream",
    ) -> ArtifactRef:
        relative = _safe_relative_key(key)
        path = (self.root / relative).resolve()
        if self.root not in path.parents:
            raise ValueError("artifact key escapes configured root")

        def write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(data)
            temporary.replace(path)

        await asyncio.to_thread(write)
        return ArtifactRef(
            kind=kind,
            uri=path.as_uri(),
            sha256=_digest(data),
            size_bytes=len(data),
            content_type=content_type,
        )

    async def put_json(self, *, key: str, value: Any, kind: ArtifactKind) -> ArtifactRef:
        data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
        return await self.put_bytes(
            key=key, data=data, kind=kind, content_type="application/json"
        )

    async def get_bytes(self, ref: ArtifactRef) -> bytes:
        prefix = "file://"
        if not ref.uri.startswith(prefix):
            raise ValueError("local store only accepts file:// artifact references")
        path = _file_uri_path(ref.uri)
        if self.root not in path.parents:
            raise ValueError("artifact reference escapes configured root")
        data = await asyncio.to_thread(path.read_bytes)
        if _digest(data) != ref.sha256 or len(data) != ref.size_bytes:
            raise ValueError("artifact integrity check failed")
        return data

    async def get_json(self, ref: ArtifactRef) -> Any:
        return json.loads((await self.get_bytes(ref)).decode())

    async def exists(self, ref: ArtifactRef) -> bool:
        if not ref.uri.startswith("file://"):
            return False
        path = _file_uri_path(ref.uri)
        return self.root in path.parents and await asyncio.to_thread(path.is_file)


class GCSArtifactStore:
    """Google Cloud Storage adapter; the SDK and credentials are resolved lazily."""

    def __init__(self, *, bucket: str, prefix: str = "", project: str | None = None) -> None:
        try:
            storage = importlib.import_module("google.cloud.storage")
        except ImportError as exc:  # pragma: no cover - optional cloud SDK
            raise RuntimeError("GCS support requires the google-cloud-storage package") from exc
        self.bucket_name = bucket
        self.prefix = prefix.strip("/")
        self._client = storage.Client(project=project)
        self._bucket = self._client.bucket(bucket)

    def _blob_name(self, key: str) -> str:
        relative = _safe_relative_key(key).as_posix()
        return f"{self.prefix}/{relative}" if self.prefix else relative

    def _blob_for_ref(self, ref: ArtifactRef) -> Any:
        expected_prefix = f"gs://{self.bucket_name}/"
        if not ref.uri.startswith(expected_prefix):
            raise ValueError("artifact reference belongs to a different GCS bucket")
        name = ref.uri.removeprefix(expected_prefix)
        if self.prefix and not name.startswith(f"{self.prefix}/"):
            raise ValueError("artifact reference escapes configured GCS prefix")
        return self._bucket.blob(name)

    async def put_bytes(
        self,
        *,
        key: str,
        data: bytes,
        kind: ArtifactKind,
        content_type: str = "application/octet-stream",
    ) -> ArtifactRef:
        name = self._blob_name(key)
        blob = self._bucket.blob(name)
        await asyncio.to_thread(blob.upload_from_string, data, content_type=content_type)
        return ArtifactRef(
            kind=kind,
            uri=f"gs://{self.bucket_name}/{name}",
            sha256=_digest(data),
            size_bytes=len(data),
            content_type=content_type,
        )

    async def put_json(self, *, key: str, value: Any, kind: ArtifactKind) -> ArtifactRef:
        data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
        return await self.put_bytes(
            key=key, data=data, kind=kind, content_type="application/json"
        )

    async def get_bytes(self, ref: ArtifactRef) -> bytes:
        blob = self._blob_for_ref(ref)
        data = bytes(await asyncio.to_thread(blob.download_as_bytes))
        if _digest(data) != ref.sha256 or len(data) != ref.size_bytes:
            raise ValueError("artifact integrity check failed")
        return data

    async def get_json(self, ref: ArtifactRef) -> Any:
        return json.loads((await self.get_bytes(ref)).decode())

    async def exists(self, ref: ArtifactRef) -> bool:
        return await asyncio.to_thread(self._blob_for_ref(ref).exists)
