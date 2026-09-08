from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import cast

import pytest

from app.providers.artifacts import (
    ArtifactIntegrityError,
    ArtifactRef,
    S3ArtifactStore,
)


class _Body:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def read(self) -> bytes:
        return self.data


class _VersionedS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str, str], tuple[bytes, dict[str, str]]] = {}
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.next_version = 1

    def add(
        self,
        bucket: str,
        key: str,
        data: bytes,
        *,
        version_id: str,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        self.objects[(bucket, key, version_id)] = (data, dict(metadata or {}))

    def head_object(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("head_object", kwargs))
        version_id = str(kwargs.get("VersionId", ""))
        if not version_id:
            candidates = [
                version
                for bucket, key, version in self.objects
                if bucket == str(kwargs["Bucket"]) and key == str(kwargs["Key"])
            ]
            version_id = candidates[-1]
        data, metadata = self.objects[(str(kwargs["Bucket"]), str(kwargs["Key"]), version_id)]
        return {
            "Metadata": metadata,
            "ContentLength": len(data),
            "VersionId": version_id,
            "ETag": '"etag"',
            "ContentType": "application/gzip",
        }

    def get_object(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("get_object", kwargs))
        version_id = str(kwargs.get("VersionId", ""))
        data, metadata = self.objects[(str(kwargs["Bucket"]), str(kwargs["Key"]), version_id)]
        return {
            "Body": _Body(data),
            "Metadata": metadata,
            "ContentLength": len(data),
            "VersionId": version_id,
        }

    def put_object(self, **kwargs: object) -> dict[str, str]:
        self.calls.append(("put_object", kwargs))
        version_id = f"retained-v{self.next_version}"
        self.next_version += 1
        data = bytes(cast(bytes, kwargs["Body"]))
        metadata_value = cast(Mapping[object, object], kwargs["Metadata"])
        metadata = {str(k): str(v) for k, v in metadata_value.items()}
        self.add(
            str(kwargs["Bucket"]),
            str(kwargs["Key"]),
            data,
            version_id=version_id,
            metadata=metadata,
        )
        return {"VersionId": version_id, "ETag": '"retained"'}


def _ref(
    data: bytes,
    *,
    bucket: str = "artifacts",
    key: str = "runs/run-1/model.tar.gz",
) -> ArtifactRef:
    return ArtifactRef(
        bucket=bucket,
        key=key,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        version_id="source-v1",
    )


def test_verify_immutable_checks_version_metadata_location_size_and_download() -> None:
    data = b"checkpoint archive"
    client = _VersionedS3()
    client.add(
        "artifacts",
        "runs/run-1/model.tar.gz",
        data,
        version_id="source-v1",
        metadata={"sha256": hashlib.sha256(data).hexdigest()},
    )
    store = S3ArtifactStore("artifacts", client=client, prefix="runs")

    verified = store.verify_immutable(
        _ref(data),
        expected_sha256=hashlib.sha256(data).hexdigest(),
        expected_size_bytes=len(data),
    )

    assert verified.version_id == "source-v1"
    assert verified.sha256 == hashlib.sha256(data).hexdigest()
    assert verified.size_bytes == len(data)
    assert client.calls == [
        (
            "head_object",
            {
                "Bucket": "artifacts",
                "Key": "runs/run-1/model.tar.gz",
                "VersionId": "source-v1",
            },
        ),
        (
            "get_object",
            {
                "Bucket": "artifacts",
                "Key": "runs/run-1/model.tar.gz",
                "VersionId": "source-v1",
            },
        ),
    ]


@pytest.mark.parametrize(
    ("metadata", "expected_size", "expected_message"),
    [
        ({"sha256": "0" * 64}, 18, "metadata SHA-256"),
        ({"sha256": hashlib.sha256(b"checkpoint archive").hexdigest()}, 17, "size"),
        ({}, 18, "metadata SHA-256"),
    ],
)
def test_verify_immutable_rejects_untrusted_head_metadata(
    metadata: dict[str, str], expected_size: int, expected_message: str
) -> None:
    data = b"checkpoint archive"
    client = _VersionedS3()
    client.add(
        "artifacts",
        "runs/run-1/model.tar.gz",
        data,
        version_id="source-v1",
        metadata=metadata,
    )
    store = S3ArtifactStore("artifacts", client=client, prefix="runs")

    with pytest.raises(ArtifactIntegrityError, match=expected_message):
        store.verify_immutable(
            _ref(data),
            expected_sha256=hashlib.sha256(data).hexdigest(),
            expected_size_bytes=expected_size,
        )


def test_verify_immutable_rejects_unversioned_or_out_of_scope_reference_before_download() -> None:
    data = b"checkpoint archive"
    client = _VersionedS3()
    store = S3ArtifactStore("artifacts", client=client, prefix="runs")

    with pytest.raises(ArtifactIntegrityError, match="VersionId"):
        store.verify_immutable(
            ArtifactRef(
                bucket="artifacts",
                key="runs/run-1/model.tar.gz",
                sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
            ),
            expected_sha256=hashlib.sha256(data).hexdigest(),
            expected_size_bytes=len(data),
        )
    with pytest.raises(ArtifactIntegrityError, match="allowed bucket"):
        store.verify_immutable(
            _ref(data, bucket="other"),
            expected_sha256=hashlib.sha256(data).hexdigest(),
            expected_size_bytes=len(data),
        )
    assert client.calls == []


def test_verify_immutable_rejects_download_digest_mismatch() -> None:
    expected = b"checkpoint archive"
    actual = b"tampered archive!!"
    client = _VersionedS3()
    client.add(
        "artifacts",
        "runs/run-1/model.tar.gz",
        actual,
        version_id="source-v1",
        metadata={"sha256": hashlib.sha256(expected).hexdigest()},
    )
    store = S3ArtifactStore("artifacts", client=client, prefix="runs")

    with pytest.raises(ArtifactIntegrityError, match="downloaded bytes"):
        store.verify_immutable(
            _ref(expected),
            expected_sha256=hashlib.sha256(expected).hexdigest(),
            expected_size_bytes=len(expected),
        )


def test_canonicalize_sagemaker_output_retains_downloaded_bytes_by_content_hash() -> None:
    data = b"sagemaker output archive"
    digest = hashlib.sha256(data).hexdigest()
    client = _VersionedS3()
    client.add(
        "artifacts",
        "jobs/run-1/model.tar.gz",
        data,
        version_id="mutable-output-v7",
        # This metadata is deliberately not used as the retained checkpoint hash.
        metadata={"sha256": "f" * 64},
    )
    store = S3ArtifactStore("artifacts", client=client, prefix="retained")

    retained = store.canonicalize_sagemaker_output(
        "s3://artifacts/jobs/run-1/model.tar.gz",
        retained_prefix="checkpoints/run-1",
        allowed_source_bucket="artifacts",
        allowed_source_prefix="jobs/run-1",
    )

    assert retained.bucket == "artifacts"
    assert retained.key == f"retained/checkpoints/run-1/{digest}.tar.gz"
    assert retained.sha256 == digest
    assert retained.size_bytes == len(data)
    assert retained.version_id == "retained-v1"
    put_calls = [kwargs for name, kwargs in client.calls if name == "put_object"]
    assert len(put_calls) == 1
    assert put_calls[0]["Body"] == data
    metadata = cast(Mapping[object, object], put_calls[0]["Metadata"])
    assert metadata["sha256"] == digest


def test_canonicalize_sagemaker_output_requires_retained_version() -> None:
    data = b"sagemaker output archive"

    class UnversionedRetention(_VersionedS3):
        def put_object(self, **kwargs: object) -> dict[str, str]:
            self.calls.append(("put_object", kwargs))
            return {}

    client = UnversionedRetention()
    client.add(
        "artifacts",
        "jobs/run-1/model.tar.gz",
        data,
        version_id="mutable-output-v7",
    )
    store = S3ArtifactStore("artifacts", client=client, prefix="retained")

    with pytest.raises(ArtifactIntegrityError, match=r"retained artifact.*VersionId"):
        store.canonicalize_sagemaker_output(
            "s3://artifacts/jobs/run-1/model.tar.gz",
            retained_prefix="checkpoints/run-1",
            allowed_source_bucket="artifacts",
            allowed_source_prefix="jobs/run-1",
        )


def test_canonicalize_rejects_source_version_rebinding() -> None:
    data = b"sagemaker output archive"

    class RebindingHead(_VersionedS3):
        def head_object(self, **kwargs: object) -> dict[str, object]:
            response = super().head_object(**kwargs)
            response["VersionId"] = "source-v2"
            return response

    client = RebindingHead()
    client.add("artifacts", "jobs/run-1/model.tar.gz", data, version_id="source-v1")
    store = S3ArtifactStore("artifacts", client=client, prefix="retained")

    with pytest.raises(ArtifactIntegrityError, match="requested source VersionId"):
        store.canonicalize_sagemaker_output(
            "s3://artifacts/jobs/run-1/model.tar.gz?versionId=source-v1",
            retained_prefix="checkpoints/run-1",
            allowed_source_bucket="artifacts",
            allowed_source_prefix="jobs/run-1",
        )


@pytest.mark.parametrize(
    "output_uri",
    [
        "s3://artifacts/jobs/run-1/model.tar.gz?versionId=source-v1&versionId=source-v2",
        "s3://artifacts/jobs/run-1/model.tar.gz?versionId=",
    ],
)
def test_canonicalize_requires_explicit_source_location_and_valid_uri_version(
    output_uri: str,
) -> None:
    store = S3ArtifactStore("artifacts", client=_VersionedS3(), prefix="retained")

    with pytest.raises(ArtifactIntegrityError, match=r"VersionId|source bucket"):
        store.canonicalize_sagemaker_output(
            output_uri,
            retained_prefix="checkpoints/run-1",
            allowed_source_bucket="artifacts",
            allowed_source_prefix="jobs/run-1",
        )

    with pytest.raises(TypeError):
        store.canonicalize_sagemaker_output(
            "s3://artifacts/jobs/run-1/model.tar.gz",
            retained_prefix="checkpoints/run-1",
        )  # type: ignore[call-arg]


def test_live_uri_parser_requires_one_nonblank_version_id() -> None:
    data = b"archive"
    for uri in (
        "s3://artifacts/checkpoints/model.tar.gz",
        "s3://artifacts/checkpoints/model.tar.gz?versionId=",
        "s3://artifacts/checkpoints/model.tar.gz?versionId=v1&versionId=v2",
        "s3://artifacts/checkpoints/model.tar.gz?versionId=null",
    ):
        with pytest.raises(ValueError, match="one immutable VersionId"):
            ArtifactRef.from_live_uri(
                uri,
                sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
            )


@pytest.mark.parametrize("value", ["runs/../model", "runs//model", "runs/./model", "runs/model\n"])
def test_artifact_paths_reject_dot_empty_and_control_segments(value: str) -> None:
    store = S3ArtifactStore("artifacts", client=_VersionedS3())
    with pytest.raises(ArtifactIntegrityError, match="path"):
        store.put_bytes(value, b"bytes")

    with pytest.raises(ArtifactIntegrityError, match="path"):
        store.canonicalize_sagemaker_output(
            "s3://artifacts/jobs/run-1/model.tar.gz",
            retained_prefix=value,
            allowed_source_bucket="artifacts",
            allowed_source_prefix="jobs/run-1",
        )


@pytest.mark.parametrize("invalid_length", [True, -1])
def test_verify_rejects_boolean_or_negative_content_length(invalid_length: object) -> None:
    data = b"checkpoint archive"

    class InvalidLengthHead(_VersionedS3):
        content_length: object = invalid_length

        def head_object(self, **kwargs: object) -> dict[str, object]:
            response = super().head_object(**kwargs)
            response["ContentLength"] = self.content_length
            return response

    client = InvalidLengthHead()
    client.add(
        "artifacts",
        "runs/run-1/model.tar.gz",
        data,
        version_id="source-v1",
        metadata={"sha256": hashlib.sha256(data).hexdigest()},
    )
    store = S3ArtifactStore("artifacts", client=client, prefix="runs")
    with pytest.raises(ArtifactIntegrityError, match="ContentLength"):
        store.verify_immutable(
            _ref(data),
            expected_sha256=hashlib.sha256(data).hexdigest(),
            expected_size_bytes=len(data),
        )
