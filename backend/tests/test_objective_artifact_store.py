from __future__ import annotations

import hashlib
from collections.abc import Mapping
from io import BytesIO
from typing import Any, cast

import pytest

from app.objective.artifacts import (
    ObjectiveArtifactError,
    ObjectiveArtifactIntegrityError,
    S3ObjectiveArtifactStore,
)
from app.objective.engine import ServiceRecoveryEngine
from app.objective.models import ObjectiveSplit, ToolCall, Trajectory


class _VersionedS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str, str], tuple[bytes, dict[str, str]]] = {}
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.sequence = 0

    def _find(
        self, bucket: str, key: str, version: str | None
    ) -> tuple[str, bytes, dict[str, str]]:
        if version is None:
            candidates = [
                (item_version, data, metadata)
                for (item_bucket, item_key, item_version), (data, metadata) in self.objects.items()
                if item_bucket == bucket and item_key == key
            ]
            if not candidates:
                raise KeyError((bucket, key))
            return candidates[-1]
        try:
            data, metadata = self.objects[(bucket, key, version)]
        except KeyError:
            raise KeyError((bucket, key, version)) from None
        return version, data, metadata

    def head_object(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("head_object", kwargs))
        version, data, metadata = self._find(
            str(kwargs["Bucket"]),
            str(kwargs["Key"]),
            str(kwargs["VersionId"]) if kwargs.get("VersionId") is not None else None,
        )
        return {
            "VersionId": version,
            "ContentLength": len(data),
            "Metadata": metadata,
            "ContentType": "application/json",
        }

    def get_object(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("get_object", kwargs))
        version, data, metadata = self._find(
            str(kwargs["Bucket"]), str(kwargs["Key"]), str(kwargs["VersionId"])
        )
        return {
            "VersionId": version,
            "ContentLength": len(data),
            "Metadata": metadata,
            "Body": BytesIO(data),
        }

    def put_object(self, **kwargs: object) -> dict[str, str]:
        self.calls.append(("put_object", kwargs))
        self.sequence += 1
        version = f"v{self.sequence}"
        metadata = {
            str(key): str(value)
            for key, value in cast(Mapping[Any, Any], kwargs["Metadata"]).items()
        }
        data = bytes(cast(bytes, kwargs["Body"]))
        self.objects[(str(kwargs["Bucket"]), str(kwargs["Key"]), version)] = (data, metadata)
        return {"VersionId": version}


def _trajectory(*, split: ObjectiveSplit = ObjectiveSplit.REPLAY) -> Trajectory:
    engine = ServiceRecoveryEngine(seed=7)
    return engine.verify(
        engine.run_episode(
            "replay-001",
            [
                ToolCall(tool="edit_config", arguments={"service": "api", "content": "fixed"}),
                ToolCall(tool="run_healthcheck", arguments={"service": "api"}),
            ],
            split=split,
        )
    ).trajectory


def test_put_and_get_trajectory_uses_content_addressed_versioned_bytes() -> None:
    client = _VersionedS3()
    store = S3ObjectiveArtifactStore("objective-artifacts", client=client, prefix="runs")
    trajectory = _trajectory()

    reference = store.put(trajectory)
    restored = store.get(reference.trajectory_id)

    assert restored == trajectory
    assert reference.split is ObjectiveSplit.REPLAY
    assert reference.verified is True
    trajectory_puts = [
        call
        for name, call in client.calls
        if name == "put_object" and "trajectories/" in str(call["Key"])
    ]
    assert trajectory_puts
    key = str(trajectory_puts[0]["Key"])
    payload = bytes(cast(bytes, trajectory_puts[0]["Body"]))
    assert hashlib.sha256(payload).hexdigest() in key
    gets = [call for name, call in client.calls if name == "get_object"]
    assert gets and all("VersionId" in call for call in gets)


def test_repeated_content_addressed_put_is_idempotent() -> None:
    client = _VersionedS3()
    store = S3ObjectiveArtifactStore("objective-artifacts", client=client)
    trajectory = _trajectory()

    first = store.put(trajectory)
    second = store.put(trajectory)

    assert first == second
    assert (
        len(
            [
                1
                for name, call in client.calls
                if name == "put_object" and "trajectories/" in str(call["Key"])
            ]
        )
        == 1
    )


def test_unverified_and_hidden_trajectories_cannot_cross_public_boundary() -> None:
    store = S3ObjectiveArtifactStore("objective-artifacts", client=_VersionedS3())
    trajectory = _trajectory()
    with pytest.raises(ObjectiveArtifactIntegrityError, match="verifier-confirmed"):
        store.put(trajectory.model_copy(update={"verified": False}))
    hidden = Trajectory.model_construct(
        trajectory_id="hidden-1",
        task_id="hidden-1",
        split=ObjectiveSplit.HIDDEN,
        engine_version="service-recovery-v1",
        seed=7,
        steps=(),
        total_reward=0,
        success=False,
        done=False,
        verified=True,
    )
    with pytest.raises(ObjectiveArtifactIntegrityError, match="hidden"):
        store.put(hidden)
    with pytest.raises(ObjectiveArtifactIntegrityError, match="hidden"):
        store.get_trajectory("hidden-1", split=ObjectiveSplit.HIDDEN)


def test_validation_requires_explicit_evaluator_scope() -> None:
    client = _VersionedS3()
    store = S3ObjectiveArtifactStore("objective-artifacts", client=client)
    trajectory = _trajectory(split=ObjectiveSplit.VALIDATION)
    reference = store.put_trajectory(trajectory)

    assert store.get(reference.trajectory_id) is None
    with pytest.raises(ObjectiveArtifactIntegrityError, match="explicit evaluator scope"):
        store.get_trajectory(reference.trajectory_id, split=ObjectiveSplit.VALIDATION)
    assert (
        store.get_trajectory(
            reference.trajectory_id, split=ObjectiveSplit.VALIDATION, allow_validation=True
        )
        == trajectory
    )


def test_missing_version_or_hash_mismatch_fails_closed() -> None:
    client = _VersionedS3()
    store = S3ObjectiveArtifactStore("objective-artifacts", client=client)
    trajectory = _trajectory()
    store.put(trajectory)
    trajectory_put = next(
        call
        for name, call in client.calls
        if name == "put_object" and "trajectories/" in str(call["Key"])
    )
    key = str(trajectory_put["Key"])
    version = next(
        version
        for bucket, object_key, version in client.objects
        if bucket == "objective-artifacts" and object_key == key
    )
    data, metadata = client.objects[("objective-artifacts", key, version)]
    client.objects[("objective-artifacts", key, version)] = (data + b"tamper", metadata)
    with pytest.raises(ObjectiveArtifactIntegrityError, match=r"metadata|digest or size"):
        store.get(trajectory.trajectory_id)

    class NoVersion(_VersionedS3):
        def put_object(self, **kwargs: object) -> dict[str, str]:
            super().put_object(**kwargs)
            return {}

    with pytest.raises(ObjectiveArtifactIntegrityError, match="VersionId"):
        S3ObjectiveArtifactStore("objective-artifacts", client=NoVersion()).put(trajectory)


def test_dataset_store_excludes_validation_and_hidden_rows_and_returns_exact_uri() -> None:
    client = _VersionedS3()
    store = S3ObjectiveArtifactStore("objective-artifacts", client=client, prefix="objective")
    trajectory = _trajectory()
    engine = ServiceRecoveryEngine(seed=7)
    dataset = engine.build_dataset([trajectory], run_id="run-1", experiment_id="exp-1")

    manifest = store.put_dataset(dataset)
    restored = store.get_dataset(dataset.manifest.dataset_id)

    assert restored is not None
    assert restored.manifest.dataset_id == dataset.manifest.dataset_id
    assert manifest.s3_uri.startswith("s3://objective-artifacts/objective/datasets/")
    assert "versionId=" not in dataset.manifest.s3_uri
    assert "versionId=" in manifest.s3_uri
    with pytest.raises(ObjectiveArtifactIntegrityError, match="validation and hidden"):
        store.put_dataset(
            dataset.model_copy(
                update={
                    "rows": (
                        dataset.rows[0].model_copy(update={"split": ObjectiveSplit.VALIDATION}),
                    )
                }
            )
        )


def test_bucket_prefix_and_identifier_validation_is_strict() -> None:
    with pytest.raises(ValueError, match="bucket"):
        S3ObjectiveArtifactStore("../bucket", client=_VersionedS3())
    with pytest.raises(ValueError, match="prefix"):
        S3ObjectiveArtifactStore(
            "objective-artifacts", prefix="runs/../escape", client=_VersionedS3()
        )
    store = S3ObjectiveArtifactStore("objective-artifacts", client=_VersionedS3())
    with pytest.raises(ValueError, match="trajectory_id"):
        store.get("../secret")


def test_s3_provider_errors_are_not_treated_as_missing() -> None:
    class Denied(_VersionedS3):
        def head_object(self, **kwargs: object) -> dict[str, object]:
            raise PermissionError("access denied")

    with pytest.raises(ObjectiveArtifactError, match="S3 head failed"):
        S3ObjectiveArtifactStore("objective-artifacts", client=Denied()).get("replay-001")
