"""Stage a deterministic local service-recovery evaluator ID bundle.

This creates only opaque task IDs for the in-repository service-recovery
environment.  It does not download or package an external AgentGym dataset,
and it does not upload anything to AWS or another service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

SUITE = "AgentGym/AgentEval"
SUITE_VERSION = "agent-eval-v1"
SOURCE_ENVIRONMENT = "in-repo-service-recovery-v1"
DEFAULT_TASK_COUNT = 20
DEFAULT_OBJECTIVE_SEED = 7
MAX_TASK_COUNT = 10_000


class SealedEvaluationStagingError(ValueError):
    """Raised when a safe evaluator-only bundle cannot be staged."""


def _validate_task_count(task_count: object) -> int:
    if isinstance(task_count, bool) or not isinstance(task_count, int):
        raise SealedEvaluationStagingError("task_count must be an integer")
    if not 1 <= task_count <= MAX_TASK_COUNT:
        raise SealedEvaluationStagingError(f"task_count must be between 1 and {MAX_TASK_COUNT}")
    return task_count


def _validate_objective_seed(objective_seed: object) -> int:
    if isinstance(objective_seed, bool) or not isinstance(objective_seed, int):
        raise SealedEvaluationStagingError("objective_seed must be an integer")
    return objective_seed


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_sealed_evaluation_bundle(
    *, task_count: int = DEFAULT_TASK_COUNT, objective_seed: int = DEFAULT_OBJECTIVE_SEED
) -> dict[str, bytes]:
    """Return deterministic evaluator-only artifacts without task internals."""

    count = _validate_task_count(task_count)
    seed = _validate_objective_seed(objective_seed)
    task_ids = [f"hidden-{index:03d}" for index in range(1, count + 1)]
    task_payload = {"tasks": task_ids}
    task_bytes = _canonical_json(task_payload) + b"\n"
    unsigned_manifest = {
        "objective_seed": seed,
        "source_environment": SOURCE_ENVIRONMENT,
        "suite": SUITE,
        "suite_version": SUITE_VERSION,
        "task_bundle_sha256": _sha256(task_bytes),
        "task_count": len(task_ids),
    }
    signed_manifest = {
        **unsigned_manifest,
        "manifest_sha256": _sha256(_canonical_json(unsigned_manifest)),
    }
    manifest_bytes = _canonical_json(signed_manifest) + b"\n"
    return {"tasks.json": task_bytes, "manifest.json": manifest_bytes}


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_file_atomically(directory: Path, name: str, payload: bytes) -> None:
    path = directory / name
    with path.open("xb") as artifact:
        artifact.write(payload)
        artifact.flush()
        os.fsync(artifact.fileno())


def stage_sealed_evaluation(
    output_dir: str | Path,
    *,
    task_count: int = DEFAULT_TASK_COUNT,
    objective_seed: int = DEFAULT_OBJECTIVE_SEED,
) -> dict[str, object]:
    """Atomically create a new directory and return metadata-only hashes."""

    bundle = build_sealed_evaluation_bundle(task_count=task_count, objective_seed=objective_seed)
    manifest = json.loads(bundle["manifest.json"])
    requested_path = Path(output_dir).expanduser()
    if not requested_path.name:
        raise SealedEvaluationStagingError("output_dir must name a new directory")
    if requested_path.exists() or requested_path.is_symlink():
        raise SealedEvaluationStagingError("output_dir already exists; refusing to overwrite")

    parent = requested_path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        resolved_parent = parent.resolve()
    except OSError as exc:
        raise SealedEvaluationStagingError("output parent directory is unavailable") from exc
    target = resolved_parent / requested_path.name
    if target.exists() or target.is_symlink():
        raise SealedEvaluationStagingError("output_dir already exists; refusing to overwrite")

    lock_path = resolved_parent / f".{requested_path.name}.stage.lock"
    try:
        lock_descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise SealedEvaluationStagingError(
            "another staging operation holds the output lock"
        ) from exc
    except OSError as exc:
        raise SealedEvaluationStagingError("could not acquire output staging lock") from exc

    try:
        os.close(lock_descriptor)
        if target.exists() or target.is_symlink():
            raise SealedEvaluationStagingError("output_dir already exists; refusing to overwrite")
        try:
            with tempfile.TemporaryDirectory(
                prefix=f".{requested_path.name}.stage-", dir=resolved_parent
            ) as temporary_name:
                temporary_dir = Path(temporary_name)
                for name, payload in sorted(bundle.items()):
                    _write_file_atomically(temporary_dir, name, payload)
                _fsync_directory(temporary_dir)
                os.rename(temporary_dir, target)
                _fsync_directory(resolved_parent)
        except OSError as exc:
            raise SealedEvaluationStagingError(
                "could not atomically stage evaluator artifacts"
            ) from exc
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    files = []
    for name in sorted(bundle):
        artifact_bytes = (target / name).read_bytes()
        files.append(
            {
                "name": name,
                "sha256": _sha256(artifact_bytes),
                "size_bytes": len(artifact_bytes),
            }
        )
    return {
        "status": "STAGED_LOCAL",
        "source_environment": SOURCE_ENVIRONMENT,
        "external_agentgym_assets": False,
        "output_dir": str(target),
        "task_count": manifest["task_count"],
        "objective_seed": manifest["objective_seed"],
        "manifest_sha256": manifest["manifest_sha256"],
        "files": files,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new directory for tasks.json and manifest.json",
    )
    parser.add_argument("--task-count", type=int, default=DEFAULT_TASK_COUNT)
    parser.add_argument("--objective-seed", type=int, default=DEFAULT_OBJECTIVE_SEED)
    args = parser.parse_args(argv)
    try:
        result = stage_sealed_evaluation(
            args.output_dir,
            task_count=args.task_count,
            objective_seed=args.objective_seed,
        )
    except SealedEvaluationStagingError as exc:
        print(json.dumps({"status": "FAILED", "reason": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
