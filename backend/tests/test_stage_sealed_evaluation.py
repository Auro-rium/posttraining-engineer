from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from workers.evaluator.evaluate import _sealed_manifest, parse_evaluation_inputs


def _staging_module() -> ModuleType:
    spec = importlib.util.find_spec("scripts.stage_sealed_evaluation")
    assert spec is not None, "sealed evaluation staging command has not been implemented"
    return importlib.import_module("scripts.stage_sealed_evaluation")


def _parse_staged_bundle(root: Path, *, seed: int) -> tuple[dict[str, object], list[str]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    candidate = root.parent / f"{root.name}-candidate"
    candidate.mkdir()
    inputs = parse_evaluation_inputs(
        {
            "RUN_ID": "contract-test-run",
            "EXPERIMENT_ID": "contract-test-experiment",
            "EVALUATION_MANIFEST_SHA256": manifest["manifest_sha256"],
            "EVALUATION_SUITE_VERSION": "agent-eval-v1",
            "OBJECTIVE_SEED": str(seed),
            "SM_OUTPUT_DATA_DIR": str(root.parent / f"{root.name}-worker-output"),
        },
        {"candidate": candidate, "sealed": root},
    )
    actual_manifest, task_ids = _sealed_manifest(inputs)
    return dict(actual_manifest), task_ids


def test_bundle_is_deterministic_and_matches_the_evaluator_contract(tmp_path: Path) -> None:
    staging = _staging_module()
    first = staging.stage_sealed_evaluation(tmp_path / "first", task_count=4, objective_seed=19)
    second = staging.stage_sealed_evaluation(tmp_path / "second", task_count=4, objective_seed=19)

    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    assert (first_root / "tasks.json").read_bytes() == (second_root / "tasks.json").read_bytes()
    assert (first_root / "manifest.json").read_bytes() == (
        second_root / "manifest.json"
    ).read_bytes()
    assert first["manifest_sha256"] == second["manifest_sha256"]

    manifest, task_ids = _parse_staged_bundle(first_root, seed=19)
    assert {path.name for path in first_root.iterdir()} == {"tasks.json", "manifest.json"}
    assert task_ids == ["hidden-001", "hidden-002", "hidden-003", "hidden-004"]
    assert manifest["suite"] == "AgentGym/AgentEval"
    assert manifest["suite_version"] == "agent-eval-v1"
    assert manifest["objective_seed"] == 19
    assert manifest["task_count"] == 4
    assert manifest["source_environment"] == "in-repo-service-recovery-v1"
    assert "run_id" not in manifest
    assert "experiment_id" not in manifest
    assert (
        manifest["task_bundle_sha256"]
        == hashlib.sha256((first_root / "tasks.json").read_bytes()).hexdigest()
    )
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    assert (
        manifest["manifest_sha256"]
        == hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def test_staged_artifact_contains_only_opaque_hidden_ids(tmp_path: Path) -> None:
    staging = _staging_module()
    staging.stage_sealed_evaluation(tmp_path / "sealed", task_count=3)

    root = tmp_path / "sealed"
    tasks = json.loads((root / "tasks.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert tasks == {"tasks": ["hidden-001", "hidden-002", "hidden-003"]}
    assert set(tasks) == {"tasks"}
    assert set(manifest) == {
        "manifest_sha256",
        "objective_seed",
        "source_environment",
        "suite",
        "suite_version",
        "task_bundle_sha256",
        "task_count",
    }
    serialized = json.dumps({"tasks": tasks, "manifest": manifest}).lower()
    for forbidden in ("failure_mode", "service_name", "config_error", "dependency_failure"):
        assert forbidden not in serialized


@pytest.mark.parametrize("task_count", [0, -1, 10_001, True, 2.5])
def test_invalid_task_count_fails_without_creating_output(
    tmp_path: Path, task_count: object
) -> None:
    staging = _staging_module()

    with pytest.raises(staging.SealedEvaluationStagingError, match="task_count"):
        staging.stage_sealed_evaluation(tmp_path / "invalid", task_count=task_count)

    assert not (tmp_path / "invalid").exists()


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    staging = _staging_module()
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("leave me", encoding="utf-8")

    with pytest.raises(staging.SealedEvaluationStagingError, match="already exists"):
        staging.stage_sealed_evaluation(output)

    assert sentinel.read_text(encoding="utf-8") == "leave me"


def test_cli_reports_only_hashes_and_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    staging = _staging_module()
    output = tmp_path / "cli-output"

    assert staging.main(["--output-dir", str(output), "--task-count", "2"]) == 0

    captured = capsys.readouterr().out
    report = json.loads(captured)
    assert report["status"] == "STAGED_LOCAL"
    assert report["source_environment"] == "in-repo-service-recovery-v1"
    assert report["external_agentgym_assets"] is False
    assert report["task_count"] == 2
    assert {item["name"] for item in report["files"]} == {"tasks.json", "manifest.json"}
    for item in report["files"]:
        content = (output / item["name"]).read_bytes()
        assert item["size_bytes"] == len(content)
        assert item["sha256"] == hashlib.sha256(content).hexdigest()
    assert "hidden-001" not in captured
