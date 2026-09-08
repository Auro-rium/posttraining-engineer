"""SageMaker entrypoint for independent, sealed objective evaluation.

The evaluator is the only worker that accepts a ``sealed`` channel.  It never
serializes hidden task definitions or model responses: output is an aggregate
report bound to the supplied AgentEval manifest and verified checkpoint
artifacts.  Missing inputs and unverifiable artifacts fail before a report can
be written.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.objective.engine import ServiceRecoveryEngine
from app.objective.models import ObjectiveSplit, Task, ToolCall

EVALUATION_SUITE = "AgentGym/AgentEval"
EVALUATION_SUITE_VERSION = "agent-eval-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")
_CHANNELS = frozenset({"candidate", "champion", "sealed"})


class EvaluationWorkerError(ValueError):
    """The sealed evaluation contract or artifact verification failed."""


class EvaluationArtifactError(EvaluationWorkerError):
    """A checkpoint or sealed evaluation artifact is absent or invalid."""


@dataclass(frozen=True, slots=True)
class EvaluationInputs:
    candidate_dir: Path
    sealed_dir: Path
    output_dir: Path
    run_id: str
    experiment_id: str
    evaluation_manifest_sha256: str
    evaluation_suite_version: str
    objective_seed: int
    champion_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    task_count: int
    successful_tasks: int

    @property
    def success_rate(self) -> float:
        return self.successful_tasks / self.task_count if self.task_count else 0.0


def _required(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise EvaluationWorkerError(f"{name} is required")
    return value


def _path(value: str | Path, name: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.exists() or not candidate.is_dir():
        raise EvaluationWorkerError(f"{name} must be an existing directory")
    return candidate.resolve()


def _digest(value: str, name: str) -> str:
    if not _SHA256.fullmatch(value):
        raise EvaluationWorkerError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _channels(
    values: Mapping[str, str | Path],
) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for raw_name, raw_path in values.items():
        name = str(raw_name).strip().lower()
        if name not in _CHANNELS:
            raise EvaluationWorkerError(
                f"evaluator channel {raw_name!r} is not allowed; use candidate, champion, or sealed"
            )
        parsed[name] = _path(raw_path, f"{name} channel")
    if "candidate" not in parsed:
        raise EvaluationWorkerError("candidate channel is required")
    if "sealed" not in parsed:
        raise EvaluationWorkerError("sealed channel is required")
    return parsed


def parse_evaluation_inputs(
    env: Mapping[str, str] | None = None,
    channels: Mapping[str, str | Path] | None = None,
) -> EvaluationInputs:
    """Parse a strict evaluator contract and permit no train-side channel."""

    values = dict(os.environ if env is None else env)
    channel_values = dict(channels or {})
    for env_name, env_value in values.items():
        if env_name.startswith("SM_CHANNEL_") and env_value.strip():
            channel_name = env_name.removeprefix("SM_CHANNEL_").lower()
            if channel_name not in _CHANNELS:
                raise EvaluationWorkerError(
                    f"evaluator cannot consume train-side or unknown channel {env_name!r}"
                )
    for env_key, name in (
        ("SM_CHANNEL_CANDIDATE", "candidate"),
        ("SM_CHANNEL_CHAMPION", "champion"),
        ("SM_CHANNEL_SEALED", "sealed"),
    ):
        if env_key in values and values[env_key].strip():
            channel_values.setdefault(name, values[env_key])
    parsed_channels = _channels(channel_values)
    output_text = values.get("SM_OUTPUT_DATA_DIR", values.get("SM_MODEL_DIR", "")).strip()
    if not output_text:
        raise EvaluationWorkerError("SM_OUTPUT_DATA_DIR is required")
    output_dir = Path(output_text).expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise EvaluationWorkerError("SM_OUTPUT_DATA_DIR must be a directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    suite_version = _required(values, "EVALUATION_SUITE_VERSION")
    if suite_version != EVALUATION_SUITE_VERSION:
        raise EvaluationWorkerError(
            f"EVALUATION_SUITE_VERSION must equal {EVALUATION_SUITE_VERSION!r}"
        )
    try:
        seed = int(_required(values, "OBJECTIVE_SEED"))
    except ValueError as exc:
        raise EvaluationWorkerError("OBJECTIVE_SEED must be an integer") from exc
    return EvaluationInputs(
        candidate_dir=parsed_channels["candidate"],
        sealed_dir=parsed_channels["sealed"],
        output_dir=output_dir,
        run_id=_required(values, "RUN_ID"),
        experiment_id=_required(values, "EXPERIMENT_ID"),
        evaluation_manifest_sha256=_digest(
            _required(values, "EVALUATION_MANIFEST_SHA256"), "EVALUATION_MANIFEST_SHA256"
        ),
        evaluation_suite_version=suite_version,
        objective_seed=seed,
        champion_dir=parsed_channels.get("champion"),
    )


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise EvaluationArtifactError(f"checkpoint artifact file is absent: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_digest(files: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(list(files), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify_checkpoint_artifact(checkpoint_dir: Path) -> Mapping[str, Any]:
    """Verify a trainer manifest and every referenced checkpoint file."""

    directory = _path(checkpoint_dir, "checkpoint")
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise EvaluationArtifactError("checkpoint manifest.json is absent")
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationArtifactError("checkpoint manifest is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise EvaluationArtifactError("checkpoint manifest must be a JSON object")
    manifest_digest = payload.get("manifest_sha256")
    artifact_digest = payload.get("artifact_sha256")
    files = payload.get("artifact_files")
    if not isinstance(manifest_digest, str) or not _SHA256.fullmatch(manifest_digest):
        raise EvaluationArtifactError("checkpoint manifest_sha256 is absent or invalid")
    if not isinstance(artifact_digest, str) or not _SHA256.fullmatch(artifact_digest):
        raise EvaluationArtifactError("checkpoint artifact_sha256 is absent or invalid")
    if not isinstance(files, list) or not files:
        raise EvaluationArtifactError("checkpoint artifact_files are absent")
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    expected_manifest = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected_manifest != manifest_digest:
        raise EvaluationArtifactError("checkpoint manifest checksum does not match content")
    checked: list[dict[str, Any]] = []
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise EvaluationArtifactError("checkpoint manifest contains an invalid file entry")
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.name == "manifest.json":
            raise EvaluationArtifactError("checkpoint manifest contains an unsafe file path")
        file_path = directory / relative
        try:
            file_path.resolve().relative_to(directory)
        except ValueError as exc:
            raise EvaluationArtifactError(
                f"checkpoint manifest file escapes checkpoint directory: {relative}"
            ) from exc
        digest = _file_sha256(file_path)
        if digest != entry.get("sha256") or file_path.stat().st_size != entry.get("size_bytes"):
            raise EvaluationArtifactError(f"checkpoint checksum mismatch for {relative}")
        checked.append(
            {
                "path": relative.as_posix(),
                "size_bytes": file_path.stat().st_size,
                "sha256": digest,
            }
        )
    if _artifact_digest(checked) != artifact_digest:
        raise EvaluationArtifactError("checkpoint artifact checksum does not match content")
    return payload


def _sealed_manifest(inputs: EvaluationInputs) -> tuple[Mapping[str, Any], list[str]]:
    manifest_path = inputs.sealed_dir / "manifest.json"
    if not manifest_path.is_file():
        raise EvaluationArtifactError("sealed evaluation manifest.json is absent")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationArtifactError("sealed evaluation manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise EvaluationArtifactError("sealed evaluation manifest must be a JSON object")
    if manifest.get("manifest_sha256") != inputs.evaluation_manifest_sha256:
        raise EvaluationArtifactError("sealed evaluation manifest digest does not match job input")
    if manifest.get("suite_version") != EVALUATION_SUITE_VERSION:
        raise EvaluationArtifactError("sealed evaluation suite version does not match job input")
    task_file = next(
        (
            item
            for item in (inputs.sealed_dir / "tasks.json", inputs.sealed_dir / "sealed.json")
            if item.is_file()
        ),
        None,
    )
    if task_file is None:
        raise EvaluationArtifactError("sealed evaluation task artifact is absent")
    try:
        raw = json.loads(task_file.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationArtifactError("sealed evaluation task artifact is not valid JSON") from exc
    entries = raw.get("tasks") if isinstance(raw, dict) else raw
    if not isinstance(entries, list) or not entries:
        raise EvaluationArtifactError("sealed evaluation contains no tasks")
    task_ids = []
    for entry in entries:
        task_id = entry.get("task_id") if isinstance(entry, dict) else entry
        if not isinstance(task_id, str) or not task_id.strip():
            raise EvaluationArtifactError("sealed evaluation contains an invalid task identifier")
        task_ids.append(task_id)
    return manifest, task_ids


def _decode_actions(text: str) -> tuple[ToolCall, ...]:
    """Parse only allow-listed JSON tool calls from model output."""

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return ()
    values = value if isinstance(value, list) else [value]
    actions: list[ToolCall] = []
    for item in values:
        if not isinstance(item, dict) or not isinstance(item.get("tool"), str):
            continue
        try:
            actions.append(ToolCall(tool=item["tool"], arguments=item.get("arguments", {})))
        except Exception:
            continue
    return tuple(actions)


def _model_policy(
    checkpoint_dir: Path, manifest: Mapping[str, Any]
) -> Callable[[Task], tuple[ToolCall, ...]]:
    try:
        # Heavy dependencies are loaded only while evaluating a real checkpoint.
        from peft import PeftModel  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise EvaluationWorkerError(
            "Transformers/PEFT evaluation dependencies are unavailable"
        ) from exc
    model_id = manifest.get("base_model_id")
    revision = manifest.get("base_model_revision")
    if (
        model_id != "google/functiongemma-270m-it"
        or not isinstance(revision, str)
        or not _REVISION.fullmatch(revision)
    ):
        raise EvaluationWorkerError(
            "checkpoint manifest does not pin the FunctionGemma base revision"
        )
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=False)
        base = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, trust_remote_code=False
        )
        model = PeftModel.from_pretrained(base, checkpoint_dir)
        model.eval()
    except Exception as exc:
        raise EvaluationWorkerError(
            f"real checkpoint loading failed: {type(exc).__name__}"
        ) from exc

    def policy(task: Task) -> tuple[ToolCall, ...]:
        prompt = json.dumps(
            {
                "objective": task.objective,
                "service": task.service_name,
                "allowed_tools": task.allowed_tools,
                "output": "JSON array of {tool, arguments} objects",
            },
            sort_keys=True,
        )
        try:
            encoded = tokenizer(prompt, return_tensors="pt")
            generated = model.generate(**encoded, max_new_tokens=256)
            text = tokenizer.decode(generated[0], skip_special_tokens=True)
        except Exception:
            return ()
        return _decode_actions(text)

    return policy


def evaluate_checkpoint(
    checkpoint_dir: Path,
    *,
    sealed_task_ids: Sequence[str],
    objective_seed: int,
    policy: Callable[[Task], Sequence[ToolCall]] | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> EvaluationMetrics:
    """Evaluate one verified checkpoint against task IDs without leaking tasks."""

    checkpoint_manifest = verify_checkpoint_artifact(checkpoint_dir)
    selected_policy = policy or _model_policy(checkpoint_dir, checkpoint_manifest)
    successes = 0
    for task_id in sealed_task_ids:
        engine = ServiceRecoveryEngine(seed=objective_seed, sealed=True)
        task = engine.reset(split=ObjectiveSplit.HIDDEN, task_id=task_id)
        actions = tuple(selected_policy(task))[: task.max_steps]
        result = engine.run_episode(task_id, actions, split=ObjectiveSplit.HIDDEN)
        successes += int(result.success)
    return EvaluationMetrics(task_count=len(sealed_task_ids), successful_tasks=successes)


def build_evaluation_report(
    inputs: EvaluationInputs,
    *,
    candidate_metrics: EvaluationMetrics,
    candidate_manifest: Mapping[str, Any],
    champion_metrics: EvaluationMetrics | None = None,
    champion_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build aggregate-only report after artifact-backed evaluation."""

    payload: dict[str, Any] = {
        "schema_version": "evaluation-report-v1",
        "suite": EVALUATION_SUITE,
        "suite_version": inputs.evaluation_suite_version,
        "evaluation_manifest_sha256": inputs.evaluation_manifest_sha256,
        "run_id": inputs.run_id,
        "experiment_id": inputs.experiment_id,
        "candidate_manifest_sha256": candidate_manifest["manifest_sha256"],
        "candidate_artifact_sha256": candidate_manifest["artifact_sha256"],
        "candidate_task_count": candidate_metrics.task_count,
        "candidate_successful_tasks": candidate_metrics.successful_tasks,
        "candidate_success_rate": candidate_metrics.success_rate,
    }
    if champion_metrics is not None:
        if champion_manifest is None:
            raise EvaluationWorkerError("champion manifest is required with champion metrics")
        payload.update(
            {
                "champion_manifest_sha256": champion_manifest["manifest_sha256"],
                "champion_artifact_sha256": champion_manifest["artifact_sha256"],
                "champion_task_count": champion_metrics.task_count,
                "champion_successful_tasks": champion_metrics.successful_tasks,
                "champion_success_rate": champion_metrics.success_rate,
            }
        )
    return payload


def run_evaluation(
    inputs: EvaluationInputs,
    *,
    policy: Callable[[Task], Sequence[ToolCall]] | None = None,
) -> Path:
    """Run sealed evaluation and write a report only after all checks pass."""

    candidate_manifest = verify_checkpoint_artifact(inputs.candidate_dir)
    sealed_manifest, task_ids = _sealed_manifest(inputs)
    candidate_metrics = evaluate_checkpoint(
        inputs.candidate_dir,
        sealed_task_ids=task_ids,
        objective_seed=inputs.objective_seed,
        policy=policy,
        manifest=sealed_manifest,
    )
    champion_metrics: EvaluationMetrics | None = None
    champion_manifest: Mapping[str, Any] | None = None
    if inputs.champion_dir is not None:
        champion_manifest = verify_checkpoint_artifact(inputs.champion_dir)
        champion_metrics = evaluate_checkpoint(
            inputs.champion_dir,
            sealed_task_ids=task_ids,
            objective_seed=inputs.objective_seed,
            policy=policy,
            manifest=sealed_manifest,
        )
    payload = build_evaluation_report(
        inputs,
        candidate_metrics=candidate_metrics,
        candidate_manifest=candidate_manifest,
        champion_metrics=champion_metrics,
        champion_manifest=champion_manifest,
    )
    return write_evaluation_report(inputs.output_dir, payload)


def write_evaluation_report(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    """Persist an aggregate report with a deterministic self-checksum."""

    destination = Path(output_dir).expanduser().resolve()
    if not destination.exists() or not destination.is_dir():
        raise EvaluationArtifactError("evaluation output directory is absent")
    unsigned = dict(payload)
    unsigned.pop("report_sha256", None)
    unsigned["report_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output = destination / "evaluation.json"
    output.write_text(json.dumps(unsigned, sort_keys=True, separators=(",", ":")) + "\n")
    return output


evaluate = run_evaluation
parse_evaluator_inputs = parse_evaluation_inputs
verify_artifact = verify_checkpoint_artifact


def main() -> None:
    inputs = parse_evaluation_inputs()
    report_path = run_evaluation(inputs)
    print(json.dumps(json.loads(report_path.read_text()), sort_keys=True))


if __name__ == "__main__":
    main()
