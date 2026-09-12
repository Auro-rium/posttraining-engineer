"""SageMaker entrypoint for independent, sealed objective evaluation.

The evaluator is the only worker that accepts a ``sealed`` channel.  It never
serializes hidden task definitions or model responses: output is an aggregate
report bound to the supplied AgentEval manifest and verified checkpoint
artifacts.  Missing inputs and unverifiable artifacts fail before a report can
be written.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import re
import tarfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from app.objective.engine import ServiceRecoveryEngine
from app.objective.models import ObjectiveSplit, Task, ToolCall

EVALUATION_SUITE = "AgentGym/AgentEval"
EVALUATION_SUITE_VERSION = "agent-eval-v1"
FUNCTION_START = "<start_function_call>"
FUNCTION_END = "<end_function_call>"
FUNCTION_ESCAPE = "<escape>"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")
_CHANNELS = frozenset({"candidate", "champion", "sealed"})


class EvaluationWorkerError(ValueError):
    """The sealed evaluation contract or artifact verification failed."""


class EvaluationArtifactError(EvaluationWorkerError):
    """A checkpoint or sealed evaluation artifact is absent or invalid."""


class EvaluationRuntimeError(EvaluationWorkerError):
    """The model/runtime/environment prevented a trustworthy evaluation."""


class InvalidModelAction(EvaluationWorkerError):
    """A deterministic model output violated the tool-call protocol."""


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
    task_successes: tuple[bool, ...] = ()
    task_environments: tuple[str, ...] = ()
    invalid_action_tasks: int = 0

    def __post_init__(self) -> None:
        if self.task_count < 0 or self.successful_tasks < 0:
            raise EvaluationWorkerError("evaluation counts cannot be negative")
        if self.successful_tasks > self.task_count:
            raise EvaluationWorkerError("successful task count cannot exceed task count")
        if self.invalid_action_tasks < 0 or self.invalid_action_tasks > self.task_count:
            raise EvaluationWorkerError("invalid action count is outside task count")
        if self.task_successes and len(self.task_successes) != self.task_count:
            raise EvaluationWorkerError("evaluation task outcomes do not match task count")
        if self.task_successes and sum(self.task_successes) != self.successful_tasks:
            raise EvaluationWorkerError("evaluation task outcomes do not match success count")
        if self.task_environments and len(self.task_environments) != self.task_count:
            raise EvaluationWorkerError("evaluation task environments do not match task count")
        if self.task_environments and not self.task_successes:
            raise EvaluationWorkerError("evaluation task outcomes are required for environments")

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


def _extract_checkpoint_channel(
    channel_dir: Path, *, name: str, expected_sha256: str, destination: Path
) -> Path:
    archives = tuple(channel_dir.glob("*.tar.gz"))
    if (
        not _SHA256.fullmatch(expected_sha256)
        or len(archives) != 1
        or archives[0].name != f"{expected_sha256}.tar.gz"
        or _file_sha256(archives[0]) != expected_sha256
    ):
        raise EvaluationArtifactError(f"{name} checkpoint archive is not content-addressed")
    root = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    total_size = 0
    file_count = 0
    try:
        with tarfile.open(archives[0], mode="r:gz") as archive:
            for member in archive.getmembers():
                if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                    raise EvaluationArtifactError(
                        "checkpoint archive contains a link or special file"
                    )
                if member.size < 0 or member.size > 2 * 1024**3:
                    raise EvaluationArtifactError("checkpoint archive member is oversized")
                parts = tuple(
                    part for part in PurePosixPath(member.name).parts if part not in {"", "."}
                )
                if not parts or PurePosixPath(member.name).is_absolute() or ".." in parts:
                    raise EvaluationArtifactError("checkpoint archive contains an unsafe path")
                target = destination.joinpath(*parts)
                try:
                    target.resolve().relative_to(root)
                except ValueError as exc:
                    raise EvaluationArtifactError("checkpoint archive escapes its channel") from exc
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise EvaluationArtifactError(
                        "checkpoint archive contains an unsupported entry"
                    )
                file_count += 1
                total_size += member.size
                if file_count > 20_000 or total_size > 4 * 1024**3:
                    raise EvaluationArtifactError("checkpoint archive exceeds extraction limits")
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise EvaluationArtifactError("checkpoint archive member is unreadable")
                with source, target.open("xb") as output:
                    remaining = member.size
                    while remaining:
                        chunk = source.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise EvaluationArtifactError("checkpoint archive member is truncated")
                        output.write(chunk)
                        remaining -= len(chunk)
    except EvaluationArtifactError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise EvaluationArtifactError("checkpoint archive is invalid") from exc
    if file_count == 0:
        raise EvaluationArtifactError("checkpoint archive is empty")
    return destination


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
    for name in ("candidate", "champion"):
        channel = parsed_channels.get(name)
        if channel is None or (channel / "manifest.json").is_file():
            continue
        if not tuple(channel.glob("*.tar.gz")):
            continue
        expected = _required(values, f"{name.upper()}_ARCHIVE_SHA256")
        parsed_channels[name] = _extract_checkpoint_channel(
            channel,
            name=name,
            expected_sha256=expected,
            destination=output_dir / f"_{name}_checkpoint",
        )
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


def _verify_trained_adapter(directory: Path, payload: Mapping[str, Any]) -> None:
    """Reject self-consistent manifests that describe logs rather than a trained PEFT adapter."""

    qlora_config = payload.get("qlora_config")
    if not isinstance(qlora_config, Mapping):
        raise EvaluationArtifactError("checkpoint QLoRA config is absent")
    try:
        from app.autonomous.agents import validate_qlora_config

        expected = validate_qlora_config(qlora_config).model_dump(mode="json")
    except Exception as exc:
        raise EvaluationArtifactError(
            "checkpoint QLoRA config is outside the fixed search space"
        ) from exc
    config_path = directory / "adapter_config.json"
    if not config_path.is_file() or config_path.is_symlink():
        raise EvaluationArtifactError("checkpoint adapter_config.json is absent")
    try:
        adapter_config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationArtifactError("checkpoint adapter_config.json is not valid JSON") from exc
    if not isinstance(adapter_config, dict):
        raise EvaluationArtifactError("checkpoint adapter_config.json must be a JSON object")
    target_modules = adapter_config.get("target_modules")
    if not isinstance(target_modules, (list, tuple, set)) or any(
        not isinstance(module, str) for module in target_modules
    ):
        raise EvaluationArtifactError("checkpoint adapter target modules are invalid")
    if (
        adapter_config.get("base_model_name_or_path") != "google/functiongemma-270m-it"
        or adapter_config.get("peft_type") != "LORA"
        or adapter_config.get("task_type") != "CAUSAL_LM"
        or adapter_config.get("r") != expected["rank"]
        or adapter_config.get("lora_alpha") != expected["alpha"]
        or adapter_config.get("lora_dropout") != expected["dropout"]
        or set(target_modules) != set(expected["target_modules"])
    ):
        raise EvaluationArtifactError("checkpoint adapter config does not match its QLoRA manifest")
    weights = tuple(
        path
        for suffix in (".safetensors", ".bin")
        for path in directory.glob(f"adapter_model*{suffix}")
        if path.is_file() and not path.is_symlink() and path.stat().st_size > 0
    )
    if not weights:
        raise EvaluationArtifactError("checkpoint contains no non-empty adapter weights")
    metrics = payload.get("training_metrics")
    loss = metrics.get("train_loss") if isinstance(metrics, Mapping) else None
    if (
        not isinstance(metrics, Mapping)
        or not metrics
        or not isinstance(loss, (int, float))
        or isinstance(loss, bool)
        or not math.isfinite(float(loss))
    ):
        raise EvaluationArtifactError("checkpoint training metrics lack a finite train_loss")
    metrics_path = directory / "training_metrics.json"
    if not metrics_path.is_file() or metrics_path.is_symlink():
        raise EvaluationArtifactError("checkpoint training_metrics.json is absent")
    try:
        persisted_metrics = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationArtifactError("checkpoint training_metrics.json is not valid JSON") from exc
    if persisted_metrics != dict(metrics):
        raise EvaluationArtifactError("checkpoint metrics file does not match its manifest")


def verify_checkpoint_artifact(
    checkpoint_dir: Path,
    *,
    run_id: str | None = None,
    experiment_id: str | None = None,
) -> Mapping[str, Any]:
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
    if payload.get("kind") != "qlora-adapter":
        raise EvaluationArtifactError("checkpoint kind is not a QLoRA adapter")
    if payload.get("base_model_id") != "google/functiongemma-270m-it":
        raise EvaluationArtifactError("checkpoint base model identity is not FunctionGemma")
    if not isinstance(payload.get("base_model_revision"), str) or not _REVISION.fullmatch(
        payload["base_model_revision"]
    ):
        raise EvaluationArtifactError("checkpoint base model revision is absent or mutable")
    if payload.get("artifact_id") != f"checkpoint://{artifact_digest}":
        raise EvaluationArtifactError("checkpoint artifact ID is not content-bound")
    if run_id is not None and payload.get("run_id") != run_id:
        raise EvaluationArtifactError("checkpoint run identity does not match evaluation input")
    if experiment_id is not None and payload.get("experiment_id") != experiment_id:
        raise EvaluationArtifactError(
            "checkpoint experiment identity does not match evaluation input"
        )
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    expected_manifest = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected_manifest != manifest_digest:
        raise EvaluationArtifactError("checkpoint manifest checksum does not match content")
    checked: list[dict[str, Any]] = []
    listed_paths: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise EvaluationArtifactError("checkpoint manifest contains an invalid file entry")
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.name == "manifest.json":
            raise EvaluationArtifactError("checkpoint manifest contains an unsafe file path")
        canonical_path = relative.as_posix()
        if canonical_path in listed_paths:
            raise EvaluationArtifactError("checkpoint manifest contains duplicate file paths")
        listed_paths.add(canonical_path)
        file_path = directory / relative
        if file_path.is_symlink():
            raise EvaluationArtifactError("checkpoint manifest contains a symlink")
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
                "path": canonical_path,
                "size_bytes": file_path.stat().st_size,
                "sha256": digest,
            }
        )
    if _artifact_digest(checked) != artifact_digest:
        raise EvaluationArtifactError("checkpoint artifact checksum does not match content")
    all_entries = list(directory.rglob("*"))
    if any(item.is_symlink() for item in all_entries):
        raise EvaluationArtifactError("checkpoint directory contains a symlink")
    actual_paths = {
        item.relative_to(directory).as_posix()
        for item in all_entries
        if item.is_file() and item.name != "manifest.json"
    }
    if actual_paths != listed_paths:
        raise EvaluationArtifactError("checkpoint manifest is not complete for directory contents")
    _verify_trained_adapter(directory, payload)
    return payload


def _sealed_manifest(inputs: EvaluationInputs) -> tuple[Mapping[str, Any], list[str]]:
    manifest_path = inputs.sealed_dir / "manifest.json"
    if not manifest_path.is_file():
        raise EvaluationArtifactError("sealed evaluation manifest.json is absent")
    if manifest_path.is_symlink():
        raise EvaluationArtifactError("sealed evaluation manifest must not be a symlink")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationArtifactError("sealed evaluation manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise EvaluationArtifactError("sealed evaluation manifest must be a JSON object")
    if manifest.get("manifest_sha256") != inputs.evaluation_manifest_sha256:
        raise EvaluationArtifactError("sealed evaluation manifest digest does not match job input")
    if manifest.get("run_id") != inputs.run_id:
        raise EvaluationArtifactError("sealed evaluation run identity does not match job input")
    if manifest.get("experiment_id") != inputs.experiment_id:
        raise EvaluationArtifactError(
            "sealed evaluation experiment identity does not match job input"
        )
    if manifest.get("objective_seed") != inputs.objective_seed:
        raise EvaluationArtifactError("sealed evaluation seed does not match job input")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    expected_manifest = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected_manifest != inputs.evaluation_manifest_sha256:
        raise EvaluationArtifactError("sealed evaluation manifest checksum does not match content")
    if manifest.get("suite") != EVALUATION_SUITE:
        raise EvaluationArtifactError("sealed evaluation suite does not match job input")
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
    if task_file.is_symlink():
        raise EvaluationArtifactError("sealed evaluation task artifact must not be a symlink")
    task_digest = _file_sha256(task_file)
    if manifest.get("task_bundle_sha256") != task_digest:
        raise EvaluationArtifactError("sealed task bundle checksum does not match manifest")
    try:
        raw = json.loads(task_file.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationArtifactError("sealed evaluation task artifact is not valid JSON") from exc
    entries = raw.get("tasks") if isinstance(raw, dict) else raw
    if not isinstance(entries, list) or not entries:
        raise EvaluationArtifactError("sealed evaluation contains no tasks")
    if manifest.get("task_count") != len(entries):
        raise EvaluationArtifactError("sealed task count does not match manifest")
    if len(entries) > 10_000:
        raise EvaluationArtifactError("sealed evaluation task count exceeds the worker bound")
    task_ids = []
    for entry in entries:
        task_id = entry.get("task_id") if isinstance(entry, dict) else entry
        if not isinstance(task_id, str) or not task_id.strip():
            raise EvaluationArtifactError("sealed evaluation contains an invalid task identifier")
        task_ids.append(task_id)
    if len(set(task_ids)) != len(task_ids):
        raise EvaluationArtifactError("sealed evaluation contains duplicate task identifiers")
    return manifest, task_ids


def render_action_prompt(task: Task, observations: Sequence[Mapping[str, Any]] = ()) -> str:
    """Render the same safe policy prompt shape used by trainer SFT examples."""

    return json.dumps(
        {
            "task_id": task.task_id,
            "objective": task.objective,
            "service": task.service_name,
            "observations": [dict(item) for item in observations],
            "output": "FunctionGemma function call",
            "messages": _canonical_messages(task, observations),
            "tools": function_tool_schemas(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def function_tool_schemas() -> list[dict[str, Any]]:
    """Return the objective tools in FunctionGemma's schema format."""

    from app.objective.models import ALLOWED_TOOLS

    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Service-recovery tool: {name}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "additionalProperties": True,
                },
            },
        }
        for name in ALLOWED_TOOLS
    ]


def _canonical_messages(
    task: Task, observations: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Build the canonical FunctionGemma chat/tool input for one turn."""

    messages: list[dict[str, Any]] = [
        {
            "role": "developer",
            "content": "Use the provided service-recovery functions one call at a time.",
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task_id": task.task_id,
                    "objective": task.objective,
                    "service": task.service_name,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    ]
    messages.extend(
        {
            "role": "tool",
            "content": json.dumps(dict(observation), sort_keys=True, separators=(",", ":")),
        }
        for observation in observations
    )
    return messages


def _parse_function_call(text: str) -> ToolCall:
    """Parse FunctionGemma's ``<start_function_call>call:...`` format."""

    start = text.find(FUNCTION_START)
    end = text.find(FUNCTION_END, start + len(FUNCTION_START))
    if start < 0 or end < 0:
        raise InvalidModelAction("model output is not a FunctionGemma function call")
    value = text[start + len(FUNCTION_START) : end].strip()
    if not value.startswith("call:") or "{" not in value or not value.endswith("}"):
        raise InvalidModelAction("malformed FunctionGemma function call")
    name, body = value[5:].split("{", 1)
    name = name.strip()
    body = body[:-1].strip()
    if not name:
        raise InvalidModelAction("function call name is empty")
    arguments: dict[str, Any] = {}
    if body:
        cursor = 0
        while cursor < len(body):
            separator = body.find(":", cursor)
            if separator <= cursor:
                raise InvalidModelAction("malformed FunctionGemma arguments")
            key = body[cursor:separator].strip()
            cursor = separator + 1
            if body.startswith(FUNCTION_ESCAPE, cursor):
                begin = cursor + len(FUNCTION_ESCAPE)
                finish = body.find(FUNCTION_ESCAPE, begin)
                if finish < 0:
                    raise InvalidModelAction("unterminated escaped function argument")
                arguments[key] = body[begin:finish]
                cursor = finish + len(FUNCTION_ESCAPE)
            else:
                next_separator = body.find(",", cursor)
                raw = body[cursor:] if next_separator < 0 else body[cursor:next_separator]
                raw = raw.strip()
                try:
                    arguments[key] = json.loads(raw)
                except json.JSONDecodeError:
                    arguments[key] = raw
                cursor = len(body) if next_separator < 0 else next_separator + 1
            while cursor < len(body) and body[cursor] in " ,":
                cursor += 1
    try:
        return ToolCall(tool=name, arguments=arguments)
    except Exception as exc:
        raise InvalidModelAction("FunctionGemma emitted an unknown or invalid tool") from exc


def _decode_actions(text: str, *, allow_json: bool = False) -> tuple[ToolCall, ...]:
    """Parse one allow-listed JSON tool call from model output.

    Unknown tools are a contract violation, not an unsuccessful action that
    can be silently dropped.  The policy wrapper turns the violation into a
    failed task without crossing the sealed boundary.
    """

    if FUNCTION_START in text:
        return (_parse_function_call(text),)
    if not allow_json:
        raise InvalidModelAction("model output lacks the FunctionGemma function-call marker")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidModelAction("model output is not a supported function call") from exc
    values = value if isinstance(value, list) else [value]
    if not values:
        raise InvalidModelAction("model output contains no function call")
    actions: list[ToolCall] = []
    for item in values:
        if not isinstance(item, dict) or not isinstance(item.get("tool"), str):
            raise InvalidModelAction("model output contains an invalid tool call")
        try:
            actions.append(ToolCall(tool=item["tool"], arguments=item.get("arguments", {})))
        except Exception as exc:
            raise InvalidModelAction("model output contains an unknown or invalid tool") from exc
    return tuple(actions)


def _model_policy(
    checkpoint_dir: Path, manifest: Mapping[str, Any]
) -> Callable[[Task, Sequence[Mapping[str, Any]]], tuple[ToolCall, ...]]:
    try:
        # Heavy dependencies are loaded only while evaluating a real checkpoint.
        from peft import PeftModel  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoProcessor,
        )
    except ImportError as exc:
        raise EvaluationRuntimeError(
            "Transformers/PEFT evaluation dependencies are unavailable"
        ) from exc
    model_id = manifest.get("base_model_id")
    revision = manifest.get("base_model_revision")
    if (
        model_id != "google/functiongemma-270m-it"
        or not isinstance(revision, str)
        or not _REVISION.fullmatch(revision)
    ):
        raise EvaluationRuntimeError(
            "checkpoint manifest does not pin the FunctionGemma base revision"
        )
    try:
        processor = AutoProcessor.from_pretrained(checkpoint_dir, trust_remote_code=False)
        base = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, trust_remote_code=False
        )
        model = PeftModel.from_pretrained(base, checkpoint_dir)
        model.eval()
    except Exception as exc:
        raise EvaluationRuntimeError(
            f"real checkpoint loading failed: {type(exc).__name__}"
        ) from exc

    def policy(
        task: Task, observations: Sequence[Mapping[str, Any]] = ()
    ) -> tuple[ToolCall, ...]:
        try:
            encoded = processor.apply_chat_template(
                _canonical_messages(task, observations),
                tools=function_tool_schemas(),
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_ids = encoded.get("input_ids")
            if input_ids is None:
                raise EvaluationRuntimeError("model tokenizer returned no input IDs")
            device = getattr(model, "device", None)
            if device is None:
                device = next(model.parameters()).device
            encoded = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
            generated = model.generate(**encoded, max_new_tokens=256)
            completion = generated[0, input_ids.shape[-1] :]
            text = processor.decode(completion, skip_special_tokens=False)
        except InvalidModelAction:
            raise
        except EvaluationRuntimeError:
            raise
        except Exception as exc:
            raise EvaluationRuntimeError(
                f"model generation/runtime failed: {type(exc).__name__}"
            ) from exc
        return _decode_actions(text)[:1]

    return policy


def evaluate_checkpoint(
    checkpoint_dir: Path,
    *,
    sealed_task_ids: Sequence[str],
    objective_seed: int,
    policy: Callable[..., ToolCall | Sequence[ToolCall]] | None = None,
    manifest: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    experiment_id: str | None = None,
) -> EvaluationMetrics:
    """Evaluate one verified checkpoint against task IDs without leaking tasks."""

    checkpoint_manifest = verify_checkpoint_artifact(
        checkpoint_dir, run_id=run_id, experiment_id=experiment_id
    )
    selected_policy = policy or _model_policy(checkpoint_dir, checkpoint_manifest)
    successes = 0
    task_successes: list[bool] = []
    task_environments: list[str] = []
    invalid_action_tasks = 0

    def invoke(
        selected: Callable[..., ToolCall | Sequence[ToolCall]],
        task: Task,
        observations: Sequence[Mapping[str, Any]],
    ) -> ToolCall | Sequence[ToolCall]:
        # Existing coordinator test adapters accepted ``policy(task)``.  New
        # model policies receive the observation history at every step.  Use
        # signature inspection to preserve the old adapter without masking a
        # real TypeError raised inside a policy implementation.
        try:
            parameters = inspect.signature(selected).parameters.values()
            accepts_observations = any(
                parameter.kind is parameter.VAR_POSITIONAL
                or parameter.kind is parameter.VAR_KEYWORD
                for parameter in parameters
            ) or len(
                [
                    parameter
                    for parameter in parameters
                    if parameter.kind
                    in {parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD}
                ]
            ) >= 2
        except (TypeError, ValueError):
            accepts_observations = True
        return selected(task, observations) if accepts_observations else selected(task)

    for task_id in sealed_task_ids:
        engine = ServiceRecoveryEngine(seed=objective_seed, sealed=True)
        task = engine.reset(split=ObjectiveSplit.HIDDEN, task_id=task_id)
        observations: list[Mapping[str, Any]] = []
        success = False
        invalid_action = False
        for _ in range(task.max_steps):
            try:
                raw_action = invoke(selected_policy, task, tuple(observations))
                actions = (
                    (raw_action,)
                    if isinstance(raw_action, ToolCall)
                    else tuple(raw_action)
                )
            except InvalidModelAction:
                invalid_action = True
                break
            except EvaluationRuntimeError:
                raise
            except (TypeError, ValueError) as exc:
                raise EvaluationRuntimeError("policy invocation failed") from exc
            if not actions:
                invalid_action = True
                break
            # A policy emits one action per observation turn.  Extra actions
            # are rejected instead of being executed with hidden future state.
            if len(actions) != 1:
                invalid_action = True
                break
            if not isinstance(actions[0], ToolCall):
                invalid_action = True
                break
            try:
                action = ToolCall(tool=actions[0].tool, arguments=dict(actions[0].arguments))
                step = engine.step(action.tool, dict(action.arguments))
            except (TypeError, ValueError):
                invalid_action = True
                break
            observations.append(step.observation)
            if step.done:
                success = bool(step.success and step.reward > 0)
                break
        successes += int(success)
        invalid_action_tasks += int(invalid_action)
        task_successes.append(success)
        task_environments.append(task.service_name)
    return EvaluationMetrics(
        task_count=len(sealed_task_ids),
        successful_tasks=successes,
        task_successes=tuple(task_successes),
        task_environments=tuple(task_environments),
        invalid_action_tasks=invalid_action_tasks,
    )


def build_evaluation_report(
    inputs: EvaluationInputs,
    *,
    candidate_metrics: EvaluationMetrics,
    candidate_manifest: Mapping[str, Any],
    champion_metrics: EvaluationMetrics | None = None,
    champion_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build aggregate-only report after artifact-backed evaluation."""

    def environment_aggregates(metrics: EvaluationMetrics) -> dict[str, dict[str, Any]]:
        if not metrics.task_environments:
            return {}
        grouped: dict[str, list[bool]] = {}
        for environment, success in zip(
            metrics.task_environments, metrics.task_successes, strict=True
        ):
            grouped.setdefault(environment, []).append(success)
        return {
            environment: {
                "task_count": len(results),
                "successful_tasks": sum(results),
                "success_rate": sum(results) / len(results),
            }
            for environment, results in sorted(grouped.items())
        }

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
        "candidate_invalid_action_tasks": candidate_metrics.invalid_action_tasks,
        "candidate_metrics": {
            "task_count": candidate_metrics.task_count,
            "successful_tasks": candidate_metrics.successful_tasks,
            "success_rate": candidate_metrics.success_rate,
            "by_environment": environment_aggregates(candidate_metrics),
            "invalid_action_tasks": candidate_metrics.invalid_action_tasks,
        },
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
                "champion_invalid_action_tasks": champion_metrics.invalid_action_tasks,
                "champion_metrics": {
                    "task_count": champion_metrics.task_count,
                    "successful_tasks": champion_metrics.successful_tasks,
                    "success_rate": champion_metrics.success_rate,
                    "by_environment": environment_aggregates(champion_metrics),
                    "invalid_action_tasks": champion_metrics.invalid_action_tasks,
                },
            }
        )
        if (
            not candidate_metrics.task_successes
            or not champion_metrics.task_successes
            or len(candidate_metrics.task_successes)
            != len(champion_metrics.task_successes)
        ):
            raise EvaluationWorkerError("paired candidate/champion task outcomes are required")
        if (
            candidate_metrics.task_environments
            and champion_metrics.task_environments
            and candidate_metrics.task_environments != champion_metrics.task_environments
        ):
            raise EvaluationWorkerError("paired candidate/champion environments are required")
        regressions = sum(
            champion and not candidate
            for candidate, champion in zip(
                candidate_metrics.task_successes,
                champion_metrics.task_successes,
                strict=True,
            )
        )
        improvements = sum(
            candidate and not champion
            for candidate, champion in zip(
                candidate_metrics.task_successes,
                champion_metrics.task_successes,
                strict=True,
            )
        )
        unchanged = candidate_metrics.task_count - regressions - improvements
        decision = "REGRESSED" if regressions else "IMPROVED" if improvements else "UNCHANGED"
        environment_regression: dict[str, dict[str, Any]] = {}
        if candidate_metrics.task_environments:
            paired: dict[str, list[tuple[bool, bool]]] = {}
            for environment, candidate, champion in zip(
                candidate_metrics.task_environments,
                candidate_metrics.task_successes,
                champion_metrics.task_successes,
                strict=True,
            ):
                paired.setdefault(environment, []).append((candidate, champion))
            for environment, values in sorted(paired.items()):
                env_regressions = sum(champion and not candidate for candidate, champion in values)
                env_improvements = sum(candidate and not champion for candidate, champion in values)
                environment_regression[environment] = {
                    "task_count": len(values),
                    "candidate_successful_tasks": sum(candidate for candidate, _ in values),
                    "champion_successful_tasks": sum(champion for _, champion in values),
                    "regression_count": env_regressions,
                    "improvement_count": env_improvements,
                    "decision": "REGRESSED"
                    if env_regressions
                    else "IMPROVED"
                    if env_improvements
                    else "UNCHANGED",
                }
        evidence = {
            "candidate_manifest_sha256": candidate_manifest["manifest_sha256"],
            "champion_manifest_sha256": champion_manifest["manifest_sha256"],
            "candidate_artifact_sha256": candidate_manifest["artifact_sha256"],
            "champion_artifact_sha256": champion_manifest["artifact_sha256"],
            "task_count": candidate_metrics.task_count,
            "regression_count": regressions,
            "improvement_count": improvements,
            "unchanged_count": unchanged,
            # Outcomes are the source of truth for the aggregate counts.  The
            # evaluator never emits task IDs or hidden content, but this
            # canonical vector makes the regression decision reproducible.
            "candidate_task_successes": list(candidate_metrics.task_successes),
            "champion_task_successes": list(champion_metrics.task_successes),
            "candidate_task_environments": list(candidate_metrics.task_environments),
            "champion_task_environments": list(champion_metrics.task_environments),
        }
        paired_outcomes_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "candidate": list(candidate_metrics.task_successes),
                    "champion": list(champion_metrics.task_successes),
                    "environments": list(candidate_metrics.task_environments),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        evidence["paired_outcomes_sha256"] = paired_outcomes_sha256
        payload.update(
            {
                "regression_count": regressions,
                "improvement_count": improvements,
                "unchanged_count": unchanged,
                "regression_decision": decision,
                "paired_environment_regression": environment_regression,
                "paired_outcomes_sha256": paired_outcomes_sha256,
                "regression_evidence_sha256": hashlib.sha256(
                    json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
    return payload


def run_evaluation(
    inputs: EvaluationInputs,
    *,
    policy: Callable[..., ToolCall | Sequence[ToolCall]] | None = None,
) -> Path:
    """Run sealed evaluation and write a report only after all checks pass."""

    candidate_manifest = verify_checkpoint_artifact(
        inputs.candidate_dir, run_id=inputs.run_id, experiment_id=inputs.experiment_id
    )
    sealed_manifest, task_ids = _sealed_manifest(inputs)
    candidate_metrics = evaluate_checkpoint(
        inputs.candidate_dir,
        sealed_task_ids=task_ids,
        objective_seed=inputs.objective_seed,
        policy=policy,
        manifest=sealed_manifest,
        run_id=inputs.run_id,
        experiment_id=inputs.experiment_id,
    )
    champion_metrics: EvaluationMetrics | None = None
    champion_manifest: Mapping[str, Any] | None = None
    if inputs.champion_dir is not None:
        champion_manifest = verify_checkpoint_artifact(
            inputs.champion_dir,
            run_id=inputs.run_id,
            experiment_id=inputs.experiment_id,
        )
        champion_metrics = evaluate_checkpoint(
            inputs.champion_dir,
            sealed_task_ids=task_ids,
            objective_seed=inputs.objective_seed,
            policy=policy,
            manifest=sealed_manifest,
            run_id=inputs.run_id,
            experiment_id=inputs.experiment_id,
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
