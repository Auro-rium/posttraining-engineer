"""Real FunctionGemma execution for the service-recovery objective worker.

The benchmark adapter has no rule-based or random fallback. Production calls
require a complete local FunctionGemma snapshot pinned by commit revision and
content digest; policy inference receives only public task/observation data.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import tarfile
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast
from urllib.parse import parse_qs, unquote, urlparse

from app.objective.engine import ServiceRecoveryEngine
from app.objective.models import (
    ALLOWED_TOOLS,
    BenchmarkExecutionResult,
    BenchmarkRequest,
    ObjectiveSplit,
    Task,
    ToolCall,
    Trajectory,
)

TARGET_MODEL_ID = "google/functiongemma-270m-it"
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BASE_MODEL_URI_ENV = "OBJECTIVE_BASE_MODEL_URI"
_BASE_MODEL_SHA256_ENV = "OBJECTIVE_BASE_MODEL_SHA256"
_FUNCTION_START = "<start_function_call>"
_FUNCTION_END = "<end_function_call>"
_FUNCTION_ESCAPE = "<escape>"
_CHECKPOINT_DIR_ENV = "OBJECTIVE_MODEL_CHECKPOINT_DIR"
_REVISION_ENV = "OBJECTIVE_MODEL_REVISION"
_SHA256_ENV = "OBJECTIVE_MODEL_SHA256"
_TOOL_ARGUMENTS: dict[str, dict[str, str]] = {
    "get_logs": {"service": "string"},
    "inspect_service": {"service": "string"},
    "read_config": {"service": "string"},
    "edit_config": {"service": "string", "content": "string"},
    "restart_service": {"service": "string"},
    "run_healthcheck": {"service": "string"},
}
_MAX_ADAPTER_ARCHIVE_BYTES = 4 * 1024**3
_MAX_ADAPTER_EXTRACTED_BYTES = 4 * 1024**3
_MAX_ADAPTER_FILES = 20_000
_LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
_OBJECTIVE_STAGE_NAMES = frozenset(
    {
        "CHECKPOINT_RESOLVE",
        "PROCESSOR_LOAD",
        "MODEL_LOAD",
        "PROMPT_RENDER",
        "MODEL_GENERATE",
        "MODEL_DECODE",
        "FUNCTION_PARSE",
        "ENVIRONMENT_STEP",
        "TRAJECTORY_VERIFY",
        "S3_PERSIST",
        "BENCHMARK_COMPLETE",
    }
)
_OBJECTIVE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_OBJECTIVE_LOGGER = logging.getLogger("app.objective.execution")
_SAFE_PARSE_FAILURE_CODES = {
    "FunctionGemma output must be text": "non_text_output",
    "FunctionGemma output must contain exactly one tool call": "missing_function_call_frame",
    "FunctionGemma output contained multiple tool calls": "multiple_tool_calls",
    "FunctionGemma output contained a malformed tool call": "malformed_call_frame",
    "FunctionGemma emitted a tool outside the allow-list": "tool_not_allowlisted",
    "FunctionGemma output contained malformed arguments": "malformed_arguments",
    "FunctionGemma output contained an invalid argument name": "invalid_argument_name",
    "FunctionGemma output repeated an argument": "repeated_argument",
    "FunctionGemma output contained an unknown argument": "unknown_argument",
    "FunctionGemma arguments must use escaped string values": "unescaped_string_value",
    "FunctionGemma output contained an unterminated value": "unterminated_string_value",
    "FunctionGemma output contained an invalid string value": "invalid_string_value",
    "FunctionGemma output contained a trailing argument separator": "trailing_argument_separator",
    "FunctionGemma output omitted a required argument": "missing_required_arguments",
}


class ObjectiveExecutionUnavailable(RuntimeError):
    """A real objective model/checkpoint could not be used safely."""


@dataclass(frozen=True)
class _ObjectiveStageTrace:
    correlation_id: str
    checkpoint_revision: str | None


_CURRENT_OBJECTIVE_TRACE: ContextVar[_ObjectiveStageTrace | None] = ContextVar(
    "current_objective_stage_trace", default=None
)


def _process_rss_bytes() -> int | None:
    """Read current Linux process RSS without introducing a telemetry dependency."""

    try:
        resident_pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        rss_bytes = resident_pages * page_size
    except (OSError, ValueError, IndexError, TypeError, AttributeError):
        return None
    return rss_bytes if rss_bytes >= 0 else None


def _emit_objective_stage(stage: str, *, started_at: float, error: BaseException | None) -> None:
    """Emit an allow-listed JSON stage record; logging failures never affect inference."""

    trace = _CURRENT_OBJECTIVE_TRACE.get()
    if trace is None:
        return
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
        elapsed_ms = 0.0
    event: dict[str, Any] = {
        "correlation_id": trace.correlation_id,
        "stage": stage,
        "duration_ms": round(elapsed_ms, 3),
        "checkpoint_revision": trace.checkpoint_revision,
        "process_rss_bytes": _process_rss_bytes(),
        "status": "failed" if error is not None else "succeeded",
    }
    if error is not None:
        event["exception_class"] = type(error).__name__
        failure_code = _SAFE_PARSE_FAILURE_CODES.get(str(error))
        if stage == "FUNCTION_PARSE" and failure_code is not None:
            event["failure_code"] = failure_code
    try:
        _OBJECTIVE_LOGGER.info(json.dumps(event, sort_keys=True, separators=(",", ":")))
    except Exception:
        # Observation is best-effort and must not change the benchmark result.
        pass


@contextmanager
def objective_stage_trace(
    correlation_id: str, checkpoint_revision: str | None
) -> Iterator[None]:
    """Bind metadata-only trace context to this request/task context."""

    safe_revision = (
        checkpoint_revision
        if isinstance(checkpoint_revision, str)
        and _OBJECTIVE_REVISION_RE.fullmatch(checkpoint_revision)
        else None
    )
    token = _CURRENT_OBJECTIVE_TRACE.set(
        _ObjectiveStageTrace(
            correlation_id=correlation_id,
            checkpoint_revision=safe_revision,
        )
    )
    try:
        yield
    finally:
        _CURRENT_OBJECTIVE_TRACE.reset(token)


@contextmanager
def _objective_stage(stage: str) -> Iterator[None]:
    if stage not in _OBJECTIVE_STAGE_NAMES:
        raise ValueError("objective stage is not allow-listed")
    started_at = time.perf_counter()
    error: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        error = exc
        raise
    finally:
        _emit_objective_stage(stage, started_at=started_at, error=error)


def _run_objective_stage(stage: str, operation: Callable[[], Any]) -> Any:
    """Run a synchronous operation under a finite, safe telemetry stage."""

    with _objective_stage(stage):
        return operation()


@dataclass(frozen=True)
class FunctionGemmaAdapterReadiness:
    """Static readiness facts; probing never loads weights or calls inference."""

    checkpoint_verified: bool
    inference_runtime_available: bool
    blockers: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.checkpoint_verified and self.inference_runtime_available and not self.blockers


def _local_inference_runtime_available() -> bool:
    """Check importability only; do not resolve remote models or load weights."""

    try:
        import safetensors
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoProcessor,
        )
    except Exception:
        return False
    return bool(
        getattr(torch, "__version__", None)
        and getattr(safetensors, "__version__", None)
        and callable(PeftModel.from_pretrained)
        and callable(AutoModelForCausalLM.from_pretrained)
        and callable(AutoProcessor.from_pretrained)
    )


class ActionPolicy(Protocol):
    def __call__(
        self,
        task: Task,
        observations: tuple[dict[str, Any], ...],
    ) -> ToolCall: ...


def checkpoint_snapshot_sha256(files: Sequence[Any]) -> str:
    """Hash the validated per-file snapshot identity in a stable order."""

    entries = [
        {
            "path": item.path,
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
        }
        for item in sorted(files, key=lambda file: file.path)
    ]
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_local_checkpoint(
    checkpoint_dir: str | Path,
    *,
    revision: str,
    expected_sha256: str,
) -> Path:
    if not _REVISION_RE.fullmatch(revision):
        raise ObjectiveExecutionUnavailable(
            "FunctionGemma checkpoint revision must be an immutable 40-character commit SHA"
        )
    if not _SHA256_RE.fullmatch(expected_sha256):
        raise ObjectiveExecutionUnavailable(
            "FunctionGemma checkpoint requires a lowercase SHA-256 snapshot digest"
        )
    path = Path(checkpoint_dir).expanduser()
    try:
        from scripts.stage_functiongemma_checkpoint import (
            CheckpointStagingError,
            validate_checkpoint_directory,
        )
    except ImportError as exc:
        raise ObjectiveExecutionUnavailable(
            "FunctionGemma checkpoint validator is unavailable in this runtime"
        ) from exc
    try:
        files = validate_checkpoint_directory(
            path,
            revision=revision,
            model_id=TARGET_MODEL_ID,
        )
    except (OSError, CheckpointStagingError) as exc:
        message = str(exc)
        if "checkpoint is incomplete" in message or message == "checkpoint directory is empty":
            reason = f"checkpoint is incomplete: {message}"
        else:
            reason = "local FunctionGemma checkpoint is absent, incomplete, or invalid"
        raise ObjectiveExecutionUnavailable(reason) from exc
    actual_sha256 = checkpoint_snapshot_sha256(files)
    if actual_sha256 != expected_sha256:
        raise ObjectiveExecutionUnavailable(
            "local FunctionGemma checkpoint digest does not match the configured snapshot SHA-256"
        )
    return path.resolve()


def _is_immutable_s3_model_uri(value: str) -> bool:
    parsed = urlparse(value)
    versions = parse_qs(parsed.query, keep_blank_values=True).get("versionId", [])
    return bool(
        parsed.scheme == "s3"
        and parsed.netloc
        and parsed.path
        and len(versions) == 1
        and versions[0].strip()
        and versions[0].strip().lower() != "null"
        and not parsed.fragment
    )


def _archive_location(model_uri: str, expected_sha256: str) -> tuple[str, str, str]:
    """Parse the exact immutable champion S3 reference and content-addressed key."""

    parsed = urlparse(model_uri)
    versions = parse_qs(parsed.query, keep_blank_values=True).get("versionId", [])
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.strip("/")
        or len(versions) != 1
        or not versions[0].strip()
        or versions[0].strip().lower() == "null"
        or parsed.fragment
    ):
        raise ObjectiveExecutionUnavailable("requested champion has no immutable S3 version")
    key = unquote(parsed.path.lstrip("/"))
    if key.rsplit("/", 1)[-1] != f"{expected_sha256}.tar.gz":
        raise ObjectiveExecutionUnavailable("requested champion archive identity is invalid")
    return parsed.netloc, key, versions[0].strip()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_adapter_archive(archive_path: Path, destination: Path) -> Path:
    """Extract a bounded regular-file-only SageMaker adapter archive."""

    root = destination.resolve()
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    total_size = 0
    file_count = 0
    seen_paths: set[str] = set()
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                    raise ObjectiveExecutionUnavailable(
                        "champion archive contains an unsupported entry"
                    )
                raw_name = member.name.rstrip("/")
                parts = raw_name.split("/")
                if (
                    not raw_name
                    or PurePosixPath(raw_name).is_absolute()
                    or any(part in {"", ".", ".."} for part in parts)
                ):
                    raise ObjectiveExecutionUnavailable("champion archive contains an unsafe path")
                normalized = PurePosixPath(*parts).as_posix()
                if normalized in seen_paths:
                    raise ObjectiveExecutionUnavailable("champion archive contains duplicate paths")
                seen_paths.add(normalized)
                output = destination.joinpath(*parts)
                try:
                    output.resolve(strict=False).relative_to(root)
                except ValueError as exc:
                    raise ObjectiveExecutionUnavailable(
                        "champion archive escapes its extraction directory"
                    ) from exc
                if member.isdir():
                    output.mkdir(mode=0o700, parents=True, exist_ok=True)
                    continue
                if not member.isfile() or member.size < 0:
                    raise ObjectiveExecutionUnavailable(
                        "champion archive contains an unsupported entry"
                    )
                file_count += 1
                total_size += member.size
                if (
                    file_count > _MAX_ADAPTER_FILES
                    or total_size > _MAX_ADAPTER_EXTRACTED_BYTES
                    or member.size > _MAX_ADAPTER_EXTRACTED_BYTES
                ):
                    raise ObjectiveExecutionUnavailable("champion archive exceeds safe limits")
                output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ObjectiveExecutionUnavailable("champion archive is unreadable")
                with source, output.open("xb") as target:
                    remaining = member.size
                    while remaining:
                        chunk = source.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ObjectiveExecutionUnavailable(
                                "champion archive contains a truncated file"
                            )
                        target.write(chunk)
                        remaining -= len(chunk)
                output.chmod(0o600)
    except ObjectiveExecutionUnavailable:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise ObjectiveExecutionUnavailable("champion archive is invalid") from exc
    if file_count == 0:
        raise ObjectiveExecutionUnavailable("champion archive is empty")
    return destination


def _adapter_manifest_files(directory: Path) -> list[dict[str, Any]]:
    entries = list(directory.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ObjectiveExecutionUnavailable("champion artifact contains a symlink")
    files = sorted(
        (path for path in entries if path.is_file() and path.name != "manifest.json"),
        key=lambda path: path.relative_to(directory).as_posix(),
    )
    if not files:
        raise ObjectiveExecutionUnavailable("champion artifact contains no files")
    return [
        {
            "path": path.relative_to(directory).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _file_digest(path),
        }
        for path in files
    ]


def _validate_adapter_config(
    directory: Path,
    *,
    qlora_config: Mapping[str, Any],
    training_metrics: Mapping[str, Any],
) -> None:
    """Check the archive is a trained FunctionGemma LoRA, not an arbitrary model."""

    allowed: dict[str, tuple[object, ...]] = {
        "rank": (8, 16, 32),
        "alpha": (16, 32, 64),
        "dropout": (0.0, 0.05, 0.1),
        "learning_rate": (1e-4, 2e-4, 5e-4),
        "epochs": (1, 2, 3),
        "sequence_length": (512, 1024),
        "batch_size": (1, 2, 4),
        "gradient_accumulation_steps": (4, 8, 16),
    }
    if set(qlora_config) != {*allowed, "target_modules"}:
        raise ObjectiveExecutionUnavailable("champion QLoRA provenance is incomplete")
    for name, values in allowed.items():
        value = qlora_config[name]
        if isinstance(value, bool) or value not in values:
            raise ObjectiveExecutionUnavailable("champion QLoRA provenance is invalid")
        if name in {"dropout", "learning_rate"} and type(value) is not float:
            raise ObjectiveExecutionUnavailable("champion QLoRA provenance is invalid")
        if name not in {"dropout", "learning_rate"} and type(value) is not int:
            raise ObjectiveExecutionUnavailable("champion QLoRA provenance is invalid")
    target_modules = qlora_config.get("target_modules")
    if (
        not isinstance(target_modules, (list, tuple))
        or tuple(target_modules) != _LORA_TARGET_MODULES
    ):
        raise ObjectiveExecutionUnavailable("champion QLoRA provenance is invalid")
    if (
        not training_metrics
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in training_metrics.values()
        )
        or not isinstance(training_metrics.get("train_loss"), (int, float))
    ):
        raise ObjectiveExecutionUnavailable("champion training provenance is invalid")
    metrics_path = directory / "training_metrics.json"
    if not metrics_path.is_file() or metrics_path.is_symlink():
        raise ObjectiveExecutionUnavailable("champion training metrics are absent")
    try:
        archived_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ObjectiveExecutionUnavailable("champion training metrics are invalid") from exc
    if archived_metrics != dict(training_metrics):
        raise ObjectiveExecutionUnavailable("champion training provenance does not match content")

    config_path = directory / "adapter_config.json"
    if not config_path.is_file() or config_path.is_symlink():
        raise ObjectiveExecutionUnavailable("champion adapter config is absent")
    try:
        adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ObjectiveExecutionUnavailable("champion adapter config is invalid") from exc
    if not isinstance(adapter_config, dict):
        raise ObjectiveExecutionUnavailable("champion adapter config is invalid")
    modules = adapter_config.get("target_modules")
    if (
        adapter_config.get("base_model_name_or_path") != TARGET_MODEL_ID
        or adapter_config.get("peft_type") != "LORA"
        or adapter_config.get("task_type") != "CAUSAL_LM"
        or adapter_config.get("r") != qlora_config["rank"]
        or adapter_config.get("lora_alpha") != qlora_config["alpha"]
        or adapter_config.get("lora_dropout") != qlora_config["dropout"]
        or not isinstance(modules, (list, tuple))
        or tuple(modules) != _LORA_TARGET_MODULES
    ):
        raise ObjectiveExecutionUnavailable(
            "champion adapter is not an approved FunctionGemma LoRA"
        )
    weight_files = tuple(
        path
        for suffix in (".safetensors", ".bin")
        for path in directory.glob(f"adapter_model*{suffix}")
        if path.is_file() and not path.is_symlink() and path.stat().st_size > 0
    )
    if not weight_files:
        raise ObjectiveExecutionUnavailable("champion adapter weights are absent")


def _validate_champion_archive(
    directory: Path,
    *,
    request_run_id: str,
    base_model_revision: str,
) -> Mapping[str, Any]:
    """Validate archive manifest, self-digests, training provenance and LoRA shape."""

    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ObjectiveExecutionUnavailable("champion provenance manifest is absent")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ObjectiveExecutionUnavailable("champion provenance manifest is invalid") from exc
    if not isinstance(payload, dict):
        raise ObjectiveExecutionUnavailable("champion provenance manifest is invalid")
    manifest_sha256 = payload.get("manifest_sha256")
    artifact_sha256 = payload.get("artifact_sha256")
    artifact_files = payload.get("artifact_files")
    if (
        payload.get("schema_version") != "trainer-manifest-v1"
        or payload.get("kind") != "qlora-adapter"
        or not isinstance(manifest_sha256, str)
        or not _SHA256_RE.fullmatch(manifest_sha256)
        or not isinstance(artifact_sha256, str)
        or not _SHA256_RE.fullmatch(artifact_sha256)
        or not isinstance(artifact_files, list)
        or not artifact_files
        or payload.get("artifact_id") != f"checkpoint://{artifact_sha256}"
        or payload.get("run_id") != request_run_id
        or not isinstance(payload.get("dataset_id"), str)
        or not payload["dataset_id"].strip()
        or not isinstance(payload.get("dataset_sha256"), str)
        or not _SHA256_RE.fullmatch(payload["dataset_sha256"])
        or payload.get("base_model_id") != TARGET_MODEL_ID
        or payload.get("base_model_revision") != base_model_revision
    ):
        raise ObjectiveExecutionUnavailable("champion provenance does not match this request")
    experiment_id = payload.get("experiment_id")
    experiment_number = (
        experiment_id.removeprefix(f"{request_run_id}-") if isinstance(experiment_id, str) else ""
    )
    if not isinstance(experiment_id, str) or not re.fullmatch(r"[1-9][0-9]*", experiment_number):
        raise ObjectiveExecutionUnavailable("champion provenance does not match this request")
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    actual_manifest_digest = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if actual_manifest_digest != manifest_sha256:
        raise ObjectiveExecutionUnavailable("champion manifest digest does not match content")
    actual_files = _adapter_manifest_files(directory)
    if actual_files != artifact_files:
        raise ObjectiveExecutionUnavailable("champion artifact files do not match provenance")
    actual_artifact_digest = hashlib.sha256(
        json.dumps(actual_files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if actual_artifact_digest != artifact_sha256:
        raise ObjectiveExecutionUnavailable("champion artifact digest does not match content")
    qlora_config = payload.get("qlora_config")
    training_metrics = payload.get("training_metrics")
    if not isinstance(qlora_config, Mapping) or not isinstance(training_metrics, Mapping):
        raise ObjectiveExecutionUnavailable("champion training provenance is incomplete")
    _validate_adapter_config(
        directory, qlora_config=qlora_config, training_metrics=training_metrics
    )
    return payload


def _tool_schemas() -> list[dict[str, Any]]:
    descriptions = {
        "get_logs": "Read recent service logs.",
        "inspect_service": "Inspect service health without changing it.",
        "read_config": "Read the service configuration.",
        "edit_config": "Update the service configuration.",
        "restart_service": "Restart the service.",
        "run_healthcheck": "Run the service health check.",
    }
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": descriptions[name],
                "parameters": {
                    "type": "object",
                    "properties": {
                        argument: {"type": kind} for argument, kind in _TOOL_ARGUMENTS[name].items()
                    },
                    "required": list(_TOOL_ARGUMENTS[name]),
                    "additionalProperties": False,
                },
            },
        }
        for name in ALLOWED_TOOLS
    ]


def _messages(task: Task, observations: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Construct the model context without verifier rewards or task internals."""

    messages = [
        {
            "role": "developer",
            # FunctionGemma's documented activation prompt is intentionally
            # exact: additional directives degrade the base model's tool-call
            # emission before task-specific fine-tuning.
            "content": "You are a model that can do function calling with the following functions",
        },
        {
            "role": "user",
            "content": (
                f"The {task.service_name} service failed its health check. {task.objective}."
            ),
        },
    ]
    # We retain only public observations, not an assistant tool-call transcript.
    # FunctionGemma's template correctly rejects an orphan ``tool`` turn, so an
    # observation is supplied as a normal user turn for the next single action.
    messages.extend(
        {
            "role": "user",
            "content": "Result of the previous action: "
            + json.dumps(dict(observation), sort_keys=True, separators=(",", ":")),
        }
        for observation in observations
    )
    return messages


def _parse_function_call(text: str) -> ToolCall:
    if not isinstance(text, str):
        raise ObjectiveExecutionUnavailable("FunctionGemma output must be text")
    value = text.strip()
    if not value.startswith(_FUNCTION_START) or not value.endswith(_FUNCTION_END):
        raise ObjectiveExecutionUnavailable(
            "FunctionGemma output must contain exactly one tool call"
        )
    frame = value[len(_FUNCTION_START) : -len(_FUNCTION_END)]
    if _FUNCTION_START in frame or _FUNCTION_END in frame:
        raise ObjectiveExecutionUnavailable("FunctionGemma output contained multiple tool calls")
    if not frame.startswith("call:") or "{" not in frame or not frame.endswith("}"):
        raise ObjectiveExecutionUnavailable("FunctionGemma output contained a malformed tool call")
    name, body = frame[5:].split("{", 1)
    name = name.strip()
    body = body[:-1]
    expected_arguments = _TOOL_ARGUMENTS.get(name)
    if expected_arguments is None:
        raise ObjectiveExecutionUnavailable("FunctionGemma emitted a tool outside the allow-list")
    arguments: dict[str, Any] = {}
    cursor = 0
    while cursor < len(body):
        while cursor < len(body) and body[cursor] == " ":
            cursor += 1
        if cursor == len(body):
            break
        separator = body.find(":", cursor)
        if separator <= cursor:
            raise ObjectiveExecutionUnavailable(
                "FunctionGemma output contained malformed arguments"
            )
        key = body[cursor:separator].strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ObjectiveExecutionUnavailable(
                "FunctionGemma output contained an invalid argument name"
            )
        if key in arguments:
            raise ObjectiveExecutionUnavailable("FunctionGemma output repeated an argument")
        if key not in expected_arguments:
            raise ObjectiveExecutionUnavailable(
                "FunctionGemma output contained an unknown argument"
            )
        cursor = separator + 1
        while cursor < len(body) and body[cursor] == " ":
            cursor += 1
        if not body.startswith(_FUNCTION_ESCAPE, cursor):
            raise ObjectiveExecutionUnavailable(
                "FunctionGemma arguments must use escaped string values"
            )
        begin = cursor + len(_FUNCTION_ESCAPE)
        finish = body.find(_FUNCTION_ESCAPE, begin)
        if finish < 0:
            raise ObjectiveExecutionUnavailable(
                "FunctionGemma output contained an unterminated value"
            )
        argument_value = body[begin:finish]
        if not argument_value.strip() or any(
            token in argument_value for token in (_FUNCTION_START, _FUNCTION_END, _FUNCTION_ESCAPE)
        ):
            raise ObjectiveExecutionUnavailable(
                "FunctionGemma output contained an invalid string value"
            )
        arguments[key] = argument_value
        cursor = finish + len(_FUNCTION_ESCAPE)
        while cursor < len(body) and body[cursor] == " ":
            cursor += 1
        if cursor < len(body):
            if body[cursor] != ",":
                raise ObjectiveExecutionUnavailable(
                    "FunctionGemma output contained malformed arguments"
                )
            cursor += 1
            if not body[cursor:].strip():
                raise ObjectiveExecutionUnavailable(
                    "FunctionGemma output contained a trailing argument separator"
                )
    if set(arguments) != set(expected_arguments):
        raise ObjectiveExecutionUnavailable("FunctionGemma output omitted a required argument")
    try:
        return ToolCall(tool=name, arguments=arguments)
    except Exception as exc:
        raise ObjectiveExecutionUnavailable(
            "FunctionGemma emitted a tool outside the allow-list"
        ) from exc


def _first_complete_function_call(text: str) -> str:
    """Return precisely the next complete function-call frame.

    FunctionGemma can continue planning after a valid tool call.  The objective
    environment is deliberately one-action-at-a-time, so only the first complete
    frame is eligible for execution; later generated text is never interpreted
    as another action.
    """

    if not isinstance(text, str) or not text.startswith(_FUNCTION_START):
        raise ObjectiveExecutionUnavailable(
            "FunctionGemma output must contain exactly one tool call"
        )
    finish = text.find(_FUNCTION_END, len(_FUNCTION_START))
    if finish < 0:
        raise ObjectiveExecutionUnavailable(
            "FunctionGemma output must contain exactly one tool call"
        )
    return text[: finish + len(_FUNCTION_END)]


class FunctionGemmaLocalPolicy:
    """Local-only FunctionGemma inference on a digest-pinned base snapshot."""

    def __init__(self, checkpoint_dir: Path, processor: Any, model: Any) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.processor = processor
        self.model = model
        self._generation_ready = False

    @property
    def generation_ready(self) -> bool:
        """True only after a local generation produced a valid tool call."""

        return self._generation_ready

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        *,
        revision: str,
        expected_sha256: str,
        adapter_dir: str | Path | None = None,
    ) -> FunctionGemmaLocalPolicy:
        path = _run_objective_stage(
            "CHECKPOINT_RESOLVE",
            lambda: _validate_local_checkpoint(
                checkpoint_dir,
                revision=revision,
                expected_sha256=expected_sha256,
            ),
        )
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoProcessor,
            )

            def load_processor() -> Any:
                return AutoProcessor.from_pretrained(
                    str(path),
                    revision=revision,
                    local_files_only=True,
                    trust_remote_code=False,
                )  # type: ignore[no-untyped-call]

            processor: Any = _run_objective_stage("PROCESSOR_LOAD", load_processor)

            def load_model() -> Any:
                model_value: Any = AutoModelForCausalLM.from_pretrained(
                    str(path),
                    revision=revision,
                    local_files_only=True,
                    trust_remote_code=False,
                )
                if adapter_dir is not None:
                    from peft import PeftModel

                    adapter_path = Path(adapter_dir).expanduser()
                    if not adapter_path.is_dir() or adapter_path.is_symlink():
                        raise ObjectiveExecutionUnavailable(
                            "verified FunctionGemma adapter directory is unavailable"
                        )
                    model_value = PeftModel.from_pretrained(
                        model_value,
                        str(adapter_path.resolve()),
                        local_files_only=True,
                        is_trainable=False,
                    )
                model_value.eval()
                return model_value

            model = _run_objective_stage("MODEL_LOAD", load_model)
        except ImportError as exc:
            raise ObjectiveExecutionUnavailable(
                "Transformers and PEFT are required for local FunctionGemma inference"
            ) from exc
        except Exception as exc:
            raise ObjectiveExecutionUnavailable(
                "local FunctionGemma inference could not load its pinned snapshot: "
                f"{type(exc).__name__}"
            ) from exc
        return cls(path, processor, model)

    def __call__(
        self,
        task: Task,
        observations: tuple[dict[str, Any], ...],
    ) -> ToolCall:
        try:
            encoded = _run_objective_stage(
                "PROMPT_RENDER",
                lambda: self.processor.apply_chat_template(
                    _messages(task, observations),
                    tools=_tool_schemas(),
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ),
            )
            input_ids = encoded.get("input_ids")
            if input_ids is None:
                raise ObjectiveExecutionUnavailable("FunctionGemma tokenizer returned no input IDs")
            device = getattr(self.model, "device", None)
            if device is None:
                device = next(self.model.parameters()).device
            encoded = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
            eos_token_id = getattr(self.processor, "eos_token_id", None)
            if eos_token_id is None:
                raise ObjectiveExecutionUnavailable(
                    "FunctionGemma processor has no configured EOS token"
                )
            generated = _run_objective_stage(
                "MODEL_GENERATE",
                lambda: self.model.generate(
                    **encoded,
                    pad_token_id=eos_token_id,
                    eos_token_id=eos_token_id,
                    max_new_tokens=128,
                    do_sample=False,
                ),
            )
            completion = generated[0, input_ids.shape[-1] :]
            text = _run_objective_stage(
                "MODEL_DECODE",
                lambda: self.processor.decode(completion, skip_special_tokens=True),
            )
            action = cast(
                ToolCall,
                _run_objective_stage(
                    "FUNCTION_PARSE",
                    lambda: _parse_function_call(_first_complete_function_call(text)),
                ),
            )
            self._generation_ready = True
            return action
        except ObjectiveExecutionUnavailable:
            raise
        except Exception as exc:
            raise ObjectiveExecutionUnavailable(
                f"local FunctionGemma inference failed: {type(exc).__name__}"
            ) from exc


class _DeferredLocalPolicy:
    """Defer the heavy model load until the first authenticated benchmark call."""

    def __init__(self, path: Path, revision: str, expected_sha256: str) -> None:
        self.path = path
        self.revision = revision
        self.expected_sha256 = expected_sha256
        self._policy: FunctionGemmaLocalPolicy | None = None

    @property
    def model_load_ready(self) -> bool:
        return self._policy is not None

    @property
    def generation_ready(self) -> bool:
        return self._policy is not None and self._policy.generation_ready

    def __call__(
        self,
        task: Task,
        observations: tuple[dict[str, Any], ...],
    ) -> ToolCall:
        if self._policy is None:
            self._policy = FunctionGemmaLocalPolicy.from_checkpoint(
                self.path,
                revision=self.revision,
                expected_sha256=self.expected_sha256,
            )
        return self._policy(task, observations)

    def readiness(self) -> FunctionGemmaAdapterReadiness:
        """Revalidate the pinned local artifact and runtime without loading it."""

        try:
            _validate_local_checkpoint(
                self.path,
                revision=self.revision,
                expected_sha256=self.expected_sha256,
            )
        except ObjectiveExecutionUnavailable:
            return FunctionGemmaAdapterReadiness(
                checkpoint_verified=False,
                inference_runtime_available=False,
                blockers=("checkpoint_unverified",),
            )
        if not _local_inference_runtime_available():
            return FunctionGemmaAdapterReadiness(
                checkpoint_verified=True,
                inference_runtime_available=False,
                blockers=("inference_runtime_unavailable",),
            )
        return FunctionGemmaAdapterReadiness(
            checkpoint_verified=True,
            inference_runtime_available=True,
            blockers=(),
        )


class FunctionGemmaBenchmarkExecutionAdapter:
    """Run the pinned base model or a per-request verified champion LoRA."""

    def __init__(
        self,
        policy: ActionPolicy,
        *,
        base_model_uri: str | None = None,
        base_model_sha256: str | None = None,
        s3_client: Any | None = None,
        aws_region: str | None = None,
    ) -> None:
        self.policy = policy
        self.base_model_uri = base_model_uri
        self.base_model_sha256 = base_model_sha256
        self._s3_client = s3_client
        self.aws_region = aws_region

    def _model_identity_ready(self) -> bool:
        return bool(
            isinstance(self.base_model_uri, str)
            and _is_immutable_s3_model_uri(self.base_model_uri)
            and isinstance(self.base_model_sha256, str)
            and _SHA256_RE.fullmatch(self.base_model_sha256)
        )

    @property
    def configuration_ready(self) -> bool:
        return self._model_identity_ready() and isinstance(self.policy, _DeferredLocalPolicy)

    @property
    def model_load_ready(self) -> bool:
        return isinstance(self.policy, _DeferredLocalPolicy) and self.policy.model_load_ready

    @property
    def generation_ready(self) -> bool:
        return isinstance(self.policy, _DeferredLocalPolicy) and self.policy.generation_ready

    def readiness(self) -> FunctionGemmaAdapterReadiness:
        """Attest only adapters created from a verified local checkpoint config."""

        if not self._model_identity_ready():
            return FunctionGemmaAdapterReadiness(
                checkpoint_verified=False,
                inference_runtime_available=False,
                blockers=("model_identity_unavailable",),
            )
        if not isinstance(self.policy, _DeferredLocalPolicy):
            return FunctionGemmaAdapterReadiness(
                checkpoint_verified=False,
                inference_runtime_available=False,
                blockers=("functiongemma_adapter_unavailable",),
            )
        return self.policy.readiness()

    def _s3_client_or_create(self) -> Any:
        if self._s3_client is not None:
            return self._s3_client
        try:
            import boto3  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - dependency is present in deployment
            raise ObjectiveExecutionUnavailable("S3 client is unavailable") from exc
        self._s3_client = boto3.client("s3", region_name=self.aws_region)
        return self._s3_client

    def _download_champion_archive(self, request: BenchmarkRequest, destination: Path) -> Path:
        if not isinstance(request.model_sha256, str) or not _SHA256_RE.fullmatch(
            request.model_sha256
        ):
            raise ObjectiveExecutionUnavailable("requested champion digest is invalid")
        bucket, key, version_id = _archive_location(request.model_uri, request.model_sha256)
        try:
            response = self._s3_client_or_create().get_object(
                Bucket=bucket,
                Key=key,
                VersionId=version_id,
            )
        except ObjectiveExecutionUnavailable:
            raise
        except Exception as exc:
            raise ObjectiveExecutionUnavailable(
                "requested champion object version could not be read"
            ) from exc
        if response.get("VersionId") != version_id:
            raise ObjectiveExecutionUnavailable(
                "S3 did not return the requested champion object version"
            )
        body = response.get("Body")
        if body is None or not callable(getattr(body, "read", None)):
            raise ObjectiveExecutionUnavailable("requested champion object body is unreadable")
        content_length = response.get("ContentLength")
        if content_length is not None and (
            type(content_length) is not int
            or content_length < 0
            or content_length > _MAX_ADAPTER_ARCHIVE_BYTES
        ):
            raise ObjectiveExecutionUnavailable("requested champion archive exceeds safe limits")
        digest = hashlib.sha256()
        size = 0
        try:
            with body, destination.open("xb") as output:
                while chunk := body.read(1024 * 1024):
                    size += len(chunk)
                    if size > _MAX_ADAPTER_ARCHIVE_BYTES:
                        raise ObjectiveExecutionUnavailable(
                            "requested champion archive exceeds safe limits"
                        )
                    digest.update(chunk)
                    output.write(chunk)
        except ObjectiveExecutionUnavailable:
            raise
        except Exception as exc:
            raise ObjectiveExecutionUnavailable(
                "requested champion archive could not be streamed"
            ) from exc
        if content_length is not None and size != content_length:
            raise ObjectiveExecutionUnavailable(
                "requested champion archive size does not match S3 metadata"
            )
        if digest.hexdigest() != request.model_sha256:
            raise ObjectiveExecutionUnavailable(
                "requested champion archive SHA-256 does not match the request"
            )
        return destination

    def _policy_for_request(self, request: BenchmarkRequest) -> ActionPolicy:
        if (
            request.model_uri == self.base_model_uri
            and request.model_sha256 == self.base_model_sha256
        ):
            return self.policy
        if not isinstance(self.policy, _DeferredLocalPolicy):
            raise ObjectiveExecutionUnavailable(
                "requested champion cannot be materialized by this objective worker"
            )
        with tempfile.TemporaryDirectory(prefix="objective-champion-") as temporary:
            scratch = Path(temporary)
            archive_path = scratch / "adapter.tar.gz"
            adapter_directory = scratch / "adapter"
            self._download_champion_archive(request, archive_path)
            _extract_adapter_archive(archive_path, adapter_directory)
            _validate_champion_archive(
                adapter_directory,
                request_run_id=request.run_id,
                base_model_revision=self.policy.revision,
            )
            return FunctionGemmaLocalPolicy.from_checkpoint(
                self.policy.path,
                revision=self.policy.revision,
                expected_sha256=self.policy.expected_sha256,
                adapter_dir=adapter_directory,
            )

    def execute_benchmark(
        self,
        request: BenchmarkRequest,
        engine: ServiceRecoveryEngine,
    ) -> BenchmarkExecutionResult:
        if request.split not in {ObjectiveSplit.TRAIN, ObjectiveSplit.REPLAY}:
            raise ObjectiveExecutionUnavailable("benchmark execution is limited to train/replay")
        if not self._model_identity_ready():
            raise ObjectiveExecutionUnavailable(
                "objective worker has no immutable base model identity configured"
            )
        policy = _run_objective_stage(
            "CHECKPOINT_RESOLVE", lambda: self._policy_for_request(request)
        )
        trajectories: list[Trajectory] = []
        for task_id in request.execution_task_ids:
            task = engine.reset(split=request.split, task_id=task_id)
            observations: list[dict[str, Any]] = []
            actions: list[ToolCall] = []
            for _ in range(task.max_steps):
                try:
                    action = policy(task, tuple(dict(item) for item in observations))
                    if not isinstance(action, ToolCall):
                        raise ObjectiveExecutionUnavailable(
                            "target model returned an invalid tool call"
                        )

                    def perform_environment_step(selected_action: ToolCall = action) -> Any:
                        return engine.step(
                            selected_action.tool, dict(selected_action.arguments)
                        )

                    step = _run_objective_stage(
                        "ENVIRONMENT_STEP",
                        perform_environment_step,
                    )
                except ObjectiveExecutionUnavailable:
                    raise
                except Exception as exc:
                    raise ObjectiveExecutionUnavailable(
                        f"target-model objective step failed: {type(exc).__name__}"
                    ) from exc
                actions.append(action)
                observations.append(dict(step.observation))
                if step.done:
                    break
            try:
                trajectory = engine.run_episode(task_id, actions, split=request.split)
                if not isinstance(trajectory, Trajectory):
                    raise ObjectiveExecutionUnavailable("objective trajectory crossed sealed scope")

                def verify_trajectory(
                    selected_trajectory: Trajectory = trajectory,
                ) -> Trajectory:
                    return engine.verify(selected_trajectory).trajectory

                confirmed = _run_objective_stage(
                    "TRAJECTORY_VERIFY",
                    verify_trajectory,
                )
            except ObjectiveExecutionUnavailable:
                raise
            except Exception as exc:
                raise ObjectiveExecutionUnavailable(
                    f"deterministic objective verification failed: {type(exc).__name__}"
                ) from exc
            trajectories.append(confirmed)
        return BenchmarkExecutionResult(trajectories=tuple(trajectories))


class _UnavailableBenchmarkExecutionAdapter:
    def __init__(self, reason: str) -> None:
        self.reason = reason

    def execute_benchmark(
        self,
        request: BenchmarkRequest,
        engine: ServiceRecoveryEngine,
    ) -> BenchmarkExecutionResult:
        del request, engine
        raise ObjectiveExecutionUnavailable(self.reason)


def build_benchmark_execution_adapter(
    config: Mapping[str, str] | None = None,
) -> FunctionGemmaBenchmarkExecutionAdapter | _UnavailableBenchmarkExecutionAdapter:
    """Build from explicit pinned local config; never resolve a remote model ID."""

    values = os.environ if config is None else config
    checkpoint_dir = values.get(_CHECKPOINT_DIR_ENV, "").strip()
    revision = values.get(_REVISION_ENV, "").strip()
    expected_sha256 = values.get(_SHA256_ENV, "").strip()
    base_model_uri = values.get(_BASE_MODEL_URI_ENV, "").strip()
    base_model_sha256 = values.get(_BASE_MODEL_SHA256_ENV, "").strip()
    if not checkpoint_dir and not revision and not expected_sha256:
        return _UnavailableBenchmarkExecutionAdapter(
            "no immutable local FunctionGemma checkpoint is configured"
        )
    if not checkpoint_dir or not revision or not expected_sha256:
        return _UnavailableBenchmarkExecutionAdapter(
            "immutable local FunctionGemma checkpoint path, revision, and SHA-256 are all required"
        )
    if not base_model_uri or not base_model_sha256:
        return _UnavailableBenchmarkExecutionAdapter(
            "immutable S3 base model URI and SHA-256 are required"
        )
    if not _is_immutable_s3_model_uri(base_model_uri):
        return _UnavailableBenchmarkExecutionAdapter(
            "objective base model URI must pin an immutable S3 object version"
        )
    if not _SHA256_RE.fullmatch(base_model_sha256):
        return _UnavailableBenchmarkExecutionAdapter(
            "objective base model requires a lowercase SHA-256 bundle digest"
        )
    try:
        path = _validate_local_checkpoint(
            checkpoint_dir,
            revision=revision,
            expected_sha256=expected_sha256,
        )
    except ObjectiveExecutionUnavailable as exc:
        return _UnavailableBenchmarkExecutionAdapter(str(exc))
    return FunctionGemmaBenchmarkExecutionAdapter(
        _DeferredLocalPolicy(path, revision, expected_sha256),
        base_model_uri=base_model_uri,
        base_model_sha256=base_model_sha256,
        aws_region=values.get("AWS_REGION") or os.environ.get("AWS_REGION") or None,
    )
