"""Real FunctionGemma execution for the service-recovery objective worker.

The benchmark adapter has no rule-based or random fallback. Production calls
require a complete local FunctionGemma snapshot pinned by commit revision and
content digest; policy inference receives only public task/observation data.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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


class ObjectiveExecutionUnavailable(RuntimeError):
    """A real objective model/checkpoint could not be used safely."""


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
        import safetensors  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoProcessor,
        )
    except Exception:
        return False
    return bool(
        getattr(torch, "__version__", None)
        and getattr(safetensors, "__version__", None)
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


class FunctionGemmaLocalPolicy:
    """Local-only FunctionGemma inference on a digest-pinned base snapshot."""

    def __init__(self, checkpoint_dir: Path, processor: Any, model: Any) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.processor = processor
        self.model = model

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        *,
        revision: str,
        expected_sha256: str,
    ) -> FunctionGemmaLocalPolicy:
        path = _validate_local_checkpoint(
            checkpoint_dir,
            revision=revision,
            expected_sha256=expected_sha256,
        )
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoProcessor,
            )

            processor = AutoProcessor.from_pretrained(
                str(path),
                revision=revision,
                local_files_only=True,
                trust_remote_code=False,
            )
            model = AutoModelForCausalLM.from_pretrained(
                str(path),
                revision=revision,
                local_files_only=True,
                trust_remote_code=False,
            )
            model.eval()
        except ImportError as exc:
            raise ObjectiveExecutionUnavailable(
                "Transformers is required for local FunctionGemma inference"
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
            encoded = self.processor.apply_chat_template(
                _messages(task, observations),
                tools=_tool_schemas(),
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
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
            generated = self.model.generate(**encoded, max_new_tokens=256)
            completion = generated[0, input_ids.shape[-1] :]
            text = self.processor.decode(completion, skip_special_tokens=False)
            return _parse_function_call(text)
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
    """Run actual target-model tool calls and return deterministic replay proof."""

    def __init__(self, policy: ActionPolicy) -> None:
        self.policy = policy

    def readiness(self) -> FunctionGemmaAdapterReadiness:
        """Attest only adapters created from a verified local checkpoint config."""

        if not isinstance(self.policy, _DeferredLocalPolicy):
            return FunctionGemmaAdapterReadiness(
                checkpoint_verified=False,
                inference_runtime_available=False,
                blockers=("functiongemma_adapter_unavailable",),
            )
        return self.policy.readiness()

    def execute_benchmark(
        self,
        request: BenchmarkRequest,
        engine: ServiceRecoveryEngine,
    ) -> BenchmarkExecutionResult:
        if request.split not in {ObjectiveSplit.TRAIN, ObjectiveSplit.REPLAY}:
            raise ObjectiveExecutionUnavailable("benchmark execution is limited to train/replay")
        trajectories: list[Trajectory] = []
        for task_id in request.task_ids:
            task = engine.reset(split=request.split, task_id=task_id)
            observations: list[dict[str, Any]] = []
            actions: list[ToolCall] = []
            for _ in range(task.max_steps):
                try:
                    action = self.policy(task, tuple(dict(item) for item in observations))
                    if not isinstance(action, ToolCall):
                        raise ObjectiveExecutionUnavailable(
                            "target model returned an invalid tool call"
                        )
                    step = engine.step(action.tool, dict(action.arguments))
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
                confirmed = engine.verify(trajectory).trajectory
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
    if not checkpoint_dir and not revision and not expected_sha256:
        return _UnavailableBenchmarkExecutionAdapter(
            "no immutable local FunctionGemma checkpoint is configured"
        )
    if not checkpoint_dir or not revision or not expected_sha256:
        return _UnavailableBenchmarkExecutionAdapter(
            "immutable local FunctionGemma checkpoint path, revision, and SHA-256 are all required"
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
        _DeferredLocalPolicy(path, revision, expected_sha256)
    )
