"""Deterministic, in-memory service-recovery objective engine."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, overload

from .models import (
    ALLOWED_TOOLS,
    Dataset,
    DatasetManifest,
    DatasetRow,
    ObjectiveSplit,
    ReplayResult,
    StepResult,
    Task,
    ToolCall,
    Trajectory,
)

ENGINE_VERSION = "service-recovery-v1"
_SERVICES = ("api", "web", "database", "cache")
_FAILURE_MODES = ("config_error", "dependency_failure", "healthcheck_failure")
_MAX_EPISODE_STEPS = 3


class TrajectoryNotAdmissible(ValueError):
    """Raised when an unverified or invalid trajectory is used for training."""


@dataclass(frozen=True, slots=True)
class _TaskDefinition:
    task_id: str
    split: ObjectiveSplit
    service_name: str
    failure_mode: str


@dataclass(frozen=True, slots=True)
class _SealedEvaluation:
    """Private hidden-evaluation result with no serializing model methods."""

    task_id: str
    reward: int
    success: bool
    steps: int


class ServiceRecoveryEngine:
    """Own task reset, tool execution, replay, and deterministic verification.

    ``sealed`` is intentionally opt-in for the independent evaluator.  The
    public objective service uses the default and cannot access hidden tasks.
    """

    def __init__(self, *, seed: int = 0, sealed: bool = False) -> None:
        self.seed = seed
        self.sealed = sealed
        self._definition: _TaskDefinition | None = None
        self._active_issue = False
        self._step = 0

    def _make_definition(self, task_id: str, split: ObjectiveSplit) -> _TaskDefinition:
        digest = hashlib.sha256(f"{self.seed}:{split.value}:{task_id}".encode()).digest()
        return _TaskDefinition(
            task_id=task_id,
            split=split,
            service_name=_SERVICES[digest[0] % len(_SERVICES)],
            failure_mode=_FAILURE_MODES[digest[1] % len(_FAILURE_MODES)],
        )

    def reset(
        self,
        *,
        split: ObjectiveSplit = ObjectiveSplit.TRAIN,
        task_id: str = "train-001",
    ) -> Task:
        """Reset one deterministic task and return only its safe metadata."""

        split = ObjectiveSplit(split)
        if split is ObjectiveSplit.HIDDEN and not self.sealed:
            raise ValueError("hidden split is sealed")
        if not task_id.strip():
            raise ValueError("task_id must not be blank")
        self._definition = self._make_definition(task_id, split)
        self._active_issue = True
        self._step = 0
        return Task(
            task_id=task_id,
            split=split,
            service_name=self._definition.service_name,
            objective="restore the service and pass its health check",
            allowed_tools=ALLOWED_TOOLS,
            max_steps=_MAX_EPISODE_STEPS,
            engine_version=ENGINE_VERSION,
            seed=self.seed,
        )

    @property
    def current_task(self) -> Task | None:
        if self._definition is None:
            return None
        definition = self._definition
        return Task(
            task_id=definition.task_id,
            split=definition.split,
            service_name=definition.service_name,
            objective="restore the service and pass its health check",
            allowed_tools=ALLOWED_TOOLS,
            max_steps=_MAX_EPISODE_STEPS,
            engine_version=ENGINE_VERSION,
            seed=self.seed,
        )

    def _public_observation(self, *, service: str, message: str, healthy: bool) -> dict[str, Any]:
        return {"service": service, "healthy": healthy, "message": message}

    def step(self, tool: str, arguments: dict[str, Any] | None = None) -> StepResult:
        """Execute one allow-listed tool with a stable, timestamp-free result."""

        if self._definition is None:
            raise RuntimeError("reset must be called before step")
        if tool not in ALLOWED_TOOLS:
            raise ValueError(f"tool {tool!r} is not allowed")
        call = ToolCall(tool=tool, arguments=arguments or {})
        self._step += 1
        service = str(call.arguments.get("service", self._definition.service_name))
        if service not in _SERVICES:
            return self._result(call, False, f"invalid service: {service}", service=service)

        if tool == "edit_config":
            content = call.arguments.get("content")
            if not isinstance(content, str) or not content.strip():
                return self._result(call, False, "content is required", service=service)
            if (
                self._definition.failure_mode == "config_error"
                and service == self._definition.service_name
                and "broken_value_should_be_fixed" not in content
            ):
                self._active_issue = False
                return self._result(call, True, "configuration updated", service=service)
            return self._result(
                call, False, "configuration did not resolve the issue", service=service
            )

        if tool == "run_healthcheck":
            healthy = not self._active_issue
            return self._result(
                call,
                healthy,
                "health check passed" if healthy else "health check failed",
                service=service,
                healthy=healthy,
            )

        if tool == "restart_service":
            if (
                self._active_issue
                and service == self._definition.service_name
                and self._definition.failure_mode in {"dependency_failure", "healthcheck_failure"}
            ):
                self._active_issue = False
            healthy = not self._active_issue
            return self._result(
                call,
                healthy,
                "service restarted" if healthy else "service still requires repair",
                service=service,
                healthy=healthy,
            )

        if tool == "read_config":
            if not self._active_issue:
                content = "setting1=value1\nsetting2=value2\n"
            elif self._definition.failure_mode == "config_error":
                content = "setting1=value1\nsetting2=broken_value_should_be_fixed\n"
            elif self._definition.failure_mode == "dependency_failure":
                content = "setting1=value1\ndependency=unavailable\n"
            else:
                content = "setting1=value1\nhealthcheck=failed\n"
            return self._result(call, True, content, service=service, content=content)

        if tool == "get_logs":
            message = (
                "service startup error observed"
                if self._active_issue
                else "service started successfully"
            )
            return self._result(call, True, message, service=service)

        # inspect_service is deliberately descriptive but does not reveal the
        # private failure mode or verifier state.
        healthy = not self._active_issue
        return self._result(
            call, True, "service inspection complete", service=service, healthy=healthy
        )

    def _result(
        self,
        call: ToolCall,
        success: bool,
        message: str,
        *,
        service: str,
        healthy: bool | None = None,
        **extra: Any,
    ) -> StepResult:
        observation = self._public_observation(
            service=service,
            message=message,
            healthy=(not self._active_issue if healthy is None else healthy),
        )
        observation.update(extra)
        done = self._active_issue is False or self._step >= 10
        return StepResult(
            call=call,
            success=success,
            reward=int(done and not self._active_issue),
            done=done,
            step=self._step,
            observation=observation,
            error=None if success else message,
        )

    @overload
    def run_episode(  # type: ignore[overload-overlap]
        self,
        task_id: str,
        actions: list[ToolCall] | tuple[ToolCall, ...],
        *,
        split: Literal[ObjectiveSplit.HIDDEN],
        repaired_from_trajectory_id: str | None = None,
    ) -> _SealedEvaluation: ...

    @overload
    def run_episode(
        self,
        task_id: str,
        actions: list[ToolCall] | tuple[ToolCall, ...],
        *,
        split: ObjectiveSplit | None = None,
        repaired_from_trajectory_id: str | None = None,
    ) -> Trajectory: ...

    def run_episode(
        self,
        task_id: str,
        actions: list[ToolCall] | tuple[ToolCall, ...],
        *,
        split: ObjectiveSplit | None = None,
        repaired_from_trajectory_id: str | None = None,
    ) -> Trajectory | _SealedEvaluation:
        if split is None:
            split = ObjectiveSplit.REPLAY if task_id.startswith("replay-") else ObjectiveSplit.TRAIN
        else:
            split = ObjectiveSplit(split)
        if split is ObjectiveSplit.HIDDEN and repaired_from_trajectory_id is not None:
            raise ValueError("repair provenance cannot cross the sealed hidden boundary")
        episode_engine = ServiceRecoveryEngine(seed=self.seed, sealed=self.sealed)
        task = episode_engine.reset(split=split, task_id=task_id)
        results: list[StepResult] = []
        for action in actions:
            result = episode_engine.step(action.tool, dict(action.arguments))
            results.append(result)
            if result.done:
                break
        if split is ObjectiveSplit.HIDDEN:
            success = bool(results and results[-1].done and not episode_engine._active_issue)
            return _SealedEvaluation(
                task_id=task.task_id,
                reward=int(success),
                success=success,
                steps=len(results),
            )
        executed_actions = [result.call.model_dump(mode="json") for result in results]
        identity: dict[str, Any] = {
            "seed": self.seed,
            "task_id": task.task_id,
            "split": task.split.value,
            "actions": executed_actions,
        }
        if repaired_from_trajectory_id is not None:
            identity["repaired_from_trajectory_id"] = repaired_from_trajectory_id
        trajectory_id = "traj-" + hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
        return Trajectory(
            trajectory_id=trajectory_id,
            task_id=task.task_id,
            split=task.split,
            engine_version=ENGINE_VERSION,
            seed=self.seed,
            steps=tuple(results),
            total_reward=round(sum(result.reward for result in results), 8),
            success=bool(results and results[-1].done and not episode_engine._active_issue),
            done=bool(results and results[-1].done),
            repaired_from_trajectory_id=repaired_from_trajectory_id,
        )

    def verify(self, trajectory: Trajectory) -> ReplayResult:
        if trajectory.engine_version != ENGINE_VERSION or trajectory.seed != self.seed:
            raise ValueError("trajectory provenance does not match engine")
        if trajectory.split is ObjectiveSplit.HIDDEN:
            raise ValueError("hidden split cannot be replayed by the objective service")
        if not trajectory.steps:
            raise TrajectoryNotAdmissible("verifier requires at least one trajectory step")
        actions = tuple(step.call for step in trajectory.steps)
        replayed = self.run_episode(
            trajectory.task_id,
            actions,
            split=trajectory.split,
            repaired_from_trajectory_id=trajectory.repaired_from_trajectory_id,
        )
        if not isinstance(replayed, Trajectory):
            raise ValueError("trajectory replay crossed the sealed hidden boundary")
        if replayed.trajectory_id != trajectory.trajectory_id:
            raise ValueError("trajectory provenance does not match replay")
        if (
            replayed.steps != trajectory.steps
            or replayed.total_reward != trajectory.total_reward
            or replayed.success != trajectory.success
            or replayed.done != trajectory.done
            or replayed.task_id != trajectory.task_id
            or replayed.split != trajectory.split
            or replayed.engine_version != trajectory.engine_version
            or replayed.seed != trajectory.seed
        ):
            raise ValueError("trajectory replay outcome or provenance does not match verifier")
        confirmed = trajectory.model_copy(
            update={"verified": True, "verifier_success": replayed.success}
        )
        return ReplayResult(
            trajectory=confirmed,
            verified=True,
            replayed_reward=replayed.total_reward,
            replayed_success=replayed.success,
            reason="deterministic verifier confirmed replay",
        )

    replay = verify
    replay_trajectory = verify
    execute_tool = step

    def build_dataset(
        self,
        trajectories: list[Trajectory] | tuple[Trajectory, ...],
        *,
        run_id: str,
        experiment_id: str,
        scope: ObjectiveSplit = ObjectiveSplit.REPLAY,
    ) -> Dataset:
        scope = ObjectiveSplit(scope)
        if scope not in {ObjectiveSplit.TRAIN, ObjectiveSplit.REPLAY}:
            raise TrajectoryNotAdmissible("dataset scope is limited to train and replay")
        rows: list[DatasetRow] = []
        for trajectory in trajectories:
            if trajectory.split is not scope:
                raise TrajectoryNotAdmissible("trajectory split does not match dataset scope")
            if not trajectory.verified:
                raise TrajectoryNotAdmissible(
                    "only verifier-confirmed trajectories may enter a dataset"
                )
            try:
                confirmed = self.verify(trajectory).trajectory
            except ValueError as exc:
                raise TrajectoryNotAdmissible(
                    f"only verifier-confirmed trajectories may enter a dataset: {exc}"
                ) from exc
            if not confirmed.success or confirmed.verifier_success is not True:
                raise TrajectoryNotAdmissible(
                    "only successful trajectories or repaired trajectories that pass verifier "
                    "replay may enter a dataset"
                )
            rows.append(
                DatasetRow(
                    source_trajectory_id=confirmed.trajectory_id,
                    task_id=confirmed.task_id,
                    split=confirmed.split,
                    messages=tuple(
                        {
                            "role": "tool",
                            "name": step.call.tool,
                            "arguments": step.call.arguments,
                            "observation": step.observation,
                        }
                        for step in confirmed.steps
                    )
                    or ({"role": "tool", "name": "noop", "observation": {}},),
                    failure_label="service_recovery",
                    verifier_confirmed=True,
                    verifier_success=True,
                    repaired_from_trajectory_id=confirmed.repaired_from_trajectory_id,
                    source_type=(
                        "repaired_replay"
                        if confirmed.repaired_from_trajectory_id is not None
                        else "successful_replay"
                    ),
                )
            )
        if not rows:
            raise TrajectoryNotAdmissible(
                "dataset requires at least one verifier-confirmed successful trajectory"
            )
        row_payload = "\n".join(row.canonical_json() for row in rows)
        digest = hashlib.sha256(row_payload.encode()).hexdigest()
        dataset_id = (
            "dataset-"
            + hashlib.sha256(f"{run_id}:{experiment_id}:{digest}".encode()).hexdigest()[:24]
        )
        manifest = DatasetManifest(
            dataset_id=dataset_id,
            run_id=run_id,
            experiment_id=experiment_id,
            row_count=len(rows),
            sha256=digest,
            s3_uri=f"s3://objective-datasets/{dataset_id}.jsonl",
            created_at=datetime.now(UTC),
            source_trajectory_ids=tuple(row.source_trajectory_id for row in rows),
            target_failure_classes=("service_recovery",),
        )
        return Dataset(manifest=manifest, rows=tuple(rows))

    admit_for_training = build_dataset
