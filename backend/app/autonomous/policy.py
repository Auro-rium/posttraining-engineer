"""Pure policy primitives for the autonomous live run.

This module deliberately has no provider, clock, or persistence dependency.  A
supervisor supplies ``now`` and records the returned decisions durably before
performing an external side effect.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from math import isfinite
from typing import Any, Final, cast

MAX_EXPERIMENTS: Final = 5
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
_OPERATION_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,511}$")
_PHASE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_JOB_NAME = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")


def _finite_number(value: object, name: str, *, nonnegative: bool = True) -> float:
    """Validate a JSON/Python numeric policy input and return a float."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _required_text(value: object, name: str, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without surrounding whitespace")
    if pattern is not None and not pattern.fullmatch(value):
        raise ValueError(f"{name} contains unsupported characters")
    return value


def _experiment_number(value: object, *, allow_baseline: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("experiment_number must be an integer")
    minimum = 0 if allow_baseline else 1
    if not minimum <= value <= MAX_EXPERIMENTS:
        raise ValueError(f"experiment_number must be between {minimum} and {MAX_EXPERIMENTS}")
    return value


def canonical_operation_key(run_id: str, experiment_number: int, phase: str) -> str:
    """Return the stable operation identity used for at-least-once execution."""

    run = _required_text(run_id, "run_id", _RUN_ID)
    number = _experiment_number(experiment_number)
    if hasattr(phase, "value"):
        phase = cast(Any, phase).value
    phase_text = _required_text(phase, "phase").lower()
    if not _PHASE.fullmatch(phase_text):
        raise ValueError("phase contains unsupported characters")
    return f"{run}:{number}:{phase_text}"


def _canonical_json(value: object, path: str = "$") -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"non-finite number at {path}")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"mapping keys at {path} must be non-empty strings")
            result[key] = _canonical_json(item, f"{path}.{key}")
        return {key: result[key] for key in sorted(result)}
    if isinstance(value, (list, tuple)):
        return [_canonical_json(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"unsupported value at {path}: {type(value).__name__}")


def canonical_request_hash(request: Mapping[str, object]) -> str:
    """Hash a strict, JSON-compatible request independent of mapping order."""

    if not isinstance(request, Mapping):
        raise TypeError("request must be a mapping")
    payload = _canonical_json(request)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def canonical_job_name(operation_key: str, request_hash: str) -> str:
    """Build a deterministic SageMaker-safe job name from operation identity."""

    operation = _required_text(operation_key, "operation_key", _OPERATION_KEY)
    digest = _required_text(request_hash, "request_hash")
    if not _SHA256.fullmatch(digest):
        raise ValueError("request_hash must be a lowercase SHA-256 digest")
    identity = hashlib.sha256(f"{operation}:{digest}".encode()).hexdigest()
    name = f"autonomous-{identity[:48]}"
    if not _JOB_NAME.fullmatch(name):  # pragma: no cover - construction is constrained above
        raise AssertionError("generated job name violated provider contract")
    return name


# Descriptive aliases used by adapters that call the digest an operation hash.
request_hash = canonical_request_hash
canonical_request_digest = canonical_request_hash
deterministic_job_name = canonical_job_name


def bounded_exponential_polling_schedule(
    initial_seconds: float = 1.0,
    factor: float = 2.0,
    max_seconds: float = 30.0,
    max_attempts: int = 5,
) -> tuple[float, ...]:
    """Return a finite, capped schedule; no unbounded retry is permitted."""

    initial = _finite_number(initial_seconds, "initial_seconds", nonnegative=False)
    multiplier = _finite_number(factor, "factor")
    maximum = _finite_number(max_seconds, "max_seconds", nonnegative=False)
    if initial <= 0 or maximum <= 0:
        raise ValueError("poll intervals must be positive")
    if multiplier < 1:
        raise ValueError("factor must be at least one")
    if maximum < initial:
        raise ValueError("max_seconds must be at least initial_seconds")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
        raise TypeError("max_attempts must be an integer")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    values: list[float] = []
    current = initial
    for _ in range(max_attempts):
        values.append(min(current, maximum))
        current = min(current * multiplier, maximum)
    return tuple(values)


exponential_poll_schedule = bounded_exponential_polling_schedule


@dataclass(slots=True)
class BudgetLedger:
    """Hard budget ledger with per-operation reservations.

    Reservations are incremental: committed cost plus all outstanding
    reservations can never exceed the approved cap.  Reconciliation is
    atomic from the caller's perspective (validation happens before mutation).
    """

    approved_budget_usd: float
    spent_budget_usd: float = 0.0
    reserved_budget_usd: float = 0.0
    _reservations: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _reconciled: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self.approved_budget_usd = _finite_number(
            self.approved_budget_usd, "approved_budget_usd"
        )
        self.spent_budget_usd = _finite_number(self.spent_budget_usd, "spent_budget_usd")
        self.reserved_budget_usd = _finite_number(
            self.reserved_budget_usd, "reserved_budget_usd"
        )
        if self.approved_budget_usd <= 0:
            raise ValueError("approved_budget_usd must be positive")
        if self.spent_budget_usd + self.reserved_budget_usd > self.approved_budget_usd:
            raise ValueError("committed budget exceeds approved budget")

    @property
    def remaining_budget_usd(self) -> float:
        return self.approved_budget_usd - self.spent_budget_usd - self.reserved_budget_usd

    @property
    def available_budget_usd(self) -> float:
        return self.remaining_budget_usd

    def reserve(self, operation_key: str, estimated_cost_usd: float) -> None:
        key = _required_text(operation_key, "operation_key", _OPERATION_KEY)
        cost = _finite_number(estimated_cost_usd, "estimated_cost_usd")
        if key in self._reservations or key in self._reconciled:
            raise ValueError(f"budget reservation already exists for {key!r}")
        if self.spent_budget_usd + self.reserved_budget_usd + cost > self.approved_budget_usd:
            raise ValueError("budget reservation exceeds approved budget")
        self._reservations[key] = cost
        self.reserved_budget_usd += cost

    def reconcile(self, operation_key: str, actual_cost_usd: float) -> None:
        key = _required_text(operation_key, "operation_key", _OPERATION_KEY)
        actual = _finite_number(actual_cost_usd, "actual_cost_usd")
        if key not in self._reservations:
            raise ValueError(f"no budget reservation exists for {key!r}")
        reservation = self._reservations[key]
        new_spent = self.spent_budget_usd + actual
        if new_spent + self.reserved_budget_usd - reservation > self.approved_budget_usd:
            raise ValueError("actual cost exceeds approved budget")
        del self._reservations[key]
        self._reconciled.add(key)
        self.reserved_budget_usd -= reservation
        self.spent_budget_usd = new_spent

    def reservation(self, operation_key: str) -> float:
        key = _required_text(operation_key, "operation_key", _OPERATION_KEY)
        try:
            return self._reservations[key]
        except KeyError as exc:
            raise ValueError(f"no budget reservation exists for {key!r}") from exc

    reserve_incremental = reserve
    reconcile_actual = reconcile


def reserve_incremental_budget(
    ledger: BudgetLedger, operation_key: str, estimated_cost_usd: float
) -> None:
    if not isinstance(ledger, BudgetLedger):
        raise TypeError("ledger must be a BudgetLedger")
    ledger.reserve(operation_key, estimated_cost_usd)


def reconcile_actual_cost(ledger: BudgetLedger, operation_key: str, actual_cost_usd: float) -> None:
    if not isinstance(ledger, BudgetLedger):
        raise TypeError("ledger must be a BudgetLedger")
    ledger.reconcile(operation_key, actual_cost_usd)


reserve_budget = reserve_incremental_budget
reconcile_cost = reconcile_actual_cost


def _aware_datetime(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def approval_is_expired(expires_at: datetime, now: datetime) -> bool:
    """Compare approval expiry against an explicitly supplied instant."""

    return _aware_datetime(now, "now") >= _aware_datetime(expires_at, "expires_at")


class PolicyAction(StrEnum):
    CONTINUE = "CONTINUE"
    CANCEL = "CANCEL"
    SAFE_STOP = "SAFE_STOP"
    APPROVAL_EXPIRED = "APPROVAL_EXPIRED"
    MAX_EXPERIMENTS = "MAX_EXPERIMENTS"
    TARGET_REACHED = "TARGET_REACHED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    action: PolicyAction
    reason: str
    allow_active_job_completion: bool
    request_provider_stop: bool
    may_submit_later_phase: bool

    @property
    def should_stop(self) -> bool:
        return self.action is not PolicyAction.CONTINUE


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def target_score_reached(champion_score: float, target_score: float) -> bool:
    champion = _finite_number(champion_score, "champion_score")
    target = _finite_number(target_score, "target_score")
    return champion >= target


should_stop_for_target_score = target_score_reached


def can_start_experiment(experiment_count: int, max_experiments: int = MAX_EXPERIMENTS) -> bool:
    if isinstance(experiment_count, bool) or not isinstance(experiment_count, int):
        raise TypeError("experiment_count must be an integer")
    if experiment_count < 0:
        raise ValueError("experiment_count must be non-negative")
    if isinstance(max_experiments, bool) or not isinstance(max_experiments, int):
        raise TypeError("max_experiments must be an integer")
    if not 1 <= max_experiments <= MAX_EXPERIMENTS:
        raise ValueError(f"max_experiments must be between 1 and {MAX_EXPERIMENTS}")
    return experiment_count < max_experiments


def decide_run_control(
    *,
    cancellation_requested: bool = False,
    safe_stop_requested: bool = False,
    approval_expires_at: datetime | None = None,
    now: datetime | None = None,
    active_provider_job: bool = False,
    experiment_count: int = 0,
    max_experiments: int = MAX_EXPERIMENTS,
    champion_score: float | None = None,
    target_score: float | None = None,
    remaining_budget_usd: float | None = None,
) -> PolicyDecision:
    """Apply stop conditions in a stable priority order.

    Cancellation is the only condition that asks a provider to stop an active
    job.  Safe-stop and approval expiry drain the active job but prohibit all
    subsequent phase submissions.
    """

    cancel = _strict_bool(cancellation_requested, "cancellation_requested")
    safe = _strict_bool(safe_stop_requested, "safe_stop_requested")
    active = _strict_bool(active_provider_job, "active_provider_job")
    if approval_expires_at is not None:
        if now is None:
            raise ValueError("now is required when approval expiry is supplied")
        expired = approval_is_expired(approval_expires_at, now)
    else:
        expired = False
        if now is not None:
            _aware_datetime(now, "now")
    if not can_start_experiment(experiment_count, max_experiments):
        cap_reached = True
    else:
        cap_reached = False
    if remaining_budget_usd is not None:
        remaining = _finite_number(remaining_budget_usd, "remaining_budget_usd")
        budget_exhausted = remaining <= 0
    else:
        budget_exhausted = False
    target_reached = False
    if champion_score is not None or target_score is not None:
        if champion_score is None or target_score is None:
            raise ValueError("champion_score and target_score must be supplied together")
        target_reached = target_score_reached(champion_score, target_score)

    if cancel:
        return PolicyDecision(
            PolicyAction.CANCEL,
            "cancellation requested",
            False,
            active,
            False,
        )
    if safe:
        return PolicyDecision(PolicyAction.SAFE_STOP, "safe stop requested", True, False, False)
    if expired:
        return PolicyDecision(
            PolicyAction.APPROVAL_EXPIRED, "approval expired", True, False, False
        )
    if target_reached:
        return PolicyDecision(
            PolicyAction.TARGET_REACHED, "target score reached", True, False, False
        )
    if cap_reached:
        return PolicyDecision(
            PolicyAction.MAX_EXPERIMENTS, "maximum experiments reached", True, False, False
        )
    if budget_exhausted:
        return PolicyDecision(
            PolicyAction.BUDGET_EXHAUSTED, "hard budget exhausted", True, False, False
        )
    return PolicyDecision(PolicyAction.CONTINUE, "policy permits next phase", True, False, True)


stop_decision = decide_run_control
operation_key = canonical_operation_key
job_name = canonical_job_name
polling_schedule = bounded_exponential_polling_schedule
is_target_score_reached = target_score_reached


__all__ = [
    "MAX_EXPERIMENTS",
    "BudgetLedger",
    "PolicyAction",
    "PolicyDecision",
    "approval_is_expired",
    "bounded_exponential_polling_schedule",
    "can_start_experiment",
    "canonical_job_name",
    "canonical_operation_key",
    "canonical_request_digest",
    "canonical_request_hash",
    "decide_run_control",
    "deterministic_job_name",
    "exponential_poll_schedule",
    "is_target_score_reached",
    "job_name",
    "operation_key",
    "polling_schedule",
    "reconcile_actual_cost",
    "reconcile_cost",
    "request_hash",
    "reserve_budget",
    "reserve_incremental_budget",
    "should_stop_for_target_score",
    "stop_decision",
    "target_score_reached",
]
