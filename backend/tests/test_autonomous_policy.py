"""Focused contracts for the deterministic autonomous run policy."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from math import inf, nan

import pytest

from app.autonomous.policy import (
    BudgetLedger,
    PolicyAction,
    approval_is_expired,
    bounded_exponential_polling_schedule,
    can_start_experiment,
    canonical_job_name,
    canonical_operation_key,
    canonical_request_hash,
    decide_run_control,
    target_score_reached,
)


def test_canonical_provider_identity_is_stable_and_safe() -> None:
    key = canonical_operation_key("run-1", 2, "TRAINING")
    assert key == "run-1:2:training"
    assert key == canonical_operation_key("run-1", 2, "training")

    request = {"z": [2, 1], "a": {"b": 2, "a": 1}}
    assert canonical_request_hash(request) == canonical_request_hash(
        {"a": {"a": 1, "b": 2}, "z": [2, 1]}
    )
    job_name = canonical_job_name(key, canonical_request_hash(request))
    assert len(job_name) <= 63
    assert job_name == canonical_job_name(key, canonical_request_hash(request))
    assert job_name.replace("-", "").isalnum()


@pytest.mark.parametrize("value", ["", "run 1", "run/1"])
def test_canonical_operation_key_rejects_unsafe_run_ids(value: str) -> None:
    with pytest.raises(ValueError):
        canonical_operation_key(value, 1, "training")


def test_hash_rejects_non_json_and_nonfinite_values() -> None:
    with pytest.raises(ValueError):
        canonical_request_hash({"cost": nan})
    with pytest.raises(ValueError):
        canonical_request_hash({"cost": inf})
    with pytest.raises(TypeError):
        canonical_request_hash({"opaque": object()})


def test_polling_schedule_is_bounded_and_deterministic() -> None:
    assert bounded_exponential_polling_schedule(1, 2, 5, 6) == (1.0, 2.0, 4.0, 5.0, 5.0, 5.0)
    with pytest.raises(ValueError):
        bounded_exponential_polling_schedule(0, 2, 5, 2)
    with pytest.raises(ValueError):
        bounded_exponential_polling_schedule(1, 1, 5, 0)


def test_incremental_budget_reservation_and_actual_reconciliation_are_hard() -> None:
    ledger = BudgetLedger(approved_budget_usd=10.0)
    ledger.reserve("op-1", 4.0)
    with pytest.raises(ValueError, match="budget"):
        ledger.reserve("op-2", 6.01)

    ledger.reconcile("op-1", 3.5)
    assert ledger.spent_budget_usd == 3.5
    assert ledger.reserved_budget_usd == 0.0
    with pytest.raises(ValueError, match="budget"):
        ledger.reconcile("op-1", 10.0)
    with pytest.raises(ValueError):
        ledger.reserve("op-1", 1.0)
    with pytest.raises(ValueError):
        BudgetLedger(approved_budget_usd=10.0).reserve("bad", -1.0)


def test_budget_rejects_nonfinite_costs() -> None:
    ledger = BudgetLedger(approved_budget_usd=10.0)
    for cost in (-1.0, nan, inf, True):
        with pytest.raises((TypeError, ValueError)):
            ledger.reserve("op", cost)


def test_budget_uses_exact_cents_and_rejects_sub_cent_ambiguity() -> None:
    ledger = BudgetLedger(approved_budget_usd=Decimal("0.30"))
    ledger.reserve("op-1", Decimal("0.10"))
    ledger.reserve("op-2", 0.20)
    assert ledger.remaining_budget_usd == 0.0
    with pytest.raises(ValueError, match="whole cents"):
        BudgetLedger(approved_budget_usd=1).reserve("op-3", Decimal("0.001"))


def test_budget_snapshot_hydrates_in_flight_and_reconciled_operations() -> None:
    ledger = BudgetLedger(approved_budget_usd=10)
    ledger.reserve("op-1", Decimal("4.25"))
    ledger.reserve("op-2", Decimal("1.75"))
    ledger.reconcile("op-2", Decimal("1.25"))

    serialized = json.loads(json.dumps(ledger.snapshot()))
    restored = BudgetLedger.from_snapshot(serialized)
    assert restored.reservation("op-1") == 4.25
    assert restored.reserved_budget_usd == 4.25
    restored.reconcile("op-1", Decimal("4.00"))
    with pytest.raises(ValueError, match="already exists"):
        restored.reserve("op-1", 1)
    with pytest.raises(ValueError, match="already exists"):
        restored.reserve("op-2", 1)

    with pytest.raises(ValueError, match="aggregate"):
        BudgetLedger.from_snapshot({**ledger.snapshot(), "reserved_budget_cents": 0})


def test_experiment_cap_and_target_score_stop() -> None:
    assert can_start_experiment(0)
    assert can_start_experiment(4)
    assert not can_start_experiment(5)
    with pytest.raises(ValueError):
        can_start_experiment(1, max_experiments=6)
    assert target_score_reached(0.81, 0.8)
    assert not target_score_reached(0.79, 0.8)


def test_stop_policy_encodes_cancel_safe_stop_and_expiry_semantics() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    expired = now - timedelta(seconds=1)

    cancelled = decide_run_control(cancellation_requested=True, active_provider_job=True, now=now)
    assert cancelled.action is PolicyAction.CANCEL
    assert cancelled.request_provider_stop
    assert not cancelled.may_submit_later_phase

    safe = decide_run_control(safe_stop_requested=True, active_provider_job=True, now=now)
    assert safe.action is PolicyAction.SAFE_STOP
    assert safe.allow_active_job_completion
    assert not safe.request_provider_stop
    assert not safe.may_submit_later_phase

    expiry = decide_run_control(approval_expires_at=expired, active_provider_job=True, now=now)
    assert expiry.action is PolicyAction.APPROVAL_EXPIRED
    assert expiry.allow_active_job_completion
    assert not expiry.may_submit_later_phase

    assert approval_is_expired(expired, now)
