"""A job the provider rejected trained nothing, so it must not use up the token or USD cap.

Live finding (2026-11-02): two jobs rejected at dataset validation (0 trained tokens) kept their
launch estimates forever, 1,951,824 tokens of the 10M unpriced cap, because reconcile ignored 0.
"""

from __future__ import annotations

import pytest
from test_approval import settings as approval_settings
from test_domain import GOOD

from posttrain_agent.config import Settings
from posttrain_agent.experiments.domain import LocalStubBackend, make_tools
from posttrain_agent.tools import budget


def _settings(tmp_path, **kw):
    return Settings(state_dir=tmp_path / "state", data_dir=tmp_path / "data", **kw)


@pytest.mark.parametrize("status", ["failed", "cancelled", "FAILED"])
def test_an_ended_job_with_zero_tokens_clears_its_estimate(tmp_path, status):
    s = _settings(tmp_path)
    budget.record_spend(s, "j", 0.0, "d.jsonl", 934_632)
    assert budget.reconcile(s, "j", 0, status) is True
    assert budget.spent_tokens(s) == 0
    assert budget.reconcile(s, "j", 0, status) is False  # idempotent


@pytest.mark.parametrize("status", ["queued", "running", "validating", None])
def test_a_job_still_in_progress_with_zero_tokens_keeps_its_estimate(tmp_path, status):
    s = _settings(tmp_path)
    budget.record_spend(s, "j", 0.0, "d.jsonl", 934_632)
    assert budget.reconcile(s, "j", 0, status) is False
    assert budget.spent_tokens(s) == 934_632


def test_a_rejected_job_also_clears_its_priced_cost(tmp_path):
    s = _settings(tmp_path, price_per_mtok=1.0)
    budget.record_spend(s, "j", 0.93, "d.jsonl", 934_632)
    assert budget.reconcile(s, "j", 0, "failed") is True
    assert budget.spent_usd(s) == 0.0


def test_a_failed_job_that_did_train_is_charged_for_what_it_trained(tmp_path):
    s = _settings(tmp_path)
    budget.record_spend(s, "j", 0.0, "d.jsonl", 934_632)
    assert budget.reconcile(s, "j", 5_000, "failed") is True
    assert budget.spent_tokens(s) == 5_000


class _Rejecting(LocalStubBackend):
    def train(self, config, dataset_path, base_model):
        return {"error": "Line 0: Provided dataset does not fall into any supported formats",
                "provider_job_id": "ftjob-rejected", "status": "BLOCKED",
                "stage": "dataset_validation"}

    def job_status(self, job_id):
        return {"evidence": "LIVE", "provider_job_id": job_id, "status": "failed",
                "trained_tokens": 0, "checkpoint": None}


def test_a_launch_blocked_at_dataset_validation_records_no_spend(tmp_path):
    s = approval_settings(tmp_path)
    out = make_tools(_Rejecting(), s)["launch_training"](GOOD, "d.jsonl")
    assert out["status"] == "BLOCKED" and out["stage"] == "dataset_validation"
    assert budget.spent_tokens(s) == 0 and not budget.spend_path(s).exists()


def test_polling_a_rejected_job_clears_an_estimate_already_in_the_ledger(tmp_path):
    s = approval_settings(tmp_path)
    budget.record_spend(s, "ftjob-old", 0.0, "d.jsonl", 934_632)
    make_tools(_Rejecting(), s)["get_training_status"]("ftjob-old")
    assert budget.spent_tokens(s) == 0


def test_waiting_on_a_rejected_job_clears_its_estimate(tmp_path):
    s = approval_settings(tmp_path)
    budget.record_spend(s, "ftjob-old", 0.0, "d.jsonl", 934_632)
    make_tools(_Rejecting(), s)["wait_for_training"]("ftjob-old")
    assert budget.spent_tokens(s) == 0
