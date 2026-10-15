import copy
import json

import pytest
from test_domain import ROW

from posttrain_agent import budget, dataset
from posttrain_agent.config import Settings


def row(**over):
    r = copy.deepcopy(ROW)
    r.update(over)
    return r


def test_valid_rows_pass_and_hash_is_stable():
    a, b = dataset.verify_rows([ROW]), dataset.verify_rows([ROW])
    assert a["ok"] and a["sha256"] == b["sha256"] and a["evidence"] == ["LIVE"]


@pytest.mark.parametrize("bad, fragment", [
    ({"messages": []}, "non-empty"),
    ({"messages": [{"role": "robot", "content": "x"}]}, "invalid role"),
    ({"messages": [{"role": "user", "content": "x"}]}, "last message"),
    ({"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "rm_rf", "arguments": "{}"}}]}]}, "allow-listed"),
    ({"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "get_logs", "arguments": "[1]"}}]}]}, "JSON object"),
    ({"messages": [{"role": "user", "content": "see heldout://t9"},
                   {"role": "assistant", "content": "ok"}]}, "sealed"),
])
def test_bad_rows_rejected(bad, fragment):
    rep = dataset.verify_rows([bad])
    assert not rep["ok"] and any(fragment in e for e in rep["errors"])


def test_duplicates_and_empty_dataset_rejected():
    assert any("duplicate" in e for e in dataset.verify_rows([ROW, ROW])["errors"])
    assert not dataset.verify_rows([])["ok"]


def test_explanation_rows_warn_and_save_requires_verification(tmp_path):
    sim = row(metadata={"evidence": "EXPLANATION"})
    assert dataset.verify_rows([sim])["warnings"]
    assert not dataset.save_rows(tmp_path, "x.jsonl", [{"messages": []}])["ok"]
    assert not (tmp_path / "x.jsonl").exists()
    assert dataset.save_rows(tmp_path, "x.jsonl", [ROW])["ok"]
    assert dataset.verify_file(tmp_path, "x.jsonl")["ok"]


@pytest.mark.parametrize("ref", ["../x.jsonl", "a/b.jsonl", "x.txt", "/etc/passwd"])
def test_path_traversal_refused(tmp_path, ref):
    with pytest.raises(ValueError):
        dataset.resolve(tmp_path, ref)
    assert not dataset.verify_file(tmp_path, ref)["ok"]


def test_budget_math_cap_and_unknown_price(tmp_path):
    (tmp_path / "data").mkdir()
    big = row(messages=[{"role": "user", "content": "x" * 4000},
                        {"role": "assistant", "content": "y" * 4000}])
    dataset.save_rows(tmp_path / "data", "d.jsonl", [big])
    hp = {"n_epochs": 2, "context_length": 8192}
    s = Settings(state_dir=tmp_path, data_dir=tmp_path / "data", price_per_mtok=10.0, max_cost_usd=1.0)
    e = budget.estimate(s, "d.jsonl", hp)
    assert e["estimated_trained_tokens"] == 2 * (8032 // 4)  # two epochs of 2008 tokens
    assert e["estimated_cost_usd"] == pytest.approx(4016 / 1e6 * 10, abs=1e-4)
    assert e["within_budget"] is True
    budget.record_spend(s, "j1", 0.99, "d.jsonl")
    assert budget.estimate(s, "d.jsonl", hp)["within_budget"] is False
    unpriced = budget.estimate(Settings(state_dir=tmp_path, data_dir=tmp_path / "data"), "d.jsonl", hp)
    assert unpriced["within_budget"] is None and unpriced["estimated_cost_usd"] is None
    assert "error" in budget.estimate(s, "nope.jsonl", hp)


def test_context_truncation_counted(tmp_path):
    (tmp_path / "data").mkdir()
    big = row(messages=[{"role": "user", "content": "x" * 100_000},
                        {"role": "assistant", "content": "ok"}])
    dataset.save_rows(tmp_path / "data", "d.jsonl", [big])
    s = Settings(state_dir=tmp_path, data_dir=tmp_path / "data", price_per_mtok=1.0)
    e = budget.estimate(s, "d.jsonl", {"n_epochs": 1, "context_length": 8192})
    assert e["rows_over_context"] == 1 and e["estimated_trained_tokens"] == 8192
    assert json.dumps(e)
