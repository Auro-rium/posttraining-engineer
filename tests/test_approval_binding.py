"""The human approves content, not a path: a dataset edited after the approval prompt must not
be uploaded."""

import json

import pytest
from test_domain import GOOD, ROW

from posttrain_agent.agent import approval_description
from posttrain_agent.config import Settings
from posttrain_agent.experiments.domain import LocalStubBackend, make_tools
from posttrain_agent.tools import dataset
from posttrain_agent.tools.dataset import save_rows

pytestmark = pytest.mark.real_gate


def _settings(tmp_path, **kw):
    s = Settings(state_dir=tmp_path / "state", data_dir=tmp_path / "state" / "data",
                 suite_dir=tmp_path / "nosuite", price_per_mtok=1.0, **kw)
    return s, save_rows(s.data_dir, "d.jsonl", [ROW])["sha256"]


def test_check_approved_sha_cases():
    assert dataset.check_approved_sha("AB", "ab", True) is None  # case-insensitive match
    assert "required" in dataset.check_approved_sha(None, "ab", True)
    assert dataset.check_approved_sha(None, "ab", False) is None  # approval off: nothing to bind
    assert "changed since it was approved" in dataset.check_approved_sha("cd", "ab", True)


def test_launch_without_sha_is_blocked_when_approval_is_on(tmp_path):
    s, _ = _settings(tmp_path)
    out = make_tools(LocalStubBackend(), s)["launch_training"](GOOD, "d.jsonl")
    assert out["status"] == "BLOCKED" and out["stage"] == "approval_binding"


def test_dataset_edited_after_approval_is_refused_and_nothing_is_spent(tmp_path):
    s, sha = _settings(tmp_path)
    edited = save_rows(s.data_dir, "d.jsonl", [ROW, ROW | {"note": "edited after approval"}])
    assert edited["ok"] and edited["sha256"] != sha  # the file really changed
    out = make_tools(LocalStubBackend(), s)["launch_training"](GOOD, "d.jsonl", dataset_sha256=sha)
    assert out["status"] == "BLOCKED" and out["stage"] == "approval_binding"
    assert "changed since it was approved" in out["errors"][0]
    assert not list(s.state_dir.glob("spend*"))  # nothing was recorded as spent


def test_approval_prompt_shows_the_live_hash_and_flags_a_lying_call(tmp_path):
    s, sha = _settings(tmp_path)
    describe = approval_description(s)
    ok = describe({"args": {"hyperparameters": GOOD, "dataset_ref": "d.jsonl",
                            "dataset_sha256": sha}}, None, None)
    assert sha in ok and "bound" in ok and "MISMATCH" not in ok
    bad = describe({"args": {"hyperparameters": GOOD, "dataset_ref": "d.jsonl",
                             "dataset_sha256": "0" * 64}}, None, None)
    assert "MISMATCH" in bad and sha in bad
    none = describe({"args": {"hyperparameters": GOOD, "dataset_ref": "d.jsonl"}}, None, None)
    assert "NOT BOUND" in none
    assert json.dumps(GOOD, sort_keys=True) in ok
