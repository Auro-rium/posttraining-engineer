"""The regression split must be compared like-for-like, and the manual gate path must not be a
weaker way to a PROMOTE than ``run_gate``."""

from test_gate_power import R

from posttrain_agent.config import Settings
from posttrain_agent.experiments.domain import LocalStubBackend, make_tools, promotion_gate


def REG(score, **kw):
    return R(score, n=60, split="regression", **kw)


def _strong():
    return R(0.30, n=300, tasks_sha256="h"), R(0.50, n=300, tasks_sha256="h")


def test_regression_results_on_different_task_sets_are_rejected():
    champ, cand = _strong()
    out = promotion_gate(champ, cand, champion_regression=REG(0.5, tasks_sha256="a"),
                         candidate_regression=REG(0.5, tasks_sha256="b"))
    assert out["decision"] == "REJECT"
    assert any("regression tasks_sha256 differs" in r for r in out["reasons"])


def test_regression_results_on_the_same_task_set_still_pass():
    champ, cand = _strong()
    out = promotion_gate(champ, cand, champion_regression=REG(0.5, tasks_sha256="a"),
                         candidate_regression=REG(0.5, tasks_sha256="a"))
    assert out["decision"] == "PROMOTE", out["reasons"]


def test_decide_promotion_cannot_promote_without_the_regression_check(tmp_path):
    s = Settings(state_dir=tmp_path / "state", data_dir=tmp_path / "state" / "data")
    champ, cand = _strong()
    out = make_tools(LocalStubBackend(), s)["decide_promotion"](champ, cand)
    assert out["decision"] == "REJECT"
    assert any("regression split not evaluated" in r for r in out["reasons"])
