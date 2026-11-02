"""The promotion gate must expose its statistical-power limit, not hide it behind a REJECT.

At n=100 per side and base 0.19 the 80%-power MDE is about 0.138 absolute (13.8 points), while a
+10% relative gain is only about 0.019. These tests document that mismatch and pin the paired exact
test selection.
"""

import pytest

from posttrain_agent.agent import ORCHESTRATOR_PROMPT
from posttrain_agent.experiments.domain import promotion_gate
from posttrain_agent.experiments.gate import sign_test_p
from posttrain_agent.subagents import ROLES, load_prompt


def R(score, n=100, **kw):
    return {"score": score, "evidence": "LIVE", "suite": "s", "split": "heldout", "seed": 7,
            "n": n, "parse_rate": 1.0, "judged_by": "j", "served_by": "v", **kw}


def _verdicts(n, base_ok, cand_ok):
    return ("".join("1" if i in base_ok else "0" for i in range(n)),
            "".join("1" if i in cand_ok else "0" for i in range(n)))


def test_ten_percent_relative_gain_at_n100_is_rejected_as_not_significant():
    out = promotion_gate(R(0.19), R(0.209))  # +10% relative = +0.019 absolute
    assert out["decision"] == "REJECT"
    assert out["relative_improvement"] == pytest.approx(0.10, abs=1e-9)
    assert out["test"] == "two-proportion z (unpaired; no per-task verdicts)"
    assert out["z"] < 1.645
    assert any("statistically significant" in reason for reason in out["reasons"])


def test_power_block_reports_underpowered_gain_and_keeps_the_warning():
    out = promotion_gate(R(0.19), R(0.209))
    power = out["power"]
    assert power["adequately_powered"] is False
    assert power["observed_gain"] == pytest.approx(0.019, abs=1e-9)
    assert power["mde_80pct_power"] == pytest.approx(0.138, abs=0.001)
    assert power["mde_80pct_power"] == out["mde_80pct_power"]
    assert power["recommended_n"] == 300 and power["n"] == 100
    assert "power_note" in out
    assert any("below the minimum detectable effect" in w for w in out["warnings"])


def test_large_gain_clears_significance_and_is_not_rejected_for_power():
    out = promotion_gate(R(0.19), R(0.60))
    assert out["decision"] == "PROMOTE"
    assert out["power"]["adequately_powered"] is True
    assert not any("statistically significant" in reason for reason in out["reasons"])
    assert not any("minimum detectable effect" in warning for warning in out["warnings"])


def test_paired_verdicts_with_matching_tasks_sha_use_the_paired_exact_test():
    n = 100
    a, b = _verdicts(n, set(range(50)), set(range(60)))  # candidate fixes 10, breaks none
    out = promotion_gate(R(0.5, verdicts=a, tasks_sha256="t"),
                         R(0.6, verdicts=b, tasks_sha256="t"))
    assert out["test"] == "paired-exact (McNemar sign test)"
    assert out["p_value"] == pytest.approx(sign_test_p(10, 0), abs=1e-5)
    assert out["decision"] == "PROMOTE"


def test_prompts_require_test_power_parse_rate_and_reject_honesty():
    cm = load_prompt(ROLES["champion_manager"][0])
    td = load_prompt(ROLES["training_designer"][0])
    for text in (ORCHESTRATOR_PROMPT, cm):
        assert "paired" in text and "two-proportion" in text
        assert "mde_80pct_power" in text
        assert "parse rate" in text
        assert "REJECT is a valid outcome" in text
    assert "detectable effect" in td and "undetectable" in td
