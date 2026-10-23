"""Fix round 4: large balanced suites + phrasing slice, exit invariant per candidate and for runs
that never launched, cost visibility."""

import json
import time

from test_phase_a import HP, _rows
from test_phase_b import R, put

from posttrain_agent.agent import approval_description
from posttrain_agent.config import Settings
from posttrain_agent.experiments import games
from posttrain_agent.experiments import runstate as rs
from posttrain_agent.experiments.domain import cost_summary
from posttrain_agent.runtime.exit_gate import exit_problem
from posttrain_agent.tools import dataset

# ---- (1) suites


def _suite():
    used, out = set(), {}
    for name, seed, n, styles in (("train", 1, 200, (0,)), ("heldout", 2, 300, (0,)),
                                  ("regression", 3, 150, (0,)),
                                  ("heldout_phrasing", 4, 120, games.HELDOUT_STYLES)):
        out[name] = games.make_split(seed, n, used, styles)
        used |= {t["state"] for t in out[name]}
    return out


def test_large_suites_are_full_balanced_disjoint_and_use_the_shared_system_prompt():
    suites = _suite()
    seen: set[str] = set()
    for name, tasks in suites.items():
        states = {t["state"] for t in tasks}
        assert len(states) == len(tasks) and not states & seen
        seen |= states
        counts = {g: sum(t["id"].startswith(g + "-") for t in tasks) for g in games.GAMES}
        assert max(counts.values()) - min(counts.values()) <= 1, (name, counts)
        assert all(t["messages"][0] == {"role": "system", "content": games.SYSTEM_PROMPT}
                   for t in tasks)
    assert len(suites["heldout"]) == 300 and len(suites["regression"]) == 150


def test_phrasing_slice_uses_wording_sft_never_sees_and_tasks_stay_correct():
    for t in _suite()["heldout_phrasing"][:40]:
        name, args = games.expected_from_state(t["state"])
        assert t["expected"] == {"name": name, "arguments": args}
    spec = games.GAMES["rps"](__import__("random").Random(5))
    seen_texts = {spec["render"](s) for s in games.HELDOUT_STYLES}
    assert not seen_texts & {spec["render"](s) for s in range(games.STYLES)}
    rows, _ = games.generate_sft(1200, 1, set())
    assert {r["metadata"]["style"] for r in rows} <= set(range(games.STYLES))
    assert "heldout_phrasing" in games.SPLIT_FILES  # SFT excludes its states too


def test_sft_leaves_room_after_all_suites_and_rps_nim_grow_past_old_caps():
    suites = _suite()
    exclude = {t["state"] for ts in suites.values() for t in ts}
    rows, counts = games.generate_sft(1200, 7, exclude)
    assert not {r["metadata"]["state"] for r in rows} & exclude
    assert counts["rps"] > 108 and counts["nim"] > 225
    assert dataset.verify_rows(rows, max_rows=1200)["ok"]


# ---- (2)/(3) exit invariant


def launch(s, job, status="succeeded", ckpt=None, run="r", key=None, ts=None):
    rs.write_run_state(s.state_dir, run=run, key=key or f"k-{job}", job_id=job, status=status,
                       checkpoint=ckpt)
    time.sleep(0.01)


def test_run_that_never_launched_is_not_complete_until_aborted(tmp_path):
    s = Settings(state_dir=tmp_path)
    assert "no training job was ever launched" in exit_problem(s.state_dir, "r")
    rs.record_abort(s.state_dir, "r", "goal needs no candidate")
    assert exit_problem(s.state_dir, "r") is None


def test_every_launched_candidate_needs_its_own_verdict_or_abort(tmp_path):
    s = Settings(state_dir=tmp_path)
    launch(s, "ftjob-1", ckpt="ck1")
    launch(s, "ftjob-2", ckpt="ck2")
    for ck in ("ck1", "ck2"):
        for sp in ("heldout", "regression"):
            put(s, ck, sp, R(0.5, sp))
    rs.record_gate(s.state_dir, run="r", champion="base", candidate="ck2", candidate_job="ftjob-2",
                   decision="REJECT")
    msg = exit_problem(s.state_dir, "r")  # the NEWEST is gated, the older one is not
    assert msg and "ck1" in msg and "run_gate(champion, 'ftjob-1')" in msg
    rs.record_gate(s.state_dir, run="r", champion="base", candidate="ck1", candidate_job="ftjob-1",
                   decision="REJECT")
    assert exit_problem(s.state_dir, "r") is None


def test_abort_covers_earlier_candidates_but_not_ones_launched_after_it(tmp_path):
    s = Settings(state_dir=tmp_path)
    launch(s, "ftjob-1", "failed")
    rs.record_abort(s.state_dir, "r", "first failed")
    assert exit_problem(s.state_dir, "r") is None
    time.sleep(0.01)
    launch(s, "ftjob-2", "running")
    assert "ftjob-2" in exit_problem(s.state_dir, "r")


# ---- (4) cost visibility


def test_approval_prompt_and_cost_summary_show_tokens_cap_steps_and_unpriced(tmp_path):
    s = Settings(state_dir=tmp_path / "state", data_dir=tmp_path / "data", allow_unpriced=True,
                 max_trained_tokens=5_000_000)
    dataset.save_rows(s.data_dir, "d.jsonl", _rows(10))
    c = cost_summary(s, "d.jsonl", HP)
    assert c["estimated_trained_tokens"] > 0 and c["max_trained_tokens"] == 5_000_000
    assert c["estimated_steps"] and "UNPRICED" in c["usd"]
    text = approval_description(s)(
        {"args": {"dataset_ref": "d.jsonl", "hyperparameters": json.dumps(HP)}}, None, None)
    assert "COST:" in text and "cap 5000000" in text and "optimizer steps: ~" in text
    assert "UNPRICED" in text


def test_priced_cost_line_shows_known_usd(tmp_path):
    s = Settings(state_dir=tmp_path / "state", data_dir=tmp_path / "data", price_per_mtok=1.0)
    dataset.save_rows(s.data_dir, "d.jsonl", _rows(10))
    c = cost_summary(s, "d.jsonl", HP)
    assert "known USD" in c["cost_line"] and "UNPRICED" not in c["cost_line"]
