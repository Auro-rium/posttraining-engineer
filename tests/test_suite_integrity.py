"""The evaluation suite is a hard boundary: SFT data must never train on a state it can be
evaluated on, including the wording-only ``heldout_phrasing`` slice."""

import json
from pathlib import Path

import pytest

from posttrain_agent.config import Settings
from posttrain_agent.experiments import games
from posttrain_agent.experiments import pretokenize as pt
from posttrain_agent.experiments.domain import LocalStubBackend, make_tools
from posttrain_agent.tools import dataset

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "state" / "data" / "exp1_data_n1200.jsonl"
V2 = ROOT / "state" / "suite_v2"
HP = {"lora_r": 8, "lora_alpha": 8, "lora_dropout": 0.0, "learning_rate": 1e-4,
      "n_epochs": 3, "batch_size": 4, "context_length": 8192, "packing": False}

needs_old = pytest.mark.skipif(not OLD.exists(), reason="state/data/exp1_data_n1200.jsonl absent")
needs_v2 = pytest.mark.skipif(not V2.is_dir(), reason="state/suite_v2 absent")
needs_tok = pytest.mark.skipif(
    not pt.available(), reason="jinja2/tokenizers or cached Qwen3 tokenizer missing")


def test_default_suite_directory_is_suite_v2(monkeypatch):
    monkeypatch.delenv("POSTTRAIN_SUITE_DIR", raising=False)
    monkeypatch.delenv("POSTTRAIN_STATE_DIR", raising=False)
    assert Settings().suite_dir == Path("state/suite_v2")
    assert Settings.from_env().suite_dir == Path("state/suite_v2")
    monkeypatch.setenv("POSTTRAIN_STATE_DIR", "runs/x")
    assert Settings.from_env().suite_dir == Path("runs/x/suite_v2")


def _write_suite(tmp_path: Path) -> Path:
    """All four split files, written the same way scripts/make_game_suites.py does it."""
    suite = tmp_path / "suite"
    suite.mkdir()
    used: set[str] = set()
    splits = (("train", 1, 24, (0,)), ("heldout", 2, 24, (0,)),
              ("regression", 3, 12, (0,)), ("heldout_phrasing", 4, 12, games.HELDOUT_STYLES))
    for name, seed, n, styles in splits:
        tasks = games.make_split(seed, n, used, styles)
        used |= {t["state"] for t in tasks}
        (suite / f"{name}.jsonl").write_text(
            "".join(json.dumps(t) + "\n" for t in tasks))
    return suite


@needs_old
@needs_v2
def test_old_n1200_dataset_is_blocked_on_suite_v2():
    overlap = dataset.check_suite_overlap(OLD.parent, OLD.name, V2)
    assert overlap["ok"] is False
    assert overlap["overlap_total"] == 91  # 15 train + 40 heldout + 18 regression + 18 phrasing
    assert overlap["overlap_counts"] == {"train": 15, "heldout": 40, "regression": 18,
                                         "heldout_phrasing": 18}
    assert overlap["overlap_states"]["heldout"]
    assert any("'heldout'" in e and "40 training state(s)" in e for e in overlap["errors"])

    # check_chat_template blocks BEFORE it needs the tokenizer
    blocked = dataset.check_chat_template(OLD.parent, OLD.name, suite_dir=V2)
    assert blocked["ok"] is False and blocked["stage"] == "contamination"
    assert blocked["overlap"]["overlap_total"] == 91


@needs_old
@needs_v2
def test_launch_training_blocks_the_old_dataset_against_the_configured_v2_suite(tmp_path):
    s = Settings(state_dir=tmp_path / "state", data_dir=OLD.parent, suite_dir=V2,
                 require_approval=False, price_per_mtok=1.0)
    out = make_tools(LocalStubBackend(), s)["launch_training"](HP, OLD.name)
    assert out["status"] == "BLOCKED" and out["stage"] == "contamination"
    assert out["overlap"]["overlap_total"] == 91
    assert any("heldout" in e for e in out["errors"])


@needs_v2
def test_clean_generated_dataset_has_zero_overlap_on_suite_v2(tmp_path):
    suite_states = games.suite_states(V2, include_phrasing=True)
    rows, _ = games.generate_sft(300, 31, suite_states)
    assert len(rows) == 300
    assert dataset.save_rows(tmp_path, "d.jsonl", rows)["ok"]
    overlap = dataset.check_suite_overlap(tmp_path, "d.jsonl", V2)
    assert overlap["ok"] is True and overlap["overlap_total"] == 0
    assert set(overlap["overlap_counts"].values()) == {0}
    assert overlap["dataset_states"] == 300
    assert not dataset.row_states(rows) & suite_states


@needs_v2
@needs_tok
def test_clean_generated_dataset_passes_chat_template_and_launch_on_suite_v2(tmp_path):
    suite_states = games.suite_states(V2)
    rows, _ = games.generate_sft(300, 32, suite_states)
    s = Settings(state_dir=tmp_path / "state", data_dir=tmp_path / "data", suite_dir=V2,
                 require_approval=False, price_per_mtok=1.0)
    assert dataset.save_rows(s.data_dir, "d.jsonl", rows)["ok"]
    tools = make_tools(LocalStubBackend(), s)
    checked = tools["check_chat_template"]("d.jsonl")
    assert checked["ok"], checked.get("errors")
    launched = tools["launch_training"](HP, "d.jsonl")
    assert launched.get("provider_job_id"), launched


def test_heldout_phrasing_states_are_included_and_excluded_from_generation(tmp_path):
    suite = _write_suite(tmp_path)
    per_split = games.suite_split_states(suite)
    assert set(per_split) == set(games.SPLIT_FILES)
    all_states = set().union(*per_split.values())
    assert games.suite_states(suite) == all_states
    assert per_split["heldout_phrasing"]
    # legacy view stays available for callers that only know the three gate splits
    assert games.suite_states(suite, include_phrasing=False) == set().union(
        *(per_split[split] for split in games.GATE_SPLITS))

    rows, _ = games.generate_sft(500, 7, all_states)
    assert rows
    assert not dataset.row_states(rows) & per_split["heldout_phrasing"]
    assert not dataset.row_states(rows) & all_states
    assert {r["metadata"]["style"] for r in rows} <= set(range(games.STYLES))


def test_generate_game_sft_tool_excludes_the_phrasing_split(tmp_path):
    suite = _write_suite(tmp_path)
    s = Settings(state_dir=tmp_path / "state", data_dir=tmp_path / "data", suite_dir=suite,
                 max_dataset_rows=500)
    out = make_tools(LocalStubBackend(), s)["generate_game_sft"]("game-sft", n_rows=500)
    assert out["ok"], out.get("errors")
    saved = dataset.load_rows(s.data_dir, "game-sft.jsonl")
    phrasing = games.suite_split_states(suite)["heldout_phrasing"]
    assert phrasing and not dataset.row_states(saved) & phrasing


def test_launch_rechecks_a_suite_that_changed_after_generation(tmp_path):
    suite = _write_suite(tmp_path)
    rows, _ = games.generate_sft(60, 41, games.suite_states(suite))
    s = Settings(state_dir=tmp_path / "state", data_dir=tmp_path / "data", suite_dir=suite,
                 require_approval=False, price_per_mtok=1.0)
    assert dataset.save_rows(s.data_dir, "d.jsonl", rows)["ok"]
    assert dataset.check_suite_overlap(s.data_dir, "d.jsonl", suite)["ok"]
    # the suite moves underneath the already-saved dataset: launch must re-read it and block
    leaked = rows[0]["metadata"]["state"]
    task = {"id": "leak", "state": leaked, "messages": [], "expected": {}}
    with (suite / "train.jsonl").open("a") as fh:
        fh.write(json.dumps(task) + "\n")
    out = make_tools(LocalStubBackend(), s)["launch_training"](HP, "d.jsonl")
    assert out["status"] == "BLOCKED" and out["stage"] == "contamination"
    assert out["overlap"]["overlap_counts"]["train"] == 1
    assert any("'train'" in e and "1 training state(s)" in e for e in out["errors"])


def test_v1_suite_does_not_block_a_clean_dataset(tmp_path):
    v1 = ROOT / "state" / "suite"
    if not v1.is_dir():
        pytest.skip("state/suite (v1) absent")
    rows, _ = games.generate_sft(300, 33, games.suite_states(v1))
    assert dataset.save_rows(tmp_path, "d.jsonl", rows)["ok"]
    overlap = dataset.check_suite_overlap(tmp_path, "d.jsonl", v1)
    assert overlap["ok"] is True and overlap["overlap_total"] == 0
