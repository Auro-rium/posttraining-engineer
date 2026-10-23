"""Phase A: pretokenized masked-label training rows, train/serve parity, dataset checks, step guard,
tokenizer-based budget, and the launch gates."""

import json
import math
import shutil
from pathlib import Path

import pytest

from posttrain_agent.config import Settings
from posttrain_agent.experiments import games
from posttrain_agent.experiments import pretokenize as pt
from posttrain_agent.experiments.domain import (
    LocalStubBackend,
    estimate_steps,
    make_tools,
    step_findings,
    validate_hparams,
)
from posttrain_agent.experiments.sandbox_student import parse_tool_calls
from posttrain_agent.tools import budget, dataset

needs_tok = pytest.mark.skipif(not pt.available(), reason="jinja2/tokenizers or cached Qwen3 tokenizer missing")
OLD = Path(__file__).resolve().parents[1] / "state" / "data" / "exp1_data_n1200.jsonl"
# state/ is gitignored: the live-06b dataset only exists on the machine that ran it
needs_old = pytest.mark.skipif(not OLD.exists(), reason="state/data/exp1_data_n1200.jsonl absent")
HP = {"lora_r": 8, "lora_alpha": 8, "lora_dropout": 0.0, "learning_rate": 1e-4, "n_epochs": 3,
      "batch_size": 4, "context_length": 8192, "packing": False}


def _rows(n=60, seed=3):
    return games.generate_sft(n, seed, set())[0]


# ------------------------------------------------------------- games / rows


def test_train_and_eval_prompts_share_the_system_line():
    spec = games.GAMES["nim"](__import__("random").Random(1))
    task, row = games.as_task(spec, 0), games.as_sft_row(spec, 0)
    assert task["messages"][0] == {"role": "system", "content": games.SYSTEM_PROMPT}
    assert row["messages"][:2] == task["messages"]


def test_every_game_state_roundtrips_through_the_rule_recompute():
    for row in _rows(200):
        name, args = games.expected_from_state(row["metadata"]["state"])
        call = row["messages"][2]["tool_calls"][0]["function"]
        assert (call["name"], json.loads(call["arguments"])) == (name, args)


def test_verify_rows_rejects_a_semantically_wrong_answer():
    good = _rows(4)[0]
    assert dataset.verify_rows([good])["ok"]
    bad = json.loads(json.dumps(good))
    call = bad["messages"][2]["tool_calls"][0]["function"]
    key = next(iter(json.loads(call["arguments"])))
    wrong = json.loads(call["arguments"])
    wrong[key] = "scissors" if wrong[key] != "scissors" else "rock" if isinstance(wrong[key], str) \
        else wrong[key] + 1
    call["arguments"] = json.dumps(wrong)
    rep = dataset.verify_rows([bad])
    assert not rep["ok"] and any("disagrees with the rule answer" in e for e in rep["errors"])


# ------------------------------------------------------------- pretokenize


@needs_tok
def test_labels_decode_to_exactly_the_tool_call_and_parse():
    tok = pt.get_tokenizer()
    for row in _rows(40):
        ex = pt.pretokenize_row(row, tok)
        ids, labels = ex["token_ids"], ex["labels"]
        assert len(ids) == len(labels) == len(ex["attention_mask"]) and set(ex["attention_mask"]) == {1}
        sup = [i for i, lab in enumerate(labels) if lab != -100]
        assert sup == list(range(sup[0], len(ids)))  # one trailing span
        assert all(labels[i] == ids[i] for i in sup)
        text = tok.decode([ids[i] for i in sup])
        call = row["messages"][2]["tool_calls"][0]["function"]
        want = {"name": call["name"], "arguments": json.loads(call["arguments"])}
        assert text == f"<tool_call>\n{json.dumps(want)}\n</tool_call><|im_end|>"
        assert parse_tool_calls(text) == [want]


@needs_tok
@needs_old
def test_target_equals_what_the_qwen3_template_renders_itself():
    tok = pt.get_tokenizer()
    for row in _rows(40) + [json.loads(OLD.read_text().splitlines()[0])]:
        assert pt.derived_target(tok, row) == pt.tool_call_target(pt.split_row(row)[1])


@needs_tok
def test_train_prompt_equals_serving_prompt_even_for_old_sorted_key_datasets():
    """Serving renders the suite task (tools in games._tool key order). Training rows loaded from a
    sort_keys file must still produce byte-identical prompt text."""
    tok = pt.get_tokenizer()
    rng = __import__("random").Random(5)
    for game in games.GAMES:
        spec = games.GAMES[game](rng)
        task = games.as_task(spec, 0)
        serve = tok.render(task["messages"], task["tools"], True)  # sandbox_student.py:161-163
        row = json.loads(json.dumps(games.as_sft_row(spec, 0), sort_keys=True))  # as saved on disk
        assert pt.serving_prompt(tok, row) == serve
        ids = pt.pretokenize_row(row, tok)
        n_prompt = sum(1 for lab in ids["labels"] if lab == -100)
        assert tok.decode(ids["token_ids"][:n_prompt]) == serve


@pytest.mark.real_gate
@needs_tok
def test_check_chat_template_writes_marker_and_pretok_file(tmp_path):
    ref = "d.jsonl"
    assert dataset.save_rows(tmp_path, ref, _rows(30))["ok"]
    sha = dataset.verify_file(tmp_path, ref)["sha256"]
    assert not dataset.check_marker_ok(tmp_path, ref, sha)  # nothing checked yet
    rep = dataset.check_chat_template(tmp_path, ref)
    assert rep["ok"], rep["errors"] and 0.0 < rep["supervised_token_fraction"] < 0.5
    assert dataset.check_marker_ok(tmp_path, ref, sha)
    pre = [json.loads(line) for line in (tmp_path / "d.pretok.jsonl").read_text().splitlines()]
    assert len(pre) == 30 and set(pre[0]) == {"token_ids", "labels", "attention_mask"}
    assert (tmp_path / "d.pretok.sha256").exists()
    # any change to the dataset invalidates the marker
    assert dataset.save_rows(tmp_path, ref, _rows(31, seed=4))["ok"]
    assert not dataset.check_marker_ok(tmp_path, ref, dataset.verify_file(tmp_path, ref)["sha256"])


@needs_tok
def test_inspect_dataset_reports_counts_and_samples(tmp_path):
    dataset.save_rows(tmp_path, "d.jsonl", _rows(40))
    rep = dataset.inspect_dataset(tmp_path, "d.jsonl", n=2, game="nim")
    assert rep["row_count"] == 40 and set(rep["by_game"]) == set(games.GAMES)
    assert len(rep["samples"]) == 2 and rep["samples"][0]["state"].startswith("nim:")
    assert rep["samples"][0]["training_target"].startswith("<tool_call>")
    assert games.SYSTEM_PROMPT in rep["samples"][0]["serving_prompt"]


def test_check_chat_template_degrades_without_a_tokenizer(tmp_path, monkeypatch):
    monkeypatch.setenv("POSTTRAIN_TOKENIZER_DIR", str(tmp_path / "nope"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "nohf"))
    dataset.save_rows(tmp_path, "d.jsonl", _rows(5))
    rep = dataset.check_chat_template(tmp_path, "d.jsonl")
    assert rep["ok"] is False and rep["stage"] == "tokenizer"


# ------------------------------------------------------------- step guard


def test_estimate_steps_formula_and_packing():
    hp = {"n_epochs": 3, "batch_size": 16, "context_length": 8192, "packing": True}
    assert estimate_steps(hp, 296_000, 250) == math.ceil(296_000 * 3 / (16 * 8192))  # the 9-step run
    assert estimate_steps({**hp, "packing": False, "batch_size": 4}, 296_000, 250) == 888


@pytest.mark.parametrize("hp, kind", [
    ({"n_epochs": 3, "batch_size": 16, "context_length": 8192, "packing": True}, "block"),
    ({"n_epochs": 3, "batch_size": 64, "context_length": 8192, "packing": False}, "warn"),
    ({"n_epochs": 3, "batch_size": 4, "context_length": 8192, "packing": False}, "ok"),
])
def test_step_guard_blocks_warns_and_accepts(hp, kind):
    steps, errors, warns = step_findings(hp, 296_000, 250)
    assert bool(errors) == (kind == "block") and bool(warns) == (kind == "warn"), steps
    full = {"lora_r": 8, "lora_alpha": 8, "lora_dropout": 0.0, "learning_rate": 1e-4, **hp}
    assert bool(validate_hparams(full, total_tokens=296_000, avg_row_tokens=250)) == (kind == "block")
    assert validate_hparams(full) == []  # without dataset numbers only the search space applies


def _settings(tmp_path, **kw):
    return Settings(state_dir=tmp_path / "state", data_dir=tmp_path / "data", price_per_mtok=1.0,
                    require_approval=False, **kw)


@pytest.mark.real_gate
def test_validate_hparams_tool_accepts_json_string_and_applies_step_guard(tmp_path):
    s = _settings(tmp_path)
    dataset.save_rows(s.data_dir, "d.jsonl", _rows(400))
    t = make_tools(LocalStubBackend(), s)
    packed = {**HP, "packing": True, "batch_size": 16}
    out = t["validate_hparams"](json.dumps(packed), dataset_ref="d.jsonl")
    assert out["valid"] is False and "optimizer steps" in out["errors"][0] and out["steps"] < 50
    ok = t["validate_hparams"](json.dumps(HP), dataset_ref="d.jsonl")
    assert ok["valid"] is True and ok["steps"] >= 100
    assert t["estimate_budget"]("d.jsonl", json.dumps(HP))["rows"] == 400


@pytest.mark.real_gate
@needs_tok
def test_launch_requires_a_passing_template_check_and_enough_steps(tmp_path):
    s = _settings(tmp_path)
    dataset.save_rows(s.data_dir, "d.jsonl", _rows(300))
    t = make_tools(LocalStubBackend(), s)
    out = t["launch_training"](HP, "d.jsonl")
    assert out["status"] == "BLOCKED" and out["stage"] == "template_check"
    assert t["check_chat_template"]("d.jsonl")["ok"]
    low = t["launch_training"]({**HP, "packing": True, "batch_size": 16}, "d.jsonl")
    assert low["status"] == "BLOCKED" and low["stage"] == "steps"
    ok = t["launch_training"](json.dumps(HP), "d.jsonl")
    assert ok.get("provider_job_id") and ok["trained_file"] == "d.pretok.jsonl"
    # tampering with the dataset after the check blocks launch again
    dataset.save_rows(s.data_dir, "d.jsonl", _rows(300, seed=9))
    assert t["launch_training"](HP, "d.jsonl")["stage"] == "template_check"


# ------------------------------------------------------------- budget


@needs_tok
@needs_old
def test_budget_estimate_is_within_25_percent_of_the_observed_889k(tmp_path):
    (tmp_path / "data").mkdir()
    shutil.copy(OLD, tmp_path / "data" / "d.jsonl")
    e = budget.estimate(_settings(tmp_path), "d.jsonl", {"n_epochs": 3, "context_length": 8192})
    assert abs(e["estimated_trained_tokens"] - 889_116) / 889_116 < 0.25
    assert e["basis"].startswith("tokenizer")


@needs_old
def test_budget_fallback_is_4x_the_old_char_estimate(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, "_tokenizer", lambda: None)
    (tmp_path / "data").mkdir()
    shutil.copy(OLD, tmp_path / "data" / "d.jsonl")
    e = budget.estimate(_settings(tmp_path), "d.jsonl", {"n_epochs": 3, "context_length": 8192})
    assert "fallback" in e["basis"]
    assert abs(e["estimated_trained_tokens"] - 889_116) / 889_116 < 0.6  # old heuristic was 3.7x low


def test_actual_trained_tokens_are_reconciled_and_unpriced_cap_is_cumulative(tmp_path):
    s = Settings(state_dir=tmp_path / "state", data_dir=tmp_path / "data", allow_unpriced=True,
                 max_trained_tokens=100_000)
    dataset.save_rows(s.data_dir, "d.jsonl", _rows(50))
    hp = {"n_epochs": 1, "context_length": 8192}
    first = budget.estimate(s, "d.jsonl", hp)
    assert first["within_budget"] is True and "UNPRICED" in first["note"]
    budget.record_spend(s, "job-1", 0.0, "d.jsonl", first["estimated_trained_tokens"])
    assert budget.spent_tokens(s) == first["estimated_trained_tokens"]
    assert budget.reconcile(s, "job-1", 90_000) is True and budget.spent_tokens(s) == 90_000
    assert budget.reconcile(s, "job-1", 90_000) is False  # idempotent
    again = budget.estimate(s, "d.jsonl", hp)
    assert again["spent_trained_tokens"] == 90_000 and again["within_budget"] is False  # cumulative


def test_reconcile_updates_cost_when_priced(tmp_path):
    s = _settings(tmp_path)
    budget.record_spend(s, "j", 0.5, "d.jsonl", 100)
    budget.reconcile(s, "j", 2_000_000)
    assert budget.spent_usd(s) == pytest.approx(2.0)


def test_approval_text_says_unpriced(tmp_path):
    from posttrain_agent.agent import approval_description

    s = Settings(state_dir=tmp_path / "state", data_dir=tmp_path / "data", allow_unpriced=True)
    dataset.save_rows(s.data_dir, "d.jsonl", _rows(10))
    text = approval_description(s)(
        {"args": {"dataset_ref": "d.jsonl", "hyperparameters": json.dumps(HP)}}, None, None)
    assert "UNPRICED" in text and "budget estimate: {" in text
