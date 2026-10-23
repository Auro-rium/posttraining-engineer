import json

import httpx

from posttrain_agent.config import Settings
from posttrain_agent.experiments import games
from posttrain_agent.experiments.domain import LocalStubBackend, make_tools, promotion_gate
from posttrain_agent.experiments.nebius import NebiusBackend
from posttrain_agent.tools import dataset


def _row(i):
    return {"messages": [{"role": "user", "content": f"u{i}"},
                         {"role": "assistant", "content": "a"}]}


def test_dataset_cap_is_1200_and_can_only_be_lowered():
    assert dataset.verify_rows([_row(i) for i in range(1200)])["ok"]
    over = dataset.verify_rows([_row(i) for i in range(1201)])
    assert not over["ok"] and "1200-row cap" in over["errors"][0]
    assert not dataset.verify_rows([_row(i) for i in range(11)], max_rows=10)["ok"]
    assert not dataset.verify_rows([_row(i) for i in range(1201)], max_rows=5000)["ok"]  # no raising
    assert Settings.from_env().max_dataset_rows == 1200


def test_env_can_lower_but_not_raise_the_cap(monkeypatch):
    monkeypatch.setenv("POSTTRAIN_MAX_DATASET_ROWS", "5000")
    assert Settings.from_env().max_dataset_rows == 1200
    monkeypatch.setenv("POSTTRAIN_MAX_DATASET_ROWS", "300")
    assert Settings.from_env().max_dataset_rows == 300


def _suite(tmp_path):
    suite = tmp_path / "suite"
    suite.mkdir()
    for name, seed, n in (("train", 1, 40), ("heldout", 2, 40), ("regression", 3, 20)):
        used = {json.loads(ln)["state"] for f in suite.glob("*.jsonl")
                for ln in f.read_text().splitlines()}
        (suite / f"{name}.jsonl").write_text(
            "".join(json.dumps(t) + "\n" for t in games.make_split(seed, n, used)))
    return suite


def test_generate_game_sft_tool_saves_capped_verified_rows_excluding_eval_states(tmp_path):
    suite = _suite(tmp_path)
    s = Settings(suite_dir=suite, data_dir=tmp_path / "data", max_dataset_rows=300)
    out = make_tools(LocalStubBackend(), s)["generate_game_sft"]("game-sft", n_rows=5000)
    assert out["ok"] and out["dataset_ref"] == "game-sft.jsonl"
    assert 0 < out["row_count"] <= 300 and sum(out["by_game"].values()) == out["row_count"]
    eval_states = games.suite_states(suite)
    saved = [json.loads(x) for x in (tmp_path / "data" / "game-sft.jsonl").read_text().splitlines()]
    assert len(saved) == out["row_count"]
    assert not {r["metadata"]["state"] for r in saved} & eval_states


def test_generate_game_sft_refuses_when_suite_has_no_state_keys(tmp_path):
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "train.jsonl").write_text(json.dumps({"id": "x", "messages": []}) + "\n")
    s = Settings(suite_dir=suite, data_dir=tmp_path / "data")
    out = make_tools(LocalStubBackend(), s)["generate_game_sft"]("d")
    assert out["ok"] is False and "state" in out["errors"][0]


def _res(score, n, evidence="LIVE"):
    return {"score": score, "n": n, "evidence": evidence, "suite": "s", "split": "heldout",
            "seed": 7, "parse_rate": 1.0}


def test_gate_rejects_a_gain_that_could_be_sampling_noise():
    # +10 points on 24 tasks is within noise; the same gain on 400 tasks is not
    small = promotion_gate(_res(0.50, 24), _res(0.60, 24))
    assert small["decision"] == "REJECT"
    assert any("statistically significant" in r for r in small["reasons"])
    big = promotion_gate(_res(0.50, 400), _res(0.60, 400))
    assert big["decision"] == "PROMOTE" and big["z"] > 1.645


def test_gate_promotes_a_real_win_over_base_and_rejects_when_n_is_missing():
    assert promotion_gate(_res(0.30, 100), _res(0.85, 100))["decision"] == "PROMOTE"
    no_n = {k: v for k, v in _res(0.5, 1).items() if k != "n"}
    out = promotion_gate(no_n, {**no_n, "score": 0.6})  # fail-closed: no n, no significance
    assert out["decision"] == "REJECT" and any("n missing" in r for r in out["reasons"])


def test_gate_still_requires_verified_evidence_even_with_a_huge_gain():
    assert promotion_gate(_res(0.1, 100), _res(0.9, 100, "EXPLANATION"))["decision"] == "REJECT"


HP = {"lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "learning_rate": 1e-4,
      "n_epochs": 2, "batch_size": 8, "context_length": 8192}


def _unpriced(tmp_path, **kw):
    suite = _suite(tmp_path)
    s = Settings(suite_dir=suite, data_dir=tmp_path / "data", state_dir=tmp_path / "state",
                 require_approval=False, **kw)
    tools = make_tools(LocalStubBackend(), s)
    assert tools["generate_game_sft"]("d", n_rows=200)["ok"]
    return s, tools


def test_launch_is_blocked_without_a_price_unless_unpriced_is_opted_into(tmp_path):
    s, tools = _unpriced(tmp_path)
    out = tools["launch_training"](HP, "d.jsonl")
    assert out["status"] == "BLOCKED" and out["stage"] == "budget"
    assert "POSTTRAIN_ALLOW_UNPRICED" in out["estimate"]["note"]


def test_unpriced_launch_is_capped_by_trained_tokens_and_records_zero_usd(tmp_path):
    s, tools = _unpriced(tmp_path, allow_unpriced=True, max_trained_tokens=10_000_000)
    est = tools["estimate_budget"]("d.jsonl", HP)
    assert est["within_budget"] is True and est["estimated_cost_usd"] is None
    assert "UNPRICED" in est["note"]
    out = tools["launch_training"](HP, "d.jsonl")
    assert out.get("provider_job_id") and out["estimate"]["estimated_cost_usd"] is None
    assert budget_spent(s) == 0.0
    (tmp_path / "t").mkdir()
    _, tight = _unpriced(tmp_path / "t", allow_unpriced=True, max_trained_tokens=10)
    assert tight["launch_training"](HP, "d.jsonl")["status"] == "BLOCKED"


def budget_spent(s):
    from posttrain_agent.tools import budget
    return budget.spent_usd(s)


def test_nebius_upload_drops_private_metadata_and_scores_per_game(tmp_path):
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/files"):
            seen["body"] = req.content.decode()
            return httpx.Response(200, json={"id": "file-1"})
        return httpx.Response(200, json={"id": "job-1", "status": "queued"})

    be = NebiusBackend(Settings(nebius_api_key="k"),
                       lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    path = tmp_path / "d.jsonl"
    rows, _ = games.generate_sft(8, 1, set())
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert be.train({"lora_r": 8}, path, "Qwen/Qwen3-0.6B")["provider_job_id"] == "job-1"
    assert '"metadata"' not in seen["body"] and '"tool_calls"' in seen["body"]
