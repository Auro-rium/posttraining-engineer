import json

from posttrain_agent.config import Settings
from posttrain_agent.domain import LocalStubBackend, make_tools, promotion_gate, validate_hparams

GOOD = {"lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "learning_rate": 2e-4,
        "n_epochs": 3, "batch_size": 8, "context_length": 8192}

ROW = {"messages": [
    {"role": "user", "content": "api is down"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "get_logs", "arguments": json.dumps({"service": "api"})}}]}],
    "metadata": {"evidence": "LIVE"}}


def ev(score, evidence="LIVE", split="heldout"):
    return {"score": score, "evidence": evidence, "suite": "s", "split": split, "seed": 7}


def tools(tmp_path, **kw):
    s = Settings(state_dir=tmp_path, data_dir=tmp_path / "data", price_per_mtok=1.0, **kw)
    return make_tools(LocalStubBackend(), s), s


def test_hparams_accept_valid_and_reject_out_of_space():
    assert validate_hparams(GOOD) == []
    assert validate_hparams({**GOOD, "lora_r": 4})
    assert validate_hparams({**GOOD, "learning_rate": 1.0})
    assert validate_hparams({**GOOD, "n_epochs": 2.5})
    assert validate_hparams({**GOOD, "n_epochs": True})
    assert validate_hparams({**GOOD, "evil": 1})
    assert validate_hparams({**GOOD, "lora_alpha": 200})  # > 4 * lora_r
    assert validate_hparams({k: v for k, v in GOOD.items() if k != "batch_size"})
    assert validate_hparams({**GOOD, "lora": False})  # LoRA is not optional


def test_gate_promotes_clear_verified_improvement():
    assert promotion_gate(ev(0.5), ev(0.6))["decision"] == "PROMOTE"


def test_gate_fails_closed_on_unverified_evidence():
    r = promotion_gate(ev(0.5, "EXPLANATION"), ev(0.9, "EXPLANATION"))
    assert r["decision"] == "REJECT" and any("not verified" in x for x in r["reasons"])


def test_gate_rejects_small_gain_mismatched_inputs_and_regression():
    assert promotion_gate(ev(0.5), ev(0.52))["decision"] == "REJECT"
    assert promotion_gate(ev(0.5), ev(0.9, split="regression"))["decision"] == "REJECT"
    assert promotion_gate(ev(0.5), ev(0.4))["decision"] == "REJECT"
    assert promotion_gate({"score": "x"}, ev(0.4))["decision"] == "REJECT"
    assert promotion_gate(ev(float("nan")), ev(0.4))["decision"] == "REJECT"


def test_sealed_split_enforced_and_stub_labelled(tmp_path):
    t, _ = tools(tmp_path)
    assert "error" in t["run_benchmark"]("base", split="heldout")
    assert t["run_benchmark"]("base")["evidence"] == "EXPLANATION"
    assert "error" in t["evaluate_checkpoint"]("base", split="train")


def test_launch_is_gated_on_hparams_dataset_and_budget(tmp_path):
    t, s = tools(tmp_path)
    assert t["launch_training"]({"lora_r": 3}, "d.jsonl")["stage"] == "hyperparameters"
    assert t["launch_training"](GOOD, "missing.jsonl")["stage"] == "dataset"
    assert t["save_dataset"]("d", [ROW])["ok"]
    ok = t["launch_training"](GOOD, "d.jsonl")
    assert ok["checkpoint"].startswith("stub-ckpt-") and ok["dataset_sha256"]
    # no price configured -> fail closed
    t2 = make_tools(LocalStubBackend(), Settings(state_dir=tmp_path, data_dir=s.data_dir))
    assert t2["launch_training"](GOOD, "d.jsonl")["stage"] == "budget"
    # cumulative cap: the first launch recorded spend
    t3 = make_tools(LocalStubBackend(), Settings(state_dir=tmp_path, data_dir=s.data_dir,
                                                 price_per_mtok=1.0, max_cost_usd=1e-9))
    assert t3["launch_training"](GOOD, "d.jsonl")["stage"] == "budget"


def test_stub_results_can_never_be_promoted(tmp_path):
    t, _ = tools(tmp_path)
    t["save_dataset"]("d", [ROW])
    ckpt = t["launch_training"](GOOD, "d.jsonl")["checkpoint"]
    out = t["decide_promotion"](t["evaluate_checkpoint"]("base"), t["evaluate_checkpoint"](ckpt))
    assert out["decision"] == "REJECT"
