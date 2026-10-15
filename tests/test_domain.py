from posttrain_agent.domain import (
    LocalStubBackend,
    make_tools,
    promotion_gate,
    validate_qlora,
)

GOOD = {"rank": 16, "alpha": 32, "dropout": 0.05, "learning_rate": 2e-4, "epochs": 3,
        "sequence_length": 1024, "batch_size": 4, "gradient_accumulation_steps": 4,
        "target_modules": ["q_proj", "v_proj"]}


def ev(score, evidence="LIVE", split="heldout"):
    return {"score": score, "evidence": evidence, "suite": "s", "split": split, "seed": 7}


def test_qlora_accepts_valid_and_rejects_out_of_space():
    assert validate_qlora(GOOD) == []
    assert validate_qlora({**GOOD, "rank": 3})
    assert validate_qlora({**GOOD, "learning_rate": 1.0})
    assert validate_qlora({**GOOD, "target_modules": ["lm_head"]})
    assert validate_qlora({**GOOD, "evil": 1})
    assert validate_qlora({**GOOD, "epochs": True})


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


def test_tools_enforce_sealed_split_and_stub_is_labelled():
    t = make_tools(LocalStubBackend())
    assert "error" in t["run_benchmark"]("base", split="heldout")
    assert t["run_benchmark"]("base")["evidence"] == "EXPLANATION"
    assert "error" in t["evaluate_checkpoint"]("base", split="train")
    assert t["launch_training"]({"rank": 3}, "d", "base")["status"] == "BLOCKED"
    trained = t["launch_training"](GOOD, "d", "base")
    assert trained["checkpoint"].startswith("stub-ckpt-") and trained["provider_job_id"] is None


def test_stub_results_can_never_be_promoted():
    t = make_tools(LocalStubBackend())
    champ = t["evaluate_checkpoint"]("base")
    cand = t["evaluate_checkpoint"](t["launch_training"](GOOD, "d", "base")["checkpoint"])
    assert t["decide_promotion"](champ, cand)["decision"] == "REJECT"
