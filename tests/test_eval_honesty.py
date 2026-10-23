"""Phase C: retries, structured transient errors, parse-rate preflight, persisted generations,
fail-closed judging, the regression/paired/parse-rate gate, checkpoint ordering, the ledger."""

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from test_sandbox_student import BASE, CFG, FakeContree, _answer_for  # noqa: F401

from posttrain_agent.config import Settings
from posttrain_agent.experiments import game_env, games, ledger, retry
from posttrain_agent.experiments import sandbox as sandbox_mod
from posttrain_agent.experiments import sandbox_student as ss
from posttrain_agent.experiments.domain import make_tools, promotion_gate
from posttrain_agent.experiments.gate import sign_test_p
from posttrain_agent.experiments.nebius import NebiusBackend, final_checkpoint
from posttrain_agent.experiments.sandbox import SandboxJudge, SandboxUnavailable
from posttrain_agent.observability.events import CURRENT_RUN


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr(retry, "BASE_DELAY", 0.0)


class ApiTimeoutError(Exception):  # same class name as contree_sdk's; matched by name
    pass


def be(tmp_path, handler, **kw):
    s = Settings(state_dir=tmp_path, data_dir=tmp_path, suite_dir=tmp_path / "suite",
                 nebius_api_key="k", nebius_base_url="https://api.example/v1/",
                 **({"serve": "serverless"} | kw))
    b = NebiusBackend(s, lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    b._sleep = lambda _s: None
    return b


# ------------------------------------------------------------------ retries


def test_req_retries_dns_and_5xx_then_succeeds(tmp_path):
    seen = []

    def h(req):
        seen.append(1)
        if len(seen) == 1:
            raise httpx.ConnectError("[Errno -3] Temporary failure in name resolution")
        if len(seen) == 2:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"ok": 1})

    assert be(tmp_path, h)._req("GET", "/x") == {"ok": 1} and len(seen) == 3


def test_req_exhausted_is_a_structured_transient_error_not_a_value(tmp_path):
    seen = []

    def h(req):
        seen.append(1)
        raise httpx.ReadTimeout("slow")

    out = be(tmp_path, h)._req("GET", "/x")
    assert out["transient"] is True and out["stage"] == "request" and "ReadTimeout" in out["error"]
    assert len(seen) == retry.TRIES
    assert be(tmp_path, lambda r: httpx.Response(502))._req("GET", "/x")["transient"] is True


def test_4xx_is_not_retried_and_job_creation_is_never_retried(tmp_path):
    seen = []

    def h(req):
        seen.append(req.url.path)
        if req.url.path.endswith("/files"):
            return httpx.Response(200, json={"id": "f1"})
        return httpx.Response(503, text="x")

    b = be(tmp_path, h)
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"messages": []}) + "\n")
    out = b.train({"lora_r": 8}, p, "m")
    assert out["status"] == "BLOCKED" and out["stage"] == "create_job"
    assert seen.count("/v1/fine_tuning/jobs") == 1  # a lost response must not double-launch
    four = []
    assert "error" in be(tmp_path, lambda r: four.append(1) or httpx.Response(404))._req("GET", "/y")
    assert len(four) == 1


def test_get_bytes_retries_transport_errors(tmp_path):
    seen = []

    def h(req):
        seen.append(1)
        if len(seen) < 3:
            raise httpx.ConnectError("dns")
        return httpx.Response(200, content=b"w")

    assert be(tmp_path, h)._get_bytes("/files/f/content") == b"w"


def test_sandbox_judge_uses_the_600s_transport_timeout(monkeypatch):
    got = {}
    monkeypatch.setattr(sandbox_mod, "contree_client",
                        lambda s, transport_timeout=None, operation_timeout=None:
                        got.setdefault("t", transport_timeout))
    SandboxJudge(Settings())._client()
    assert got["t"] == 600.0 == ss.TRANSPORT_TIMEOUT_S


class FlakySandbox:
    def __init__(self, fail_times, exc=ApiTimeoutError):
        self.fail, self.exc, self.use_calls, self.run_calls = fail_times, exc, 0, 0
        self.images = SimpleNamespace(use=self._use)

    async def _use(self, image, strict=True):
        self.use_calls += 1
        if self.use_calls <= self.fail:
            raise self.exc("timeout")
        return SimpleNamespace(run=self._run)

    async def _run(self, shell, disposable, timeout=None):
        self.run_calls += 1
        import subprocess
        p = subprocess.run(["bash", "-c", shell], capture_output=True, text=True, check=False)
        return SimpleNamespace(exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)


def _items(n=3):
    return [{"id": t["id"], "state": t["state"], **t["expected"]} | {"arguments": t["expected"]["arguments"]}
            for t in games.make_split(1, n)]


def test_sandbox_judge_retries_images_use_then_succeeds_and_exhaustion_is_transient():
    items = [{"id": t["id"], "state": t["state"], "name": t["expected"]["name"],
              "arguments": t["expected"]["arguments"]} for t in games.make_split(1, 3)]
    fake = FlakySandbox(2)
    assert all(r["ok"] for r in SandboxJudge(Settings(), lambda: fake).judge(items))
    with pytest.raises(SandboxUnavailable) as e:
        SandboxJudge(Settings(), lambda: FlakySandbox(99)).judge(items)
    assert e.value.transient is True


def test_student_retries_build_and_each_batch_individually_and_keeps_finished_batches(tmp_path):
    tasks = games.make_split(3, 12)
    fake = FakeContree(_answer_for(tasks))
    real_use, state = fake._use, {"fail": 2}

    async def use(ref, strict=False):  # the image lookup times out twice, then works
        if state["fail"]:
            state["fail"] -= 1
            raise ApiTimeoutError("t")
        return await real_use(ref, strict)

    fake.images.use = use
    s = Settings(nebius_api_key="k", nebius_project_id="p", base_model=BASE)
    st = ss.SandboxStudent(s, lambda: fake, batch_size=4)
    assert len(st.generate(tasks)) == 12

    # one batch keeps failing: the others are returned in .partial, the error is transient
    bad = FakeContree(_answer_for(tasks))
    orig = bad._run

    async def run(ref, kw):
        if kw["disposable"] and json.dumps(tasks[4]["messages"][-1]["content"])[1:-1] in kw["stdin"]:
            raise ApiTimeoutError("dead batch")
        return await orig(ref, kw)

    bad._run = run
    with pytest.raises(ss.StudentError) as e:
        ss.SandboxStudent(s, lambda: bad, batch_size=4).generate(tasks)
    assert e.value.transient and e.value.done == 8
    assert [o is None for o in e.value.partial] == [False] * 4 + [True] * 4 + [False] * 4


def test_retry_does_not_retry_deterministic_errors():
    calls = []

    def boom():
        calls.append(1)
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        retry.retry_sync(boom)
    assert len(calls) == 1
    assert asyncio.run(retry.retry_async(lambda: asyncio.sleep(0, "ok"))) == "ok"


# ------------------------------------------------------------------ evaluation honesty


def _suite(tmp_path, n=8, seed=2):
    (tmp_path / "suite").mkdir(exist_ok=True)
    tasks = games.make_split(seed, n)
    for split in ("heldout", "regression"):
        (tmp_path / "suite" / f"{split}.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    return tasks


def _wrong_legal(task):
    game = task["state"].split(":")[0]
    pool = {"ttt": range(9), "nim": range(1, 4), "guess": range(0, 400),
            "rps": ["rock", "paper", "scissors"]}[game]
    arg = game_env.TOOLS[game][1]
    for v in pool:
        ok, reason = game_env.judge(task["state"], task["expected"]["name"], {arg: v})
        if not ok and not reason.startswith("illegal"):
            return {arg: v}
    raise AssertionError


def _chat(tasks, behave):
    by = {t["messages"][-1]["content"]: (i, t) for i, t in enumerate(tasks)}

    def handler(req):
        body = json.loads(req.content)
        i, t = by[body["messages"][-1]["content"]]
        kind = behave(i)
        if kind == "prose":
            return httpx.Response(200, json={"choices": [{"message": {"content": "Guessed 82."}}]})
        name, args = t["expected"]["name"], t["expected"]["arguments"]
        if kind == "tool":
            name = "other_tool"
        elif kind == "illegal":
            args = {}
        elif kind == "wrong":
            args = _wrong_legal(t)
        call = {"function": {"name": name, "arguments": json.dumps(args)}}
        return httpx.Response(200, json={"choices": [{"message": {"tool_calls": [call]}}]})
    return handler


def test_failure_kinds_parse_rate_and_per_task_verdicts(tmp_path):
    tasks = _suite(tmp_path, 8)
    kinds = ["ok", "ok", "ok", "prose", "tool", "illegal", "wrong", "ok"]
    r = be(tmp_path, _chat(tasks, lambda i: kinds[i]), eval_judge="local").evaluate(
        "base", "s", "heldout", 7)
    assert r["n"] == 8 and r["n_parsed"] == 7 and r["parse_rate"] == 0.875
    assert (r["n_unparsed"], r["n_wrong_tool"], r["n_illegal"], r["n_wrong_move"]) == (1, 1, 1, 1)
    assert r["score"] == 0.5 and r["verdicts"] == "11100001"[:3] + "00" + "00" + "1"
    assert {f["kind"] for f in r["failures"]} == {"unparsed", "wrong_tool", "illegal", "wrong_move"}
    assert "expected" not in json.dumps(r["failures"])  # held-out tasks stay sealed


def test_prose_only_adapter_fails_preflight_clearly_not_as_a_zero_score(tmp_path):
    tasks = _suite(tmp_path, 40)
    r = be(tmp_path, _chat(tasks, lambda i: "prose"), eval_judge="local").evaluate(
        "ft:x", "s", "heldout", 7)
    assert "score" not in r and r["stage"] == "preflight" and r["transient"] is False
    assert r["parse_rate"] == 0.0 and "does not emit tool calls" in r["error"]
    assert "Guessed" in r["sample_output"][0] or r["sample_output"]


def test_preflight_generates_only_a_sample_first(tmp_path):
    tasks = _suite(tmp_path, 40)
    seen = []

    def behave(i):
        seen.append(i)
        return "prose"

    be(tmp_path, _chat(tasks, behave), eval_judge="local").evaluate("ft:x", "s", "heldout", 7)
    assert 0 < len(seen) <= 16


def test_generations_are_persisted_and_reused_after_a_judge_failure(tmp_path):
    tasks = _suite(tmp_path, 8)
    calls = []
    inner = _chat(tasks, lambda i: "ok")

    def handler(req):
        calls.append(1)
        return inner(req)

    broken = SandboxJudge(Settings(), lambda: FlakySandbox(99))
    b = be(tmp_path, handler, eval_judge="auto")
    b._sandbox = broken
    token = CURRENT_RUN.set("run-a")
    try:
        r = b.evaluate("base", "s", "heldout", 7)
        assert "score" not in r and r["stage"] == "judge" and r["transient"] is True
        gen = tmp_path / "operator_evals" / "run-a-heldout-gen.jsonl"
        assert r["gen_file"] == str(gen) and len(gen.read_text().splitlines()) == 8
        assert len(calls) == 8
        b._sandbox = SandboxJudge(Settings(), lambda: FlakySandbox(0))  # judge recovers
        ok = b.evaluate("base", "s", "heldout", 7)
        assert ok["score"] == 1.0 and ok["judged_by"] == "nebius-sandbox"
        assert len(calls) == 8  # nothing regenerated
        # a fresh backend (a rerun in a new process) reuses the file too
        b2 = be(tmp_path, handler, eval_judge="local")
        assert b2.evaluate("base", "s", "heldout", 7)["score"] == 1.0 and len(calls) == 8
    finally:
        CURRENT_RUN.reset(token)


def test_inference_outage_keeps_completed_generations_and_reports_transient(tmp_path):
    tasks = _suite(tmp_path, 8)
    inner, n = _chat(tasks, lambda i: "ok"), []

    def handler(req):
        n.append(1)
        if len(n) > 5:
            raise httpx.ConnectError("dns")
        return inner(req)

    token = CURRENT_RUN.set("run-b")
    try:
        r = be(tmp_path, handler, eval_judge="local").evaluate("base", "s", "heldout", 7)
    finally:
        CURRENT_RUN.reset(token)
    assert "score" not in r and r["transient"] is True and r["stage"] == "inference"
    assert r["n_tasks"] == 8 and r["n_generated"] == 5
    assert len((tmp_path / "operator_evals" / "run-b-heldout-gen.jsonl").read_text().splitlines()) == 5


def test_max_tasks_allows_the_recommended_split_sizes():
    from posttrain_agent.experiments import nebius
    assert nebius.MAX_TASKS >= 300 and nebius.MAX_TASKS >= 150


def test_final_checkpoint_is_chosen_by_step_not_list_order():
    items = [{"fine_tuned_model_checkpoint": f"ft:m:ckpt-step-{n}"} for n in (9, 3, 6)]
    assert final_checkpoint(items)["fine_tuned_model_checkpoint"].endswith("step-9")
    items = [{"fine_tuned_model_checkpoint": "a", "step_number": 2},
             {"fine_tuned_model_checkpoint": "b", "step_number": 10},
             {"fine_tuned_model_checkpoint": "c", "step_number": 4}]
    assert final_checkpoint(items)["fine_tuned_model_checkpoint"] == "b"
    assert final_checkpoint([]) is None


def test_job_status_returns_the_highest_step_checkpoint(tmp_path):
    job = {"id": "ftjob-1", "status": "succeeded", "error": None}
    ck = {"data": [{"fine_tuned_model_checkpoint": f"ft:m:ckpt-step-{n}", "result_files": []}
                   for n in (6, 9, 3)]}
    b = be(tmp_path, lambda r: httpx.Response(
        200, json=ck if r.url.path.endswith("/checkpoints") else job))
    assert b.job_status("ftjob-1")["checkpoint"] == "ft:m:ckpt-step-9"


# ------------------------------------------------------------------ the gate


def R(score, n=400, **kw):
    return {"score": score, "n": n, "evidence": "LIVE", "suite": "s", "split": "heldout",
            "seed": 7, "parse_rate": 1.0, "judged_by": "j", "served_by": "v", **kw}


def REG(score, n=150, **kw):
    return R(score, n, split="regression", **kw)


def test_regression_split_drop_blocks_promotion():
    ok = promotion_gate(R(0.4), R(0.8), champion_regression=REG(0.9), candidate_regression=REG(0.88))
    assert ok["decision"] == "PROMOTE" and ok["regression_checked"] is True
    bad = promotion_gate(R(0.4), R(0.8), champion_regression=REG(0.9), candidate_regression=REG(0.7))
    assert bad["decision"] == "REJECT" and any("regression split dropped" in r for r in bad["reasons"])
    only = promotion_gate(R(0.4), R(0.8), champion_regression=REG(0.9))
    assert only["decision"] == "REJECT"
    errored = promotion_gate(R(0.4), R(0.8), champion_regression=REG(0.9),
                             candidate_regression={"error": "x", "stage": "judge"})
    assert errored["decision"] == "REJECT"


def test_missing_regression_warns_unless_required():
    soft = promotion_gate(R(0.4), R(0.8))
    assert soft["decision"] == "PROMOTE" and soft["regression_checked"] is False and soft["warnings"]
    hard = promotion_gate(R(0.4), R(0.8), require_regression=True)
    assert hard["decision"] == "REJECT"


def test_missing_n_rejects_fail_closed():
    no_n = {k: v for k, v in R(0.4).items() if k != "n"}
    out = promotion_gate(no_n, R(0.9))
    assert out["decision"] == "REJECT" and any("n missing" in r for r in out["reasons"])


def test_prose_only_adapter_is_invalid_via_parse_rate_even_with_a_good_score():
    out = promotion_gate(R(0.4, parse_rate=1.0), R(0.8, parse_rate=0.5))
    assert out["decision"] == "REJECT" and any("parse_rate" in r for r in out["reasons"])
    assert promotion_gate(R(0.4), {k: v for k, v in R(0.8).items() if k != "parse_rate"}
                          )["decision"] == "REJECT"
    # base itself parsed 0.99: candidate may be at most 0.02 below it; floor is 0.90 otherwise
    assert promotion_gate(R(0.4, parse_rate=0.99), R(0.8, parse_rate=0.95))["decision"] == "REJECT"
    assert promotion_gate(R(0.4, parse_rate=0.5), R(0.8, parse_rate=0.92))["decision"] == "PROMOTE"
    assert promotion_gate(R(0.4, parse_rate=0.5), R(0.8, parse_rate=0.85))["decision"] == "REJECT"


def _verdicts(n, base_ok, cand_ok):
    return ("".join("1" if i in base_ok else "0" for i in range(n)),
            "".join("1" if i in cand_ok else "0" for i in range(n)))


def test_paired_exact_test_is_more_powerful_than_the_unpaired_z():
    n = 100
    a, b = _verdicts(n, set(range(50)), set(range(60)))  # candidate fixes 10, breaks none
    base = R(0.5, n, verdicts=a, tasks_sha256="t")
    cand = R(0.6, n, verdicts=b, tasks_sha256="t")
    paired = promotion_gate(base, cand)
    assert paired["decision"] == "PROMOTE" and paired["test"].startswith("paired")
    assert paired["p_value"] == pytest.approx(sign_test_p(10, 0), abs=1e-5)
    unpaired = promotion_gate({k: v for k, v in base.items() if k != "verdicts"},
                              {k: v for k, v in cand.items() if k != "verdicts"})
    assert unpaired["decision"] == "REJECT" and unpaired["test"].startswith("two-proportion")
    # same marginal scores but the candidate also breaks tasks: paired test is not significant
    a2, b2 = _verdicts(n, set(range(50)), set(range(40, 100)))  # fixes 50, breaks 40... -> wins 50
    mixed = promotion_gate(R(0.5, n, verdicts=a2, tasks_sha256="t"),
                           R(0.6, n, verdicts=b2, tasks_sha256="t"))
    assert mixed["decision"] == "REJECT" and any("paired exact" in r for r in mixed["reasons"])
    other = promotion_gate(R(0.5, n, verdicts=a, tasks_sha256="t"),
                           R(0.6, n, verdicts=b, tasks_sha256="u"))
    assert other["decision"] == "REJECT" and any("tasks_sha256" in r for r in other["reasons"])


def test_sign_test_values():
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(5, 5) > 0.5
    assert sign_test_p(10, 0) == pytest.approx(1 / 1024)


def test_gate_reports_mde_and_underpower_warning():
    out = promotion_gate(R(0.50, 100), R(0.60, 100))
    assert out["mde_80pct_power"] == pytest.approx(0.16, abs=0.02) and "power_note" in out
    assert any("below the minimum detectable effect" in w for w in out["warnings"])
    assert any("below the recommended 300" in w for w in out["warnings"])


# ------------------------------------------------------------------ ledger + operator script


class FakeBackend:
    def __init__(self, fail_regression=0):
        self.calls, self.fail_regression = [], fail_regression

    def job_status(self, job_id):
        return {"status": "succeeded", "checkpoint": "ft:m:ckpt-step-9"}

    def evaluate(self, checkpoint, suite, split, seed):
        self.calls.append((checkpoint, split))
        if split == "regression" and checkpoint != "base" and self.fail_regression:
            self.fail_regression -= 1
            return {"error": "dns", "stage": "inference", "transient": True}
        good = checkpoint != "base"
        return {"evidence": "LIVE", "checkpoint": checkpoint, "suite": suite, "split": split,
                "seed": seed, "score": 0.8 if good else 0.4, "n": 400, "parse_rate": 1.0,
                "judged_by": "j", "served_by": "v", "tasks_sha256": "t"}

    def benchmark(self, *a): ...
    def train(self, *a): ...
    def job_events(self, *a): ...
    def job_id(self, *a): ...


def _script():
    spec = importlib.util.spec_from_file_location(
        "eval_candidate", Path(__file__).parent.parent / "scripts" / "eval_candidate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_evaluate_checkpoint_appends_ok_and_error_records_to_the_ledger(tmp_path):
    s = Settings(state_dir=tmp_path, data_dir=tmp_path)
    t = make_tools(FakeBackend(fail_regression=1), s)
    t["evaluate_checkpoint"]("ftjob-1", "heldout")
    t["evaluate_checkpoint"]("ftjob-1", "regression")
    recs = ledger.read(tmp_path)
    assert [(r["split"], r["status"], r["job"], r["checkpoint"]) for r in recs] == [
        ("heldout", "ok", "ftjob-1", "ft:m:ckpt-step-9"),
        ("regression", "error", "ftjob-1", "ft:m:ckpt-step-9")]
    assert recs[1]["result"]["transient"] is True and "score" not in recs[1]["result"]


def test_eval_candidate_discards_both_splits_on_failure_and_resume_reuses_finished_ones(tmp_path):
    mod = _script()
    s = Settings(state_dir=tmp_path, data_dir=tmp_path)
    fb = FakeBackend(fail_regression=1)
    failed = mod.evaluate("run1", "ftjob-1", s, fb)
    assert failed["status"] == "FAILED" and "heldout" not in failed and "score" not in json.dumps(failed)
    assert failed["errors"]["regression"]["transient"] is True
    n_heldout = sum(1 for c in fb.calls if c == ("ft:m:ckpt-step-9", "heldout"))
    done = mod.evaluate("run1", "ftjob-1", s, fb, resume=True)
    assert done["status"] == "SUCCEEDED" and done["gate"]["decision"] == "PROMOTE"
    assert sum(1 for c in fb.calls if c == ("ft:m:ckpt-step-9", "heldout")) == n_heldout
    assert done["gate"]["regression_checked"] is True and done["base"]["heldout"]["score"] == 0.4
    assert done["heldout"]["evidence"] == "LIVE"  # taken from the result, not hard-coded
    # base results come from the ledger the second time around
    again = mod.evaluate("run2", "ftjob-1", s, fb)
    assert again["status"] == "SUCCEEDED"
    assert sum(1 for c in fb.calls if c[0] == "base") == 2  # evaluated once per split, ever
