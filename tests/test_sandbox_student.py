"""Sandbox CPU serving: pure adapter/tool-call logic, the student with a fake ConTree client, and
NebiusBackend routing (serverless vs sandbox) with a fake httpx transport. No network."""

import asyncio
import json
import subprocess
import sys
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.tools import StructuredTool

from posttrain_agent.config import Settings
from posttrain_agent.experiments import games
from posttrain_agent.experiments import sandbox_student as ss
from posttrain_agent.experiments.domain import LocalStubBackend, make_tools, promotion_gate
from posttrain_agent.experiments.nebius import NebiusBackend

KEY = "nb-secretsecret123"
BASE = "Qwen/Qwen3-0.6B"
CKPT = "ft:Qwen/Qwen3-0.6B-2026-09-30:org:games-v1:ID:ckpt-step-6"
CFG = json.dumps({"base_model_name_or_path": BASE, "r": 8, "peft_type": "LORA"}).encode()


# ------------------------------------------------------------------ pure logic

def test_peft_state_keys_adds_the_missing_prefix_once():
    keys = ["model.layers.0.self_attn.q_proj.lora_A.weight",
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]
    assert ss.peft_state_keys(keys) == {
        keys[0]: "base_model.model." + keys[0], keys[1]: keys[1]}


def test_adapter_load_problem_is_strict_about_lora_keys_only():
    state = ["base_model.model.model.layers.0.q_proj.lora_A.weight",
             "base_model.model.model.layers.0.q_proj.lora_B.weight"]
    base_missing = ["base_model.model.model.layers.0.q_proj.base_layer.weight"]
    assert ss.adapter_load_problem(state, [], base_missing) is None  # base weights: expected
    assert "1 unexpected" in ss.adapter_load_problem(state, ["model.x.lora_A.weight"], [])
    assert "1 missing lora" in ss.adapter_load_problem(
        state, [], ["base_model.model.model.layers.1.q_proj.lora_A.default.weight"])
    assert "no LoRA tensors" in ss.adapter_load_problem([], [], [])
    assert "no LoRA tensors" in ss.adapter_load_problem(["model.embed.weight"], [], [])


def test_generation_script_ships_the_tested_helpers_and_compiles():
    for fn in ("def peft_state_keys", "def adapter_load_problem", "set_peft_model_state_dict",
               "enable_thinking=False", "do_sample=False", "skip_special_tokens=False",
               "merge_and_unload", ss.OUT_MARKER):
        assert fn in ss.GEN_SCRIPT, fn
    compile(ss.GEN_SCRIPT, "gen.py", "exec")
    # really run the script's head + spliced helpers in a clean interpreter (no package imports,
    # as in the sandbox): catches names the helpers need that the script does not define
    head = ss.GEN_SCRIPT[:ss.GEN_SCRIPT.index(ss._GEN_BODY)] + (
        "\nprint(json.dumps([peft_state_keys(['m.lora_A.weight']), len(batch),"
        " adapter_load_problem(['x.lora_A.weight'], [], [])]))\n")
    p = subprocess.run([sys.executable, "-I", "-c", head, "--base", "/models/base"],
                       input='{"items": []}', capture_output=True, text=True, check=False)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout) == [{"m.lora_A.weight": "base_model.model.m.lora_A.weight"}, 0,
                                    None]
    assert "--adapter /work/adapter" in ss.gen_command(160, True)
    assert "--adapter" not in ss.gen_command(160, False)
    assert "--index-url https://download.pytorch.org/whl/cpu" in ss.setup_shell(BASE)


def test_tool_call_parsing_handles_qwen_output_shapes():
    one = '<tool_call>\n{"name": "take_stones", "arguments": {"count": 3}}\n</tool_call><|im_end|>'
    assert ss.first_tool_call(one) == {"name": "take_stones", "arguments": {"count": 3}}
    think = "<think>\nhmm\n</think>\n\n" + one
    assert ss.first_tool_call(think)["arguments"] == {"count": 3}
    as_str = '<tool_call>{"name": "f", "arguments": "{\\"x\\": 1}"}</tool_call>'
    assert ss.first_tool_call(as_str) == {"name": "f", "arguments": {"x": 1}}
    unclosed = '<tool_call>{"name": "f", "arguments": {"x": 2}}'  # cut at max_new_tokens
    assert ss.first_tool_call(unclosed)["arguments"] == {"x": 2}
    two = ('<tool_call>{"name": "a", "arguments": {}}</tool_call>'
           '<tool_call>{"name": "b", "arguments": {}}</tool_call>')
    assert [c["name"] for c in ss.parse_tool_calls(two)] == ["a", "b"]
    assert ss.first_tool_call('<tool_call>{"name": "f", "arguments": 5}</tool_call>') == {
        "name": "f", "arguments": None}
    assert ss.first_tool_call("<tool_call>not json</tool_call>") is None
    assert ss.first_tool_call('{"name": "f", "arguments": {}}') is None  # no tags: no call
    assert ss.first_tool_call("I would take 3 stones.") is None
    after_end = 'ok<|im_end|><tool_call>{"name": "f", "arguments": {}}</tool_call>'
    assert ss.first_tool_call(after_end) is None


def test_prepare_adapter_validates_base_and_renames_weights():
    ad = ss.prepare_adapter({"adapter_config.json": CFG, "weights.safetensors": b"W",
                             "tokenizer.json": b"{}"}, BASE)
    assert set(ad.files) == {"adapter_config.json", "adapter_model.safetensors"}
    assert ad.files["adapter_model.safetensors"] == b"W" and len(ad.sha256) == 64
    with pytest.raises(ss.StudentError, match="expected 'other/model'"):
        ss.prepare_adapter({"adapter_config.json": CFG, "a.safetensors": b"W"}, "other/model")
    with pytest.raises(ss.StudentError, match="no adapter_config.json"):
        ss.prepare_adapter({"a.safetensors": b"W"}, BASE)
    with pytest.raises(ss.StudentError, match="no single adapter weights"):
        ss.prepare_adapter({"adapter_config.json": CFG, "a.safetensors": b"1",
                            "b.safetensors": b"2"}, BASE)


# ------------------------------------------------------------------ fake ConTree


def _answer_for(tasks):
    by_prompt = {t["messages"][0]["content"]: t for t in tasks}

    def answer(item):
        exp = by_prompt[item["messages"][0]["content"]]["expected"]
        return f"<tool_call>\n{json.dumps(exp)}\n</tool_call><|im_end|>"
    return answer


class FakeContree:
    """Records every sandbox call; generation jobs answer through ``answer(item)``."""

    def __init__(self, answer=lambda item: "", job_exit=0, sample_error=None):
        self.answer, self.job_exit, self.sample_error = answer, job_exit, sample_error
        self.runs, self.uses = [], []
        self.inflight = self.max_inflight = 0
        self.images = SimpleNamespace(use=self._use)

    async def _use(self, ref, strict=False):
        self.uses.append((ref, strict))
        return SimpleNamespace(run=lambda **kw: self._run(ref, kw))

    async def _run(self, ref, kw):
        self.runs.append({"ref": ref, **kw})
        if not kw["disposable"]:
            n = sum(not r["disposable"] for r in self.runs)
            return SimpleNamespace(exit_code=0, uuid=f"img-{n}", stdout="", stderr="")
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        await asyncio.sleep(0.01)
        self.inflight -= 1
        if self.job_exit:
            return SimpleNamespace(exit_code=self.job_exit, uuid=None, stdout="",
                                   stderr="ADAPTER_LOAD_FAILED: 3 unexpected keys")
        items = json.loads(kw["stdin"])["items"]
        res = [{"text": "", "error": self.sample_error} if self.sample_error else
               {"text": self.answer(it), "error": None, "new_tokens": 20, "seconds": 0.1}
               for it in items]
        out = {"results": res, "load_s": 1.0, "gen_s": 0.5, "n": len(items),
               "versions": {"torch": "x"}, "adapter_tensors": 2 if "--adapter" in kw["shell"]
               else None}
        return SimpleNamespace(exit_code=0, uuid=None, stdout="log\n" + ss.OUT_MARKER +
                               json.dumps(out) + "\n", stderr="")


def _settings(tmp_path, **kw):
    return Settings(nebius_api_key=KEY, nebius_project_id="proj", base_model=BASE,
                    suite_dir=tmp_path / "suite", eval_judge="local",
                    nebius_base_url="https://api.example/v1/", **kw)


def test_student_builds_once_batches_bounded_and_never_ships_the_key(tmp_path):
    tasks = games.make_split(3, 37)
    fake = FakeContree(_answer_for(tasks))
    st = ss.SandboxStudent(_settings(tmp_path), lambda: fake, batch_size=5, concurrency=10)
    adapter = ss.prepare_adapter({"adapter_config.json": CFG, "w.safetensors": b"W"}, BASE)
    outs = st.generate(tasks)
    outs2 = st.generate(tasks, adapter)
    st.generate(tasks[:3], adapter)
    assert len(outs) == len(outs2) == 37
    assert ss.first_tool_call(outs[0]["text"]) == tasks[0]["expected"]
    builds = [r for r in fake.runs if not r["disposable"]]
    assert len(builds) == 2  # one heavy image, one adapter layer, both reused
    assert builds[0]["ref"] == "python:3.12-slim" and ss.SCRIPT_PATH in builds[0]["files"]
    assert builds[1]["ref"] == "img-1" and set(builds[1]["files"]) == {
        "/work/adapter/adapter_config.json", "/work/adapter/adapter_model.safetensors"}
    assert fake.max_inflight <= ss.CONCURRENCY  # capped at 4 whatever was asked
    jobs = [r for r in fake.runs if r["disposable"]]
    assert len(jobs) == 8 + 8 + 1 and all(r["timeout"] <= 240 + 45 * 5 for r in jobs)
    assert all(r["ref"] == "img-2" for r in jobs[8:])
    blob = json.dumps([{k: (str(v) if k == "files" else v) for k, v in r.items()}
                       for r in fake.runs])
    assert KEY not in blob and "proj" not in blob and "env" not in fake.runs[0]
    assert all(b"nb-secret" not in c for r in builds for c in r["files"].values())


def test_student_failures_raise_with_partial_counts_never_outputs(tmp_path):
    tasks = games.make_split(3, 6)
    with pytest.raises(ss.StudentError, match="ADAPTER_LOAD_FAILED") as e:
        ss.SandboxStudent(_settings(tmp_path), lambda: FakeContree(job_exit=1)).generate(tasks)
    assert e.value.done == 0 and e.value.total == 6
    with pytest.raises(ss.StudentError, match="samples failed"):
        ss.SandboxStudent(_settings(tmp_path),
                          lambda: FakeContree(sample_error="OOM")).generate(tasks)


# ------------------------------------------------------------------ backend routing


def _write_suite(tmp_path, n=8):
    (tmp_path / "suite").mkdir()
    tasks = games.make_split(2, n)
    for split in ("heldout", "train"):
        (tmp_path / "suite" / f"{split}.jsonl").write_text(
            "".join(json.dumps(t) + "\n" for t in tasks))
    return tasks


def _nebius(models, calls, chat=None):
    files = {"file-cfg": ("adapter_config.json", CFG),
             "file-w": ("checkpoints/step6/adapter_model.safetensors", b"WEIGHTS")}

    def h(req: httpx.Request) -> httpx.Response:
        p = req.url.path
        calls.append((req.method, p))
        assert req.headers["authorization"] == f"Bearer {KEY}"
        assert "//" not in p  # base url's trailing slash is stripped
        if p == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": m} for m in models]})
        if p == "/v1/chat/completions" and chat:
            return httpx.Response(200, json=chat(json.loads(req.content)))
        if p == "/v1/fine_tuning/jobs":
            return httpx.Response(200, json={"data": [
                {"id": "ftjob-old", "status": "failed"}, {"id": "ftjob-2", "status": "succeeded"},
                {"id": "ftjob-1", "status": "succeeded"}]})
        if p == "/v1/fine_tuning/jobs/ftjob-2/checkpoints":
            return httpx.Response(200, json={"data": [{"fine_tuned_model_checkpoint": "ft:other",
                                                       "result_files": ["file-x"]}]})
        if p == "/v1/fine_tuning/jobs/ftjob-1/checkpoints":
            return httpx.Response(200, json={"data": [
                {"fine_tuned_model_checkpoint": CKPT, "result_files": ["file-cfg", "file-w"]}]})
        if p.startswith("/v1/files/"):
            fid = p.split("/")[3]
            if p.endswith("/content"):
                return httpx.Response(200, content=files[fid][1])
            return httpx.Response(200, json={"id": fid, "filename": files[fid][0]})
        return httpx.Response(404, text="no route")
    return lambda: httpx.Client(transport=httpx.MockTransport(h))


def test_auto_serves_unlisted_base_and_checkpoint_in_the_sandbox(tmp_path):
    tasks = _write_suite(tmp_path)
    fake, calls = FakeContree(_answer_for(tasks)), []
    s = _settings(tmp_path)
    be = NebiusBackend(s, _nebius(["Qwen/Qwen3-30B-A3B-Instruct-2507"], calls),
                       student=ss.SandboxStudent(s, lambda: fake))
    base = be.evaluate("base", "s", "heldout", 7)
    cand = be.evaluate(CKPT, "s", "heldout", 7)
    bench = be.benchmark("base", "s", 7)
    for r in (base, cand, bench):
        assert r["served_by"] == "nebius-sandbox-cpu" and r["judged_by"] == "local-env"
        assert r["evidence"] == "LIVE" and r["score"] == 1.0 and r["n"] == 8
        assert "not served by /chat/completions" in r["serve_note"]
    assert ("POST", "/v1/chat/completions") not in calls
    assert calls.count(("GET", "/v1/models")) == 1  # probed once, cached
    assert ("GET", "/v1/files/file-w/content") in calls
    layer = [r for r in fake.runs if not r["disposable"]][1]
    assert layer["files"]["/work/adapter/adapter_model.safetensors"] == b"WEIGHTS"
    assert any("--adapter" in r["shell"] for r in fake.runs if r["disposable"])
    assert promotion_gate(base, {**cand, "score": 1.0})["decision"] == "REJECT"  # no gain


def test_auto_keeps_serverless_for_a_listed_base_model(tmp_path):
    tasks = _write_suite(tmp_path)
    by_prompt = {t["messages"][0]["content"]: t for t in tasks}

    def chat(body):
        exp = by_prompt[body["messages"][0]["content"]]["expected"]
        return {"choices": [{"message": {"tool_calls": [{"function": {
            "name": exp["name"], "arguments": json.dumps(exp["arguments"])}}]}}]}

    fake, calls = FakeContree(), []
    s = _settings(tmp_path)
    be = NebiusBackend(s, _nebius([BASE], calls, chat), student=ss.SandboxStudent(s, lambda: fake))
    r = be.evaluate("base", "s", "heldout", 7)
    assert r["served_by"] == "nebius-serverless" and r["score"] == 1.0 and "serve_note" not in r
    assert not fake.runs


def test_sandbox_failures_are_errors_with_counts_not_scores(tmp_path):
    _write_suite(tmp_path)
    s = _settings(tmp_path, serve="sandbox")
    broken = NebiusBackend(s, _nebius([], []),
                           student=ss.SandboxStudent(s, lambda: FakeContree(job_exit=1)))
    r = broken.evaluate("base", "s", "heldout", 7)
    assert "score" not in r and r["stage"] == "inference" and r["served_by"] == "nebius-sandbox-cpu"
    assert r["n_tasks"] == 8 and r["n_generated"] == 0 and "ADAPTER_LOAD_FAILED" in r["error"]
    missing = NebiusBackend(s, _nebius([], []), student=ss.SandboxStudent(s, lambda: FakeContree()))
    m = missing.evaluate("ft:nope", "s", "heldout", 7)
    assert m["stage"] == "checkpoint" and "not found" in m["error"] and "score" not in m
    unconfigured = NebiusBackend(Settings(nebius_api_key=KEY, suite_dir=s.suite_dir,
                                          serve="sandbox"), _nebius([], []))
    assert "NEBIUS_PROJECT_ID" in unconfigured.evaluate("base", "s", "heldout", 7)["error"]


def test_job_status_remembers_checkpoint_files_for_sandbox_serving(tmp_path):
    calls = []
    s = _settings(tmp_path)
    be = NebiusBackend(s, _nebius([], calls))
    be._req = lambda m, p, **kw: (  # noqa: SLF001 - minimal stub for one status call
        {"status": "succeeded"} if p == "/fine_tuning/jobs/ftjob-1" else
        {"data": [{"fine_tuned_model_checkpoint": CKPT, "result_files": [{"id": "file-w"}]}]})
    assert be.job_status("ftjob-1")["checkpoint"] == CKPT
    assert be._resolve_checkpoint(CKPT) == {"job_id": "ftjob-1", "result_files": ["file-w"]}


def test_gate_never_compares_two_serving_paths():
    def res(score, served):
        return {"score": score, "n": 400, "evidence": "LIVE", "suite": "s", "split": "heldout",
                "seed": 7, "judged_by": "nebius-sandbox", "served_by": served}
    mixed = promotion_gate(res(0.4, "nebius-serverless"), res(0.9, "nebius-sandbox-cpu"))
    assert mixed["decision"] == "REJECT" and "served_by differs" in mixed["reasons"][0]
    same = promotion_gate(res(0.4, "nebius-sandbox-cpu"), res(0.9, "nebius-sandbox-cpu"))
    assert same["decision"] == "PROMOTE"


# ------------------------------------------------------------------ validate_hparams tool


def test_validate_hparams_tool_accepts_nested_and_flat_calls_through_langchain(tmp_path):
    fn = make_tools(LocalStubBackend(), Settings(state_dir=tmp_path))["validate_hparams"]
    tool = StructuredTool.from_function(fn)
    user_06b = {"lora_r": 8, "lora_alpha": 8, "lora_dropout": 0.0, "learning_rate": 1e-5,
                "n_epochs": 1, "batch_size": 8, "context_length": 8192, "packing": True}
    assert tool.invoke({"hyperparameters": user_06b})["valid"] is True
    flat = tool.invoke(user_06b)  # the call shape that used to crash
    assert flat["valid"] is True and flat["checked"] == user_06b
    bad = tool.invoke({**user_06b, "lora_r": 8.5})  # reaches our validator, not pydantic
    assert bad["valid"] is False and any("lora_r" in e for e in bad["errors"])
    assert tool.invoke({})["valid"] is False  # nothing given: missing keys, no TypeError
    for r, bs in ((8, 4), (16, 8), (32, 16)):  # 0.6B ranges; no tokens-per-batch minimum
        assert tool.invoke({**user_06b, "lora_r": r, "lora_alpha": 2 * r, "batch_size": bs})[
            "valid"]
