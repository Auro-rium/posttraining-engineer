import json

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from posttrain_agent import telemetry
from posttrain_agent.config import Settings
from posttrain_agent.hf import hf_tools
from posttrain_agent.nebius import NebiusBackend


def aclient(handler):
    return lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_hf_search_and_info_and_not_found():
    seen = []

    def h(req: httpx.Request) -> httpx.Response:
        seen.append(str(req.url))
        if "nope" in req.url.path:
            return httpx.Response(404)
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json=[{"id": "a/b", "downloads": 5, "tags": ["x"] * 30}])
        return httpx.Response(200, json={"id": "a/b", "sha": "abc", "gated": False,
                                         "cardData": {"license": "apache-2.0"},
                                         "safetensors": {"total": 8_000_000_000}})

    t = hf_tools(Settings(), aclient(h))
    found = await t["hf_search"]("llama")
    assert found["results"][0]["id"] == "a/b" and len(found["results"][0]["tags"]) == 15
    info = await t["hf_model_info"]("a/b")
    assert info["license"] == "apache-2.0" and info["parameters"] == 8_000_000_000
    assert info["revision"] == "abc"
    assert "error" in await t["hf_model_info"]("nope/x")
    assert "error" in await t["hf_search"]("x", kind="spaces")
    assert any("sort=downloads" in u for u in seen)


def mock_nebius(tmp_path, routes, key="k"):
    s = Settings(state_dir=tmp_path, data_dir=tmp_path, suite_dir=tmp_path, nebius_api_key=key)
    calls = []

    def h(req: httpx.Request) -> httpx.Response:
        calls.append((req.method, req.url.path, req.content))
        assert req.headers["authorization"] == "Bearer k"
        for (m, p), resp in routes.items():
            if req.method == m and req.url.path.endswith(p):
                return httpx.Response(200, json=resp(req) if callable(resp) else resp)
        return httpx.Response(404, text="no route")

    return NebiusBackend(s, lambda: httpx.Client(transport=httpx.MockTransport(h))), calls


def test_nebius_train_uploads_then_creates_lora_job(tmp_path):
    f = tmp_path / "d.jsonl"
    f.write_text("{}\n")
    b, calls = mock_nebius(tmp_path, {("POST", "/files"): {"id": "file-1"},
                                      ("POST", "/fine_tuning/jobs"): {"id": "ftjob-1", "status": "queued"}})
    out = b.train({"lora_r": 16, "n_epochs": 2}, f, "meta-llama/Llama-3.1-8B-Instruct")
    assert out["provider_job_id"] == "ftjob-1" and out["evidence"] == "LIVE"
    body = json.loads(calls[1][2])
    assert body["training_file"] == "file-1" and body["model"].startswith("meta-llama")
    assert body["hyperparameters"] == {"lora_r": 16, "n_epochs": 2, "lora": True}
    assert b"purpose" in calls[0][2]


def test_nebius_errors_block_not_crash(tmp_path):
    f = tmp_path / "d.jsonl"
    f.write_text("{}\n")
    b, _ = mock_nebius(tmp_path, {})
    assert b.train({}, f, "m")["status"] == "BLOCKED"
    nokey, _ = mock_nebius(tmp_path, {}, key="")
    assert "NEBIUS_API_KEY" in nokey.job_status("j")["error"]


def test_nebius_status_checkpoint_and_events(tmp_path):
    b, _ = mock_nebius(tmp_path, {
        ("GET", "/fine_tuning/jobs/j1"): {"status": "succeeded", "trained_steps": 9, "total_steps": 9},
        ("GET", "/fine_tuning/jobs/j1/checkpoints"): {"data": [
            {"fine_tuned_model_checkpoint": "ft:m:ckpt-step-1"}, {"fine_tuned_model_checkpoint": "ft:m:ckpt-step-9"}]},
        ("GET", "/fine_tuning/jobs/j1/events"): {"data": [{"level": "info", "message": "m" * 500}]},
    })
    st = b.job_status("j1")
    assert st["status"] == "succeeded" and st["checkpoint"] == "ft:m:ckpt-step-9"
    assert len(b.job_events("j1")["events"][0]["message"]) == 300


def test_nebius_evaluation_scores_exact_match_and_keeps_heldout_sealed(tmp_path):
    task = lambda i, name: json.dumps({"id": i, "messages": [{"role": "user", "content": "q"}],
                                       "expected": {"name": name, "arguments": {"service": "api"}}})
    (tmp_path / "heldout.jsonl").write_text(task("h1", "get_logs") + "\n" + task("h2", "restart_service") + "\n")
    (tmp_path / "train.jsonl").write_text(task("t1", "restart_service") + "\n")

    def completion(req):
        return {"choices": [{"message": {"tool_calls": [{"function": {
            "name": "get_logs", "arguments": json.dumps({"service": "api"})}}]}}]}

    b, calls = mock_nebius(tmp_path, {("POST", "/chat/completions"): completion})
    ev = b.evaluate("base", "s", "heldout", 7)
    assert ev["score"] == 0.5 and ev["n"] == 2 and "failures" not in ev and ev["evidence"] == "LIVE"
    bench = b.benchmark("ft:x", "s", 7)
    assert bench["score"] == 0.0 and bench["failures"][0]["task_id"] == "t1"
    assert json.loads(calls[0][2])["model"] == b.s.base_model  # 'base' alias
    assert "error" in b.evaluate("base", "s", "regression", 7)


def test_telemetry_handler_and_summary(tmp_path):
    from uuid import uuid4

    path = tmp_path / "t.jsonl"
    h = telemetry.TelemetryHandler(path)
    r1, r2, r3 = uuid4(), uuid4(), uuid4()
    h.on_chat_model_start({}, [[HumanMessage("secret prompt")]], run_id=r1,
                          metadata={"ls_provider": "backboard", "ls_model_name": "m"})
    msg = AIMessage("secret reply", usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14})
    h.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]), run_id=r1)
    h.on_chat_model_start({}, [[]], run_id=r2, metadata={"ls_provider": "backboard", "ls_model_name": "m"})
    h.on_llm_error(TimeoutError("x"), run_id=r2)
    h.on_tool_start({"name": "recall"}, "{}", run_id=r3)
    h.on_tool_end("ok", run_id=r3)
    raw = path.read_text()
    assert "secret" not in raw  # prompts/completions are never logged
    s = telemetry.summarize(path)
    m = s["by_name"]["model:backboard:m"]
    assert m["calls"] == 2 and m["errors"] == 1 and m["input_tokens"] == 10 and m["output_tokens"] == 4
    assert s["by_name"]["tool:recall"]["calls"] == 1
    assert s["error_types"] == {"TimeoutError": 1}
    assert "by_name" not in telemetry.summarize(path, only_errors=True)
    assert telemetry.summarize(tmp_path / "none.jsonl")["events"] == 0
    assert telemetry.telemetry_tool(path)()["events"] == 3
