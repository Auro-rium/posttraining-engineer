"""Resilience regressions: late checkpoint listings, adapter upload retries, operator resume."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import httpx
import pytest
from test_sandbox_student import (
    BASE,
    CFG,
    CKPT,
    FakeContree,
    _answer_for,
    _nebius,
    _settings,
    _write_suite,
)

from posttrain_agent.config import Settings
from posttrain_agent.experiments import games, ledger, nebius, retry
from posttrain_agent.experiments import sandbox_student as ss
from posttrain_agent.experiments.domain import make_tools
from posttrain_agent.experiments.nebius import NebiusBackend
from posttrain_agent.observability.events import CURRENT_RUN

JOB = {"id": "ftjob-1", "status": "succeeded", "error": None, "trained_steps": 9,
       "total_steps": 9, "trained_tokens": 90}


class ApiTimeoutError(Exception):  # same class name as the SDK's; matched by name
    pass


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr(retry, "BASE_DELAY", 0.0)


def _backend(tmp_path, handler):
    s = Settings(state_dir=tmp_path, data_dir=tmp_path, suite_dir=tmp_path / "suite",
                 nebius_api_key="k", nebius_base_url="https://api.example/v1/",
                 base_model=BASE)
    return NebiusBackend(s, lambda: httpx.Client(transport=httpx.MockTransport(handler)))


# ------------------------------------------------------------------ checkpoint listings


def test_job_status_retries_a_briefly_empty_checkpoint_list(tmp_path):
    listings = iter([[], [{"fine_tuned_model_checkpoint": "ft:m:ckpt-step-3"},
                           {"fine_tuned_model_checkpoint": "ft:m:ckpt-step-9"}]])
    calls, sleeps = [], []

    def h(req):
        calls.append((req.method, req.url.path))
        if req.url.path == "/v1/fine_tuning/jobs/ftjob-1":
            return httpx.Response(200, json=JOB)
        if req.url.path.endswith("/checkpoints"):
            return httpx.Response(200, json={"data": next(listings)})
        return httpx.Response(404, text="unexpected")

    b = _backend(tmp_path, h)
    b._sleep = sleeps.append  # tests must not wait for real backoff
    out = b.job_status("ftjob-1")
    assert out["status"] == "succeeded" and out["checkpoint"] == "ft:m:ckpt-step-9"
    assert out["error"] is None and "stage" not in out
    assert sum(path.endswith("/checkpoints") for _m, path in calls) == 2
    assert sleeps == [nebius.CHECKPOINT_BASE_DELAY_S]
    assert all(method == "GET" for method, _path in calls)  # never re-creates the job


def test_permanently_empty_checkpoint_list_is_an_error_not_a_score(tmp_path):
    calls, sleeps = [], []

    def h(req):
        calls.append((req.method, req.url.path))
        if req.url.path == "/v1/fine_tuning/jobs/ftjob-1":
            return httpx.Response(200, json=JOB)
        if req.url.path.endswith("/checkpoints"):
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404, text="unexpected")

    b = _backend(tmp_path, h)
    b._sleep = sleeps.append
    out = b.job_status("ftjob-1")
    assert out["checkpoint"] is None and out["status"] == "succeeded"
    assert out["stage"] == "checkpoint" and out["checkpoint_missing"] is True
    assert "no checkpoint" in out["error"] or "without a checkpoint" in out["error"]
    assert out["attempts"] == nebius.CHECKPOINT_TRIES
    assert "score" not in out
    assert sum(path.endswith("/checkpoints") for _m, path in calls) == nebius.CHECKPOINT_TRIES
    assert len(sleeps) == nebius.CHECKPOINT_TRIES - 1
    expected = [min(nebius.CHECKPOINT_MAX_DELAY_S,
                    nebius.CHECKPOINT_BASE_DELAY_S * 2 ** i)
                for i in range(nebius.CHECKPOINT_TRIES - 1)]
    assert sleeps == expected
    assert all(method == "GET" for method, _path in calls)

    # the tool layer must turn that status into an error, never a number
    res = make_tools(b, b.s)["evaluate_checkpoint"]("ftjob-1", "heldout")
    assert "score" not in res and res["stage"] == "checkpoint"
    assert ledger.read(tmp_path)[-1]["status"] == "error"


# ------------------------------------------------------------------ sandbox adapter uploads


def _flaky_upload(tasks, failures):
    """A fake sandbox whose adapter-layer build times out ``failures`` times (the setup build
    and all generation jobs work)."""
    fake = FakeContree(_answer_for(tasks))
    orig = fake._run

    async def run(ref, kw):
        if not kw["disposable"]:
            fake.builds = getattr(fake, "builds", 0) + 1
            if fake.builds > 1 and fake.failures:
                fake.failures -= 1
                raise ApiTimeoutError("write timeout")
        return await orig(ref, kw)

    fake.failures = failures
    fake._run = run
    return fake


def test_adapter_upload_timeout_is_retried_then_generation_proceeds(tmp_path):
    tasks = games.make_split(3, 6)
    fake = _flaky_upload(tasks, failures=1)
    st = ss.SandboxStudent(_settings(tmp_path, state_dir=tmp_path), lambda: fake,
                           batch_size=8)
    adapter = ss.prepare_adapter({"adapter_config.json": CFG, "w.safetensors": b"W"}, BASE)
    outs = st.generate(tasks, adapter)
    assert len(outs) == 6
    assert [ss.first_tool_call(o["text"]) for o in outs] == [t["expected"] for t in tasks]
    assert fake.builds == 3  # setup image, timed-out adapter upload, successful retry
    assert fake.failures == 0


def test_exhausted_adapter_upload_is_a_structured_error_with_counts(tmp_path):
    tasks = _write_suite(tmp_path, n=8)
    fake = _flaky_upload(tasks, failures=99)
    s = _settings(tmp_path, serve="sandbox", state_dir=tmp_path)
    be = NebiusBackend(s, _nebius([], []),
                       student=ss.SandboxStudent(s, lambda: fake, batch_size=8))
    r = be.evaluate(CKPT, "s", "heldout", 7)
    assert "score" not in r
    assert r["stage"] == "image" and r["transient"] is True
    assert r["n_tasks"] == 8 and r["n_generated"] == 0
    assert f"attempt {ss.UPLOAD_TRIES}/{ss.UPLOAD_TRIES}" in r["error"]
    assert fake.builds == 1 + ss.UPLOAD_TRIES  # setup build, then every adapter attempt


def test_partial_sandbox_generations_survive_and_only_the_rest_is_regenerated(tmp_path):
    tasks = _write_suite(tmp_path, n=8)
    dead = json.dumps(tasks[4]["messages"][-1]["content"])[1:-1]
    bad = FakeContree(_answer_for(tasks))
    orig = bad._run

    async def run(ref, kw):
        if kw["disposable"] and dead in kw["stdin"]:
            raise ApiTimeoutError("dead batch")
        return await orig(ref, kw)

    bad._run = run
    s = _settings(tmp_path, serve="sandbox", state_dir=tmp_path)
    token = CURRENT_RUN.set("run-partial")
    try:
        be = NebiusBackend(s, _nebius([], []),
                           student=ss.SandboxStudent(s, lambda: bad, batch_size=4))
        failed = be.evaluate("base", "s", "heldout", 7)
        assert "score" not in failed
        assert failed["n_tasks"] == 8 and failed["n_generated"] == 4
        gen = tmp_path / "operator_evals" / "run-partial-heldout-gen.jsonl"
        assert len(gen.read_text().splitlines()) == 4

        good = FakeContree(_answer_for(tasks))
        be._student = ss.SandboxStudent(s, lambda: good, batch_size=4)
        ok = be.evaluate("base", "s", "heldout", 7)
        assert ok["score"] == 1.0 and ok["n"] == 8
        jobs = [json.loads(r["stdin"])["items"] for r in good.runs if r["disposable"]]
        assert sum(len(items) for items in jobs) == 4  # only the unfinished batch
    finally:
        CURRENT_RUN.reset(token)


# ------------------------------------------------------------------ eval_candidate --resume


class _ResumeBackend:
    def __init__(self, fail_regression=0):
        self.calls, self.fail_regression = [], fail_regression

    def job_status(self, job_id):
        return {"status": "succeeded", "checkpoint": "ft:m:ckpt-step-9"}

    def evaluate(self, checkpoint, suite, split, seed):
        self.calls.append((checkpoint, split))
        if split == "regression" and checkpoint != "base" and self.fail_regression:
            self.fail_regression -= 1
            return {"error": "timeout", "stage": "inference", "transient": True}
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


def test_eval_candidate_resume_reuses_finished_split_and_retries_only_failed(tmp_path):
    mod = _script()
    s = Settings(state_dir=tmp_path, data_dir=tmp_path)
    fb = _ResumeBackend(fail_regression=1)
    failed = mod.evaluate("run1", "ftjob-1", s, fb)
    assert failed["status"] == "FAILED"
    assert "heldout" not in failed and "score" not in json.dumps(failed)
    heldout = ("ft:m:ckpt-step-9", "heldout")
    assert fb.calls.count(heldout) == 1

    done = mod.evaluate("run1", "ftjob-1", s, fb, resume=True)
    assert done["status"] == "SUCCEEDED"
    assert fb.calls.count(heldout) == 1  # the finished split came from the ledger
    assert fb.calls.count(("ft:m:ckpt-step-9", "regression")) == 2  # failed once, then retried
    assert sum(1 for c in fb.calls if c[0] == "base") == 2
    assert done["heldout"]["score"] == 0.8 and done["gate"]["decision"] == "PROMOTE"


def test_eval_candidate_resume_accepts_a_checkpoint_name_ref(tmp_path):
    mod = _script()
    s = Settings(state_dir=tmp_path, data_dir=tmp_path)
    fb = _ResumeBackend(fail_regression=1)
    ckpt = "ft:m:ckpt-step-9"
    failed = mod.evaluate("run2", ckpt, s, fb)
    assert failed["status"] == "FAILED" and fb.calls.count((ckpt, "heldout")) == 1

    done = mod.evaluate("run2", ckpt, s, fb, resume=True)
    assert done["status"] == "SUCCEEDED"
    assert fb.calls.count((ckpt, "heldout")) == 1  # reused by checkpoint name
    assert fb.calls.count((ckpt, "regression")) == 2
