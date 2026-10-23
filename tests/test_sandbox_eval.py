import json
import subprocess
from types import SimpleNamespace

import httpx
import pytest

from posttrain_agent.config import Settings
from posttrain_agent.experiments import game_env, games
from posttrain_agent.experiments.domain import promotion_gate
from posttrain_agent.experiments.nebius import NebiusBackend
from posttrain_agent.experiments.sandbox import SandboxJudge, SandboxUnavailable


def _item(task, name=None, arguments=None):
    exp = task["expected"]
    return {"id": task["id"], "state": task["state"], "name": name or exp["name"],
            "arguments": exp["arguments"] if arguments is None else arguments}


def test_judge_accepts_every_expected_answer_and_rejects_wrong_ones():
    tasks = games.make_split(5, 200)
    assert all(r["ok"] for r in game_env.judge_batch([_item(t) for t in tasks]))
    assert not any(r["ok"] for r in game_env.judge_batch(
        [_item(t, name="wrong_tool") for t in tasks]))
    assert not any(r["ok"] for r in game_env.judge_batch(
        [_item(t, arguments={}) for t in tasks]))  # missing argument
    assert not any(r["ok"] for r in game_env.judge_batch(
        [_item(t, arguments={**t["expected"]["arguments"], "x": 1}) for t in tasks]))  # extra arg


def test_judge_accepts_any_winning_cell_not_just_the_first():
    state = "ttt:XX.X....."  # X at 0,1,3: cell 2 completes the top row, cell 6 the left column
    for cell in (2, 6):
        assert game_env.judge(state, "place_mark", {"cell": cell})[0], cell
    for cell in (4, 5, 7, 8):
        assert not game_env.judge(state, "place_mark", {"cell": cell})[0], cell
    assert not game_env.judge(state, "place_mark", {"cell": 0})[0]  # occupied: illegal
    assert not game_env.judge("nim:18", "take_stones", {"count": True})[0]  # bool is not an int
    assert not game_env.judge("nim:18", "take_stones", {"count": 3})[0]  # leaves 15


class FakeSandbox:
    """Stands in for contree: runs the exact shell command locally with bash."""

    def __init__(self, fail=False):
        self.commands, self.fail = [], fail
        self.images = SimpleNamespace(use=self._use)

    async def _use(self, image, strict=True):
        self.image = image
        return SimpleNamespace(run=self._run)

    async def _run(self, shell, disposable, timeout=None):
        self.commands.append(shell)
        if self.fail:
            return SimpleNamespace(exit_code=2, stdout="", stderr="boom")
        p = subprocess.run(["bash", "-c", shell], capture_output=True, text=True, check=False)
        return SimpleNamespace(exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)


def test_sandbox_judge_runs_the_real_command_in_batches_and_never_gets_the_api_key():
    s = Settings(nebius_api_key="nb-secretsecret123", nebius_project_id="proj")
    fake = FakeSandbox()
    tasks = games.make_split(6, 95)
    out = SandboxJudge(s, lambda: fake).judge([_item(t) for t in tasks])
    assert [r["id"] for r in out] == [t["id"] for t in tasks] and all(r["ok"] for r in out)
    assert len(fake.commands) == 3  # 95 calls in batches of 40
    assert fake.image == "python:3.12-slim"
    assert all("nb-secretsecret123" not in c for c in fake.commands)


def test_sandbox_failure_raises_instead_of_inventing_verdicts():
    with pytest.raises(SandboxUnavailable, match="exit 2"):
        SandboxJudge(Settings(), lambda: FakeSandbox(fail=True)).judge(
            [_item(games.make_split(1, 4)[0])])


def _suite(tmp_path, n=8):
    suite = tmp_path / "suite"
    suite.mkdir()
    tasks = games.make_split(2, n)
    (suite / "heldout.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    return suite, tasks


def _backend(tmp_path, tasks, judge_mode, sandbox=None, right=True):
    suite, _ = tmp_path / "suite", tasks
    by_prompt = {t["messages"][-1]["content"]: t for t in tasks}

    def handler(req: httpx.Request) -> httpx.Response:
        task = by_prompt[json.loads(req.content)["messages"][-1]["content"]]
        args = task["expected"]["arguments"] if right else {}
        call = {"function": {"name": task["expected"]["name"], "arguments": json.dumps(args)}}
        return httpx.Response(200, json={"choices": [{"message": {"tool_calls": [call]}}]})

    s = Settings(nebius_api_key="k", suite_dir=suite, eval_judge=judge_mode)
    return NebiusBackend(s, lambda: httpx.Client(transport=httpx.MockTransport(handler)), sandbox)


def test_evaluation_uses_the_sandbox_when_available_and_labels_it(tmp_path):
    _, tasks = _suite(tmp_path)
    be = _backend(tmp_path, tasks, "auto", SandboxJudge(Settings(), lambda: FakeSandbox()))
    r = be.evaluate("base", "s", "heldout", 7)
    assert r["judged_by"] == "nebius-sandbox" and r["score"] == 1.0 and r["n"] == 8
    assert r["failures"] == [] and r["parse_rate"] == 1.0 and set(r["by_game"]) <= {"ttt", "nim", "guess", "rps"}


def test_judge_fails_closed_when_the_sandbox_judge_fails_in_auto_and_sandbox_modes(tmp_path):
    _, tasks = _suite(tmp_path)
    broken = SandboxJudge(Settings(), lambda: FakeSandbox(fail=True))
    auto = _backend(tmp_path, tasks, "auto", broken).evaluate("base", "s", "heldout", 7)
    assert "score" not in auto and auto["stage"] == "judge" and "judged_by" not in auto
    local = _backend(tmp_path, tasks, "local").evaluate("base", "s", "heldout", 7)  # explicit
    assert local["judged_by"] == "local-env" and local["score"] == 1.0
    strict = _backend(tmp_path, tasks, "sandbox", broken).evaluate("base", "s", "heldout", 7)
    assert "error" in strict and strict["stage"] == "judge"
    unset = _backend(tmp_path, tasks, "sandbox").evaluate("base", "s", "heldout", 7)
    assert "NEBIUS_PROJECT_ID" in unset["error"]


def test_wrong_moves_score_zero_and_benchmark_reports_the_reason(tmp_path):
    _, tasks = _suite(tmp_path)
    (tmp_path / "suite" / "train.jsonl").write_text(
        "".join(json.dumps(t) + "\n" for t in tasks))
    be = _backend(tmp_path, tasks, "local", right=False)
    assert be.evaluate("base", "s", "heldout", 7)["score"] == 0.0
    bench = be.benchmark("base", "s", 7)
    assert bench["judged_by"] == "local-env" and bench["failures"][0]["got"]["arguments"] == {}


def test_gate_never_compares_results_from_different_judges():
    def res(score, judge):
        return {"score": score, "n": 400, "evidence": "LIVE", "suite": "s", "split": "heldout",
                "seed": 7, "judged_by": judge, "parse_rate": 1.0}
    mixed = promotion_gate(res(0.4, "local-env"), res(0.9, "nebius-sandbox"))
    assert mixed["decision"] == "REJECT" and "judged_by differs" in mixed["reasons"][0]
    assert promotion_gate(res(0.4, "nebius-sandbox"), res(0.9, "nebius-sandbox"))["decision"] == "PROMOTE"
