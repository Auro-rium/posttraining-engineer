"""End-to-end exit invariant: a recorded candidate cannot end as COMPLETED on a final message.

The candidate is already evaluated on both splits here, which is the dangerous case: the model may
try to write the final report before champion_manager runs the gate. The live ExitGate must
re-prompt and replay must report INCOMPLETE; a recorded gate verdict (even REJECT) releases the
run."""

from fakes import FakeBackboard, text
from test_approval import settings as approval_settings
from test_phase_b import R, put, seed_candidate

from posttrain_agent.agent import build_agent
from posttrain_agent.experiments import runstate as rs
from posttrain_agent.runtime.exit_gate import exit_problem
from posttrain_agent.runtime.persistence import open_runtime
from posttrain_agent.runtime.replay import run_with_recovery

RUN = "e2e"


def _evaluated_candidate(s):
    seed_candidate(s, "succeeded", run=RUN)
    for split in ("heldout", "regression"):
        put(s, "ckpt-1", split, R(0.5, split))


async def _run(s, log):
    async with open_runtime(s.state_dir) as rt:
        graph = await build_agent(
            s, rt, client_factory=lambda: FakeBackboard(lambda kw: text("final report"), log))
        return await run_with_recovery(
            graph, "go", RUN, max_attempts=1,
            exit_check=lambda thread: exit_problem(s.state_dir, thread))


async def test_final_message_cannot_complete_without_a_gate_verdict(tmp_path):
    s = approval_settings(tmp_path)
    _evaluated_candidate(s)
    assert "no gate verdict" in exit_problem(s.state_dir, RUN)

    log: list = []
    rep = await _run(s, log)
    assert rep.status == "INCOMPLETE"
    assert "ended without a gate verdict" in rep.errors[-1]
    assert len(log) == 4  # the final message plus 3 exit-gate re-prompts
    prompts = [m for m in rep.result["messages"] if "[exit gate" in str(m.content)]
    assert len(prompts) == 3 and "ftjob-1" in prompts[0].content


async def test_recorded_reject_verdict_lets_the_run_complete(tmp_path):
    s = approval_settings(tmp_path)
    _evaluated_candidate(s)
    rs.record_gate(s.state_dir, run=RUN, champion="base", candidate="ckpt-1",
                   candidate_job="ftjob-1", decision="REJECT")
    assert exit_problem(s.state_dir, RUN) is None

    log: list = []
    rep = await _run(s, log)
    assert rep.status == "COMPLETED" and len(log) == 1
    assert rep.result["messages"][-1].content == "final report"
