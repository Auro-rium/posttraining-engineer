"""Generate web/public/sample-run.jsonl: a SIMULATED run for the dashboard's sample mode.

Runs the stub-backed fake agent (scripted model, local stub training backend, approval
included) and writes its event journal, one JSON envelope per line. Every score and job
event carries evidence EXPLANATION: nothing here is live. The run id starts with "sample-".

    python scripts/make_sample_journal.py --out web/public/sample-run.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "tests"):  # the fake model client lives with the tests
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fakes import FakeBackboard, call, text  # noqa: E402
from test_domain import GOOD, ROW  # noqa: E402

from posttrain_agent.agent import build_agent  # noqa: E402
from posttrain_agent.config import Settings  # noqa: E402
from posttrain_agent.dataset import save_rows  # noqa: E402
from posttrain_agent.domain import LocalStubBackend, make_tools  # noqa: E402
from posttrain_agent.events import CURRENT_RUN, Journal, emit  # noqa: E402
from posttrain_agent.journal_handler import JournalHandler  # noqa: E402
from posttrain_agent.persistence import open_runtime  # noqa: E402
from posttrain_agent.replay import decide, run_with_recovery  # noqa: E402

RUN = "sample-simulated-run"
LAUNCH = [
    call("task", {"description": "launch the job", "subagent_type": "training_executor"}),
    call("launch_training", {"hyperparameters": GOOD, "dataset_ref": "d.jsonl"}, id="call_2"),
]


def _factory(seq: list):
    return lambda: FakeBackboard(lambda kw: seq.pop(0), [])


async def _run(tmp: Path, journal: Journal) -> None:
    s = Settings(state_dir=tmp / "state", data_dir=tmp / "state" / "data",
                 model_retries=0, price_per_mtok=1.0)
    save_rows(s.data_dir, "d.jsonl", [ROW])
    cbs = [JournalHandler(journal, primary="sample:simulated-model")]
    async with open_runtime(s.state_dir) as rt:
        graph = await build_agent(s, rt, client_factory=_factory(list(LAUNCH)), journal=journal)
        await run_with_recovery(graph, "sample run", RUN, store=rt.store, max_attempts=1,
                                journal=journal, backend="stub", experiment_budget=3, callbacks=cbs)
    done = [text('{"status": "SUCCEEDED"}'), text("all done")]
    async with open_runtime(s.state_dir) as rt:
        graph = await build_agent(s, rt, client_factory=_factory(done), journal=journal)
        await decide(graph, RUN, True, store=rt.store, max_attempts=1, journal=journal,
                     callbacks=cbs)

    # Evaluation, job polling and the promotion gate, on the same stub backend and run id.
    token = CURRENT_RUN.set(RUN)
    try:
        tools = make_tools(LocalStubBackend(), s, journal=journal)
        tools["run_benchmark"]("base")
        tools["evaluate_checkpoint"]("stub-ckpt-1", "heldout")
        out = tools["launch_training"](GOOD, "d.jsonl")
        tools["get_training_status"](out["provider_job_id"])
        tools["read_training_logs"](out["provider_job_id"])
        tools["decide_promotion"]({"score": 0.3}, {"score": 0.5})
        emit(journal, "telemetry.snapshot", {"p50_ms": 120.0, "p95_ms": 480.0, "error_rate": 0.0,
                                             "input_tokens": 12, "output_tokens": 8})
    finally:
        CURRENT_RUN.reset(token)


def make_sample(out: Path) -> None:
    """Run the simulated agent and write its journal to ``out`` (one envelope per line)."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        journal = Journal(tmp / "events.sqlite")
        asyncio.run(_run(tmp, journal))
        events = journal.read(RUN, limit=10_000)
    lines = [json.dumps({"seq": e.seq, "ts": e.ts, "run": e.run, "type": e.type, "data": e.data},
                        separators=(",", ":")) for e in events]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "web" / "public" / "sample-run.jsonl")
    make_sample(ap.parse_args().out)


if __name__ == "__main__":
    main()
