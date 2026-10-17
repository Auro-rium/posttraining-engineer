"""Live, cost-bounded smoke test of sandbox CPU serving (not a pytest test).

Serves the BASE model on a few held-out tasks through ``NebiusBackend.evaluate`` (so the /models
probe, the sandbox student and the judge all run for real), prints raw outputs, parsed calls,
verdicts and timings. With ``--adapter-job`` it also proves the adapter path (checkpoint lookup,
download, strict LoRA load) on a couple of tasks; that checkpoint's quality is irrelevant.

Needs NEBIUS_API_KEY and NEBIUS_PROJECT_ID in ./.env or the environment. Prints no secrets.

    POSTTRAIN_BASE_MODEL=Qwen/Qwen3-0.6B .venv/bin/python scripts/smoke_sandbox_serving.py \
        --tasks 4 --adapter-job ftjob-... --adapter-step 6 --adapter-tasks 2
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from posttrain_agent.config import Settings, load_dotenv
from posttrain_agent.experiments.nebius import NebiusBackend
from posttrain_agent.experiments.sandbox_student import SandboxStudent, clean_output, first_tool_call


def _suite(tasks: list[dict]) -> Path:
    d = Path(tempfile.mkdtemp(prefix="smoke-suite-"))
    (d / "heldout.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    return d


def _run(be: NebiusBackend, student: SandboxStudent, checkpoint: str, tasks: list[dict]) -> None:
    captured: list[dict] = []
    orig = student.generate

    def spy(items, adapter=None):
        outs = orig(items, adapter)
        captured.extend(outs)
        return outs

    student.generate = spy  # type: ignore[method-assign]
    t = time.monotonic()
    res = be.evaluate(checkpoint, "smoke", "heldout", be.s.seed)
    wall = time.monotonic() - t
    student.generate = orig  # type: ignore[method-assign]
    print(f"\n== {checkpoint[:90]} on {len(tasks)} tasks: {wall:.1f}s wall")
    print(json.dumps(res, indent=1))
    for task, out in zip(tasks, captured):
        print(f"- {task['id']}: expected {json.dumps(task['expected'])}")
        print(f"  raw: {clean_output(out.get('text') or '')[:300]!r}")
        print(f"  parsed: {json.dumps(first_tool_call(out.get('text') or ''))}  "
              f"new_tokens={out.get('new_tokens')} seconds={out.get('seconds')}")
    print("timings:", json.dumps(student.timings[-3:]))
    student.timings.clear()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=4)
    ap.add_argument("--adapter-job")
    ap.add_argument("--adapter-step", type=int)
    ap.add_argument("--adapter-tasks", type=int, default=2)
    a = ap.parse_args()
    load_dotenv()
    s = Settings.from_env()
    if not (s.nebius_api_key and s.nebius_project_id):
        print("NEBIUS_API_KEY and NEBIUS_PROJECT_ID are required", file=sys.stderr)
        return 2
    heldout = [json.loads(x) for x in (s.suite_dir / "heldout.jsonl").read_text().splitlines()
               if x.strip()]
    print(f"base model {s.base_model}; serve={s.serve}; judge={s.eval_judge}; "
          f"image {s.sandbox_image}")
    student = SandboxStudent(s)
    tasks = heldout[:a.tasks]
    be = NebiusBackend(replace(s, suite_dir=_suite(tasks)), student=student)
    print("serving decision:", be.serving())
    _run(be, student, "base", tasks)
    if a.adapter_job:
        tasks2 = heldout[:a.adapter_tasks]
        be2 = NebiusBackend(replace(s, suite_dir=_suite(tasks2), serve="sandbox"), student=student)
        items = be2._index_checkpoints(a.adapter_job)  # noqa: SLF001 - smoke script
        names = [c.get("fine_tuned_model_checkpoint") for c in items]
        pick = [n for n in names if n and (a.adapter_step is None
                                           or n.endswith(f"ckpt-step-{a.adapter_step}"))]
        print(f"\ncheckpoints of {a.adapter_job}: {len(names)}; using {pick[-1:]}")
        if not pick:
            return 1
        _run(be2, student, pick[-1], tasks2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
