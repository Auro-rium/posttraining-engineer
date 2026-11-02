"""Operator-run candidate evaluation + gate, for when the agent loop cannot reach the checkpoint.

    python scripts/eval_candidate.py RUN ftjob-... [--resume] [--base-checkpoint base]

Uses the same ``evaluate_checkpoint`` tool and sealed splits the evaluator agent uses (a job id
resolves to its final checkpoint). Every attempt is appended to the eval ledger
(``state/eval_ledger.jsonl``, see ``experiments/ledger.py``); this script reads results back from
that ledger, never from agent chat text. Base results are taken from the ledger when one exists
for the same tasks / judge / serving path, else evaluated now. Raw generations are kept in
``state/operator_evals/<run>-<split>-gen.jsonl``.

If either split fails, BOTH are discarded: ``state/operator_evals/<run>-candidate.json`` then holds
``status: FAILED`` and the structured errors, no scores, and the exit code is 1. ``--resume``
reuses this run's finished splits from the ledger and retries only the failed ones.
Success writes both splits, the base results and the gate verdict (regression split enforced).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from posttrain_agent.config import Settings, load_dotenv
from posttrain_agent.experiments import ledger
from posttrain_agent.experiments.domain import make_tools, promotion_gate
from posttrain_agent.experiments.nebius import NebiusBackend
from posttrain_agent.observability.events import CURRENT_RUN, Journal

SPLITS = ("heldout", "regression")
SUITE = "AgentGym/AgentEval"  # make_tools' default, stored in the ledger for resume matching


def _candidate_split(tools: dict, s: Settings, run: str, job: str, split: str,
                     resume: bool) -> dict[str, Any]:
    if resume:
        # Reuse only this run/job's split on the same suite and seed; a changed seed or suite
        # means the old score is not comparable and must be recomputed.
        rec = ledger.latest_ok(s.state_dir, run=run, job=job, split=split, suite=SUITE,
                               seed=s.seed)
        if rec is None and not job.startswith("ftjob-"):
            # a checkpoint-name ref is recorded with job=None, its name in "checkpoint"
            rec = ledger.latest_ok(s.state_dir, run=run, checkpoint=job, split=split,
                                   suite=SUITE, seed=s.seed)
        if rec:
            return rec["result"]
    return tools["evaluate_checkpoint"](job, split)


def _base_split(tools: dict, s: Settings, base: str, split: str, like: dict) -> dict[str, Any]:
    rec = ledger.latest_matching(s.state_dir, checkpoint=base, split=split, like=like)
    return rec["result"] if rec else tools["evaluate_checkpoint"](base, split)


def evaluate(run: str, job: str, settings: Settings, backend: Any, *, resume: bool = False,
             base: str = "base") -> dict[str, Any]:
    journal = Journal(settings.state_dir / "events.sqlite")
    tools = make_tools(backend, settings, suite=SUITE, journal=journal)
    token = CURRENT_RUN.set(run)
    try:
        cand = {sp: _candidate_split(tools, settings, run, job, sp, resume) for sp in SPLITS}
        bad = {sp: r for sp, r in cand.items() if "score" not in r or "error" in r}
        if bad:  # one failed split invalidates the artifact: never keep a half-valid pair
            return {"status": "FAILED", "role": "candidate", "operator_run": True, "run": run,
                    "job": job, "errors": {sp: {k: r.get(k) for k in
                                                ("error", "stage", "transient", "gen_file",
                                                 "n_generated", "n_tasks", "parse_rate")
                                                if r.get(k) is not None} for sp, r in bad.items()},
                    "hint": "rerun with --resume; finished splits and generations are reused"}
        champ = {sp: _base_split(tools, settings, base, sp, cand[sp]) for sp in SPLITS}
        bad_base = {sp: r for sp, r in champ.items() if "score" not in r or "error" in r}
        if bad_base:
            return {"status": "FAILED", "role": "candidate", "operator_run": True, "run": run,
                    "job": job, "errors": {f"base-{sp}": {k: r.get(k) for k in
                                                         ("error", "stage", "transient")
                                                         if r.get(k) is not None}
                                           for sp, r in bad_base.items()},
                    "hint": "base evaluation failed; rerun with --resume"}
    finally:
        CURRENT_RUN.reset(token)
    gate = promotion_gate(champ["heldout"], cand["heldout"],
                          champion_regression=champ["regression"],
                          candidate_regression=cand["regression"], require_regression=True)
    return {"status": "SUCCEEDED", "role": "candidate", "operator_run": True, "run": run,
            "job": job, "checkpoint": cand["heldout"].get("checkpoint"), **cand,
            "base": champ, "gate": gate,
            "regression_delta": round(cand["regression"]["score"] - champ["regression"]["score"], 4)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("job", help="ftjob-... id or a checkpoint name")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--base-checkpoint", default="base")
    args = ap.parse_args(argv)
    load_dotenv()
    s = Settings.from_env()
    out = evaluate(args.run, args.job, s, NebiusBackend(s), resume=args.resume,
                   base=args.base_checkpoint)
    d = Path(s.state_dir) / "operator_evals"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{args.run}-candidate.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k not in (*SPLITS, "base")}, indent=2))
    for sp in SPLITS:
        v = out.get(sp)
        if v:
            print(sp, v["score"], "n=", v["n"], "parse_rate=", v.get("parse_rate"),
                  v.get("by_game"))
    return 0 if out["status"] == "SUCCEEDED" else 1


if __name__ == "__main__":
    sys.exit(main())
