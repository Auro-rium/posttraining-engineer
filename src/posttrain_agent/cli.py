"""``posttrain`` command line."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from .agent import build_agent
from .config import Settings
from .persistence import open_runtime
from .replay import RunReport, decide, fork, history, ledger, pending_actions, run_with_recovery
from .telemetry import TelemetryHandler, summarize


def _show(report: RunReport) -> int:
    final = None
    if report.result and report.status == "COMPLETED":
        final = report.result["messages"][-1].content
    print(json.dumps({
        "thread": report.thread_id, "status": report.status, "attempts": report.attempts,
        "errors": report.errors, "final": final,
        "awaiting_approval": [{"action": a.get("name"), "description": a.get("description")}
                              for a in report.pending],
    }, indent=2))
    if report.status == "AWAITING_APPROVAL":
        print(f"\nReview, then: posttrain approve --thread {report.thread_id}"
              f"   or   posttrain reject --thread {report.thread_id} -m 'why'", file=sys.stderr)
        return 3
    return 0 if report.status == "COMPLETED" else 1


async def _main(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if args.cmd == "telemetry":  # needs no model or network
        print(json.dumps(summarize(settings.state_dir / "telemetry.jsonl", args.last,
                                   args.errors), indent=2))
        return 0
    if not settings.backboard_api_key and settings.cassette_mode != "replay":
        print("BACKBOARD_API_KEY is not set", file=sys.stderr)
        return 2
    cbs: list[Any] = [TelemetryHandler(settings.state_dir / "telemetry.jsonl")]
    opts = {"max_attempts": settings.max_recovery_attempts, "callbacks": cbs}
    async with open_runtime(settings.state_dir) as rt:
        graph = await build_agent(settings, rt)
        if args.cmd == "run":
            return _show(await run_with_recovery(graph, args.goal, args.thread, store=rt.store, **opts))
        if args.cmd == "resume":
            return _show(await run_with_recovery(graph, "", args.thread, store=rt.store, **opts))
        if args.cmd == "pending":
            print(json.dumps(await pending_actions(graph, args.thread), indent=2, default=str))
        elif args.cmd in {"approve", "reject"}:
            return _show(await decide(graph, args.thread, args.cmd == "approve",
                                      message=getattr(args, "message", None),
                                      store=rt.store, **opts))
        elif args.cmd == "history":
            print(json.dumps(await history(graph, args.thread), indent=2))
        elif args.cmd == "fork":
            out = await fork(graph, args.thread, args.checkpoint, cbs)
            print(out["messages"][-1].content)
        elif args.cmd == "runs":
            print(json.dumps(await ledger(rt.store), indent=2))
    return 0


def main() -> None:
    p = argparse.ArgumentParser(prog="posttrain")
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="start (or continue) a run on a thread")
    run.add_argument("goal")
    run.add_argument("--thread", default="default")
    for name in ("resume", "pending", "approve", "history"):
        sub.add_parser(name).add_argument("--thread", default="default")
    rej = sub.add_parser("reject", help="reject the pending training launch")
    rej.add_argument("--thread", default="default")
    rej.add_argument("-m", "--message", help="reason shown to the agent")
    f = sub.add_parser("fork", help="replay from a past checkpoint")
    f.add_argument("--thread", default="default")
    f.add_argument("checkpoint")
    sub.add_parser("runs", help="show the run ledger")
    t = sub.add_parser("telemetry", help="summarise model/tool telemetry")
    t.add_argument("--last", type=int, default=500)
    t.add_argument("--errors", action="store_true", help="only errors")
    raise SystemExit(asyncio.run(_main(p.parse_args())))
