"""``posttrain`` command line."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .agent import build_agent
from .config import Settings
from .persistence import open_runtime
from .replay import fork, history, ledger, run_with_recovery


async def _main(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if not settings.backboard_api_key and not settings.cassette_mode == "replay":
        print("BACKBOARD_API_KEY is not set", file=sys.stderr)
        return 2
    async with open_runtime(settings.state_dir) as rt:
        graph = await build_agent(settings, rt)
        if args.cmd == "run":
            report = await run_with_recovery(
                graph, args.goal, args.thread, store=rt.store,
                max_attempts=settings.max_recovery_attempts,
            )
            final = report.result["messages"][-1].content if report.result else None
            print(json.dumps({"thread": report.thread_id, "status": report.status,
                              "attempts": report.attempts, "errors": report.errors,
                              "final": final}, indent=2))
            return 0 if report.status == "COMPLETED" else 1
        if args.cmd == "resume":
            report = await run_with_recovery(graph, "", args.thread, store=rt.store,
                                             max_attempts=settings.max_recovery_attempts)
            print(report.status)
            return 0 if report.status == "COMPLETED" else 1
        if args.cmd == "history":
            print(json.dumps(await history(graph, args.thread), indent=2))
        elif args.cmd == "fork":
            out = await fork(graph, args.thread, args.checkpoint)
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
    for name in ("resume", "history"):
        sub.add_parser(name).add_argument("--thread", default="default")
    f = sub.add_parser("fork", help="replay from a past checkpoint")
    f.add_argument("--thread", default="default")
    f.add_argument("checkpoint")
    sub.add_parser("runs", help="show the run ledger")
    raise SystemExit(asyncio.run(_main(p.parse_args())))
