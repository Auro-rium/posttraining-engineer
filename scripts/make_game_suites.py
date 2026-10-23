"""Generate the evaluation suites (train / heldout / regression) from the four mini-games.

Each task shows a game state and one tool; the model must call it with the provably best move.
Splits are disjoint by game state. SFT data is generated separately by the agent's
`generate_game_sft` tool and excludes every state found here.

    python scripts/make_game_suites.py state/suite --train 200 --heldout 100 --regression 60

For an honest gate use ``--heldout 300 --regression 150`` (the defaults below; nebius.MAX_TASKS
allows up to 1000). WARNING: this overwrites the sealed split files, which invalidates every
stored base result (the ledger's ``tasks_sha256`` stops matching); write to a NEW directory and
point POSTTRAIN_SUITE_DIR at it. rps has only 108 distinct states and Nim 222, so large splits
come out unbalanced; the script prints per-game counts and warns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posttrain_agent.experiments.games import GAMES, make_split


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--train", type=int, default=200)
    p.add_argument("--heldout", type=int, default=300)  # n=100 has ~27% power for a 6-point gain
    p.add_argument("--regression", type=int, default=150)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    splits = {"train": (1, args.train), "heldout": (2, args.heldout),
              "regression": (3, args.regression)}
    used: set[str] = set()  # states already in an earlier split: sealed splits never reuse them
    for name, (seed, n) in splits.items():
        tasks = make_split(seed, n, used)
        used |= {t["state"] for t in tasks}
        (args.out / f"{name}.jsonl").write_text(
            "".join(json.dumps(t, separators=(",", ":")) + "\n" for t in tasks))
        counts = {g: sum(t["id"].startswith(g + "-") for t in tasks) for g in GAMES}
        print(f"{name}: {len(tasks)} tasks {counts} -> {args.out}")
        if len(tasks) < n or (tasks and min(counts.values()) < 0.6 * len(tasks) / len(GAMES)):
            print(f"WARNING: {name} is short or unbalanced (distinct states ran out)")


if __name__ == "__main__":
    main()
