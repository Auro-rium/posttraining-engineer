"""Generate the evaluation suites (train / heldout / regression) from the four mini-games.

Each task shows a game state and one tool; the model must call it with the provably best move.
Splits are disjoint by game state. SFT data is generated separately by the agent's
`generate_game_sft` tool and excludes every state found here.

    python scripts/make_game_suites.py state/suite_v2 --train 200 --heldout 300 --regression 150

For an honest gate use ``--heldout 300 --regression 150`` (the defaults below; nebius.MAX_TASKS
allows up to 1000). WARNING: this overwrites the sealed split files, which invalidates every
stored base result (the ledger's ``tasks_sha256`` stops matching); write to a NEW directory and
point POSTTRAIN_SUITE_DIR at it (suites are not versioned in config: state/suite_v2 is the
new one). heldout_phrasing uses wording styles 3-4 that SFT never sees; it is reported separately
and is not part of the gate. State spaces are large enough (rps 348, nim 746) for a balanced 300/150. The script prints per-game counts and warns if a split is short or unbalanced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from posttrain_agent.experiments.games import GAMES, HELDOUT_STYLES, make_split


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--train", type=int, default=200)
    p.add_argument("--heldout", type=int, default=300)  # n=100 has ~27% power for a 6-point gain
    p.add_argument("--regression", type=int, default=150)
    p.add_argument("--phrasing", type=int, default=120,
                   help="heldout_phrasing tasks (unseen wording; NOT part of the gate's main n)")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    splits = {"train": (1, args.train), "heldout": (2, args.heldout),
              "regression": (3, args.regression),
              "heldout_phrasing": (4, args.phrasing)}
    used: set[str] = set()  # states already in an earlier split: sealed splits never reuse them
    for name, (seed, n) in splits.items():
        tasks = make_split(seed, n, used,
                           HELDOUT_STYLES if name == "heldout_phrasing" else (0,))
        used |= {t["state"] for t in tasks}
        (args.out / f"{name}.jsonl").write_text(
            "".join(json.dumps(t, separators=(",", ":")) + "\n" for t in tasks))
        counts = {g: sum(t["id"].startswith(g + "-") for t in tasks) for g in GAMES}
        sha = hashlib.sha256((args.out / f"{name}.jsonl").read_bytes()).hexdigest()[:12]
        print(f"{name}: {len(tasks)} tasks sha {sha} {counts} -> {args.out}")
        if len(tasks) < n or (tasks and min(counts.values()) < 0.6 * len(tasks) / len(GAMES)):
            print(f"WARNING: {name} is short or unbalanced (distinct states ran out)")


if __name__ == "__main__":
    main()
