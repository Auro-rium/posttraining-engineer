"""Generate the evaluation suites (train / heldout / regression) from the four mini-games.

Each task shows a game state and one tool; the model must call it with the provably best move.
Splits are disjoint by game state. SFT data is generated separately by the agent's
`generate_game_sft` tool and excludes every state found here.

    python scripts/make_game_suites.py state/suite --train 200 --heldout 100 --regression 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posttrain_agent.experiments.games import make_split


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--train", type=int, default=200)
    p.add_argument("--heldout", type=int, default=100)  # larger n makes a real gain detectable
    p.add_argument("--regression", type=int, default=60)
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
        kinds = sorted({t["id"].split("-")[0] for t in tasks})
        print(f"{name}: {len(tasks)} tasks ({', '.join(kinds)}) -> {args.out}")


if __name__ == "__main__":
    main()
