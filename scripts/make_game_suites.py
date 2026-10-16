"""Generate tool-calling task suites from four small deterministic games.

Each task shows a game state and one tool; the model must call it with the provably best move.
The expected answer is computed, never invented. Splits use disjoint seeds, so held-out and
regression states are not in train.

    python scripts/make_game_suites.py state/suite --train 60 --heldout 20 --regression 20
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
BEATS = {"rock": "paper", "paper": "scissors", "scissors": "rock"}  # key loses to value


def _tool(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


def _task(tid: str, prompt: str, tool: dict, name: str, args: dict) -> dict:
    return {"id": tid, "messages": [{"role": "user", "content": prompt}], "tools": [tool],
            "expected": {"name": name, "arguments": args}}


def _winner_move(board: list[str], mark: str) -> int | None:
    for a, b, c in LINES:
        cells = [board[a], board[b], board[c]]
        if cells.count(mark) == 2 and cells.count(".") == 1:
            return (a, b, c)[cells.index(".")]
    return None


def tictactoe(rng: random.Random, i: int) -> dict:
    """X to move with a forced answer: win now if possible, else block O."""
    while True:
        board = ["."] * 9
        for cell in rng.sample(range(9), rng.randint(3, 6)):
            board[cell] = rng.choice("XO")
        if abs(board.count("X") - board.count("O")) > 1 or any(
                len({board[a], board[b], board[c]}) == 1 and board[a] != "." for a, b, c in LINES):
            continue
        win, block = _winner_move(board, "X"), _winner_move(board, "O")
        if win is None and block is None:
            continue
        move = win if win is not None else block
        grid = "\n".join(" ".join(board[r * 3:r * 3 + 3]) for r in range(3))
        tool = _tool("place_mark", "Place X on a cell, index 0-8 row by row.",
                     {"cell": {"type": "integer"}}, ["cell"])
        return _task(f"ttt-{i}", f"Tic-tac-toe, you are X and it is your move. Board (. = empty):"
                     f"\n{grid}\nPlay the move that wins now, otherwise blocks O's win.", tool,
                     "place_mark", {"cell": move})


def nim(rng: random.Random, i: int) -> dict:
    """Take 1-3 stones; whoever takes the last stone wins. Leave a multiple of 4."""
    while True:
        pile = rng.randint(5, 40)
        if pile % 4:
            break
    tool = _tool("take_stones", "Remove 1 to 3 stones from the pile.",
                 {"count": {"type": "integer"}}, ["count"])
    return _task(f"nim-{i}", f"Nim: the pile has {pile} stones. Each turn you take 1, 2 or 3; "
                 "taking the last stone wins. Make the winning move.", tool,
                 "take_stones", {"count": pile % 4})


def guess(rng: random.Random, i: int) -> dict:
    """Binary search: guess the midpoint of the range still possible."""
    lo = rng.randint(1, 50)
    hi = lo + rng.randint(4, 60)
    tool = _tool("guess_number", "Guess the secret number.", {"value": {"type": "integer"}},
                 ["value"])
    return _task(f"guess-{i}", f"Number guessing. The secret is an integer between {lo} and "
                 f"{hi} inclusive. Make the guess that halves the range (round down).", tool,
                 "guess_number", {"value": (lo + hi) // 2})


def rps(rng: random.Random, i: int) -> dict:
    """The opponent cycles rock->paper->scissors (or the reverse); play what beats its next move."""
    order = ["rock", "paper", "scissors"]
    step = rng.choice([1, -1])
    start = rng.randrange(3)
    history = [order[(start + step * k) % 3] for k in range(rng.randint(3, 5))]
    nxt = order[(start + step * len(history)) % 3]
    tool = _tool("play_move", "Play rock, paper or scissors.",
                 {"move": {"type": "string", "enum": order}}, ["move"])
    return _task(f"rps-{i}", "Rock-paper-scissors. Your opponent follows a fixed cycle. Their "
                 f"moves so far: {', '.join(history)}. Predict their next move and play the move "
                 "that beats it.", tool, "play_move", {"move": BEATS[nxt]})


GAMES = (tictactoe, nim, guess, rps)


def make_split(seed: int, n: int, exclude: set[str] = frozenset()) -> list[dict]:
    rng, out, seen = random.Random(seed), [], set(exclude)
    for i in itertools.count():
        if len(out) >= n or i > n * 50:
            break
        t = GAMES[i % len(GAMES)](rng, i)  # rotate by attempt so a saturated game can't stall
        key = t["messages"][0]["content"]
        if key not in seen:  # no duplicate states within a split
            seen.add(key)
            out.append(t)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--train", type=int, default=60)
    p.add_argument("--heldout", type=int, default=20)
    p.add_argument("--regression", type=int, default=20)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    splits = {"train": (1, args.train), "heldout": (2, args.heldout),
              "regression": (3, args.regression)}
    used: set[str] = set()  # states already in an earlier split: sealed splits never reuse them
    for name, (seed, n) in splits.items():
        tasks = make_split(seed, n, used)
        used |= {t["messages"][0]["content"] for t in tasks}
        (args.out / f"{name}.jsonl").write_text(
            "".join(json.dumps(t, separators=(",", ":")) + "\n" for t in tasks))
        kinds = sorted({t["id"].split("-")[0] for t in tasks})
        print(f"{name}: {len(tasks)} tasks ({', '.join(kinds)}) -> {args.out}")


if __name__ == "__main__":
    main()
