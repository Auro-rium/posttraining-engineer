"""Four deterministic mini-games used for evaluation suites AND for SFT data.

Every task shows a game state and one tool; the correct tool call is computed from the game rule,
never invented. A task has a canonical ``state`` key ("nim:18"). Splits and SFT data are kept
disjoint by state key, so a held-out state can never be trained on, whatever the phrasing.

Evaluation tasks always use phrasing style 0. SFT rows may also use styles 1 and 2 (paraphrases of
the same state) so the model learns the rule, not one sentence.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
ORDER = ["rock", "paper", "scissors"]
BEATS = {"rock": "paper", "paper": "scissors", "scissors": "rock"}  # key loses to value
STYLES = 3
RULE_VERIFIED = "RULE_VERIFIED"  # answer recomputed from the game rule (not a measurement)
SPLIT_FILES = ("train", "heldout", "regression")


def _tool(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


def _winner_move(board: list[str], mark: str) -> int | None:
    for a, b, c in LINES:
        cells = [board[a], board[b], board[c]]
        if cells.count(mark) == 2 and cells.count(".") == 1:
            return (a, b, c)[cells.index(".")]
    return None


def _spec(game: str, key: str, render: Callable[[int], str], tool: dict, name: str,
          args: dict, done: str) -> dict:
    return {"game": game, "key": f"{game}:{key}", "render": render, "tool": tool, "name": name,
            "args": args, "done": done}


def ttt(rng: random.Random) -> dict:
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
        texts = [
            "Tic-tac-toe, you are X and it is your move. Board (. = empty):\n" + grid +
            "\nPlay the move that wins now, otherwise blocks O's win.",
            "You are X in tic-tac-toe. Cells are numbered 0-8, left to right, top to bottom.\n" +
            grid + "\nIf you can win this turn, do it; otherwise stop O from winning.",
            "Board:\n" + grid + "\nX to move. Call place_mark with the cell that wins "
            "immediately, or else blocks O."]
        tool = _tool("place_mark", "Place X on a cell, index 0-8 row by row.",
                     {"cell": {"type": "integer"}}, ["cell"])
        return _spec("ttt", "".join(board), lambda s, t=texts: t[s], tool, "place_mark",
                     {"cell": move}, f"Placed X in cell {move}.")


def nim(rng: random.Random) -> dict:
    """Take 1-3 stones; whoever takes the last stone wins. Leave a multiple of 4."""
    pile = rng.choice([p for p in range(5, 301) if p % 4])
    texts = [
        f"Nim: the pile has {pile} stones. Each turn you take 1, 2 or 3; taking the last stone "
        "wins. Make the winning move.",
        f"There are {pile} stones in a pile. Players alternate removing 1, 2 or 3 stones; whoever "
        "removes the last stone wins. It is your turn. Remove the number that forces a win.",
        f"Nim game, {pile} stones left, take 1-3 per turn, last stone wins. Your move."]
    tool = _tool("take_stones", "Remove 1 to 3 stones from the pile.",
                 {"count": {"type": "integer"}}, ["count"])
    return _spec("nim", str(pile), lambda s, t=texts: t[s], tool, "take_stones",
                 {"count": pile % 4}, f"Took {pile % 4} stone{'s' if pile % 4 > 1 else ''}.")


def guess(rng: random.Random) -> dict:
    """Binary search: guess the midpoint of the range still possible, rounded down."""
    lo = rng.randint(1, 50)
    hi = lo + rng.randint(4, 120)
    texts = [
        f"Number guessing. The secret is an integer between {lo} and {hi} inclusive. Make the "
        "guess that halves the range (round down).",
        f"I am thinking of a whole number from {lo} to {hi}, both included. Guess the middle "
        "value, rounding down, to cut the possibilities in half.",
        f"Secret number in [{lo}, {hi}]. Best first guess for binary search (floor of the middle)?"]
    tool = _tool("guess_number", "Guess the secret number.", {"value": {"type": "integer"}},
                 ["value"])
    mid = (lo + hi) // 2
    return _spec("guess", f"{lo}-{hi}", lambda s, t=texts: t[s], tool, "guess_number",
                 {"value": mid}, f"Guessed {mid}.")


def rps(rng: random.Random) -> dict:
    """The opponent cycles rock->paper->scissors (or the reverse); beat its next move."""
    step, start = rng.choice([1, -1]), rng.randrange(3)
    history = [ORDER[(start + step * k) % 3] for k in range(rng.randint(3, 20))]
    nxt = ORDER[(start + step * len(history)) % 3]
    listing = ", ".join(history)
    texts = [
        "Rock-paper-scissors. Your opponent follows a fixed cycle. Their moves so far: "
        f"{listing}. Predict their next move and play the move that beats it.",
        f"Your opponent plays rock-paper-scissors in a repeating cycle. History: {listing}. "
        "Work out what they will play next and counter it.",
        f"Opponent moves: {listing} (a fixed cycle). Play the move that beats their next one."]
    tool = _tool("play_move", "Play rock, paper or scissors.",
                 {"move": {"type": "string", "enum": ORDER}}, ["move"])
    return _spec("rps", f"{step}:{start}:{len(history)}", lambda s, t=texts: t[s], tool,
                 "play_move", {"move": BEATS[nxt]}, f"Played {BEATS[nxt]}.")


GAMES: dict[str, Callable[[random.Random], dict]] = {"ttt": ttt, "nim": nim, "guess": guess,
                                                      "rps": rps}


def as_task(spec: dict, i: int, style: int = 0) -> dict:
    return {"id": f"{spec['game']}-{i}", "state": spec["key"],
            "messages": [{"role": "user", "content": spec["render"](style)}],
            "tools": [spec["tool"]],
            "expected": {"name": spec["name"], "arguments": spec["args"]}}


def as_sft_row(spec: dict, style: int) -> dict:
    """Nebius conversational row: tools, user, assistant tool call, tool result, closing reply."""
    call = {"id": "call_1", "type": "function",
            "function": {"name": spec["name"], "arguments": json.dumps(spec["args"])}}
    return {
        "tools": [spec["tool"]],
        "messages": [
            {"role": "user", "content": spec["render"](style)},
            {"role": "assistant", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call_1", "name": spec["name"],
             "content": json.dumps({"ok": True, "applied": spec["args"]})},
            {"role": "assistant", "content": spec["done"]},
        ],
        "metadata": {"evidence": RULE_VERIFIED, "source": "rule_generated",
                     "game": spec["game"], "state": spec["key"], "style": style},
    }


def make_split(seed: int, n: int, exclude: set[str] | frozenset[str] = frozenset()) -> list[dict]:
    """``n`` tasks (style 0) over all games, balanced, with distinct states not in ``exclude``."""
    rng, out, seen = random.Random(seed), [], set(exclude)
    names, misses = list(GAMES), dict.fromkeys(GAMES, 0)
    i = 0
    while len(out) < n and any(m < 300 for m in misses.values()):
        game = names[i % len(names)]
        i += 1
        if misses[game] >= 300:
            continue
        spec = GAMES[game](rng)
        if spec["key"] in seen:
            misses[game] += 1
            continue
        misses[game] = 0
        seen.add(spec["key"])
        out.append(as_task(spec, len(out)))
    return out


def suite_states(suite_dir: Path | str) -> set[str]:
    """Canonical states of every eval split on disk (empty if the files carry no ``state``)."""
    states: set[str] = set()
    for split in SPLIT_FILES:
        path = Path(suite_dir) / f"{split}.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    state = json.loads(line).get("state")
                    if isinstance(state, str):
                        states.add(state)
    return states


def generate_sft(n: int, seed: int, exclude: set[str] | frozenset[str],
                 cap: int = 1200) -> tuple[list[dict], dict[str, int]]:
    """Up to ``min(n, cap)`` rule-correct SFT rows, balanced over games, never on ``exclude``.

    Fewer rows come back when a game runs out of distinct (state, phrasing) pairs; the per-game
    counts say which. Rows are deterministic for a given seed.
    """
    n = max(0, min(n, cap))
    rng = random.Random(seed)
    names, misses = list(GAMES), dict.fromkeys(GAMES, 0)
    seen: set[tuple[str, int]] = set()
    rows: list[dict] = []
    counts = dict.fromkeys(GAMES, 0)
    i = 0
    while len(rows) < n and any(m < 300 for m in misses.values()):
        game = names[i % len(names)]
        i += 1
        if misses[game] >= 300:
            continue
        spec = GAMES[game](rng)
        style = rng.randrange(STYLES)
        key = (spec["key"], style)
        if spec["key"] in exclude or key in seen:
            misses[game] += 1
            continue
        misses[game] = 0
        seen.add(key)
        rows.append(as_sft_row(spec, style))
        counts[game] += 1
    rng.shuffle(rows)
    return rows, counts
