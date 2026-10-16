"""Game environment judge: executes a model's tool call against the game and says if it is right.

Standard library only and self-contained, so the same file runs on the host or inside a Nebius
sandbox (``python3 game_env.py < batch.json``). Judging by outcome rather than string equality
means every optimal move counts (tic-tac-toe can have several winning cells) and malformed calls
(wrong tool, extra or mistyped arguments, illegal move) fail with a reason.

Input (stdin JSON):  {"items": [{"id": ..., "state": "nim:18", "name": "take_stones",
                                 "arguments": {"count": 2}}, ...]}
Output (stdout JSON): {"results": [{"id": ..., "ok": true, "reason": ""}, ...]}
"""

from __future__ import annotations

import json
import sys

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
ORDER = ["rock", "paper", "scissors"]
TOOLS = {"ttt": ("place_mark", "cell"), "nim": ("take_stones", "count"),
         "guess": ("guess_number", "value"), "rps": ("play_move", "move")}


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _forced_cells(board: list[str]) -> set[int]:
    """Cells that win now for X; if none, cells that stop O from winning."""
    for mark in ("X", "O"):
        found = set()
        for a, b, c in LINES:
            cells = [board[a], board[b], board[c]]
            if cells.count(mark) == 2 and cells.count(".") == 1:
                found.add((a, b, c)[cells.index(".")])
        if found:
            return found
    return set()


def judge(state: str, name: object, arguments: object) -> tuple[bool, str]:
    game, _, rest = state.partition(":")
    if game not in TOOLS:
        return False, f"unknown game {game!r}"
    tool, arg = TOOLS[game]
    if name != tool:
        return False, f"expected tool {tool}, got {name!r}"
    if not isinstance(arguments, dict) or set(arguments) != {arg}:
        return False, f"{tool} takes exactly one argument {arg!r}"
    v = arguments[arg]
    try:
        if game == "ttt":
            board = list(rest)
            if not _is_int(v) or not 0 <= v <= 8 or board[v] != ".":
                return False, "illegal move: not an empty cell 0-8"
            return (True, "") if v in _forced_cells(board) else (False, "not the winning/blocking cell")
        if game == "nim":
            if not _is_int(v) or not 1 <= v <= 3:
                return False, "illegal move: take 1, 2 or 3 stones"
            return (True, "") if (int(rest) - v) % 4 == 0 else (False, "does not leave a multiple of 4")
        if game == "guess":
            lo, hi = (int(x) for x in rest.split("-"))
            if not _is_int(v):
                return False, "guess must be an integer"
            return (True, "") if v == (lo + hi) // 2 else (False, "not the midpoint (rounded down)")
        step, start, length = (int(x) for x in rest.split(":"))
        nxt = ORDER[(start + step * length) % 3]
        beat = ORDER[(ORDER.index(nxt) + 1) % 3]  # paper beats rock, scissors beats paper, ...
        if v not in ORDER:
            return False, "move must be rock, paper or scissors"
        return (True, "") if v == beat else (False, f"does not beat the predicted {nxt}")
    except (ValueError, IndexError):
        return False, f"malformed state {state!r}"


def judge_batch(items: list[dict]) -> list[dict]:
    out = []
    for it in items:
        ok, reason = judge(str(it.get("state", "")), it.get("name"), it.get("arguments"))
        out.append({"id": it.get("id"), "ok": ok, "reason": reason})
    return out


def main() -> None:
    items = json.load(sys.stdin).get("items", [])
    json.dump({"results": judge_batch(items)}, sys.stdout)


if __name__ == "__main__":
    main()
