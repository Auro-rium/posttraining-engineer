import json

from posttrain_agent.experiments import games
from posttrain_agent.tools import dataset


def _prompt(t):
    return t["messages"][0]["content"]


def _expected_from_state(state: str):
    """Recompute the right answer from the canonical state key alone (independent of games.py)."""
    game, _, rest = state.partition(":")
    if game == "nim":
        return "take_stones", {"count": int(rest) % 4}
    if game == "guess":
        lo, hi = map(int, rest.split("-"))
        return "guess_number", {"value": (lo + hi) // 2}
    if game == "rps":
        step, start, length = map(int, rest.split(":"))
        order = ["rock", "paper", "scissors"]
        nxt = order[(start + step * length) % 3]
        return "play_move", {"move": {"rock": "paper", "paper": "scissors", "scissors": "rock"}[nxt]}
    board = list(rest)
    lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
    for mark in ("X", "O"):  # win now first, else block
        for a, b, c in lines:
            cells = [board[a], board[b], board[c]]
            if cells.count(mark) == 2 and cells.count(".") == 1:
                return "place_mark", {"cell": (a, b, c)[cells.index(".")]}
    raise AssertionError("no forced move")


def test_splits_are_full_disjoint_by_state_and_cover_all_games():
    used: set[str] = set()
    for seed, n in ((1, 120), (2, 60), (3, 40)):
        tasks = games.make_split(seed, n, used)
        states = {t["state"] for t in tasks}
        assert len(tasks) == n and len(states) == n and not states & used
        assert {t["id"].split("-")[0] for t in tasks} == {"ttt", "nim", "guess", "rps"}
        used |= states


def test_eval_answers_are_correct_and_match_the_state():
    for t in games.make_split(9, 120):
        name, args = _expected_from_state(t["state"])
        assert t["expected"] == {"name": name, "arguments": args}, t["state"]


def test_sft_rows_are_verified_correct_capped_and_never_use_excluded_states():
    eval_states = {t["state"] for seed in (1, 2, 3) for t in games.make_split(seed, 100)}
    rows, counts = games.generate_sft(5000, seed=100, exclude=eval_states, cap=1200)
    assert 0 < len(rows) <= 1200 and sum(counts.values()) == len(rows)
    assert dataset.verify_rows(rows, max_rows=1200)["ok"]  # schema, allow-list, duplicates
    for row in rows:
        meta, msgs = row["metadata"], row["messages"]
        assert meta["state"] not in eval_states
        name, args = _expected_from_state(meta["state"])
        call = msgs[1]["tool_calls"][0]["function"]
        assert call["name"] == name and json.loads(call["arguments"]) == args
        assert row["tools"][0]["function"]["name"] == name
        assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant"]
        assert msgs[-1]["content"]  # Nebius: the final message is natural language
    assert {m["metadata"]["style"] for m in rows} == {0, 1, 2}  # paraphrases are used
    assert min(counts["ttt"], counts["nim"], counts["guess"]) > 100  # roughly balanced


def test_sft_generation_is_deterministic_and_honours_a_smaller_n():
    a, _ = games.generate_sft(50, seed=7, exclude=set())
    b, _ = games.generate_sft(50, seed=7, exclude=set())
    assert a == b and len(a) == 50
    assert len(games.generate_sft(10, seed=8, exclude=set(), cap=3)[0]) == 3
