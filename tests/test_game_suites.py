import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "games", Path(__file__).parent.parent / "scripts" / "make_game_suites.py")
games = importlib.util.module_from_spec(spec)
spec.loader.exec_module(games)


def _prompt(t):
    return t["messages"][0]["content"]


def test_splits_are_full_disjoint_and_cover_all_games():
    used: set[str] = set()
    for seed, n in ((1, 40), (2, 12), (3, 12)):
        tasks = games.make_split(seed, n, used)
        prompts = {_prompt(t) for t in tasks}
        assert len(tasks) == n and len(prompts) == n
        assert not prompts & used
        assert {t["id"].split("-")[0] for t in tasks} == {"ttt", "nim", "guess", "rps"}
        used |= prompts


def test_expected_moves_are_correct():
    for t in games.make_split(9, 60):
        kind, args = t["id"].split("-")[0], t["expected"]["arguments"]
        text = _prompt(t)
        if kind == "nim":
            pile = int(text.split("has ")[1].split()[0])
            assert (pile - args["count"]) % 4 == 0 and 1 <= args["count"] <= 3
        elif kind == "guess":
            lo, hi = map(int, text.split("between ")[1].split(" inclusive")[0].split(" and "))
            assert args["value"] == (lo + hi) // 2
        elif kind == "ttt":
            rows = text.split("\n")[1:4]
            board = [c for r in rows for c in r.split()]
            assert board[args["cell"]] == "."
            assert games._winner_move(board, "X") in (None, args["cell"])
        else:
            moves = text.split("so far: ")[1].split(". Predict")[0].split(", ")
            order = ["rock", "paper", "scissors"]
            step = (order.index(moves[1]) - order.index(moves[0])) % 3
            nxt = order[(order.index(moves[-1]) + step) % 3]
            assert games.BEATS[nxt] == args["move"]
