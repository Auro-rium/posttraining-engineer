"""Base vs adapter eval charts for one run, straight from the event journal (SVG, no dependencies).

    python scripts/plot_evals.py live-06b [out.svg]

Reads the operator bundle written by eval_candidate.py (ledger-backed, preferred) or the evaluator
results (``a2a.message`` evaluator -> coordinator, older runs) and draws per-game
scores with 95% Wilson intervals for the held-out and regression splits. A role that has no
finished evaluation yet is drawn as "pending", never as zero.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from pathlib import Path

GAMES = [("ttt", "tic-tac-toe"), ("nim", "Nim"), ("guess", "guessing"), ("rps", "rock-paper-scissors")]
COLORS = {"champion": "#8a93a6", "candidate": "#22d3a6"}
LABELS = {"champion": "base", "candidate": "adapter"}


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load(db: Path, run: str) -> dict[str, dict]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out: dict[str, dict] = {}
    for (data,) in conn.execute(
        "select data from events where run=? and type='a2a.message' order by seq", (run,)
    ):
        d = json.loads(data)
        if d.get("from") != "evaluator" or not d.get("ok", True):
            continue
        try:
            r = json.loads(d["text"].strip().removeprefix("```json").removesuffix("```"))
        except (ValueError, KeyError):
            continue
        if r.get("status") == "SUCCEEDED" and r.get("role") in COLORS:
            out[r["role"]] = r  # the latest finished evaluation per role wins
    op = db.parent / "operator_evals" / f"{run}-candidate.json"  # scripts/eval_candidate.py
    if op.is_file():  # ledger-backed bundle (both splits valid + base results): preferred
        r = json.loads(op.read_text(encoding="utf-8"))
        if r.get("status") == "SUCCEEDED":
            out["candidate"] = r
            if isinstance(r.get("base"), dict):
                out["champion"] = r["base"]
    return out


def panel(res: dict[str, dict], split: str, x0: int, title: str) -> str:
    w, h, top, left = 520, 300, 70, 50
    s = [f'<text x="{x0 + left}" y="62" class="t">{title}</text>']
    for i in range(6):
        y = top + h - i * h / 5
        s.append(f'<line x1="{x0 + left}" x2="{x0 + left + w}" y1="{y}" y2="{y}" class="g"/>'
                 f'<text x="{x0 + left - 8}" y="{y + 4}" class="a" text-anchor="end">{i / 5:.1f}</text>')
    cats = [(g, n) for g, n in GAMES] + [("all", "overall")]
    slot = w / len(cats)
    for ci, (g, name) in enumerate(cats):
        cx = x0 + left + slot * (ci + 0.5)
        s.append(f'<text x="{cx}" y="{top + h + 22}" class="a" text-anchor="middle">{name}</text>')
        for ri, role in enumerate(("champion", "candidate")):
            r = (res.get(role) or {}).get(split)
            bx = cx - 34 + ri * 36
            if not r:
                s.append(f'<text x="{bx + 14}" y="{top + h - 8}" class="p" text-anchor="middle" '
                         f'transform="rotate(-90 {bx + 14} {top + h - 8})">pending</text>')
                continue
            cell = r if g == "all" else (r.get("by_game") or {}).get(g)
            if not cell:
                continue
            p, n = float(cell["score"]), int(cell["n"])
            lo, hi = wilson(p, n)
            bh = p * h
            s.append(f'<rect x="{bx}" y="{top + h - bh}" width="28" height="{max(bh, 1)}" rx="3" '
                     f'fill="{COLORS[role]}"/>')
            xm = bx + 14
            s.append(f'<line x1="{xm}" x2="{xm}" y1="{top + h - hi * h}" y2="{top + h - lo * h}" class="e"/>')
            s.append(f'<text x="{xm}" y="{top + h - hi * h - 6}" class="v" text-anchor="middle">{p:.2f}</text>')
            s.append(f'<text x="{xm}" y="{top + h + 40}" class="n" text-anchor="middle">n={n}</text>'
                     if ri == 0 else "")
    return "\n".join(s)


def render(run: str, res: dict[str, dict]) -> str:
    legend = "".join(
        f'<rect x="{60 + i * 120}" y="14" width="14" height="14" rx="3" fill="{COLORS[r]}"/>'
        f'<text x="{80 + i * 120}" y="26" class="a">{LABELS[r]}</text>'
        for i, r in enumerate(("champion", "candidate")))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="470" viewBox="0 0 1200 470">
<style>
text{{font-family:Inter,Segoe UI,Arial,sans-serif;fill:#d7dce6}}
.t{{font-size:17px;font-weight:600}} .a{{font-size:12px;fill:#9aa3b5}}
.v{{font-size:12px;font-weight:600}} .n{{font-size:11px;fill:#6b7488}} .p{{font-size:11px;fill:#6b7488}}
.g{{stroke:#262c3a;stroke-width:1}} .e{{stroke:#e5e9f2;stroke-width:1.5}}
</style>
<rect width="1200" height="470" fill="#0d1017"/>
{legend}
<text x="1140" y="26" class="a" text-anchor="end">run {run} · Qwen3-0.6B · bars = score, whiskers = 95% Wilson interval</text>
{panel(res, "heldout", 0, "Held-out split")}
{panel(res, "regression", 600, "Regression split")}
</svg>"""


def main() -> None:
    run = sys.argv[1]
    out = Path(sys.argv[2] if len(sys.argv) > 2 else f"state/media/evals-{run}.svg")
    res = load(Path("state/events.sqlite"), run)
    out.write_text(render(run, res), encoding="utf-8")
    print(out, "| roles with results:", sorted(res))


if __name__ == "__main__":
    main()
