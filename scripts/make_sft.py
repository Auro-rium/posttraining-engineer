"""Build the v2 SFT dataset deterministically and prove it is disjoint from every eval split.

The rows are computed by ``games.generate_sft`` (the same generator the agent's
``generate_game_sft`` tool uses) with every state in the configured suite excluded, including
``heldout_phrasing``. This script never calls an LLM. It saves the JSONL, then runs the dataset
verifier, the inspector and the chat-template check, and prints one JSON summary.

    python scripts/make_sft.py
    python scripts/make_sft.py --suite state/suite_v2 --out state/data/game-sft-v2.jsonl

If the optional tokenizer extra or cached Qwen3 tokenizer is unavailable, the JSONL is still
written and verified; ``chat_template`` in the summary reports that clearly and the script exits
successfully as long as verification and the state-overlap check passed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posttrain_agent.config import Settings
from posttrain_agent.experiments import games
from posttrain_agent.experiments import pretokenize as pt
from posttrain_agent.tools import dataset


def build_summary(out: Path, suite: Path, rows: int, seed: int) -> dict:
    data_dir = out.parent
    ref = out.name
    split_states = games.suite_split_states(suite)
    excluded = set().union(*split_states.values()) if split_states else set()
    if not excluded:
        return {"ok": False, "out": str(out), "suite_dir": str(suite),
                "errors": [(f"suite {suite} carries no 'state' keys; refusing to generate data "
                            "that cannot be proven disjoint")]}
    requested = min(max(0, rows), dataset.MAX_ROWS)
    generated, counts = games.generate_sft(requested, seed, excluded, dataset.MAX_ROWS)
    generated_states = dataset.row_states(generated)
    generated_overlap = sorted(generated_states & excluded)
    if generated_overlap:
        return {"ok": False, "out": str(out), "suite_dir": str(suite),
                "generated_overlap_count": len(generated_overlap),
                "errors": [(f"generator leaked {len(generated_overlap)} excluded eval state(s); "
                            "nothing was saved")]}

    save = dataset.save_rows(data_dir, ref, generated, dataset.MAX_ROWS)
    verify = dataset.verify_file(data_dir, ref, dataset.MAX_ROWS) if save["ok"] else save
    inspect = (dataset.inspect_dataset(data_dir, ref, n=1, max_rows=dataset.MAX_ROWS)
               if verify["ok"] else {"ok": False, "errors": verify.get("errors", [])})
    overlap = dataset.check_suite_overlap(data_dir, ref, suite)
    template = (dataset.check_chat_template(data_dir, ref, dataset.MAX_ROWS, suite_dir=suite)
                if verify["ok"] else {"ok": False, "stage": "verify",
                                      "errors": verify.get("errors", [])})
    tokenizer_available = pt.available()
    tokenizer_missing = template.get("stage") == "tokenizer"
    data_ok = (bool(save["ok"]) and bool(verify["ok"]) and bool(inspect.get("ok"))
               and bool(overlap["ok"]) and not generated_overlap)
    template_ok = bool(template.get("ok"))
    hard_ok = data_ok and (template_ok or tokenizer_missing)
    return {
        "ok": hard_ok,
        "out": str(out),
        "suite_dir": str(suite),
        "seed": seed,
        "requested_rows": requested,
        "row_count": len(generated),
        "distinct_states": len(generated_states),
        "max_rows": dataset.MAX_ROWS,
        "suite_states": {split: len(states) for split, states in split_states.items()},
        "excluded_eval_states": len(excluded),
        "generated_overlap_count": len(generated_overlap),
        "by_game": counts,
        "by_style": inspect.get("by_style", {}),
        "sha256": verify.get("sha256"),
        "verify": {"ok": verify.get("ok"),
                   "errors": verify.get("errors", []),
                   "warnings": verify.get("warnings", [])},
        "inspect": {"ok": inspect.get("ok"), "row_count": inspect.get("row_count"),
                    "distinct_states": inspect.get("distinct_states"),
                    "by_game": inspect.get("by_game", {}),
                    "by_style": inspect.get("by_style", {}),
                    "errors": inspect.get("errors", []),
                    "warnings": inspect.get("warnings", [])},
        "overlap": {"ok": overlap.get("ok"),
                    "overlap_total": overlap.get("overlap_total"),
                    "overlap_counts": overlap.get("overlap_counts", {}),
                    "errors": overlap.get("errors", []),
                    "warnings": overlap.get("warnings", [])},
        "tokenizer_available": tokenizer_available,
        "chat_template": {
            "ok": template.get("ok"),
            "stage": template.get("stage"),
            "errors": template.get("errors", []),
            "pretok_ref": template.get("pretok_ref"),
            "supervised_token_fraction": template.get("supervised_token_fraction"),
            "note": ("tokenizer extra/cache unavailable; JSONL was still created and verified"
                     if tokenizer_missing else None)},
        "never_calls_a_model": True,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=None,
                   help="output JSONL (default: <configured data_dir>/game-sft-v2.jsonl)")
    p.add_argument("--suite", type=Path, default=None,
                   help="suite directory (default: configured POSTTRAIN_SUITE_DIR)")
    p.add_argument("--rows", type=int, default=dataset.MAX_ROWS,
                   help=f"row cap, at most {dataset.MAX_ROWS} (default: {dataset.MAX_ROWS})")
    p.add_argument("--seed", type=int, default=100, help="deterministic generator seed")
    args = p.parse_args(argv)

    settings = Settings.from_env()
    out = args.out or Path(settings.data_dir) / "game-sft-v2.jsonl"
    suite = args.suite or Path(settings.suite_dir)
    summary = build_summary(out, suite, args.rows, args.seed)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
