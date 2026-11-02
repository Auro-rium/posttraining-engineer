"""Deterministic SFT dataset verification and storage (chat-format JSONL).

This is a structural and policy check, not an environment replay: it proves a
dataset is well-formed, uses only allow-listed tools, contains no sealed
references and no duplicates. It never calls an LLM.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

OBJECTIVE_TOOLS = frozenset({
    "get_logs", "inspect_service", "read_config", "edit_config",
    "restart_service", "run_healthcheck",
    # game suites (scripts/make_game_suites.py)
    "place_mark", "take_stones", "guess_number", "play_move"})
ROLES = frozenset({"system", "user", "assistant", "tool"})
SEALED_MARKERS = ("heldout://", "sealed://", "heldout/", "sealed/")
VERIFIED = frozenset({"LIVE", "PRIOR_VERIFIED_RUN"})
MAX_ROWS = 1200  # hard cap on SFT dataset size
EVAL_SPLITS = ("train", "heldout", "regression", "heldout_phrasing")
RULE_VERIFIED = "RULE_VERIFIED"  # answer recomputed from the game rule by code
_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def _row_errors(row: Any) -> list[str]:
    if not isinstance(row, dict) or not isinstance(row.get("messages"), list) or not row["messages"]:
        return ["row must be an object with a non-empty 'messages' list"]
    errs: list[str] = []
    msgs = row["messages"]
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or m.get("role") not in ROLES:
            errs.append(f"message {i}: invalid role")
            continue
        content = m.get("content")
        if content is not None and not isinstance(content, str):
            errs.append(f"message {i}: content must be a string")
        calls = m.get("tool_calls") or []
        if calls and m["role"] != "assistant":
            errs.append(f"message {i}: only assistant messages may carry tool_calls")
        for c in calls:
            fn = (c or {}).get("function") or {}
            if fn.get("name") not in OBJECTIVE_TOOLS:
                errs.append(f"message {i}: tool {fn.get('name')!r} is not allow-listed")
            args = fn.get("arguments")
            try:
                parsed = json.loads(args) if isinstance(args, str) else args
                if not isinstance(parsed, dict):
                    raise ValueError
            except ValueError:
                errs.append(f"message {i}: tool arguments must be a JSON object")
        if not content and not calls:
            errs.append(f"message {i}: empty message")
    if msgs[-1].get("role") != "assistant":
        errs.append("last message must be from the assistant (the training target)")
    blob = json.dumps(row)
    if any(marker in blob for marker in SEALED_MARKERS):
        errs.append("row references sealed/held-out data")
    ev = (row.get("metadata") or {}).get("evidence")
    if ev is not None and ev not in VERIFIED | {"EXPLANATION", RULE_VERIFIED}:
        errs.append(f"unknown evidence label {ev!r}")
    return errs


def _semantic_errors(row: Any) -> list[str]:
    """Recompute the correct move from ``metadata.state`` with the game rules and compare it with
    the row's tool call. Rows without a game state (hand-built, ops tools) are not checked."""
    from ..experiments import games  # local import: games has no dependency on this module

    if not isinstance(row, dict) or not isinstance(row.get("metadata"), dict):
        return []
    state = row["metadata"].get("state")
    if not isinstance(state, str) or state.partition(":")[0] not in games.GAMES:
        return []
    want = games.expected_from_state(state)
    if want is None:
        return [f"metadata.state {state!r} is malformed or has no forced move"]
    calls = [c for m in row.get("messages") or [] if isinstance(m, dict)
             for c in m.get("tool_calls") or []]
    if len(calls) != 1:
        return [f"expected exactly one tool call, found {len(calls)}"]
    fn = (calls[0] or {}).get("function") or {}
    args = fn.get("arguments")
    try:
        got_args = json.loads(args) if isinstance(args, str) else args
    except ValueError:
        got_args = None
    if (fn.get("name"), got_args) != want:
        return [f"tool call {fn.get('name')}({got_args}) disagrees with the rule answer "
                f"{want[0]}({want[1]}) for state {state!r}"]
    return []


def verify_rows(rows: list[Any], *, min_rows: int = 1,
                max_rows: int = MAX_ROWS) -> dict[str, Any]:
    """Return {ok, row_count, errors, warnings, sha256, evidence}. Errors are capped at 20.

    ``max_rows`` can lower the limit but never raise it above the hard ``MAX_ROWS`` cap.
    """
    max_rows = min(max_rows, MAX_ROWS)
    errors: list[str] = []
    warnings: list[str] = []
    seen: dict[str, int] = {}
    labels: set[str] = set()
    for i, row in enumerate(rows):
        errors += [f"row {i}: {e}" for e in _row_errors(row)]
        errors += [f"row {i}: {e}" for e in _semantic_errors(row)]
        key = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        if key in seen:
            errors.append(f"row {i}: duplicate of row {seen[key]}")
        seen.setdefault(key, i)
        if isinstance(row, dict) and isinstance(row.get("metadata"), dict):
            if row["metadata"].get("evidence"):
                labels.add(row["metadata"]["evidence"])
    if len(rows) < min_rows:
        errors.append(f"need at least {min_rows} rows, got {len(rows)}")
    if len(rows) > max_rows:
        errors.append(f"too many rows: {len(rows)} > the {max_rows}-row cap; drop or sample rows")
    if "EXPLANATION" in labels:
        warnings.append("dataset contains EXPLANATION-labelled rows (simulated, not verified)")
    if not labels:
        warnings.append("no row carries metadata.evidence; provenance is unverified")
    digest = hashlib.sha256("\n".join(json.dumps(r, sort_keys=True) for r in rows).encode())
    return {"ok": not errors, "row_count": len(rows), "errors": errors[:20],
            "warnings": warnings, "sha256": digest.hexdigest(), "evidence": sorted(labels)}


def resolve(data_dir: Path, ref: str) -> Path:
    """Map a dataset ref to a file strictly inside ``data_dir`` (no traversal)."""
    if not _NAME.match(ref) or not ref.endswith(".jsonl"):
        raise ValueError("dataset ref must be a plain '<name>.jsonl' file name")
    return Path(data_dir) / ref


def load_rows(data_dir: Path, ref: str) -> list[Any]:
    path = resolve(data_dir, ref)
    rows: list[Any] = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                rows.append({"__invalid_json_line__": n})
    return rows


def verify_file(data_dir: Path, ref: str, max_rows: int = MAX_ROWS) -> dict[str, Any]:
    try:
        rows = load_rows(data_dir, ref)
    except (OSError, ValueError) as exc:
        return {"ok": False, "errors": [str(exc)], "row_count": 0, "warnings": [],
                "sha256": None, "evidence": []}
    return verify_rows(rows, max_rows=max_rows)


def row_states(rows: Iterable[Any]) -> set[str]:
    """Distinct canonical ``metadata.state`` values carried by dataset rows."""
    states: set[str] = set()
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("metadata"), dict):
            state = row["metadata"].get("state")
            if isinstance(state, str) and state:
                states.add(state)
    return states


def check_suite_overlap(data_dir: Path, ref: str, suite_dir: Path | str | None,
                        splits: Iterable[str] = EVAL_SPLITS) -> dict[str, Any]:
    """Compare a saved dataset's ``metadata.state`` values with every configured eval split.

    Deterministic and LLM-free. A missing suite directory is reported as a warning (nothing is
    configured, so there is nothing to compare); an existing suite whose files carry no ``state``
    keys is an error because disjointness cannot be proven. Any overlapping state blocks:
    ``errors`` names the split and how many training states it contains.
    """
    try:
        rows = load_rows(data_dir, ref)
    except (OSError, ValueError) as exc:
        return {"ok": False, "dataset_ref": ref, "errors": [str(exc)], "warnings": []}
    from ..experiments import games  # local import: games has no dependency on this module

    names = tuple(splits)
    training_states = row_states(rows)
    root = Path(suite_dir) if suite_dir is not None else None
    base: dict[str, Any] = {
        "dataset_ref": ref, "dataset_states": len(training_states),
        "suite_dir": str(root) if root is not None else None, "split_states": {},
        "overlap_counts": dict.fromkeys(names, 0), "overlap_total": 0,
        "overlap_states": {}, "errors": [], "warnings": []}
    if root is None or not root.is_dir():
        base["warnings"].append(
            f"suite directory {root} does not exist; train/eval state overlap not checked")
        return {**base, "ok": True}
    per_split = games.suite_split_states(root, names)
    base["split_states"] = {split: len(states) for split, states in per_split.items()}
    if not any(per_split.values()):
        return {**base, "ok": False, "errors": [(
            f"evaluation suite {root} carries no 'state' keys; cannot prove training data is "
            "disjoint from evaluation data (regenerate it with scripts/make_game_suites.py)")]}
    errors: list[str] = []
    overlaps: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    for split in names:
        hit = sorted(training_states & per_split.get(split, set()))
        counts[split] = len(hit)
        if hit:
            overlaps[split] = hit
            errors.append(
                f"contamination: {len(hit)} training state(s) also appear in eval split "
                f"'{split}' ({root / f'{split}.jsonl'}); SFT data must exclude every "
                "evaluation state")
    base.update(overlap_counts=counts, overlap_total=len(set().union(*overlaps.values())),
                overlap_states=overlaps, errors=errors)
    return {**base, "ok": not overlaps}


def save_rows(data_dir: Path, ref: str, rows: list[Any],
              max_rows: int = MAX_ROWS) -> dict[str, Any]:
    """Verify then write ``rows`` as JSONL. Nothing is written unless verification passes."""
    try:
        path = resolve(data_dir, ref)
    except ValueError as exc:
        return {"ok": False, "errors": [str(exc)]}
    report = verify_rows(rows, max_rows=max_rows)
    if not report["ok"]:
        return report
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))  # keep key order (tool schemas)
    return {**report, "dataset_ref": ref}


# ---------------------------------------------------------------- inspection and template check


def check_marker_path(data_dir: Path, ref: str) -> Path:
    p = resolve(data_dir, ref)
    return p.with_name(p.stem + ".check.json")


def check_marker_ok(data_dir: Path, ref: str, sha256: str | None) -> bool:
    """True only if ``check_chat_template`` passed for THIS exact dataset content (sha256) and the
    pretokenized file it wrote is still the one the marker names."""
    from ..experiments import pretokenize as pt

    if not sha256:
        return False
    try:
        marker = json.loads(check_marker_path(data_dir, ref).read_text())
        pre = pt.pretok_path(resolve(data_dir, ref))
        pre_sha = hashlib.sha256(pre.read_bytes()).hexdigest()
    except (OSError, ValueError):
        return False
    return (marker.get("ok") is True and marker.get("sha256") == sha256
            and marker.get("pretok_sha256") == pre_sha)


def check_approved_sha(approved: str | None, actual: str, require: bool) -> str | None:
    """Return an error when the dataset a human approved is not the dataset about to be uploaded.

    The approval prompt shows the sha256 of the file as it was when the prompt was built; the agent
    echoes that value back in ``launch_training(dataset_sha256=...)``. Any edit between approval and
    upload changes ``actual`` and is refused. A missing value is refused when approval is on, since
    the approval would otherwise cover a path rather than content.
    """
    if not approved:
        return ("dataset_sha256 is required so the approval covers this exact file; pass the "
                "sha256 returned by check_chat_template" if require else None)
    if approved.strip().lower() != actual:
        return (f"dataset changed since it was approved: approved sha256 {approved}, current "
                f"{actual}; re-run check_chat_template and request approval again")
    return None


def inspect_dataset(data_dir: Path, ref: str, n: int = 3, game: str | None = None,
                    max_rows: int = MAX_ROWS) -> dict[str, Any]:
    """Counts per game / style / evidence plus ``n`` sample rows (optionally of one game) as the
    model will see them: serving prompt text and the training target."""
    report = verify_file(data_dir, ref, max_rows)
    try:
        rows = [r for r in load_rows(data_dir, ref) if isinstance(r, dict)]
    except (OSError, ValueError) as exc:
        return {"ok": False, "errors": [str(exc)]}
    per_game: dict[str, int] = {}
    styles: dict[str, int] = {}
    states: set[str] = set()
    for r in rows:
        meta = r.get("metadata") or {}
        per_game[str(meta.get("game", "?"))] = per_game.get(str(meta.get("game", "?")), 0) + 1
        styles[str(meta.get("style", "?"))] = styles.get(str(meta.get("style", "?")), 0) + 1
        if meta.get("state"):
            states.add(meta["state"])
    picked = [r for r in rows if game is None or (r.get("metadata") or {}).get("game") == game]
    samples = []
    for r in picked[:max(0, min(n, 10))]:
        item: dict[str, Any] = {"state": (r.get("metadata") or {}).get("state")}
        try:
            from ..experiments import pretokenize as pt

            tok = pt.get_tokenizer()
            item["serving_prompt"] = pt.serving_prompt(tok, r)
            item["training_target"] = pt.tool_call_target(pt.split_row(r)[1])
        except Exception as exc:  # noqa: BLE001 - inspection must still show the raw messages
            item["messages"] = r.get("messages")
            item["note"] = f"template rendering unavailable: {exc}"
        samples.append(item)
    return {"ok": report["ok"], "dataset_ref": ref, "sha256": report["sha256"],
            "row_count": len(rows), "distinct_states": len(states), "by_game": per_game,
            "by_style": styles, "evidence": report["evidence"], "errors": report["errors"],
            "warnings": report["warnings"], "samples": samples}


def check_chat_template(data_dir: Path, ref: str, max_rows: int = MAX_ROWS, *,
                        suite_dir: Path | str | None = None) -> dict[str, Any]:
    """Pretokenize the dataset and prove the training rows match serving. On success writes the
    pretokenized file and the check marker that ``launch_training`` requires.

    Per row: the unmasked labels are ONE contiguous span at the end that decodes to exactly the
    tool call (and parses with the evaluator's ``parse_tool_calls``); the masked prefix equals the
    serving prompt (same template call as ``sandbox_student``); template-derived target equals the
    literal one; tokenisation is stable across the prompt/target boundary.

    With ``suite_dir`` it first proves the dataset's states are disjoint from every evaluation
    split (train, heldout, regression, heldout_phrasing) and blocks contaminated data.
    """
    from ..experiments import pretokenize as pt
    from ..experiments.sandbox_student import parse_tool_calls

    report = verify_file(data_dir, ref, max_rows)
    if not report["ok"]:
        return {"ok": False, "stage": "verify", "errors": report["errors"]}
    overlap = check_suite_overlap(data_dir, ref, suite_dir)
    if not overlap["ok"]:
        return {"ok": False, "stage": "contamination", "errors": overlap["errors"],
                "overlap": overlap}
    try:
        tok = pt.get_tokenizer()
    except pt.TokenizerUnavailable as exc:
        return {"ok": False, "stage": "tokenizer", "errors": [str(exc)]}
    path = resolve(data_dir, ref)
    errors: list[str] = []
    sup = total = 0
    per_game: dict[str, int] = {}
    lengths: list[int] = []
    for i, row in enumerate(load_rows(data_dir, ref)):
        try:
            prompt, call = pt.split_row(row)
            ex = pt.pretokenize_row(row, tok)
            ids, labels = ex["token_ids"], ex["labels"]
            sup_idx = [j for j, lab in enumerate(labels) if lab != pt.IGNORE]
            if not sup_idx or sup_idx != list(range(sup_idx[0], len(ids))):
                errors.append(f"row {i}: unmasked labels are not one contiguous trailing span")
                continue
            if any(labels[j] != ids[j] for j in sup_idx):
                errors.append(f"row {i}: labels differ from token_ids on the target")
            target = tok.decode(ids[sup_idx[0]:])
            if target != pt.tool_call_target(call):
                errors.append(f"row {i}: unmasked span decodes to {target!r}, not the tool call")
            if pt.derived_target(tok, row) != target:
                errors.append(f"row {i}: target differs from what the chat template renders")
            if tok.decode(ids[:sup_idx[0]]) != pt.serving_prompt(tok, row):
                errors.append(f"row {i}: masked prefix is not the serving prompt")
            parsed = parse_tool_calls(target)
            want = pt._call_obj(call)
            if parsed != [{"name": want["name"], "arguments": want["arguments"]}]:
                errors.append(f"row {i}: parse_tool_calls gives {parsed}, expected {want}")
            if tok.encode(pt.serving_prompt(tok, row) + target) != ids:
                errors.append(f"row {i}: tokenisation differs across the prompt/target boundary")
            if prompt[-1].get("role") != "user":
                errors.append(f"row {i}: last prompt message is {prompt[-1].get('role')!r}, not user")
            sup += len(sup_idx)
            total += len(ids)
            lengths.append(len(ids))
            g = str((row.get("metadata") or {}).get("game", "?"))
            per_game[g] = per_game.get(g, 0) + 1
        except Exception as exc:  # noqa: BLE001 - one bad row is a reported error, not a crash
            errors.append(f"row {i}: {type(exc).__name__}: {exc}")
        if len(errors) >= 20:
            break
    out: dict[str, Any] = {
        "ok": not errors, "dataset_ref": ref, "sha256": report["sha256"], "row_count": report["row_count"],
        "supervised_token_fraction": round(sup / total, 4) if total else 0.0,
        "avg_row_tokens": round(total / len(lengths), 1) if lengths else 0,
        "total_tokens": total, "by_game": per_game, "errors": errors[:20]}
    if errors:
        return out
    info = pt.pretokenize_file(path, report["sha256"], tok)
    marker = {"ok": True, "sha256": report["sha256"], "pretok_sha256": info["pretok_sha256"],
              "pretok_ref": info["pretok_ref"], "rows": report["row_count"],
              "total_tokens": total, "avg_row_tokens": out["avg_row_tokens"],
              "supervised_token_fraction": out["supervised_token_fraction"]}
    check_marker_path(data_dir, ref).write_text(json.dumps(marker, sort_keys=True) + "\n")
    return {**out, "pretok_ref": info["pretok_ref"], "pretok_sha256": info["pretok_sha256"]}
