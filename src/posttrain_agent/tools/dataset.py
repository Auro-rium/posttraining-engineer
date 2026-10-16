"""Deterministic SFT dataset verification and storage (chat-format JSONL).

This is a structural and policy check, not an environment replay: it proves a
dataset is well-formed, uses only allow-listed tools, contains no sealed
references and no duplicates. It never calls an LLM.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

OBJECTIVE_TOOLS = frozenset({"get_logs", "inspect_service", "read_config", "edit_config",
                             "restart_service", "run_healthcheck"})
ROLES = frozenset({"system", "user", "assistant", "tool"})
SEALED_MARKERS = ("heldout://", "sealed://", "heldout/", "sealed/")
VERIFIED = frozenset({"LIVE", "PRIOR_VERIFIED_RUN"})
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
    if ev is not None and ev not in VERIFIED | {"EXPLANATION"}:
        errs.append(f"unknown evidence label {ev!r}")
    return errs


def verify_rows(rows: list[Any], *, min_rows: int = 1) -> dict[str, Any]:
    """Return {ok, row_count, errors, warnings, sha256, evidence}. Errors are capped at 20."""
    errors: list[str] = []
    warnings: list[str] = []
    seen: dict[str, int] = {}
    labels: set[str] = set()
    for i, row in enumerate(rows):
        errors += [f"row {i}: {e}" for e in _row_errors(row)]
        key = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        if key in seen:
            errors.append(f"row {i}: duplicate of row {seen[key]}")
        seen.setdefault(key, i)
        if isinstance(row, dict) and isinstance(row.get("metadata"), dict):
            if row["metadata"].get("evidence"):
                labels.add(row["metadata"]["evidence"])
    if len(rows) < min_rows:
        errors.append(f"need at least {min_rows} rows, got {len(rows)}")
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


def verify_file(data_dir: Path, ref: str) -> dict[str, Any]:
    try:
        rows = load_rows(data_dir, ref)
    except (OSError, ValueError) as exc:
        return {"ok": False, "errors": [str(exc)], "row_count": 0, "warnings": [],
                "sha256": None, "evidence": []}
    return verify_rows(rows)


def save_rows(data_dir: Path, ref: str, rows: list[Any]) -> dict[str, Any]:
    """Verify then write ``rows`` as JSONL. Nothing is written unless verification passes."""
    try:
        path = resolve(data_dir, ref)
    except ValueError as exc:
        return {"ok": False, "errors": [str(exc)]}
    report = verify_rows(rows)
    if not report["ok"]:
        return report
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return {**report, "dataset_ref": ref}
