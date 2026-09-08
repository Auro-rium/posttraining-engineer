#!/usr/bin/env python3
"""Validate living documentation and changelog discipline.

The checker has no third-party dependencies so it can run before the backend
environment is installed. On pull requests, pass ``--base-ref`` to require a
ChangeLog.md edit whenever implementation or operational files changed.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LIVING_DOCUMENTS = (
    "AGENTS.md",
    "ChangeLog.md",
    "Decisions.md",
    "Flow.md",
    "README.md",
)
TRACKED_PREFIXES = (
    ".github/workflows/",
    "backend/app/",
    "backend/tests/",
)
TRACKED_FILES = {
    ".env.example",
    ".gitignore",
    "AGENTS.md",
    "backend/Dockerfile",
    "backend/pyproject.toml",
    "backend/uv.lock",
    "docker-compose.yml",
}
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
FLOW_REFERENCE = re.compile(r"<!--\s*flow-ref:\s*([^:>\s]+)::([A-Za-z_]\w*)\s*-->")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-ref",
        help="Git revision compared with HEAD to enforce ChangeLog.md updates.",
    )
    return parser.parse_args()


def changed_files(base_ref: str) -> set[str]:
    command = ["git", "diff", "--name-only", "--diff-filter=ACMRT", f"{base_ref}...HEAD"]
    result = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def requires_changelog(path: str) -> bool:
    return (
        path in TRACKED_FILES
        or path == "backend/scripts/check_docs_sync.py"
        or path.startswith(TRACKED_PREFIXES)
    )


def validate_required_documents(errors: list[str]) -> None:
    for relative_path in LIVING_DOCUMENTS:
        if not (REPOSITORY_ROOT / relative_path).is_file():
            errors.append(f"missing required living document: {relative_path}")


def validate_local_links(errors: list[str]) -> None:
    for relative_path in LIVING_DOCUMENTS:
        document = REPOSITORY_ROOT / relative_path
        if not document.is_file():
            continue
        lines = document.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            for match in MARKDOWN_LINK.finditer(line):
                target = match.group(1).strip().strip("<>")
                if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                    continue
                path_text = unquote(target.split("#", maxsplit=1)[0])
                candidate = (document.parent / path_text).resolve()
                try:
                    candidate.relative_to(REPOSITORY_ROOT)
                except ValueError:
                    errors.append(
                        f"{relative_path}:{line_number}: local link escapes repository: {target}"
                    )
                    continue
                if not candidate.exists():
                    errors.append(
                        f"{relative_path}:{line_number}: missing local link target: {target}"
                    )


def top_level_python_symbols(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def validate_flow_references(errors: list[str]) -> None:
    flow_path = REPOSITORY_ROOT / "Flow.md"
    if not flow_path.is_file():
        return
    cache: dict[Path, set[str]] = {}
    for line_number, line in enumerate(flow_path.read_text(encoding="utf-8").splitlines(), start=1):
        for relative_path, symbol in FLOW_REFERENCE.findall(line):
            source = REPOSITORY_ROOT / relative_path
            if not source.is_file():
                errors.append(f"Flow.md:{line_number}: missing flow source: {relative_path}")
                continue
            if source.suffix != ".py":
                continue
            try:
                symbols = cache.setdefault(source, top_level_python_symbols(source))
            except SyntaxError as exc:
                errors.append(f"Flow.md:{line_number}: cannot parse {relative_path}: {exc.msg}")
                continue
            if symbol not in symbols:
                errors.append(
                    f"Flow.md:{line_number}: {symbol} is not a top-level symbol "
                    f"in {relative_path}"
                )


def validate_changelog(base_ref: str | None, errors: list[str]) -> None:
    if not base_ref:
        return
    try:
        paths = changed_files(base_ref)
    except RuntimeError as exc:
        errors.append(f"unable to enforce changelog: {exc}")
        return
    implementation_changes = sorted(path for path in paths if requires_changelog(path))
    if implementation_changes and "ChangeLog.md" not in paths:
        preview = ", ".join(implementation_changes[:5])
        errors.append(f"implementation changed without ChangeLog.md update: {preview}")


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    validate_required_documents(errors)
    validate_local_links(errors)
    validate_flow_references(errors)
    validate_changelog(args.base_ref, errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Living documentation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
