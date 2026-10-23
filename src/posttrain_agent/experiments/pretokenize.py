"""Pretokenize SFT rows so ONLY the tool call is trained on (labels masked elsewhere).

Why: the provider's conversational format has no loss mask and wants a prose last message, so the
tool-call turn is probably not the training target. Here every row becomes
``{token_ids, labels, attention_mask}``: the prompt is exactly what serving renders
(``apply_chat_template(messages_up_to_user, tools=..., add_generation_prompt=True,
enable_thinking=False)``, see ``sandbox_student.py``), the target is the tool call the Qwen3
template itself renders for an assistant ``tool_calls`` turn, and ``labels`` is -100 elsewhere.

Needs the optional extra ``tokenize`` (``jinja2``, ``tokenizers``) and the base model's
``tokenizer.json`` + ``tokenizer_config.json`` in the Hugging Face cache (or the directory in
``POSTTRAIN_TOKENIZER_DIR``). Without them everything here raises ``TokenizerUnavailable`` and
callers fall back to an estimate (budget) or refuse (launch).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

IGNORE = -100
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


class TokenizerUnavailable(RuntimeError):
    """jinja2/tokenizers or the cached tokenizer files are missing."""


def find_tokenizer_dir(model: str = DEFAULT_MODEL) -> Path | None:
    env = os.environ.get("POSTTRAIN_TOKENIZER_DIR")
    if env and (Path(env) / "tokenizer.json").is_file():
        return Path(env)
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    pattern = str(home / "hub" / f"models--{model.replace('/', '--')}" / "snapshots" / "*")
    for d in sorted(glob.glob(pattern)):
        if (Path(d) / "tokenizer.json").is_file() and (Path(d) / "tokenizer_config.json").is_file():
            return Path(d)
    return None


def canonical_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Key order of ``games._tool`` / the eval suites. Datasets saved by older versions have sorted
    keys, but the template prints tools in dict order, so the order must be restored for train/serve
    parity (exact for the game tools; top-level order only for any other tool)."""
    from .games import TOOL_SCHEMAS

    fn = tool.get("function", {}) if isinstance(tool, dict) else {}
    known = TOOL_SCHEMAS.get(fn.get("name"))
    if known is not None and json.dumps(known, sort_keys=True) == json.dumps(tool, sort_keys=True):
        return known  # same content as a game tool: use its exact (suite) key order, nested too
    params = fn.get("parameters") or {}
    cparams = {k: params[k] for k in ("type", "properties", "required") if k in params}
    cparams.update({k: v for k, v in params.items() if k not in cparams})
    cfn = {k: fn[k] for k in ("name", "description") if k in fn}
    cfn["parameters"] = cparams
    cfn.update({k: v for k, v in fn.items() if k not in cfn})
    return {"type": tool.get("type", "function"), "function": cfn}


class ChatTokenizer:
    """Qwen3 chat template (rendered with jinja2) + tokenizers, no ``transformers`` needed."""

    def __init__(self, directory: Path) -> None:
        try:
            import jinja2
            from jinja2.sandbox import ImmutableSandboxedEnvironment
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - exercised by the missing-extra path
            raise TokenizerUnavailable(
                "install the 'tokenize' extra: pip install jinja2 tokenizers") from exc
        cfg = json.loads((directory / "tokenizer_config.json").read_text())
        template = cfg.get("chat_template")
        if isinstance(template, list):  # named templates: use the default one
            template = next((t["template"] for t in template if t.get("name") == "default"),
                            template[0]["template"])
        if not isinstance(template, str):
            raise TokenizerUnavailable(f"no chat_template in {directory}/tokenizer_config.json")

        def raise_exception(message: str) -> None:
            raise jinja2.exceptions.TemplateError(message)

        def tojson(x: Any, ensure_ascii: bool = False, indent: int | None = None,
                   separators: Any = None, sort_keys: bool = False) -> str:
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                              separators=separators, sort_keys=sort_keys)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        self._template = env.from_string(template)
        self._tok = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.directory = directory

    def render(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
               add_generation_prompt: bool) -> str:
        """Same arguments as ``sandbox_student.py`` uses at serving time."""
        return self._template.render(messages=messages, tools=tools or None,
                                     add_generation_prompt=add_generation_prompt,
                                     enable_thinking=False)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)

    def count(self, text: str) -> int:
        return len(self.encode(text))


@lru_cache(maxsize=4)
def _load(directory: str) -> ChatTokenizer:
    return ChatTokenizer(Path(directory))


def get_tokenizer(model: str = DEFAULT_MODEL) -> ChatTokenizer:
    d = find_tokenizer_dir(model)
    if d is None:
        raise TokenizerUnavailable(
            f"tokenizer files for {model} not found in the Hugging Face cache; set "
            "POSTTRAIN_TOKENIZER_DIR to a directory with tokenizer.json and tokenizer_config.json")
    return _load(str(d))


def available(model: str = DEFAULT_MODEL) -> bool:
    try:
        get_tokenizer(model)
        return True
    except TokenizerUnavailable:
        return False


# ------------------------------------------------------------------ rows


def split_row(row: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(prompt messages = everything before the first assistant turn, that assistant tool call)."""
    msgs = row["messages"]
    idx = next((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), None)
    if idx is None or idx == 0 or not msgs[idx].get("tool_calls"):
        raise ValueError("row needs prompt messages followed by an assistant tool_calls turn")
    prompt = [{k: v for k, v in m.items() if k in ("role", "content", "name")} for m in msgs[:idx]]
    return prompt, msgs[idx]["tool_calls"][0]


def _call_obj(call: dict[str, Any]) -> dict[str, Any]:
    fn = call["function"]
    args = fn.get("arguments")
    return {"name": fn["name"], "arguments": json.loads(args) if isinstance(args, str) else args}


def tool_call_target(call: dict[str, Any]) -> str:
    """The text the model must generate after the serving prompt."""
    obj = _call_obj(call)
    body = json.dumps({"name": obj["name"], "arguments": obj["arguments"]}, ensure_ascii=False)
    return f"<tool_call>\n{body}\n</tool_call><|im_end|>"


def derived_target(tok: ChatTokenizer, row: dict[str, Any]) -> str:
    """The same target, but derived by rendering the row through the template and diffing."""
    prompt, call = split_row(row)
    tools = [canonical_tool(t) for t in row.get("tools") or []]
    head = tok.render(prompt, tools, add_generation_prompt=True)
    asst = {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "function": _call_obj(call)}]}
    full = tok.render([*prompt, asst], tools, add_generation_prompt=False)
    if not full.startswith(head):
        raise ValueError("template output for the tool-call turn does not extend the serving prompt")
    return full[len(head):].removesuffix("\n")


def serving_prompt(tok: ChatTokenizer, row: dict[str, Any]) -> str:
    prompt, _ = split_row(row)
    return tok.render(prompt, [canonical_tool(t) for t in row.get("tools") or []],
                      add_generation_prompt=True)


def pretokenize_row(row: dict[str, Any], tok: ChatTokenizer) -> dict[str, list[int]]:
    prompt_ids = tok.encode(serving_prompt(tok, row))
    target_ids = tok.encode(tool_call_target(split_row(row)[1]))
    ids = prompt_ids + target_ids
    return {"token_ids": ids, "labels": [IGNORE] * len(prompt_ids) + target_ids,
            "attention_mask": [1] * len(ids)}


def pretok_path(dataset_path: Path) -> Path:
    return dataset_path.with_name(dataset_path.stem + ".pretok.jsonl")


def marker_path(dataset_path: Path) -> Path:
    return dataset_path.with_name(dataset_path.stem + ".pretok.sha256")


def pretokenize_file(dataset_path: Path, source_sha256: str | None = None,
                     tok: ChatTokenizer | None = None) -> dict[str, Any]:
    """Write ``<stem>.pretok.jsonl`` and the marker ``<stem>.pretok.sha256`` next to a dataset."""
    tok = tok or get_tokenizer()
    rows = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    body = "".join(json.dumps(pretokenize_row(r, tok)) + "\n" for r in rows)
    out = pretok_path(dataset_path)
    out.write_text(body)
    digest = hashlib.sha256(body.encode()).hexdigest()
    marker = {"pretok_sha256": digest, "source_sha256": source_sha256, "rows": len(rows),
              "tokenizer_dir": str(tok.directory)}
    marker_path(dataset_path).write_text(json.dumps(marker, sort_keys=True) + "\n")
    return {"pretok_ref": out.name, **marker}


def read_marker(dataset_path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(marker_path(dataset_path).read_text())
    except (OSError, ValueError):
        return None
