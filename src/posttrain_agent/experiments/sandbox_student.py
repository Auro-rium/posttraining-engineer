"""Serve the base model or a LoRA checkpoint on CPU inside a Nebius Sandbox (ConTree).

Why: Nebius serverless ``/chat/completions`` does not serve ``Qwen/Qwen3-0.6B`` nor the ``ft:``
LoRA checkpoints trained on it (HTTP 404 "model does not exist"). Training still runs on Nebius
Token Factory; evaluation generates with ``transformers`` (+ ``peft``) on sandbox CPU.

Adapted from distillery's ``sandbox_student.py``, which measured live (Qwen3-0.6B, bf16, 4 CPUs,
~4 GB): pip install of CPU torch + transformers/peft ~63 s, weights download ~20 s, load ~1.3 s,
~7.8 s per sample at ~900 prompt tokens and ``max_new_tokens=160``.

Token Factory LoRA checkpoints are ``adapter_config.json`` + ``adapter_model.safetensors`` whose
tensor keys LACK peft's ``base_model.model.`` prefix. ``PeftModel.from_pretrained`` silently loads
nothing from such keys and leaves a random adapter, so the script adds the prefix and loads
strictly: any unexpected key or missing LoRA key fails the job (``adapter_load_problem``).

Flow per process: the heavy image (pip + base weights + script) is built ONCE as a non-disposable
branch of ``POSTTRAIN_SANDBOX_IMAGE``; each adapter is a small extra layer on top of it (uploaded
as bytes); generation batches then run as disposable jobs, at most ``CONCURRENCY`` at a time, with
the batch on stdin. Greedy decoding, ``max_new_tokens`` <= 160, thinking disabled.

Secrets: the API key stays on the host (it authenticates the SDK and the adapter download). The
sandbox receives only the generation script, task messages/tools and adapter bytes; no ``env`` is
passed and no token of any kind (Nebius or Hugging Face) is put into a command, stdin or file.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import shlex
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..config import Settings
from .sandbox import SandboxUnavailable, _run_sync, contree_client

SERVED_BY = "nebius-sandbox-cpu"
OUT_MARKER = "POSTTRAIN_OUT:"
BASE_DIR = "/models/base"
ADAPTER_DIR = "/work/adapter"
SCRIPT_PATH = "/work/gen.py"
MAX_NEW_TOKENS = 160  # hard ceiling; a tool call needs ~20-40 tokens
BATCH_SIZE = 8  # samples per generation job (one model load per job)
CONCURRENCY = 4  # generation jobs in flight
SETUP_TIMEOUT_S = 1500.0
ADAPTER_TIMEOUT_S = 600.0
JOB_BASE_TIMEOUT_S = 240.0  # python + torch import + model load, with headroom
JOB_PER_SAMPLE_TIMEOUT_S = 45.0
TOTAL_TIMEOUT_S = 3600.0  # one whole generate() call
TRUNCATE_OUTPUT_AT = 1_000_000  # the SDK default (64 KiB) would cut the JSON result
TRANSPORT_TIMEOUT_S = 600.0  # SDK default 10 s timed out uploading a LoRA adapter (live)
ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")  # the names peft loads
END_TOKENS = ("<|im_end|>", "<|endoftext|>")


class StudentError(RuntimeError):
    """Serving failed. ``done``/``total`` count samples generated before the failure."""

    def __init__(self, message: str, *, done: int = 0, total: int = 0, stage: str = "inference"):
        super().__init__(message)
        self.done, self.total, self.stage = done, total, stage


# ------------------------------------------------- pure logic (also shipped inside the sandbox)


def peft_state_keys(keys: Iterable[str], prefix: str = "base_model.model.") -> dict[str, str]:
    """Map each checkpoint tensor key to the key peft expects (adds the missing model prefix)."""
    return {k: (k if k.startswith(prefix) else prefix + k) for k in keys}


def adapter_load_problem(state_keys: Sequence[str], unexpected_keys: Sequence[str],
                         missing_keys: Sequence[str]) -> str | None:
    """Why a strict LoRA load failed, or None when every adapter tensor was applied.

    ``missing_keys`` from ``load_state_dict(strict=False)`` lists every base weight too; only
    missing LoRA keys count. A state without any LoRA tensor is a failure as well.
    """
    unexpected = list(unexpected_keys)
    lora_missing = [k for k in missing_keys if "lora_" in k]
    n_lora = sum("lora_" in k for k in state_keys)
    if not state_keys or n_lora == 0:
        return f"adapter has no LoRA tensors ({len(state_keys)} tensors in the file)"
    if unexpected or lora_missing:
        return ("adapter weights did not load cleanly: %d tensors, %d unexpected keys, %d missing "
                "lora keys; e.g. %s" % (len(state_keys), len(unexpected), len(lora_missing),
                                        (unexpected + lora_missing)[:3]))
    return None


# ------------------------------------------------- the generation script

_GEN_HEAD = """\
from __future__ import annotations  # the spliced helpers' annotations name typing types

import argparse
import json
import os
import resource
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter")
ap.add_argument("--max-new-tokens", type=int, default=160)
args = ap.parse_args()
batch = json.load(sys.stdin)["items"]
"""

_GEN_BODY = """\
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(os.cpu_count() or 4)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(args.base)
try:
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
except TypeError:  # older transformers
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
versions = {"torch": torch.__version__, "transformers": transformers.__version__}
adapter_tensors = None
if args.adapter:
    import peft
    from peft import PeftConfig, get_peft_model, set_peft_model_state_dict

    versions["peft"] = peft.__version__
    pm = get_peft_model(model, PeftConfig.from_pretrained(args.adapter))
    safe = os.path.join(args.adapter, "adapter_model.safetensors")
    if os.path.exists(safe):
        from safetensors.torch import load_file

        state = load_file(safe)
    else:
        state = torch.load(os.path.join(args.adapter, "adapter_model.bin"), map_location="cpu")
    mapping = peft_state_keys(list(state))
    state = {mapping[k]: v for k, v in state.items()}
    res = set_peft_model_state_dict(pm, state)
    problem = adapter_load_problem(list(state), list(res.unexpected_keys), list(res.missing_keys))
    if problem:
        raise SystemExit("ADAPTER_LOAD_FAILED: " + problem)
    model = pm.merge_and_unload()  # merged only after the strict load proved every tensor applied
    adapter_tensors = len(state)
model.eval()
load_s = time.time() - t0

results = []
t1 = time.time()
for item in batch:
    t = time.time()
    try:
        text = tok.apply_chat_template(
            item["messages"], tools=item.get("tools") or None, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        ids = tok(text, return_tensors="pt")
        with torch.no_grad():
            seqs = model.generate(
                **ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None,
            )
        new = seqs[0][ids["input_ids"].shape[1]:]
        results.append({
            "text": tok.decode(new, skip_special_tokens=False), "error": None,
            "prompt_tokens": int(ids["input_ids"].shape[1]), "new_tokens": int(len(new)),
            "seconds": round(time.time() - t, 2),
        })
    except Exception as exc:
        results.append({"text": "", "error": type(exc).__name__ + ": " + str(exc)[:300],
                        "seconds": round(time.time() - t, 2)})
print("POSTTRAIN_OUT:" + json.dumps({
    "results": results, "load_s": round(load_s, 2), "gen_s": round(time.time() - t1, 2),
    "n": len(batch), "versions": versions, "adapter_tensors": adapter_tensors,
    "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
}))
"""

# The pure helpers are spliced in from their own source so the tested code is the code that runs.
GEN_SCRIPT = "\n\n".join((_GEN_HEAD, inspect.getsource(peft_state_keys),
                          inspect.getsource(adapter_load_problem), _GEN_BODY))


def setup_shell(base_model: str) -> str:
    """Builds the heavy image: CPU-only torch, the HF stack, the base weights (public repo, no
    token), and a syntax check of the script."""
    dl = ("from huggingface_hub import snapshot_download; "
          f"snapshot_download({json.dumps(base_model)}, local_dir={json.dumps(BASE_DIR)})")
    return ("pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu"
            " && pip install --no-cache-dir transformers accelerate peft safetensors huggingface_hub"
            f" && HF_HUB_DISABLE_TELEMETRY=1 python -c {shlex.quote(dl)}"
            f" && python -m py_compile {SCRIPT_PATH}")


def gen_command(max_new_tokens: int, adapter: bool) -> str:
    cmd = f"python {SCRIPT_PATH} --base {BASE_DIR} --max-new-tokens {int(max_new_tokens)}"
    return cmd + (f" --adapter {ADAPTER_DIR}" if adapter else "")


def parse_output(stdout: str) -> dict[str, Any]:
    for line in reversed((stdout or "").splitlines()):
        if line.startswith(OUT_MARKER):
            parsed = json.loads(line[len(OUT_MARKER):])
            if isinstance(parsed, dict):
                return parsed
    raise ValueError(f"no {OUT_MARKER} line in generation output")


# ------------------------------------------------- tool-call parsing (host side)

_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.S | re.I)
_TOOL_CALL = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)


def clean_output(text: str) -> str:
    """Generated text without end tokens (and what follows them) and without think blocks."""
    for end in END_TOKENS:
        text = text.split(end, 1)[0]
    return _THINK.sub("", text).strip()


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Qwen ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` blocks, in order, as
    ``{"name", "arguments"}`` (the serverless path's shape). ``arguments`` is None when it is not
    a JSON object (or JSON string of one); a block without a usable name is skipped."""
    calls = []
    for body in _TOOL_CALL.findall(clean_output(text)):
        body = body.strip()
        try:
            obj, _ = json.JSONDecoder().raw_decode(body)
        except ValueError:
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            continue
        args = obj.get("arguments", obj.get("parameters"))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = None
        calls.append({"name": obj["name"], "arguments": args if isinstance(args, dict) else None})
    return calls


def first_tool_call(text: str) -> dict[str, Any] | None:
    calls = parse_tool_calls(text)
    return calls[0] if calls else None


# ------------------------------------------------- adapter files


@dataclass(frozen=True)
class PreparedAdapter:
    files: dict[str, bytes]  # file name inside ADAPTER_DIR -> content
    sha256: str
    base_model_name_or_path: str


def prepare_adapter(files: Mapping[str, bytes], base_model: str) -> PreparedAdapter:
    """Validate a downloaded checkpoint before anything is uploaded: ``adapter_config.json`` must
    name ``base_model`` and exactly one weights file must exist (renamed to what peft loads).
    Only these two files are shipped."""
    listing = ", ".join(sorted(files)) or "(none)"
    if ADAPTER_CONFIG not in files:
        raise StudentError(f"no {ADAPTER_CONFIG} among the checkpoint files: {listing}",
                           stage="adapter")
    try:
        cfg = json.loads(files[ADAPTER_CONFIG])
    except ValueError as exc:
        raise StudentError(f"{ADAPTER_CONFIG} is not valid JSON ({exc})", stage="adapter") from exc
    declared = cfg.get("base_model_name_or_path") if isinstance(cfg, dict) else None
    if declared != base_model:
        raise StudentError(f"adapter base_model_name_or_path is {declared!r}, expected "
                           f"{base_model!r}", stage="adapter")
    weights = next(((n, files[n]) for n in ADAPTER_WEIGHTS if n in files), None)
    if weights is None:
        safes = [n for n in files if n.endswith(".safetensors")]
        bins = [n for n in files if n.endswith(".bin")]
        if len(safes) == 1:
            weights = (ADAPTER_WEIGHTS[0], files[safes[0]])
        elif not safes and len(bins) == 1:
            weights = (ADAPTER_WEIGHTS[1], files[bins[0]])
        else:
            raise StudentError(f"no single adapter weights file among: {listing}", stage="adapter")
    chosen = {ADAPTER_CONFIG: files[ADAPTER_CONFIG], weights[0]: weights[1]}
    h = hashlib.sha256()
    for name in sorted(chosen):
        h.update(f"{name}:{hashlib.sha256(chosen[name]).hexdigest()}\n".encode())
    return PreparedAdapter(chosen, h.hexdigest(), str(declared))


# ------------------------------------------------- the server


def _tail(text: Any, n: int = 400) -> str:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return str(text or "")[-n:]


class SandboxStudent:
    """Generates completions for ``[{messages, tools}]`` with the base model or one adapter.

    Images are cached on the instance (one heavy build per process, one layer per adapter), so a
    backend that keeps one student pays the build once for benchmark, heldout and regression.
    Any failed job fails the whole call with a ``StudentError`` carrying partial counts; no
    output is ever invented.
    """

    def __init__(self, settings: Settings, client_factory: Callable[[], Any] | None = None, *,
                 batch_size: int = BATCH_SIZE, concurrency: int = CONCURRENCY,
                 max_new_tokens: int = MAX_NEW_TOKENS,
                 job_base_timeout_s: float = JOB_BASE_TIMEOUT_S,
                 job_per_sample_timeout_s: float = JOB_PER_SAMPLE_TIMEOUT_S,
                 total_timeout_s: float = TOTAL_TIMEOUT_S) -> None:
        if batch_size < 1 or concurrency < 1:
            raise ValueError("batch_size and concurrency must be >= 1")
        self.s = settings
        self._factory = client_factory
        self.batch_size = batch_size
        self.concurrency = min(concurrency, CONCURRENCY)
        self.max_new_tokens = max(1, min(int(max_new_tokens), MAX_NEW_TOKENS))
        self._job_base_s, self._job_per_s = job_base_timeout_s, job_per_sample_timeout_s
        self._total_s = total_timeout_s
        self._lock = threading.Lock()  # concurrent evaluations must not build the image twice
        self._slots = threading.BoundedSemaphore(self.concurrency)  # jobs in flight, all calls
        self._deps: str | None = None
        self._adapters: dict[str, str] = {}
        self.timings: list[dict[str, Any]] = []  # image builds and per-batch load/gen seconds

    def _client(self) -> Any:
        if self._factory:
            return self._factory()
        return contree_client(self.s, transport_timeout=TRANSPORT_TIMEOUT_S,
                              operation_timeout=SETUP_TIMEOUT_S + 300)

    # -- images ---------------------------------------------------------------------
    async def _branch(self, image_ref: str, shell: str, files: dict[str, bytes], timeout: float,
                      strict: bool, what: str) -> str:
        client = self._client()
        image = await client.images.use(image_ref, strict=strict)
        t = time.monotonic()
        try:
            res = await asyncio.wait_for(
                image.run(shell=shell, files=files, disposable=False, timeout=timeout,
                          truncate_output_at=TRUNCATE_OUTPUT_AT), timeout=timeout + 60)
        except Exception as exc:  # noqa: BLE001 - any SDK/network failure or our backstop
            raise StudentError(f"{what} build failed: {type(exc).__name__}: {str(exc)[:200]}",
                               stage="image") from exc
        if res.exit_code != 0 or not getattr(res, "uuid", None):
            raise StudentError(f"{what} build failed: exit {res.exit_code}: {_tail(res.stderr)}",
                               stage="image")
        self.timings.append({"build": what, "seconds": round(time.monotonic() - t, 1)})
        return str(res.uuid)

    def image_for(self, adapter: PreparedAdapter | None) -> str:
        """Image uuid holding deps (+ the adapter), built on first use."""
        with self._lock:
            if self._deps is None:
                self._deps = _run_sync(lambda: self._branch(
                    self.s.sandbox_image, setup_shell(self.s.base_model),
                    {SCRIPT_PATH: GEN_SCRIPT.encode()}, SETUP_TIMEOUT_S, True, "serving image"))
            if adapter is None:
                return self._deps
            if adapter.sha256 not in self._adapters:
                deps = self._deps
                self._adapters[adapter.sha256] = _run_sync(lambda: self._branch(
                    deps, "true", {f"{ADAPTER_DIR}/{n}": b for n, b in adapter.files.items()},
                    ADAPTER_TIMEOUT_S, False, "adapter layer"))
            return self._adapters[adapter.sha256]

    # -- generation -------------------------------------------------------------------
    def job_timeout(self, n: int) -> float:
        return self._job_base_s + self._job_per_s * n

    def generate(self, items: Sequence[Mapping[str, Any]],
                 adapter: PreparedAdapter | None = None) -> list[dict[str, Any]]:
        """One output ``{text, new_tokens, seconds}`` per item, in order; raises StudentError."""
        if not items:
            return []
        try:
            image = self.image_for(adapter)
        except SandboxUnavailable as exc:
            raise StudentError(str(exc), total=len(items), stage="image") from exc
        payload = [{"messages": it["messages"], "tools": it.get("tools") or None} for it in items]
        return _run_sync(lambda: self._generate(image, payload, adapter is not None))

    async def _generate(self, image_uuid: str, items: list[dict[str, Any]],
                        has_adapter: bool) -> list[dict[str, Any]]:
        client = self._client()
        cmd = gen_command(self.max_new_tokens, has_adapter)
        chunks = [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]

        async def one(chunk: list[dict[str, Any]]) -> dict[str, Any]:
            # The bound is shared by every generate() call on this student (each call runs in
            # its own event loop/thread), so parallel evaluations together stay within it.
            # Polling keeps cancellation (total timeout) from leaking a slot.
            while not self._slots.acquire(blocking=False):
                await asyncio.sleep(0.05)
            try:
                return await run_job(chunk)
            finally:
                self._slots.release()

        async def run_job(chunk: list[dict[str, Any]]) -> dict[str, Any]:
            timeout = self.job_timeout(len(chunk))
            try:
                image = await client.images.use(image_uuid)
                res = await asyncio.wait_for(
                    image.run(shell=cmd, stdin=json.dumps({"items": chunk}), disposable=True,
                              timeout=timeout, truncate_output_at=TRUNCATE_OUTPUT_AT),
                    timeout=timeout + 30)
            except Exception as exc:  # noqa: BLE001 - timeouts, SDK and network failures
                return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            if res.exit_code != 0:
                return {"error": f"exit {res.exit_code}: {_tail(res.stderr)}"}
            try:
                out = parse_output(_tail(res.stdout, TRUNCATE_OUTPUT_AT))
                results = out["results"]
            except (ValueError, KeyError, TypeError) as exc:
                return {"error": f"unreadable generation output: {exc}"}
            if not isinstance(results, list) or len(results) != len(chunk):
                return {"error": "generation returned the wrong number of outputs"}
            return {"results": results,
                    "stats": {k: out.get(k) for k in ("load_s", "gen_s", "n", "peak_rss_mb",
                                                      "versions", "adapter_tensors")}}

        try:
            parts = await asyncio.wait_for(asyncio.gather(*(one(c) for c in chunks)),
                                           timeout=self._total_s)
        except TimeoutError as exc:
            raise StudentError(f"generation exceeded {self._total_s:.0f}s", total=len(items)) from exc
        outputs: list[dict[str, Any]] = []
        errors = [p["error"] for p in parts if "error" in p]
        done = sum(len(p["results"]) for p in parts if "results" in p)
        for p in parts:
            if "stats" in p:
                self.timings.append(p["stats"])
        if errors:
            raise StudentError(f"{len(errors)} of {len(chunks)} generation batches failed; first: "
                               f"{errors[0]}", done=done, total=len(items))
        for p in parts:
            outputs.extend(p["results"])
        bad = [o for o in outputs if not isinstance(o, dict) or o.get("error")]
        if bad:
            first = bad[0].get("error") if isinstance(bad[0], dict) else "malformed result"
            raise StudentError(f"{len(bad)} of {len(outputs)} samples failed inside the sandbox; "
                               f"first: {first}", done=len(outputs) - len(bad), total=len(items))
        return outputs
