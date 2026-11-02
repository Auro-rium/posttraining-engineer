"""Nebius Token Factory backend: fine-tuning jobs plus deterministic task evaluation.

Uses the OpenAI-compatible REST API (https://api.tokenfactory.nebius.com/v1):
files (purpose=fine-tune), fine_tuning/jobs (+ /events, /checkpoints) for training.
Scoring serves the model one of two ways, decided once per backend (``POSTTRAIN_SERVE``):
``nebius-serverless`` (``/chat/completions``, when ``/models`` lists the base model) or
``nebius-sandbox-cpu`` (``sandbox_student``: the base model, plus the checkpoint's LoRA files
downloaded via ``/files/{id}/content``, run with transformers on Nebius Sandbox CPU). Every
result carries ``served_by``; the promotion gate never compares two serving paths.

Evaluation is deterministic: each task in ``<suite_dir>/<split>.jsonl`` is
``{"id", "messages", "tools"?, "expected": {"name", "arguments"}}`` and scores 1 when
the model's first tool call matches name and arguments exactly. Held-out tasks are
read only here; callers receive scores, never task contents.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from ..config import Settings
from ..observability.events import CURRENT_RUN
from . import game_env
from .retry import TransientError, is_transient, retry_sync
from .sandbox import SandboxJudge, SandboxUnavailable, sandbox_configured
from .sandbox_student import SERVED_BY as SERVED_BY_SANDBOX
from .sandbox_student import SandboxStudent, StudentError, first_tool_call, prepare_adapter

MAX_TASKS = 1000  # held-out n>=300 and regression n>=150 must fit; suites set the real size
PREFLIGHT_N = 12  # tasks generated first; the full eval only runs if enough of them parse
PREFLIGHT_MIN_PARSE = 0.5  # lenient on purpose: the gate enforces the strict parse-rate rule
FAILURES_SHOWN = 10  # failure records returned by evaluate()
TERMINAL = {"succeeded", "failed", "cancelled"}
SERVED_BY_SERVERLESS = "nebius-serverless"
MAX_JOBS_SCANNED = 30  # succeeded jobs searched when resolving a checkpoint name
CHECKPOINT_TRIES = 5  # a succeeded job's checkpoint list can lag (seen live: empty, then 3)
CHECKPOINT_BASE_DELAY_S = 1.0
CHECKPOINT_MAX_DELAY_S = 8.0
_STUDENT_LOCK = threading.Lock()


_STEP = re.compile(r"step[-_]?(\d+)", re.IGNORECASE)


def checkpoint_step(c: dict[str, Any]) -> int:
    """Training step of a checkpoint record: its ``step_number`` field, else the ``step-N`` in its
    name; -1 when neither is present."""
    n = c.get("step_number")
    if isinstance(n, int) and not isinstance(n, bool):
        return n
    m = _STEP.search(str(c.get("fine_tuned_model_checkpoint") or ""))
    return int(m.group(1)) if m else -1


def final_checkpoint(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The checkpoint with the highest step (ties: newest ``created_at``, then last listed);
    never simply the last list element, because the provider does not promise an order."""
    best = max(enumerate(items), default=None, key=lambda ic: (
        checkpoint_step(ic[1]), ic[1].get("created_at") or 0, ic[0]))
    return best[1] if best else None


def _note(r: dict[str, Any]) -> dict[str, Any]:
    return {k: r[k] for k in ("judge_note", "serve_note") if r.get(k)}


def _provider_jsonl(path: Path) -> bytes:
    """The dataset as the provider's format allows: our private ``metadata`` key is not sent."""
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                row.pop("metadata", None)
            out.append(json.dumps(row, sort_keys=True))
    return ("\n".join(out) + "\n").encode()


def _swap_token_key(data: bytes, failure: dict[str, Any]) -> bytes | None:
    """The same pretokenized file with ``token_ids`` <-> ``input_ids`` renamed, or None when the
    failure is not a 4xx that names the key (or the file is not pretokenized)."""
    text = f"{failure.get('error', '')} {failure.get('detail', '')}"
    if "HTTP 4" not in text or not ("token_ids" in text or "input_ids" in text):
        return None
    rows = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    if not rows or not all(isinstance(r, dict) for r in rows):
        return None
    old, new = ("token_ids", "input_ids") if "token_ids" in rows[0] else ("input_ids", "token_ids")
    if old not in rows[0]:
        return None
    out = [json.dumps({(new if k == old else k): v for k, v in r.items()}, sort_keys=True)
           for r in rows]
    return ("\n".join(out) + "\n").encode()


class NebiusBackend:
    def __init__(self, settings: Settings,
                 client_factory: Callable[[], httpx.Client] | None = None,
                 sandbox: SandboxJudge | None = None, *,
                 student: SandboxStudent | None = None,
                 gen_dir: Path | str | None = None) -> None:
        self.s = settings
        self._sleep = time.sleep  # tests replace it so retries do not wait
        self._gen_dir = Path(gen_dir) if gen_dir else None
        self._sandbox = sandbox
        self._student = student
        self._make = client_factory or (lambda: httpx.Client(timeout=120))
        self._serving: tuple[str, str | None] | None = None  # (served_by, note), probed once
        self._ckpts: dict[str, dict[str, Any]] = {}  # checkpoint name -> {job_id, result_files}
        self._adapter_files: dict[str, dict[str, bytes]] = {}

    @staticmethod
    def _transient_error(exc: BaseException, what: str) -> dict[str, Any]:
        return {"error": f"{what} failed after retries: {type(exc).__name__}: {str(exc)[:200]}",
                "stage": "request", "transient": True}

    def _get_bytes(self, path: str) -> bytes | dict[str, Any]:
        def once() -> httpx.Response:
            with self._make() as client:
                # large files are served through a redirect; httpx drops the key on other hosts
                r = client.get(f"{self.s.nebius_base_url.rstrip('/')}{path}",
                               headers={"Authorization": f"Bearer {self.s.nebius_api_key}"},
                               follow_redirects=True)
            if r.status_code >= 500 or r.status_code == 429:
                raise TransientError(f"nebius HTTP {r.status_code}")
            return r

        try:
            resp = retry_sync(once, sleep=self._sleep)
        except Exception as exc:  # only transient failures become structured errors
            if not is_transient(exc):
                raise
            return self._transient_error(exc, f"GET {path}")
        if resp.status_code >= 400:
            return {"error": f"nebius HTTP {resp.status_code}", "detail": resp.text[:300]}
        if not resp.content:
            return {"error": f"nebius returned an empty body for {path}"}
        return resp.content

    # -- serving ------------------------------------------------------------------
    def serving(self) -> tuple[str, str | None]:
        """``(served_by, note)`` for this backend's base model and all of its checkpoints.

        One decision per backend (``POSTTRAIN_SERVE``; ``auto`` asks ``/models`` once), so the
        base model and every checkpoint are always served the same way.
        """
        if self._serving is None:
            mode = self.s.serve
            if mode == "sandbox":
                self._serving = (SERVED_BY_SANDBOX, None)
            elif mode == "serverless":
                self._serving = (SERVED_BY_SERVERLESS, None)
            else:
                try:
                    listed = self._req("GET", "/models")
                except Exception as exc:  # noqa: BLE001 - keep the previous behaviour on any failure
                    listed = {"error": type(exc).__name__}
                ids = {m.get("id") for m in listed.get("data", []) if isinstance(m, dict)} \
                    if isinstance(listed, dict) and "error" not in listed else None
                if ids is None:  # not cached: the next call asks again
                    return (SERVED_BY_SERVERLESS,
                            f"/models probe failed ({listed.get('error')}); assumed serverless")
                self._serving = ((SERVED_BY_SERVERLESS, None) if self.s.base_model in ids else
                                 (SERVED_BY_SANDBOX,
                                  (f"{self.s.base_model} is not served by "
                                   "/chat/completions; served on sandbox CPU")))
        return self._serving

    # -- plumbing ---------------------------------------------------------------
    def _req(self, method: str, path: str, *, tries: int | None = None,
             **kw: Any) -> dict[str, Any]:
        """One API call, retried with backoff on transport/DNS errors, 5xx and 429. After the last
        try the result is ``{error, stage, transient: True}``, never a partial value."""
        if not self.s.nebius_api_key:
            return {"error": "NEBIUS_API_KEY is not set"}

        def once() -> httpx.Response:
            with self._make() as client:
                r = client.request(method, f"{self.s.nebius_base_url.rstrip('/')}{path}",
                                   headers={"Authorization": f"Bearer {self.s.nebius_api_key}"},
                                   **kw)
            if r.status_code >= 500 or r.status_code == 429:
                raise TransientError(f"nebius HTTP {r.status_code}: {r.text[:120]}")
            return r

        try:
            resp = retry_sync(once, tries=tries, sleep=self._sleep)
        except Exception as exc:  # only transient failures become structured errors
            if not is_transient(exc):
                raise
            return self._transient_error(exc, f"{method} {path}")
        if resp.status_code >= 400:
            return {"error": f"nebius HTTP {resp.status_code}", "detail": resp.text[:300]}
        return resp.json()

    # -- training ---------------------------------------------------------------
    def train(self, config: dict[str, Any], dataset_path: Path, base_model: str) -> dict[str, Any]:
        """Upload then create the job. Nebius's docs name the pretokenized key both ``token_ids``
        and ``input_ids``; if the provider rejects the file or job with a 4xx naming the key,
        re-upload ONCE with the other name. A rejected creation made no job, so no double launch."""
        data = _provider_jsonl(dataset_path)
        for attempt in (1, 2):
            up = self._req("POST", "/files", data={"purpose": "fine-tune"},
                           files={"file": (dataset_path.name, data)})
            job = up if "error" in up else self._req("POST", "/fine_tuning/jobs", json={
                "model": base_model, "training_file": up["id"], "seed": self.s.seed,
                "suffix": dataset_path.stem[:64],
                "hyperparameters": {**config, "lora": True},
            }, tries=1)  # never auto-retry job creation: a lost response could double-launch
            if not (job.get("error") and "id" not in job):
                break
            swapped = _swap_token_key(data, job)
            if attempt == 2 or swapped is None:
                stage = "upload" if "error" in up else "create_job"
                return {**job, "status": "BLOCKED", "stage": stage}
            data = swapped
        return {"evidence": "LIVE", "status": job.get("status", "queued"),
                "provider_job_id": job["id"], "training_file": up["id"], "checkpoint": None}

    def job_status(self, job_id: str) -> dict[str, Any]:
        job = self._req("GET", f"/fine_tuning/jobs/{job_id}")
        if job.get("error") and "status" not in job:
            return job
        out = {"evidence": "LIVE", "provider_job_id": job_id, "status": job.get("status"),
               "trained_steps": job.get("trained_steps"), "total_steps": job.get("total_steps"),
               "trained_tokens": job.get("trained_tokens"), "error": job.get("error"),
               "checkpoint": None}
        if job.get("status") == "succeeded":
            last, missing = self._final_checkpoint(job_id)  # highest step, not list order
            if last:
                out["checkpoint"] = last.get("fine_tuned_model_checkpoint")
            else:
                out.update(missing or {})
        return out

    def _final_checkpoint(self, job_id: str
                          ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """The highest-step checkpoint of a succeeded job, retrying a briefly empty listing.

        Checkpoint listings can lag the terminal job status (seen live: a succeeded job returned
        an empty list, then three checkpoints on a later request). Only the idempotent listing is
        retried, with bounded exponential backoff; the training job is never created again.
        """
        detail = "empty checkpoint list"
        for attempt in range(CHECKPOINT_TRIES):
            items, err = self._checkpoint_items(job_id)
            last = final_checkpoint(items)
            if last and last.get("fine_tuned_model_checkpoint"):
                return last, None
            if err:
                detail = str(err.get("error") or err)[:300]
            if attempt + 1 < CHECKPOINT_TRIES:
                self._sleep(min(CHECKPOINT_MAX_DELAY_S,
                                CHECKPOINT_BASE_DELAY_S * 2 ** attempt))
        return None, {
            "error": f"job {job_id} reports succeeded but no checkpoint was listed "
                     f"after {CHECKPOINT_TRIES} attempts ({detail}); refusing to evaluate "
                     "without a checkpoint",
            "stage": "checkpoint", "checkpoint_missing": True, "transient": True,
            "attempts": CHECKPOINT_TRIES,
        }

    def _index_checkpoints(self, job_id: str) -> list[dict[str, Any]]:
        """The job's checkpoints from one listing; remembers name -> (job, result files) for
        sandbox serving. See ``_final_checkpoint`` for the retrying succeeded-job path."""
        items, _err = self._checkpoint_items(job_id)
        return items

    def _checkpoint_items(self, job_id: str
                          ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """One checkpoint listing and its structured error (if the request failed)."""
        ck = self._req("GET", f"/fine_tuning/jobs/{job_id}/checkpoints")
        if "error" in ck:
            return [], ck
        items = [c for c in (ck.get("data") or []) if isinstance(c, dict)]
        for c in items:
            if name := c.get("fine_tuned_model_checkpoint"):
                files = [f.get("id") if isinstance(f, dict) else f
                         for f in c.get("result_files") or []]
                self._ckpts[name] = {"job_id": job_id, "result_files": [f for f in files if f]}
        return items, None

    def _resolve_checkpoint(self, checkpoint: str) -> dict[str, Any]:
        """Find the job and result files of a checkpoint name (exact match, never by position)."""
        if checkpoint in self._ckpts:
            return self._ckpts[checkpoint]
        jobs = self._req("GET", "/fine_tuning/jobs", params={"limit": 100})
        if "error" in jobs:
            return {**jobs, "stage": "checkpoint"}
        done = [j for j in jobs.get("data", []) if isinstance(j, dict)
                and j.get("status") == "succeeded" and j.get("id")]
        for job in done[:MAX_JOBS_SCANNED]:
            self._index_checkpoints(job["id"])
            if checkpoint in self._ckpts:
                return self._ckpts[checkpoint]
        return {"error": f"checkpoint {checkpoint!r} not found among the last "
                         f"{min(len(done), MAX_JOBS_SCANNED)} succeeded fine-tuning jobs",
                "stage": "checkpoint"}

    def _download_adapter(self, checkpoint: str) -> dict[str, bytes] | dict[str, Any]:
        """Checkpoint files as ``{basename: bytes}`` (host side; the key never leaves it)."""
        if checkpoint in self._adapter_files:
            return self._adapter_files[checkpoint]
        ref = self._resolve_checkpoint(checkpoint)
        if "error" in ref:
            return ref
        files: dict[str, bytes] = {}
        for fid in ref["result_files"]:
            meta = self._req("GET", f"/files/{fid}")
            if "error" in meta:
                return {**meta, "stage": "adapter_download"}
            name = Path(str(meta.get("filename") or "")).name
            if name in ("", ".", ".."):
                return {"error": f"unusable filename for checkpoint file {fid}",
                        "stage": "adapter_download"}
            content = self._get_bytes(f"/files/{fid}/content")
            if isinstance(content, dict):
                return {**content, "stage": "adapter_download"}
            files[name] = content
        self._adapter_files[checkpoint] = files
        return files

    def job_events(self, job_id: str, limit: int = 50) -> dict[str, Any]:
        ev = self._req("GET", f"/fine_tuning/jobs/{job_id}/events", params={"limit": limit})
        if "error" in ev:
            return ev
        return {"evidence": "LIVE", "provider_job_id": job_id,
                "events": [{"created_at": e.get("created_at"), "level": e.get("level"),
                            "message": (e.get("message") or "")[:300]} for e in ev.get("data", [])]}
    # -- generation store (raw outputs survive judge failures and reruns) -------------------
    def _gen_file(self, split: str) -> Path | None:
        """``state/operator_evals/<run>-<split>-gen.jsonl``. Without a run id (and no explicit
        ``gen_dir``) nothing is persisted, so ad-hoc calls never touch ``state/``."""
        run = CURRENT_RUN.get()
        if self._gen_dir is None and not run:
            return None
        base = self._gen_dir or Path(self.s.state_dir) / "operator_evals"
        return base / f"{run or 'adhoc'}-{split}-gen.jsonl"

    @staticmethod
    def _gen_key(checkpoint: str, task: dict) -> str:
        blob = json.dumps([checkpoint, task["id"], task["messages"], task.get("tools")],
                          sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def _load_gens(self, split: str) -> dict[str, dict[str, Any]]:
        path = self._gen_file(split)
        gens: dict[str, dict[str, Any]] = {}
        if path and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue  # a torn last line loses one generation, not the file
                if isinstance(rec, dict) and rec.get("key"):
                    if isinstance(rec.get("text"), str):  # re-parse: the parser may have improved
                        rec["got"] = first_tool_call(rec["text"])
                    gens[rec["key"]] = rec
        return gens

    def _save_gens(self, split: str, new: list[dict[str, Any]]) -> None:
        path = self._gen_file(split)
        if path and new:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.writelines(json.dumps(r, sort_keys=True) + "\n" for r in new)

    def _ensure_gens(self, checkpoint: str, tasks: list[dict], split: str, served_by: str,
                     gens: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
        """Generate what is missing from ``gens`` (saving every completed output at once).
        Returns a structured error, or None when every task has a generation."""
        missing = [t for t in tasks if self._gen_key(checkpoint, t) not in gens]
        if not missing:
            return None
        if served_by == SERVED_BY_SANDBOX:  # phase 1: real inference, on sandbox CPU ...
            entries, err = self._infer_sandbox(checkpoint, missing)
        else:  # ... or on Token Factory serverless
            entries, err = self._infer_serverless(checkpoint, missing)
        new = []
        for t, e in zip(missing, entries, strict=False):
            if e is not None:
                rec = {"key": self._gen_key(checkpoint, t), "checkpoint": checkpoint,
                       "split": split, "task_id": t["id"], **e}
                gens[rec["key"]] = rec
                new.append(rec)
        self._save_gens(split, new)
        if err:
            have = sum(self._gen_key(checkpoint, t) in gens for t in tasks)
            out = {**err, "n_tasks": len(tasks), "n_generated": have, "served_by": served_by}
            if path := self._gen_file(split):
                out["gen_file"] = str(path)
            return out
        return None

    @staticmethod
    def _parsed(got: dict | None) -> bool:
        return isinstance(got, dict) and isinstance(got.get("arguments"), dict) \
            and isinstance(got.get("name"), str)

    # -- scoring ----------------------------------------------------------------
    def _score(self, checkpoint: str, split: str) -> dict[str, Any]:
        path = Path(self.s.suite_dir) / f"{split}.jsonl"
        if not path.exists():
            return {"error": f"no task file for split {split!r}", "stage": "suite"}
        tasks = [json.loads(x) for x in path.read_text().splitlines() if x.strip()][:MAX_TASKS]
        if not tasks:
            return {"error": f"split {split!r} has no tasks", "stage": "suite"}
        served_by, serve_note = self.serving()
        gens = self._load_gens(split)
        # preflight: a small spread of tasks first, so an adapter that never emits a tool call
        # fails here, clearly, instead of as a full-length run of zeros
        step = max(1, len(tasks) // PREFLIGHT_N)
        sample = tasks[::step][:PREFLIGHT_N]
        if err := self._ensure_gens(checkpoint, sample, split, served_by, gens):
            return err
        outs = [gens[self._gen_key(checkpoint, t)] for t in sample]
        pre_rate = sum(self._parsed(o.get("got")) for o in outs) / len(outs)
        if pre_rate < PREFLIGHT_MIN_PARSE:
            return {"error": f"preflight failed: {sum(self._parsed(o.get('got')) for o in outs)} of "
                             f"{len(outs)} sample outputs contained a parseable tool call "
                             f"(parse_rate {pre_rate:.2f} < {PREFLIGHT_MIN_PARSE}); the model does "
                             "not emit tool calls, so no score was computed",
                    "stage": "preflight", "transient": False, "parse_rate": round(pre_rate, 4),
                    "n_preflight": len(outs), "served_by": served_by,
                    "sample_output": [str(o.get("text") or o.get("got"))[:160] for o in outs[:2]]}
        if err := self._ensure_gens(checkpoint, tasks, split, served_by, gens):
            return err
        gots = [gens[self._gen_key(checkpoint, t)].get("got") for t in tasks]
        verdicts = self._judge(tasks, gots)  # phase 2: execute the calls against the game rules
        if "error" in verdicts:
            gen_file = self._gen_file(split)
            return {**verdicts, "served_by": served_by,
                    **({"gen_file": str(gen_file)} if gen_file else {})}
        hits, failures = 0, []
        kinds = {"unparsed": 0, "wrong_tool": 0, "illegal": 0, "wrong_move": 0}
        per_game: dict[str, list[int]] = {}  # game -> [hits, n]
        bits = []
        for t, got, v in zip(tasks, gots, verdicts["results"], strict=True):
            game = per_game.setdefault(str(t["id"]).split("-")[0], [0, 0])
            game[1] += 1
            bits.append("1" if v["ok"] and self._parsed(got) else "0")
            if v["ok"] and self._parsed(got):
                hits += 1
                game[0] += 1
                continue
            kind = self._failure_kind(t, got, v)
            kinds[kind] += 1
            failures.append({"task_id": t["id"], "kind": kind, "expected": t["expected"],
                             "got": got, "reason": v.get("reason", "")})
        n = len(tasks)
        n_parsed = n - kinds["unparsed"]
        ident = hashlib.sha256("\n".join(f"{t['id']}|{t.get('state', '')}" for t in tasks)
                               .encode()).hexdigest()[:16]
        out = {"n": n, "score": round(hits / n, 4), "n_parsed": n_parsed,
               "parse_rate": round(n_parsed / n, 4), "n_unparsed": kinds["unparsed"],
               "n_wrong_tool": kinds["wrong_tool"], "n_illegal": kinds["illegal"],
               "n_wrong_move": kinds["wrong_move"],
               "by_game": {g: {"score": round(h / m, 4), "n": m} for g, (h, m) in per_game.items()},
               "judged_by": verdicts["judged_by"], "served_by": served_by, "failures": failures,
               "verdicts": "".join(bits), "tasks_sha256": ident}
        if verdicts.get("note"):
            out["judge_note"] = verdicts["note"]
        if serve_note:
            out["serve_note"] = serve_note
        return out

    def _failure_kind(self, task: dict, got: dict | None, verdict: dict) -> str:
        """unparsed (no usable tool call) vs wrong_tool vs illegal vs wrong_move."""
        if not self._parsed(got):
            return "unparsed"
        reason = str(verdict.get("reason") or "")
        if reason.startswith("expected tool") or (not reason and got["name"] != task["expected"].get("name")):
            return "wrong_tool"
        if reason.startswith(("illegal", "guess must", "move must")) or "takes exactly one" in reason:
            return "illegal"
        return "wrong_move"

    def _infer_serverless(self, checkpoint: str, tasks: list[dict]
                          ) -> tuple[list[dict | None], dict[str, Any] | None]:
        """``(entries, error)``: one ``{text, got}`` per task (None where not generated)."""
        model = self.s.base_model if checkpoint == "base" else checkpoint
        entries: list[dict | None] = []
        for t in tasks:
            body: dict[str, Any] = {"model": model, "messages": t["messages"], "temperature": 0}
            if t.get("tools"):
                body["tools"] = t["tools"]
            resp = self._req("POST", "/chat/completions", json=body)  # retried inside _req
            if "error" in resp:
                entries.extend([None] * (len(tasks) - len(entries)))
                return entries, {**resp, "stage": "inference"}
            msg = resp["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            got = None
            if calls:
                fn = calls[0]["function"]
                try:
                    got = {"name": fn["name"], "arguments": json.loads(fn["arguments"])}
                except ValueError:
                    got = {"name": fn["name"], "arguments": None}
            entries.append({"text": None, "got": got,
                            "content": str(msg.get("content") or "")[:300]})
        return entries, None

    def _infer_sandbox(self, checkpoint: str, tasks: list[dict]
                       ) -> tuple[list[dict | None], dict[str, Any] | None]:
        """Generate with the base model (+ the checkpoint's LoRA) on sandbox CPU, then parse the
        first ``<tool_call>``. Fails closed: any serving error is returned, never a score."""
        none: list[dict | None] = [None] * len(tasks)
        with _STUDENT_LOCK:  # parallel evaluator calls must share one student (one image build)
            if self._student is None:
                if not sandbox_configured(self.s):
                    return none, {"error": "sandbox serving needs NEBIUS_PROJECT_ID, "
                                           "NEBIUS_API_KEY and contree-sdk", "stage": "serving"}
                self._student = SandboxStudent(self.s)
        adapter = None
        try:
            if checkpoint not in ("base", self.s.base_model):
                files = self._download_adapter(checkpoint)
                if "error" in files:
                    return none, files  # type: ignore[return-value]
                adapter = prepare_adapter(files, self.s.base_model)  # type: ignore[arg-type]
            outs = self._student.generate(tasks, adapter)
        except StudentError as exc:
            part = list(exc.partial) + [None] * (len(tasks) - len(exc.partial))
            entries = [None if o is None else {"text": o.get("text") or "",
                                               "got": first_tool_call(o.get("text") or "")}
                       for o in part]
            return entries, {"error": str(exc)[:500], "stage": exc.stage, "transient": exc.transient}
        return [{"text": o.get("text") or "", "got": first_tool_call(o.get("text") or "")}
                for o in outs], None

    def _judge(self, tasks: list[dict], gots: list[dict | None]) -> dict[str, Any]:
        """Verdicts per task. Stateful tasks are judged by outcome in the sandbox. Fails closed:
        if the sandbox judge is configured (or injected) and fails, that is an error, never a
        silent local re-judge. Local judging happens only when asked for (``local``) or when no
        sandbox is configured at all (``auto``), and is then labelled ``local-env``."""
        if not all("state" in t for t in tasks):  # old-format suites: exact match to `expected`
            return {"judged_by": "exact-match", "results": [
                {"ok": g == t["expected"], "reason": ""} for t, g in zip(tasks, gots, strict=True)]}
        items = [{"id": t["id"], "state": t["state"], "name": (g or {}).get("name"),
                  "arguments": (g or {}).get("arguments")} for t, g in zip(tasks, gots, strict=True)]
        mode, note = self.s.eval_judge, None
        if mode in ("auto", "sandbox"):
            judge = self._sandbox or (SandboxJudge(self.s) if sandbox_configured(self.s) else None)
            if judge is None and mode == "sandbox":
                return {"error": "POSTTRAIN_EVAL_JUDGE=sandbox but the Nebius sandbox is not "
                                 "configured (need NEBIUS_PROJECT_ID and contree-sdk)",
                        "stage": "judge"}
            if judge is not None:
                try:
                    return {"judged_by": "nebius-sandbox", "results": judge.judge(items)}
                except SandboxUnavailable as exc:
                    return {"error": f"sandbox judge failed: {exc}", "stage": "judge",
                            "transient": exc.transient}
            note = "sandbox not configured, judged locally"
        return {"judged_by": "local-env", "note": note,
                "results": game_env.judge_batch(items)}

    def benchmark(self, checkpoint: str, suite: str, seed: int) -> dict[str, Any]:
        r = self._score(checkpoint, "train")
        if "error" in r:
            return r
        return {"evidence": "LIVE", "checkpoint": checkpoint, "suite": suite, "split": "train",
                "seed": seed, "score": r["score"], "n": r["n"], "by_game": r["by_game"],
                "parse_rate": r["parse_rate"], "judged_by": r["judged_by"],
                "served_by": r["served_by"], **_note(r),
                "failures": r["failures"][:20]}  # train is not sealed: expected values included

    def evaluate(self, checkpoint: str, suite: str, split: str, seed: int) -> dict[str, Any]:
        r = self._score(checkpoint, split)
        if "error" in r:
            return r
        # scores and failure KINDS only: tasks stay sealed, so no expected values or reasons
        failures = [{"task_id": f["task_id"], "kind": f["kind"], "got": _short(f["got"])}
                    for f in r["failures"][:FAILURES_SHOWN]]
        return {"evidence": "LIVE", "checkpoint": checkpoint, "suite": suite, "split": split,
                "seed": seed, "score": r["score"], "n": r["n"], "n_parsed": r["n_parsed"],
                "parse_rate": r["parse_rate"], "n_unparsed": r["n_unparsed"],
                "n_wrong_tool": r["n_wrong_tool"], "n_illegal": r["n_illegal"],
                "n_wrong_move": r["n_wrong_move"], "by_game": r["by_game"],
                "judged_by": r["judged_by"], "served_by": r["served_by"],
                "verdicts": r["verdicts"], "tasks_sha256": r["tasks_sha256"],
                "failures": failures, **_note(r)}


def _short(got: Any) -> Any:
    return None if got is None else json.dumps(got, sort_keys=True)[:160]
