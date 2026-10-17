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

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from ..config import Settings
from . import game_env
from .sandbox import SandboxJudge, SandboxUnavailable, sandbox_configured
from .sandbox_student import (SERVED_BY as SERVED_BY_SANDBOX, SandboxStudent, StudentError,
                              first_tool_call, prepare_adapter)

MAX_TASKS = 300
TERMINAL = {"succeeded", "failed", "cancelled"}
SERVED_BY_SERVERLESS = "nebius-serverless"
MAX_JOBS_SCANNED = 30  # succeeded jobs searched when resolving a checkpoint name
_STUDENT_LOCK = threading.Lock()


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


class NebiusBackend:
    def __init__(self, settings: Settings,
                 client_factory: Callable[[], httpx.Client] | None = None,
                 sandbox: SandboxJudge | None = None, *,
                 student: SandboxStudent | None = None) -> None:
        self.s = settings
        self._sandbox = sandbox
        self._student = student
        self._make = client_factory or (lambda: httpx.Client(timeout=120))
        self._serving: tuple[str, str | None] | None = None  # (served_by, note), probed once
        self._ckpts: dict[str, dict[str, Any]] = {}  # checkpoint name -> {job_id, result_files}
        self._adapter_files: dict[str, dict[str, bytes]] = {}

    def _get_bytes(self, path: str) -> bytes | dict[str, Any]:
        with self._make() as client:
            resp = client.get(f"{self.s.nebius_base_url.rstrip('/')}{path}",
                              headers={"Authorization": f"Bearer {self.s.nebius_api_key}"})
        if resp.status_code >= 400:
            return {"error": f"nebius HTTP {resp.status_code}", "detail": resp.text[:300]}
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
                                 (SERVED_BY_SANDBOX, f"{self.s.base_model} is not served by "
                                                     "/chat/completions; served on sandbox CPU"))
        return self._serving

    # -- plumbing ---------------------------------------------------------------
    def _req(self, method: str, path: str, **kw: Any) -> dict[str, Any]:
        if not self.s.nebius_api_key:
            return {"error": "NEBIUS_API_KEY is not set"}
        with self._make() as client:
            resp = client.request(method, f"{self.s.nebius_base_url.rstrip('/')}{path}",
                                  headers={"Authorization": f"Bearer {self.s.nebius_api_key}"}, **kw)
        if resp.status_code >= 400:
            return {"error": f"nebius HTTP {resp.status_code}", "detail": resp.text[:300]}
        return resp.json()

    # -- training ---------------------------------------------------------------
    def train(self, config: dict[str, Any], dataset_path: Path, base_model: str) -> dict[str, Any]:
        up = self._req("POST", "/files", data={"purpose": "fine-tune"},
                       files={"file": (dataset_path.name, _provider_jsonl(dataset_path))})
        if "error" in up:
            return {"status": "BLOCKED", "stage": "upload", **up}
        job = self._req("POST", "/fine_tuning/jobs", json={
            "model": base_model, "training_file": up["id"], "seed": self.s.seed,
            "suffix": dataset_path.stem[:64],
            "hyperparameters": {**config, "lora": True},
        })
        if "error" in job:
            return {"status": "BLOCKED", "stage": "create_job", **job}
        return {"evidence": "LIVE", "status": job.get("status", "queued"),
                "provider_job_id": job["id"], "training_file": up["id"], "checkpoint": None}

    def job_status(self, job_id: str) -> dict[str, Any]:
        job = self._req("GET", f"/fine_tuning/jobs/{job_id}")
        if "error" in job:
            return job
        out = {"evidence": "LIVE", "provider_job_id": job_id, "status": job.get("status"),
               "trained_steps": job.get("trained_steps"), "total_steps": job.get("total_steps"),
               "trained_tokens": job.get("trained_tokens"), "error": job.get("error"),
               "checkpoint": None}
        if job.get("status") == "succeeded":
            items = self._index_checkpoints(job_id)
            out["checkpoint"] = items[-1].get("fine_tuned_model_checkpoint") if items else None
        return out

    def _index_checkpoints(self, job_id: str) -> list[dict[str, Any]]:
        """The job's checkpoints; remembers name -> (job, result files) for sandbox serving."""
        ck = self._req("GET", f"/fine_tuning/jobs/{job_id}/checkpoints")
        items = [c for c in ck.get("data", []) if isinstance(c, dict)] if "error" not in ck else []
        for c in items:
            if name := c.get("fine_tuned_model_checkpoint"):
                files = [f.get("id") if isinstance(f, dict) else f
                         for f in c.get("result_files") or []]
                self._ckpts[name] = {"job_id": job_id, "result_files": [f for f in files if f]}
        return items

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

    # -- scoring ----------------------------------------------------------------
    def _score(self, checkpoint: str, split: str) -> dict[str, Any]:
        path = Path(self.s.suite_dir) / f"{split}.jsonl"
        if not path.exists():
            return {"error": f"no task file for split {split!r}"}
        tasks = [json.loads(x) for x in path.read_text().splitlines() if x.strip()][:MAX_TASKS]
        served_by, serve_note = self.serving()
        if served_by == SERVED_BY_SANDBOX:  # phase 1: real inference, on sandbox CPU ...
            inferred = self._infer_sandbox(checkpoint, tasks)
        else:  # ... or on Token Factory serverless
            inferred = self._infer_serverless(checkpoint, tasks)
        if "error" in inferred:
            return {**inferred, "served_by": served_by}
        gots = inferred["gots"]
        verdicts = self._judge(tasks, gots)  # phase 2: execute the calls against the game rules
        if "error" in verdicts:
            return {**verdicts, "served_by": served_by}
        hits, failures = 0, []
        per_game: dict[str, list[int]] = {}  # game -> [hits, n]
        for t, got, v in zip(tasks, gots, verdicts["results"], strict=True):
            game = per_game.setdefault(str(t["id"]).split("-")[0], [0, 0])
            game[1] += 1
            if v["ok"]:
                hits += 1
                game[0] += 1
            else:
                failures.append({"task_id": t["id"], "expected": t["expected"], "got": got,
                                 "reason": v.get("reason", "")})
        out = {"n": len(tasks), "score": round(hits / len(tasks), 4) if tasks else 0.0,
               "by_game": {g: {"score": round(h / m, 4), "n": m} for g, (h, m) in per_game.items()},
               "judged_by": verdicts["judged_by"], "served_by": served_by, "failures": failures}
        if verdicts.get("note"):
            out["judge_note"] = verdicts["note"]
        if serve_note:
            out["serve_note"] = serve_note
        return out

    def _infer_serverless(self, checkpoint: str, tasks: list[dict]) -> dict[str, Any]:
        model = self.s.base_model if checkpoint == "base" else checkpoint
        gots: list[dict | None] = []
        for t in tasks:
            body: dict[str, Any] = {"model": model, "messages": t["messages"], "temperature": 0}
            if t.get("tools"):
                body["tools"] = t["tools"]
            resp = self._req("POST", "/chat/completions", json=body)
            if "error" in resp:
                return {**resp, "stage": "inference", "n_tasks": len(tasks),
                        "n_generated": len(gots)}
            calls = (resp["choices"][0]["message"].get("tool_calls") or [])
            got = None
            if calls:
                fn = calls[0]["function"]
                try:
                    got = {"name": fn["name"], "arguments": json.loads(fn["arguments"])}
                except ValueError:
                    got = {"name": fn["name"], "arguments": None}
            gots.append(got)
        return {"gots": gots}

    def _infer_sandbox(self, checkpoint: str, tasks: list[dict]) -> dict[str, Any]:
        """Generate with the base model (+ the checkpoint's LoRA) on sandbox CPU, then parse the
        first ``<tool_call>``. Fails closed: any serving error is returned, never a score."""
        with _STUDENT_LOCK:  # parallel evaluator calls must share one student (one image build)
            if self._student is None:
                if not sandbox_configured(self.s):
                    return {"error": "sandbox serving needs NEBIUS_PROJECT_ID, NEBIUS_API_KEY and "
                                     "contree-sdk", "stage": "serving"}
                self._student = SandboxStudent(self.s)
        adapter = None
        try:
            if checkpoint not in ("base", self.s.base_model):
                files = self._download_adapter(checkpoint)
                if "error" in files:
                    return files  # type: ignore[return-value]
                adapter = prepare_adapter(files, self.s.base_model)  # type: ignore[arg-type]
            outs = self._student.generate(tasks, adapter)
        except StudentError as exc:
            return {"error": str(exc)[:500], "stage": exc.stage, "n_tasks": len(tasks),
                    "n_generated": exc.done}
        return {"gots": [first_tool_call(o.get("text") or "") for o in outs]}

    def _judge(self, tasks: list[dict], gots: list[dict | None]) -> dict[str, Any]:
        """Verdicts per task. Stateful tasks are judged by outcome (sandbox, else labelled local)."""
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
                    if mode == "sandbox":
                        return {"error": str(exc), "stage": "judge"}
                    note = f"sandbox failed, judged locally: {exc}"
            else:
                note = "sandbox not configured, judged locally"
        return {"judged_by": "local-env", "note": note,
                "results": game_env.judge_batch(items)}

    def benchmark(self, checkpoint: str, suite: str, seed: int) -> dict[str, Any]:
        r = self._score(checkpoint, "train")
        if "error" in r:
            return r
        return {"evidence": "LIVE", "checkpoint": checkpoint, "suite": suite, "split": "train",
                "seed": seed, "score": r["score"], "n": r["n"], "by_game": r["by_game"],
                "judged_by": r["judged_by"], "served_by": r["served_by"], **_note(r),
                "failures": r["failures"][:20]}

    def evaluate(self, checkpoint: str, suite: str, split: str, seed: int) -> dict[str, Any]:
        r = self._score(checkpoint, split)
        if "error" in r:
            return r
        return {"evidence": "LIVE", "checkpoint": checkpoint, "suite": suite, "split": split,
                "seed": seed, "score": r["score"], "n": r["n"],
                "by_game": r["by_game"], "judged_by": r["judged_by"],
                "served_by": r["served_by"], **_note(r)}  # scores only: tasks stay sealed
