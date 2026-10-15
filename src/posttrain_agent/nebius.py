"""Nebius Token Factory backend: fine-tuning jobs plus deterministic task evaluation.

Uses the OpenAI-compatible REST API (https://api.tokenfactory.nebius.com/v1):
files (purpose=fine-tune), fine_tuning/jobs (+ /events, /checkpoints) and
chat/completions for scoring. Request/response fields follow Nebius' published
FineTuningRequest schema. Only unit-tested against a mock transport; no live call
has been made.

Evaluation is deterministic: each task in ``<suite_dir>/<split>.jsonl`` is
``{"id", "messages", "tools"?, "expected": {"name", "arguments"}}`` and scores 1 when
the model's first tool call matches name and arguments exactly. Held-out tasks are
read only here; callers receive scores, never task contents.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from .config import Settings

MAX_TASKS = 200
TERMINAL = {"succeeded", "failed", "cancelled"}


class NebiusBackend:
    def __init__(self, settings: Settings,
                 client_factory: Callable[[], httpx.Client] | None = None) -> None:
        self.s = settings
        self._make = client_factory or (lambda: httpx.Client(timeout=120))

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
                       files={"file": (dataset_path.name, dataset_path.read_bytes())})
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
            ck = self._req("GET", f"/fine_tuning/jobs/{job_id}/checkpoints")
            items = ck.get("data", []) if "error" not in ck else []
            out["checkpoint"] = items[-1].get("fine_tuned_model_checkpoint") if items else None
        return out

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
        model = self.s.base_model if checkpoint == "base" else checkpoint
        hits, failures = 0, []
        for t in tasks:
            body: dict[str, Any] = {"model": model, "messages": t["messages"], "temperature": 0}
            if t.get("tools"):
                body["tools"] = t["tools"]
            resp = self._req("POST", "/chat/completions", json=body)
            if "error" in resp:
                return {**resp, "stage": "inference"}
            calls = (resp["choices"][0]["message"].get("tool_calls") or [])
            got = None
            if calls:
                fn = calls[0]["function"]
                try:
                    got = {"name": fn["name"], "arguments": json.loads(fn["arguments"])}
                except ValueError:
                    got = {"name": fn["name"], "arguments": None}
            if got == t["expected"]:
                hits += 1
            else:
                failures.append({"task_id": t["id"], "expected": t["expected"], "got": got})
        return {"n": len(tasks), "score": round(hits / len(tasks), 4) if tasks else 0.0,
                "failures": failures}

    def benchmark(self, checkpoint: str, suite: str, seed: int) -> dict[str, Any]:
        r = self._score(checkpoint, "train")
        if "error" in r:
            return r
        return {"evidence": "LIVE", "checkpoint": checkpoint, "suite": suite, "split": "train",
                "seed": seed, "score": r["score"], "n": r["n"], "failures": r["failures"][:20]}

    def evaluate(self, checkpoint: str, suite: str, split: str, seed: int) -> dict[str, Any]:
        r = self._score(checkpoint, split)
        if "error" in r:
            return r
        return {"evidence": "LIVE", "checkpoint": checkpoint, "suite": suite, "split": split,
                "seed": seed, "score": r["score"], "n": r["n"]}  # scores only: tasks stay sealed
