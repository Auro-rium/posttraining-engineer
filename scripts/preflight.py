"""Cheap live pre-flight before a real run: does each configured service accept our settings?

Reads ./.env itself (never prints a key). Costs a few tokens: one trivial call per distinct
Backboard model, a model-list call to Nebius, and (if configured) one sandbox judge call.

    python scripts/preflight.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import httpx

from posttrain_agent.config import Settings
from posttrain_agent.experiments.sandbox import SandboxJudge, sandbox_configured
from posttrain_agent.llm.memory import ensure_assistant
from posttrain_agent.llm.resilience import build_model


def load_env(path: Path = Path(".env")) -> None:
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, _, v = line.partition("=")
            if v.strip():
                os.environ[k.strip()] = v.strip()


async def backboard(s: Settings) -> None:
    s_with_assistant = Settings(**{**s.__dict__, "assistant_id": await ensure_assistant(s)})
    pairs = {(s.llm_provider, s.model_name)} | set(s.role_models.values()) | set(s.fallback_models)
    for provider, model in sorted(pairs):
        llm = build_model(s_with_assistant, provider=provider, model=model)
        try:
            out = await llm.ainvoke("Reply with the single word OK.")
            print(f"backboard {provider}:{model}: OK ({str(out.content)[:30]!r})")
        except Exception as exc:  # noqa: BLE001 - report every failure
            print(f"backboard {provider}:{model}: FAIL {type(exc).__name__}: {str(exc)[:160]}")


def nebius(s: Settings) -> None:
    r = httpx.get(f"{s.nebius_base_url.rstrip('/')}/models",
                  headers={"Authorization": f"Bearer {s.nebius_api_key}"}, timeout=30)
    ids = [m.get("id", "") for m in r.json().get("data", [])] if r.status_code == 200 else []
    print(f"nebius /models: HTTP {r.status_code}, {len(ids)} models; "
          f"base model {s.base_model!r} listed: {s.base_model in ids}")
    qwen = [i for i in ids if "qwen3" in i.lower() and ("0.6" in i or "1.7" in i)]
    print("nebius small qwen3 ids:", qwen[:6])


def sandbox(s: Settings) -> None:
    if not sandbox_configured(s):
        print("sandbox: NOT configured (need NEBIUS_PROJECT_ID, NEBIUS_API_KEY and contree-sdk)")
        return
    item = {"id": "t", "state": "nim:18", "name": "take_stones", "arguments": {"count": 2}}
    try:
        print("sandbox verdict:", SandboxJudge(s).judge([item]))
    except Exception as exc:  # noqa: BLE001
        print(f"sandbox: FAIL {type(exc).__name__}: {str(exc)[:200]}")


def main() -> None:
    load_env()
    s = Settings.from_env()
    asyncio.run(backboard(s))
    nebius(s)
    sandbox(s)


if __name__ == "__main__":
    main()
