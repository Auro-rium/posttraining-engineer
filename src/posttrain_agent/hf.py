"""Hugging Face Hub lookup tools (read-only metadata)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from urllib.parse import quote

import httpx

from .config import Settings

HUB = "https://huggingface.co/api"


def hf_tools(
    settings: Settings, client_factory: Callable[[], httpx.AsyncClient] | None = None
) -> dict[str, Callable[..., Any]]:
    make = client_factory or (lambda: httpx.AsyncClient(timeout=30))
    headers = {"Authorization": f"Bearer {settings.hf_token}"} if settings.hf_token else {}

    async def _get(path: str, params: dict[str, Any] | None = None) -> Any:
        async with make() as client:
            resp = await client.get(f"{HUB}{path}", params=params, headers=headers)
        if resp.status_code == 404:
            return {"error": "not found (or gated/private)"}
        if resp.status_code >= 400:
            return {"error": f"huggingface HTTP {resp.status_code}"}
        return resp.json()

    def _slim(d: dict[str, Any]) -> dict[str, Any]:
        card = d.get("cardData") or {}
        return {"id": d.get("id"), "downloads": d.get("downloads"), "likes": d.get("likes"),
                "pipeline_tag": d.get("pipeline_tag"), "library": d.get("library_name"),
                "license": card.get("license"), "gated": d.get("gated"),
                "last_modified": d.get("lastModified"), "revision": d.get("sha"),
                "tags": (d.get("tags") or [])[:15],
                "parameters": (d.get("safetensors") or {}).get("total")}

    async def hf_search(query: str, kind: str = "models", limit: int = 5) -> dict[str, Any]:
        """Search the Hugging Face Hub ('models' or 'datasets'), most downloaded first."""
        if kind not in {"models", "datasets"}:
            return {"error": "kind must be 'models' or 'datasets'"}
        data = await _get(f"/{kind}", {"search": query, "limit": max(1, min(limit, 20)),
                                        "sort": "downloads", "direction": -1})
        if isinstance(data, dict):
            return data
        return {"source": "huggingface", "evidence": "EXPLANATION", "results": [_slim(d) for d in data]}

    async def hf_model_info(repo_id: str) -> dict[str, Any]:
        """Look up a model repo: license, size, revision (pin it!), gating, tags."""
        data = await _get(f"/models/{quote(repo_id, safe='/')}")
        return data if "error" in data else {"source": "huggingface", "evidence": "EXPLANATION",
                                             **_slim(data)}

    async def hf_dataset_info(repo_id: str) -> dict[str, Any]:
        """Look up a dataset repo: license, size, revision, gating, tags."""
        data = await _get(f"/datasets/{quote(repo_id, safe='/')}")
        if "error" in data:
            return data
        info = {"source": "huggingface", "evidence": "EXPLANATION", **_slim(data)}
        info["description"] = ((data.get("cardData") or {}).get("pretty_name")
                               or (data.get("description") or "")[:300])
        return info

    return {"hf_search": hf_search, "hf_model_info": hf_model_info,
            "hf_dataset_info": hf_dataset_info}
