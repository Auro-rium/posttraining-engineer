"""Tavily web tools (search + page extract) for research-type roles."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx

from .config import Settings

TAVILY = "https://api.tavily.com"
_SEALED_MARKERS = ("heldout://", "sealed://")


def web_tools(
    settings: Settings, client_factory: Callable[[], httpx.AsyncClient] | None = None
) -> dict[str, Callable[..., Any]]:
    """Return async tools; without ``TAVILY_API_KEY`` they report 'unavailable' instead of failing."""
    make = client_factory or (lambda: httpx.AsyncClient(timeout=30))

    async def _post(path: str, body: dict[str, Any]) -> dict[str, Any]:
        if not settings.tavily_api_key:
            return {"error": "web tools unavailable: TAVILY_API_KEY is not set"}
        async with make() as client:
            resp = await client.post(
                f"{TAVILY}{path}", json=body,
                headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
            )
        if resp.status_code >= 400:
            return {"error": f"tavily HTTP {resp.status_code}"}
        return resp.json()

    async def web_search(query: str, max_results: int = 5, topic: str = "general") -> dict[str, Any]:
        """Search the web (Tavily) for papers, docs or prior art. Never include held-out data."""
        if any(m in query for m in _SEALED_MARKERS):
            return {"error": "queries must not contain sealed/held-out references"}
        data = await _post("/search", {
            "query": query, "max_results": max(1, min(max_results, 10)),
            "topic": topic if topic in {"general", "news"} else "general",
            "include_answer": True,
        })
        if "error" in data:
            return data
        return {"evidence": "EXPLANATION", "answer": data.get("answer"),
                "results": [{"title": r.get("title"), "url": r.get("url"),
                             "snippet": (r.get("content") or "")[:600]}
                            for r in data.get("results", [])]}

    async def web_extract(urls: list[str]) -> dict[str, Any]:
        """Fetch the cleaned text of up to 5 URLs (Tavily extract), truncated per page."""
        data = await _post("/extract", {"urls": urls[:5]})
        if "error" in data:
            return data
        return {"evidence": "EXPLANATION",
                "pages": [{"url": r.get("url"), "text": (r.get("raw_content") or "")[:4000]}
                          for r in data.get("results", [])],
                "failed": data.get("failed_results", [])}

    return {"web_search": web_search, "web_extract": web_extract}
