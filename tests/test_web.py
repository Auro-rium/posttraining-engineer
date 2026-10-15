import httpx

from posttrain_agent.config import Settings
from posttrain_agent.subagents import ROLES
from posttrain_agent.web import web_tools


def tools(handler, key="k"):
    return web_tools(
        Settings(tavily_api_key=key),
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


async def test_search_maps_results_and_sends_bearer():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"], seen["url"] = req.headers["authorization"], str(req.url)
        return httpx.Response(200, json={"answer": "A", "results": [
            {"title": "T", "url": "http://x", "content": "c" * 900}]})

    out = await tools(handler)["web_search"]("qlora rank")
    assert seen == {"auth": "Bearer k", "url": "https://api.tavily.com/search"}
    assert out["evidence"] == "EXPLANATION" and len(out["results"][0]["snippet"]) == 600


async def test_missing_key_and_http_error_and_sealed_query_fail_soft():
    boom = lambda r: httpx.Response(500)
    assert "TAVILY_API_KEY" in (await tools(boom, key="")["web_search"]("x"))["error"]
    assert "500" in (await tools(boom)["web_search"]("x"))["error"]
    assert "sealed" in (await tools(boom)["web_search"]("see heldout://t1"))["error"]


async def test_extract_truncates():
    h = lambda r: httpx.Response(200, json={"results": [{"url": "u", "raw_content": "z" * 9000}]})
    assert len((await tools(h)["web_extract"](["u"]))["pages"][0]["text"]) == 4000


def test_only_research_roles_get_web_tools():
    with_web = {n for n, (_, _, t, _) in ROLES.items() if "web_search" in t}
    assert with_web == {"researcher", "training_designer"}
