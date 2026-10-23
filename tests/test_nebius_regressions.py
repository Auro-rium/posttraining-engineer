"""Regressions found in the live 0.6B run."""

import httpx

from posttrain_agent.config import Settings
from posttrain_agent.experiments.nebius import NebiusBackend

JOB = {"id": "ftjob-1", "status": "succeeded", "error": None, "trained_tokens": 9}
CK = {"data": [{"fine_tuned_model_checkpoint": "ft:m:ckpt-step-9", "result_files": []}]}


def backend(tmp_path, handler):
    s = Settings(state_dir=tmp_path, data_dir=tmp_path, suite_dir=tmp_path, nebius_api_key="k",
                 nebius_base_url="https://api.example/v1/")
    return NebiusBackend(s, lambda: httpx.Client(transport=httpx.MockTransport(handler)))


def test_job_status_with_null_error_still_returns_the_checkpoint(tmp_path):
    def h(req):  # Nebius sends "error": null on healthy jobs
        return httpx.Response(200, json=CK if req.url.path.endswith("/checkpoints") else JOB)

    out = backend(tmp_path, h).job_status("ftjob-1")
    assert out["status"] == "succeeded" and out["checkpoint"] == "ft:m:ckpt-step-9"


def test_get_bytes_follows_redirects_and_rejects_empty_bodies(tmp_path):
    def h(req):
        if req.url.path == "/v1/files/f1/content":
            return httpx.Response(302, headers={"location": "https://store.example/blob"})
        if req.url.host == "store.example":
            assert "authorization" not in req.headers  # the key is not sent to the storage host
            return httpx.Response(200, content=b"weights")
        return httpx.Response(200, content=b"")

    b = backend(tmp_path, h)
    assert b._get_bytes("/files/f1/content") == b"weights"
    assert "empty body" in b._get_bytes("/files/f2/content")["error"]
