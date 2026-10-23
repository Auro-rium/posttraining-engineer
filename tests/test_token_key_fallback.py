"""Nebius's docs name the pretokenized key both token_ids and input_ids: retry once on a key 4xx."""

import httpx

from posttrain_agent.config import Settings
from posttrain_agent.experiments.nebius import NebiusBackend


def backend(tmp_path, handler):
    s = Settings(state_dir=tmp_path, data_dir=tmp_path, suite_dir=tmp_path, nebius_api_key="k",
                 nebius_base_url="https://api.example/v1/")
    return NebiusBackend(s, lambda: httpx.Client(transport=httpx.MockTransport(handler)))


def _train(tmp_path, job_replies):
    ds = tmp_path / "d.pretok.jsonl"
    ds.write_text('{"token_ids": [1, 2], "labels": [-100, 2], "attention_mask": [1, 1]}\n')
    uploads, creates = [], []

    def h(req):
        if req.url.path.endswith("/files"):
            uploads.append(req.content)
            return httpx.Response(200, json={"id": f"file-{len(uploads)}"})
        creates.append(1)
        return job_replies[len(creates) - 1]

    return backend(tmp_path, h).train({}, ds, "Qwen/Qwen3-0.6B"), uploads, creates


def test_retries_once_with_input_ids_when_the_provider_names_the_key(tmp_path):
    bad = httpx.Response(422, json={"detail": "field required: input_ids"})
    ok = httpx.Response(200, json={"id": "ftjob-9", "status": "queued"})
    out, uploads, creates = _train(tmp_path, [bad, ok])
    assert out["provider_job_id"] == "ftjob-9" and len(creates) == 2
    assert b'"token_ids"' in uploads[0] and b'"input_ids"' in uploads[1]
    assert b'"token_ids"' not in uploads[1]


def test_unrelated_rejection_is_not_retried(tmp_path):
    out, uploads, creates = _train(tmp_path, [httpx.Response(400, json={"detail": "bad lr"})])
    assert out["status"] == "BLOCKED" and out["stage"] == "create_job" and len(creates) == 1


def test_a_second_key_rejection_stops_the_loop(tmp_path):
    key = httpx.Response(422, json={"detail": "bad input_ids"})
    out, uploads, creates = _train(tmp_path, [key, key])
    assert out["status"] == "BLOCKED" and len(creates) == 2 and len(uploads) == 2
