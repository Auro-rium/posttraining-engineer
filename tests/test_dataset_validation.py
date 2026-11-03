"""One call to train() makes at most one upload and one job; dataset rejections are reported, never
retried. The pretokenized wire key is PRETOK_TOKEN_KEY (input_ids, proven live), renamed at upload only."""

import json

import httpx

from posttrain_agent.config import Settings
from posttrain_agent.experiments.nebius import PRETOK_TOKEN_KEY, NebiusBackend

PRETOK_ROW = ('{"token_ids": [1, 2], "labels": [-100, 2], "attention_mask": [1, 1], '
              '"metadata": {"id": "t1"}}\n')


def backend(tmp_path, handler):
    s = Settings(state_dir=tmp_path, data_dir=tmp_path, suite_dir=tmp_path, nebius_api_key="k",
                 nebius_base_url="https://api.example/v1/")
    return NebiusBackend(s, lambda: httpx.Client(transport=httpx.MockTransport(handler)))


def _train(tmp_path, job_replies, get_replies=None, rows=PRETOK_ROW, name="d.pretok.jsonl"):
    ds = tmp_path / name
    ds.write_text(rows)
    uploads, creates = [], []

    def h(req):
        if req.url.path.endswith("/files"):
            uploads.append(req.content)
            return httpx.Response(200, json={"id": f"file-{len(uploads)}"})
        if req.method == "GET":  # validation watch: by default the job is already running
            return (get_replies or [httpx.Response(200, json={"status": "running"})]).pop(0)
        creates.append(1)
        return job_replies[len(creates) - 1]

    b = backend(tmp_path, h)
    b._sleep = lambda s: None
    return b.train({}, ds, "Qwen/Qwen3-0.6B"), uploads, creates


def _queued(n=1):
    return httpx.Response(200, json={"id": f"ftjob-{n}", "status": "queued"})


# Live finding (2026-11-02): Nebius accepted the upload and job, then the job ENDED `failed` ~40s
# later with this message and 0 trained tokens.
def _rejected():
    return httpx.Response(200, json={"status": "failed", "error": {
        "message": "Error while processing dataset 'training': Line 0: Provided dataset does not "
                   "fall into any supported formats: CLM, Instruction, Chat, Pretokenized"}})


def test_upload_carries_the_wire_key_not_token_ids(tmp_path):
    out, uploads, creates = _train(tmp_path, [_queued()])
    assert out["provider_job_id"] == "ftjob-1" and len(uploads) == 1 and len(creates) == 1
    body = uploads[0]
    assert f'"{PRETOK_TOKEN_KEY}": [1, 2]'.encode() in body
    assert b'"token_ids"' not in body
    assert b'"labels": [-100, 2]' in body and b'"attention_mask": [1, 1]' in body
    assert b'"metadata"' not in body


def test_messages_row_is_uploaded_unchanged(tmp_path):
    row = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
    out, uploads, _ = _train(tmp_path, [_queued()], rows=json.dumps(row) + "\n", name="d.jsonl")
    assert out["provider_job_id"] == "ftjob-1"
    assert (json.dumps(row, sort_keys=True) + "\n").encode() in uploads[0]
    assert PRETOK_TOKEN_KEY.encode() not in uploads[0]


def test_local_dataset_file_is_byte_identical_after_train(tmp_path):
    ds = tmp_path / "d.pretok.jsonl"
    _train(tmp_path, [_queued()])
    assert ds.read_bytes() == PRETOK_ROW.encode()  # token_ids and metadata still on disk


def test_synchronous_4xx_naming_the_key_blocks_without_retry(tmp_path):
    bad = httpx.Response(422, json={"detail": "field required: input_ids"})
    out, uploads, creates = _train(tmp_path, [bad, _queued(2)])
    assert out["status"] == "BLOCKED" and out["stage"] == "create_job"
    assert "id" not in out and "422" in out["error"]
    assert len(uploads) == 1 and len(creates) == 1


def test_unrelated_rejection_is_blocked_at_create_job(tmp_path):
    out, uploads, creates = _train(tmp_path, [httpx.Response(400, json={"detail": "bad lr"})])
    assert out["status"] == "BLOCKED" and out["stage"] == "create_job"
    assert len(uploads) == 1 and len(creates) == 1


def test_async_rejection_blocks_with_the_provider_message_and_no_retry(tmp_path):
    out, uploads, creates = _train(tmp_path, [_queued(1), _queued(2)], [_rejected()])
    assert out["status"] == "BLOCKED" and out["stage"] == "dataset_validation"
    assert "supported formats" in out["error"] and out["provider_job_id"] == "ftjob-1"
    assert "id" not in out
    assert len(uploads) == 1 and len(creates) == 1


def test_a_running_status_does_not_end_the_watch_before_the_rejection_arrives(tmp_path):
    running = httpx.Response(200, json={"status": "running", "trained_steps": None})
    out, uploads, creates = _train(tmp_path, [_queued()], [running, running, _rejected()])
    assert out["status"] == "BLOCKED" and out["stage"] == "dataset_validation"
    assert "id" not in out and len(uploads) == 1 and len(creates) == 1


def test_training_progress_ends_the_watch(tmp_path):
    progress = httpx.Response(200, json={"status": "running", "trained_steps": 3})
    out, uploads, creates = _train(tmp_path, [_queued()], [progress])
    assert out["provider_job_id"] == "ftjob-1" and out["status"] == "queued"
    assert len(uploads) == 1 and len(creates) == 1


def test_a_failed_poll_does_not_block_the_launch(tmp_path):
    no_status = httpx.Response(404, json={"detail": "not found"})
    out, uploads, creates = _train(tmp_path, [_queued()], [no_status])
    assert out["provider_job_id"] == "ftjob-1" and "stage" not in out
    assert len(uploads) == 1 and len(creates) == 1
