"""evaluate_checkpoint accepts a fine-tuning job id and resolves it to the final checkpoint."""

from posttrain_agent.config import Settings
from posttrain_agent.experiments.domain import make_tools


class Backend:
    def __init__(self, status):
        self.status, self.evaluated = status, []

    def job_status(self, job_id):
        return self.status

    def evaluate(self, checkpoint, suite, split, seed):
        self.evaluated.append(checkpoint)
        return {"evidence": "LIVE", "checkpoint": checkpoint, "score": 0.5, "n": 10}


def tools(status, tmp_path):
    s = Settings(state_dir=tmp_path, data_dir=tmp_path, suite_dir=tmp_path)
    b = Backend(status)
    return make_tools(b, s)["evaluate_checkpoint"], b


def test_job_id_resolves_even_with_a_null_error_key(tmp_path):
    ev, b = tools({"status": "succeeded", "error": None, "checkpoint": "ft:m:ckpt-step-9"}, tmp_path)
    assert ev("ftjob-1", "heldout")["checkpoint"] == "ft:m:ckpt-step-9"
    assert b.evaluated == ["ft:m:ckpt-step-9"]


def test_job_without_a_checkpoint_is_an_error_and_evaluates_nothing(tmp_path):
    ev, b = tools({"status": "running", "error": None, "checkpoint": None}, tmp_path)
    assert "no finished checkpoint" in ev("ftjob-1", "heldout")["error"]
    assert b.evaluated == []
