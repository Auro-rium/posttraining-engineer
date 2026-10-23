"""launch_training now needs a passing check_chat_template marker and >= 50 optimizer steps.
Tests that exercise other things with tiny fake datasets skip both gates; tests that prove the
gates themselves are marked ``@pytest.mark.real_gate``."""

import pytest

from posttrain_agent.tools import budget as budget_mod
from posttrain_agent.tools import dataset as dataset_mod


def pytest_configure(config):
    config.addinivalue_line("markers", "real_gate: keep the template-check and step gates on")


@pytest.fixture(autouse=True)
def _bypass_launch_gates(request, monkeypatch):
    if request.node.get_closest_marker("real_gate"):
        return
    monkeypatch.setattr(dataset_mod, "check_marker_ok", lambda *a, **k: True)
    real = budget_mod.training_token_info
    monkeypatch.setattr(
        budget_mod, "training_token_info",
        lambda s, ref: {"total_tokens": 10**7, "avg_row_tokens": 300.0, "rows": 1, "source": "test"}
        if "error" not in real(s, ref) else real(s, ref))
