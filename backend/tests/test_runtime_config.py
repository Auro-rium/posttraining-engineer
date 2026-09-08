"""Runtime budget, objective, and telemetry configuration contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.runtime_config import RuntimeConfig


def test_defaults_match_bounded_five_run_demo_contract() -> None:
    settings = RuntimeConfig(_env_file=None)

    assert settings.max_experiments == 5
    assert settings.max_cost_usd == 25.0
    assert settings.objective_suite == "AgentGym/AgentEval"
    assert settings.objective_suite_version == "agent-eval-v1"
    assert settings.telemetry_enabled is True
    assert settings.telemetry_exporter == "logging"


def test_budget_limits_cannot_be_widened_by_configuration() -> None:
    with pytest.raises(ValidationError):
        RuntimeConfig(_env_file=None, max_experiments=6)
    with pytest.raises(ValidationError):
        RuntimeConfig(_env_file=None, max_cost_usd=25.01)


def test_otlp_telemetry_requires_an_endpoint() -> None:
    with pytest.raises(ValidationError, match="telemetry_otlp_endpoint"):
        RuntimeConfig(_env_file=None, telemetry_exporter="otlp")

    settings = RuntimeConfig(
        _env_file=None,
        telemetry_exporter="otlp",
        telemetry_otlp_endpoint="http://otel-collector:4318",
    )
    assert settings.telemetry_otlp_endpoint == "http://otel-collector:4318"


def test_telemetry_can_be_disabled_for_local_diagnostics() -> None:
    settings = RuntimeConfig(_env_file=None, telemetry_enabled=False, telemetry_exporter="none")

    assert settings.telemetry_enabled is False
    assert settings.telemetry_exporter == "none"
