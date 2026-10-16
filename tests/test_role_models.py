from posttrain_agent.config import Settings
from posttrain_agent.subagents import ROLES, build_subagents


def test_role_models_parsed_from_env(monkeypatch):
    monkeypatch.setenv("POSTTRAIN_MODEL_EVALUATOR", "anthropic:claude-haiku-5-5")
    monkeypatch.setenv("POSTTRAIN_MODEL_RETRIES", "4")  # not a role
    s = Settings.from_env()
    assert s.role_models == {"evaluator": ("anthropic", "claude-haiku-5-5")}
    assert s.model_retries == 4


def test_only_overridden_roles_get_a_model():
    tools = {t: (lambda: None) for _, _, names, _ in ROLES.values() for t in names}
    specs = {s["name"]: s for s in build_subagents(tools, models={"evaluator": "m"})}
    assert specs["evaluator"]["model"] == "m"
    assert "model" not in specs["benchmark"]
