from posttrain_agent.config import Settings
from posttrain_agent.subagents import ROLES, build_subagents


def test_role_models_parsed_from_env(monkeypatch):
    monkeypatch.setenv("POSTTRAIN_MODEL_EVALUATOR", "anthropic:claude-haiku-5-5")
    monkeypatch.setenv("POSTTRAIN_MODEL_RETRIES", "4")  # not a role
    s = Settings.from_env()
    assert s.role_models == {"evaluator": ("anthropic", "claude-haiku-5-5")}
    assert s.model_retries == 4


def test_dotenv_fills_missing_values_but_never_overrides_the_environment(tmp_path, monkeypatch):
    from posttrain_agent.config import load_dotenv
    env = tmp_path / ".env"
    env.write_text("# c\nFOO_A=1\nFOO_B=\nFOO_C='quoted'\nbad line\nFOO_D=keep=this\n")
    monkeypatch.setenv("FOO_D", "from-shell")
    for k in ("FOO_A", "FOO_B", "FOO_C"):
        monkeypatch.delenv(k, raising=False)
    assert load_dotenv(env) == 2  # A and C; B is blank, D is already set
    import os
    assert os.environ["FOO_A"] == "1" and os.environ["FOO_C"] == "quoted"
    assert "FOO_B" not in os.environ and os.environ["FOO_D"] == "from-shell"
    assert load_dotenv(tmp_path / "missing") == 0
    for k in ("FOO_A", "FOO_C"):
        monkeypatch.delenv(k, raising=False)


def test_only_overridden_roles_get_a_model():
    tools = {t: (lambda: None) for _, _, names, _ in ROLES.values() for t in names}
    specs = {s["name"]: s for s in build_subagents(tools, models={"evaluator": "m"})}
    assert specs["evaluator"]["model"] == "m"
    assert "model" not in specs["benchmark"]
