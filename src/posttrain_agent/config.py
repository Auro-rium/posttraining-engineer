"""Environment-driven settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _split_models(raw: str) -> tuple[tuple[str, str], ...]:
    """Parse ``provider:model,provider:model`` into (provider, model) pairs."""
    pairs = []
    for item in filter(None, (p.strip() for p in raw.split(","))):
        provider, _, model = item.partition(":")
        if not model:
            raise ValueError(f"fallback model {item!r} must look like provider:model")
        pairs.append((provider, model))
    return tuple(pairs)


@dataclass(frozen=True)
class Settings:
    backboard_api_key: str = ""
    backboard_base_url: str = "https://app.backboard.io/api"
    llm_provider: str = "anthropic"
    model_name: str = "claude-sonnet-4-5"
    fallback_models: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    memory_mode: str = "Readonly"  # Backboard memory on LLM calls: Auto | Readonly | off
    max_input_tokens: int = 128_000
    request_timeout: int = 120
    state_dir: Path = Path("state")
    cassette_path: Path | None = None
    cassette_mode: str = "auto"  # auto | record | replay
    max_recovery_attempts: int = 3
    model_retries: int = 2
    assistant_id: str | None = None
    tavily_api_key: str = ""
    hf_token: str = ""
    nebius_api_key: str = ""
    nebius_base_url: str = "https://api.tokenfactory.nebius.com/v1"
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_cost_usd: float = 25.0  # total estimated spend across all launched jobs
    price_per_mtok: float | None = None  # USD per 1M trained tokens; Nebius publishes none
    data_dir: Path = Path("state/data")
    suite_dir: Path = Path("state/suite")  # <split>.jsonl task files (train/heldout/regression)
    require_approval: bool = True  # human approval before every training job
    seed: int = 7
    max_parallel_subagents: int = 2
    # role -> (provider, model) overrides from POSTTRAIN_MODEL_<ROLE>; other roles use model_name
    role_models: dict[str, tuple[str, str]] = field(default_factory=dict)
    # hard caps per invocation (see budget_guard.py); the coordinator gets the larger pair
    subagent_max_calls: int = 12
    subagent_max_tokens: int = 150_000  # cumulative estimated tokens over all calls of one subagent
    coordinator_max_calls: int = 80
    coordinator_max_tokens: int = 1_000_000
    run_max_tokens: int = 2_000_000  # whole run: coordinator + all subagents, persisted in state
    max_dataset_rows: int = 1200  # SFT rows per dataset; can only be lowered (hard cap 1200)
    # Nebius Sandboxes (Beta) run the game-environment judge; see experiments/sandbox.py
    allow_unpriced: bool = False  # launch without a USD price, capped by max_trained_tokens
    max_trained_tokens: int = 10_000_000  # per-job cap on estimated trained tokens when unpriced
    nebius_project_id: str = ""
    sandbox_image: str = "python:3.12-slim"
    eval_judge: str = "auto"  # auto: sandbox if configured, else local | sandbox: required | local

    @classmethod
    def from_env(cls) -> Settings:
        e = os.environ.get
        cassette = e("POSTTRAIN_CASSETTE")
        return cls(
            backboard_api_key=e("BACKBOARD_API_KEY", ""),
            backboard_base_url=e("BACKBOARD_BASE_URL", cls.backboard_base_url),
            llm_provider=e("POSTTRAIN_LLM_PROVIDER", cls.llm_provider),
            model_name=e("POSTTRAIN_MODEL", cls.model_name),
            fallback_models=_split_models(e("POSTTRAIN_FALLBACK_MODELS", "")),
            memory_mode=e("POSTTRAIN_MEMORY_MODE", cls.memory_mode),
            state_dir=Path(e("POSTTRAIN_STATE_DIR", "state")),
            cassette_path=Path(cassette) if cassette else None,
            cassette_mode=e("POSTTRAIN_CASSETTE_MODE", cls.cassette_mode),
            max_recovery_attempts=int(e("POSTTRAIN_MAX_RECOVERY", "3")),
            model_retries=int(e("POSTTRAIN_MODEL_RETRIES", "2")),
            assistant_id=e("BACKBOARD_ASSISTANT_ID") or None,
            tavily_api_key=e("TAVILY_API_KEY", ""),
            hf_token=e("HF_TOKEN", ""),
            nebius_api_key=e("NEBIUS_API_KEY", ""),
            nebius_base_url=e("NEBIUS_BASE_URL", cls.nebius_base_url),
            base_model=e("POSTTRAIN_BASE_MODEL", cls.base_model),
            max_cost_usd=float(e("POSTTRAIN_MAX_COST_USD", "25")),
            price_per_mtok=float(e("POSTTRAIN_PRICE_PER_MTOK")) if e("POSTTRAIN_PRICE_PER_MTOK") else None,
            data_dir=Path(e("POSTTRAIN_DATA_DIR", str(Path(e("POSTTRAIN_STATE_DIR", "state")) / "data"))),
            suite_dir=Path(e("POSTTRAIN_SUITE_DIR", str(Path(e("POSTTRAIN_STATE_DIR", "state")) / "suite"))),
            require_approval=e("POSTTRAIN_REQUIRE_APPROVAL", "1").lower() not in {"0", "false", "no"},
            seed=int(e("POSTTRAIN_SEED", "7")),
            max_parallel_subagents=max(1, int(e("POSTTRAIN_MAX_PARALLEL", "2"))),
            subagent_max_calls=max(1, int(e("POSTTRAIN_SUBAGENT_MAX_CALLS", "12"))),
            subagent_max_tokens=max(1, int(e("POSTTRAIN_SUBAGENT_MAX_TOKENS", "150000"))),
            coordinator_max_calls=max(1, int(e("POSTTRAIN_COORDINATOR_MAX_CALLS", "80"))),
            coordinator_max_tokens=max(1, int(e("POSTTRAIN_COORDINATOR_MAX_TOKENS", "1000000"))),
            run_max_tokens=max(1, int(e("POSTTRAIN_RUN_MAX_TOKENS", "2000000"))),
            max_dataset_rows=max(1, min(1200, int(e("POSTTRAIN_MAX_DATASET_ROWS", "1200")))),
            allow_unpriced=e("POSTTRAIN_ALLOW_UNPRICED", "0").lower() in {"1", "true", "yes"},
            max_trained_tokens=max(1, int(e("POSTTRAIN_MAX_TRAINED_TOKENS", "10000000"))),
            nebius_project_id=e("NEBIUS_PROJECT_ID", ""),
            sandbox_image=e("POSTTRAIN_SANDBOX_IMAGE", cls.sandbox_image),
            eval_judge=e("POSTTRAIN_EVAL_JUDGE", "auto").lower(),
            role_models={k[len("POSTTRAIN_MODEL_"):].lower(): _split_models(v)[0]
                         for k, v in os.environ.items()
                         if k.startswith("POSTTRAIN_MODEL_") and k != "POSTTRAIN_MODEL_RETRIES" and v},
        )
