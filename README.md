# posttrain-agent

An autonomous post-training engineer. It is a LangChain `deepagents` harness whose
LLM calls and long-term memory go through [Backboard](https://app.backboard.io). A
coordinator agent delegates each experiment step to eight specialist subagents, and
deterministic Python code (not the LLM) validates training configs and decides
promotion. Specialists can also use optional Tavily web search.

The project is an early-stage harness. Its experiment backend is a simulation; see
the [Evidence boundary](#evidence-boundary) section before reading any result.

## Install

Requires Python 3.12 or 3.13.

```bash
uv sync --extra dev
```

This creates `.venv/` and installs the `posttrain` console script
(`.venv/bin/posttrain`).

## Configuration

Settings are read from environment variables by `Settings.from_env()`
(`src/posttrain_agent/config.py`).

| Variable | Default | Purpose |
| --- | --- | --- |
| `BACKBOARD_API_KEY` | none | Backboard API key. Required, except when `POSTTRAIN_CASSETTE_MODE=replay`. The CLI exits with status 2 if it is missing. |
| `BACKBOARD_BASE_URL` | `https://app.backboard.io/api` | Backboard API base URL. |
| `BACKBOARD_ASSISTANT_ID` | unset | Use this existing Backboard assistant. If unset, the id is read from `<state>/backboard.json`, or a new assistant is created and written there. |
| `POSTTRAIN_LLM_PROVIDER` | `anthropic` | Provider name passed to Backboard with each call. |
| `POSTTRAIN_MODEL` | `claude-sonnet-4-5` | Model name passed to Backboard. |
| `POSTTRAIN_FALLBACK_MODELS` | empty | Comma-separated `provider:model` pairs tried in order when the primary model fails, e.g. `openai:gpt-4o,anthropic:claude-haiku-4-5`. A malformed entry raises `ValueError`. |
| `POSTTRAIN_MEMORY_MODE` | `Readonly` | Value passed as the `memory` argument of Backboard's `send_message` (the code's comment lists `Auto`, `Readonly`, `off`). |
| `POSTTRAIN_STATE_DIR` | `state` | Directory for SQLite files and `backboard.json`. |
| `POSTTRAIN_CASSETTE` | unset | Path to a JSONL cassette file for record/replay of model responses. |
| `POSTTRAIN_CASSETTE_MODE` | `auto` | `auto` (replay hits, record misses), `record` (always call live), or `replay` (fail on a miss). The cassette is only attached when `POSTTRAIN_CASSETTE` is set. |
| `POSTTRAIN_MAX_RECOVERY` | `3` | Attempts per `run`/`resume` before the run is marked `FAILED`. |
| `POSTTRAIN_MODEL_RETRIES` | `2` | Retries per model on transient errors. |
| `TAVILY_API_KEY` | empty | Optional. Enables the `web_search` and `web_extract` tools. Without it, both tools return an "unavailable" error instead of failing the run. |

`max_input_tokens` (128000) and `request_timeout` (120 seconds) have no environment
variable; they use the defaults in `Settings`.

## Web tools

`src/posttrain_agent/web.py` provides two async tools that call the Tavily HTTP API
directly with `httpx`:

- `web_search(query, max_results=5, topic="general")`: returns an answer and up to
  10 results (title, URL, 600-character snippet). Queries containing `heldout://` or
  `sealed://` are rejected.
- `web_extract(urls)`: fetches cleaned text for up to 5 URLs, truncated to 4000
  characters per page.

Both return `evidence: EXPLANATION`, since web content is not a measurement. Which
agents get them:

- coordinator: `web_search`
- `researcher`: `web_search` and `web_extract`
- `training_designer`: `web_search`

Other roles get no web tools.

## CLI usage

```bash
posttrain run "<goal>" [--thread NAME]      # start, or continue, a run on a thread
posttrain resume [--thread NAME]            # resume an existing thread from its last checkpoint
posttrain history [--thread NAME]           # list checkpoints of a thread, newest first (JSON)
posttrain fork CHECKPOINT_ID [--thread NAME]  # re-execute from a past checkpoint
posttrain runs                              # print the run ledger (JSON)
```

`--thread` defaults to `default`. `run` and `resume` print a JSON report (or the
status for `resume`) and exit 0 only when the status is `COMPLETED`. Application state
is stored under `POSTTRAIN_STATE_DIR`:

- `checkpoints.sqlite`: LangGraph checkpoints
- `store.sqlite`: the `/memories/` file store and the run ledger
- `backboard.json`: the assistant id, if the assistant was created by this tool

## Running tests

```bash
.venv/bin/python -m pytest
```

Run from the repository root. The suite has 29 tests in `tests/`. They use
`tests/fakes.py`, a scripted stand-in for `BackboardClient`, together with a real
deepagents graph and SQLite persistence. The web tool tests use an `httpx`
`MockTransport`. No test calls the live Backboard or Tavily APIs.

## Evidence boundary

- **The default experiment backend is `LocalStubBackend`.** `build_agent` uses it
  whenever no backend is passed, and the CLI never passes one. Benchmarks, training
  jobs and evaluations are therefore deterministic simulations computed from hashes
  (`src/posttrain_agent/domain.py`).
- **Every stub result is labelled `EXPLANATION`.** Stub checkpoints also receive a
  fixed +0.08 score bonus in `evaluate`, so a stub candidate can look like an
  improvement.
- **The promotion gate fails closed.** `promotion_gate` rejects any result whose
  evidence is not `LIVE` or `PRIOR_VERIFIED_RUN`. Stub runs therefore can never
  return `PROMOTE`. The gate also requires matching suite, split and seed, a relative
  improvement of at least 10%, and an absolute regression no greater than 0.05. It
  only returns a decision; nothing in this codebase persists or applies a champion.
- **Web results are not verified evidence.** They are labelled `EXPLANATION` and
  cannot feed the promotion gate.
- **No real training adapter exists.** There is no SageMaker or GPU implementation.
  To add one, implement the `ExperimentBackend` protocol in `domain.py`
  (`benchmark`, `train`, `evaluate`), returning provider-measured numbers labelled
  `LIVE`, and pass it to `build_agent(..., experiments=...)`.
- **Live Backboard behaviour has not been exercised.** The tests use a fake Backboard
  client. Request and response shapes, memory semantics and error codes are
  unverified against the real service. The Tavily integration is likewise
  unverified against the live API.

## Layout

```
src/posttrain_agent/
  agent.py           deep-agent assembly (coordinator prompt, tools, subagents)
  backboard_model.py ChatBackboard chat model and Cassette record/replay
  cli.py             `posttrain` command
  config.py          Settings from environment variables
  domain.py          deterministic QLoRA validation, promotion gate, stub backend, tools
  memory.py          assistant bootstrap and remember/recall tools
  persistence.py     SQLite checkpointer, store and CompositeBackend
  replay.py          run_with_recovery, history, fork, ledger
  resilience.py      retry/fallback middleware and transient-error classification
  subagents.py       eight specialist roles
  web.py             Tavily web_search / web_extract tools
  prompts/*.md       subagent system prompts
tests/               pytest suite
docs/architecture.md component and data-flow description
```
