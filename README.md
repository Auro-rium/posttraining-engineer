# posttrain-agent

posttrain-agent is an autonomous post-training engineer. You give it a goal, such as "reduce
tool-call failures of this checkpoint". A coordinator agent, built on LangChain's deepagents
harness and using Backboard as its LLM and memory, delegates to eight specialist subagents. The
specialists benchmark the checkpoint on a train split, cluster failures, build a verified chat
SFT dataset, choose bounded LoRA hyperparameters, and propose a fine-tuning job on Nebius Token
Factory. The job cannot start until a human approves it at the CLI. The candidate and champion are
then scored on sealed held-out tasks, and a deterministic gate decides PROMOTE or REJECT. The
agents propose; deterministic Python code validates datasets and hyperparameters, enforces the
budget and the held-out boundary, and makes the promotion decision. State is durable SQLite, so
runs survive crashes and can be resumed, forked, or replayed offline from a recorded cassette.

---

## Contents

- [How it works](#how-it-works)
- [Features](#features)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
- [CLI reference](#cli-reference)
- [Configuration reference](#configuration-reference)
- [Dataset format](#dataset-format)
- [Hyperparameter search space](#hyperparameter-search-space)
- [Budget model](#budget-model)
- [Safety model](#safety-model)
- [Operating guide](#operating-guide)
- [Extending](#extending)
- [Testing](#testing)
- [Status and limitations](#status-and-limitations)
- [More detail](#more-detail)

---

## How it works

### The loop

The coordinator prompt (`ORCHESTRATOR_PROMPT` in `src/posttrain_agent/agent.py`) asks it to run one
experiment as the sequence below, delegating each step with the `task` tool. The order is an
instruction in the prompt; the code does not enforce it.

```
goal
  |
  v
coordinator (deepagents + Backboard LLM)           tools: remember, recall, web_search, read_telemetry
  |
  |  1 benchmark ............ run_benchmark            (TRAIN split only)
  |  2 failure_analyst ...... read_training_logs
  |  3 researcher ........... web_search, web_extract, hf_search, hf_model_info, hf_dataset_info
  |  4 data_curator ......... verify_dataset, save_dataset, hf_dataset_info
  |  5 training_designer .... validate_hparams, estimate_budget, web_search, hf_model_info
  |  6 training_executor .... launch_training  <== HUMAN APPROVAL GATE
  |                           get_training_status, read_training_logs
  |  7 evaluator ............ evaluate_checkpoint      (heldout | regression, scores only)
  |  8 champion_manager ..... decide_promotion         (deterministic gate)
  v
PROMOTE or REJECT, with an evidence label on every number (LIVE | PRIOR_VERIFIED_RUN | EXPLANATION)
```

### The eight subagent roles

| Role | Gets these tools | Plus memory |
|------|------------------|-------------|
| `benchmark` | `run_benchmark` | recall |
| `failure_analyst` | `read_training_logs` | recall |
| `researcher` | `web_search`, `web_extract`, `hf_search`, `hf_model_info`, `hf_dataset_info` | recall, remember |
| `data_curator` | `verify_dataset`, `save_dataset`, `hf_dataset_info` | recall |
| `training_designer` | `validate_hparams`, `estimate_budget`, `web_search`, `hf_model_info` | recall |
| `training_executor` | `launch_training`, `get_training_status`, `read_training_logs` | recall |
| `evaluator` | `evaluate_checkpoint` | recall |
| `champion_manager` | `decide_promotion` | recall |

Each role's system prompt is a file in `src/posttrain_agent/prompts/`, prefixed by a shared
preamble in `subagents.py`. Most tools are deterministic Python (`domain.py`, `dataset.py`,
`budget.py`, `nebius.py`). `web_*` and `hf_*` are read-only lookups against external services.

### Components

- **Backboard** is the LLM and the semantic memory. `ChatBackboard` is a LangChain chat model
  that sends each turn to the Backboard API. Backboard routes to the model named by
  `POSTTRAIN_LLM_PROVIDER` and `POSTTRAIN_MODEL`. Long-term memory is Backboard assistant
  memory, exposed as the `remember` and `recall` tools.
- **deepagents** is the harness. It provides the coordinator loop, planning (`write_todos`), the
  `task` tool that spawns subagents, file tools, and the human-in-the-loop `interrupt_on` mechanism.
- **Nebius Token Factory** runs fine-tuning jobs (LoRA SFT) and chat completions for evaluation.
  It is reached through its OpenAI-compatible REST API.
- **Tavily** provides web search and page extraction. **Hugging Face Hub** provides read-only
  model and dataset metadata. Both are optional.
- **SQLite** holds LangGraph checkpoints, a cross-thread store, the run ledger, telemetry, and the
  spend ledger.

See [docs/architecture.md](docs/architecture.md) for the data flow and internals.

---

## Features

- **Human approval before any training job.** `launch_training` is interrupted and pauses the
  run durably. Nothing is submitted until `posttrain approve`.
- **Budget estimate and cumulative spend cap.** Each approval prompt shows an estimate. Launches
  are blocked when spend plus the estimate would exceed `POSTTRAIN_MAX_COST_USD`.
- **Dataset verifier.** Structural and policy checks on chat-format SFT rows: schema, allow-listed
  tools, JSON arguments, no sealed references, no duplicates. It is not an environment replay.
- **Hugging Face lookup.** Search, model metadata and dataset metadata, read-only.
- **Tavily web search.** Search and page extraction for the researcher and training designer.
- **Telemetry and training-log readers.** Per-model and per-tool counts, error rates, latency
  percentiles and token totals, plus the provider's job event log.
- **SQLite persistence.** Checkpoints, memory store and run ledger survive process restarts.
- **Crash recovery.** A failed run retries from its last checkpoint with backoff.
- **Checkpoint fork and replay.** Re-execute from any past checkpoint with `fork`.
- **Model fallback chain.** Retries transient errors on the primary model, then tries each
  fallback in order.
- **Cassette record and replay.** Records model responses to a JSONL file and replays them
  without calling Backboard.

---

## Requirements

- Python 3.12 or 3.13 (`requires-python = ">=3.12,<3.14"`).
- [uv](https://docs.astral.sh/uv/) for installation and running.
- A Backboard API key. This is required for every run except cassette replay.
- A Nebius Token Factory API key for real training and evaluation. Without it the backend is a
  labelled simulation (see [Status and limitations](#status-and-limitations)).
- Optional: a Tavily API key (web tools), a Hugging Face token (authenticated Hub requests and
  gated repositories).
- Network access to `app.backboard.io`, `api.tokenfactory.nebius.com`, `api.tavily.com` and
  `huggingface.co` as applicable.

---

## Quickstart

Run these from the repository root.

**1. Install the dependencies.**

```bash
uv sync --extra dev
```

This creates `.venv/` and installs the `posttrain` console script into it. Run commands with
`uv run posttrain ...` or `.venv/bin/posttrain ...`.

**2. Set environment variables.**

```bash
cp .env.example .env
# edit .env, then load it into your shell (the CLI does not read .env files itself)
set -a; . ./.env; set +a
```

At minimum set:

| Variable | Why |
|----------|-----|
| `BACKBOARD_API_KEY` | Required for every run. |
| `NEBIUS_API_KEY` | Required for real training and evaluation. Leave empty only for the simulation. |
| `POSTTRAIN_PRICE_PER_MTOK` | **Required for any launch.** Nebius publishes no per-token fine-tuning price, so you supply one. Find your rate in your Nebius Token Factory account or billing console, or in your contract or quote. Enter the USD cost per 1 million trained tokens. Without it every launch is blocked at the budget stage. |

`TAVILY_API_KEY` and `HF_TOKEN` are optional. Without Tavily the web tools return an
"unavailable" error and the researcher continues without them.

**3. Create the evaluation task files.**

The default suite directory is `state/suite` (`POSTTRAIN_SUITE_DIR` overrides it). Create three
JSONL files there:

```bash
mkdir -p state/suite
```

| File | Used by | Visible to agents |
|------|---------|-------------------|
| `train.jsonl` | `run_benchmark` | Scores and up to 20 failures, including expected and got values |
| `heldout.jsonl` | `evaluate_checkpoint(split="heldout")` | Scores only |
| `regression.jsonl` | `evaluate_checkpoint(split="regression")` | Scores only |

Each line of each file is one task object:

```json
{"id": "api-502-001",
 "messages": [{"role": "user", "content": "The api service is returning 502s. Investigate."}],
 "tools": [{"type": "function", "function": {"name": "get_logs",
            "description": "Read recent logs for a service.",
            "parameters": {"type": "object",
                           "properties": {"service": {"type": "string"}},
                           "required": ["service"]}}}],
 "expected": {"name": "get_logs", "arguments": {"service": "api"}}}
```

| Field | Required | Meaning |
|-------|----------|---------|
| `id` | yes | Task identifier, echoed in failure reports. |
| `messages` | yes | Chat messages sent as-is to Nebius `/chat/completions`. |
| `tools` | no | Tool schemas forwarded as-is. Without them the model has no tool to call. |
| `expected` | yes | `{"name": ..., "arguments": {...}}`. |

Scoring: the model is called with `temperature: 0`. A task scores 1 only if the model's **first**
tool call has exactly this `name` and `arguments` (a parsed JSON object compared for equality).
Otherwise it scores 0. At most 200 tasks per split are read. Use the same six tool names the
dataset verifier allows (`get_logs`, `inspect_service`, `read_config`, `edit_config`,
`restart_service`, `run_healthcheck`) so that the behaviour you train and the behaviour you measure
are the same.

The held-out and regression files are sealed. Agents receive only `{evidence, score, n}` for them.
Task contents never leave `nebius.py`.

**4. Start a run.**

```bash
uv run posttrain run "Reduce tool-call failures of the base model on API incidents" --thread demo
```

The command prints a JSON status. Exit code `3` means the run is paused for approval.

**5. Review the approval request.**

```bash
uv run posttrain pending --thread demo
```

The `description` field shows the base model, dataset ref, hyperparameters and the budget
estimate (`estimated_trained_tokens`, `estimated_cost_usd`, `within_budget`, `spent_usd`, `cap_usd`).
The description does not include the dataset hash. The hash is returned in the tool result after
launch.

**6. Approve or reject.**

```bash
uv run posttrain approve --thread demo
# or
uv run posttrain reject --thread demo -m "Too expensive; use fewer epochs"
```

Approval resumes the paused run, which then submits the job, monitors it, runs the evaluations and
makes the gate decision. A rejection tells the agent why, and the agent revises its plan. It does not
retry the same launch.

**7. Resume if needed.**

```bash
uv run posttrain resume --thread demo
```

Use this after a crash or a `FAILED` status. Running `posttrain run` on a thread that already has
saved state also resumes it, and the goal text is ignored.

**8. Inspect the results.**

```bash
uv run posttrain history --thread demo      # checkpoints, newest first
uv run posttrain runs                       # run ledger: status, attempts, errors, pending actions
uv run posttrain telemetry --errors         # model and tool errors
cat state/spend.jsonl                       # estimated spend per launched job
ls state/data/                              # verified datasets
```

Look for `"evidence": "LIVE"` on measurements. Only results labelled `LIVE` (or
`PRIOR_VERIFIED_RUN`) can produce `PROMOTE`. `EXPLANATION` results are simulated or inferred.

---

## CLI reference

Entry point: `posttrain` (`posttrain_agent.cli:main`). Every command except `telemetry` needs
`BACKBOARD_API_KEY`, or `POSTTRAIN_CASSETTE_MODE=replay`.

| Command | Arguments | What it does |
|---------|-----------|--------------|
| `run` | `GOAL` `--thread ID` (default `default`) | Starts the run on the thread, or resumes it if the thread already has state. |
| `resume` | `--thread ID` | Continues a thread from its last checkpoint. |
| `pending` | `--thread ID` | Prints the actions waiting for a human decision. |
| `approve` | `--thread ID` | Approves every pending action and continues. Errors if nothing is pending. |
| `reject` | `--thread ID` `-m/--message TEXT` | Rejects every pending action. The message is shown to the agent. |
| `history` | `--thread ID` | Lists checkpoints, newest first: id, step, source, next node, message count. |
| `fork` | `--thread ID` `CHECKPOINT_ID` | Re-executes from a past checkpoint. Original history is kept. Prints the final message. |
| `runs` | none | Prints the run ledger across threads (up to 100 entries). |
| `telemetry` | `--last N` (default 500) `--errors` | Summarises `state/telemetry.jsonl`. Needs no key or network. |

Exit codes for `run`, `resume`, `approve` and `reject`:

| Code | Meaning |
|------|---------|
| `0` | `COMPLETED` |
| `1` | `FAILED` (all recovery attempts exhausted) |
| `2` | `BACKBOARD_API_KEY` is not set |
| `3` | `AWAITING_APPROVAL` |

---

## Configuration reference

Settings are read from the process environment by `Settings.from_env()` in
`src/posttrain_agent/config.py`. The code does not load `.env` files. Use `.env.example` as a
template.

| Variable | Default | Purpose |
|----------|---------|---------|
| `BACKBOARD_API_KEY` | `""` | Backboard API key. |
| `BACKBOARD_BASE_URL` | `https://app.backboard.io/api` | Backboard API base URL. |
| `BACKBOARD_ASSISTANT_ID` | unset | Use this existing assistant instead of creating one. |
| `POSTTRAIN_LLM_PROVIDER` | `anthropic` | Provider name passed to Backboard for the primary model. |
| `POSTTRAIN_MODEL` | `claude-sonnet-4-5` | Model name passed to Backboard for the primary model. |
| `POSTTRAIN_FALLBACK_MODELS` | `""` | Comma-separated `provider:model` fallbacks, tried in order. |
| `POSTTRAIN_MEMORY_MODE` | `Readonly` | Backboard memory mode sent with each LLM call. The config comment lists `Auto`, `Readonly`, `off`. |
| `POSTTRAIN_MODEL_RETRIES` | `2` | Retries of a transient model error on the same model before falling back. |
| `POSTTRAIN_MAX_RECOVERY` | `3` | Attempts per run in `run_with_recovery`. |
| `POSTTRAIN_STATE_DIR` | `state` | Root of all durable state. |
| `POSTTRAIN_CASSETTE` | unset | Path to a cassette JSONL file. Enables recording or replay. |
| `POSTTRAIN_CASSETTE_MODE` | `auto` | `auto` replays hits and records misses; `record` always calls live; `replay` never calls and fails on a miss. |
| `NEBIUS_API_KEY` | `""` | Nebius key. Empty selects the labelled simulation backend. |
| `NEBIUS_BASE_URL` | `https://api.tokenfactory.nebius.com/v1` | Nebius OpenAI-compatible base URL. |
| `POSTTRAIN_BASE_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | Base model for fine-tuning and for the `base` checkpoint alias. |
| `POSTTRAIN_PRICE_PER_MTOK` | unset | USD per 1M trained tokens. **Required to launch.** |
| `POSTTRAIN_MAX_COST_USD` | `25` | Cumulative estimated spend cap across all launched jobs. |
| `POSTTRAIN_DATA_DIR` | `<STATE_DIR>/data` | Where verified datasets are stored. |
| `POSTTRAIN_SUITE_DIR` | `<STATE_DIR>/suite` | Directory containing `train.jsonl`, `heldout.jsonl`, `regression.jsonl`. |
| `POSTTRAIN_REQUIRE_APPROVAL` | `1` | Set to `0`, `false` or `no` to disable the approval gate. Only do this for unattended runs you accept. |
| `POSTTRAIN_SEED` | `7` | Seed for benchmark and evaluation requests and for Nebius jobs. |
| `TAVILY_API_KEY` | `""` | Enables `web_search` and `web_extract`. |
| `HF_TOKEN` | `""` | Bearer token for Hugging Face Hub requests. |

Not environment-driven: `request_timeout` (120 seconds, used for Backboard calls) and
`max_input_tokens` (128000, passed to the chat model profile). Both use fixed defaults. The Nebius
client uses a fixed 120-second timeout, and the Hugging Face and Tavily clients use 30 seconds.

---

## Dataset format

Datasets are chat-format SFT files: one JSON object per line, stored as `<name>.jsonl` in
`POSTTRAIN_DATA_DIR`. Names must match `^[A-Za-z0-9_.-]{1,64}$` and end in `.jsonl`. Path separators
are refused.

```json
{"messages": [
   {"role": "user", "content": "The api service is returning 502s. Investigate."},
   {"role": "assistant", "content": "", "tool_calls": [
      {"id": "call_1", "type": "function",
       "function": {"name": "get_logs", "arguments": "{\"service\": \"api\"}"}}]}],
 "metadata": {"evidence": "LIVE", "source_task_id": "api-502-001"}}
```

Rules enforced by `dataset.verify_rows`:

- Each row is an object with a non-empty `messages` list.
- Each message has `role` in `system`, `user`, `assistant`, `tool`.
- `content` is a string or null. A message needs content or tool calls.
- Only `assistant` messages may carry `tool_calls`.
- Every tool call's `function.name` is in the allow-list: `get_logs`, `inspect_service`,
  `read_config`, `edit_config`, `restart_service`, `run_healthcheck`.
- Tool arguments parse to a JSON object (a string that parses, or an object).
- The last message is from the `assistant`. It is the training target.
- No row contains `heldout://`, `sealed://`, `heldout/` or `sealed/`.
- `metadata.evidence`, if present, is `LIVE`, `PRIOR_VERIFIED_RUN` or `EXPLANATION`.
- No two rows are identical (SHA-256 of the canonical JSON).
- At least one row.

Warnings, not errors: rows labelled `EXPLANATION`, and datasets with no evidence labels at all.

**What the verifier does not check.** It does not replay tool calls in an environment, so it does
not know whether a call is correct. It does not check that a `tool` message follows the matching
tool call, that `id` values are unique or linked, or that `source_task_id` refers to a real task.
It does not check licence or provenance of external data, and it does not detect contamination
beyond the sealed-marker substrings. Token length is estimated, not enforced (see
[Budget model](#budget-model)). Verification is structural and policy-based. It is not a
correctness check.

`save_rows` writes nothing unless verification passes. `launch_training` re-verifies the dataset
before submitting it, and reports its SHA-256 in the result.

---

## Hyperparameter search space

Source: `HPARAM_SPACE` and `OPTIONAL_HPARAMS` in `src/posttrain_agent/domain.py`. Bounds are
inclusive. LoRA is always on.

| Key | Type | Range | Required |
|-----|------|-------|----------|
| `lora_r` | integer | 8 to 128 | yes |
| `lora_alpha` | integer | 8 to 256, and at most `4 * lora_r` | yes |
| `lora_dropout` | float | 0.0 to 0.1 | yes |
| `learning_rate` | float | 1e-6 to 5e-4 | yes |
| `n_epochs` | integer | 1 to 5 | yes |
| `batch_size` | integer | 1 to 64 | yes |
| `context_length` | integer | 8192 to 32768 | yes |
| `warmup_ratio` | float | 0.0 to 0.2 | no |
| `weight_decay` | float | 0.0 to 0.1 | no |
| `packing` | boolean | `true` or `false` | no |

`validate_hparams` returns a list of violations. An empty list means valid. It rejects missing
required keys, unknown keys (including `lora`), booleans used as numbers, non-finite numbers, and
non-integer values for integer keys. `launch_training` re-runs this check, so a bad configuration
cannot reach Nebius.

---

## Budget model

Source: `src/posttrain_agent/budget.py`.

1. **Tokens per row** is a heuristic: `max(1, chars // 4)`, where chars is the sum over messages of
   `len(content) + 16`, plus the tool name and arguments length for each tool call.
2. **Trained tokens** is `sum(min(row_tokens, context_length)) * n_epochs`. Defaults are
   `n_epochs = 3` and `context_length = 8192` when absent. Rows longer than the context are
   counted at the context length and reported in `rows_over_context`.
3. **Estimated cost** is `trained_tokens / 1e6 * POSTTRAIN_PRICE_PER_MTOK`.
4. **Spent** is the sum of `estimated_cost_usd` in `state/spend.jsonl`. It is recorded after each
   successful submission that returns a provider job ID. Stub launches are recorded too.
5. **Within budget** is `spent + cost <= POSTTRAIN_MAX_COST_USD`.

Without `POSTTRAIN_PRICE_PER_MTOK` the estimate reports token counts only, with
`estimated_cost_usd: null` and `within_budget: null`. `launch_training` then returns
`BLOCKED` at the `budget` stage. This is deliberate: the system fails closed.

The estimate is an estimate. Nebius bills by compute, and the spend ledger records estimates, not
invoices. Reconcile against your Nebius billing before relying on the cap.

---

## Safety model

- **Sealed held-out data.** `run_benchmark` accepts only the `train` split. `evaluate_checkpoint`
  accepts only `heldout` or `regression`, and returns only `evidence`, `score`, `n` and the seed
  (no per-task failures). Web queries containing `heldout://` or `sealed://` are refused, and
  dataset rows containing sealed markers are refused.
- **Fail-closed promotion gate.** `decide_promotion` (`promotion_gate` in `domain.py`) returns
  `REJECT` unless all of the following hold:
  - both results have verified evidence (`LIVE` or `PRIOR_VERIFIED_RUN`);
  - suite, split and seed match between champion and candidate;
  - the relative improvement is at least 10% (if the champion score is 0, any positive candidate
    score passes this check);
  - the regression is at most 0.05 absolute.
  Only the deterministic code decides. No agent can set a decision. The gate does not itself
  require the split to be `heldout`; it requires the two inputs to match.
- **Evidence labels.** Every measurement carries `evidence`. `LIVE` means provider-measured.
  `PRIOR_VERIFIED_RUN` is accepted by the gate, but no code path produces it. `EXPLANATION` means
  simulated or inferred, and it can never be promoted. Subagent prompts require labels to be
  copied unchanged from tool results.
- **Human approval gate.** `launch_training` is the only path to a training job. The agent
  interrupts before it runs. Approve or reject is recorded durably, in the checkpoint.
- **Stub by default.** With no `NEBIUS_API_KEY`, the backend is `LocalStubBackend`. Every result it
  returns is `EXPLANATION`, and the stub cannot produce a promotion.
- **What telemetry logs.** `state/telemetry.jsonl` contains event type, tool or model name,
  latency, success flag, token counts and the exception class name. It does not contain prompts,
  completions or tool arguments.
- **What telemetry does not protect.** Other state is sensitive. `checkpoints.sqlite` stores full
  transcripts, including tool results. `data/` stores training examples. The run ledger in
  `store.sqlite` stores exception messages in `errors`. Treat `state/` as sensitive.

---

## Operating guide

### State directory layout

Everything is relative to `POSTTRAIN_STATE_DIR` (default `state/`, resolved from the current
directory) unless noted.

```
state/
  checkpoints.sqlite     LangGraph checkpoints: full thread state for every step
  store.sqlite           /memories/ files (including AGENTS.md) and the run ledger
  backboard.json         Backboard assistant id, written on first use
  telemetry.jsonl        model and tool events (append-only)
  spend.jsonl            one line per launched job: estimated cost, dataset, job id
  data/                  POSTTRAIN_DATA_DIR: verified <name>.jsonl datasets
  suite/                 POSTTRAIN_SUITE_DIR: train.jsonl, heldout.jsonl, regression.jsonl
```

The cassette file is wherever `POSTTRAIN_CASSETTE` points. Scratch files the agents write under
`/work/` live in graph state (inside `checkpoints.sqlite`), not on disk.

### Recovering from crashes

1. **Inside one process.** A failing step is retried from the last checkpoint, up to
   `POSTTRAIN_MAX_RECOVERY` attempts, with backoff of 2 s, 4 s, and so on. Completed steps are not
   re-executed because their results are checkpointed.
2. **After the process died.** Run `posttrain resume --thread <id>`. It loads the last checkpoint
   in a new process.
3. **Check for an orphaned launch.** The job is submitted first, then `spend.jsonl` is written.
   A crash between those two steps leaves a job on Nebius with no spend record. Check the Nebius
   job list before resuming a run that was in the launch step. The code has no idempotency key for
   launches, so a launch that was submitted but not yet checkpointed can be submitted again on
   resume.
4. **Fatal errors** are not retried: `CassetteMiss`, `KeyboardInterrupt` and task cancellation.

### Replaying offline with the cassette

The cassette stores **model responses only**, keyed by a SHA-256 of provider, model, system prompt,
rendered transcript and bound tools.

```bash
# 1. Record a live run into a cassette
POSTTRAIN_CASSETTE=state/cassette.jsonl POSTTRAIN_CASSETTE_MODE=record \
  uv run posttrain run "goal" --thread rec1

# 2. Replay it into a fresh state directory, with no Backboard calls for model turns
POSTTRAIN_STATE_DIR=state-replay POSTTRAIN_CASSETTE=state/cassette.jsonl \
  POSTTRAIN_CASSETTE_MODE=replay uv run posttrain run "goal" --thread rec1
```

Points to know before you rely on replay:

- Tools are **re-executed** during replay. Only the model calls come from the cassette. If
  `NEBIUS_API_KEY` is set, replayed `launch_training` or evaluation calls reach Nebius again. Unset
  it for offline replay. The stub backend then runs. Its results can differ from the recording if
  the recording used live Nebius.
- A request that was not recorded raises `CassetteMiss` (fatal). The rendered transcript must match
  the recording exactly, so the same goal, thread and tool results are needed.
- Replay with `POSTTRAIN_CASSETTE_MODE=replay` does not need `BACKBOARD_API_KEY`, but the assistant
  id still has to resolve. Keep the recorded `backboard.json`, or set `BACKBOARD_ASSISTANT_ID`.
  Otherwise the code tries to create an assistant over the network.
- The approval gate applies during replay too. You must approve the replayed launch.

### Forking and history

```bash
uv run posttrain history --thread demo
uv run posttrain fork --thread demo <checkpoint_id>
```

`fork` re-executes from that checkpoint with the configured model (or the cassette, if set). It
does not delete later checkpoints.

### Reading telemetry

```bash
uv run posttrain telemetry --last 500
uv run posttrain telemetry --errors
```

The output has `by_name` keys of the form `model:<provider>:<model>` and `tool:<name>`. Each entry
has `calls`, `errors`, `p50_ms`, `p95_ms`, `input_tokens` and `output_tokens`. Top-level keys
include `error_types` and `recent_errors`. The coordinator can call the same summary through
`read_telemetry`.

### Job status

There is no standalone CLI command to poll a provider job. The `training_executor` role fetches
status with `get_training_status` while it runs, and it can be asked again by resuming the thread.
Job events come from `read_training_logs`.

---

## Extending

### Add another training or evaluation provider

Implement the `ExperimentBackend` protocol in `src/posttrain_agent/domain.py`:

```python
class ExperimentBackend(Protocol):
    def benchmark(self, checkpoint: str, suite: str, seed: int) -> dict[str, Any]: ...
    def train(self, config: dict[str, Any], dataset_path: Path, base_model: str) -> dict[str, Any]: ...
    def evaluate(self, checkpoint: str, suite: str, split: str, seed: int) -> dict[str, Any]: ...
    def job_status(self, job_id: str) -> dict[str, Any]: ...
    def job_events(self, job_id: str, limit: int = 50) -> dict[str, Any]: ...
```

Contract points taken from the existing code:

- `train` must return `provider_job_id` if the job was submitted. Otherwise no spend is recorded.
- `evaluate` must return `score`, `evidence`, and echo `suite`, `split` and `seed`, because the gate
  compares them.
- Return `"evidence": "LIVE"` only for provider-measured values. Use `EXPLANATION` otherwise.
- Keep held-out task contents out of the return value.

Pass your backend to `build_agent(settings, runtime, experiments=...)`. The CLI picks the backend
with `default_backend` in `src/posttrain_agent/agent.py`, so change that function, or add a flag to
the CLI.

### Add a tool

1. Write a function with a docstring. The docstring is the tool description the model sees.
2. Expose it. For a deterministic tool, add it to the dictionary returned by `make_tools` in
   `domain.py`. For a web-style tool, add it to the dict returned by a factory, and merge that dict
   into `tools` in `build_agent`.
3. Give it to the roles that need it by adding its name to the `ROLES` tuple in `subagents.py`. Or
   add it to the coordinator's `tools` list in `build_agent`.
4. Add a test. Keep any tool that changes state behind the approval gate, or make it read-only.

---

## Testing

```bash
uv sync --extra dev
.venv/bin/python -m pytest
```

The pytest config in `pyproject.toml` uses `asyncio_mode = "auto"` and adds `tests/` and `src/` to
the path. Lint with `.venv/bin/ruff check src tests` (line length 100).

| Test file | Covers |
|-----------|--------|
| `test_approval.py` | Pause at `launch_training`, resume in a new process, reject with a message, approval disabled |
| `test_backboard_model.py` | Stateless calls, tool-call round trip, malformed arguments, empty responses, cassette record and replay |
| `test_data_budget.py` | Dataset verifier rules, path traversal, budget math, cap and unpriced behaviour |
| `test_domain.py` | Hyperparameter validation, promotion gate, sealed split, launch gating, stub promotion refusal |
| `test_e2e.py` | Full graph with SQLite persistence: assistant reuse, memory tool, crash then resume, history and fork, file memory across threads |
| `test_hf_nebius_telemetry.py` | Hugging Face tools, Nebius upload, job create, status, events, scoring, telemetry redaction |
| `test_resilience.py` | Transient classification, retry on the same model, fallback order, non-transient fall-through |
| `test_web.py` | Tavily search and extract, missing key, sealed query refusal, which roles get web tools |

Tests use a scripted fake Backboard client (`tests/fakes.py`) and `httpx.MockTransport` for Nebius,
Tavily and Hugging Face. No test calls a live service.

---

## Status and limitations

Read this before relying on the system.

- **No live calls have been exercised.** The tests use a fake Backboard client and mocked HTTP
  transports. No live Backboard, Nebius, Tavily or Hugging Face call has been made by the test
  suite. The request shapes follow the published APIs as implemented in the code, but they are
  unverified against the live services.
- **Serving a fine-tuned checkpoint for evaluation is untested.** Evaluation sends chat completions
  with the checkpoint string as the `model` value. Whether Nebius serves a newly fine-tuned
  checkpoint under that name has not been tested.
- **Token estimates are heuristic.** The chars/4 estimate is not Nebius's tokenizer. Billing is by
  compute. Spend records are estimates.
- **No live price.** Nebius publishes no per-token fine-tuning price, so `POSTTRAIN_PRICE_PER_MTOK`
  is operator-supplied. The cap is only as accurate as that value.
- **Simulation without a key.** With no `NEBIUS_API_KEY`, every benchmark, training and evaluation
  result comes from `LocalStubBackend`. These results are labelled `EXPLANATION`, and the gate
  always rejects them. The sandbox default is the stub.
- **Subagent outputs are not schema-checked.** The OUTPUT CONTRACT blocks in the prompt files
  describe JSON shapes. The code does not validate them. Some prompt inputs, such as `run_id` and
  `manifest_hash`, are not supplied by any tool. Treat the prompts as guidance.
- **No champion store or deployment.** `decide_promotion` returns a decision. The code does not
  record a champion, deploy a checkpoint, or pick the next candidate. The coordinator prompt
  describes a champion, but nothing persists it.
- **Approval shows the dataset reference, not its hash.** The dataset could change between approval
  and submission. `launch_training` re-verifies the file before upload, and its result reports the
  SHA-256, but the approval text does not include the hash.
- **Crash between submit and spend record.** See [Recovering from crashes](#recovering-from-crashes).
- **Replay re-executes tools.** See [Replaying offline with the cassette](#replaying-offline-with-the-cassette).
- **Approval can be turned off.** `POSTTRAIN_REQUIRE_APPROVAL=0` removes the only human check on
  spend. Use it only for runs you accept.
- **Web and Hugging Face lookups are external and not cached.** They are not recorded in the
  cassette, so a replay makes them again.
- **Not included:** no UI, no multi-process scheduling, no retry of a rejected launch, no
  cancellation of a running provider job, and no automatic promotion or deployment.

---

## More detail

- [docs/architecture.md](docs/architecture.md): components, data flow, memory tiers, resilience
  middleware, the approval mechanism, replay, telemetry, and the Nebius API calls.
- `.env.example`: every environment variable with a placeholder and a comment.
- Source: `src/posttrain_agent/` (`agent.py` assembles the graph; `domain.py` holds the
  deterministic rules).

## License

MIT. See [LICENSE](LICENSE).
