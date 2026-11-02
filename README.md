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
- [Live dashboard](#live-dashboard)
- [Extending](#extending)
- [Testing](#testing)
- [Known limitations / honest status](#known-limitations--honest-status)
- [More detail](#more-detail)

---

## How it works

### The loop

The coordinator prompt (`ORCHESTRATOR_PROMPT` in `src/posttrain_agent/agent.py`) asks it to run one
experiment as the sequence below, delegating each step with the `task` tool. The order is an
instruction in the prompt; the code does not enforce it. Some steps may run side by side; see
[Parallel subagents](#parallel-subagents).

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
  |                           wait_for_training, read_training_logs
  |  7 evaluator ............ evaluate_checkpoint      (sealed splits, scores only)
  |  8 champion_manager ..... run_gate                 (deterministic gate)
  v
PROMOTE or REJECT, with an evidence label on every number
(LIVE | PRIOR_VERIFIED_RUN | RULE_VERIFIED | EXPLANATION)
```

### The eight subagent roles

| Role | Gets these tools | Plus memory |
|------|------------------|-------------|
| `benchmark` | `run_benchmark` | recall |
| `failure_analyst` | `read_training_logs` | recall |
| `researcher` | `web_search`, `web_extract`, `hf_search`, `hf_model_info`, `hf_dataset_info` | recall, remember |
| `data_curator` | `generate_game_sft`, `verify_dataset`, `save_dataset`, `inspect_dataset`, `check_chat_template`, `hf_dataset_info` | recall |
| `training_designer` | `validate_hparams`, `estimate_budget`, `web_search`, `hf_model_info` | recall |
| `training_executor` | `launch_training`, `wait_for_training`, `read_training_logs` | recall |
| `evaluator` | `evaluate_checkpoint` | recall |
| `champion_manager` | `run_gate` | recall |

Each role's system prompt is a file in `src/posttrain_agent/prompts/`, prefixed by a shared
preamble in `subagents.py`. Most tools are deterministic Python (`experiments/domain.py`, `tools/dataset.py`,
`tools/budget.py`, `experiments/nebius.py`). `web_*` and `hf_*` are read-only lookups against external services.

`decide_promotion` still exists in the deterministic tool set, but no role is given it: the
`champion_manager` calls `run_gate`, which reads the evaluation ledger itself. `abort_experiment` is
a coordinator tool, not a role tool. `run_gate` requires the regression split; a missing split
returns an error naming it rather than a verdict.

### Parallel subagents

The coordinator can run up to `POSTTRAIN_MAX_PARALLEL` subagents at once (default `2`). When it
issues several independent `task` calls in one message, they run concurrently. A semaphore
middleware (`SubagentConcurrencyMiddleware`, attached to the coordinator only) queues the extra
calls until a slot is free. Queued calls are not dropped.

The prompt names the steps that may overlap:

- Step 3, a researcher literature and prior-art scan, alongside the `failure_analyst` (step 2).
- Step 7, one `evaluator` task for the champion and one for the candidate.

The prompt forbids parallel `training_executor` tasks: one training job per experiment. The code
enforces only the concurrency cap, not which steps are grouped together.

If several subagents pause for approval at the same time, `posttrain pending` lists all of them,
and one `posttrain approve` or `posttrain reject` answers every pending interrupt.

The cap counts subagents only. Ordinary tool calls such as `recall` or `web_search` are not limited.

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
  labelled simulation (see [Known limitations / honest status](#known-limitations--honest-status)).
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
# edit .env; the `posttrain` CLI loads ./.env at startup without overriding variables
# already exported. To load it into your shell as well:
set -a; . ./.env; set +a
```

At minimum set:

| Variable | Why |
|----------|-----|
| `BACKBOARD_API_KEY` | Required for every run. |
| `NEBIUS_API_KEY` | Required for real training and evaluation. Leave empty only for the simulation. |
| `POSTTRAIN_PRICE_PER_MTOK` | **Required to launch unless you opt into unpriced mode.** Nebius publishes no per-token fine-tuning price, so you supply one: USD per 1 million trained tokens. Without a price, `POSTTRAIN_ALLOW_UNPRICED=1` permits launches under a cumulative `POSTTRAIN_MAX_TRAINED_TOKENS` cap instead; otherwise every launch is blocked at the budget stage. |

`TAVILY_API_KEY` and `HF_TOKEN` are optional. Without Tavily the web tools return an
"unavailable" error and the researcher continues without them.

**3. Create the evaluation task files.**

The default suite is `state/suite_v2` (`POSTTRAIN_SUITE_DIR` selects the directory; the code's
fallback is `<STATE_DIR>/suite_v2`). The bundled generator writes four splits from four
deterministic mini-games (tic-tac-toe, Nim, number guessing, rock-paper-scissors): 200 train, 300
held-out, 150 regression and 120 `heldout_phrasing` tasks, disjoint by game state. The phrasing
split uses wording styles that SFT never sees and is reported separately; it is not part of the
gate's main `n`.

```bash
python scripts/make_game_suites.py state/suite_v2
```

`state/suite` is the legacy v1 suite (200 train / 100 held-out / 60 regression). Keep it only for
old results; do not mix v1 data with v2.

**What we fine-tune on.** Supervised fine-tuning (LoRA) of `Qwen/Qwen3-0.6B` on at most **1200**
chat-format rows (game state and tool schema in, the correct tool call out, then a tool result and a
short reply, the format Nebius requires). The rows are generated by code from the game rules
(`generate_game_sft`), so every answer is correct by construction; one row per distinct state, with
phrasing styles rotated across states. No state from any evaluation split, including
`heldout_phrasing`, is ever used for training, and the verifier rejects datasets over 1200 rows
(`POSTTRAIN_MAX_DATASET_ROWS` can only lower the cap).

Generate a fresh dataset against the suite you will evaluate on:

```bash
python scripts/make_sft.py --suite state/suite_v2 --out state/data/game-sft-v2.jsonl
```

`generate_game_sft` inside the agent does the same thing. The old
`state/data/exp1_data_n1200.jsonl` is a v1 dataset and must not be reused with `suite_v2`: it shares
91 states with v2 (15 train, 40 held-out, 18 regression, 18 held-out-phrasing), so the state-overlap
check blocks it at `check_chat_template` and `launch_training`.

**How we prove it beat the base model.** Training runs on Nebius Token Factory, but its serverless
`/chat/completions` serves neither `Qwen/Qwen3-0.6B` nor `ft:` LoRA checkpoints. So both the base
model and the checkpoint are **served on CPU inside a Nebius Sandbox**
(`experiments/sandbox_student.py`): one image with CPU torch, transformers, peft and the base weights
is built once per process; the checkpoint's `adapter_config.json` and `adapter_model.safetensors` are
downloaded on the host (`/fine_tuning/jobs/{id}/checkpoints` -> `/files/{id}/content`) and added as a
small layer, loaded strictly (the provider's tensor keys lack peft's `base_model.model.` prefix) and
merged; greedy generation (`max_new_tokens` 160, thinking off) runs in at most 4 disposable sandboxes
at a time, and the first `<tool_call>{json}</tool_call>` is the model's answer. Expect minutes per
evaluation (the first one also builds the image). `POSTTRAIN_SERVE=auto` keeps serverless for a base
model that `/models` lists. The tool call is then executed against the real game rules by a judge
inside a **Nebius Sandbox** (an isolated VM), and counted only if the move is legal and optimal.

The first champion is the untouched base model. `run_gate` promotes a candidate only if all of these
hold (see [Safety model](#safety-model) for the exact list):

- both results have verified evidence (`LIVE` or `PRIOR_VERIFIED_RUN`) and identical
  `suite`, `split`, `seed`, `judged_by` and `served_by`;
- both report an integer `n > 0`;
- the candidate's `parse_rate` is at least `max(0.90, base parse_rate - 0.02)`;
- the candidate gains at least 10% relative and does not drop more than 0.05 absolute;
- the gain is statistically significant: an exact paired McNemar test when both sides carry
  per-task `verdicts` over the same `tasks_sha256`, otherwise a one-sided two-proportion z-test with
  `z >= 1.645`;
- the regression split is present and passes the same checks (`run_gate` requires it).

The verdict also reports the minimum detectable effect (MDE) at 80% power and flags an underpowered
comparison. At a 0.19 base score and n=100, the MDE is about 0.138 absolute (13.8 points), so a 10%
relative gain (about 1.9 points) is not detectable; the default 300-task held-out split is sized for
that. Results carry `judged_by` (`nebius-sandbox`, or `local-env` if the sandbox was unavailable,
with a note), `served_by` (`nebius-sandbox-cpu` or `nebius-serverless`; the gate never compares the
two) and per-game accuracy. A serving failure returns an `error` with `n_tasks`/`n_generated`, never
a score.

Sandboxes are a Nebius Beta product. Set `NEBIUS_PROJECT_ID` (Token Factory, Project settings), ask
contree@nebius.com for access if the project is not activated, and `pip install ".[sandbox]"`. The
API key never enters the sandbox; only the judge source, the model's tool calls, the generation
script, task prompts and adapter bytes do. Live smoke: `scripts/smoke_sandbox_serving.py`.
`POSTTRAIN_EVAL_JUDGE=sandbox` makes the sandbox mandatory, `local` never uses it.

The task files, if you write your own, are JSONL:

```bash
mkdir -p state/suite_v2
```

| File | Used by | Visible to agents |
|------|---------|-------------------|
| `train.jsonl` | `run_benchmark` | Scores and up to 20 failures, including expected and got values |
| `heldout.jsonl` | `evaluate_checkpoint(split="heldout")` | Scores only |
| `regression.jsonl` | `evaluate_checkpoint(split="regression")` | Scores only |
| `heldout_phrasing.jsonl` | `evaluate_checkpoint(split="heldout_phrasing")` | Scores only; informational, never gates |

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
| `state` | no | Canonical game state, e.g. `nim:18`. With it, a call is judged by executing it against the game rules; without it, by exact match to `expected`. |

Scoring: the model is called with `temperature: 0`. A task scores 1 only if the model's **first**
tool call has exactly this `name` and `arguments` (a parsed JSON object compared for equality).
Otherwise it scores 0. At most 1000 tasks per split are read. Use the same tool names the dataset
verifier allows: the four game tools (`place_mark`, `take_stones`, `guess_number`, `play_move`) and
the six ops tools (`get_logs`, `inspect_service`, `read_config`, `edit_config`, `restart_service`,
`run_healthcheck`) so that the behaviour you train and the behaviour you measure are the same.

The held-out, regression and phrasing files are sealed. Agents receive only
`{evidence, score, n}` for them. Task contents never leave `experiments/nebius.py`.

**4. Start a run.**

```bash
uv run posttrain run "Reduce tool-call failures of the base model on API incidents" --thread demo
```

The command prints a JSON status. Exit code `3` means the run is paused for approval.

**5. Review the approval request.**

```bash
uv run posttrain pending --thread demo
```

The `description` field shows the base model, dataset ref, hyperparameters, a cost line and the
budget estimate (`estimated_trained_tokens`, `estimated_cost_usd`, `within_budget`, `spent_usd`,
`cap_usd`; in unpriced mode the token cap instead). The description does not include the dataset
hash. The hash is returned in the tool result after launch.

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
`RULE_VERIFIED` marks dataset rows whose answer was recomputed from the game rule; it is valid
training provenance, not measured model evidence, and the promotion gate does not accept it.

---

## CLI reference

Entry point: `posttrain` (`posttrain_agent.cli:main`). Every command except `telemetry`, `a2a` and
`note` needs `BACKBOARD_API_KEY`, or `POSTTRAIN_CASSETTE_MODE=replay`.

| Command | Arguments | What it does |
|---------|-----------|--------------|
| `run` | `GOAL` `--thread ID` (default `default`) `--auto-resume` `--max-resumes N` | Starts the run on the thread, or resumes it if the thread already has state. `--auto-resume` retries a `FAILED_TRANSIENT` outage up to `N` times (default 8). |
| `resume` | `--thread ID` | Continues a thread from its last checkpoint. |
| `pending` | `--thread ID` | Prints the actions waiting for a human decision, from every paused subagent. |
| `approve` | `--thread ID` | Approves every pending action, across all paused subagents, and continues. Errors if nothing is pending. |
| `reject` | `--thread ID` `-m/--message TEXT` | Rejects every pending action, across all paused subagents. The message is shown to the agent. |
| `note` | `--thread ID` `TEXT` | Leaves an operator note the coordinator reads before its next model call. Needs no key or network. |
| `history` | `--thread ID` | Lists checkpoints, newest first: id, step, source, next node, message count. |
| `fork` | `--thread ID` `CHECKPOINT_ID` | Re-executes from a past checkpoint. Original history is kept. Prints the final message. |
| `runs` | none | Prints the run ledger across threads (up to 100 entries). |
| `a2a` | `--thread ID` `--out FILE` | Exports the run's agent-to-agent conversation from the event journal. Needs no key or network. |
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
`src/posttrain_agent/config.py`. The `posttrain` CLI loads `./.env` at startup without overriding
already-exported variables; library callers get only the process environment. Use `.env.example`
as a template.

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
| `POSTTRAIN_MAX_PARALLEL` | `2` | Maximum subagents running at once. Extra `task` calls queue. Values below 1 are read as 1. |
| `POSTTRAIN_STATE_DIR` | `state` | Root of all durable state. |
| `POSTTRAIN_CASSETTE` | unset | Path to a cassette JSONL file. Enables recording or replay. |
| `POSTTRAIN_CASSETTE_MODE` | `auto` | `auto` replays hits and records misses; `record` always calls live; `replay` never calls and fails on a miss. |
| `NEBIUS_API_KEY` | `""` | Nebius key. Empty selects the labelled simulation backend. |
| `NEBIUS_BASE_URL` | `https://api.tokenfactory.nebius.com/v1` | Nebius OpenAI-compatible base URL. |
| `POSTTRAIN_BASE_MODEL` | `Qwen/Qwen3-0.6B` | Base model for fine-tuning and for the `base` checkpoint alias. |
| `POSTTRAIN_PRICE_PER_MTOK` | unset | USD per 1M trained tokens. **Required to launch unless `POSTTRAIN_ALLOW_UNPRICED=1`.** |
| `POSTTRAIN_ALLOW_UNPRICED` | `0` | `1`/`true`/`yes` allows launches without a price, capped by cumulative trained tokens instead of USD. |
| `POSTTRAIN_MAX_TRAINED_TOKENS` | `10000000` | Cumulative trained-token cap when unpriced (this job plus earlier ones). |
| `POSTTRAIN_MAX_COST_USD` | `25` | Cumulative estimated spend cap across all launched jobs when priced. |
| `POSTTRAIN_DATA_DIR` | `<STATE_DIR>/data` | Where verified datasets are stored. |
| `POSTTRAIN_SUITE_DIR` | `<STATE_DIR>/suite_v2` | Directory containing `train.jsonl`, `heldout.jsonl`, `regression.jsonl`, `heldout_phrasing.jsonl`. `state/suite` is the legacy v1 suite. |
| `POSTTRAIN_MAX_DATASET_ROWS` | `1200` | SFT row cap. Can only lower the hard 1200-row limit. |
| `POSTTRAIN_REQUIRE_APPROVAL` | `1` | Set to `0`, `false` or `no` to disable the approval gate. Only do this for unattended runs you accept. |
| `POSTTRAIN_SEED` | `7` | Seed for benchmark and evaluation requests and for Nebius jobs. |
| `POSTTRAIN_SERVE` | `auto` | `auto` serves the base model serverlessly if `/models` lists it, else on sandbox CPU; `serverless`/`sandbox` force one path. |
| `POSTTRAIN_EVAL_JUDGE` | `auto` | `auto` uses the sandbox judge when configured, else local rules; `sandbox` requires it; `local` never uses it. |
| `NEBIUS_PROJECT_ID` | `""` | Nebius project for Sandboxes (Beta), used for serving and judging. |
| `POSTTRAIN_SANDBOX_IMAGE` | `python:3.12-slim` | Base image for sandbox jobs. |
| `POSTTRAIN_TOKENIZER_DIR` | unset | Directory with the base model's `tokenizer.json`/`tokenizer_config.json`; otherwise the Hugging Face cache is searched. |
| `POSTTRAIN_SUBAGENT_MAX_CALLS` / `POSTTRAIN_SUBAGENT_MAX_TOKENS` | `12` / `150000` | Per-subagent model-call and token caps. |
| `POSTTRAIN_COORDINATOR_MAX_CALLS` / `POSTTRAIN_COORDINATOR_MAX_TOKENS` | `80` / `1000000` | Per-coordinator model-call and token caps. |
| `POSTTRAIN_RUN_MAX_TOKENS` | `2000000` | Whole-run token cap, persisted in `state/run_tokens.json`. |
| `POSTTRAIN_MODEL_<ROLE>` | unset | Per-role `provider:model` override (for example `POSTTRAIN_MODEL_EVALUATOR=anthropic:claude-sonnet-4-5`). |
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
- Every tool call's `function.name` is in the allow-list: the four game tools (`place_mark`,
  `take_stones`, `guess_number`, `play_move`) and the six ops tools (`get_logs`,
  `inspect_service`, `read_config`, `edit_config`, `restart_service`, `run_healthcheck`).
- Tool arguments parse to a JSON object (a string that parses, or an object).
- The last message is from the `assistant`. It is the training target.
- No row contains `heldout://`, `sealed://`, `heldout/` or `sealed/`.
- `metadata.evidence`, if present, is `LIVE`, `PRIOR_VERIFIED_RUN`, `RULE_VERIFIED` or
  `EXPLANATION`.
- If `metadata.state` names a game state, the tool call is recomputed from the game rule and must
  match.
- No two rows are identical (SHA-256 of the canonical JSON).
- At least one row.

Warnings, not errors: rows labelled `EXPLANATION`, and datasets with no evidence labels at all.
`RULE_VERIFIED` is the normal label for rows produced by `generate_game_sft`: it means the answer
was recomputed by code, not measured from a model.

**What the verifier does not check.** For ops-style rows it does not replay tool calls in an
environment, so it cannot know whether a call is correct. It does not check that a `tool` message
follows the matching tool call, that `id` values are unique or linked, or that `source_task_id`
refers to a real task. It does not check licence or provenance of external data. `verify_rows`
does not compare game states against the evaluation suite; `check_chat_template` and
`launch_training` run a separate state-overlap check and reject any training state that appears in
any eval split, including `heldout_phrasing`. Token length is estimated, not enforced (see
[Budget model](#budget-model)). Verification is structural, policy-based and rule-based for game
rows; it is not a full environment replay.

`save_rows` writes nothing unless verification passes. `check_chat_template` must pass for the
current file SHA-256, and `launch_training` re-verifies the dataset, re-checks suite overlap and
requires that template check before submitting; it reports the SHA-256 in the result.

---

## Hyperparameter search space

Source: `HPARAM_SPACE` and `OPTIONAL_HPARAMS` in `src/posttrain_agent/experiments/domain.py`. Bounds are
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
non-integer values for integer keys. With a `dataset_ref` it also estimates optimizer steps and
blocks fewer than about 50 (warns below 100). `launch_training` re-runs both checks, so a bad
configuration cannot reach Nebius.

---

## Budget model

Source: `src/posttrain_agent/tools/budget.py`.

1. **Tokens per row** comes from the base model's tokenizer and chat template, with the tools schema
   included (`experiments/pretokenize.py`). It uses the cached Qwen3 tokenizer, or
   `POSTTRAIN_TOKENIZER_DIR`. If the optional `tokenize` extra or tokenizer files are unavailable it
   falls back to `4 * max(1, chars // 4)` (the old estimate undercounted the tools schema and chat
   template). The estimate reports which `basis` it used.
2. **Trained tokens** is `sum(min(row_tokens, context_length)) * n_epochs`. Defaults are
   `n_epochs = 3` and `context_length = 8192` when absent. Rows longer than the context are
   counted at the context length and reported in `rows_over_context`.
3. **Estimated cost** is `trained_tokens / 1e6 * POSTTRAIN_PRICE_PER_MTOK`.
4. **Spent** is the sum of `estimated_cost_usd` in `state/spend.jsonl`. It is recorded after each
   successful submission that returns a provider job ID. Stub launches are recorded too. When the
   provider later reports `trained_tokens`, a `reconcile` record replaces the estimate.
5. **Within budget** is `spent + cost <= POSTTRAIN_MAX_COST_USD` when priced. In unpriced mode it is
   `spent_trained_tokens + trained_tokens <= POSTTRAIN_MAX_TRAINED_TOKENS`.

Without `POSTTRAIN_PRICE_PER_MTOK` the estimate reports token counts only, with
`estimated_cost_usd: null`. If `POSTTRAIN_ALLOW_UNPRICED=1`, `within_budget` is computed against the
cumulative token cap. Otherwise `within_budget` is `null` and `launch_training` returns `BLOCKED` at
the `budget` stage. This is deliberate: the system fails closed.

The estimate is an estimate. Nebius bills by compute, and the spend ledger records estimates (or
provider-reported token counts after reconciliation), not invoices. Reconcile against your Nebius
billing before relying on the cap.

---

## Safety model

- **Sealed held-out data.** `run_benchmark` accepts only the `train` split. `evaluate_checkpoint`
  accepts `heldout`, `regression` or `heldout_phrasing`, and returns only scores and metadata (no
  task contents). The phrasing split is informational and never gates. Web queries containing
  `heldout://` or `sealed://` are refused, datasets with sealed markers are refused, and
  `check_chat_template` / `launch_training` block any dataset state that appears in any eval split.
- **Fail-closed promotion gate.** `run_gate` reads the evaluation ledger and calls the deterministic
  `promotion_gate` (`experiments/gate.py`). It returns `REJECT` unless all of the following hold:
  - both results have verified evidence (`LIVE` or `PRIOR_VERIFIED_RUN`);
  - `suite`, `split`, `seed`, `judged_by` and `served_by` match between champion and candidate;
  - both report an integer `n > 0`;
  - the candidate's `parse_rate` is at least `max(0.90, base parse_rate - 0.02)`; a missing
    candidate parse rate rejects;
  - the relative improvement is at least 10% (if the champion score is 0, any positive candidate
    score passes this check) and no absolute drop exceeds 0.05;
  - the gain is significant: an exact paired McNemar/sign test when both results carry per-task
    `verdicts` and the same `tasks_sha256`, otherwise a one-sided two-proportion z-test with
    `z >= 1.645`;
  - the regression split is supplied for both sides and passes the same validity, parse-rate and
    `<= 0.05` drop checks (`run_gate` passes `require_regression=True`);
  - differing `tasks_sha256` values reject as "not computed on the same tasks", on the held-out
    split and on the regression split.
  `decide_promotion` is still available to code, takes caller-supplied scores, and now also
  requires both regression results (otherwise REJECT); no agent role gets it, and only `run_gate`
  records the verdict the exit gate accepts. Only deterministic code decides; no agent can set a decision. The gate does
  not itself require the split to be `heldout`; it requires the two inputs to match.
  The verdict also reports `mde_80pct_power`, `power_note` and a `power` block (observed gain,
  adequacy, recommended n=300) at 80% power and one-sided 5%.
- **Evidence labels.** `LIVE` means provider-measured. `PRIOR_VERIFIED_RUN` is accepted by the gate,
  but no code path produces it. `RULE_VERIFIED` marks a dataset row whose answer was recomputed by
  code from the game rule; the dataset verifier accepts it, but the promotion gate does not.
  `EXPLANATION` means simulated or inferred and can never be promoted. Subagent prompts require
  labels to be copied unchanged from tool results.
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
  spend.jsonl            launch estimates and reconciled trained tokens per job
  eval_ledger.jsonl      every held-out/regression evaluation result (run_gate reads this)
  run_state.jsonl        launch intent, job status, checkpoint and trained tokens
  gate_log.jsonl         one record per run_gate verdict
  aborts.jsonl           one record per abort_experiment
  notes.jsonl            operator notes and read cursors
  events.sqlite          redacted event journal for the dashboard/API
  run_tokens.json        persisted whole-run token total
  data/                  POSTTRAIN_DATA_DIR: verified <name>.jsonl datasets
  suite_v2/              current suite: train, heldout, regression, heldout_phrasing
  suite/                 legacy v1 suite: train, heldout, regression
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

There is no standalone CLI command to poll a provider job. The `training_executor` role calls
`wait_for_training`, which blocks inside the tool until the job is terminal or its timeout expires;
on timeout it returns `timed_out: true` and can be called again with the same job id. Job events
come from `read_training_logs`. `get_training_status` is available in the deterministic tool set but
is not one of the role's assigned tools.

---

## Extending

### Add another training or evaluation provider

Implement the `ExperimentBackend` protocol in `src/posttrain_agent/experiments/domain.py`:

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
- `evaluate` must return `score`, `evidence`, `n`, `parse_rate`, and echo `suite`, `split`, `seed`,
  `judged_by` and `served_by`, because the gate compares them. Per-task `verdicts` and
  `tasks_sha256` enable the paired exact test; without them the gate uses the unpaired z-test.
- Return `"evidence": "LIVE"` only for provider-measured values. Use `EXPLANATION` otherwise.
- Keep held-out task contents out of the return value.

Pass your backend to `build_agent(settings, runtime, experiments=...)`. The CLI picks the backend
with `default_backend` in `src/posttrain_agent/agent.py`, so change that function, or add a flag to
the CLI.

### Add a tool

1. Write a function with a docstring. The docstring is the tool description the model sees.
2. Expose it. For a deterministic tool, add it to the dictionary returned by `make_tools` in
   `experiments/domain.py`. For a web-style tool, add it to the dict returned by a factory, and merge that dict
   into `tools` in `build_agent`.
3. Give it to the roles that need it by adding its name to the `ROLES` tuple in `subagents.py`. Or
   add it to the coordinator's `tools` list in `build_agent`.
4. Add a test. Keep any tool that changes state behind the approval gate, or make it read-only.

---

## Live dashboard

The dashboard is the Next.js app in `web/` (see [web/README.md](web/README.md)); it is not part of
the Python package. Every run appends redacted events (statuses, scores, job progress, budget,
approvals) to a SQLite journal under the state directory. A read-only API streams them to the
dashboard over Server-Sent Events. Failures are journaled as exception type names only; the journal
never holds raw model prompts or completions, tool arguments or secrets.

**Agent-to-agent conversation.** The one exception is the coordinator/subagent traffic: each
`task` delegation is stored as two `a2a.message` events, the coordinator's brief and the
subagent's reply, as JSON in the same journal and shown in the dashboard's A2A panel. Their text is
scrubbed of known API keys, bearer tokens and `key=value` credentials, and cut to 2000 characters
(`text_len` keeps the original length). Briefs are written by the coordinator and may quote task
details, so treat the journal as sensitive. Export a run's full conversation as JSON:

```bash
posttrain a2a --thread t1 --out a2a-t1.json     # from the journal; no API key needed
curl "http://127.0.0.1:8787/v1/runs/t1/a2a?token=$READ_TOKEN"   # same, over the API
```

```bash
cp .env.example .env                  # set READ_TOKEN (required), ALLOWED_ORIGIN, UID/GID
mkdir -p state                        # so Docker does not create ./state root-owned
docker compose up --build             # api only: 127.0.0.1:8787, state in ./state
docker compose --profile tunnel up    # also the named Cloudflare tunnel (needs TUNNEL_TOKEN)
docker compose --profile quick up     # or a temporary trycloudflare.com URL, no token
```

### Why the API is containerized and the agent is sandboxed

They guard different things.

- **API (internet-facing).** The tunnel exposes it publicly, so it holds only `READ_TOKEN`, opens the
  journal read-only and has no model or search keys.
- **Agent (untrusted input).** It reads web and Hugging Face content a third party can write, so a
  prompt injection is possible, and it holds the Backboard, Nebius and Tavily keys. Today its files
  are virtual and it has no shell tool, so it cannot touch the host. That is a property of the code,
  not an enforced boundary. The container enforces it and makes a future code-execution tool safer.

```bash
docker compose --profile agent run --rm agent run "improve the checkpoint" --thread t1
docker compose --profile agent run --rm agent approve --thread t1
```

The `agent` service runs as a non-root user with a read-only root filesystem, no capabilities,
`no-new-privileges`, PID and memory limits, and only `./state` mounted. It has its own network, so it
shares nothing with the API or tunnel. Compose cannot restrict which hosts it may reach; use an
egress proxy if you need that.

- On a fresh checkout the API answers 503 (`journal unavailable`) until the first run writes
  `state/events.sqlite`; the tunnels start regardless. The dashboard shows RECONNECTING until then.
- The api runs as `${UID}:${GID}` so it can open the host-owned `./state` files. bash does not
  export `UID`, so either set `UID`/`GID` in `.env` or run `export UID GID=$(id -g)` first.
- `./state` is mounted read-write on purpose: a WAL reader needs a writable `-shm` file. The API
  opens the journal read-only and never writes events.
- `GET /v1/health` needs no token (used by the Docker healthcheck); it returns 503 when the journal
  cannot be read. Every other route needs `?token=<READ_TOKEN>`.
- `READ_TOKEN` is required even in replay mode. Set `REPLAY_FILE=/replay/sample-run.jsonl` in
  `.env` to serve a recorded journal instead of the live one (`./web/public` is mounted at
  `/replay`, read-only). Outside compose, use the file's own path.
- cloudflared runs on the `latest` tag; pin a version for production.
- `web/public/sample-run.jsonl` is a **simulated sample**: a stub-backed fake run with approval
  included, every score and job carrying evidence `EXPLANATION`. It is never live data. Regenerate
  it with `python scripts/make_sample_journal.py --out web/public/sample-run.jsonl`.

---

## Testing

```bash
uv sync --extra dev
.venv/bin/python -m pytest
```

The pytest config in `pyproject.toml` uses `asyncio_mode = "auto"` and adds `tests/` and `src/` to
the path. Lint with `.venv/bin/ruff check src tests` (line length 100); the command is useful, but
the branch is not lint-clean at the moment.

The suite currently collects 300 tests in this checkout (the audit recorded 276 before later test
additions).

| Test files | Covers |
|------------|--------|
| `test_approval.py`, `test_e2e.py`, `test_exit_gate_e2e.py` | Approval pause/resume, SQLite persistence, crash recovery, history/fork, termination rule |
| `test_backboard_model.py`, `test_resilience.py` | Backboard chat model, cassette, transient classification, retries and fallback |
| `test_data_budget.py`, `test_sft_and_gate.py`, `test_gate_power.py` | Dataset verifier, budget/tokenizer math, unpriced cap, promotion gate and power reporting |
| `test_suite_integrity.py` | v1/v2 suite boundary, the 91-state overlap, contamination blocking, phrasing exclusion |
| `test_domain.py`, `test_nebius_regressions.py`, `test_eval_resilience.py`, `test_eval_honesty.py` | Hyperparameters, sealed splits, launch gating, Nebius scoring/retries and error honesty |
| `test_parallel.py`, `test_budget_guard.py`, `test_role_models.py` | Concurrency cap, per-agent budgets, role model overrides |
| `test_hf_nebius_telemetry.py`, `test_web.py`, `test_token_key_fallback.py` | External tool request shapes, telemetry redaction, token-key fallback |
| `test_api.py`, `test_events.py`, `test_journal_wiring.py`, `test_a2a.py`, `test_fold.py` | Journal/API/SSE, dashboard folds and A2A export |
| `test_sandbox_eval.py`, `test_sandbox_student.py`, `test_game_suites.py`, `test_prompts.py` | Sandbox judge/serving, suite generation, prompt contracts |

Tests use a scripted fake Backboard client (`tests/fakes.py`) and `httpx.MockTransport` for Nebius,
Tavily and Hugging Face. No test calls a live Backboard, Nebius, Tavily or Hugging Face service; the
request shapes are exercised against mocks, not the live APIs.

---

## Known limitations / honest status

Read this before relying on the system.

- **The last live run did not promote anything.** It ended without a gate verdict: the candidate
  held-out evaluation timed out uploading the adapter (`ApiTimeoutError` on the Sandbox file
  upload), the candidate regression scored 0.0, and no candidate held-out score existed. The base
  scored 0.19 on held-out (n=100) and 0.1833 on regression (n=60). Do not present this as a
  successful promotion.
- **The gate's 10% relative threshold is below the MDE at n=100.** At a 0.19 base and n=100, the
  reported 80%-power MDE is about 0.138 absolute (13.8 points); a 10% relative gain is only about
  1.9 points, so it cannot clear the significance test. Use the 300-task held-out suite for a real
  comparison and read `mde_80pct_power` and the `power` block.
- **Tests do not exercise live services.** The suite uses a fake Backboard client and mocked HTTP
  transports. No live Backboard, Nebius, Tavily or Hugging Face call is made by the tests.
- **The old v1 dataset is unsafe on suite_v2.** `state/data/exp1_data_n1200.jsonl` shares 91 states
  with `suite_v2` (15 train, 40 held-out, 18 regression, 18 held-out-phrasing). The overlap check
  blocks it; generate fresh data with `generate_game_sft` or `scripts/make_sft.py`.
- **Candidate serving is not proven end to end.** The last live run reached adapter upload but timed
  out there. A fine-tuned checkpoint has not been fully evaluated through the gate.
- **Token estimates can be approximate.** The tokenizer/chat-template estimate is used when the
  optional tokenizer is available; otherwise the code falls back to a scaled chars/4 heuristic.
  Nebius bills by compute, and spend records are estimates unless reconciled with reported trained
  tokens.
- **No live price.** Nebius publishes no per-token fine-tuning price, so `POSTTRAIN_PRICE_PER_MTOK`
  is operator-supplied. In unpriced mode the dollar cost is unknown; only the token cap applies.
- **Simulation without a key.** With no `NEBIUS_API_KEY`, every benchmark, training and evaluation
  result comes from `LocalStubBackend`. These results are labelled `EXPLANATION`, and the gate
  always rejects them.
- **Subagent outputs are not schema-checked.** The OUTPUT CONTRACT blocks in the prompt files
  describe JSON shapes. The code does not validate them, so a malformed hand-off is caught only
  by the next tool's own checks. Treat the prompts as guidance. (The prompts no longer ask for
  inputs that no tool supplies; `dataset_sha256` is passed from `data_curator` through to
  `launch_training`.)
- **No champion store or deployment.** `run_gate` returns a decision. The code does not record a
  champion, deploy a checkpoint, or pick the next candidate.
- **Approval is bound to the dataset hash, but only as strongly as the human reads it.** The approval
  prompt shows the SHA-256 of the file as it is on disk at that moment and flags `NOT BOUND` or
  `MISMATCH` when the call's `dataset_sha256` is missing or differs. `launch_training` refuses to
  upload (`stage: approval_binding`) when `dataset_sha256` is missing (approval on) or differs from
  the file's current hash, so an edit after approval cannot be uploaded. With
  `POSTTRAIN_REQUIRE_APPROVAL=0` no hash is demanded.
- **Crash between submit and spend record.** See [Recovering from crashes](#recovering-from-crashes).
- **Replay re-executes tools.** See [Replaying offline with the cassette](#replaying-offline-with-the-cassette).
- **Approval can be turned off.** `POSTTRAIN_REQUIRE_APPROVAL=0` removes the only human check on
  spend. Use it only for runs you accept.
- **Web and Hugging Face lookups are external and not cached.** They are not recorded in the
  cassette, so a replay makes them again.
- **Not included:** no multi-process scheduling, no retry of a rejected launch, no cancellation of a
  running provider job, and no automatic promotion or deployment.

---

## More detail

- [docs/architecture.md](docs/architecture.md): components, data flow, memory tiers, resilience
  middleware, the approval mechanism, replay, telemetry, and the Nebius API calls.
- `.env.example`: every environment variable with a placeholder and a comment.
- Source: `src/posttrain_agent/` (`agent.py` assembles the graph; `experiments/domain.py` holds the
  deterministic rules).

## License

MIT. See [LICENSE](LICENSE).
