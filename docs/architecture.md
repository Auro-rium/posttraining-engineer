# Architecture

posttrain-agent is a deepagents coordinator running on LangGraph. Its LLM calls and semantic
memory go through Backboard, via a custom LangChain chat model. Deterministic Python code owns
every decision that affects promotion, budget, and the held-out boundary. This document describes
the code as it is; see the README "Known limitations / honest status" section for what is
unverified.

## Components

| Module | Responsibility |
|--------|----------------|
| `agent.py` | `build_agent` assembles the deep agent: model, middleware, tools, subagents, memory, checkpointer, store, and the approval interrupt. Holds `ORCHESTRATOR_PROMPT` and `default_backend`. |
| `runtime/concurrency.py` | `SubagentConcurrencyMiddleware`: semaphore capping concurrent `task` calls at `POSTTRAIN_MAX_PARALLEL`. Attached to the coordinator only. |
| `runtime/budget_guard.py` | `BudgetGuard` middleware (per-invocation model-call and token caps, wrap-up warning at 75%, clean `BUDGET_EXHAUSTED` stop) and `RunTokenLedger` (persisted per-run token total, default 2M). On the coordinator and every subagent. |
| `cli.py` | `posttrain` command. Opens the runtime, builds the agent, and dispatches to `runtime/replay.py` and `observability/telemetry.py`. |
| `config.py` | `Settings` dataclass, `Settings.from_env()` and `load_dotenv`. The CLI loads `./.env`; `pretokenize` also reads `POSTTRAIN_TOKENIZER_DIR`. |
| `llm/backboard_model.py` | `ChatBackboard` (LangChain chat model over the Backboard API), `Cassette` (record/replay store), and transcript rendering. |
| `llm/resilience.py` | `is_transient`, `build_model`, and `build_middleware` (fallback, retry, tool retry). |
| `llm/memory.py` | `ensure_assistant` (persistent Backboard assistant) and the `remember` / `recall` tools. |
| `runtime/persistence.py` | `open_runtime`: LangGraph SQLite checkpointer, SQLite store, and the composite file backend with `/memories/` routed to the store. Seeds `/memories/AGENTS.md`. |
| `subagents.py` | `ROLES`: the eight role contracts (prompt file, description, tool names, memory access). `build_subagents` turns them into deepagents `SubAgent` specs. |
| `experiments/domain.py` | `Protocol ExperimentBackend`, `LocalStubBackend`, `validate_hparams`, `make_tools` (the deterministic tool functions), `run_gate`, and `abort_experiment`. |
| `experiments/gate.py` | `promotion_gate`: evidence, matching, parse-rate, relative-gain, regression and significance rules, plus MDE/power reporting. Re-exported by `domain.py`. |
| `experiments/games.py`, `experiments/game_env.py` | Four deterministic mini-games, rule-correct SFT generation and the local game judge. |
| `experiments/ledger.py` | Append-only `eval_ledger.jsonl`: every evaluation attempt, status and result. |
| `experiments/runstate.py` | Append-only run state, gate log, aborts and operator notes. |
| `experiments/pretokenize.py` | Qwen3 chat-template rendering and pretokenization; only the tool call is supervised. Optional `tokenize` extra. |
| `experiments/sandbox.py`, `experiments/sandbox_student.py` | Nebius Sandbox judge and CPU serving of the base model and LoRA adapters. |
| `tools/dataset.py` | Chat-format SFT verification (`verify_rows`), safe file references (`resolve`), state-overlap checking, inspection, load and save. |
| `tools/budget.py` | Tokenizer-based token estimate with a fallback heuristic, cost estimate, `spend.jsonl` ledger, reconciliation and `within_budget`. |
| `experiments/nebius.py` | `NebiusBackend`: fine-tuning jobs and deterministic scoring over the Nebius REST API, serverless or sandbox CPU. |
| `tools/web.py` | Tavily `web_search` and `web_extract`. |
| `tools/hf.py` | Hugging Face Hub `hf_search`, `hf_model_info`, `hf_dataset_info`. |
| `runtime/replay.py` | `run_with_recovery`, `decide`, `pending_actions`, `history`, `fork`, `ledger`; the run ledger. |
| `runtime/exit_gate.py` | `ExitGate` termination rule (a launched candidate needs a gate verdict or an abort) and `NotesInjector` for operator notes. |
| `observability/telemetry.py` | `TelemetryHandler` (LangChain callback that writes JSONL events) and `summarize` / `telemetry_tool`. |
| `observability/events.py`, `journal_handler.py`, `fold.py` | Redacted event journal, LangChain journal handler, and folds for snapshots and A2A export. |
| `api.py` | Read-only FastAPI service (`posttrain-api`) streaming `events.sqlite` over SSE; serves the `web/` dashboard. |
| `prompts/*.md` | Per-role system prompts, appended to the shared preamble in `subagents.py`. |

## Data flow

```
 posttrain run "<goal>" --thread T
        |
        v
 +--------------------------------------------------------------------------+
 | cli.py -> open_runtime(state/)                                           |
 |   AsyncSqliteSaver  -> state/checkpoints.sqlite  (per-step thread state) |
 |   AsyncSqliteStore  -> state/store.sqlite        (/memories/, run ledger)|
 +--------------------------------------------------------------------------+
        |
        v
 build_agent: ensure_assistant -> state/backboard.json (or BACKBOARD_ASSISTANT_ID)
        |
        v
 +---------------------------- deep agent graph ----------------------------+
 |  coordinator  --task-->  subagent (benchmark, ..., champion_manager)     |
 |      |                        |                                          |
 |      | ChatBackboard          | ChatBackboard  (same middleware stack)   |
 |      v                        v                                          |
 |  Backboard API            Backboard API                                  |
 |                                                                          |
 |  deterministic tools (make_tools, verify, budget, nebius, web, hf)       |
 |     -> data/*.jsonl, spend.jsonl, Nebius REST, Tavily, Hugging Face      |
 |                                                                          |
 |  launch_training  --interrupt--> AWAITING_APPROVAL (checkpoint saved)    |
 +--------------------------------------------------------------------------+
        |
        v
 TelemetryHandler -> state/telemetry.jsonl  (names, latencies, tokens; no prompts)
        |
        v
 replay.run_with_recovery -> RunReport -> cli prints JSON, exit code 0 | 1 | 3
 replay.run_with_recovery also appends a ledger row to store.sqlite
```

The approval resume follows a separate path: `posttrain approve` loads the same checkpoint in a new
process, then `replay.decide` resumes the graph with a `Command(resume=...)`.

Deterministic tools also write their own durable records outside the graph: `evaluate_checkpoint`
appends to `eval_ledger.jsonl`, `launch_training` / `wait_for_training` write `run_state.jsonl`,
`run_gate` writes `gate_log.jsonl`, and `abort_experiment` writes `aborts.jsonl`. The journal handler
writes redacted events to `events.sqlite`; `api.py` serves that read-only and `web/` renders it.

## Suites and datasets

The current suite is `state/suite_v2` (`POSTTRAIN_SUITE_DIR`; the code fallback is
`<STATE_DIR>/suite_v2`):

| Split | Tasks | Role |
|-------|-------|------|
| `train` | 200 | benchmark only; failures are visible |
| `heldout` | 300 | main gate comparison |
| `regression` | 150 | forgetting check; required by `run_gate` |
| `heldout_phrasing` | 120 | unseen wording, informational only |

`state/suite` is the legacy v1 suite (200/100/60) and has no `heldout_phrasing` file. Splits are
disjoint by canonical game state; the generator (`scripts/make_game_suites.py`) carries used states
forward. `generate_game_sft` and `scripts/make_sft.py` exclude every state in all four splits,
including the phrasing split.

The v1 dataset `state/data/exp1_data_n1200.jsonl` is **not** safe with `suite_v2`: 91 of its states
appear in v2 (15 train, 40 heldout, 18 regression, 18 heldout_phrasing). `check_suite_overlap`
rejects it, and `check_chat_template` and `launch_training` block it with `stage: "contamination"`.
Generate fresh data against the configured suite (`generate_game_sft` in the data-curator role, or
`python scripts/make_sft.py --suite state/suite_v2 --out state/data/game-sft-v2.jsonl`).

## Memory tiers

| Tier | Storage | Scope | Written by | Read by |
|------|---------|-------|------------|---------|
| 1. Thread state | LangGraph checkpoints in `checkpoints.sqlite` | One thread. Every step is a checkpoint. | LangGraph | `history`, `fork`, `resume`, crash recovery |
| 2. File memory | `/memories/` routed to `store.sqlite` (namespace `posttrain/memories`) | Across all threads | Agents writing `/memories/...` through the file backend | Seeded into the prompt as `AGENTS.md` (`memory=[SEED_MEMORY]`) |
| 2a. Scratch files | `StateBackend` inside graph state (no path prefix) | One thread, inside checkpoints | Agents (for example under `/work/`) | Agents in the same thread |
| 3. Semantic memory | Backboard assistant memories | Across all threads, per assistant | `remember` (researcher and coordinator) | `recall` (all roles, and the coordinator) |

The assistant id is fixed for the whole installation. Precedence: `BACKBOARD_ASSISTANT_ID`, then
`state/backboard.json`, then a newly created assistant (name `posttrain-agent`), which is written to
`backboard.json`.

## ChatBackboard: stateless per call

`ChatBackboard` is a `BaseChatModel` whose `_call` sends one `send_message` per model turn.

- **No Backboard thread per graph thread.** Each call omits `thread_id`, so Backboard opens a fresh
  thread. LangGraph owns conversation state. Mapping graph threads to Backboard threads would
  diverge from the checkpoint on the first replay, fork, or isolated subagent. The fixed
  `assistant_id` is the only long-lived Backboard object, so assistant memory accumulates.
- **Transcript rendering.** `render_messages` splits the LangChain transcript into a system prompt
  and one content string. Human, assistant, assistant tool-call and tool-result entries are written
  as labelled blocks (`[user]`, `[assistant]`, `[assistant tool_call id=... name=...]`,
  `[tool_result id=... name=...]`). A single human message is sent verbatim.
- **Tools re-sent every turn.** `bind_tools` converts tools to OpenAI-style schemas and stores them.
  Each call passes them as `tools=`. Backboard does not keep per-turn tool state.
- **Tool results go back as text**, not as a submitted tool output on a pending run. The model sees
  them in the rendered transcript.
- **Tool choice is ignored**, because Backboard has no such control.
- **Failure states.** A Backboard response with status `failed`, `error`, `cancelled` or `expired`
  raises `BackboardAPIError` with no status code. An empty reply with no text and no tool calls
  raises `EmptyResponseError`, a `BackboardAPIError` subclass.
- **Malformed tool arguments** (not a JSON object) become `invalid_tool_calls` on the AIMessage
  rather than exceptions.
- **Sync entry point.** `_generate` runs `_agenerate` through `_run_sync`, which uses a worker thread
  when an event loop is already running.
- **Token usage** is read from the last message of the Backboard response and reported as
  `usage_metadata`.

## Resilience middleware

`build_middleware` returns the stack, outermost first:

```
ModelFallbackMiddleware(fallback models...)      only when POSTTRAIN_FALLBACK_MODELS is set
  -> ModelRetryMiddleware(max_retries=POSTTRAIN_MODEL_RETRIES, retry_on=is_transient,
                          on_failure="error", initial_delay=1.0, max_delay=20.0)
    -> ToolRetryMiddleware(max_retries=1, on_failure="continue")
```

The same list is applied to the coordinator and to every subagent.

Because the fallback middleware is outermost, each fallback model gets its own retry budget before
the next model is tried. A non-transient error is not retried, but it still falls through to the next
fallback (the test suite checks this).

### Transient-error rules (`is_transient`)

| Exception | Transient? | Reason |
|-----------|-----------|--------|
| `CassetteMiss` | no | Replay-only: a missing recording is a bug in the replay setup. |
| `BackboardValidationError` | no | The request is wrong. |
| `BackboardNotFoundError` | no | Retrying will not create the resource. |
| `BackboardRateLimitError` | yes | Rate limit. |
| `BackboardServerError` | yes | Server error. |
| `BackboardAPIError` with no status code | yes | The SDK reports timeouts and connection errors this way. Backboard `failed`, `cancelled` and `expired` runs and `EmptyResponseError` also have no status. |
| `BackboardAPIError` with status >= 500 | yes | Server-side. |
| `BackboardAPIError` with status < 500 (for example 401) | no | Client-side. |
| Any other exception | no | Not a Backboard error. |

## Approval gate

The gate is a deepagents human-in-the-loop interrupt, configured in `build_agent` when
`POSTTRAIN_REQUIRE_APPROVAL` is true:

```python
interrupt_on = {"launch_training": {
    "allowed_decisions": ["approve", "reject"],
    "description": approval_description(settings),
}}
create_deep_agent(..., interrupt_on=interrupt_on, checkpointer=..., store=...)
```

Mechanics:

1. **Inheritance.** deepagents passes the top-level `interrupt_on` to declarative `SubAgent` specs
   that do not set their own. `training_executor` therefore pauses when it calls `launch_training`.
   The test `test_training_pauses_for_approval_then_runs_after_approve_in_new_process` exercises this
   through the `task` tool.
2. **Pause.** The interrupt stops the graph before the tool runs. The checkpoint is written to SQLite,
   so the pause is durable. `run_with_recovery` reads the pending state with `pending_actions`, which
   collects `action_requests` from the interrupt values, and records status `AWAITING_APPROVAL`.
3. **Description.** `approval_description` builds the text shown to the human: base model, dataset
   ref, hyperparameters, and the output of `budget.estimate` for those arguments. An error in the
   estimate is shown inline and does not block the prompt.
4. **Decision.** `replay.decide` sends one decision per pending action:
   `{"type": "approve"}`, or `{"type": "reject", "message": ...}`, using
   `Command(resume={"decisions": [...]})`. When one subagent is paused, that is the whole payload.
   When several are paused (see [Parallel subagents](#parallel-subagents)), the payload is
   `Command(resume={interrupt_id: {"decisions": [...]}, ...})`, one entry per interrupt, so a single
   decision answers them all. It then calls `run_with_recovery` with that resume payload.
5. **Reject.** The tool does not run. The rejection message reaches the subagent, which is told to
   return `REJECTED` and not retry the same launch.
6. **Process independence.** A decision can come from a new process, because the graph is rebuilt
   over the same SQLite files and the checkpoint is loaded by thread id.

Inside `launch_training`, which runs only after approval, the code checks hyperparameters, dataset
verification (`verify_file`), suite-state overlap, the `check_chat_template` marker for the current
dataset SHA-256, the optimizer-step floor and `estimate` with `within_budget is True`. Any failure
returns `BLOCKED` with a stage. The job is submitted only when all pass. After a successful
submission, `record_spend` appends to `spend.jsonl`.

## Promotion gate

`champion_manager` calls `run_gate(champion_ref, candidate_ref)`. It resolves checkpoint/job names,
reads the newest ok held-out and regression results from `eval_ledger.jsonl`, and calls
`promotion_gate(..., require_regression=True)`. A missing split returns an error naming it, never a
verdict. `decide_promotion` remains in `make_tools` and calls the same gate, but no role is given it;
it does not require regression unless the caller passes both sides.

`gate.promotion_gate` fails closed. Every failed rule adds a reason; any reason means `REJECT`:

1. Both scores are finite, and both results carry evidence in `LIVE` or `PRIOR_VERIFIED_RUN`.
   `RULE_VERIFIED` is dataset provenance, not gate evidence.
2. `suite`, `split`, `seed`, `judged_by` and `served_by` are identical.
3. Both `n` values are integers greater than zero.
4. The candidate has a finite `parse_rate >= max(0.90, base parse_rate - 0.02)`. A missing candidate
   parse rate rejects; a missing base rate uses the 0.90 floor.
5. The candidate gains at least 10% relative (with a zero champion, any positive candidate passes)
   and does not drop more than 0.05 absolute.
6. Significance: if both sides carry per-task `verdicts` of the same length over the same
   `tasks_sha256`, use the exact paired McNemar sign test (one-sided 5%). Otherwise use the
   unpaired one-sided two-proportion z-test and require `z >= 1.645`. Differing `tasks_sha256`
   values reject.
7. Regression: both regression results must have verified evidence, matching keys, `n > 0`, the
   candidate parse-rate floor, `split == "regression"` and a drop of at most 0.05.

When `n` is usable the verdict reports `mde_80pct_power`, `power_note` and a `power` block
(`observed_gain`, `adequately_powered`, `recommended_n=300`, `n`) for 80% power at one-sided 5%. It
warns when the observed gain is below the MDE and when `n < 300`. A small but real gain can
therefore be rejected as underpowered; that is not evidence that training had no effect.

## Budget and tokenization

`budget.estimate` counts each row with the base model's tokenizer and chat template, including the
tools schema (`experiments/pretokenize.py`), when the optional `tokenize` extra and tokenizer files
are available. Otherwise it uses `4 * chars/4` as a conservative fallback and reports which `basis`
it used. Trained tokens are `sum(min(row_tokens, context_length)) * n_epochs`; cost is that total
times the operator-supplied `POSTTRAIN_PRICE_PER_MTOK`.

Without a price, `POSTTRAIN_ALLOW_UNPRICED=1` caps the cumulative trained tokens at
`POSTTRAIN_MAX_TRAINED_TOKENS`; otherwise `within_budget` is `None` and the launch is blocked.
Provider-reported `trained_tokens` are reconciled into `spend.jsonl` on status/wait, replacing the
launch estimate. Nebius bills by compute, so the ledger is an estimate unless reconciled.

## Replay module

`runtime/replay.py` provides the operations on a thread.

| Function | Behaviour |
|----------|-----------|
| `run_with_recovery(graph, goal, thread_id, ...)` | If the thread has state, resumes with no new input; otherwise starts from the goal. On an exception, saves the checkpoint id and retries with `base_delay * 2**(attempt-1)` seconds of backoff, up to `max_attempts`. `FATAL = (CassetteMiss, KeyboardInterrupt, CancelledError)` is re-raised immediately. Records the run ledger before and after each attempt. |
| `pending_actions(graph, thread_id)` | Reads the thread state and returns `action_requests` from pending interrupts. |
| `decide(graph, thread_id, approve, message=...)` | Raises `ValueError` if nothing is pending. Otherwise resumes with a decision for each pending action. If several subagents are paused, every interrupt is answered in the same resume, keyed by interrupt id. |
| `history(graph, thread_id)` | Lists checkpoints newest first, with checkpoint id, step, source, next nodes, creation time, message count. |
| `fork(graph, thread_id, checkpoint_id, callbacks)` | Invokes the graph with `input=None` at a given checkpoint id. The original history stays. |
| `ledger(store)` | Returns up to 100 items in the `posttrain/runs` namespace. Each has `thread_id`, status, attempts, errors, last checkpoint id, update time, and pending actions. |

The run ledger is written by `_record` at the start of each attempt and at the end of the run.

### Cassette (`llm/backboard_model.py`)

`Cassette` is a JSONL file of `{"key", "response"}` rows.

- **Key.** SHA-256 over JSON of `[llm_provider, model_name, system, content, bound_tools]`. The key
  does not include `assistant_id` or `memory`.
- **Modes.** `auto` returns hits and records misses. `record` always calls the client. `replay` returns
  hits and raises `CassetteMiss` for misses, and never writes.
- **Scope.** The cassette sits inside `ChatBackboard._call`. Tool execution is not cached. A replayed
  run therefore re-runs every tool, including web, Hugging Face, and Nebius calls.

## Telemetry module

`TelemetryHandler` is a LangChain `BaseCallbackHandler` with `run_inline = True`, so events are
written in order under asyncio. The CLI attaches it to every run, approval, and fork.

| Callback | Event written |
|----------|---------------|
| `on_chat_model_start` | Stores start time and name `"<ls_provider>:<ls_model_name>"`. Nothing is written yet. |
| `on_llm_end` | `{"event": "model", "name", "ms", "ok": true, "input_tokens", "output_tokens"}` |
| `on_llm_error` | `{"event": "model", "name", "ms", "ok": false, "error": "<ExceptionClass>"}` |
| `on_tool_start` / `on_tool_end` | `{"event": "tool", "name", "ms", "ok": true}` |
| `on_tool_error` | `{"event": "tool", "name", "ms", "ok": false, "error": "<ExceptionClass>"}` |

Every row also has `ts`. Writes are serialised by a lock. Prompts, completions, tool arguments, and
exception messages are not written; only the exception class name is.

`summarize(path, last_n, only_errors)` reads the last `last_n` lines and returns `events`, `window`,
`by_name` (keyed `event:name`, with `calls`, `errors`, `p50_ms`, `p95_ms`, token sums), `error_types`,
and `recent_errors` (last five). `only_errors` drops `by_name`. `telemetry_tool` exposes the same
summary to the coordinator as `read_telemetry`. The `telemetry` CLI command runs without a key.

## Nebius API calls (`experiments/nebius.py`)

The base URL is `NEBIUS_BASE_URL` (default `https://api.tokenfactory.nebius.com/v1`). Each request sends
`Authorization: Bearer <NEBIUS_API_KEY>` and uses a synchronous `httpx.Client` with a 120-second timeout.
Responses with status 400 or more are returned as `{"error": "nebius HTTP <code>", "detail": ...}`.
Nothing is raised. Callers turn the error into a `BLOCKED` or failed result.

| Call | Made by | Request | Used for |
|------|---------|---------|----------|
| `POST /files` | `train` | Multipart: `purpose=fine-tune`, `file=<dataset>.jsonl` | Upload the verified dataset. The returned `id` is the training file. |
| `POST /fine_tuning/jobs` | `train` | JSON: `model` (base model), `training_file`, `seed`, `suffix` (dataset stem, up to 64 characters), `hyperparameters` (validated config plus `lora: true`) | Create the LoRA fine-tuning job. The returned `id` is the provider job id, and the job's `status` (default `queued`) is returned. |
| `GET /fine_tuning/jobs/{id}` | `job_status` | none | Status, `trained_steps`, `total_steps`, `trained_tokens`, `error`. |
| `GET /fine_tuning/jobs/{id}/checkpoints` | `job_status` | none, only after status `succeeded` | Takes the checkpoint with the highest step (ties: newest `created_at`) as the final checkpoint name. |
| `GET /fine_tuning/jobs/{id}/events` | `job_events` | query `limit` (1 to 200) | Event log. Messages are cut to 300 characters. |
| `GET /models` | `serving` | none, once per backend when `POSTTRAIN_SERVE=auto` | Decides `served_by`: `nebius-serverless` if the base model is listed, else `nebius-sandbox-cpu`. A failed probe keeps serverless (not cached). |
| `POST /chat/completions` | `_infer_serverless` | JSON: `model` (base model for `base`, else the checkpoint name), `messages`, `temperature: 0`, and `tools` if the task has them | Serverless inference; the first tool call is judged. |
| `GET /fine_tuning/jobs`, `GET /fine_tuning/jobs/{id}/checkpoints` | `_resolve_checkpoint` | `limit=100`; up to 30 succeeded jobs scanned | Sandbox serving: maps a checkpoint name (exact match) to its `result_files`. `job_status` fills the same cache. |
| `GET /files/{id}`, `GET /files/{id}/content` | `_download_adapter` | none | Sandbox serving: filename (basename) and bytes of each checkpoint file, downloaded on the host. |

**Sandbox CPU serving** (`experiments/sandbox_student.py`). Used because serverless serves neither
`Qwen/Qwen3-0.6B` nor its `ft:` checkpoints. A non-disposable branch of `POSTTRAIN_SANDBOX_IMAGE`
gets CPU torch, transformers, peft and the base weights (built once per process); an adapter is one
more small layer (`adapter_config.json`, `adapter_model.safetensors`, validated by `prepare_adapter`:
the config must name the base model). The generation script adds peft's `base_model.model.` prefix
to the provider's tensor keys and fails on any unexpected or missing LoRA key
(`adapter_load_problem`) before merging. Batches of 8 prompts go on stdin to disposable jobs, at
most 4 in flight, each with a hard timeout; greedy, `max_new_tokens` 160, chat template with the
task's `tools` and thinking off. `parse_tool_calls` reads `<tool_call>{json}</tool_call>`. Any failed
batch or sample makes the whole split an `error` (with `n_tasks`, `n_generated`), never a score. The
key never enters a sandbox; no env is passed.

`_score` reads `<suite_dir>/<split>.jsonl`, at most 1000 tasks. It runs a small preflight sample
first; if fewer than half of those outputs parse, it returns a preflight error instead of a
full-length run of zeros. Any inference error stops the whole split and returns the error with
`stage: "inference"` (and `n_tasks`/`n_generated` for serving failures). Benchmark results (`train`)
include up to 20 failures with `task_id`, `expected`, and `got`. Evaluation results (`heldout`,
`regression`, `heldout_phrasing`) include `n`, `score`, `parse_rate`, `by_game`, `judged_by`,
`served_by`, per-task `verdicts` and `tasks_sha256`, but no task contents or expected values.

The fine-tuning request shape follows Nebius' published schema, as noted in the module docstring. The
Nebius client and sandbox paths have been exercised against mocks and a live job; the test suite
itself does not call Nebius.

## Subagent contracts

Each role's prompt (`prompts/*.md`) has an INPUT CONTRACT and an OUTPUT CONTRACT. The shared preamble
in `subagents.py` tells every role that held-out inputs are sealed and that evidence labels must be
copied exactly. `ROLES` gives `data_curator` `generate_game_sft`, `inspect_dataset` and
`check_chat_template`; `training_executor` `wait_for_training`; and `champion_manager` `run_gate`.
`decide_promotion` remains a deterministic tool but is not assigned to a role; `abort_experiment` is
a coordinator tool. Some prompt text refers to inputs that no tool provides (for example `run_id`,
`manifest_hash`, and an "objective worker"). The code does not validate subagent output against these
contracts. The enforced rules are the ones in the deterministic tools: the sealed split restriction,
state-overlap and dataset validation, the approval gate, the budget, and the promotion gate.

## Parallel subagents

`SubagentConcurrencyMiddleware` (`runtime/concurrency.py`) is attached to the coordinator only. Each `task`
tool call acquires a slot from a semaphore sized by `POSTTRAIN_MAX_PARALLEL` (default `2`, values
below 1 read as 1). When the coordinator emits several `task` calls in one message, LangGraph runs
them concurrently. Calls beyond the limit wait for a slot; none are dropped.

- **Two semaphores.** The async path uses an `asyncio.Semaphore`, created on first use so it binds to
  the running loop. The sync path uses a `threading.Semaphore`. `peak` records the highest number of
  `task` calls observed at once.
- **Scope.** Only `task` calls are limited. Other tool calls, such as `recall` or `web_search`, are
  not queued.
- **Which steps overlap.** `ORCHESTRATOR_PROMPT` asks the coordinator to issue together (a) the
  researcher prior-art scan (step 3) with the `failure_analyst` (step 2), and (b) one `evaluator`
  task for the champion and one for the candidate (step 7). It forbids parallel `training_executor`
  tasks. The grouping is a prompt instruction. The code enforces only the cap.
- **Several approvals at once.** Each paused subagent contributes one interrupt. `pending_actions`
  and `_pauses` collect them. `decide` answers all of them in one resume, keyed by interrupt id.

## Sealed-data boundaries, in code

| Boundary | Enforced in |
|----------|-------------|
| `run_benchmark` accepts only `train` | `domain.make_tools.run_benchmark` |
| `evaluate_checkpoint` accepts `heldout`, `regression` or `heldout_phrasing`, and returns scores/metadata only | `domain.make_tools.evaluate_checkpoint`, `nebius._score` / `evaluate` |
| `heldout_phrasing` is informational and never gates | `gate.promotion_gate` (only the held-out and regression inputs are passed) |
| Failure details are returned only for `train` | `nebius.benchmark` (failures) vs `nebius.evaluate` (no task contents) |
| Dataset rows containing sealed markers are rejected | `dataset._row_errors` |
| Training states overlapping any eval split are rejected | `dataset.check_suite_overlap` in `check_chat_template` and `launch_training` |
| Web queries containing `heldout://` or `sealed://` are refused | `web.web_search` |

## Known gaps

- The last live run produced no gate verdict: candidate held-out evaluation timed out uploading the
  adapter, candidate regression was 0.0, base held-out was 0.19 (n=100) and base regression 0.1833
  (n=60). It is not a successful promotion.
- At base 0.19 and n=100 the 80%-power MDE is about 0.138 absolute, while the 10% relative threshold
  is only about 0.019. The default 300-task held-out split is the intended fix; read the `power`
  block, not just the decision.
- `state/data/exp1_data_n1200.jsonl` is a v1 dataset with 91 states in `suite_v2`; the contamination
  check blocks it. Generate fresh data with `generate_game_sft` or `scripts/make_sft.py`.
- The approval description shows the dataset ref and the estimate, not the dataset hash.
- A crash between provider submission and `record_spend` loses the spend record. No idempotency key
  is sent on job creation.
- The orchestrator's loop order is a prompt instruction, not an enforced state machine.
- `PRIOR_VERIFIED_RUN` is accepted by the gate and the verifier but no code path produces it;
  `RULE_VERIFIED` is dataset provenance and is not accepted as gate evidence.
- The promotion gate returns a decision only. No champion is stored and no checkpoint is deployed.
- The tests do not call Backboard, Nebius, Tavily or Hugging Face. The request shapes are mocked, not
  live-verified.
