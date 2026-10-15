# Architecture

posttrain-agent is a deepagents coordinator running on LangGraph. Its LLM calls and semantic
memory go through Backboard, via a custom LangChain chat model. Deterministic Python code owns
every decision that affects promotion, budget, and the held-out boundary. This document describes
the code as it is; see the README "Status and limitations" section for what is unverified.

## Components

| Module | Responsibility |
|--------|----------------|
| `agent.py` | `build_agent` assembles the deep agent: model, middleware, tools, subagents, memory, checkpointer, store, and the approval interrupt. Holds `ORCHESTRATOR_PROMPT` and `default_backend`. |
| `concurrency.py` | `SubagentConcurrencyMiddleware`: semaphore capping concurrent `task` calls at `POSTTRAIN_MAX_PARALLEL`. Attached to the coordinator only. |
| `cli.py` | `posttrain` command. Opens the runtime, builds the agent, and dispatches to `replay.py` and `telemetry.py`. |
| `config.py` | `Settings` dataclass and `Settings.from_env()`. The only place environment variables are read. |
| `backboard_model.py` | `ChatBackboard` (LangChain chat model over the Backboard API), `Cassette` (record/replay store), and transcript rendering. |
| `resilience.py` | `is_transient`, `build_model`, and `build_middleware` (fallback, retry, tool retry). |
| `memory.py` | `ensure_assistant` (persistent Backboard assistant) and the `remember` / `recall` tools. |
| `persistence.py` | `open_runtime`: LangGraph SQLite checkpointer, SQLite store, and the composite file backend with `/memories/` routed to the store. Seeds `/memories/AGENTS.md`. |
| `subagents.py` | `ROLES`: the eight role contracts (prompt file, description, tool names, memory access). `build_subagents` turns them into deepagents `SubAgent` specs. |
| `domain.py` | `Protocol ExperimentBackend`, `LocalStubBackend`, `validate_hparams`, `promotion_gate`, and `make_tools` (the deterministic tool functions). |
| `dataset.py` | Chat-format SFT verification (`verify_rows`), safe file references (`resolve`), load and save. |
| `budget.py` | Token heuristic, cost estimate, `spend.jsonl` ledger, `within_budget`. |
| `nebius.py` | `NebiusBackend`: fine-tuning jobs and deterministic scoring over the Nebius REST API. |
| `web.py` | Tavily `web_search` and `web_extract`. |
| `hf.py` | Hugging Face Hub `hf_search`, `hf_model_info`, `hf_dataset_info`. |
| `replay.py` | `run_with_recovery`, `decide`, `pending_actions`, `history`, `fork`, `ledger`; the run ledger. |
| `telemetry.py` | `TelemetryHandler` (LangChain callback that writes JSONL events) and `summarize` / `telemetry_tool`. |
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

Inside `launch_training`, which runs only after approval, the code repeats three checks: hyperparameter
validation, dataset verification (`verify_file`), and `estimate` with `within_budget is True`. Any
failure returns `BLOCKED` with a stage. The job is submitted only when all three pass. After a
successful submission, `record_spend` appends to `spend.jsonl`.

## Replay module

`replay.py` provides the operations on a thread.

| Function | Behaviour |
|----------|-----------|
| `run_with_recovery(graph, goal, thread_id, ...)` | If the thread has state, resumes with no new input; otherwise starts from the goal. On an exception, saves the checkpoint id and retries with `base_delay * 2**(attempt-1)` seconds of backoff, up to `max_attempts`. `FATAL = (CassetteMiss, KeyboardInterrupt, CancelledError)` is re-raised immediately. Records the run ledger before and after each attempt. |
| `pending_actions(graph, thread_id)` | Reads the thread state and returns `action_requests` from pending interrupts. |
| `decide(graph, thread_id, approve, message=...)` | Raises `ValueError` if nothing is pending. Otherwise resumes with a decision for each pending action. If several subagents are paused, every interrupt is answered in the same resume, keyed by interrupt id. |
| `history(graph, thread_id)` | Lists checkpoints newest first, with checkpoint id, step, source, next nodes, creation time, message count. |
| `fork(graph, thread_id, checkpoint_id, callbacks)` | Invokes the graph with `input=None` at a given checkpoint id. The original history stays. |
| `ledger(store)` | Returns up to 100 items in the `posttrain/runs` namespace. Each has `thread_id`, status, attempts, errors, last checkpoint id, update time, and pending actions. |

The run ledger is written by `_record` at the start of each attempt and at the end of the run.

### Cassette (`backboard_model.py`)

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

## Nebius API calls (`nebius.py`)

The base URL is `NEBIUS_BASE_URL` (default `https://api.tokenfactory.nebius.com/v1`). Each request sends
`Authorization: Bearer <NEBIUS_API_KEY>` and uses a synchronous `httpx.Client` with a 120-second timeout.
Responses with status 400 or more are returned as `{"error": "nebius HTTP <code>", "detail": ...}`.
Nothing is raised. Callers turn the error into a `BLOCKED` or failed result.

| Call | Made by | Request | Used for |
|------|---------|---------|----------|
| `POST /files` | `train` | Multipart: `purpose=fine-tune`, `file=<dataset>.jsonl` | Upload the verified dataset. The returned `id` is the training file. |
| `POST /fine_tuning/jobs` | `train` | JSON: `model` (base model), `training_file`, `seed`, `suffix` (dataset stem, up to 64 characters), `hyperparameters` (validated config plus `lora: true`) | Create the LoRA fine-tuning job. The returned `id` is the provider job id, and the job's `status` (default `queued`) is returned. |
| `GET /fine_tuning/jobs/{id}` | `job_status` | none | Status, `trained_steps`, `total_steps`, `trained_tokens`, `error`. |
| `GET /fine_tuning/jobs/{id}/checkpoints` | `job_status` | none, only after status `succeeded` | Takes `fine_tuned_model_checkpoint` from the last item as the checkpoint name. |
| `GET /fine_tuning/jobs/{id}/events` | `job_events` | query `limit` (1 to 200) | Event log. Messages are cut to 300 characters. |
| `POST /chat/completions` | `_score` | JSON: `model` (base model for `base`, else the checkpoint name), `messages`, `temperature: 0`, and `tools` if the task has them | Evaluation. The score is exact match on the first tool call's name and parsed arguments. |

`_score` reads `<suite_dir>/<split>.jsonl`, at most 200 tasks. Any inference error stops the whole
split and returns the error with `stage: "inference"`. Benchmark results (`train`) include up to 20
failures, each with `task_id`, `expected`, and `got`. Evaluation results (`heldout`, `regression`)
include only `n` and `score`.

The fine-tuning request shape follows Nebius' published schema, as noted in the module docstring. The
module has only been tested against a mock transport.

## Subagent contracts

Each role's prompt (`prompts/*.md`) has an INPUT CONTRACT and an OUTPUT CONTRACT. The shared preamble
in `subagents.py` tells every role that held-out inputs are sealed and that evidence labels must be
copied exactly. Some prompt text refers to inputs that no tool provides (for example `run_id`,
`manifest_hash`, and an "objective worker"). The code does not validate subagent output against these
contracts. The enforced rules are the ones in the deterministic tools: the sealed split restriction,
hyperparameter and dataset validation, the approval gate, the budget, and the promotion gate.

## Parallel subagents

`SubagentConcurrencyMiddleware` (`concurrency.py`) is attached to the coordinator only. Each `task`
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
| `evaluate_checkpoint` accepts only `heldout` or `regression`, and returns scores only | `domain.make_tools.evaluate_checkpoint`, `nebius._score` / `evaluate` |
| Failure details are returned only for `train` | `nebius.benchmark` (failures) vs `nebius.evaluate` (no failures) |
| Dataset rows containing sealed markers are rejected | `dataset._row_errors` |
| Web queries containing `heldout://` or `sealed://` are refused | `web.web_search` |

## Known gaps

- The approval description shows the dataset ref and the estimate, not the dataset hash.
- A crash between provider submission and `record_spend` loses the spend record. No idempotency key
  is sent on job creation.
- The orchestrator's loop order is a prompt instruction, not an enforced state machine.
- `PRIOR_VERIFIED_RUN` is accepted by the gate and the verifier but no code path produces it.
- The promotion gate returns a decision only. No champion is stored and no checkpoint is deployed.
