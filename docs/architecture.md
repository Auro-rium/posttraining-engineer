# Architecture

posttrain-agent is a `deepagents` coordinator that runs on LangGraph. It takes its
LLM calls and semantic memory from Backboard through a custom LangChain chat model.
Deterministic code owns every decision that affects promotion.

## Diagram

```
                     posttrain CLI (cli.py)
                               |
                               v
        replay.py: run_with_recovery | history | fork | ledger
                               |
                               v
   +------------------- deep agent (agent.py) -------------------+
   | coordinator: model=ChatBackboard                            |
   |   tools: remember, recall, web_search, deepagents built-ins |
   |   middleware: Fallback > ModelRetry > ToolRetry             |
   |                                                             |
   |   subagents (subagents.py), 8 roles, same middleware        |
   |     each -> recall; researcher also -> remember             |
   |     deterministic tools (domain.py) and web tools (web.py)  |
   |                                                             |
   |   backend: CompositeBackend                                 |
   |     "/memories/" -> StoreBackend (store.sqlite)             |
   |     everything else -> StateBackend (graph state)           |
   +-------+------------------------+----------------------------+
           |                        |
           v                        v
  ChatBackboard._call        checkpoints.sqlite (LangGraph)
   | Cassette? hit -> return      store.sqlite (/memories/, runs)
   v
  Backboard API: send_message, assistant, memories (remember/recall)
  Tavily API (web.py, only if TAVILY_API_KEY is set)
```

## Components

### Chat model: `backboard_model.py`

`ChatBackboard` is a `BaseChatModel` that delegates generation to Backboard.

- **Stateless per call.** LangGraph owns conversation state through its checkpoints.
  Each model call sends one `send_message` with the rendered transcript (system prompt
  split out, prior turns flattened into `[user]`, `[assistant]`, `[tool_result]`
  blocks) and no Backboard thread id. The assistant id is always passed, so the
  assistant-level memory accumulates across calls. The reason for this design is that
  mapping graph threads onto Backboard threads would diverge after the first replay,
  fork or isolated subagent call.
- **Tools are re-sent on every call.** `bind_tools` stores OpenAI-format tool
  definitions, which go out with each request. Tool results are not submitted back to
  a pending run; they are rendered into the next transcript.
- **Tool arguments** that are not a JSON object become `invalid_tool_calls`.
- **Empty replies** (no text and no tool calls) raise `EmptyResponseError`, which is
  transient. Run statuses `failed`, `error`, `cancelled` or `expired` raise
  `BackboardAPIError`.
- `max_input_tokens` is only recorded in the model profile. Nothing truncates the
  transcript.

### Cassette: record/replay

`Cassette` is a JSONL file of `{key, response}` rows. The key is a SHA-256 of the
provider, model, system prompt, rendered content and bound tools. It does not include
the assistant id or memory mode.

`_call` checks the cassette before it opens a client, so a hit makes no network call.

| Mode | Hit | Miss |
| --- | --- | --- |
| `auto` | replay | call live, record |
| `record` | call live, record | call live, record |
| `replay` | replay | raise `CassetteMiss` |

Because the key covers the full rendered prompt and tool list, changing a prompt or a
tool schema makes every replay miss. `CassetteMiss` is fatal for recovery.

### Memory: three tiers

1. **Thread state (LangGraph checkpoints).** `AsyncSqliteSaver` writes
   `state/checkpoints.sqlite`. This is the per-thread conversation and graph state.
2. **File memory (`/memories/`).** `persistence.build_backend` builds a
   `CompositeBackend`. Paths under `/memories/` go to a `StoreBackend` on
   `state/store.sqlite`, namespace `("posttrain", "memories")`, so they survive across
   threads. All other paths go to `StateBackend`, which is scratch space in graph
   state. On startup, `open_runtime` seeds `/memories/AGENTS.md` if `/memories/` is
   empty. The coordinator loads that file as its memory.
3. **Semantic memory (Backboard).** `memory.py` provides `remember` (`add_memory`
   with a `kind` metadata field) and `recall` (`search_memories`, returns JSON hits or
   `"no relevant memories"`). Both are explicit tool calls. Nothing is written
   automatically.

`ensure_assistant` resolves the assistant id in this order: `BACKBOARD_ASSISTANT_ID`,
then `state/backboard.json`, then a newly created assistant (which is written to
`backboard.json`).

### Resilience: `resilience.py`

`build_middleware` returns this stack, outermost first:

1. **`ModelFallbackMiddleware`**, added only when `POSTTRAIN_FALLBACK_MODELS` is set.
   It is built from one `ChatBackboard` per fallback `provider:model` pair.
2. **`ModelRetryMiddleware`** with `max_retries = model_retries` (default 2),
   `retry_on = is_transient`, `initial_delay=1.0`, `max_delay=20.0`, and
   `on_failure="error"`. It sits inside the fallback layer, so each model, primary or
   fallback, gets its own retries before the next model is tried.
3. **`ToolRetryMiddleware`** with `max_retries=1` and `on_failure="continue"`.

The same middleware list is given to the coordinator and to every subagent.

`is_transient` classifies errors as follows:

| Error | Retried |
| --- | --- |
| `BackboardRateLimitError`, `BackboardServerError` | yes |
| `BackboardAPIError` with `status_code >= 500` | yes |
| `BackboardAPIError` with no status code (timeouts, connection errors, run failures, `EmptyResponseError`) | yes |
| `BackboardAPIError` with a 4xx status | no |
| `BackboardValidationError`, `BackboardNotFoundError`, `CassetteMiss` | no |
| any other exception | no |

Non-transient errors are not retried, but they still reach the fallback layer, as
`test_non_transient_error_is_not_retried_but_still_falls_back` checks.

### Replay and recovery: `replay.py`

- **`run_with_recovery(graph, goal, thread_id, ...)`** starts a new thread with the
  goal as the first user message, or resumes an existing thread if it has state. On an
  exception it records the error, sleeps `base_delay * 2**(attempt-1)` seconds (base
  2.0), and resumes the same thread from its last checkpoint. Completed steps are not
  re-run. `CassetteMiss`, `KeyboardInterrupt` and `CancelledError` are re-raised
  immediately.
- **`history(graph, thread_id)`** lists checkpoints, newest first, with step, source,
  next nodes and message count.
- **`fork(graph, thread_id, checkpoint_id)`** re-invokes the graph from a past
  checkpoint. Original history is preserved. With a cassette in `replay` mode, the
  model calls come from the recording, so the run is deterministic and offline.
- **Run ledger.** `_record` writes to the store namespace `("posttrain", "runs")`,
  keyed by thread id. Each write replaces the previous row for that thread, so the
  ledger holds the latest status, attempt count and errors for each thread, not a
  row per attempt. `ledger()` returns up to 100 rows.

### Deterministic rules: `domain.py`

Nothing in this module calls an LLM.

- `validate_qlora` checks a config against a fixed search space: allowed values for
  rank, alpha, sequence length, batch size and gradient accumulation; inclusive ranges
  for dropout, learning rate and epochs; and a non-empty subset of seven target
  modules. Unknown or missing keys are errors.
- `promotion_gate` returns `PROMOTE` only when both results have verified evidence
  (`LIVE` or `PRIOR_VERIFIED_RUN`), suite, split and seed match, relative improvement
  is at least 10%, and regression is at most 0.05 absolute. It returns a decision and
  reasons. It does not write any state.
- `make_tools` binds five tools to an `ExperimentBackend`. `run_benchmark` accepts only
  the `train` split. `evaluate_checkpoint` accepts only `heldout` or `regression`.
  `launch_training` validates the config first and returns `BLOCKED` with the errors
  if it is invalid.
- `LocalStubBackend` is the default backend. Its results are all labelled
  `EXPLANATION`.

### Web tools: `web.py`

`web_tools` returns `web_search` and `web_extract`, which call the Tavily API with
`httpx`. Without `TAVILY_API_KEY`, each returns `{"error": ...}` and makes no request.
Queries containing `heldout://` or `sealed://` are rejected. Results are labelled
`EXPLANATION`. HTTP errors (status 400 or higher) return `{"error": "tavily HTTP <code>"}`.

### Subagents: `subagents.py`

Each of the eight roles is a deepagents `SubAgent` with a system prompt from
`prompts/<file>.md` (prefixed by a shared preamble), the shared middleware, and the
tools below. Every role gets `recall`. Only `researcher` also gets `remember`.

| Role | Prompt file | Deterministic or web tools |
| --- | --- | --- |
| `benchmark` | `benchmark.md` | `run_benchmark` |
| `failure_analyst` | `failure_analyst.md` | none |
| `researcher` | `research.md` | `web_search`, `web_extract` (plus `remember`) |
| `data_curator` | `data_curator.md` | none |
| `training_designer` | `training_designer.md` | `validate_qlora_config`, `web_search` |
| `training_executor` | `training_executor.md` | `launch_training` |
| `evaluator` | `eval.md` | `evaluate_checkpoint` |
| `champion_manager` | `champion_manager.md` | `decide_promotion` |

The coordinator's prompt describes the loop: benchmark, failure analysis, research,
curation, training design, training, evaluation of champion and candidate on the same
split, then the promotion decision. It uses `write_todos` (a deepagents built-in) for
planning and the `task` tool to delegate.

## Data flow for one run

1. The CLI builds `Settings` and opens `checkpoints.sqlite` and `store.sqlite`.
2. `build_agent` resolves the assistant id, builds the middleware and the model, and
   assembles the deep agent.
3. `run_with_recovery` invokes the coordinator. Each model call goes through the
   middleware, then through `ChatBackboard._call`, which checks the cassette before it
   calls Backboard.
4. The coordinator delegates through `task` to a subagent. The subagent calls its
   deterministic or web tools and returns a JSON result, which is written under `/work/`
   (scratch state) for later steps to read.
5. Each checkpoint is saved. After a failure, the run resumes from the last
   checkpoint. The ledger row is updated on each attempt and at the end.
