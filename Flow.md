# Codebase Flow

This is the current operational map, not a history. Update it whenever control flow, state, APIs, events, dependencies, or failure behavior changes. Historical changes belong in `ChangeLog.md`; architectural reasoning belongs in `Decisions.md`.

## System boundaries

```text
HTTP client
    -> Coordinator FastAPI service
       -> typed run state + append-only events
       -> authenticated Research A2A service
          -> Failure Analyst -> Research Agent -> Data Curator -> Training Designer
          -> integrity-checked GCS RAG corpus, excluding held-out tasks
       -> authenticated Execution A2A service
          -> Benchmark Runner -> Training Executor -> Evaluation Agent -> Champion Manager
       -> Vertex AI custom training

Research + Execution A2A services
    -> authenticated external objective-evidence worker
       -> FunctionGemma inference + sandboxed AgentGym WebShop
       -> hashed benchmark, verified-curation, evaluation, and training artifacts

Canonical metadata: Firestore (memory adapter locally)
Immutable artifacts: GCS (local-file adapter locally)
Telemetry: OpenTelemetry -> Cloud Trace and Cloud Logging
Secrets: Secret Manager -> process memory only
```

The coordinator is a workflow controller and is not counted among the eight specialists. `SERVICE_ROLE` selects the coordinator, research, or execution application from the same container.

## End-to-end research loop

1. `POST /api/runs` validates FunctionGemma/WebShop scope, creates a typed run, appends the initial event, and returns its ID.
2. A step or auto command acquires the run version, rejects cancelled or terminal work, and dispatches the next phase idempotently.
3. Benchmark Runner sends a typed A2A operation to Execution, which obtains train-side trajectories and objective rewards from the authenticated evidence worker. The coordinator persists the hashed trajectory artifact and trajectory IDs for restart-safe failure analysis.
4. The coordinator calls Research through authenticated A2A. Failure Analyst loads the hashed train trajectories and clusters grounded failures. Research Agent retrieves leakage-safe GCS RAG citations and forms one cited hypothesis. Data Curator proposes actions, but only the objective worker's `/v1/verify-curation` response can admit rows. Training Designer selects an unused bounded QLoRA configuration.
5. Training Executor submits a Vertex custom training job, polls it within the 55-minute deployment budget, then asks Execution to resolve hashed checkpoint and log evidence from `/v1/training-evidence`.
6. Evaluation Agent asks Execution to compare champion and candidate through `/v1/evaluate` on identical sealed inputs. Evaluation content remains in the objective worker; only validated aggregate metrics and hashed artifacts return.
7. Champion Manager applies the fixed improvement, regression, validity, provenance, and budget gates. A rejected candidate remains in experiment history. A promoted candidate atomically changes the champion reference.
8. The loop stops after promotion, cancellation, terminal failure, or two candidate attempts. Every transition appends an ordered event that the SSE endpoint can resume by event ID.

## RAG and data-admission flow

- The operator uploads a JSON corpus of FunctionGemma/WebShop documentation, training recipes, and previous experiment summaries to the configured GCS URI, optionally pinned by SHA-256. Terraform does not create the corpus.
- Research downloads and integrity-checks the corpus once, then `LeakageSafeRAG` rejects held-out/regression scopes and indexes deterministic lexical chunks.
- Hypothesis generation receives the retrieved citations, must select a nonempty subset, and is canonicalized back to the retrieved records. Retrieval spans export counts, never passages.
- A proposed SFT action carries source trajectory lineage. The external objective worker replays it through `/v1/verify-curation`; only strictly verified examples enter the hashed dataset manifest.

## A2A flow

- Coordinator sends official A2A v1 `message/send` JSON-RPC requests containing exactly one protobuf data part with `operation`, `run_id`, `schema_version`, and typed payload.
- Cloud Run ID tokens authenticate requests. Trace context and idempotency key travel in HTTP headers; retries are bounded to configured transient transport/status failures.
- Research accepts only analysis, hypothesis, curation, and design operations. Execution accepts only benchmark, evaluation, and training-evidence operations.
- The custom A2A `AgentExecutor` validates the request, dispatches `A2AOperationService`, and emits exactly one typed data artifact. Invalid schemas, cross-role operations, ungrounded evidence, leakage, or provider failures fail closed.

## State and artifact ownership

| Data | Owner | Storage | Mutation rule |
| --- | --- | --- | --- |
| Run phase, champion pointer, experiment count | Coordinator | Firestore / memory | Optimistic version check. |
| Ordered run events | Coordinator | Firestore / memory | Append only. |
| Failure reports and hypotheses | Research service | Firestore plus artifact copy | New immutable version per attempt. |
| SFT datasets and manifests | Data Curator | GCS / local files | Content-addressed after replay verification. |
| Training checkpoints and provider manifests | Execution service | GCS | Immutable provider job and content hashes. |
| Evaluation reports | Evaluation Agent | GCS plus aggregate state | Sealed task content stays inside execution boundary. |
| Promotion decision | Champion Manager | Firestore plus report artifact | Deterministic and append only. |

Artifact bytes are written and hashed before their URI is committed to state. If the state write fails, an unreferenced artifact may be garbage-collected later; state never points at a partial artifact.

The Vertex job handle is currently held only while the training phase polls. A coordinator restart during that poll cannot resume the job safely; `/auto` is therefore best-effort and a failed in-flight training phase requires operator inspection before another step.

## API and failure flow

- `GET /health` reports process role/readiness without credentials or remote payloads.
- `POST /api/runs` creates a run. Invalid model, environment, budget, or schema returns a 4xx response without state mutation.
- `GET /api/runs/{run_id}` and `/experiments` return safe typed state or 404.
- `GET /api/runs/{run_id}/events` streams ordered SSE records and resumes after a supplied last event ID. Disconnecting a client does not cancel work.
- `step` and `cancel` serialize per run. `auto` starts one process-local background task and is best-effort even though deployment keeps one always-CPU coordinator instance. Cancellation prevents new work but does not claim a provider job was stopped.
- `POST /api/demo/verify` compares a prepared manifest with a fresh baseline/candidate run and preserves `LIVE` versus `PRIOR_VERIFIED_RUN` evidence labels.
- Contract and leakage errors are terminal. Network throttling and transient provider errors are retried with bounded attempts. Exhausted retries append a sanitized failure event.

## Telemetry and error propagation

Every run, experiment, A2A request, retrieval, tool call, Vertex job, and evaluation span shares `run_id` and `trace_id`. Attributes use an allowlist: service/agent name, operation, safe artifact ID, duration, token counts, retry count, and outcome. Secrets, prompts, retrieved passages, trajectory observations, and held-out contents are forbidden.

Exceptions are translated at the boundary that owns recovery: an adapter identifies retryable provider failures; the orchestrator decides retry versus terminal transition; the API maps typed domain failures to HTTP responses. Logs and events contain sanitized codes, never raw credential-bearing exceptions.

## Function catalogue

The catalogue covers project-owned top-level functions, classes with behavior, and HTTP handlers. Slash-separated method names inherit the class prefix from the first name. Protocol-only signatures and pure Pydantic data containers are grouped under their module unless they implement custom validation. Explicit `flow-ref` markers are checked against Python source by `backend/scripts/check_docs_sync.py`.

### ASGI bootstrap and API

- `_repository_for` — `backend/app/main.py`. Selects Firestore only for explicit cloud mode and otherwise returns a process-local repository; missing cloud dependencies/credentials fail application startup. <!-- flow-ref: backend/app/main.py::_repository_for -->
- `RunController.__init__` — `backend/app/main.py`. Retains repository/orchestrator ports, per-run locks, and background auto-task references for one process.
- `RunController.step` / `cancel` — `backend/app/main.py`. Serialize each run mutation through its lock and delegate to the orchestrator; repository and state errors propagate to typed API handlers.
- `RunController.start_auto` / `_run_auto` — `backend/app/main.py`. Start at most one background task per run and execute up to 16 locked steps; specialist errors have already been persisted by the orchestrator and are consumed to avoid raw provider output in server logs.
- `RunController.stop` — `backend/app/main.py`. Cancel and await unfinished background tasks during ASGI shutdown; it does not rewrite run state.
- `_event_sse` — `backend/app/main.py`. Serialize one typed event into an ID/event/data SSE frame consumed by reconnecting clients. <!-- flow-ref: backend/app/main.py::_event_sse -->
- `create_app` — `backend/app/main.py`. Composes settings, repository, decision provider, orchestrator, controller, telemetry lifespan, CORS, error handlers, and the eight API routes; injectable ports let tests avoid credentials. <!-- flow-ref: backend/app/main.py::create_app -->
- `create_app.lifespan` — Configure metadata-only Cloud Trace when explicitly enabled, then stop background tasks on shutdown; missing project/export dependencies fail startup.
- `create_app.not_found_handler` / `conflict_handler` / `invalid_state_handler` — Convert typed repository/state failures into small sanitized 404/409 JSON envelopes.
- `create_app.health` — Return only readiness, role, and environment; it reads runtime configuration and no credentials/state.
- `create_app.create_run` — Validate the fixed model/environment and bounded baseline inputs, persist a new run and `run.created` event, then return its ID/phase/evidence label. A duplicate or persistence failure produces no success response.
- `create_app.get_run` / `get_experiments` — Read typed run state or its experiment list without mutation; unknown IDs map to 404.
- `create_app.step_run` / `auto_run` / `cancel_run` — Execute one serialized step, accept one background auto loop, or cancel a non-terminal run; terminal auto requests return 409.
- `create_app.stream_events` — Validate the run, resume after `Last-Event-ID`, emit ordered SSE events and heartbeats, and close after disconnect or a terminal idle poll. It never cancels the run.
- `create_app.demo_verify` — Load the run and evaluate its latest candidate without mutation; missing candidates return 409.
- `_error_response` — `backend/app/main.py`. Serialize an allowlisted error code/message into the common JSON envelope without raw provider context. <!-- flow-ref: backend/app/main.py::_error_response -->
- `bootstrap_app` — `backend/app/main.py`. Return the coordinator FastAPI app for `coordinator` or the ADK A2A ASGI app for `research`/`execution`; invalid roles are rejected by settings. <!-- flow-ref: backend/app/main.py::bootstrap_app -->

### Settings and cloud adapters

- `Settings` — `backend/app/settings.py`. Validates service role, target/environment, candidate ceiling, backend selection, and cloud configuration from environment. It reads process environment, writes no state, and is consumed by application and adapter factories. Invalid configuration fails service startup.
- `get_settings` — `backend/app/settings.py`. Returns one cached `Settings` instance; it has no writes beyond its process cache and propagates validation errors to startup. <!-- flow-ref: backend/app/settings.py::get_settings -->
- `SecretManagerReader` — `backend/app/cloud.py`. Lazily creates the Google Secret Manager client and returns a requested secret version to the immediate caller. It never logs or persists the value; unavailable SDKs or provider failures become cloud integration errors.
- `VertexTrainingLauncher` — `backend/app/cloud.py`. Validates an experiment request, constructs safe job metadata, launches a Vertex custom training job, and returns its provider handle. It writes only through Vertex and propagates sanitized provider failures for orchestrator retry handling.
- `safe_cloud_metadata` — `backend/app/cloud.py`. Reduces provider metadata to scalar allowlisted values for logs/manifests; unsupported or sensitive objects are omitted. <!-- flow-ref: backend/app/cloud.py::safe_cloud_metadata -->

### Deployable cloud decision provider

- `GoogleIdentityTokenProvider.token` — `backend/app/cloud_provider.py`. Mints a Cloud Run audience-bound ID token from workload identity in a worker thread; missing credentials/dependencies fail the request without logging the token.
- `AuthenticatedHTTPA2ATransport.__init__` / `request` — `backend/app/cloud_provider.py`. Validate HTTPS and retry bounds, build an official A2A request, attach identity/idempotency/trace headers, and retry only transient transport or status failures; malformed/non-success responses become `CloudProviderError`.
- `_build_send_message_params` — `backend/app/cloud_provider.py`. Encodes one operation request as the official v1 protobuf JSON `SendMessage` parameter shape; transport consumes it. <!-- flow-ref: backend/app/cloud_provider.py::_build_send_message_params -->
- `_extract_a2a_payload` — `backend/app/cloud_provider.py`. Accepts task or message success responses, requires exactly one data artifact/part, validates operation/schema, and returns its payload; text, ambiguity, or remote failure fails closed. <!-- flow-ref: backend/app/cloud_provider.py::_extract_a2a_payload -->
- `BenchmarkExecutionResult.require_verifiable_train_evidence` / `EvaluationExecutionResult.require_verifiable_report` — `backend/app/cloud_provider.py`. Reject explanatory/incomplete evidence, wrong artifact kinds, non-train research trajectories, or missing evaluation reports before orchestration can use them.
- `_typed_request` — `backend/app/cloud_provider.py`. Sends one typed operation with a stable per-phase idempotency key and validates the response model; schema errors become sanitized cloud-provider failures. <!-- flow-ref: backend/app/cloud_provider.py::_typed_request -->
- `A2AResearchOperations.analyze_failures` / `form_hypothesis` / `curate_dataset` / `design_training` — `backend/app/cloud_provider.py`. Map research domain requests to typed A2A calls and verify dataset count consistency before returning specialist outputs.
- `A2AExecutionOperations.execute_benchmark` / `execute_evaluation` / `resolve` — `backend/app/cloud_provider.py`. Map objective work and training-evidence resolution to the execution A2A endpoint and validate response contracts.
- `CloudDecisionProvider.__init__` — `backend/app/cloud_provider.py`. Composes research, execution, Vertex, artifact-bucket, polling, and timeout ports and keeps only immediate benchmark evidence in process memory.
- `CloudDecisionProvider.benchmark` / `benchmark_evidence` — `backend/app/cloud_provider.py`. Obtain verified train trajectories, retain their hashed artifact/IDs, and expose them for immediate durable run-state persistence; missing evidence fails.
- `CloudDecisionProvider.analyze_failures` / `form_hypothesis` / `curate_dataset` / `design_training` — `backend/app/cloud_provider.py`. Delegate each grounded research phase using persisted benchmark provenance and reject absent prerequisites or local artifacts.
- `CloudDecisionProvider.launch_training` — `backend/app/cloud_provider.py`. Require a GCS dataset, submit Vertex, poll known states within budget, resolve matching checkpoint/log evidence, and reject timeout, unknown/failed state, mismatched job ID, or missing hashes. The in-flight handle is not yet durable.
- `CloudDecisionProvider.evaluate` — `backend/app/cloud_provider.py`. Request objective evaluation and return only a verified report; wrong result types fail.
- `build_cloud_decision_provider` — `backend/app/cloud_provider.py`. Fail fast unless cloud mode, team A2A URLs, and artifact bucket are configured, then compose authenticated transports and Vertex with deployment timeouts. <!-- flow-ref: backend/app/cloud_provider.py::build_cloud_decision_provider -->

### Research and execution operation services

- `ADKStructuredResearchGenerator.__init__` / `generate` — `backend/app/service_operations.py`. Create a single-turn Gemini ADK agent with a Pydantic output schema, pass only the bounded structured input, and reject empty or schema-invalid final output; Gemini cannot emit authoritative metrics or hashes.
- `GCSResearchEvidenceLoader.__init__` / `load_trajectories` — `backend/app/service_operations.py`. Read a hashed trajectory artifact through the GCS store, validate requested IDs/bounds/train-only split, and return grounded trajectories; integrity, decoding, absence, or leakage fails closed.
- `GCSGroundedRetriever.__init__` / `_load` / `search` — `backend/app/service_operations.py`. Validate the GCS corpus URI and optional SHA pin, download/index allowed `KnowledgeDocument` records once through `LeakageSafeRAG`, and return cited lexical results; empty, corrupt, held-out, or hash-mismatched corpora fail.
- `RemoteObjectiveEvidenceExecutor.__init__` / `_post` — `backend/app/service_operations.py`. Authenticate to the required HTTPS evidence worker with a Cloud Run ID token, post typed input within the configured timeout, and validate the typed response; the token and raw content are never persisted.
- `RemoteObjectiveEvidenceExecutor.benchmark` / `verify_curation` / `evaluate` / `training_evidence` — `backend/app/service_operations.py`. Call `/v1/benchmark`, `/v1/verify-curation`, `/v1/evaluate`, and `/v1/training-evidence` respectively; the external worker owns FunctionGemma, AgentGym, replay, sealed evaluation, and artifact hashing.
- `A2AOperationService.__init__` / `handle` — `backend/app/service_operations.py`. Require the role-specific ADK/RAG/objective dependencies, enforce the operation allowlist, wrap dispatch in metadata-only telemetry, and return the matching typed response.
- `A2AOperationService._handle_research` — `backend/app/service_operations.py`. Ground clusters in loaded trajectories; ground hypotheses in a canonical nonempty RAG citation subset; let Gemini propose repairs before objective verification; and validate bounded unique QLoRA output.
- `A2AOperationService._handle_execution` — `backend/app/service_operations.py`. Dispatch benchmark, evaluation, or training-evidence operations only to the objective worker and return its validated domain evidence.
- `A2AOperationService._load_trajectories` — `backend/app/service_operations.py`. Resolve bounded train evidence through the configured loader, with an explicit all-train sentinel only for curation.
- `_validate_proposals` — `backend/app/service_operations.py`. Reject repair proposals whose trajectory or step does not exist before the objective worker sees them. <!-- flow-ref: backend/app/service_operations.py::_validate_proposals -->

### ADK and specialist construction

- `build_specialists` — `backend/app/adk_agents.py`. Builds the eight role definitions with bounded tools/instructions from validated settings. It reads configuration, writes no run state, and is consumed by service bootstrap. Missing ADK dependencies fail explicitly. <!-- flow-ref: backend/app/adk_agents.py::build_specialists -->
- `build_service_agent` — `backend/app/adk_agents.py`. Selects the coordinator, research, or execution agent graph for `SERVICE_ROLE`; it calls `build_specialists`, writes no state, and rejects unknown roles. <!-- flow-ref: backend/app/adk_agents.py::build_service_agent -->
- `_operation_request_from_message` / `_data_part` — `backend/app/adk_service.py`. Decode exactly one protobuf data part into a typed request and encode one typed response part; absent dependencies, text parts, or invalid schemas fail closed. <!-- flow-ref: backend/app/adk_service.py::_operation_request_from_message --> <!-- flow-ref: backend/app/adk_service.py::_data_part -->
- `build_operation_agent_executor` — `backend/app/adk_service.py`. Builds the official A2A `AgentExecutor` bridge, tracks task lifecycle, dispatches the operation service, emits one data artifact, and returns coded content-free failures. <!-- flow-ref: backend/app/adk_service.py::build_operation_agent_executor -->
- `_build_operation_service` — `backend/app/adk_service.py`. Requires research/execution role and objective-worker URL; for research it also composes Gemini ADK, GCS trajectory loading, and pinned GCS RAG. <!-- flow-ref: backend/app/adk_service.py::_build_operation_service -->
- `create_a2a_app` — `backend/app/adk_service.py`. Wraps the role agent plus custom typed executor through the official ADK A2A adapter, optionally publishing an exact HTTPS Agent Card. Missing hard dependencies fail startup. <!-- flow-ref: backend/app/adk_service.py::create_a2a_app -->
- `_bootstrap_a2a_app` — `backend/app/adk_service.py`. Builds the module ASGI object for team roles and leaves coordinator imports side-effect free. <!-- flow-ref: backend/app/adk_service.py::_bootstrap_a2a_app -->

### Eight specialist contracts

- `BenchmarkRunner` — `backend/app/agents.py`. Accepts train-side task references, invokes the benchmark adapter, and produces trajectories/objective rewards. Invalid action or environment failures become typed execution failures.
- `FailureAnalyst` — `backend/app/agents.py`. Consumes trajectory evidence and asks the decision provider for clusters whose claims retain trajectory IDs; malformed or ungrounded output is rejected.
- `ResearchAgent` — `backend/app/agents.py`. Combines a failure report with allowed RAG citations to produce one falsifiable hypothesis. It cannot read held-out data.
- `DataCurator` — `backend/app/agents.py`. Proposes repaired action rows and calls the replay verifier; only verified rows reach the dataset manifest.
- `TrainingDesigner` — `backend/app/agents.py`. Selects an unused QLoRA configuration and calls `validate_qlora_config`; invalid, duplicate, or over-budget choices are rejected.
- `TrainingExecutor` — `backend/app/agents.py`. Passes a validated experiment and dataset manifest to the training launcher and returns provider/checkpoint provenance.
- `EvaluationAgent` — `backend/app/agents.py`. Calls the evaluator with identical champion/candidate settings and returns aggregate metrics without exposing sealed task contents.
- `ChampionManager` — `backend/app/agents.py`. Calls the deterministic promotion policy and produces the immutable promotion/rejection decision; no decision-provider override exists.
- `validate_qlora_config` — `backend/app/agents.py`. Enforces the allowed rank, learning rate, epoch, and dropout sets before launch; it writes no state and raises `AgentContractError` on violation. <!-- flow-ref: backend/app/agents.py::validate_qlora_config -->

### Domain validation

- `utc_now` / `new_id` — `backend/app/models.py`. Produce timezone-aware timestamps and prefixed random IDs for every persisted contract; they read/write no external state and feed all model factories. <!-- flow-ref: backend/app/models.py::utc_now --> <!-- flow-ref: backend/app/models.py::new_id -->
- `SFTExample.require_verified_train_improvement` — `backend/app/models.py`. Rejects non-train, unverified, or non-improving examples during model construction; admitted rows feed dataset creation.
- `DatasetManifest.reject_eval_data` — `backend/app/models.py`. Requires train-only source splits and a dataset artifact; invalid manifests cannot reach training design.
- `QLoRAConfig.allowed_learning_rate` / `allowed_dropout` — `backend/app/models.py`. Normalize incoming numeric values to the exact allowed literals and reject out-of-grid hyperparameters before design or launch.
- `TrainingResult.successful_job_has_checkpoint` — `backend/app/models.py`. Requires every successful job to have a checkpoint artifact; failures stop evaluation.
- `EvaluationReport.success_delta` / `regression_delta` — `backend/app/models.py`. Derive comparison deltas from stored metrics without mutation; Champion Manager consumes them.
- `RunState.enforce_experiment_budget` — `backend/app/models.py`. Rejects state with more than two attempts or more experiments than its configured ceiling; repositories cannot persist invalid state.

### A2A contracts and handoff log

- `A2AEnvelope.validate_typed_payload` / `typed_payload` — `backend/app/a2a.py`. Validate payload type against the declared artifact and reject self-addressed messages; downstream handlers receive the reconstructed domain model.
- `A2AEnvelope.from_artifact` — `backend/app/a2a.py`. Builds a versioned envelope from a type-compatible artifact, IDs, roles, and trace ID; a mismatch fails before delivery.
- `InMemoryHandoffLog.__init__` / `record` — `backend/app/a2a.py`. Initialize a locked process-local audit map and idempotently record an envelope by message ID; callers receive defensive copies.
- `InMemoryHandoffLog.mark_delivered` / `mark_failed` / `_update` — `backend/app/a2a.py`. Atomically increment attempts and move a known record to its delivery outcome; unknown IDs fail and the run timeline consumes the updated copy.
- `InMemoryHandoffLog.list_for_run` — `backend/app/a2a.py`. Returns defensive copies ordered by creation time for one run; it writes nothing.

### Artifact storage

- `_digest` / `_safe_relative_key` / `_file_uri_path` — `backend/app/artifacts.py`. Hash bytes, reject absolute/traversing keys, and safely decode local file URIs; store methods use them to enforce content and path integrity. <!-- flow-ref: backend/app/artifacts.py::_digest --> <!-- flow-ref: backend/app/artifacts.py::_safe_relative_key --> <!-- flow-ref: backend/app/artifacts.py::_file_uri_path -->
- `LocalArtifactStore.__init__` — `backend/app/artifacts.py`. Resolves and creates the configured local root; invalid filesystem permissions fail initialization.
- `LocalArtifactStore.put_bytes` / `put_json` — `backend/app/artifacts.py`. Validate the relative key, write through a temporary file, atomically replace the target, and return a hashed `ArtifactRef`; subsequent state may persist that reference.
- `LocalArtifactStore.get_bytes` / `get_json` / `exists` — `backend/app/artifacts.py`. Restrict reads to the configured root and verify hash/size before returning content; missing, escaped, or corrupt artifacts fail closed.
- `GCSArtifactStore.__init__` / `_blob_name` / `_blob_for_ref` — `backend/app/artifacts.py`. Lazily initialize the GCS bucket and constrain keys/references to its configured prefix; missing SDK/credentials or cross-bucket references fail.
- `GCSArtifactStore.put_bytes` / `put_json` — `backend/app/artifacts.py`. Upload immutable bytes and return their hash, size, type, and `gs://` URI for manifests.
- `GCSArtifactStore.get_bytes` / `get_json` / `exists` — `backend/app/artifacts.py`. Download or check an in-prefix object and verify content integrity before downstream use; provider failures propagate to retry policy.

### Persistence

- `InMemoryRunRepository.__init__` / `create_run` / `get_run` — `backend/app/repository.py`. Maintain locked process-local run/event maps, reject duplicate IDs, and return defensive copies; the API and orchestrator consume the state. Missing runs raise `RunNotFoundError`.
- `InMemoryRunRepository.save_run` / `list_runs` — `backend/app/repository.py`. Enforce optimistic versions, stamp the next version, and return bounded newest-first snapshots; stale writes raise `VersionConflictError`.
- `InMemoryRunRepository.append_event` / `list_events` — `backend/app/repository.py`. Append known-run events idempotently and return ordered events after an optional ID; unknown runs fail and SSE consumes the list.
- `FirestoreRunRepository.__init__` / `_run_ref` — `backend/app/repository.py`. Lazily initialize the synchronous Firestore client and resolve a run document; missing cloud SDK or credentials fail cloud startup.
- `FirestoreRunRepository.create_run` / `get_run` / `list_runs` — `backend/app/repository.py`. Execute Firestore operations in worker threads and validate stored documents as `RunState`; duplicate or missing records become typed repository errors.
- `FirestoreRunRepository.save_run` — `backend/app/repository.py`. Uses a Firestore transaction to compare and increment the run version; concurrent stale writers fail without overwriting state.
- `FirestoreRunRepository.append_event` / `list_events` — `backend/app/repository.py`. Persist event subdocuments and query by creation time after an optional cursor event; API streaming consumes the validated results.

### Orchestration and local explanation provider

- `_metric` / `decide_promotion` / `_provenance_complete` — `backend/app/orchestrator.py`. Read required comparison metrics, apply the fixed 5-point improvement/2-point regression/non-decreasing-validity/provenance gates, and return reasons; missing metrics or evidence reject or fail the decision path. <!-- flow-ref: backend/app/orchestrator.py::_metric --> <!-- flow-ref: backend/app/orchestrator.py::decide_promotion --> <!-- flow-ref: backend/app/orchestrator.py::_provenance_complete -->
- `Orchestrator.__init__` — `backend/app/orchestrator.py`. Receives repository and decision-provider ports, constructs all eight wrappers, and creates a process-local trajectory cache.
- `Orchestrator.step` — `backend/app/orchestrator.py`. Loads one run, handles terminal/cancel state, dispatches exactly one phase handler, and records a sanitized failure transition if work raises.
- `Orchestrator.auto` — `backend/app/orchestrator.py`. Repeats `step` until a terminal phase with a 16-step safety bound; overflow fails rather than looping indefinitely.
- `Orchestrator.cancel` — `backend/app/orchestrator.py`. Idempotently moves a non-terminal run to cancelled and emits an event; completed or failed states remain unchanged.
- `Orchestrator._benchmark` / `_analyze` / `_research` / `_curate` — `backend/app/orchestrator.py`. Invoke the first four specialist wrappers, persist each typed output and next phase, and append the corresponding event; missing prerequisites fail the run.
- `Orchestrator._design` / `_train` / `_evaluate` / `_promote` — `backend/app/orchestrator.py`. Enforce budget, create/replace the current experiment, require successful training, attach evaluation, apply deterministic promotion, update the champion only on pass, and either complete or begin the final allowed attempt.
- `Orchestrator._require_run` / `_current_experiment` — `backend/app/orchestrator.py`. Resolve required state/current experiment; missing values raise typed failures before mutation.
- `Orchestrator._save` / `_save_replaced_experiment` — `backend/app/orchestrator.py`. Construct the next immutable state and persist it using the caller's version, preventing lost updates.
- `Orchestrator._event` / `_set_terminal` — `backend/app/orchestrator.py`. Build and append typed events around terminal/state transitions; repository failures propagate so completion is not falsely reported.
- `_digest` / `_artifact` — `backend/app/demo.py`. Create deterministic `demo://` artifact metadata for local explanation mode only; it never qualifies as cloud provenance. <!-- flow-ref: backend/app/demo.py::_digest --> <!-- flow-ref: backend/app/demo.py::_artifact -->
- `LocalDemoDecisionProvider.benchmark` / `analyze_failures` / `form_hypothesis` — `backend/app/demo.py`. Generate fixed train-side explanatory trajectories, grounded clusters, and a cited hypothesis; absent measured failure evidence raises instead of fabricating a cluster.
- `LocalDemoDecisionProvider.curate_dataset` / `design_training` / `launch_training` / `evaluate` — `backend/app/demo.py`. Exercise dataset/config/job/evaluation contracts with deterministic `EXPLANATION` artifacts; provenance remains incomplete so the gate cannot promote them.
- `verify_demo` — `backend/app/demo.py`. Re-evaluates the latest candidate through the supplied provider without changing run state; a run without candidates is rejected. <!-- flow-ref: backend/app/demo.py::verify_demo -->

### RAG leakage boundary

- `KnowledgeDocument.reject_heldout_material` — `backend/app/rag.py`. Rejects held-out or regression-scoped documents at construction, before an index can observe them.
- `_tokens` — `backend/app/rag.py`. Produces stable lowercase lexical terms without external calls; chunk ranking consumes them. <!-- flow-ref: backend/app/rag.py::_tokens -->
- `chunk_document` — `backend/app/rag.py`. Rechecks scope, validates minimum chunk size, and creates deterministic non-overlapping chunks; forbidden input raises `LeakageBoundaryError`. <!-- flow-ref: backend/app/rag.py::chunk_document -->
- `LeakageSafeRAG.__init__` / `ingest` — `backend/app/rag.py`. Maintain an in-memory chunk map, validate every document before committing any pending chunks, and return the count indexed.
- `LeakageSafeRAG.search` / `chunk_count` — `backend/app/rag.py`. Rank allowed chunks by deterministic lexical overlap and return bounded cited excerpts; empty queries return no results and corrupted forbidden chunks are skipped.

### Telemetry

- `safe_attributes` — `backend/app/telemetry.py`. Applies a sensitive-name denylist plus an explicit metadata allowlist and scalar conversion; exporters see only the returned attributes. <!-- flow-ref: backend/app/telemetry.py::safe_attributes -->
- `NoOpSpan.set_attribute` / `record_exception` / `set_status` — `backend/app/telemetry.py`. Preserve call compatibility without storing data when OpenTelemetry is unavailable.
- `_set_attributes` / `telemetry_span` / `async_telemetry_span` — `backend/app/telemetry.py`. Sanitize attributes and open a real or no-op span around synchronous/asynchronous work; application exceptions propagate unchanged. <!-- flow-ref: backend/app/telemetry.py::_set_attributes --> <!-- flow-ref: backend/app/telemetry.py::telemetry_span --> <!-- flow-ref: backend/app/telemetry.py::async_telemetry_span -->
- `current_trace_id` — `backend/app/telemetry.py`. Returns the active valid trace ID or `None` without requiring the SDK; events and A2A envelopes consume it. <!-- flow-ref: backend/app/telemetry.py::current_trace_id -->
- `configure_cloud_trace` — `backend/app/telemetry.py`. Installs a service-named GCP exporter/provider from project configuration; missing optional dependencies fail cloud setup explicitly. <!-- flow-ref: backend/app/telemetry.py::configure_cloud_trace -->

### AgentGym WebShop adapter

- `tool_call_to_action` — `backend/app/webshop.py`. Maps validated FunctionGemma `search(keywords)` and `click(item)` calls to WebShop syntax and rejects unsupported tools, empty values, or bracket injection. <!-- flow-ref: backend/app/webshop.py::tool_call_to_action -->
- `AgentGymWebShopClient.__init__` / `__aenter__` / `__aexit__` — `backend/app/webshop.py`. Configure an async HTTP client, create an environment on entry, and close it on exit.
- `AgentGymWebShopClient.create` / `_require_env` — `backend/app/webshop.py`. Request and validate an integer environment ID; all later methods reject use before creation.
- `AgentGymWebShopClient.reset` / `observation` / `available_actions` — `backend/app/webshop.py`. Call the corresponding AgentGym endpoints and validate response shapes; HTTP and protocol errors propagate to execution recovery.
- `AgentGymWebShopClient.step` — `backend/app/webshop.py`. Convert a tool call if needed, execute it, and validate state/reward/done/info into `WebShopStep`; malformed responses raise `WebShopProtocolError`.
- `AgentGymWebShopClient.rollout` — `backend/app/webshop.py`. Reset one session and execute actions until completion, returning the last objective reward.
- `AgentGymWebShopClient.verify_repair` — `backend/app/webshop.py`. Replay the original and candidate complete continuations from the same session/prefix and mark verified only for strictly higher reward.
- `AgentGymWebShopClient.close` — `backend/app/webshop.py`. Close the remote environment when created, clear its ID, and close the HTTP client; provider failures remain visible.

### Documentation enforcement

- `changed_files` — `backend/scripts/check_docs_sync.py`. Runs a read-only three-dot Git diff from the supplied base revision and returns changed paths; Git errors fail the documentation check. <!-- flow-ref: backend/scripts/check_docs_sync.py::changed_files -->
- `requires_changelog` — `backend/scripts/check_docs_sync.py`. Classifies application, test, operational, packaging, and checker changes as requiring a changelog entry. <!-- flow-ref: backend/scripts/check_docs_sync.py::requires_changelog -->
- `validate_required_documents` — `backend/scripts/check_docs_sync.py`. Confirms all living documents exist and appends failures to the caller-owned error list. <!-- flow-ref: backend/scripts/check_docs_sync.py::validate_required_documents -->
- `validate_local_links` — `backend/scripts/check_docs_sync.py`. Resolves Markdown links inside the repository boundary and reports missing or escaping local targets. External links are not fetched. <!-- flow-ref: backend/scripts/check_docs_sync.py::validate_local_links -->
- `top_level_python_symbols` — `backend/scripts/check_docs_sync.py`. Parses a Python file with `ast` and returns documented top-level function/class names. <!-- flow-ref: backend/scripts/check_docs_sync.py::top_level_python_symbols -->
- `validate_flow_references` — `backend/scripts/check_docs_sync.py`. Validates explicit `flow-ref` paths and top-level Python symbols against source. <!-- flow-ref: backend/scripts/check_docs_sync.py::validate_flow_references -->
- `validate_changelog` — `backend/scripts/check_docs_sync.py`. On pull requests, rejects tracked implementation changes when `ChangeLog.md` is absent from the diff. <!-- flow-ref: backend/scripts/check_docs_sync.py::validate_changelog -->
- `main` — `backend/scripts/check_docs_sync.py`. Runs all documentation checks, prints actionable errors, and returns a CI-compatible status code. <!-- flow-ref: backend/scripts/check_docs_sync.py::main -->
