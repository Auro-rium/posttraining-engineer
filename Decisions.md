# Technical and Product Decisions

This file is append-only. A later decision may supersede an earlier one, but existing records must not be rewritten. No entry may include secrets, private prompts, or held-out task contents.

## DEC-001 — Python, FastAPI, and Google ADK for backend services

- **Date / run:** 2026-08-15 Asia/Kolkata / `BUILD-2026-08-15-001`
- **Status:** Accepted
- **Context:** The demo needs Google ADK agents, typed HTTP interfaces, cloud SDK integration, and ML training orchestration.
- **Decision:** Implement the backend in Python 3.12 with FastAPI and Google ADK.
- **Alternatives:** TypeScript services; mixed TypeScript and Python services.
- **Reason:** ADK and the model-training ecosystem are Python-native, while FastAPI maps Pydantic contracts directly to an inspectable API. One language keeps the hackathon implementation small.
- **Trade-offs:** Python provides less compile-time enforcement than TypeScript and needs runtime schema validation.
- **Affected components:** All backend services, schemas, tests, and the container image.
- **Validation:** Type checks, schema tests, API tests, and successful ADK service startup.

## DEC-002 — Three service roles from one immutable image

- **Date / run:** 2026-08-15 Asia/Kolkata / `BUILD-2026-08-15-001`
- **Status:** Accepted
- **Context:** Eight responsibilities must be visible, but eight deployments would add cost and operational work without strengthening the demo.
- **Decision:** Run one image in `coordinator`, `research`, and `execution` modes selected by `SERVICE_ROLE`. Keep eight logical specialist contracts within those boundaries.
- **Alternatives:** One monolithic deployment; eight independent services and images.
- **Reason:** Three boundaries demonstrate real A2A handoffs while retaining a tractable deployment and a single dependency set.
- **Trade-offs:** Roles share a release cadence, and one image contains code unused by a given process.
- **Affected components:** Service bootstrap, Docker Compose, Cloud Run, A2A routing, and telemetry.
- **Validation:** Start all three roles from the same image and complete coordinator-to-research-to-execution handoffs.

## DEC-003 — Firestore for state and Cloud Storage for immutable artifacts

- **Date / run:** 2026-08-15 Asia/Kolkata / `BUILD-2026-08-15-001`
- **Status:** Accepted
- **Context:** Runs need queryable typed state, while datasets, checkpoints, and reports can be large and must retain provenance.
- **Decision:** Use Firestore as canonical run/event metadata and a versioned GCS bucket for content-addressed artifacts. Use memory and local-file adapters for tests and local development.
- **Alternatives:** PostgreSQL for all state; GCS-only state; local filesystem only.
- **Reason:** The split is small, serverless, and native to the selected Google Cloud architecture without treating large artifacts as database documents.
- **Trade-offs:** Cross-store updates are not atomic, so state records must refer only to completed, hashed artifacts.
- **Affected components:** Repositories, artifact manifests, Terraform, and recovery paths.
- **Validation:** Adapter contract tests and a cloud smoke run that reads every recorded artifact URI and verifies its hash.

## DEC-004 — Deterministic verification and checkpoint promotion

- **Date / run:** 2026-08-15 Asia/Kolkata / `BUILD-2026-08-15-001`
- **Status:** Accepted
- **Context:** Letting Gemini approve its own repaired data or checkpoint would make the result subjective and vulnerable to leakage.
- **Decision:** Use Gemini only for research proposals. Admit repairs through environment replay, enforce budgets in code, compare candidates on identical held-out inputs, and apply promotion thresholds in deterministic code.
- **Alternatives:** LLM-as-judge; manual judge approval; highest observed score without regression gates.
- **Reason:** Objective promotion is the central credibility claim of the hackathon demo.
- **Trade-offs:** The system cannot promote plausible improvements that fail the fixed gate, and a small held-out set may have limited statistical power.
- **Affected components:** Data Curator, evaluator, Champion Manager, run state, and tests.
- **Validation:** Tests for invalid repairs, budget exhaustion, missing provenance, insufficient improvement, and regression rejection.

## DEC-005 — SSE for resumable run events

- **Date / run:** 2026-08-15 Asia/Kolkata / `BUILD-2026-08-15-001`
- **Status:** Accepted
- **Context:** Clients need ordered progress updates, but the public control surface is request/response and does not need bidirectional sockets.
- **Decision:** Expose run events through Server-Sent Events and use ordinary POST requests for commands.
- **Alternatives:** WebSockets; repeated polling only.
- **Reason:** SSE is a smaller unidirectional protocol with native event IDs and straightforward reconnection.
- **Trade-offs:** Clients cannot issue commands on the event connection and must use separate POST requests.
- **Affected components:** Coordinator API, event store, client contract, and reconnect tests.
- **Validation:** API tests reconnect with the last event ID and receive only subsequent events in order.

## DEC-006 — OpenTelemetry with metadata-only cloud export

- **Date / run:** 2026-08-15 Asia/Kolkata / `BUILD-2026-08-15-001`
- **Status:** Accepted
- **Context:** Judges need evidence of agent handoffs and cloud jobs, while prompts, credentials, and sealed tasks must not leak through observability.
- **Decision:** Instrument agent, RAG, A2A, training, and evaluation boundaries with OpenTelemetry. Export identifiers, timing, token counts, outcomes, and sanitized provider metadata—not content.
- **Alternatives:** Full prompt capture; application logs only; no telemetry.
- **Reason:** Metadata provides end-to-end correlation without weakening the evaluation or credential boundary.
- **Trade-offs:** Debugging content-specific failures requires controlled local reproduction rather than reading production traces.
- **Affected components:** Telemetry helpers, agent calls, cloud adapters, logging policy, and leakage tests.
- **Validation:** Span tests assert the allowlist and reject sensitive attribute names or held-out payloads.

## DEC-007 — Locked uv environments and one append-only change record

- **Date / run:** 2026-08-15 Asia/Kolkata / `BUILD-2026-08-15-001`
- **Status:** Accepted
- **Context:** The image, local setup, and CI must resolve the same backend dependencies, and future agents need a reliable history of repository mutations.
- **Decision:** Use `pyproject.toml` plus `uv.lock` across local development, CI, and Docker. Require `ChangeLog.md` to change whenever implementation or operational files change in a pull request.
- **Alternatives:** Separate requirements files; unpinned pip installation; manually requested documentation updates.
- **Reason:** A single lock removes dependency drift, while a mechanical CI gate makes the living-document promise enforceable.
- **Trade-offs:** Dependency edits require refreshing the lock, and documentation-only fixups may be needed before a pull request passes.
- **Affected components:** Backend packaging, Dockerfile, CI, and documentation checker.
- **Validation:** Frozen dependency installation, container build, tests, and a negative changelog-gate test.

## DEC-008 — Separate objective evidence worker

- **Date / run:** 2026-08-15 Asia/Kolkata / `BUILD-2026-08-15-002`
- **Status:** Accepted; extends DEC-002
- **Context:** Gemini/ADK agents may propose research decisions but must not manufacture FunctionGemma trajectories, replay verdicts, evaluation metrics, or training artifacts.
- **Decision:** Require a separately deployed, sandboxed objective worker behind authenticated HTTPS. It owns FunctionGemma inference, AgentGym, repair replay, sealed evaluation, and artifact hashing through `/v1/benchmark`, `/v1/verify-curation`, `/v1/evaluate`, and `/v1/training-evidence`. Terraform accepts its URL but does not provision it.
- **Alternatives:** Run objective tools inside the LLM services; trust structured Gemini output; simulate evidence locally.
- **Reason:** A hard process and credential boundary preserves the claim that research is agentic while evidence and promotion remain independently verifiable.
- **Trade-offs:** Deployment has an external prerequisite and another identity/latency boundary; this repository alone cannot execute a real run.
- **Affected components:** Research/execution A2A services, cloud provider, IAM instructions, Terraform inputs, and evidence tests.
- **Validation:** Contract tests reject absent/malformed/explanatory evidence; a live smoke run must verify all four worker endpoints and artifact hashes.

## DEC-009 — GCS-pinned leakage-safe RAG

- **Date / run:** 2026-08-15 Asia/Kolkata / `BUILD-2026-08-15-002`
- **Status:** Accepted
- **Context:** Research hypotheses must cite useful prior knowledge without exposing held-out or regression tasks to Gemini.
- **Decision:** Load a pre-uploaded GCS JSON corpus, optionally pin its SHA-256, validate every record through `LeakageSafeRAG`, perform deterministic lexical retrieval, and require Gemini to select a nonempty subset of canonical retrieved citations.
- **Alternatives:** Vertex AI vector search; prompt-only documentation; uncited model knowledge.
- **Reason:** The lexical index is sufficient for the hackathon scope, credential-light, deterministic, and allows the leakage boundary to be enforced before prompting.
- **Trade-offs:** Retrieval quality is below an embedding index and the operator must build/upload the corpus separately.
- **Affected components:** Research service settings, hypothesis operation, telemetry, Terraform inputs, and leakage tests.
- **Validation:** Tests cover forbidden scopes, hash mismatch, empty retrieval, invented citations, and successful canonical citation grounding.

## DEC-010 — Best-effort automatic mode with bounded request execution

- **Date / run:** 2026-08-15 Asia/Kolkata / `BUILD-2026-08-15-002`
- **Status:** Accepted for the hackathon; durability follow-up required
- **Context:** `/auto` is an in-process task and Vertex polling can be long. Firestore persists completed phases but not the in-flight Vertex job handle.
- **Decision:** Keep one coordinator instance warm with CPU always allocated, set Cloud Run and A2A timeouts below 60 minutes, cap compute at 55 minutes, label `/auto` best-effort, and use explicit `/step` calls for the judge path.
- **Alternatives:** Cloud Workflows/Tasks with durable polling; synchronous multi-hour requests; claiming Firestore alone makes work resumable.
- **Reason:** This is the smallest honest deployment for the hackathon while making the interruption boundary explicit.
- **Trade-offs:** Coordinator minimum instances cost money, process replacement can interrupt auto mode, and a lost training poll requires operator inspection before retry.
- **Affected components:** Cloud Run resources, settings, API documentation, orchestration operations, and runbook.
- **Validation:** API concurrency/cancel tests plus a future restart test after the Vertex handle is persisted.

## DEC-011 — Dependency-ordered Cloud Run creation

- **Date / run:** 2026-08-15 Asia/Kolkata / `BUILD-2026-08-15-002`
- **Status:** Accepted
- **Context:** Coordinator needs exact generated research/execution Cloud Run URIs, but all three services in one `for_each` cannot self-reference without a Terraform cycle.
- **Decision:** Create the research and execution services first, then create the coordinator with their computed URIs in one apply. Keep optional URL variables only for a second pass that corrects each team's published Agent Card self URL.
- **Alternatives:** Manual two-pass coordinator routing; service discovery; hard-coded predictable URLs.
- **Reason:** The deployable call graph becomes correct automatically while isolating the unavoidable self-URL issue to optional metadata.
- **Trade-offs:** Terraform contains separate team/coordinator resource blocks and an exact Agent Card URL may still need one optional reapply.
- **Affected components:** Terraform Cloud Run resources, outputs, IAM bindings, and deployment instructions.
- **Validation:** HCL/provider validation and a deployed check that coordinator environment URLs equal Terraform team outputs.

## DEC-012 — Ignore generated live-document snapshots

- **Date / run:** 2026-08-15 Asia/Kolkata / `MAINT-2026-08-15-001`
- **Status:** Accepted
- **Context:** Runtime or tooling integrations may create local live-document snapshots that should not be committed alongside the canonical append-only living documents.
- **Decision:** Ignore `live-docs/` and `live_docs/` directories while keeping root `ChangeLog.md`, `Decisions.md`, and `Flow.md` tracked.
- **Alternatives:** Ignore all Markdown files; commit every generated snapshot; use a machine-specific global ignore rule.
- **Reason:** Generated snapshots are environment artifacts, while the root documents are the repository’s shared source of truth.
- **Trade-offs:** A generated snapshot must be copied into the canonical documents deliberately if it contains information worth preserving.
- **Affected components:** Repository hygiene and documentation maintenance.
- **Validation:** Git status remains clean when either generated snapshot directory is present; canonical living documents remain trackable.

## DEC-013 — Expand fine-tuning schema without inventing WebShop actions

- **Date / run:** 2026-08-15 Asia/Kolkata / `CHANGE-2026-08-15-003`
- **Status:** Accepted
- **Context:** FunctionGemma fine-tuning needs an explicit, useful tools section, but native AgentGym WebShop exposes only `search` and `click` actions. Adding checkout-specific functions would create training labels the evaluator cannot execute.
- **Decision:** Ship canonical JSON function schemas for `search(keywords)` and `click(item)`, with richer descriptions and strict required arguments, and inject them into training rows when a row omits its tools field.
- **Alternatives:** Add synthetic `add_to_cart`/`checkout` functions; keep tools omitted from rows; replace WebShop with a custom commerce environment.
- **Reason:** This increases the quality and consistency of the fine-tuning signal while preserving environment compatibility and deterministic replay.
- **Trade-offs:** The model still has two native action names; broader tool coverage requires a separately scoped environment adapter and benchmark.
- **Affected components:** `backend/app/webshop_tools.py`, training renderer, WebShop tests, and living documentation.
- **Validation:** Schema tests assert both functions, required arguments, full test suite, Ruff, mypy, and documentation sync.

## DEC-014 — Revert the optional fine-tuning tool-schema expansion

- **Date / run:** 2026-08-15 Asia/Kolkata / `REVERT-2026-08-15-001`
- **Status:** Supersedes DEC-013
- **Context:** The requested tool-schema expansion was not part of the desired implementation step.
- **Decision:** Restore the prior training renderer and WebShop contract; retain only the native environment action handling already present.
- **Reason:** Keep the backend scoped to the previously approved behavior.
- **Affected components:** Training renderer, WebShop tests, and flow documentation.
- **Validation:** Full backend tests, Ruff, mypy, and documentation sync after the revert.

## DEC-015 — Add Wordle, BabyAI, and Movie environment contracts

- **Date / run:** 2026-08-15 Asia/Kolkata / `CHANGE-2026-08-15-004`
- **Status:** Accepted
- **Context:** The hackathon now needs broader AgentGym coverage than WebShop while retaining one shared post-training loop.
- **Decision:** Accept `AgentGym/WebShop`, `AgentGym/Wordle`, `AgentGym/BabyAI`, and `AgentGym/Movie` at the API boundary. Route action discovery and deterministic replay through the selected external objective worker; keep the local explanation provider WebShop-only.
- **Alternatives:** Implement separate local simulators; accept arbitrary environment strings; keep WebShop-only.
- **Reason:** Typed allow-listing prevents unsupported or misspelled environments while avoiding fabricated local metrics and keeping environment-specific action schemas inside their authoritative workers.
- **Trade-offs:** New environments can be created through the API but require deployed objective-worker support before a live run; the local fixture cannot execute them.
- **Affected components:** Environment registry, run API, ADK prompts, cloud objective-worker contract, tests, README, and Flow.
- **Validation:** API acceptance tests for all three environments plus full tests, Ruff, mypy, and documentation sync.

## DEC-016 — Compile explicit role prompts and tool allow-lists

- **Date / run:** 2026-08-21 Asia/Kolkata / `CHANGE-2026-08-21-001`
- **Status:** Accepted
- **Context:** Eight long-horizon agents need predictable boundaries and a concrete sharing protocol instead of relying on implicit swarm memory.
- **Decision:** Compile every role prompt with an explicit allowed-tool list, typed handoff target, resumability rules, retry policy, and shared-state protocol. Tools represent server-side capabilities; agents cannot call peers directly or mutate arbitrary state.
- **Alternatives:** Give every agent the same broad tool set; rely on prose-only role descriptions; use an untyped shared message bus.
- **Reason:** Least-privilege tool access and typed A2A/Firestore/GCS handoffs make failures auditable, resumable, and testable across long runs.
- **Trade-offs:** Adding a new capability requires updating the role allow-list and contract tests; prompts are longer but more deterministic.
- **Affected components:** ADK prompt compilation, role tests, A2A handoffs, Firestore state, GCS artifacts, telemetry, and Flow documentation.
- **Validation:** Prompt tests assert every role has allowed tools, a handoff, safety clauses, and the common resumability protocol.

## DEC-017 — Make AgentEval the explicit sealed evaluation suite

- **Date / run:** 2026-08-21 Asia/Kolkata / `CHANGE-2026-08-21-002`
- **Status:** Accepted
- **Context:** The evaluation path was sealed and objective but did not identify which AgentGym benchmark produced its metrics.
- **Decision:** Set `AgentGym/AgentEval` (`agent-eval-v1`) as the typed evaluation suite. Require live evaluation reports to include a lowercase SHA-256 hash of the immutable AgentEval manifest and reject reports whose suite/version does not match the request.
- **Alternatives:** Keep the suite implicit; allow each worker to choose an arbitrary held-out set; expose task contents to the coordinator.
- **Reason:** Explicit suite provenance makes champion/candidate comparisons reproducible without leaking evaluation tasks.
- **Trade-offs:** The external objective worker must pin and return the AgentEval manifest hash before a candidate can be promoted.
- **Affected components:** Evaluation request/report schemas, cloud provider validation, objective-worker contract, tests, and deployment documentation.
- **Validation:** Contract tests cover live manifest requirements and the full test, type, lint, and documentation suites.

## DEC-018 — Persist and reconcile in-flight training work

- **Date / run:** 2026-08-21 Asia/Kolkata / `CHANGE-2026-08-21-003`
- **Status:** Accepted
- **Context:** A process restart during Vertex polling previously lost the in-memory job handle and could cause an unsafe duplicate submission.
- **Decision:** Persist a `PENDING`/`RUNNING` `TrainingResult.job_id` before polling, use a deterministic Vertex display name to reconcile a submission after a crash, and resume all non-terminal Firestore runs on service startup.
- **Alternatives:** Keep `/auto` process-local; ask an operator to inspect every restart; introduce a new workflow engine during the hackathon.
- **Reason:** Durable state plus provider-side idempotent lookup makes the existing coordinator recoverable without adding another platform dependency.
- **Trade-offs:** Recovery still depends on Vertex listing permissions and the external objective worker; a provider outage remains a real blocker and fails closed.
- **Affected components:** Orchestrator, coordinator startup, Vertex launcher, cloud provider, tests, README, and Flow.
- **Validation:** Recovery tests cover persisted job reconciliation; full tests, Ruff, mypy, documentation sync, and diff checks pass.

## DEC-019 — Strands Agents as the AWS hackathon runtime

- **Date / run:** 2026-09-06 / `AWS-STRANDS-HACKATHON-001`
- **Status:** Accepted; supersedes the AWS-facing runtime choice only
- **Context:** The AWS Agents for Humans hackathon requires Strands Agents and evaluates real professional workflow execution; a frontend is not required for the backend submission.
- **Decision:** Add a Strands-native specialist package and retain the state-driven eight-role workflow. Keep the scientific record in explicit run state and artifacts rather than agent memory. AgentCore remains an optional deployment target, not a source of truth.
- **Alternatives:** Keep Google ADK as the primary runtime; replace the workflow with one general-purpose agent; build a frontend before validating the backend loop.
- **Reason:** Strands satisfies the mandatory platform requirement while preserving role specialization and deterministic evaluation/promotion boundaries.
- **Trade-offs:** The migration currently coexists with legacy Google-specific modules; live Bedrock/SageMaker/AgentCore execution still requires AWS deployment configuration.
- **Affected components:** Strands agent package, orchestrator, service-recovery environment, API bootstrap, backend packaging, and architecture documentation.
- **Validation:** Compilation, specialist initialization, full local workflow, and API smoke tests pass; no live AWS training or model improvement is claimed.

## DEC-020 — Remove legacy Google implementation from AWS submission

- **Date / run:** 2026-09-06 / `AWS-STRANDS-HACKATHON-002`
- **Status:** Accepted
- **Context:** The repository is being submitted to the AWS Agents for Humans hackathon; retaining the earlier Google ADK/Vertex/Firestore/GCS path made the runtime, dependencies, tests, and deployment instructions contradictory.
- **Decision:** Remove the legacy Google source modules, Google-only tests and training worker, and Google Terraform stack. Keep the AWS Strands workflow and retain only concise historical notes in this append-only file and `ChangeLog.md`.
- **Alternatives:** Keep both runtimes; move the Google implementation to a separate branch; silently leave the mixed documentation.
- **Reason:** One authoritative AWS path is easier for judges to install, inspect, and run, and avoids presenting stale Google infrastructure as part of the submission.
- **Trade-offs:** Historical Google implementation is no longer runnable from this checkout; live AWS integrations remain to be implemented and verified.
- **Affected components:** `backend/app`, `backend/tests`, `backend/training`, `infra/terraform`, `.env.example`, Docker/CI configuration, and living documentation.
- **Validation:** AWS-only import/compile and smoke checks must pass; no live AWS capability is claimed by this cleanup.

## DEC-021 — Isolate deterministic continuous post-training domain logic

- **Date / run:** 2026-09-06 / `POSTTRAINING-DOMAIN-001`
- **Status:** Accepted
- **Context:** Continuous cycles need an auditable record of immutable artifacts, objective evidence, promotion decisions, and rollback history without relying on Strands agent output or wall-clock/random state.
- **Decision:** Add an independent `app.posttraining` package with content-addressed artifact models, provenance-bearing evidence models, a fail-closed promotion gate, explicit approval/rejection/rollback transitions, and fixed-seed callback-based benchmark utilities.
- **Alternatives:** Extend the agent classes directly; use untyped dictionaries; let the Champion Manager decide from generated text; use process-global randomness for benchmark ordering.
- **Reason:** Keeping these rules as typed, deterministic domain primitives makes the continuous path testable and prevents simulated or incompatible evidence from being promoted.
- **Trade-offs:** Callers must provide lowercase SHA-256 digests and compatible verified evaluation manifests; benchmarks still need an objective predictor adapter to run live model evaluations.
- **Affected components:** `backend/app/posttraining`, focused post-training tests, and the backend test discovery configuration.
- **Validation:** Focused pytest, Ruff, mypy, and diff checks pass; no live AWS or model evaluation was performed.

## DEC-022 — Keep continuous post-training HTTP control plane independently injectable

- **Date / run:** 2026-09-06 / `POSTTRAINING-API-001`
- **Status:** Accepted
- **Context:** Trace intake and operator cycle decisions need a typed HTTP boundary, while the existing AWS Strands coordinator bootstrap and local domain primitives are changing independently.
- **Decision:** Add an isolated FastAPI router under `backend/app/api/continuous_post_training.py` with repository and service protocols, app-state dependency resolution, and a lock-protected in-memory repository for local tests. Do not import or mutate `app.main`.
- **Alternatives:** Add routes directly to `main.py`; use a module-global mutable store; couple the API to a concrete AWS or training provider.
- **Reason:** Explicit inclusion and dependency overrides make the contract testable and allow a durable adapter to be introduced without changing handlers. Approval records intent and queues work; it does not claim that training or artifacts exist.
- **Trade-offs:** The default repository is process-local and the router does not provide authentication, background execution, or durable artifact storage until the host application supplies those integrations.
- **Affected components:** `backend/app/api/continuous_post_training.py`, API tests, and integration documentation.
- **Validation:** Focused API pytest, Ruff, mypy, Python compilation, documentation sync, and diff checks pass; no live AWS or model training run was performed.

## DEC-023 — Add a fail-closed AWS deployment foundation

- **Date / run:** 2026-09-06 / `AWS-DEPLOYMENT-FOUNDATION-001`
- **Status:** Accepted
- **Context:** The Strands local workflow needed explicit AWS configuration, role discovery, and deployable storage/runtime foundations without adding a frontend.
- **Decision:** Add typed runtime settings, wire the continuous control-plane router into the app, publish role Agent Cards, and provision S3, DynamoDB, ECR, ECS, IAM, VPC, and CloudWatch foundations with CDK. Keep live AWS orchestration disabled until provider-backed state, artifacts, training, and evaluation are connected.
- **Reason:** Explicit fail-closed boundaries prevent the local simulation from being presented as live evidence while making the next deployment step reproducible.
- **Trade-offs:** CDK infrastructure is not sufficient for a live model-improvement claim; the current workflow remains local until provider wiring is completed.
- **Affected components:** `backend/app/runtime_config.py`, `backend/app/main.py`, `infra/cdk/`, and deployment documentation.
- **Validation:** Pytest, compilation, CDK synthesis, documentation sync, and diff checks pass.

## DEC-024 — Bound comparable runs and make observation metadata-only

- **Date / run:** 2026-09-08 / `REAL-POSTTRAINING-RUNS-001`
- **Status:** Accepted
- **Context:** The hackathon demo needs repeatable improvement evidence across multiple candidate runs while keeping telemetry safe to inspect.
- **Decision:** Permit at most five sequential top-level run IDs. Reserve each slot atomically in DynamoDB with a counter and next-run condition; compare only compatible verified evidence. Expose JSON/SVG comparison endpoints. Emit run/phase/job/promotion telemetry with correlation IDs, latency/cost, recursive redaction, immutable attributes, and optional OpenTelemetry.
- **Alternatives:** Process-local counters; random promotion values; raw prompt/completion logging; graphing incomplete records.
- **Reason:** The registry and gate make promotion reproducible, while metadata-only observation prevents task or secret leakage.
- **Trade-offs:** Local mode remains process-local; a live `LIVE` result still requires an external objective worker, pre-existing AWS resources, model checkpoint, and hashed evaluation manifest.
- **Affected components:** Run history, DynamoDB repository, comparison graph/API, objective benchmark/SageMaker lifecycle boundary, runtime configuration, orchestrator guards, telemetry, and documentation.
- **Validation:** Full backend pytest passed; focused Ruff and mypy checks passed. No live AWS training, held-out evaluation, checkpoint improvement, or promotion was executed.

## DEC-025 — Pin Nemotron reasoning and make prompts auditable

- **Date / run:** 2026-09-08 / `NEMOTRON-PROMPT-CONTRACT-001`
- **Status:** Accepted
- **Context:** The live demo needs one predictable reasoning model for every working agent while still allowing useful, creative experiment design. Free-form role prompts make model drift, unsafe evidence claims, and irreproducible handoffs difficult to review.
- **Decision:** Pin all eight specialist agents and coordinator reasoning to NVIDIA Nemotron Super 3 120B, `nvidia.nemotron-super-3-120b`. Keep FunctionGemma as the separate post-training target. Compile each role from a versioned `AgentPromptContract` with typed inputs/outputs, preconditions, stop conditions, evidence labels, sealed-data rules, forbidden actions, and a bounded creativity lane. Persist only the model ID, prompt version, and prompt SHA-256 in manifests and metadata-only telemetry.
- **Alternatives:** Allow per-agent model overrides; use one untyped general prompt; log full prompts and completions for debugging.
- **Reason:** A single pinned reasoning model and explicit contracts make agent behavior comparable across five runs, preserve the held-out boundary, and let Nemotron generate inventive but falsifiable hypotheses without becoming the source of metrics or promotion decisions.
- **Trade-offs:** Changing the reasoning model or a role contract is a provenance change and requires a new prompt version/hash; raw prompt debugging is intentionally unavailable in telemetry.
- **Affected components:** Strands agent factories, prompt contracts, runtime configuration, run manifests, telemetry, live scripts, and documentation.
- **Validation:** Contract tests must verify model identity, prompt structure/hash stability, blocked behavior, redaction, and absence of fabricated evidence. No live SageMaker training or FunctionGemma improvement is claimed by this decision.

## DEC-026 — Show execution as a safe animated observer

- **Date / run:** 2026-09-08 / `EXECUTION-OBSERVER-001`
- **Status:** Accepted
- **Context:** The hackathon audience needs to understand how multiple agents collaborate during a run, including pauses and failures, without treating visual activity as proof of AWS work.
- **Decision:** Provide a browser observer in which role bots move through launcher, benchmark, failure analysis, data curation, training, evaluation, and promotion. Drive state from lifecycle metadata/events and render run comparisons and graphs from the same API records. Animation cannot create approval, advance a phase, or invent a metric, artifact, or provider job.
- **Alternatives:** Show a static dashboard; expose raw agent transcripts; let visual completion imply phase completion.
- **Reason:** A phase-linked observer makes the autonomous workflow legible while preserving the evidence boundary and metadata-only telemetry policy.
- **Trade-offs:** The observer is read-only and may show a blocked or incomplete run; authenticated durable events remain a live deployment prerequisite.
- **Affected components:** Frontend observer, run/event APIs, comparison graph, telemetry, and demo documentation.
- **Validation:** UI tests must verify event-driven phase movement, visible blocked/failed states, one-to-five run comparisons, and the absence of raw prompt/completion/task content.

## DEC-027 — Enforce durable telemetry lifecycle and correlation contracts

- **Date / run:** 2026-09-12 / `TELEMETRY-LIFECYCLE-HARDENING-001`
- **Status:** Accepted
- **Context:** Supervisor transitions did not provide a complete run/phase lifecycle, and unbounded event labels or free-form identifier values could weaken telemetry's metadata-only boundary.
- **Decision:** Restrict durable event records to the finite lifecycle vocabulary; validate recorder identifiers and allow-listed metadata as opaque values; persist run starts, phase outcomes, and terminal events through the durable telemetry bridge; stop the supervisor on durable validation or persistence errors; and attach the durable event ID plus exact autonomous event type to OpenTelemetry spans.
- **Alternatives:** Keep generic state-transition events only; accept arbitrary event strings and metadata values; treat durable validation errors as optional-observer failures.
- **Reason:** Precise, bounded lifecycle records make recovery and audit meaningful without capturing model/task content or making an incomplete durable write look successful.
- **Trade-offs:** Adding event semantics requires updating the model vocabulary; legacy persisted records with unrecognized event types will need explicit migration before they can be decoded.
- **Affected components:** `backend/app/autonomous/models.py`, `backend/app/autonomous/telemetry.py`, `backend/app/autonomous/supervisor.py`, `backend/app/observability.py`, telemetry tests, and `Flow.md`.
- **Validation:** Focused telemetry/supervisor/repository tests, Ruff, mypy, documentation sync, and diff checks; no live AWS run is implied.

## DEC-028 — Separate explanatory coordinator output from objective-worker evidence

- **Date / run:** 2026-09-12 / `OBJECTIVE-CHECKPOINT-EXECUTION-001`
- **Status:** Accepted
- **Context:** The ordinary local `/api/runs` flow uses in-memory state and explanatory fixtures, while the isolated objective service now has a real FunctionGemma execution adapter. Treating both as one path would blur the boundary between a local demo, real checkpoint inference, and a complete AWS post-training run.
- **Decision:** Keep `/api/runs` explicitly labeled `EXPLANATION`; run objective benchmark work only through the separately configured `SERVICE_ROLE=objective` service with a complete local FunctionGemma snapshot, immutable revision, expected snapshot digest, authentication, and artifact persistence. Load model files locally without Hub fallback, execute train/replay only, and admit artifacts only after deterministic verifier replay. Document `/api/live` as a separate, approval-gated AWS control plane; scope deployment/live-run activity to the AWS hackathon project only.
- **Alternatives:** Reuse the coordinator's explanatory fixture as benchmark evidence; resolve a mutable remote model reference at runtime; combine local-demo and AWS-control APIs.
- **Reason:** Explicitly separated paths make it possible to distinguish implementation and contract tests from real model execution and AWS-backed training/evaluation evidence.
- **Trade-offs:** The objective worker requires a separately staged target checkpoint and configured artifact storage. Implemented adapters and passing tests do not prove that a real checkpoint loaded, AWS resources deployed, or a post-training run completed.
- **Affected components:** `.env.example`, `README.md`, `backend/README.md`, `Flow.md`, objective-worker configuration and `/api/live` runbook.
- **Validation:** Objective execution/artifact contract tests passed (26 tests); `backend/scripts/check_docs_sync.py` and `git diff --check` passed. No AWS calls, deployment, real checkpoint inference, SageMaker job, or live-run completion is asserted by this decision.

## DEC-029 — Compare champion and candidate only through the paired sealed evaluator

- **Date / run:** 2026-09-12 / `SEALED-PAIRED-EVALUATION-001`
- **Status:** Accepted
- **Context:** The supervisor requested `split=baseline` through the objective benchmark endpoint, which is restricted to train/replay trajectories. Treating baseline as train would create false held-out provenance.
- **Decision:** Do not benchmark the baseline through the training endpoint. Obtain both active champion and candidate scores from one SageMaker sealed-evaluator report bound to the same evaluation manifest, checkpoint digests, task ordering, paired-outcome digest, and environment aggregates. Reject non-train/replay benchmark requests and fail closed on missing or inconsistent paired evidence.
- **Alternatives:** Map baseline to train; compare a separately measured baseline result; expose hidden tasks through the objective benchmark API.
- **Reason:** One sealed report is the authoritative evidence that both models were evaluated against the same held-out suite and task sequence.
- **Trade-offs:** The objective worker alone cannot establish a baseline; the SageMaker evaluator and its immutable report are prerequisites for promotion.
- **Affected components:** Autonomous supervisor, live objective/evaluation adapters, objective benchmark request contract, evaluator report validation, focused tests, and Flow.
- **Validation:** Focused supervisor, live execution, objective workflow, objective service, and objective execution tests. No AWS run or held-out model evaluation is implied by local tests.

## DEC-030 — Require offline model-load smokes for SageMaker worker images

- **Date / run:** 2026-09-12 / `WORKER-IMAGE-SMOKE-001`
- **Status:** Accepted
- **Context:** Local contract tests and Dockerfile syntax checks do not exercise CUDA/bitsandbytes model loading or evaluator adapter loading, while those failures occur before a live experiment can produce evidence.
- **Decision:** Ship separate network-disabled smoke entrypoints in both amd64 worker images. The trainer smoke loads the mounted staged FunctionGemma snapshot locally in 4-bit and performs exactly one throwaway LoRA optimizer step; the evaluator smoke locally loads that base and adapter and performs one forward pass without sealed tasks. The throwaway adapter is explicitly not a SageMaker artifact or promotion candidate.
- **Alternatives:** Treat image build success as runtime proof; run smoke checks against a mutable Hub model; use real sealed evaluation data to exercise evaluator model loading.
- **Reason:** The checks cover the CUDA, quantization, PEFT, local-path, and adapter-interoperability boundaries without contacting the Hub, generating sealed reports, or confusing smoke outputs with model improvement evidence.
- **Trade-offs:** Trainer smoke requires GPU-capable Docker and both smokes require an already staged local base; actual SageMaker execution remains a separate proof obligation.
- **Affected components:** Trainer/evaluator image entrypoints and Dockerfiles, CDK deployment instructions, worker-flow documentation, and contract tests.
- **Validation:** The local contract test suite verifies offline/local-only loading and one-step/forward behavior is present in image entrypoints. Docker GPU smoke requires an external CUDA runtime and staged checkpoint and is not claimed unless run.

## DEC-031 — Admit only verifier-success trajectories as SFT targets

- **Date / run:** 2026-09-12 / `SFT-SUCCESS-ADMISSION-001`
- **Status:** Accepted
- **Context:** Deterministic replay proves that a trajectory is reproducible, but it does not prove the task succeeded; training on a replayable failed action sequence would reinforce failure.
- **Decision:** Keep verified failures as benchmark evidence but never as SFT rows. Admit a successful original trajectory only after verifier replay succeeds. Admit a repair only when it carries content-addressed source-failure lineage, the objective service resolves and replays that same-task/same-split stored failure, the repair replay succeeds, and artifact persistence rechecks both outcomes.
- **Alternatives:** Treat all replayable trajectories as training data; rely on the curation agent's claim; synthesize or accept repairs without environment replay.
- **Reason:** Only successful environment behavior is a desirable supervised target; exact failed-source linkage keeps correction provenance auditable without allowing the failed original actions into the dataset.
- **Trade-offs:** A fresh run with no successful originals or successful repaired trajectories stops at curation. Existing stored trajectory artifacts without the verifier-success outcome need replay and re-persistence before being used for SFT.
- **Affected components:** Objective trajectory identity and replay, curation service, dataset row contract, in-memory and S3 artifact stores, live dataset handoff, focused tests, and `Flow.md`.
- **Validation:** Focused objective engine, service, and artifact-store tests; backend type/lint and documentation checks. No teacher repair is generated and no live AWS run is implied.

## DEC-032 — Recover durable AWS runs continuously and expose a restricted coordinator route

- **Date / run:** 2026-09-12 / `AWS-COORDINATOR-RECOVERY-INGRESS-001`
- **Status:** Accepted
- **Context:** One-time process startup recovery can block application readiness for the duration of a run and cannot recover work that becomes eligible later; the coordinator also needs an operator route, while SageMaker reconciliation must read job tags.
- **Decision:** Run immediate and 30-second durable dispatcher recovery in a cancellable nonblocking task, relying on the existing repository lease to prevent duplicate work. Add an internet-facing coordinator ALB but require a strict operator IPv4 CIDR allowlist at synth/deploy time. Grant `sagemaker:ListTags` only for tagged processing/training job ARNs. Set the bounded approval TTL to 24 hours to cover the maximum five experiments, with training and evaluation jobs each bounded at two hours.
- **Alternatives:** Startup-only recovery; open ALB ingress; process-local retry state; broad SageMaker wildcard tag permissions; retain the 15-minute approval window.
- **Reason:** Periodic scans recover leases and interrupted runs without blocking health startup; leases preserve at-most-one active owner; an explicit CIDR makes the route usable without silently opening it; scoped tag reads permit reconciliation; the approval window covers the configured worst-case provider duration.
- **Trade-offs:** The demo ALB uses HTTP and is only suitable from the narrow configured trusted network; use TLS/private ingress before broader exposure. Source-IP changes require a CDK update. The approval packet is bounded to at most 24 hours.
- **Affected components:** `backend/app/api/autonomous_live.py`, recovery API tests, CDK coordinator ALB/IAM, approval environment defaults, `infra/cdk/README.md`, and `Flow.md`.
- **Validation:** Autonomous live API tests and all CDK stack contract tests passed. No AWS deployment, resource mutation, or live run was performed.

## DEC-033 — Replay structured repair proposals against stored failures

- **Date / run:** 2026-09-12 / `VERIFIER-BACKED-CORRECTIONS-001`
- **Status:** Accepted
- **Context:** The curator could select verifier-confirmed trajectories, but a replayable failed trajectory is not a valid SFT target and the agent had no bounded way to propose a repair for it.
- **Decision:** Let the DataCuratorAgent return strict train/replay action proposals tied to supplied failure references. The authenticated objective worker must resolve the exact stored verifier-confirmed failed trajectory, verify task and split identity, and deterministically replay the proposal. Persist and expose only PASS results with `repaired_from_trajectory_id` lineage; reject without persistence when replay fails. Curation omits failed originals from SFT rows, while the lower-level dataset builder continues to reject them.
- **Alternatives:** Train on the failed source actions; trust an agent-authored success flag; persist unsuccessful repair attempts; allow validation/hidden repair proposals.
- **Reason:** Agent creativity can suggest alternatives, but only the objective worker has authority to establish successful behavior and preserve source provenance.
- **Trade-offs:** A correction-only curation plan still fails if every proposed replay is rejected and no successful original is available. Legacy coordinator-only references that are not canonical objective-worker references cannot be used as repair sources.
- **Affected components:** Data curator handoff schema/prompt, objective correction contracts and replay endpoint, objective service, live objective client, focused tests, and `Flow.md`.
- **Validation:** Curator binding tests and objective endpoint tests cover accepted/rejected replay, source lineage, failed-source exclusion, and split boundaries. Local tests are not live AgentGym/GPU evidence.

## DEC-034 — Disable process-local mutation routes in AWS mode

- **Date / run:** 2026-09-12 / `AWS-LEGACY-ROUTE-GUARD-001`
- **Status:** Accepted
- **Context:** The coordinator still exposes process-local `/api/runs` create/step/auto/cancel and demo environment reset routes. In AWS mode these could be mistaken for the durable SageMaker-backed run API and cannot survive process restart.
- **Decision:** Return `410 Gone` for legacy and demo mutation routes when `APP_MODE=aws`; direct users to the approval-gated `/api/live/runs` control plane. Keep read-only comparison routes available.
- **Alternatives:** Leave both mutation surfaces active; silently proxy legacy calls; remove the local demo entirely.
- **Reason:** One production mutation surface preserves the durable supervisor as owner of live state while retaining local demo behavior for development.
- **Trade-offs:** Existing clients that invoke legacy mutation routes must use `/api/live` in AWS deployments.
- **Affected components:** FastAPI legacy mutation handlers, integration tests, and `Flow.md`.
- **Validation:** Focused route integration tests and source checks passed. No AWS deployment or live training run was performed.

## DEC-035 — Right-size the internal objective Fargate task

- **Date / run:** 2026-09-12 / `OBJECTIVE-FARGATE-SIZING-001`
- **Status:** Accepted
- **Context:** The internal objective task loads Python, PyTorch, Transformers, and FunctionGemma, but its 0.5 vCPU / 1 GiB allocation is below the proposed hackathon baseline.
- **Decision:** Give only the internal objective Fargate task a fixed 2 vCPU / 4 GiB allocation. Keep the coordinator at 1 vCPU / 2 GiB; add no general sizing override or changes to the external objective-worker path.
- **Alternatives:** Retain the undersized task; add broad context-configurable CPU and memory values.
- **Reason:** The objective worker needs more headroom for its Python/model-loading process while keeping the infrastructure change explicit and bounded to that service.
- **Trade-offs:** The internal service requests more Fargate resources. Synthesis does not prove image startup, model load, or a live objective run.
- **Affected components:** `infra/cdk/stacks/post_training_stack.py`, its CDK contract test, and `Flow.md`.
- **Validation:** CDK synthesis assertions verify the objective task's 2 vCPU / 4 GiB allocation and unchanged coordinator sizing. No AWS deployment or runtime proof is implied.

## DEC-037 — Resolve completed Processing outputs to one immutable evaluation report

- **Date / run:** 2026-09-12 / `PROCESSING-REPORT-PINNING-001`
- **Status:** Accepted
- **Context:** SageMaker Processing reports an S3 output prefix, not a single report object. Treating that prefix as an object fails at `HeadObject`; selecting the first configured output or first matching report could also bind evaluation to the wrong artifact.
- **Decision:** Require exactly one configured output named `evaluation` on a completed Processing job. List every page beneath that prefix and require exactly one `evaluation.json` or `evaluation.tar.gz` object. Resolve its S3 `VersionId`, download that exact version, derive SHA-256 from its bytes, and retain it under a versioned content-addressed key before parsing metrics.
- **Alternatives:** `HeadObject` the prefix; select the first output; accept the newest or first matching report; use unversioned `GetObject` after discovery.
- **Reason:** The evaluator's metrics must be tied to one concrete, immutable provider output, with ambiguity and mutable paths failing closed.
- **Trade-offs:** The artifact store needs scoped S3 list, head, get-version, and put permissions; Processing jobs without one unambiguous expected report stop without evidence.
- **Affected components:** SageMaker provider result mapping, S3 artifact canonicalization, live evaluation report reader, focused tests, and `Flow.md`.
- **Validation:** Focused provider, artifact-integrity, and live-evaluation tests passed. No AWS Processing job was run.

## DEC-038 — Cancel dispatcher child work before releasing its lease

- **Date / run:** 2026-09-12 / `AWS-DISPATCHER-CANCELLATION-001`
- **Status:** Accepted
- **Context:** Cancelling a dispatcher task during application shutdown cancelled its heartbeat but could leave its separately-created supervisor task running. The dispatcher's outer `finally` then released the lease while that child could still poll or submit provider work, allowing a replacement dispatcher to overlap it.
- **Decision:** On dispatcher cancellation, cancel and await the supervisor child before propagating cancellation and releasing the lease. Persisted provider operation IDs remain the recovery source of truth after the process stops.
- **Alternatives:** Release the lease and let the child continue; wait indefinitely for the supervisor; mark the run terminal during shutdown.
- **Reason:** Lease ownership must cover all work it authorized; cancellation must not create concurrent owners for one durable run.
- **Trade-offs:** Shutdown waits for cooperative async cancellation cleanup. A hard process kill still relies on lease expiry and provider-operation reconciliation.
- **Affected components:** `backend/app/autonomous/dispatcher.py`, dispatcher cancellation tests, and `Flow.md`.
- **Validation:** A regression test verifies the supervisor child is cancelled and awaited before the durable lease is released. No AWS deployment or live provider job was performed.

## DEC-039 — Bootstrap the objective checkpoint before serving requests

- **Date / run:** 2026-09-12 / `OBJECTIVE-STARTUP-CHECKPOINT-001`
- **Status:** Accepted
- **Context:** CDK supplied the internal objective worker with an immutable S3 checkpoint reference and digest, but the shared backend image started Uvicorn directly and never materialized that bundle into the configured local checkpoint directory.
- **Decision:** Route image startup through a role-aware Python entrypoint. For `SERVICE_ROLE=objective`, fetch exactly the versioned S3 object, verify its bundle SHA-256, safely extract and validate the pinned FunctionGemma revision, derive the local snapshot digest, and only then `exec` Uvicorn. Coordinator startup skips checkpoint download. Any bootstrap error terminates startup.
- **Alternatives:** Bake large model weights into the image; allow runtime Hugging Face fallback; start the worker and fail later on its first request.
- **Reason:** Separating immutable model data from the image keeps deployment artifacts smaller and makes invalid or unavailable checkpoint provenance fail closed before the objective health endpoint is served.
- **Trade-offs:** The internal objective task requires S3 version-read and KMS access to the configured artifact prefix and incurs model download/startup latency. Local tests do not prove an AWS task can reach or decrypt the object.
- **Affected components:** Backend Docker entrypoint, objective checkpoint bootstrap utility, bootstrap tests, backend/infra documentation, and `Flow.md`.
- **Validation:** Focused bootstrap and startup-order tests, Ruff, mypy, docs sync, and diff checks; no AWS deployment or checkpoint-backed objective inference is implied.

## DEC-040 — Require TLS on the public coordinator ingress

- **Date / run:** 2026-09-12 / `AWS-HTTPS-COORDINATOR-INGRESS-001`
- **Status:** Accepted
- **Context:** The original coordinator ingress decision described an HTTP-only public ALB, which is incompatible with transmitting the one-run approval token over an untrusted network.
- **Decision:** Keep the internet-facing ALB restricted to the configured operator IPv4 CIDR and require HTTPS on port 443 with an ACM certificate, a matching public hostname, and the corresponding Route 53 public-zone contract. Fail synthesis when these inputs are absent or inconsistent.
- **Alternatives:** Keep HTTP and rely only on source CIDR filtering; open the ALB broadly; expose an unencrypted public task address.
- **Reason:** TLS protects approval tokens and control-plane traffic in transit, while the CIDR allowlist limits who can reach the hackathon API.
- **Trade-offs:** A valid certificate and domain/zone are external deployment prerequisites; this account currently has no listed ACM certificate or Route 53 hosted zone, so ingress code is implemented but not deployed.
- **Affected components:** CDK coordinator ALB, CIDR/certificate/DNS contract tests, and `infra/cdk/README.md`.
- **Validation:** Focused CDK synthesis tests passed. No stack deployment or live endpoint was performed.

## DEC-041 — Treat every unexpired dispatcher lease as exclusively owned

- **Date / run:** 2026-09-13 / `AWS-DISPATCHER-LEASE-EXCLUSIVITY-001`
- **Status:** Accepted
- **Context:** The HTTP start path and periodic recovery loop share one dispatcher owner ID. Allowing that same owner ID to claim a live lease again can launch two supervisors for one run after concurrent scans.
- **Decision:** Reject `claim_lease` whenever any owner has an unexpired lease, including the requesting owner. Keep renewal as a distinct owner-checked, unexpired-lease compare-and-swap update. Expired leases remain claimable for recovery.
- **Alternatives:** Permit same-owner claim as implicit renewal; rely only on deterministic SageMaker job names; serialize all run work in process memory.
- **Reason:** A process identity is not an exclusive claim token when concurrent dispatcher invocations run under that process. The durable lease must reject a second claim even when owner strings match.
- **Trade-offs:** A lease owner must call the renewal API rather than claim again; expiry and provider reconciliation remain necessary after forced process termination.
- **Affected components:** Autonomous in-memory/DynamoDB lease repositories, repository and dispatcher concurrency tests, and `Flow.md`.
- **Validation:** The same-owner and stale-scan concurrent-dispatch regression tests failed before the fix and passed afterward; focused repository/supervisor tests, Ruff, and mypy passed. No AWS resource mutation or deployment was performed.

## DEC-042 — Treat tokenizer vocabulary keys as data, not access-control metadata

- **Date / run:** 2026-09-13 / `FUNCTIONGEMMA-CHECKPOINT-HANDOFF-001`
- **Status:** Accepted
- **Context:** The immutable checkpoint validator recursively scans JSON for truthy access-restriction flags. FunctionGemma's tokenizer vocabulary contains arbitrary token strings as JSON keys, so a token such as `private` with an integer token ID can be mistaken for a repository access-status field.
- **Decision:** Continue parsing and validating `tokenizer.json` as required model content, but do not interpret vocabulary keys as gated/private status metadata. Continue scanning model/config and explicit metadata/report JSON for restriction flags, and keep checkpoint identity, pinned revision, required files, and weight validation unchanged.
- **Alternatives:** Disable restriction checks for all JSON files; remove the tokenizer from checkpoint validation; allow any restricted-looking key without regard to file type.
- **Reason:** Access status is established by the authenticated pinned-revision fetch and metadata files; tokenizer vocabulary words are data, not access-policy signals.
- **Trade-offs:** A nonstandard gate marker embedded only as a key in `tokenizer.json` is not treated as authoritative access metadata; inaccessible or incomplete Hub downloads still fail before staging.
- **Affected components:** FunctionGemma immutable checkpoint validator, Hub handoff metadata cleanup, and checkpoint staging tests.
- **Validation:** Focused handoff and checkpoint validator suites passed (46 tests), with Ruff, mypy, and diff checks. The live S3 checkpoint upload then succeeded with a pinned revision, content digest, VersionId, and KMS encryption. This does not prove model inference or training.

## DEC-043 — Bind evaluator channel variables to Processing local mounts

- **Date / run:** 2026-09-13 / `PROCESSING-LOCALPATH-CONTRACT-001`
- **Status:** Accepted
- **Context:** SageMaker Processing mounts each declared `ProcessingInput` at its `S3Input.LocalPath`, while the evaluator resolves artifacts through `SM_CHANNEL_*` environment variables. A mismatch can make the worker read the wrong location or fail after an expensive job starts.
- **Decision:** Use fixed absolute local paths for `base_model`, `candidate`, `champion`, and `sealed` inputs, and set each matching `SM_CHANNEL_*` variable to the same path. Reject conflicting caller-provided channel paths. Keep `SM_OUTPUT_DATA_DIR` equal to the Processing output `LocalPath`.
- **Alternatives:** Let callers choose arbitrary paths; rely on implicit environment defaults; set environment paths independently from Processing input/output configuration.
- **Reason:** One explicit mapping keeps the SageMaker payload and strict worker parser in agreement before submission.
- **Trade-offs:** Worker path changes require a coordinated provider/test/documentation update; the local contract tests do not prove a live Processing container mount.
- **Affected components:** SageMaker Processing request mapping, evaluator input parser contract tests, and `Flow.md`.
- **Validation:** Focused provider, artifact-integrity, and live-evaluation tests, Ruff, targeted mypy, docs sync, and diff checks passed. No AWS Processing job was submitted.

## DEC-044 — Build and publish worker images in AWS CodeBuild

- **Date / run:** 2026-09-13 / `AWS-CODEBUILD-WORKER-IMAGES-001`
- **Status:** Accepted
- **Context:** Trainer and evaluator images install multi-gigabyte ML/CUDA dependencies; local builds were slow and risk exhausting workstation disk, while runtime images must be immutable ECR digests.
- **Decision:** Package only the allowlisted backend Docker build inputs into a versioned S3 source object, excluding credentials, virtual environments, local model caches, and datasets. Use a narrowly scoped privileged CodeBuild project to build all three `linux/amd64` images, push them to the bootstrap ECR repositories under a unique immutable tag, and resolve each pushed digest before runtime synthesis.
- **Alternatives:** Download and build the ML images on the developer workstation; use mutable image tags; grant the build project broad account-wide ECR/S3 access.
- **Reason:** CodeBuild moves large dependency downloads and Docker layer creation to AWS, while a restricted source/object/repository policy and digest-based runtime contract keep the build reproducible and bounded.
- **Trade-offs:** CodeBuild requires privileged mode, internet access for public base images and package indexes, and incurs build/storage charges. Image publication is not proof of GPU smoke, SageMaker training, evaluation, or promotion.
- **Affected components:** `backend/aws-image-buildspec.yml`, source staging, scoped CodeBuild IAM/project configuration, ECR image publication, and runtime image digest inputs.
- **Validation:** Buildspec YAML parsing and `git diff --check` passed. AWS CodeBuild infrastructure and actual image publication are being performed separately; no SageMaker job or GPU smoke is implied by this decision.

## DEC-045 — Diagnose objective failures with metadata-only stage telemetry

- **Date / run:** 2026-09-13 / `OBJECTIVE-STAGE-TRACE-001`
- **Status:** Accepted
- **Context:** The deployed objective worker returned a generic HTTP 503 for a real benchmark request, but its logs did not identify whether checkpoint validation, local model loading, generation, tool parsing, environment execution, verification, or S3 persistence failed.
- **Decision:** Generate one server-side correlation ID per authenticated benchmark request, emit a bounded event for each allow-listed execution stage, and return only a generic 503 plus the correlation header. Events may contain the correlation ID, stage, finite latency, immutable checkpoint revision, process RSS, status, and exception class; they must not contain exception text or task/model/credential content. Expose separate process-local objective readiness attestations for configuration, checkpoint validation, model load, valid tool generation, artifact-store interface configuration, and their aggregate.
- **Alternatives:** Return raw exception details; log prompt/trajectory contents; treat static configuration readiness as proof that model execution works; remove useful stage diagnostics.
- **Reason:** Stage-local, metadata-only telemetry identifies a failing AWS phase without exposing sensitive data, while readiness and actual artifact-producing smoke remain distinct evidence.
- **Trade-offs:** The first valid FunctionGemma tool call is required before model-load/generation readiness becomes true. Artifact-store interface readiness alone does not prove an S3 write; the benchmark smoke must verify a real version-pinned artifact.
- **Affected components:** Objective benchmark service and model adapter, objective readiness response, coordinator shallow health/build-info response, runtime task configuration, and focused tests.
- **Validation:** Full backend test suite passed when run from `backend/` with the dev extra; targeted Ruff and mypy passed for the objective/live-path files. Read-only AWS checks confirmed both stacks deployed, `/health` HTTP 200, `/api/live/readiness` returning READY, and no SageMaker jobs. The previously deployed image still returned a generic objective 503; this source change has not yet been deployed and must not be represented as live inference evidence.

## DEC-046 — Explicitly activate FunctionGemma tool-calling mode

- **Date / run:** 2026-09-13 / `FUNCTIONGEMMA-TOOLCALL-ACTIVATION-001`
- **Status:** Accepted
- **Context:** A real AWS objective trace confirmed that the immutable checkpoint, processor, model, prompt rendering, and generation worked, but generated output did not pass the strict allow-list parser. The developer prompt said only to use service-recovery functions.
- **Decision:** Use FunctionGemma's documented activation instruction, `You are a model that can do function calling with the following functions`, before the existing one-call-at-a-time instruction. Keep parser strict: do not extract arbitrary text, accept prose fallbacks, or execute unrecognized calls. Require a real post-deploy objective request and versioned artifact before marking model-generation readiness.
- **Alternatives:** Relax the parser to accept unconstrained text; synthesize a tool action when parsing fails; leave the model prompt unchanged.
- **Reason:** FunctionGemma's documented format uses a specific tool-use trigger; adding it addresses the actual observed failure while retaining closed-world tool validation. Reference: https://ai.google.dev/gemma/docs/functiongemma/formatting-and-best-practices
- **Trade-offs:** This remains a prompt-level hypothesis until a deployed real generation parses, executes, verifies, and persists successfully.
- **Affected components:** Objective prompt, focused regression test, AWS backend image publication and deployment.
- **Validation:** Regression test observed failing before the prompt change and passing afterward; real AWS trace of the pre-fix image identified `FUNCTION_PARSE`; post-fix live validation is pending.

## DEC-047 — Classify known FunctionGemma parser rejections without logging output

- **Date / run:** 2026-09-13 / `FUNCTIONGEMMA-PARSE-DIAGNOSTICS-001`
- **Status:** Accepted
- **Context:** The deployed prompt activation fix did not resolve the real AWS objective 503. Stage telemetry isolated the failure to `FUNCTION_PARSE`, but the model output and parser message are intentionally excluded from logs.
- **Decision:** Map only the parser's fixed, repository-defined rejection messages to a finite allow-list of categorical failure codes in the `FUNCTION_PARSE` stage event. Never include generated text, raw exception messages, task contents, or credentials.
- **Alternatives:** Log completion text; log arbitrary exception strings; relax the parser to accept unverified output; make another prompt-only change without identifying the rejection branch.
- **Reason:** A specific parser branch will guide the next minimal fix while preserving the objective worker's content confidentiality and strict tool allow-list.
- **Trade-offs:** Unknown errors remain visible only as exception classes; the categorical-code image must be deployed before it can clarify the live parser rejection.
- **Affected components:** Objective stage telemetry, objective execution tests, `Flow.md`, and AWS backend image release.
- **Validation:** New safe-telemetry regression failed before implementation and passed afterward; all objective execution tests, focused Ruff, and targeted mypy passed. The deployed one-episode smoke still failed at `FUNCTION_PARSE` before this diagnostic change was built; no trajectory or S3 report was produced and no SageMaker job was submitted.
