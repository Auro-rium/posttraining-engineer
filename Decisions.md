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
