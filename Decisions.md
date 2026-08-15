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
