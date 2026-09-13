# Change Log

## 2026-09-13 — `OBJECTIVE-STAGE-TRACE-001`

- **Goal:** Diagnose and then resolve the first real objective-worker HTTP 503 without leaking prompts, task data, completions, or credentials.
- **Summary of changes:** Added per-request correlation IDs and safe stage-level objective telemetry for checkpoint resolve, processor/model load, prompt render, generation/decode, tool parsing, environment actions, trajectory verification, S3 persistence, and benchmark completion. The HTTP boundary returns only a generic 503 and correlation ID. Objective readiness now separates configuration, checkpoint, model load, generation, artifact-store interface, and aggregate execution attestations. Coordinator `/health` reports shallow objective endpoint configuration and injected build provenance; runtime CDK injects bounded baseline episode count and build metadata. Updated the live FunctionGemma S3 version reference in `Flow.md`.
- **Affected files and components:** Objective worker service/execution/readiness, coordinator health and build info, runtime CDK configuration, focused tests, `Flow.md`, `Decisions.md`, and this log.
- **Tests and verification:** Full backend suite passed from `backend/` using the configured pytest asyncio mode and dev extra; focused Ruff and mypy passed; read-only AWS checks confirmed bootstrap/runtime stacks are deployed, API `/health` is HTTP 200, and `/api/live/readiness` reports READY.
- **Known limitations:** The deployed API still reflects the prior image: `/health` reports `objective_worker: not_configured` and no build-info fields. Its previous one-episode objective call returned HTTP 503 and produced no verified trajectory/report. No SageMaker training or Processing job exists. The telemetry/readiness source changes remain uncommitted and undeployed; a successful objective smoke is still the first live gate.

## 2026-09-13 — `AWS-CODEBUILD-OBJECTIVE-CONTRACT-001`

- **Goal:** Move multi-gigabyte worker-image dependency downloads to AWS and unblock the real objective-worker request contract.
- **Summary of changes:** Added an AWS CodeBuild buildspec for `linux/amd64` backend/trainer/evaluator images with immutable ECR digest reporting. Created a dedicated privileged CodeBuild project and narrowly scoped role for one versioned S3 source object, the three ECR repositories, the bootstrap KMS key, and one CloudWatch log group. Stopped the local trainer image build; no image was published by that cancelled build. Aligned objective benchmark request/response fields, configured train/replay episode counts, and verified artifact/evidence provenance. Corrected the explicit EvidenceLabel re-export and local inference adapter typing.
- **Affected files and components:** `backend/aws-image-buildspec.yml`, objective API/client/storage contracts and tests, objective model export/type annotations, `Flow.md`, `Decisions.md`, `ChangeLog.md`, and AWS CodeBuild project/role.
- **Tests and verification:** Focused objective/live-execution pytest passed 84 tests; focused Ruff and mypy passed; buildspec YAML parsing, docs synchronization, `git diff --check`, CodeBuild project configuration, trust policy, and scoped role policy were verified.
- **Known limitations and follow-up:** No source archive has been uploaded and no CodeBuild job started yet; no trainer/evaluator digest exists. API Gateway HTTP API still imposes a 30-second integration ceiling, while the benchmark is synchronous and a 10-episode inference duration has not been measured. No runtime stack, objective inference smoke, SageMaker GPU smoke/job, or live autonomous run has been executed.

## 2026-09-13 — `CONTAINER-RETRY-2026-09-13-001`

- **Goal:** Make trainer and evaluator dependency installation resilient to slow or interrupted package downloads.
- **Summary of changes:** Added a persistent BuildKit pip-cache mount and a 300-second pip timeout with five retries to both worker Dockerfiles. No application source was changed.
- **Affected files and components:** `backend/workers/trainer/Dockerfile` and `backend/workers/evaluator/Dockerfile`.
- **Tests and verification:** `backend/.venv/bin/python -m pytest -q backend/tests/test_image_smoke_contract.py` passed 3 tests; `backend/.venv/bin/python backend/scripts/check_docs_sync.py` and `git diff --check` passed. A backend image was published and its loopback `/health` endpoint returned `healthy`, but that image predates subsequent shared backend edits and is not the final deployment digest. The trainer build was cancelled before push; the evaluator image was not built.
- **Known limitations and follow-up:** Rebuild and publish the trainer/evaluator images with immutable digests, then rebuild the backend from the final shared source tree. No SageMaker job, model inference, or training run was started.

## 2026-09-08 — `MAINT-2026-09-08-001`

- **Goal:** Align local configuration and deployment documentation with the guarded Nemotron/AWS live path.
- **Summary of changes:** Set Docker Compose's default reasoning model to `nvidia.nemotron-super-3-120b`; documented all live controller variables, checkpoint and GPU admission requirements, Bedrock bearer-token precedence, the `/api/live/readiness` surface, and the existing-but-incomplete CDK foundation.
- **Affected files and components:** `.env.example`, `docker-compose.yml`, `README.md`, `backend/README.md`, and live configuration documentation.
- **Tests and verification:** `docker compose config --quiet`, `backend/scripts/check_docs_sync.py`, and `git diff --check`.
- **Known limitations and follow-up:** No secrets or AWS resources were created; live execution remains blocked until the documented external artifacts, worker, images, resources, and Bedrock authorization are configured.

## 2026-08-15 — `CHANGE-2026-08-15-003`

- **Goal:** Improve the FunctionGemma fine-tuning tools section while preserving AgentGym WebShop compatibility.
- **Summary of changes:** Added canonical strict JSON schemas for `search(keywords)` and `click(item)`, injected the schemas into training rows when omitted, and added contract coverage. The environment remains intentionally limited to its two native actions; synthetic checkout tools were not added.
- **Affected files and components:** `backend/app/webshop_tools.py`, `backend/training/train.py`, `backend/tests/test_webshop.py`, `Decisions.md`, and `Flow.md`.
- **Tests and verification:** `uv run pytest tests -q` passed 64 tests; Ruff and mypy passed for the backend app/tests/training code.
- **Known limitations and follow-up:** More native tools require a different or extended environment adapter and a matching benchmark; this change only strengthens the schema and SFT signal for WebShop's supported actions.

## 2026-08-15 — `REVERT-2026-08-15-001`

- **Goal:** Revert the preceding optional fine-tuning tools-schema expansion at the user's request.
- **Summary of changes:** Removed the added schema module, training fallback, test, and current-flow references; restored the prior renderer and WebShop contract.
- **Affected files and components:** `backend/app/webshop_tools.py`, `backend/training/train.py`, `backend/tests/test_webshop.py`, and `Flow.md`.
- **Tests and verification:** Full backend tests, Ruff, mypy, documentation sync, and `git diff --check` run after the revert.
- **Known limitations and follow-up:** The earlier change remains recorded as historical append-only documentation; no expanded tool schema is active.

## 2026-08-15 — `CHANGE-2026-08-15-004`

- **Goal:** Expand the supported AgentGym environment selection to Wordle, BabyAI, and Movie.
- **Summary of changes:** Added a typed environment allow-list and metadata registry, accepted the three environments in `POST /api/runs`, updated agent instructions and backend documentation, and added API coverage. The external objective worker remains responsible for environment-specific actions, replay, and evaluation; local explanation mode remains WebShop-only.
- **Affected files and components:** `backend/app/environments.py`, `backend/app/main.py`, `backend/app/demo.py`, `backend/app/adk_agents.py`, `backend/tests/test_api.py`, `README.md`, `Flow.md`, and `Decisions.md`.
- **Tests and verification:** `uv run pytest tests -q` passed 64 tests; Ruff and mypy passed for app/tests/training; documentation sync is pending final run.
- **Known limitations and follow-up:** No live Wordle, BabyAI, or Movie objective-worker execution has been performed. Their environment-specific schemas and benchmark workers must be deployed before claiming live results.

## 2026-08-21 — `CHANGE-2026-08-21-001`

- **Goal:** Make every specialist a detailed, least-privilege, long-horizon agent with an explicit sharing protocol.
- **Summary of changes:** Added per-role tool allow-lists and typed handoff targets; compiled common system instructions covering RunState/GCS/A2A sharing, idempotency, resumability, bounded retries, terminal-state handling, and telemetry redaction; added prompt contract tests.
- **Affected files and components:** `backend/app/adk_agents.py`, `backend/tests/test_adk_agents.py`, `Flow.md`, and `Decisions.md`.
- **Tests and verification:** Prompt tests, full backend tests, Ruff, mypy, documentation sync, and `git diff --check` passed.
- **Known limitations and follow-up:** The allow-lists describe service capabilities; actual live tool execution still depends on deployed ADK/A2A/objective-worker services.

## 2026-08-21 — `CHANGE-2026-08-21-002`

- **Goal:** Include AgentGym AgentEval as the explicit sealed evaluation benchmark.
- **Summary of changes:** Added AgentEval suite/version and manifest SHA-256 fields to evaluation contracts, passed suite identity through objective-worker requests, and made verifiable live reports fail closed without the manifest hash or a matching suite identity.
- **Affected files and components:** `backend/app/models.py`, `backend/app/cloud_provider.py`, evaluation tests, `README.md`, `Flow.md`, and `Decisions.md`.
- **Tests and verification:** `uv run pytest tests -q` passed the full backend suite; Ruff, mypy, documentation sync, and `git diff --check` passed.
- **Known limitations and follow-up:** The external objective worker must actually pin the AgentEval manifest and return its SHA-256; no live AgentEval run has been performed yet.

## 2026-08-21 — `CHANGE-2026-08-21-003`

- **Goal:** Make active runs recoverable after coordinator or Cloud Run restart.
- **Summary of changes:** Persist in-flight Vertex job status and resource name before polling, reconcile prior jobs by deterministic display name, resume non-terminal Firestore runs on startup, and increase the bounded auto-step ceiling to account for recovery polling steps.
- **Affected files and components:** `backend/app/orchestrator.py`, `backend/app/main.py`, `backend/app/cloud.py`, `backend/app/cloud_provider.py`, recovery tests, `README.md`, `Flow.md`, and `Decisions.md`.
- **Tests and verification:** Full backend tests, Ruff, mypy, documentation sync, and `git diff --check` passed.
- **Known limitations and follow-up:** Recovery requires Vertex list/get permissions and durable Firestore access; provider outages or a permanently unavailable objective worker remain genuine blockers and fail closed.

This file is append-only. Entries describe repository changes and the verification actually performed; they must not contain secrets, private prompts, or held-out evaluation contents.

## 2026-08-15 11:07:25 IST (+05:30) — `BUILD-2026-08-15-001`

- **Goal:** Implement the backend-only Autonomous Post-Training Engineer hackathon demonstration with living documentation and deployable Google Cloud boundaries.
- **Summary of changes:** Added strict run/trajectory/dataset/experiment/evaluation contracts; leakage-safe RAG; typed A2A envelopes; local and Google Cloud repositories/artifact adapters; metadata-only telemetry; AgentGym WebShop and Vertex/Secret Manager adapters; eight specialist roles; the bounded deterministic orchestrator; the coordinator FastAPI/SSE API; three `SERVICE_ROLE` modes; local explanation fixtures; deployment/CI configuration; and repository governance documentation.
- **Affected files and components:** `backend/app`, `backend/tests`, backend packaging and container files, `docker-compose.yml`, `infra/terraform`, `.github/workflows/ci.yml`, environment/git configuration, `AGENTS.md`, `README.md`, `Decisions.md`, `Flow.md`, and the documentation checker.
- **Tests and verification:** `uv run pytest tests -q` passed 36 tests; `uv run ruff check app tests scripts` passed; `uv run mypy app` passed for 16 source files; the Cloud Run container image built successfully; `docker compose config --quiet` passed; all Terraform files parsed as HCL using `python-hcl2`; the documentation checker is run after this entry is present.
- **Known limitations and follow-up:** No real Google Cloud deployment, Gemini/ADK research call, A2A network handoff, AgentGym benchmark, Vertex QLoRA job, or checkpoint evaluation was executed in this run. Terraform provider validation was not run locally because the Terraform CLI is unavailable. The local provider emits only `EXPLANATION` evidence and cannot pass provenance-based promotion; a genuine cloud decision-provider composition and live smoke run remain required before claiming an autonomous model improvement.

## 2026-08-15 11:33:52 IST (+05:30) — `BUILD-2026-08-15-002`

- **Goal:** Replace the explanatory composition with the deployment-only cloud path and make every specialist prompt and evidence boundary explicit.
- **Summary of changes:** Added the authenticated A2A v1 cloud decision provider; official typed A2A service executor; explicit role/input/output/procedure/forbidden-action/failure prompts for all eight specialists; structured Gemini ADK research; integrity-checked GCS trajectory loading; SHA-pinnable leakage-safe RAG with canonical citations; objective-worker delegation for FunctionGemma, AgentGym replay, evaluation, and training evidence; persisted benchmark provenance; metadata-only A2A/RAG/Vertex/evaluation spans; runtime Secret Manager access in the QLoRA worker; cloud-safe timeouts; and dependency-ordered Cloud Run resources. The cloud coordinator now fails closed rather than falling back to explanatory evidence.
- **Affected files and components:** `backend/app`, `backend/tests`, `backend/training`, backend packaging and lockfile, `infra/terraform`, `docker-compose.yml`, `.env.example`, `README.md`, `Decisions.md`, `Flow.md`, and `ChangeLog.md`.
- **Tests and verification:** `uv run pytest` passed 67 tests; `uv run ruff check app tests training scripts` passed; `uv run mypy app tests` passed for 29 source files; `backend/scripts/check_docs_sync.py` passed; `git diff --check` passed; the backend Docker image built; `docker compose config --quiet` passed; and all Terraform files parsed as HCL with `python-hcl2`.
- **Known limitations and follow-up:** No Google Cloud deployment, live Gemini call, network A2A handoff, AgentGym execution, Vertex job, or checkpoint improvement was run or claimed. The required authenticated objective worker and GCS RAG corpus are external deployment prerequisites not provisioned here. Terraform provider validation was not run because the Terraform CLI is unavailable. A Vertex job handle is not persisted before polling, so `/auto` is best-effort and a coordinator restart during training requires operator inspection before retry.

## 2026-08-15 11:40:00 IST (+05:30) — `MAINT-2026-08-15-001`

- **Goal:** Keep generated live-document snapshots out of version control without excluding the canonical living documents.
- **Summary of changes:** Added `live-docs/` and `live_docs/` ignore rules and documented the repository-hygiene decision.
- **Affected files and components:** `.gitignore`, `Decisions.md`, and `ChangeLog.md`.
- **Tests and verification:** Documentation and Git status checks are run after this entry; no application behavior changed.
- **Known limitations and follow-up:** Generated snapshots must be deliberately copied into the root living documents when they contain information that should be retained.

## 2026-08-15 11:32:34 IST (+05:30) — `BUILD-2026-08-15-002`

- **Goal:** Complete the deployment-only backend path with fail-closed cloud composition, real A2A/RAG boundaries, and honest operational constraints.
- **Summary of changes:** Added the cloud-only decision provider; Cloud Run identity-token A2A transport; official typed A2A service dispatch; structured Gemini ADK research; integrity-checked GCS trajectory/RAG loading; canonical cited hypotheses; remote objective benchmark, curation replay, evaluation, and training-evidence contracts; Vertex submission/polling; restart-safe benchmark provenance; and explicit eight-agent prompts. Refactored Terraform to create team services before the coordinator in one apply, use Vertex workload identity, require the external evidence-worker and RAG-corpus inputs, inject only the Hugging Face secret ID, keep coordinator CPU allocated, and bound Cloud Run/A2A work to 55 minutes. Reframed Docker Compose as a contract/packaging smoke harness rather than a deployment path.
- **Affected files and components:** Cloud provider and typed service-operation modules, A2A bootstrap, settings, orchestration/domain provenance, backend tests, Docker/Compose, Terraform variables/resources/outputs, `.env.example`, `README.md`, `Decisions.md`, and `Flow.md`.
- **Tests and verification:** `uv run pytest tests -q` passed exactly 67 tests; `uv run ruff check app tests scripts` passed; `uv run mypy app` passed for 18 source files; the final backend image built and its coordinator `/health` endpoint returned HTTP 200; `docker compose config --quiet`, the living-document checker, and `git diff --check` passed; all Terraform files parsed successfully as HCL.
- **Known limitations and follow-up:** No live Google Cloud, Gemini, A2A, objective-worker, AgentGym, Vertex training, or held-out evaluation run was performed, so no improvement metric is claimed. Terraform does not provision the required sandboxed objective-evidence worker, its FunctionGemma/AgentGym runtime, the GCS RAG corpus, or the separate QLoRA training image. Terraform provider validation remains delegated to CI because the CLI is unavailable locally. `/auto` is process-local and best-effort, and the Vertex job handle is not persisted before polling; a coordinator restart during training requires operator inspection before retry.

## 2026-09-06 — `AWS-STRANDS-HACKATHON-001`

- **Goal:** Adapt the backend demonstration for the AWS Agents for Humans Professional Agents track without building a frontend.
- **Summary of changes:** Added a Strands Agents specialist package with eight role-specific agents, a service-recovery environment, OptimizationRun state, workflow orchestrator, AWS-oriented API endpoints, requirements metadata, and architecture documentation. `/step` now advances one phase; `/auto` executes the bounded remaining workflow. Deterministic promotion logic remains separate from model-generating agents.
- **Verification:** Python compilation, the Strands implementation smoke test, full orchestrator workflow, and FastAPI health/create/step/status smoke checks passed in `backend/venv`. `uv lock --offline` could not refresh the existing lock because the offline cache lacks a compatible OpenTelemetry resolution; no live AWS, Bedrock, AgentCore, SageMaker, or Gemma training run was performed.
- **Known limitations:** The current service-recovery and training integrations are demonstration adapters; simulated metrics and artifact references must not be presented as live model improvement. The legacy Google-specific modules and tests remain in the repository and are not yet migrated to Strands.

## 2026-09-06 — `AWS-STRANDS-HACKATHON-002`

- **Goal:** Make this checkout an AWS-only submission rather than a mixed Google/AWS repository.
- **Summary of changes:** Removed legacy Google ADK/Vertex/Firestore/GCS source modules, Google-only tests and training worker, and the Google Terraform stack. Replaced Google environment/configuration references with AWS/Strands settings, updated the operational flow and CI checks, and kept historical reasoning only in the append-only records.
- **Affected files and components:** `backend/app`, `backend/tests`, `backend/training`, `infra/terraform`, `.env.example`, `docker-compose.yml`, `.github/workflows/ci.yml`, `backend/pyproject.toml`, `backend/scripts/check_docs_sync.py`, `README.md`, and `Flow.md`.
- **Verification:** Python compilation and the local Strands smoke workflow remain the validation targets; no live AWS, Bedrock, SageMaker, DynamoDB, S3, or AgentCore run was performed.
- **Known limitations:** The AWS adapters still contain simulated outputs and the AWS-native persistence/deployment stack is not yet implemented. Historical Google mentions remain in append-only records by design.

## 2026-09-06 — `DOCS-SUBMISSION-HARDENING-001`

- **Goal:** Make the AWS submission boundary and continuous workflow explicit without overstating local demonstration behavior.
- **Summary of changes:** Added the MIT license, a `LOCAL_DEMO` versus `LIVE_AWS` capability/evidence matrix, and API/event-flow documentation covering phase transitions, artifact references, durable event requirements, and restart boundaries. Corrected the documented local API list to match the current implementation.
- **Affected files and components:** `LICENSE`, `README.md`, `Flow.md`, and `ChangeLog.md`.
- **Tests and verification:** `python3 backend/scripts/check_docs_sync.py` passed; `git diff --check` passed. No Python source was modified and no live AWS run was performed.
- **Known limitations and follow-up:** The current local demo remains process-local, emits explanatory/simulated outputs, and has no durable event stream; these remain `LIVE_AWS` integration requirements.

## 2026-09-06 — `POSTTRAINING-DOMAIN-001`

- **Goal:** Add deterministic domain primitives for continuous post-training cycles.
- **Summary of changes:** Added typed content-addressed artifact and provenance-bearing evidence models; a fail-closed improvement/regression promotion gate; an approval/rejection/rollback cycle state machine with append-only transition history; and fixed-seed callback-based benchmark/evaluation utilities. Added focused tests and included them in backend pytest discovery.
- **Affected files and components:** `backend/app/posttraining/`, `backend/test_posttraining.py`, and `backend/pyproject.toml`.
- **Tests and verification:** Focused post-training pytest passed 4 tests; Ruff and mypy passed for the new package; `git diff --check` passed. The full existing test collection remains blocked by a pre-existing missing `app.agents.training_executor_agent` module in the dirty AWS migration checkout; no live AWS or model evaluation was performed.
- **Known limitations and follow-up:** The package is intentionally not wired into `main.py` or existing agents. A live predictor, artifact store, and durable cycle worker must be connected before claiming live continuous training or promotion.

## 2026-09-06 — `POSTTRAINING-API-001`

- **Goal:** Add an independently injectable HTTP contract for continuous post-training traces and cycle decisions.
- **Summary of changes:** Added the isolated `app.api.continuous_post_training` FastAPI router for trace/cycle creation, cycle status, ordered events, artifact metadata, approval, rejection, and cancellation. Added repository/service protocols, an app-state-scoped lock-protected in-memory repository, an integration-hook document, and focused API tests. `backend/app/main.py` was not modified.
- **Affected files and components:** `backend/app/api/`, `backend/docs/continuous_post_training_api.md`, `backend/tests/test_continuous_post_training_api.py`, `Flow.md`, and `Decisions.md`.
- **Tests and verification:** Focused API tests passed 4 tests; Ruff and mypy passed for the new API and tests; Python compilation, documentation sync, and `git diff --check` passed.
- **Known limitations and follow-up:** The router is not included by the current application bootstrap, and its default repository is process-local. A host application must inject durable persistence, authentication, and a worker before claiming live continuous training or durable artifacts.

## 2026-09-06 — `AWS-DEPLOYMENT-FOUNDATION-001`

- **Goal:** Implement the first AWS deployment foundation without adding a frontend.
- **Summary of changes:** Added validated local/AWS runtime configuration with fail-closed required AWS settings; wired the continuous post-training router into the FastAPI application; added role-aware Agent Card discovery; added an AWS CDK foundation for S3 versioned artifacts, DynamoDB state, ECR, ECS/Fargate, IAM, VPC, and CloudWatch logs; and added CDK dependency metadata.
- **Verification:** `uv run pytest -q`, Python compilation, CDK synthesis with the cloud extra, documentation sync, and `git diff --check` passed.
- **Known limitations:** The current coordinator workflow still uses local demonstration orchestration and must be connected to the durable AWS repository, S3 artifact store, Bedrock model wrapper, and SageMaker provider before enabling `APP_MODE=aws`. The CDK stack is a foundation and does not claim a live training/evaluation run.

## 2026-09-06 — `AWS-LIVE-SMOKE-001`

- **Goal:** Verify the configured AWS account safely without creating persistent infrastructure or compute instances.
- **Summary of changes:** Added `backend/scripts/live_smoke.py`, a non-destructive smoke test for STS identity, one Strands/Bedrock invocation, and an S3 put/get/delete artifact round trip. Added usage documentation.
- **Live verification:** Account `145023103669`, region `us-east-1`; Bedrock `amazon.nova-pro-v1:0` returned `READY`; the temporary S3 artifact round-tripped with SHA-256 verification and was deleted. No AWS resources were created by the smoke test.
- **Known limitations:** This validates credentials and basic AWS connectivity only. It does not claim live Gemma inference, SageMaker training, DynamoDB persistence, AgentCore hosting, or model improvement.

## 2026-09-06 — `AWS-LIVE-AGENTS-001`

- **Goal:** Verify that all eight specialist roles can execute as live Strands agents through Bedrock.
- **Summary of changes:** Added `backend/scripts/live_agentic_test.py`, which invokes Benchmark, Failure Analyst, Research, Data Curator, Training Designer, Training Executor, Evaluation, and Champion Manager agents with bounded role prompts.
- **Live verification:** All eight agents returned non-empty Bedrock responses using `amazon.nova-pro-v1:0` in `us-east-1`. No AWS resources were created.
- **Known limitations:** This proves live agent invocation and role-prompt responsiveness, not real Gemma inference, verified curation, SageMaker training, held-out evaluation, or checkpoint promotion.

## 2026-09-06 — `TARGET-MODEL-GEMMA3-001`

- **Goal:** Align the intended post-training target with a Gemma model available in the configured Bedrock region.
- **Summary of changes:** Changed the default target from unavailable FunctionGemma 270M metadata to `google.gemma-3-4b-it`. The reasoning model remains `nvidia.nemotron-super-3-120b`.
- **Known limitations:** This target change was superseded immediately by the required FunctionGemma-only scope; changing an identifier does not itself perform training. A real post-training run still requires a trainable FunctionGemma checkpoint, verified SFT data, SageMaker role/container, GPU budget, and independent evaluation.

## 2026-09-06 — `TARGET-MODEL-FUNCTIONGEMMA-002`

- **Goal:** Preserve the requested FunctionGemma-only target scope.
- **Summary of changes:** Reverted the default target to `google/functiongemma-270m-it`. `nvidia.nemotron-super-3-120b` remains the Strands reasoning model only.
- **Known limitations:** FunctionGemma is not listed as an active Bedrock foundation model in this account. Real post-training therefore requires the user-supplied FunctionGemma checkpoint or an approved external model artifact; no substitute Gemma target is used.

## 2026-09-08 — `REAL-POSTTRAINING-RUNS-001`

- **Goal:** Add a bounded, evidence-first five-run comparison path with telemetry and guarded AWS execution contracts.
- **Summary of changes:** Added immutable run-history records and a DynamoDB transaction that enforces unique sequential run numbers and the five-run cap; deterministic promotion evidence checks; JSON/SVG aggregate and per-environment comparison output at `/api/runs/compare` and `/api/runs/graph`; objective benchmark provenance and bounded SageMaker train-then-evaluate polling; removal of random promotion and placeholder artifact fallbacks; and metadata-only telemetry with correlation IDs, latency/cost fields, recursive redaction, immutable attributes, and optional OpenTelemetry.
- **Verification:** Full backend pytest suite passed; focused Ruff and mypy checks passed for new/changed contracts; documentation and diff checks remain required before integration.
- **Known limitations:** No live AWS training, held-out evaluation, checkpoint improvement, or promotion was executed. AWS mode constructs adapters for pre-existing configured resources but does not provision them; the objective worker, training/evaluation images, checkpoint, and manifest must be supplied before a `LIVE` result can be recorded. Local history remains process-local and terminal comparison rows require verified artifacts.

## 2026-09-08 — `NEMOTRON-PROMPT-OBSERVER-001`

- **Goal:** Make the working agents use one auditable NVIDIA Nemotron reasoning model and document the hackathon execution observer without overstating live AWS evidence.
- **Summary of changes:** Documented the fixed `nvidia.nemotron-super-3-120b` reasoning model versus the separate FunctionGemma post-training target; documented versioned prompt contracts, bounded creativity, evidence/held-out-data rules, prompt hashes, guarded live scripts, and the animated metadata-only execution view. Expanded backend setup documentation and corrected pytest discovery to include the complete `backend/tests/` contract suite.
- **Affected files and components:** `README.md`, `backend/README.md`, `Flow.md`, `Decisions.md`, `ChangeLog.md`, and `backend/pyproject.toml`.
- **Verification:** Living-document validation, backend pytest discovery, Ruff, mypy, and diff checks must pass before integration. Documentation does not constitute evidence of live SageMaker training, held-out evaluation, checkpoint improvement, or promotion.
- **Known limitations and follow-up:** Real live claims still require a successful read-only preflight, pinned checkpoint, objective-worker artifacts, SageMaker job IDs, independent evaluation evidence, and retained telemetry. The browser observer remains read-only and cannot authorize or manufacture execution.

## 2026-09-12 — `TELEMETRY-LIFECYCLE-HARDENING-001`

- **Goal:** Complete the durable run/phase telemetry lifecycle and close unsafe event/metadata contract gaps.
- **Summary of changes:** Restricted durable event types to the explicit lifecycle vocabulary; tightened observer identifiers and allow-listed string metadata; made the supervisor persist run start, phase start/completion/failure, and terminal events; propagated durable telemetry validation failures as fail-closed supervisor stops; and added durable event ID plus exact autonomous event type to OpenTelemetry spans.
- **Affected files and components:** Autonomous models, telemetry bridge, supervisor, observer recorder, telemetry/supervisor/repository tests, `Flow.md`, and `Decisions.md`.
- **Verification:** Focused pytest suite passed; focused Ruff and mypy checks passed; living-document validation and `git diff --check` passed.
- **Known limitations:** No live AWS training, held-out evaluation, checkpoint improvement, or promotion was performed. Existing persisted event types outside the new allow-list require explicit migration before they can be decoded.

## 2026-09-12 — `AWS-HACKATHON-EVIDENCE-DOCS-001`

- **Goal:** Clarify local coordinator behavior, the checkpoint-backed objective worker, the guarded AWS live API, and the current live-run evidence boundary.
- **Summary of changes:** Distinguished process-local `/api/runs` `EXPLANATION` output from the isolated FunctionGemma objective adapter and `/api/live` control plane; documented checkpoint identity and objective-worker credential requirements, approval/idempotency gates, the AWS hackathon-only scope, the historical 2026-09-06 Bedrock/S3 smoke boundary, and the operator-reported pending GPU quota request. Added objective-role credential placeholders and a dated live deployment checkpoint to the autonomous backend plan; did not claim a live run completed.
- **Affected files and components:** `.env.example`, `README.md`, `backend/README.md`, `Flow.md`, `Decisions.md`, and `docs/superpowers/plans/2026-09-08-autonomous-live-backend.md`.
- **Verification:** `python3 backend/scripts/check_docs_sync.py` passed; objective execution/artifact contract tests passed (26 tests); `git diff --check` passed. These are local checks, not AWS deployment or live model evidence.
- **Known limitations:** No AWS calls, deployment, checkpoint-backed benchmark, SageMaker job, held-out evaluation, or promotion was performed for this documentation update. The operator-reported SageMaker quota request `9a3453884e2c4230a6e8bb0004c8cca57FuK8VC5` was `PENDING` as of 2026-09-12; recheck before authorizing compute.

## 2026-09-12 — `DOCKER-DEPLOYMENT-PACKAGING-001`

- **Goal:** Reduce backend image build context and align Docker/CDK deployment instructions with digest-pinned runtime images.
- **Summary of changes:** Added a shared backend `.dockerignore` for local virtualenvs, caches, tests, local environment files, and unused docs/training data; replaced the stale `:latest` CDK example with amd64 image-build commands and a zero-task ECR bootstrap followed by digest-pinned deployment.
- **Affected files and components:** `backend/.dockerignore` and `infra/cdk/README.md`.
- **Verification:** All three Dockerfiles passed BuildKit `--check`; `docker compose config --quiet` passed; the backend image built locally and the objective-role app import passed with local-only test settings; CDK stack tests passed (22); documentation sync and `git diff --check` passed.
- **Known limitations:** The trainer image build was cancelled during its large CUDA/cuDNN dependency download to unblock integration; the evaluator image was not fully built. No image was pushed and no AWS deployment or SageMaker job was attempted.

## 2026-09-12 — `SEALED-PAIRED-EVALUATION-001`

- **Goal:** Remove unsupported baseline benchmark requests and preserve identical held-out provenance for promotion comparisons.
- **Summary of changes:** The supervisor no longer requests a separate `baseline` objective benchmark. Candidate and active-champion scores must come from one sealed SageMaker evaluator report; the reader validates both checkpoint digests, shared manifest, paired task outcomes, and per-environment metrics. The objective benchmark request type accepts only `train`/`replay`, and other splits fail before worker invocation.
- **Affected files and components:** Supervisor, live execution adapters and report reader, objective benchmark request contract, focused tests, `Flow.md`, and `Decisions.md`.
- **Verification:** Focused supervisor/live-execution/objective workflow/service/execution tests passed (104 tests). No AWS deployment or live held-out evaluation was performed by this change.
- **Known limitations:** Paired evidence can only be produced by the configured SageMaker sealed evaluator. The legacy synchronous live controller's separate baseline/held-out benchmark calls now fail closed; it must not be used as a held-out path.

## 2026-09-12 — `AWS-CDK-VERSIONED-PREFLIGHT-001`

- **Goal:** Allow the coordinator's read-only preflight to validate immutable S3 object versions without widening access beyond the configured artifact prefix.
- **Summary of changes:** Added a dedicated `s3:GetObjectVersion` permission for objects under the artifact prefix and a direct CDK assertion for its action and resource scope.
- **Affected files and components:** `infra/cdk/stacks/post_training_stack.py` and `infra/cdk/tests/test_post_training_stack.py`.
- **Verification:** CDK stack tests passed (22 tests); documentation sync and `git diff --check` were run after this entry was added.
- **Known limitations:** No AWS deployment or resource mutation was performed.

## 2026-09-12 — `WORKER-IMAGE-SMOKE-001`

- **Goal:** Add offline model-load smoke entrypoints for the amd64 SageMaker trainer and evaluator images.
- **Summary of changes:** Added a trainer image smoke that requires CUDA/bitsandbytes, loads the staged FunctionGemma directory locally in 4-bit, performs one LoRA optimizer step, and writes a throwaway adapter; added an evaluator image smoke that loads the same local base plus adapter and runs one forward pass without sealed tasks. Copied both entrypoints into their images and documented network-disabled smoke commands using the existing `linux/amd64` Buildx instructions.
- **Affected files and components:** `backend/workers/trainer/image_smoke.py`, `backend/workers/evaluator/image_smoke.py`, both worker Dockerfiles, image smoke contract tests, `infra/cdk/README.md`, `Flow.md`, and `Decisions.md`.
- **Verification:** The three image smoke contract tests passed; Ruff, mypy, Python compilation, documentation sync, and `git diff --check` passed; trainer and evaluator Dockerfiles passed BuildKit `--check` for `linux/amd64`.
- **Known limitations:** The CUDA image was not built or run, no adapter smoke was executed against a staged checkpoint, and no image was pushed or AWS job started. In the shared in-progress checkout, the broader `test_worker_contracts.py` run failed because its existing dataset fixture still uses the deprecated `verified_replay`/`verifier_confirmed` contract after the curation-admission change.

## 2026-09-12 — `AWS-COORDINATOR-RECOVERY-INGRESS-001`

- **Goal:** Close coordinator restart/reachability gaps and authorize deterministic SageMaker job-tag reconciliation.
- **Summary of changes:** Replaced blocking startup recovery with an immediate, nonblocking 30-second durable dispatcher loop; added scoped `sagemaker:ListTags`; added an internet-facing coordinator ALB restricted to required operator IPv4 CIDRs and emitted `CoordinatorUrl`; raised the approval TTL default to 24 hours to cover the bounded five-experiment worst case.
- **Affected files and components:** Autonomous live API and tests, CDK stack and contract tests, `.env.example`, `infra/cdk/cdk.json`, `infra/cdk/README.md`, `Flow.md`, and `Decisions.md`.
- **Verification:** `tests/test_autonomous_live_api.py` passed (18); `infra/cdk/tests/test_post_training_stack.py` passed (27). No AWS deployment, resource mutation, or live run was performed.
- **Known limitations:** Coordinator ALB currently uses HTTP and must remain restricted to a trusted operator network; do not transmit approval tokens over untrusted networks. Live SageMaker readiness, GPU quota, image availability, and external credentials remain separate prerequisites.

## 2026-09-12 — `VERIFIER-BACKED-CORRECTIONS-001`

- **Goal:** Let the curator propose repairs for failed trajectories without teaching the failed source actions.
- **Summary of changes:** Added a strict train/replay-only correction proposal contract to the DataCuratorAgent handoff; added authenticated `/v1/replay-corrections` that binds proposals to stored verifier-confirmed failures, replays deterministically, persists only passing repairs with lineage, and discards rejected proposals; connected the live objective client to replay passing proposals before curation; filtered failed originals from SFT rows while leaving the lower-level dataset builder fail-closed.
- **Affected files and components:** Data-curator contracts/prompt, objective models/service/engine, live objective client, focused agent/objective tests, `Flow.md`, `Decisions.md`, and `backend/README.md`.
- **Verification:** Focused autonomous-agent, objective-engine, objective-execution, prompt-contract, objective-client, artifact-store, and objective-role integration tests passed; Ruff, mypy, documentation sync, and `git diff --check` passed.
- **Known limitations:** No GPU inference, AWS deployment, or live AgentGym correction replay was performed. Correction-only curation still fails closed if no proposal passes and no successful original is available. A broader `test_main_integration.py` run in the shared dirty checkout also failed its unrelated AWS-mode coordinator-route expectations.

## 2026-09-12 — `OBJECTIVE-FARGATE-SIZING-001`

- **Goal:** Give the internal FunctionGemma objective worker sufficient bounded Fargate task resources for its Python/model-loading runtime.
- **Summary of changes:** Raised only the internal objective task definition from 0.5 vCPU / 1 GiB to a fixed 2 vCPU / 4 GiB; kept the coordinator at 1 vCPU / 2 GiB and added a synthesis contract for both allocations.
- **Affected files and components:** `infra/cdk/stacks/post_training_stack.py`, `infra/cdk/tests/test_post_training_stack.py`, `Flow.md`, and `Decisions.md`.
- **Verification:** Focused CDK stack tests passed (34). Ruff reports two unrelated existing findings in these already-modified files (`TRY004` in coordinator CIDR validation and `RUF100` for the existing `E402` suppression); with those two findings excluded, Ruff passed for both Python files. Documentation sync and `git diff --check` passed.
- **Known limitations:** No AWS deployment, Fargate startup, FunctionGemma model-load, GPU execution, or live post-training run was performed.

## 2026-09-12 — `PROCESSING-REPORT-PINNING-001`

- **Goal:** Prevent SageMaker Processing output prefixes or ambiguous outputs from being treated as verified evaluation artifacts.
- **Summary of changes:** The provider now selects exactly one named `evaluation` output from a completed Processing job. The S3 artifact store paginates the output prefix, requires exactly one `evaluation.json` or `evaluation.tar.gz`, resolves and downloads the exact source `VersionId`, hashes its bytes, and retains a versioned content-addressed copy before the live reader validates the report.
- **Affected files and components:** `backend/app/providers/sagemaker.py`, `backend/app/providers/artifacts.py`, focused provider/artifact/evaluation tests, `Flow.md`, and `Decisions.md`.
- **Tests and verification:** `uv run pytest -q tests/test_aws_providers.py tests/test_artifact_integrity.py tests/test_live_execution.py` passed (93 tests).
- **Known limitations and follow-up:** This is local contract verification only. No live SageMaker Processing job or AWS artifact discovery was performed.

## 2026-09-12 — `AWS-DISPATCHER-CANCELLATION-001`

- **Goal:** Prevent shutdown cancellation from releasing a run lease while its supervisor child is still executing.
- **Summary of changes:** The durable dispatcher now cancels and awaits its supervisor task when the dispatcher itself is cancelled, before heartbeat shutdown and the outer lease-release path. Added a regression test proving child cancellation precedes lease release and documented hard-kill recovery as lease-expiry/provider reconciliation.
- **Affected files and components:** `backend/app/autonomous/dispatcher.py`, `backend/tests/test_autonomous_supervisor.py`, `Flow.md`, and `Decisions.md`.
- **Verification:** The regression test failed before the fix and passed afterward; focused dispatcher and autonomous live API tests passed (21 tests). No AWS resource mutation or live provider job was performed.
- **Known limitations:** This covers cooperative asyncio cancellation. A forced process kill still depends on durable lease expiry and persisted provider IDs for reconciliation.

## 2026-09-12 — `OBJECTIVE-STARTUP-CHECKPOINT-001`

- **Goal:** Ensure an internal objective worker cannot serve requests before loading the approved immutable FunctionGemma checkpoint.
- **Summary of changes:** Added a role-aware backend entrypoint that bootstraps the exact versioned S3 bundle, verifies its digest and pinned snapshot identity, then replaces itself with Uvicorn; coordinator startup skips model staging, and any objective bootstrap failure prevents service startup. Documented the task's scoped S3/KMS dependency.
- **Affected files and components:** `backend/Dockerfile`, `backend/scripts/start_backend.py`, `backend/scripts/bootstrap_objective_checkpoint.py`, checkpoint bootstrap tests, `backend/README.md`, `infra/cdk/README.md`, `Flow.md`, and `Decisions.md`.
- **Verification:** Focused bootstrap/startup tests, Ruff, mypy, documentation sync, and diff checks; no AWS deployment or checkpoint-backed objective inference is implied.
- **Known limitations:** No staged checkpoint object or running internal objective task was verified in AWS; task download, decryption, and model loading remain live prerequisites.

## 2026-09-12 — `AWS-READINESS-RECONCILIATION-001`

- **Goal:** Reconcile implementation and AWS readiness against the concrete live-run caveat list without claiming deployment or model improvement.
- **Summary of changes:** Closed the duplicate objective-checkpoint-bootstrap path so the role-aware image entrypoint stages the immutable bundle once before Uvicorn, guarded the legacy process-local mutation routes with `410 Gone` in AWS mode, and corrected current documentation for dynamic per-experiment training data, TLS coordinator ingress, and the latest quota/readiness evidence.
- **Affected files and components:** Backend startup and route handlers/tests, live-execution and SageMaker adapter contracts, trainer/evaluator worker contracts, CDK ingress/IAM/resource contracts, and `Flow.md`, `Decisions.md`, `backend/README.md`, and `infra/cdk/README.md`.
- **Verification:** The focused startup, objective execution, and main integration tests passed. AWS Service Quotas returned `ml.g5.xlarge` training-job allowance `1.0`; one SigV4 Strands Nemotron `AgentResult` smoke returned the exact requested `OK`. Local `scripts/live_preflight.py` remains `BLOCKED` for absent bucket, table, role, image digests, objective URL, model/revision, and sealed evaluation URI. Read-only inventory found no matching CloudFormation stack, DynamoDB run table, dedicated artifact bucket, trainer/evaluator ECR repositories, ACM certificate, or Route 53 zone. No AWS resources were created or modified.
- **Known limitations:** The coordinator image import smoke passed earlier, but trainer/evaluator image builds stalled during the 554.6 MB PyTorch wheel download and were stopped safely. No FunctionGemma bundle, verified objective trajectory, SageMaker training/evaluation job, promotion, or end-to-end autonomous live run was produced. A quota allowance is not placement capacity, and one provider call does not validate all four agent schemas.

## 2026-09-12 — `CHECKPOINT-VERSION-PIN-001`

- **Goal:** Prevent CDK from silently generating an unversioned default checkpoint URI for live workloads.
- **Summary of changes:** CDK now requires an explicitly configured S3 checkpoint URI with exactly one non-empty `versionId` whenever a live checkpoint is required. Added regression coverage for missing and unversioned checkpoint configuration.
- **Affected files and components:** `infra/cdk/stacks/post_training_stack.py` and `infra/cdk/tests/test_post_training_stack.py`.
- **Verification:** CDK stack contract suite passed (37 tests); focused Ruff passed; local `cdk synth` passed using explicitly labeled review-only placeholder values. No placeholder was deployed.
- **Known limitations:** AWS deployment remains blocked by missing real checkpoint/version, trainer/evaluator image digests, sealed evaluation input, and HTTPS ingress domain/certificate. The quota allowance does not prove placement capacity; no live training or evaluation was submitted.

## 2026-09-13 — `AWS-DISPATCHER-LEASE-EXCLUSIVITY-001`

- **Goal:** Prevent two concurrent dispatcher scans in one process from claiming and executing the same live run lease.
- **Summary of changes:** In-memory and DynamoDB `claim_lease` now reject every unexpired lease, including one with the same owner string. Explicit renewal remains a separate owner-checked compare-and-swap path. Added repository tests for same-owner rejection and renewal, plus a stale-scan concurrent-dispatch test proving only one supervisor starts.
- **Affected files and components:** `backend/app/autonomous/repository.py`, `backend/tests/test_autonomous_repository.py`, `backend/tests/test_autonomous_supervisor.py`, `Flow.md`, and `Decisions.md`.
- **Verification:** The two in-memory/concurrency regression tests and DynamoDB lease regression test failed before the fix and passed after it. Focused `test_autonomous_repository.py` and `test_autonomous_supervisor.py` passed; Ruff and mypy passed for the changed backend files. No deployment or AWS operation was performed.
- **Known limitations:** DynamoDB conditional writes and SageMaker reconciliation were validated locally with repository/provider contracts only; no concurrent dispatcher or recovery test was run against a deployed AWS stack.

## 2026-09-13 — `RUNTIME-QUOTA-AND-BEDROCK-READINESS-001`

- **Goal:** Keep live runtime readiness truthful for GPU training, GPU evaluation, and provider-backed Nemotron agent calls.
- **Summary of changes:** Runtime CDK now requires validated, separate SageMaker training and Processing quota codes and injects both into the coordinator. Readiness queries each account quota and blocks if either is missing, unavailable, or below the requested instance count. The coordinator role also receives only `bedrock:GetFoundationModel` and `bedrock:InvokeModel` on its exact regional configured foundation-model ARN, using AWS's foundation-model ARN shape.
- **Affected files and components:** `infra/cdk/stacks/runtime_stack.py`, runtime CDK tests, `backend/app/live_execution.py`, focused live-execution tests, `Flow.md`, and `backend/README.md`.
- **Verification:** Runtime CDK synthesis tests passed (51); backend live-execution, preflight-hardening, and readiness tests passed (78); Ruff passed on all changed Python files; mypy passed for the runtime stack, its tests, and `backend/app/live_execution.py`; documentation sync and `git diff --check` passed. No AWS resources were created or deployed by this change.
- **Known limitations:** A Service Quotas allowance does not guarantee SageMaker placement capacity. Runtime deployment and live provider execution remain separate verification steps.

## 2026-09-13 — `PROCESSING-LOCALPATH-CONTRACT-001`

- **Goal:** Verify the SageMaker Processing evaluator's input mount contract alongside immutable Processing-output discovery.
- **Summary of changes:** Documented the fixed `base_model`, `candidate`, `champion`, and `sealed` `LocalPath` to `SM_CHANNEL_*` mapping and the shared output path in `Flow.md`; recorded the contract in `Decisions.md`. Audited the existing provider/artifact-store implementation and regression tests: the completed job's named output remains an S3 prefix, listing requires exactly one `evaluation.json` or `evaluation.tar.gz`, and the selected object's exact `VersionId` is used for download and byte-derived SHA-256 retention.
- **Affected files and components:** SageMaker provider and S3 artifact-store contracts/tests, `Flow.md`, and `Decisions.md`.
- **Verification:** Focused provider/artifact/live-evaluation tests passed (99); Ruff passed on the provider/artifact modules and focused tests; targeted mypy passed for `app/providers/sagemaker.py`, `app/providers/artifacts.py`, and `app/live_execution.py`; docs sync and `git diff --check` passed.
- **Known limitations:** The broader `mypy app tests` run reports 178 errors across 29 source/test files; targeted production-file mypy passes. No AWS Processing job or live artifact discovery was performed.

## 2026-09-13 — `AWS-DISPATCHER-RECOVERY-IDEMPOTENCY-001`

- **Goal:** Verify restart recovery reconciles an accepted SageMaker training request after its durable lease expires, without submitting a duplicate job.
- **Summary of changes:** Added an integrated in-memory regression that persists a deterministic SageMaker operation intent, simulates a lost create response after provider acceptance, expires the prior lease, and resumes through a fresh dispatcher/supervisor; recovery completes with one training submission.
- **Affected files and components:** `backend/tests/test_autonomous_supervisor.py`, durable dispatcher recovery, supervisor operation intents, and SageMaker reconciliation contracts.
- **Verification:** Focused dispatcher, supervisor, repository, live API, and SageMaker reconciliation suites passed (124 tests); scoped Ruff and production-file mypy passed; living-document sync and `git diff --check` passed.
- **Known limitations:** The regression uses in-memory repositories and a SageMaker contract double. It does not validate a deployed DynamoDB lease, SageMaker control-plane timing, or a live AWS recovery.
