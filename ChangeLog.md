# Change Log

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
