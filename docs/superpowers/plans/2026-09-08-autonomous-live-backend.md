# Autonomous Live Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a durable, one-call autonomous FunctionGemma post-training backend that performs up to five evidence-backed experiments without manual phase advancement.

**Architecture:** A DynamoDB-backed run/operation/event repository and lease dispatcher drive a typed `AutonomousRunSupervisor`. Judgment-heavy agent adapters and deterministic objective/training/evaluation adapters are injected, while persisted intents, deterministic provider names, artifact verification, and deterministic promotion make recovery fail closed.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, boto3/DynamoDB/S3/SageMaker/Bedrock, Strands, pytest, AWS CDK.

**Spec:** `docs/superpowers/specs/2026-09-08-autonomous-live-backend-design.md`

## Global Constraints

- Backend only; do not modify `frontend/`.
- Target model is exactly `google/functiongemma-270m-it` at a 40-character immutable Hugging Face commit SHA.
- Reasoning model is exactly `nvidia.nemotron-super-3-120b`; no silent provider substitution.
- SFT+QLoRA only; no DPO, PPO, GRPO, reward-model training, or full fine-tuning.
- Maximum five experiments and maximum approved cost USD 25.
- Held-out inputs never enter prompts, training data, curation, repair generation, or telemetry.
- Only deterministic code verifies artifacts/repairs, enforces budget, computes metrics, and promotes candidates.
- Missing adapters, artifacts, access, jobs, or provenance produce `BLOCKED` or `FAILED`, never simulated output.
- Persist a transition before the next side effect and reconcile deterministic provider operations after restart.
- Use TDD: each behavior test is run red before production implementation and green afterward.

---

### Task 1: Durable autonomous run state and repository

**Files:**
- Create: `backend/app/autonomous/models.py`
- Create: `backend/app/autonomous/repository.py`
- Create: `backend/app/autonomous/__init__.py`
- Test: `backend/tests/test_autonomous_repository.py`

**Interfaces:**
- Produces: typed `AutonomousRunState`, `RunPhase`, `AutonomousRunStatus`, `ExperimentRecord`, `RunOperation`, `RunEventRecord`, and repository protocol/implementations with create/get/transition, atomic event sequence, approval consumption, lease claim/renew/release, recoverable scan, operation intent/result, and paginated event/history reads.

- [ ] Write repository contract tests for conditional creation, atomic transition/event, approval replay rejection, leases, pagination, and operation reconciliation.
- [ ] Run the focused tests and confirm failures are caused by missing autonomous contracts.
- [ ] Implement immutable/validated models plus in-memory and DynamoDB repositories using optimistic conditional writes/transactions.
- [ ] Run focused tests, Ruff, and mypy; commit the task.

### Task 2: Deterministic service-recovery objective engine

**Files:**
- Create: `backend/app/objective/models.py`
- Create: `backend/app/objective/engine.py`
- Create: `backend/app/objective/service.py`
- Create: `backend/app/objective/__init__.py`
- Test: `backend/tests/test_objective_engine.py`

**Interfaces:**
- Produces: split-safe task/trajectory/replay/dataset contracts, `ServiceRecoveryEngine`, verifier-only admission, and authenticated FastAPI benchmark/curation endpoints for train/replay scope.

- [ ] Write failing tests for allowed tools, deterministic reward, verifier-only admission, provenance, and hidden-split rejection.
- [ ] Run focused tests red.
- [ ] Implement the narrow in-memory engine and worker contract without exposing hidden verifier fields.
- [ ] Run focused tests, Ruff, and mypy; commit the task.

### Task 3: Bounded reasoning-agent handoffs

**Files:**
- Create: `backend/app/autonomous/agents.py`
- Modify: `backend/app/agents/prompt_contract.py`
- Test: `backend/tests/test_autonomous_agents.py`

**Interfaces:**
- Consumes: verified trajectory references and experiment history.
- Produces: typed `FailureCluster`, `ResearchHypothesis`, `CuratedDatasetPlan`, and `QLoRAConfig`; provider-backed adapters parse strict JSON and fail closed; deterministic QLoRA validator enforces the exact search space.

- [ ] Write failing tests for schema validation, previous-evidence inclusion, duplicate failed-hypothesis rejection, provider failure, and QLoRA bounds.
- [ ] Run focused tests red.
- [ ] Implement minimal strict adapters using the pinned Nemotron provider and prompt contracts.
- [ ] Run focused tests, Ruff, and mypy; commit the task.

### Task 4: Trainer and evaluator worker images

**Files:**
- Create: `backend/workers/trainer/train.py`
- Create: `backend/workers/trainer/Dockerfile`
- Create: `backend/workers/trainer/requirements.txt`
- Create: `backend/workers/evaluator/evaluate.py`
- Create: `backend/workers/evaluator/Dockerfile`
- Create: `backend/workers/evaluator/requirements.txt`
- Test: `backend/tests/test_worker_contracts.py`

**Interfaces:**
- Produces: strict SageMaker channel/environment parsers, deterministic manifests/checksums, real Transformers/PEFT QLoRA training entrypoint, and separate sealed evaluator entrypoint sharing the objective engine.

- [ ] Write failing contract tests for required inputs, sealed input isolation, output manifest/checksum shape, and refusal to report absent artifacts.
- [ ] Run focused tests red.
- [ ] Implement executable workers with lazy ML imports so contract tests do not require GPU libraries.
- [ ] Run focused tests and static checks; commit the task.

### Task 5: Autonomous supervisor and recovery dispatcher

**Files:**
- Create: `backend/app/autonomous/supervisor.py`
- Create: `backend/app/autonomous/dispatcher.py`
- Test: `backend/tests/test_autonomous_supervisor.py`

**Interfaces:**
- Consumes: Task 1 repository, Task 3 agents, existing objective/SageMaker/artifact/promotion adapters.
- Produces: `AutonomousRunSupervisor.run_optimization(run_id)` and lease-based `AutonomousRunDispatcher`; deterministic operation names prevent duplicate submission and experiment N+1 receives complete prior evidence.

- [ ] Write failing tests for baseline-to-terminal progression, restart resume, duplicate prevention, five-experiment bound, budget, cancel, safe stop, expiry, provider failures, promotion lineage, and evidence carry-forward.
- [ ] Run focused tests red.
- [ ] Implement the persisted state machine, polling/reconciliation, budget reservations, stop conditions, cleanup, and telemetry hooks.
- [ ] Run focused tests, Ruff, and mypy; commit the task.

### Task 6: Live API, startup wiring, and truthful readiness

**Files:**
- Create: `backend/app/api/autonomous_live.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/api/live_readiness.py`
- Modify: `backend/app/live_execution.py`
- Test: `backend/tests/test_autonomous_live_api.py`
- Modify: `backend/tests/test_live_readiness_api.py`

**Interfaces:**
- Consumes: repository/supervisor/dispatcher and existing approval/preflight helpers.
- Produces: prepare/start/status/cancel/safe-stop/events/experiments/artifacts endpoints, startup recovery and shutdown, single bounded approval consumption, idempotency handling, and real-invocation SigV4 readiness.

- [ ] Write failing API/readiness tests including no `/step` dependency and every launch prerequisite.
- [ ] Run focused tests red.
- [ ] Implement dependency-injected router/factory and dispatcher lifecycle.
- [ ] Run focused tests, Ruff, and mypy; commit the task.

### Task 7: AWS infrastructure and immutable checkpoint staging

**Files:**
- Modify: `infra/cdk/stacks/post_training_stack.py`
- Modify: `infra/cdk/app.py`
- Create: `infra/cdk/cdk.json`
- Create: `backend/scripts/stage_functiongemma_checkpoint.py`
- Test: `backend/tests/test_checkpoint_staging.py`

**Interfaces:**
- Produces: separate trainer/evaluator ECR repositories, least-privilege SageMaker/objective roles, authenticated objective service resources, required environment outputs, and a staging command that rejects mutable/incomplete/gated checkpoints and uploads a deterministic content-addressed bundle.

- [ ] Write failing staging tests for immutable revision, required files, deterministic SHA-256, and S3 version metadata.
- [ ] Run focused tests red.
- [ ] Implement staging logic and CDK resources without deploying them.
- [ ] Run tests, static checks, and `cd infra/cdk && cdk synth`; commit the task.

### Task 8: Living docs and whole-system verification

**Files:**
- Modify: `Flow.md`
- Append: `Decisions.md`
- Append: `ChangeLog.md`
- Modify: `backend/README.md`

**Interfaces:**
- Consumes: final implemented behavior and fresh verification output.
- Produces: accurate architecture, API/runbook, evidence boundary, readiness blockers, and validation record.

- [ ] Update Flow and append a superseding AWS autonomous-supervisor decision.
- [ ] Document prepare/start/cancel/safe-stop and recovery semantics plus live prerequisite setup.
- [ ] Run `uv run pytest -q`, focused Ruff/mypy, docs sync, diff check, and CDK synth.
- [ ] Record actual readiness output and whether a live run completed; never convert blockers into success claims.
