# Autonomous Live Backend Design

## Scope

This design implements the backend-only, fail-closed autonomous post-training loop for `google/functiongemma-270m-it`. One bounded HMAC approval authorizes one top-level run containing at most five sequential experiments. The browser is an observer only; `/step` remains a legacy debugging surface and never drives `/api/live` runs.

## Control plane

`POST /api/live/runs/prepare` validates an immutable request and persists a `PREPARED` run plus canonical approval packet. `POST /api/live/runs` validates and atomically consumes the packet digest, moves the run to `QUEUED`, and returns `202`. A DynamoDB lease dispatcher claims queued or recoverable nonterminal runs and invokes `AutonomousRunSupervisor`.

The supervisor persists every phase transition before the next side effect. Provider submissions use deterministic operation keys and SageMaker job names. If submission outcome is ambiguous or the process restarts, the supervisor reconciles the persisted provider ID or deterministic name before it may submit. It never fabricates metrics, artifacts, checksums, or provider IDs.

## Durable record

Each top-level run stores immutable model/checkpoint/benchmark approval scope, status and phase, optimistic version, event sequence, lease owner/expiry, cancellation and safe-stop flags, budget ledger, baseline/champion evidence, current experiment fields, provider job IDs, artifact references, stop reason, and experiment history. DynamoDB items use `pk=RUN#<run_id>` with `STATE`, `EVENT#<sequence>`, `EXP#<number>`, and `OP#<experiment>#<phase>` sort keys. A state transition and its metadata-only event are one transaction.

## Autonomous loop

After readiness and approval verification, the supervisor obtains a real baseline through the objective worker. For each experiment it passes verified failures and all prior experiment evidence to the failure analyst and research agent, admits only verifier-confirmed SFT trajectories, validates a bounded SFT+QLoRA configuration, launches and reconciles SageMaker training, validates the produced checkpoint, launches sealed SageMaker evaluation, and applies the existing deterministic promotion gate. Rejected candidates remain in history. A promoted candidate becomes the champion and the next experiment's trainable parent adapter.

The loop stops at cancellation, safe stop, approval expiry, max experiments, hard budget, target champion score, no valid hypothesis, invalid evidence/provenance, or unrecoverable provider failure. Cancellation requests active provider job termination. Safe stop and approval expiry allow the already-active provider job to reach terminal state and persist verified output, but prohibit submission of a later phase.

## Objective boundary

The initial environment is `service-recovery-v1`, with only `get_logs`, `inspect_service`, `read_config`, `edit_config`, `restart_service`, and `run_healthcheck`. A shared deterministic engine owns task reset, tool execution, hidden verification, binary reward, trajectory provenance, and train/validation/hidden-test separation. The authenticated objective service exposes training and replay material; the independent evaluator image embeds the same engine and alone reads the sealed validation/hidden bundle. Hidden task contents never enter prompts, repair generation, or training rows.

## Data and workers

Every JSONL SFT row contains conversation/tool messages plus source trajectory, failure label, verifier confirmation, and source type. The dataset manifest records dataset/run/experiment IDs, source references, row count, SHA-256, S3 URI, creation time, and target failure classes.

The trainer image performs SFT+QLoRA only, using the pinned FunctionGemma base plus the current champion adapter when present. The fixed search space is rank 8/16/32, alpha 16/32/64, dropout 0/0.05/0.1, learning rate 1e-4/2e-4/5e-4, epochs 1/2/3, sequence length 512/1024, batch size 1/2/4, gradient accumulation 4/8/16, and attention projections `q_proj,k_proj,v_proj,o_proj`. The evaluator returns only objective metrics and provenance artifacts.

## AWS and reasoning

AWS resources are versioned encrypted S3, DynamoDB, separate ECR repositories, SageMaker roles/jobs, an authenticated ECS objective service, and CloudWatch metadata. The system may safely reuse resources only when ownership/configuration checks pass; it does not delete unrelated resources. Infrastructure creation, image push, quota requests, and live job launch remain external side effects that require an explicit execution boundary.

All judgment-heavy specialists use Strands with `nvidia.nemotron-super-3-120b` through explicit AWS SigV4. If model access fails, readiness is blocked and no substitute or simulated reasoning is used.

## API and evidence

The live API provides prepare, start, status, cancel, safe-stop, ordered events, experiments, artifacts, and readiness. Mutating calls require an idempotency key. Readiness is true only when DynamoDB, S3, role, images, quotas, checkpoint, objective worker, Nemotron, and approval secret are launch-ready. Event payloads contain only safe IDs, status, timestamps, latencies, costs, evidence labels, and artifact/provider references.

## Acceptance

Local acceptance requires the full pre-existing test suite plus tests for autonomous progression, restart reconciliation, duplicate prevention, bounded experiments, budget, cancellation, safe stop, approval scope, leakage protection, verifier admission, artifact hashes, provider failures, lineage, evidence carry-forward, deterministic promotion, and truthful readiness. Live acceptance additionally requires real objective artifacts, SageMaker job IDs, checkpoint hashes, held-out metrics, telemetry, cleanup, and a final champion or exact stop reason.
