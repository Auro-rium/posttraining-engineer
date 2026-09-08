# Autonomous Post-Training Engineer for AWS Agents for Humans Hackathon

**Professional Agents Track** - An autonomous agent system that performs end-to-end post-training optimization for Gemma models using AWS Strands Agents (mandatory) with optional Amazon AgentCore integration for enhanced Technical Implementation scoring.

This system transforms the manual post-training experimentation loop that AI/ML engineers perform into an autonomous workflow:

**Manual Process:**
```
Run model → Inspect failures → Figure out why it sucks → Collect/repair data →
Choose training recipe → Fine-tune → Run evals → Discover regression →
Change experiment → Train again
```

**Autonomous Process:**
```
Gemma + Environment + Evals
             ↓
          OPTIMIZE (via 8 specialized Strands agents)
             ↓
      Better Gemma Adapter with verified improvement
```

The system implements eight specialized Strands agents that work together to:
1. Benchmark Gemma performance in AgentGym service recovery environment
2. Analyze failure patterns
3. Generate research hypotheses
4. Curate verified post-training datasets
5. Design optimal QLoRA configurations
6. Execute training jobs
7. Evaluate checkpoints for improvement/regression
8. Make deterministic promotion decisions

Built for the AWS Agents for Humans Hackathon (deadline: September 14, 2026, 5:00 PM PDT).

## What is implemented

The Python backend provides eight logical roles behind three service modes:

| Service mode | Logical roles | Responsibility |
| --- | --- | --- |
| `coordinator` | Workflow coordinator (not counted as a specialist) | Owns run state, public API, sequencing, budget limits, and events. |
| `research` | Failure Analyst, Research Agent, Data Curator, Training Designer | Grounds a hypothesis with RAG, proposes replay-verifiable data, and selects a bounded QLoRA configuration. |
| `execution` | Benchmark Runner, Training Executor, Evaluation Agent, Champion Manager | Produces trajectories, launches training, evaluates identical task sets, and applies deterministic gates. |

The same immutable container runs all modes through `SERVICE_ROLE`. The AWS submission path is Strands-first and is designed to use Amazon Bedrock, S3, DynamoDB, SageMaker, and optional AgentCore/CloudWatch integrations. The repository now includes a bounded five-run history/graph contract, guarded objective-worker and SageMaker lifecycle boundaries, and metadata-only telemetry. The current repository run still uses in-memory state and explicit `EXPLANATION` adapters; those outputs are not a hackathon result.

### Reasoning model and post-training target

Every working agent uses the single pinned Bedrock reasoning model
`nvidia.nemotron-super-3-120b` (NVIDIA Nemotron Super 3 120B). This is the
agent brain for planning, analysis, data selection, training dispatch,
evaluation interpretation, and gate explanation. It is not the model being
improved. The post-training target remains the user-supplied
`google/functiongemma-270m-it` checkpoint, pinned by an immutable Hugging Face
revision before a live run.

Each role uses a versioned `AgentPromptContract`. Prompts state the mission,
typed inputs and outputs, preconditions, stop conditions, evidence labels,
sealed-evaluation rules, and forbidden actions. Nemotron may be inventive when
forming hypotheses, experiment choices, or repair strategies, but it cannot
invent trajectories, metrics, provider IDs, artifacts, approvals, or promotion
decisions. Prompt version and SHA-256 are recorded as safe metadata in the run
manifest and telemetry so a comparison can be reproduced without logging raw
prompts or completions.

Each specialist has an explicit safety contract: Benchmark Runner accepts train-side evidence only; Failure Analyst must ground clusters in trajectory IDs; Research Agent must return a falsifiable hypothesis; Data Curator may propose but never verify repairs; Training Designer must stay inside the QLoRA whitelist; Training Executor may report only provider-backed artifacts; Evaluation Agent returns objective evidence without deciding promotion; and Champion Manager explains but cannot override the deterministic gate.

In a live AWS deployment, the coordinator must fail fast unless its authenticated service and artifact destinations are configured. The local path is intentionally a credential-free demonstration and does not silently promote simulated evidence.

The API is intentionally small:

- `POST /api/runs`
- `GET /api/runs/{run_id}`
- `GET /api/runs/{run_id}/experiments`
- `POST /api/runs/{run_id}/step`
- `POST /api/runs/{run_id}/auto`
- `POST /api/runs/{run_id}/cancel`
- `GET /api/runs/compare`
- `GET /api/runs/graph`
- `GET /api/live/readiness`
- `GET /api/traces`
- `GET /api/cycles`
- `POST /api/demo/reset-environment`
- `GET /health`

`POST /api/runs/{run_id}/auto` runs the bounded local workflow in the current process. The comparison endpoints accept one to five run IDs and render only records with measured baseline/candidate metrics. A production AWS adapter must replace `active_runs` with durable state and persist provider job IDs before polling. Use explicit `/step` calls for the repeatable local walkthrough.

See [Flow.md](Flow.md) for control flow and [Decisions.md](Decisions.md) for the reasons behind the architecture.

## Capability and evidence matrix

The labels below are deliberate submission boundaries. `LOCAL_DEMO` describes
what this checkout can run without credentials. `LIVE_AWS` describes the
capability that must be connected and artifact-verified before it can be
claimed; no `LIVE_AWS` run is included in this repository state.

| Capability | `LOCAL_DEMO` (current checkout) | `LIVE_AWS` (claim requires evidence) |
| --- | --- | --- |
| Strands specialist workflow | Eight role contracts initialize and execute a bounded local workflow. | Bedrock-backed agent decisions and authenticated service boundaries. |
| Run control | In-memory `OptimizationRun` plus a process-local bounded history registry; `/step` advances one phase and `/auto` runs the remaining phases in-process. | Durable run ownership, idempotent commands, and restart-safe orchestration through DynamoDB. |
| Benchmark and trajectories | Service-recovery fixture returns explanatory trajectories and simulated metrics. | Objective environment worker produces immutable S3 artifacts with hashes and provider/run IDs. |
| Curation and QLoRA design | Agents return structured demonstration outputs and bounded configurations. | Replay-verified training rows and a recorded, budget-compliant training specification. |
| Training execution | Missing objective/training adapters fail closed; no checkpoint or job is fabricated. | SageMaker job submission, bounded polling, logs, and an immutable checkpoint manifest. |
| Evaluation and promotion | Deterministic gate code is exercised locally, but local metrics are simulated and cannot prove improvement. | Independent evaluation on identical sealed inputs, with artifact-backed metrics and a deterministic promotion decision. |
| Continuous trigger and event ingestion | `TraceEvent` validation, duplicate suppression, ordered in-memory storage, and deterministic per-run threshold triggering are covered locally; they are not wired to the FastAPI run routes. | EventBridge/SQS delivery into durable event storage, idempotent cycle creation, and a worker that starts the post-training run. |
| State, artifacts, and run API | Process memory plus JSON/SVG comparison responses; no durable local run/event log. | DynamoDB run history/events, S3 artifacts, and a resumable ordered event stream. |
| Agent reasoning and prompt provenance | Nemotron prompt contracts can be inspected and tested without exposing task contents. | Bedrock-backed Nemotron decisions with prompt hashes, model ID, run manifest, and provider-backed evidence. |
| Execution view | Browser demo shows phase progression and explanatory agent activity; it never presents simulated metrics as live improvement. | Browser view consumes authenticated lifecycle events and displays only safe metadata, real job states, and retained artifact references. |

Telemetry/observation is emitted at run, phase, job, and promotion transitions.
Events correlate `run_id`, run number, experiment, phase, and provider job while
redacting prompts, outputs, trajectories, held-out data, credentials, and
unknown free-form strings. Logging is the default sink; OTLP/OpenTelemetry is
optional and sink failures never change run outcomes.

The evidence labels used in outputs are `LIVE` (verified by the current AWS
request), `PRIOR_VERIFIED_RUN` (a prior run with provider IDs and hashes), and
`EXPLANATION` (fixture or simulation). The current checkout produces only
`EXPLANATION` workflow evidence.

## Evidence boundary

Every externally shown artifact has one of three labels:

- `LIVE`: produced by the request currently executing.
- `PRIOR_VERIFIED_RUN`: produced by a real earlier run with hashes and provider job identifiers.
- `EXPLANATION`: fixture or explanatory content that is never presented as measured output.

Strands model calls may analyze failures and propose hypotheses, repairs, and configurations. They may not grade their own repairs or candidates. Deterministic code enforces replay admission, the five-run budget, identical evaluation inputs, verified evidence provenance, and checkpoint promotion. Held-out tasks are excluded from prompts, repairs, training data, and telemetry.

## Credential-free contract verification

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env
cd backend
uv sync --frozen --extra cloud --extra dev
uv run pytest tests -q
uv run ruff check app tests scripts
uv run mypy app
```

These tests validate schemas, leakage barriers, deterministic orchestration, promotion gates, and API contracts without calling Bedrock, SageMaker, AgentCore, or AWS. Passing them is not evidence that a live AWS run occurred.

For a non-destructive live smoke test against an existing AWS account and S3 bucket:

```bash
cd backend
uv run --extra cloud python scripts/live_smoke.py \\
  --bucket YOUR_EXISTING_BUCKET \\
  --region us-east-1
```

The smoke test verifies STS identity, one Strands/Bedrock response, and one
S3 put/get/delete round trip. It creates no tables, instances, training jobs,
or persistent infrastructure. To exercise all eight specialist prompts live
without creating resources:

```bash
uv run --extra cloud python scripts/live_agentic_test.py
```

For the guarded live path, run the read-only preflight before approving any
compute. Then authorize one run at a time; the batch command pauses for a new
approval token before each subsequent run and never exceeds five runs or the
$25 budget:

```bash
uv run --extra cloud python scripts/live_preflight.py
uv run --extra cloud python scripts/live_run.py --approval-token TOKEN
uv run --extra cloud python scripts/live_batch.py
```

The browser execution view renders the same lifecycle metadata as the API:
animated role bots move through launcher, benchmark, failure analysis, data
curation, training, evaluation, and promotion. It must show a blocked or
failed phase plainly and never substitute an animation for a provider job,
metric, or artifact.

Docker Compose is retained only as a three-role image and healthcheck smoke harness:

```bash
docker compose up --build
curl --fail http://localhost:8000/health
curl --fail http://localhost:8001/health
curl --fail http://localhost:8002/health
```

Coordinator, research, and execution listen on ports 8000, 8001, and 8002 respectively. Responses from this harness use local `EXPLANATION` fixtures and must never be shown as model improvement, A2A, or training evidence.

Run repository and infrastructure checks from the repository root:

```bash
cd backend
uv run pytest tests -q
cd ..
backend/.venv/bin/python backend/scripts/check_docs_sync.py
docker compose config --quiet
```

## AWS deployment status

The AWS/Strands implementation has not been executed as a live post-training
run in this checkout. The guarded live controller and AWS adapters exist, but
the public local `/api/runs` workflow still uses process-local state and
`EXPLANATION` fixtures. Do not describe that path as a live AWS result.

The repository includes a CDK foundation in [infra/cdk](infra/cdk). It can
provision the retained storage, ECR, VPC, ECS/Fargate, CloudWatch, IAM, and
SageMaker role boundary described in [infra/cdk/README.md](infra/cdk/README.md).
It does not provision the objective worker, trainer/evaluator images,
FunctionGemma checkpoint or AgentEval data, Bedrock model entitlement, approval
secret, GPU placement capacity, or public ingress/authentication. Those are
deployment prerequisites and must be configured before the read-only
`/api/live/readiness` or `scripts/live_preflight.py` can report `READY`.

The live controller reuses existing resources and never deletes persistent
S3, DynamoDB, IAM, ECR, VPC, or ECS infrastructure. It only stops the current
run's in-progress SageMaker jobs during cleanup. A real run is claimable only
after a successful preflight, an explicit signed approval, provider job IDs,
immutable artifact hashes, held-out evidence, and a deterministic promotion
decision have been retained.

## Living documentation

All contributors and agents must follow [AGENTS.md](AGENTS.md): read the living documents before work, update `Flow.md` and `Decisions.md` when behavior or choices change, run checks, and append an accurate `ChangeLog.md` entry after verification. Pull-request CI rejects implementation changes without a changelog update.

## Current limitations

- No live Gemma inference, SageMaker training job, held-out evaluation, checkpoint improvement, or model metric is claimed until its real artifact, provider job IDs, and manifest digest are recorded.
- Evaluation is explicitly requested as AgentGym AgentEval (`agent-eval-v1`); live reports must include the immutable AgentEval manifest SHA-256 and are accepted only when champion and candidate use the same suite/version.
- Credential-free adapters execute only the local service-recovery explanation fixture; live environments require configured AWS workers and model access.
- S3/DynamoDB/SageMaker/AgentCore resources and the sandboxed objective worker are deployment prerequisites and are not provisioned by this repository.
- A process restart currently loses in-memory runs; durable AWS persistence and provider job reconciliation remain required before claiming recoverability.
- Authentication, multi-tenancy, billing, and multiple target models are out of scope. The hackathon execution view is a safe observer only; it does not grant approval or mutate AWS resources.
