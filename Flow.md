# AWS Strands Workflow

This is the current operational map for the AWS Agents for Humans submission.
The workflow is intentionally narrow: one professional user, one service-recovery
environment, eight specialist Strands agents, and deterministic gates.

## System boundaries

```text
HTTP client
    -> FastAPI coordinator
       -> OptimizationRun state
          -> Benchmark Agent
          -> Failure Analyst
          -> Research Agent
          -> Data Curator
          -> Training Designer
          -> Training Executor
          -> Evaluation Agent
          -> Champion Manager
       -> final status and artifact references

Live AWS target (not yet deployed):
  Bedrock -> agent decisions
  S3 -> immutable trajectories, datasets, checkpoints
  DynamoDB -> durable run state and events
  SageMaker -> QLoRA training
  AgentCore/CloudWatch -> optional hosting and observability

Run history and observation contracts:

```text
RunRegistry -> DynamoDB transaction (unique run number 1..5)
           -> ComparisonDTO -> JSON chart data + self-contained SVG
Run/phase/job transitions -> TelemetryRecorder
           -> redacted logger/exporter and optional OpenTelemetry spans
```

Telemetry is operational metadata only. Prompts, raw completions, trajectories,
held-out tasks, credentials, and secret-like values are redacted and event
attributes are immutable after emission.
```

## Reasoning model and prompt contract

All eight specialist agents use the fixed Bedrock model
`nvidia.nemotron-super-3-120b` (NVIDIA Nemotron Super 3 120B). The coordinator
may orchestrate these agents, but it does not replace their model or relax
their contracts. The optimized target is FunctionGemma, represented by a
separately pinned Hugging Face checkpoint; Nemotron is the reasoning model, not
the post-training target.

Each handoff is governed by a versioned `AgentPromptContract`. A contract
contains the role mission, required typed inputs, exact output shape,
preconditions, failure/blocked behavior, evidence-class rules, and forbidden
actions. Creative reasoning is encouraged only for falsifiable hypotheses,
experiment alternatives, and bounded repair strategies. It can never create a
trajectory, metric, provider job ID, artifact, approval, or promotion result.
The contract version, model ID, and prompt SHA-256 travel as metadata in the
manifest and telemetry. Raw prompts, completions, trajectories, and sealed
evaluation contents do not.

The current local path uses the same role boundaries but keeps state in memory
and returns `EXPLANATION` fixtures. It is a reproducible demonstration, not
live AWS evidence. `backend/scripts/live_smoke.py` separately verifies AWS
identity, one Strands/Bedrock call, and a temporary S3 artifact round trip; it
does not create compute or persistent infrastructure.

## End-to-end workflow

1. `POST /api/runs` creates an `OptimizationRun` with a target Gemma model,
   service-recovery environment, objective, and bounded budget.
2. `POST /api/runs/{run_id}/step` advances exactly one phase. `auto` advances
   the remaining phases in the current process.
3. Benchmark Agent generates train-side service-recovery trajectories and a
   baseline signal. Failure Analyst clusters the observed failures.
4. Research Agent proposes a repair hypothesis. Data Curator turns decision
   points into candidate SFT rows. Training Designer selects an allowed QLoRA
   configuration.
5. Training Executor submits or simulates a bounded training job and records
   its artifact reference. Evaluation Agent compares champion and candidate on
   the same task set.
6. Champion Manager applies deterministic improvement, regression, validity,
   budget, and provenance gates. A rejected candidate remains in history; only
   a passing candidate can replace the champion.
7. The run ends after promotion, rejection of the allowed candidates,
   cancellation, or a terminal failure.

The comparison API exposes `GET /api/runs/compare` and
`GET /api/runs/graph` for one to five persisted run IDs. A graph is only
rendered when every selected run has measured baseline/candidate metrics;
pending or explanatory records fail closed rather than becoming a score.

## Continuous API and event flow

The API is a command-and-observe loop. A client creates one run, advances it
with bounded commands, and reads the same run state after every transition:

```text
POST /api/runs
  -> run.created
  -> POST /api/runs/{run_id}/step (repeat, or one /auto request)
     -> phase.started
     -> phase.completed | phase.failed
     -> artifact.recorded (when a phase returns an artifact reference)
  -> GET /api/runs/{run_id}              (current phase/status/metrics)
  -> GET /api/runs/{run_id}/experiments  (artifact-reference history)
  -> POST /api/runs/{run_id}/cancel      (terminal cancellation)
```

The logical phase sequence is `initialized` -> `benchmark` ->
`analyze_failures` -> `research` -> `curate_data` -> `design_training` ->
`execute_training` -> `evaluate` -> `promote_decision` -> `completed`.
Each phase receives the run state and references produced by the preceding
phase; a failure stops progression and must remain visible to the caller.

In `LOCAL_DEMO`, `/step` returns the phase result inline and `/auto` schedules
the remaining phases in the current process. The current implementation does
not persist the logical events above, expose an SSE endpoint, or survive a
process restart. These are not hidden capabilities.

In `LIVE_AWS`, each transition is an append-only event containing at least the
run ID, monotonically increasing event ID, phase, event type, timestamp,
status, and references to immutable artifacts. DynamoDB is the source of
durable run/event state; S3 stores artifact bytes and manifests. A reconnecting
client reads events after its last event ID, while `GET /api/runs/{run_id}`
remains the authoritative status snapshot. Provider job IDs must be persisted
before polling so a restart can reconcile an in-flight training job rather
than submit a duplicate.

## Continuous post-training trigger flow

The continuous package accepts agent-run telemetry independently from the
request/response run API. A producer emits one immutable `TraceEvent` with an
`eventId`, `traceId`, `runId`, `eventType`, timezone-aware timestamp, source,
optional status, and structured payload. The event path is:

```text
Agent run
  -> EventBridge or SQS adapter (LIVE_AWS) / mapping (LOCAL_DEMO)
  -> validate TraceEvent
  -> claim eventId for idempotency
  -> append accepted event to the ordered trace store
  -> accumulate events per run within the configured threshold/window
  -> emit deterministic CycleRequest(event_ids, run_id, cycle_id)
  -> start one PostTrainingCycle
  -> benchmark -> evaluate -> approve | reject -> optional rollback
```

Duplicate deliveries are acknowledged as duplicates and never count toward a
cycle. Invalid events are rejected without persistence. Threshold and failure
policies are deterministic and isolated by `runId`; the cycle ID is derived
from the run and accepted event IDs, making retries safe.

`LOCAL_DEMO` currently provides the validated event contract, in-memory
deduplication/store, threshold trigger, and append-only cycle state machine for
deterministic tests. It does not connect those primitives to the FastAPI run
routes or durable storage. `LIVE_AWS` must connect EventBridge/SQS, durable
event/cycle records, and a worker that invokes the bounded run phases; a
provider or persistence failure must stop the cycle rather than manufacture a
promotion result.

The `app.posttraining` domain package keeps the cycle's scientific record
separate from agent output: artifacts require lowercase SHA-256 digests,
verified evidence carries the benchmark suite/version and manifest digest, and
the promotion gate requires compatible verified evidence plus the configured
improvement and regression thresholds. A cycle may move through
`created -> benchmarked -> evaluated -> approved | rejected`, with rollback
available only from `approved`; every transition is retained as an append-only
event. Fixed-seed benchmark helpers provide deterministic local evaluation
ordering, but a callback-backed result is not live AWS evidence by itself.

The `app.api.continuous_post_training` router is installed by `main.py` for
trace intake and operator decisions. The application currently supplies its
process-local repository; a live deployment must replace it through the
repository dependency hook with a durable implementation. Its local repository
creates cycles in `pending_approval`, records ordered status events, moves
approval to `queued`, and exposes status/events/artifact metadata. Approval is
not training execution, and the process-local implementation loses state on
restart.

## Agent communication and state

- The coordinator owns sequencing and the run state; specialists do not mutate
  one another directly.
- Handoffs are explicit method arguments and artifact references in the local
  demo. A live deployment should use authenticated Strands/AgentCore calls.
- Artifacts must be immutable and content-addressed before their reference is
  written to durable state.
- Held-out evaluation data must never enter prompts, training data, or repair
  generation.
- Model-generating agents may propose work; deterministic code verifies
  budgets, replay/evaluation results, and promotion.
- Every agent must use the declared Nemotron model and preserve run context;
  missing or contradictory inputs produce `BLOCKED`, never a guessed value.

## API surface

- `GET /health`
- `POST /api/runs`
- `GET /api/runs/{run_id}`
- `GET /api/runs/{run_id}/experiments`
- `POST /api/runs/{run_id}/step`
- `POST /api/runs/{run_id}/auto`
- `POST /api/runs/{run_id}/cancel`
- `POST /api/demo/reset-environment`
- `GET /api/runs/compare?run_ids=run-001&run_ids=run-002`
- `GET /api/runs/graph?run_ids=run-001&run_ids=run-002`

The list above is the currently implemented local surface. A live deployment
may add an authenticated resumable event-stream endpoint once durable event
storage is connected; it must not be described as available in the local demo.

The hackathon observer renders this same lifecycle as moving role bots:
launcher -> benchmark -> failure analysis -> data curation -> training ->
evaluation -> promotion. Animation is presentation only. A bot may enter a
phase when a corresponding event exists, and a blocked/failed event must stop
that phase visibly; the observer cannot approve a run or manufacture progress.

## Evidence labels

- `LIVE`: produced by a currently verified AWS request.
- `PRIOR_VERIFIED_RUN`: produced by an earlier run with provider IDs and
  hashes.
- `EXPLANATION`: local fixture or simulated output. The current repository
  produces this label only.

## Failure and recovery boundary

The local coordinator is process-local and therefore loses active runs on
restart. Runtime configuration now supports `APP_MODE=aws` and fails startup
when required S3, DynamoDB, and SageMaker settings are missing. The CDK
foundation provisions versioned S3, DynamoDB, ECR, ECS, IAM, VPC, and
CloudWatch resources. The workflow still needs its durable repository and AWS
provider adapters wired before `APP_MODE=aws` can claim live execution.
Missing credentials, unavailable AWS services, invalid artifacts, unknown job
states, or failed deterministic gates must stop the run; the system must not
manufacture progress.

The guarded command path is `live_preflight.py` (read-only), `live_run.py` (one
approved run), and `live_batch.py` (sequential runs with a fresh approval token
between runs). Preflight must pass before any SageMaker job is submitted.
