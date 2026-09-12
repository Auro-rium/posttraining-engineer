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

End-to-end AWS architecture target (this diagram is not evidence of deployment):
  Bedrock -> agent decisions
  S3 -> immutable trajectories, datasets, checkpoints
  DynamoDB -> durable run state and events
  SageMaker -> QLoRA training
  AgentCore/CloudWatch -> optional hosting and observability
```

Any deployment and live-run actions described here are scoped only to this AWS
Agents for Humans hackathon. The operator-reported SageMaker GPU quota request
`9a3453884e2c4230a6e8bb0004c8cca57FuK8VC5` is `PENDING` as of 2026-09-12;
this documents neither granted quota nor placement capacity. No completed
end-to-end live post-training result is claimed.

## Run history and observation contracts

```text
RunRegistry -> DynamoDB transaction (unique run number 1..5)
           -> ComparisonDTO -> JSON chart data + self-contained SVG
Supervisor run/phase/job/promotion lifecycle -> DurableTelemetryBridge
           -> allow-listed RunEventRecord vocabulary and opaque metadata
           -> TelemetryRecorder
           -> redacted logger/exporter and optional OpenTelemetry spans
```

Telemetry is operational metadata only. Prompts, raw completions, trajectories,
held-out tasks, credentials, and secret-like values are redacted and event
attributes are immutable after emission. Free-form IDs and allow-listed string
values are rejected or redacted unless they match the opaque metadata contract.
The supervisor records `run.started`, `phase.started`, `phase.completed`,
`phase.failed`, and precise terminal run events; phase starts and terminal state
transitions use the repository's atomic transition operation. Durable telemetry
validation or persistence failures stop the supervisor, while failures from the
optional logger/exporter/OpenTelemetry observer do not alter run outcomes.
OpenTelemetry spans include the durable event ID and exact autonomous event
type even when the observer-facing event type is a broader compatibility label.

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

The ordinary local `/api/runs` path uses the same role boundaries but keeps
state in memory and returns `EXPLANATION` fixtures. It is a reproducible
demonstration, not live AWS evidence. The isolated `SERVICE_ROLE=objective`
process has a separate checkpoint-backed benchmark adapter described below;
it is not the same path as the local coordinator workflow. The changelog's
2026-09-06 records cover a scoped AWS identity/Bedrock/S3 smoke and eight
bounded agent calls with `amazon.nova-pro-v1:0`; these historical checks do not
prove the currently pinned Nemotron call, a deployed application, training,
evaluation, or model improvement.

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

The list above is the legacy coordinator surface. A separate `/api/live`
control plane is mounted for AWS mode; it uses the durable run repository and
dispatcher when their configuration and adapters are available. It is not an
SSE endpoint: clients page ordered events with `after` and `limit`.

## Guarded AWS live control plane

The live API is separate from the local `POST /api/runs` demo:

```text
GET  /api/live/readiness                         -> read-only readiness checks
POST /api/live/runs/prepare                      -> preflight, durable PREPARED state, approval packet
POST /api/live/runs/{run_id}/start               -> preflight, consume signed one-run approval, queue
GET  /api/live/runs/{run_id}                     -> durable state snapshot
GET  /api/live/runs/{run_id}/events?after=N      -> ordered persisted lifecycle events
GET  /api/live/runs/{run_id}/experiments         -> persisted experiment records
GET  /api/live/runs/{run_id}/artifacts           -> opaque artifact references
POST /api/live/runs/{run_id}/cancel              -> idempotent cancel request
POST /api/live/runs/{run_id}/safe-stop           -> idempotent safe-stop request
```

Prepare requires an `Idempotency-Key`, passes a fresh read-only preflight, and
stores a bounded approval packet; it does not submit SageMaker work. Start
requires a new `Idempotency-Key`, reruns preflight, validates and consumes the
packet-bound single-run approval, marks the run queued, and schedules the
dispatcher. Startup attempts recovery of incomplete runs. Status, events,
experiments, and artifact routes read from the durable repository; cancel and
safe-stop request state changes, and the supervisor performs bounded cleanup.
In AWS mode the app wires a DynamoDB repository plus Bedrock, objective-worker,
S3, and SageMaker adapters, but any absent/invalid configuration leaves the
live control plane blocked rather than selecting local simulated adapters.

Successful local API/contract tests or a CDK synthesis do not establish that
these resources are deployed, accessible, or that a run completed. Deployment
state and SageMaker capacity must be checked live before authorizing compute.

The train/replay objective benchmark endpoint does not accept a `baseline` or
`held_out` split. The supervisor therefore does not request a baseline score
from that endpoint or relabel training measurements. During candidate
evaluation, SageMaker's isolated sealed evaluator measures both the active
champion checkpoint and candidate checkpoint against the same pinned sealed
manifest and task ordering. The coordinator accepts the pair only when both
checkpoint digests, shared evaluation manifest, task counts and environment
aggregates, paired outcomes digest, and regression evidence digest validate.
The candidate is gated against the champion result from that same report. A
missing champion result or any provenance mismatch blocks promotion; no hidden
task contents are returned to the coordinator.

The hackathon observer renders this same lifecycle as moving role bots:
launcher -> benchmark -> failure analysis -> data curation -> training ->
evaluation -> promotion. Animation is presentation only. A bot may enter a
phase when a corresponding event exists, and a blocked/failed event must stop
that phase visibly; the observer cannot approve a run or manufacture progress.

## Real service-recovery benchmark execution

When `SERVICE_ROLE=objective`, the application selects the isolated,
authenticated objective worker. Its `/v1/benchmark` adapter requires a local
`google/functiongemma-270m-it` snapshot plus `OBJECTIVE_MODEL_REVISION` (an
immutable 40-character commit SHA) and `OBJECTIVE_MODEL_SHA256` (the digest of
the validator's sorted per-file identities). The existing checkpoint staging
validator must accept the complete snapshot before the adapter is constructed.
Transformers loads the checkpoint from that directory with
`local_files_only=True`; the runtime never downloads a mutable Hub reference.
Missing or invalid configuration yields a blocked benchmark response, not a
rule-based or random fallback.

For each requested train/replay task, the model receives only the sanitized
objective, service name, allow-listed tool schemas, and earlier environment
observations. It does not receive verifier rewards, terminal flags, internal
failure-mode fields, or held-out tasks. Its tool calls execute through the
service-recovery engine until the task terminates or reaches its step budget.
The engine then deterministically replays each full action sequence, confirms
the reward and outcome, and the artifact store persists only verified
trajectories; the HTTP response exposes aggregate metrics and opaque artifact
references. Inference, replay, or persistence failures block the benchmark.
The current validation record contains no successful checkpoint-backed
`/v1/benchmark` result or real FunctionGemma trajectory hash. The adapter is
implemented, but a successful contract test with a test double does not prove
that the actual checkpoint loads or that any live benchmark was completed.

## Evidence labels

- `LIVE`: produced by a currently verified AWS request.
- `PRIOR_VERIFIED_RUN`: produced by an earlier run with provider IDs and
  hashes.
- `EXPLANATION`: local fixture or simulated output. The ordinary coordinator
  `/api/runs` demo produces this label; a separately verified objective worker
  is not proof of live training, held-out evaluation, improvement, or promotion.

## Failure and recovery boundary

The legacy local coordinator is process-local and therefore loses active runs
on restart. Runtime configuration supports `APP_MODE=aws`; the `/api/live`
path constructs the durable DynamoDB repository and AWS adapters only when
their required settings are available. The CDK foundation defines versioned
S3, DynamoDB, ECR, ECS, IAM, VPC, and CloudWatch resources but does not prove
they have been deployed. A real `LIVE` result still requires fresh preflight,
an explicit signed approval, provider job IDs, immutable artifact hashes,
held-out evidence, and a deterministic promotion decision.
Missing credentials, unavailable AWS services, invalid artifacts, unknown job
states, or failed deterministic gates must stop the run; the system must not
manufacture progress.

The guarded command path is `live_preflight.py` (read-only), `live_run.py` (one
approved run), and `live_batch.py` (sequential runs with a fresh approval token
between runs). Preflight must pass before any SageMaker job is submitted.
