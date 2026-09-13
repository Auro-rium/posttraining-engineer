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
Agents for Humans hackathon. A read-only Service Quotas query on 2026-09-12
confirmed an allowance of one `ml.g5.xlarge` training job in `us-east-1`; this
is not a placement reservation. The live preflight remains blocked because the
runtime and pinned artifacts are now deployed, but the real objective benchmark
has not passed its strict FunctionGemma tool-call parser. Readiness is therefore
blocked and no completed end-to-end live post-training result is claimed.

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
In `APP_MODE=aws`, legacy mutation routes (`POST /api/runs`, `/step`, `/auto`,
`/cancel`, and `/api/demo/reset-environment`) return `410 Gone`; only the
approval-gated `/api/live` controls can change live-run state. Read-only
comparison endpoints remain available and never turn local explanatory records
into live evidence.

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
dispatcher. Startup starts a nonblocking durable-recovery loop that scans
incomplete runs immediately and every 30 seconds thereafter; dispatcher leases
prevent the startup scan, periodic scan, and request-triggered dispatch from
duplicating the same run. Shutdown cancels the recovery task cleanly. Status,
events, experiments, and artifact routes read from the durable repository; cancel and
safe-stop request state changes, and the supervisor performs bounded cleanup.
Dispatcher shutdown also cancels and awaits the supervisor child before its
lease is released, so the next recovery owner cannot overlap that in-process
work with a recovered run.
In AWS mode the app wires a DynamoDB repository plus Bedrock, objective-worker,
S3, and SageMaker adapters, but any absent/invalid configuration leaves the
live control plane blocked rather than selecting local simulated adapters.
The runtime task role grants `bedrock:GetFoundationModel` and
`bedrock:InvokeModel` only on the configured regional foundation-model ARN.
Runtime synthesis requires separate validated CDK context values for the
SageMaker training-job and processing-job GPU quota IDs; coordinator preflight
queries both and blocks readiness if either quota is missing, unverifiable, or
below the requested instance count. Quota allowance is not placement capacity.

The coordinator task remains private in the VPC, behind an internet-facing
application load balancer on port 443. CDK requires an issued ACM certificate
in the stack region, a matching public hostname and Route 53 zone, and explicit
`coordinator_ingress_cidrs` IPv4 networks (/16 through /32); synthesis fails
closed when any contract is absent. The stack outputs an `https://`
`CoordinatorUrl`.
Approval TTL defaults to 86,400 seconds (24 hours), enough for five sequential
training/evaluation pairs with each provider job bounded at two hours, plus
bounded polling; configuration rejects
a window shorter than the approved experiment scope. The API is restricted to
the configured operator CIDR even though the ALB is internet-facing; the
approval token travels over TLS. An in-stack objective worker separately
requires its private TLS certificate/SAN contract, while an external worker
must be reachable at an authenticated HTTPS URL.
Coordinator IAM grants `sagemaker:ListTags` only for tagged processing/training
job ARNs so restart reconciliation can verify deterministic job fingerprints.

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

For a completed SageMaker Processing evaluation, the provider selects exactly
one output named `evaluation` and returns its S3 output prefix. The artifact
store lists that prefix (including all pages), requires exactly one report
object named `evaluation.json` or `evaluation.tar.gz`, and rejects missing or
ambiguous matches. It resolves the source object's S3 `VersionId`, downloads
that exact version, computes SHA-256 from the bytes, and copies the verified
content to the versioned, content-addressed artifact store. The evaluation
reader then validates report structure and provenance before accepting metrics.
No provider prefix itself is treated as an S3 object or evaluation evidence.

The evaluator's Processing inputs use fixed local mounts: `base_model` at
`/opt/ml/processing/input/base_model`, `candidate` at
`/opt/ml/processing/input/candidate`, `champion` at
`/opt/ml/processing/input/champion`, and `sealed` at
`/opt/ml/processing/input/sealed`. Each `SM_CHANNEL_*` environment value is
set to the matching Processing `LocalPath`; a conflicting caller-supplied path
is rejected. `SM_OUTPUT_DATA_DIR` and the Processing output `LocalPath` both
resolve to `/opt/ml/processing/output`.

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

When CDK creates this worker inside the stack, its Fargate task has a fixed
baseline of 2 vCPU and 4 GiB memory for the Python/Torch/Transformers runtime
and FunctionGemma model loading. This applies only to the internal objective
task; the coordinator remains at 1 vCPU and 2 GiB, and an externally configured
`objective_worker_url` does not create or resize an objective task. The CDK
contract verifies the synthesized allocation, not successful image startup,
model loading, or a live benchmark.

For the internal worker, the container entrypoint first downloads the exact
`OBJECTIVE_BASE_MODEL_URI` S3 object version into
`OBJECTIVE_MODEL_CHECKPOINT_DIR`, verifies its bundle SHA-256, safely extracts
the archive, checks the pinned FunctionGemma revision and model files, then
sets `OBJECTIVE_MODEL_SHA256` to the validated snapshot digest before replacing
itself with Uvicorn. CDK supplies the immutable versioned URI, bundle digest,
revision, and checkpoint destination; the objective task role has scoped
version-read and KMS permissions. Any missing input, S3/KMS error, digest or
revision mismatch, unsafe archive, or extraction failure exits before the
health endpoint starts. Coordinator containers skip model bootstrap. This is
an implemented startup contract; it does not prove an AWS task actually
downloaded or loaded a checkpoint.

For each requested train/replay task, the model receives only the sanitized
objective, service name, allow-listed tool schemas, and earlier environment
observations. It does not receive verifier rewards, terminal flags, internal
failure-mode fields, or held-out tasks. Its tool calls execute through the
service-recovery engine until the task terminates or reaches its step budget.
The engine then deterministically replays each full action sequence, confirms
the reward and outcome, and the artifact store persists replay-verified
trajectories; the HTTP response exposes aggregate metrics and opaque artifact
references. A replay-verified failure remains benchmark evidence, not an SFT
target. The DataCuratorAgent may return a strict action-sequence proposal bound
to a coordinator-supplied train/replay failure reference; that proposal is
untrusted and is not a trajectory or evidence. Authenticated
`/v1/replay-corrections` resolves the referenced artifact, checks its
verifier-confirmed failed outcome and exact task/split, then deterministically
replays the proposed actions. A failed proposal returns `REJECTED` and is not
persisted. Only a passing replay is persisted with
`repaired_from_trajectory_id` in its content-addressed identity and returned as
a verified trajectory reference. `/v1/verify-curation` replays selected
originals and accepted repairs, omits failed originals from dataset rows, and
admits only successful verifier outcomes. Dataset manifests preserve repair
lineage, and both artifact stores recheck successful verifier outcomes and the
failed parent before persistence. `build_dataset` itself still rejects failed
trajectories. Correction and curation requests accept train or replay only;
hidden and validation data cannot be proposed or replayed through the objective
service. If no successful original or verified repair is available, curation
fails closed and creates no dataset. Inference, replay, or persistence failures
block the benchmark.

Each authenticated `/v1/benchmark` request receives a server-generated
correlation ID, also returned in `X-Objective-Correlation-ID`. The worker logs
only that ID, an allow-listed stage, elapsed milliseconds, pinned checkpoint
revision, optional process RSS, outcome, and (on failure) exception class. It
never returns exception details or logs prompts, completions, credentials,
authorization headers, or trajectory/task contents. Instrumented stages are
`CHECKPOINT_RESOLVE`, `PROCESSOR_LOAD`, `MODEL_LOAD`, `PROMPT_RENDER`,
`MODEL_GENERATE`, `MODEL_DECODE`, `FUNCTION_PARSE`, `ENVIRONMENT_STEP`,
`TRAJECTORY_VERIFY`, `S3_PERSIST`, and `BENCHMARK_COMPLETE`.

Authenticated `/v1/readiness` separates `configuration_ready`,
`checkpoint_ready`, `model_load_ready`, `generation_ready`,
`artifact_store_ready`, and `execution_ready`. Model-load and generation
attestations are process-local: they remain false until that worker has loaded
the local checkpoint and generated a parseable allow-listed tool call.
`artifact_store_ready` checks the configured store interface; it is not proof
of an S3 write. Only a successful benchmark response with a version-pinned,
non-empty trajectory/report artifact proves the end-to-end objective path.
The coordinator's `/health` is intentionally shallow and reports endpoint
configuration and build provenance separately from dependency readiness.

Live AWS observation on 2026-09-13: the deployed coordinator `/health` returned
HTTP 200 but still reported the stale `objective_worker: not_configured` value
and omitted build provenance. `/api/live/readiness` returned `READY` based on
configuration and quota checks, but a one-episode authenticated objective
request had returned HTTP 503 before the stage-telemetry build was deployed.
No trajectory/report was produced and no SageMaker training or Processing job
was submitted. The local stage-telemetry/readiness changes are not live AWS
evidence until committed, published, deployed, and followed by a successful
objective smoke.

As of 2026-09-13, the pinned `google/functiongemma-270m-it` revision
`39eccb091651513a5dfb56892d3714c1b5b8276c` has been staged in the versioned,
KMS-encrypted AWS artifact bucket. Its deterministic bundle SHA-256 is
`70436508f4a6908c9b455afa57b0690f42f0f0d1df2601a54aefded914afcb79` and its
S3 `VersionId` is `L86c4JC3X7Zw8kUiOAZlVjOpvXtKtpLF`. The sealed evaluator
bundle is also staged: suite `AgentGym/AgentEval`, version `agent-eval-v1`,
seed 7, 20 task IDs, manifest SHA-256
`019b08d0ef35f192953cc6a1babdf41ea07044a2e491eb7cbf7e236ab939f39a`. Those
IDs resolve to tasks in the repository's deterministic
`in-repo-service-recovery-v1` engine; they are not downloaded upstream
AgentGym assets. These S3 writes do not prove a runtime deployment, checkpoint
load, successful `/v1/benchmark`, or real FunctionGemma trajectory hash. The
adapter is implemented, but contract tests with test doubles do not establish
live inference or a completed benchmark.

After that historical observation, commit `69560c01b5254901393862bb1e04298a8a405cfa`
was built in AWS CodeBuild and deployed. The public `/health` endpoint reported
that exact backend commit and ECR digest. A real one-episode request then passed
checkpoint resolution, processor/model load, prompt rendering, generation, and
decoding, but failed at `FUNCTION_PARSE` (correlation
`a361258cd868494db89c1353d4443c99`). The generated content was not logged or
returned; no environment action, verified trajectory, or S3 report resulted.
Authenticated objective readiness remains `BLOCKED`, specifically because a
valid tool call has not yet been generated. Source now includes FunctionGemma's
documented tool-calling activation instruction plus a regression test; that fix
still needs to be built, deployed, and proven by another real S3-backed episode.
No SageMaker training or Processing job has been submitted.

## Trainer and evaluator image smoke contract

Both SageMaker worker images include a separate, explicit smoke entrypoint and
are built for `linux/amd64`. The trainer smoke accepts only a mounted local
FunctionGemma directory, requires CUDA and bitsandbytes, disables Hub access,
loads the model in 4-bit, and performs one LoRA optimizer step. It writes a
throwaway adapter solely so the evaluator image can independently load the
same local base plus adapter and execute one inference forward pass. The
evaluator smoke uses no sealed tasks and writes no evaluation report. The
one-step smoke adapter is not a training result, has no approved lineage
manifest, and cannot be promoted. A successful image smoke does not prove a
SageMaker job or live evaluation succeeded.

Worker images are published through `backend/aws-image-buildspec.yml` in an
AWS CodeBuild environment, not by downloading the ML dependency stack on the
developer workstation. The source archive must contain only the Docker build
inputs and exclude `.env` files, virtual environments, local model caches, and
training/evaluation data. CodeBuild builds `linux/amd64`, pushes uniquely
tagged images to the three bootstrap ECR repositories, then resolves and
prints each immutable ECR digest. The runtime stack consumes those digests;
mutable tags are not deployment inputs. This build path does not run a GPU
smoke or submit SageMaker jobs, and image digests alone do not establish model
load, training, evaluation, or promotion evidence.

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

## Durable dispatcher lease ownership

A dispatcher may claim a run only when no unexpired lease exists, including
when the existing lease owner string matches its own process identity. Lease
renewal is a separate compare-and-swap operation that requires the current
owner and an unexpired lease. This prevents overlapping recovery and HTTP-start
dispatch scans in one coordinator process from treating a shared owner name as
proof that both workers own the same run. Expired leases remain claimable for
restart recovery; SageMaker intents and deterministic provider fingerprints
remain the provider-side duplicate-submission defense. Local regression tests
cover same-owner reclaims and stale concurrent dispatcher scans; these tests do
not prove AWS DynamoDB/SageMaker behavior in a deployed stack.
