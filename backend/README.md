# Backend package

The backend is the coordinator and evidence boundary for the autonomous
FunctionGemma post-training demo. It runs eight specialist roles with NVIDIA
Nemotron Super 3 120B (`nvidia.nemotron-super-3-120b`) as their fixed Bedrock
reasoning model. FunctionGemma (`google/functiongemma-270m-it`) is the separate
target checkpoint being improved by SageMaker; Nemotron is not fine-tuned by
this workflow.

## Local contract checks

From this directory:

```bash
uv sync --frozen --extra cloud --extra dev
uv run pytest -q
uv run ruff check app tests scripts
uv run mypy app
```

Pytest discovers both the root compatibility tests and the full `tests/`
contract suite. These checks are credential-free and do not prove a live AWS
run.

## Live commands

The live path is guarded and fail-closed:

```bash
uv run --extra cloud python scripts/live_preflight.py
uv run --extra cloud python scripts/live_run.py --approval-token TOKEN
uv run --extra cloud python scripts/live_batch.py
```

`live_preflight.py` is read-only. `live_run.py` executes one approved run and
`live_batch.py` pauses between sequential runs, up to five total and within
the hard budget. Missing checkpoint revisions, objective-worker artifacts,
SageMaker images/roles, provider IDs, or verified measurements produce
`BLOCKED`/`FAILED`; no fallback metrics are generated.

The HTTP live control plane is separate from the process-local `/api/runs`
demo. In AWS mode, `POST /api/live/runs/prepare` runs preflight and stores a
durable prepared run plus its immutable approval packet; `POST
/api/live/runs/{run_id}/start` rechecks preflight and consumes the signed,
single-run approval before queuing work. These mutating requests require an
`Idempotency-Key`. Status, ordered events, experiments, and artifact references
are read from the durable repository; cancel and safe-stop requests are
idempotent. This API wiring and its local contract tests do not prove an AWS
deployment or a completed post-training run.

### Objective worker and FunctionGemma checkpoint

`SERVICE_ROLE=objective` exposes the isolated `/v1/benchmark`,
`/v1/replay-corrections`, and `/v1/verify-curation` endpoints. It
requires `OBJECTIVE_AUTH_TOKEN` and `S3_ARTIFACT_BUCKET`; benchmark execution
also requires all three values below:

- `OBJECTIVE_MODEL_CHECKPOINT_DIR`: a complete local `google/functiongemma-270m-it` snapshot.
- `OBJECTIVE_MODEL_REVISION`: the immutable lowercase 40-character Hugging Face commit SHA for that snapshot.
- `OBJECTIVE_MODEL_SHA256`: the expected digest of the validator's sorted per-file identities.

The staging validator rejects incomplete, mutable-cache, gated, or malformed
checkpoints. Runtime loading uses `local_files_only=True`; it never downloads
or silently substitutes a remote model. The adapter accepts train/replay tasks
only, replays model actions through the deterministic service-recovery
verifier, and persists only verifier-confirmed trajectories. Missing
configuration, inference errors, replay failures, or artifact-store failures
block the request. The code and contract tests do not establish that a complete
checkpoint is available or that a real FunctionGemma benchmark has succeeded.

The backend image entrypoint detects `SERVICE_ROLE=objective` and, before
starting Uvicorn, runs `scripts/bootstrap_objective_checkpoint.py`. It fetches
the exact `OBJECTIVE_BASE_MODEL_URI` S3 `versionId`, verifies
`OBJECTIVE_BASE_MODEL_SHA256`, safely extracts and validates the pinned
FunctionGemma revision, then supplies the extracted snapshot digest as
`OBJECTIVE_MODEL_SHA256`. A bootstrap error is fatal and the objective service
does not become healthy. Coordinator startup skips this model download. The
internal CDK task supplies the required URI, revision, bundle digest, and local
directory and grants only artifact-prefix access plus the artifact KMS key; a
local bootstrap test is not evidence of an AWS task download or model load.
The curator may submit structured actions bound to a stored verifier-confirmed
failure. The objective worker persists only successful deterministic replays;
failed proposals are discarded, and the failed original actions are omitted
from SFT rows. Hidden and validation splits remain inaccessible to this API.

Before the first job, configure `CHECKPOINT_SHA256`,
`SAGEMAKER_GPU_QUOTA_CODE`, `SAGEMAKER_PROCESSING_GPU_QUOTA_CODE`,
`GPU_INSTANCE_ALLOWLIST`, and a secret in `LIVE_APPROVAL_SECRET`. Preflight
checks both training-job and processing-job GPU quotas; evaluation uses
SageMaker Processing, so training quota alone is insufficient. The runtime CDK
requires both quota IDs as validated context (`sagemaker_gpu_quota_code` and
`sagemaker_processing_gpu_quota_code`) and injects them into the coordinator.
The scripts print a metadata-only approval packet when no token is supplied.
An operator signs that packet with
`issue_approval_token(packet, secret)` and reruns with the resulting token.
For the included signer: `LIVE_APPROVAL_SECRET=... uv run python
scripts/issue_approval_token.py packet.json`.

The bounded live timing defaults match the five-experiment ceiling:
`LIVE_APPROVAL_TTL_SECONDS=86400` (24 hours), with each training and evaluation
job independently capped by `MAX_TRAINING_TIME_MIN=120`. Configuration rejects
an approval window shorter than `MAX_EXPERIMENTS * 2 * MAX_TRAINING_TIME_MIN`
minutes; an explicitly requested expiry must also cover its approved run size.
Provider status is checked every 30 seconds, including one terminal check after
the SageMaker runtime bound. Objective-worker requests default to
`OBJECTIVE_WORKER_TIMEOUT_SECONDS=600` and cannot be configured above 10 minutes.

Service Quotas confirms account allowance for both job types; it is not a
placement reservation, so SageMaker remains the final capacity decision after
approval. Only the current run's training/processing jobs are stopped during
cleanup; S3,
DynamoDB, IAM, ECR, and networking resources are reused and never deleted by
the controller.

The complete variable template is the repository-level `.env.example`. The
live path needs an authenticated objective worker (an external HTTPS URL or the
CDK-managed internal worker), `HF_REPO_ID`, an immutable `HF_REVISION`, a
version-pinned `CHECKPOINT_S3_URI` with its bundle SHA-256, and a staged sealed
`EVALUATION_INPUT_S3_URI`. Autonomous runs create a content-addressed training
dataset per experiment, so `TRAINING_INPUT_S3_URI` is optional and is not a
live-run readiness gate. The Hugging Face token, if the source requires one,
is used only to obtain the checkpoint for staging; workers load the verified
S3 bundle locally and have no Hugging Face fallback. `SAGEMAKER_INSTANCE_TYPE`
must be present in `GPU_INSTANCE_ALLOWLIST`; the Service Quotas check validates
account allowance, not placement capacity.

For Bedrock authentication, use the configured AWS IAM/SigV4 credential chain
with `nvidia.nemotron-super-3-120b` and leave `AWS_BEARER_TOKEN_BEDROCK`
unset. If that variable is set, the AWS SDK may select it instead of IAM; set
it only when a valid Bedrock API key is intentionally being used. A successful
STS check alone is not proof that Bedrock model invocation is authorized.

Any deployment and live-run actions described by this backend are scoped only
to the AWS Agents for Humans hackathon. A read-only Service Quotas query on
2026-09-12 confirmed allowance `1.0` for `ml.g5.xlarge` training jobs in
`us-east-1`; this is not a capacity reservation. The same day's local
`scripts/live_preflight.py` report is `BLOCKED` because runtime resource and
artifact configuration is absent. AWS inventory found no matching post-training
CloudFormation stack, no DynamoDB run table, no dedicated artifact bucket,
trainer/evaluator ECR repositories, coordinator certificate, or Route 53 zone.
No live SageMaker training, held-out evaluation, promotion, or completed
autonomous run is claimed.

All agent prompts are versioned contracts. They include explicit schemas,
evidence rules, sealed held-out-data boundaries, and bounded creative latitude.
Prompt hashes and the Nemotron model ID are safe manifest/telemetry metadata;
raw prompts, completions, trajectories, credentials, and held-out tasks are
never logged or exposed to the browser observer.

See the repository [README.md](../README.md), [Flow.md](../Flow.md), and
[Decisions.md](../Decisions.md) for the complete architecture and evidence
boundary.
