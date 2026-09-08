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

Before the first job, configure `CHECKPOINT_SHA256`,
`SAGEMAKER_GPU_QUOTA_CODE`, `GPU_INSTANCE_ALLOWLIST`, and a secret in
`LIVE_APPROVAL_SECRET`. The scripts print a metadata-only approval packet when
no token is supplied. An operator signs that packet with
`issue_approval_token(packet, secret)` and reruns with the resulting token.
For the included signer: `LIVE_APPROVAL_SECRET=... uv run python
scripts/issue_approval_token.py packet.json`.
Service Quotas confirms account allowance; it is not a placement reservation,
so SageMaker remains the final capacity decision after approval. Only the
current run's training/processing jobs are stopped during cleanup; S3,
DynamoDB, IAM, ECR, and networking resources are reused and never deleted by
the controller.

The complete variable template is the repository-level `.env.example`. The
live path additionally requires `OBJECTIVE_WORKER_URL`, `HF_REPO_ID`, an
immutable `HF_REVISION`, `TRAINING_INPUT_S3_URI`,
`EVALUATION_INPUT_S3_URI`, `CHECKPOINT_S3_URI`, and a matching lowercase
`CHECKPOINT_SHA256`. The checkpoint must already exist in versioned S3; the
controller does not download or stage a Hugging Face checkpoint automatically.
`SAGEMAKER_INSTANCE_TYPE` must be present in `GPU_INSTANCE_ALLOWLIST`, and
the quota check confirms account allowance only—it is not a placement
reservation. SageMaker makes the final capacity decision.

For Bedrock authentication, use the configured AWS IAM/SigV4 credential chain
with `nvidia.nemotron-super-3-120b` and leave `AWS_BEARER_TOKEN_BEDROCK`
unset. If that variable is set, the AWS SDK may select it instead of IAM; set
it only when a valid Bedrock API key is intentionally being used. A successful
STS check alone is not proof that Bedrock model invocation is authorized.

All agent prompts are versioned contracts. They include explicit schemas,
evidence rules, sealed held-out-data boundaries, and bounded creative latitude.
Prompt hashes and the Nemotron model ID are safe manifest/telemetry metadata;
raw prompts, completions, trajectories, credentials, and held-out tasks are
never logged or exposed to the browser observer.

See the repository [README.md](../README.md), [Flow.md](../Flow.md), and
[Decisions.md](../Decisions.md) for the complete architecture and evidence
boundary.
