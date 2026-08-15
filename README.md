# Autonomous Post-Training Engineer

A backend-only hackathon demonstration of an observable agent team that turns FunctionGemma failures into verified SFT data, bounded QLoRA experiments, and deterministic checkpoint decisions.

This repository intentionally does **not** contain a visualizer or production SaaS features. The supported target is `google/functiongemma-270m-it`; the supported environment is AgentGym WebShop; one run may launch at most two candidates.

## What is implemented

The Python backend provides eight logical roles behind three service modes:

| Service mode | Logical roles | Responsibility |
| --- | --- | --- |
| `coordinator` | Workflow coordinator (not counted as a specialist) | Owns run state, public API, sequencing, budget limits, and events. |
| `research` | Failure Analyst, Research Agent, Data Curator, Training Designer | Grounds a hypothesis with RAG, proposes replay-verifiable data, and selects a bounded QLoRA configuration. |
| `execution` | Benchmark Runner, Training Executor, Evaluation Agent, Champion Manager | Produces trajectories, launches training, evaluates identical task sets, and applies deterministic gates. |

The same immutable container runs all modes through `SERVICE_ROLE`. The deployment selects Firestore, Cloud Storage, Vertex AI, Secret Manager, Cloud Trace, and Cloud Logging. Memory, filesystem, and `EXPLANATION` adapters exist only for credential-free contract tests; their output is not a hackathon result.

Each specialist has an explicit safety contract: Benchmark Runner accepts train-side evidence only; Failure Analyst must ground clusters in trajectory IDs; Research Agent must return a falsifiable hypothesis with canonical RAG citations; Data Curator may propose but never verify repairs; Training Designer must stay inside the QLoRA whitelist; Training Executor may report only Vertex-backed artifacts; Evaluation Agent returns objective evidence without deciding promotion; and Champion Manager explains but cannot override the deterministic gate.

In cloud mode, the coordinator fails fast unless authenticated research/execution A2A destinations are configured. Team services accept only their role's typed operations and call a mandatory external objective-evidence worker for all FunctionGemma, AgentGym, replay, metric, and artifact-producing work.

The API is intentionally small:

- `POST /api/runs`
- `GET /api/runs/{run_id}`
- `GET /api/runs/{run_id}/events`
- `POST /api/runs/{run_id}/step`
- `POST /api/runs/{run_id}/auto`
- `POST /api/runs/{run_id}/cancel`
- `GET /api/runs/{run_id}/experiments`
- `POST /api/demo/verify`
- `GET /health`

`POST /api/runs/{run_id}/auto` uses an in-process background task. The deployment keeps one coordinator instance warm with CPU allocated, but Cloud Run may still restart or replace it; therefore `auto` is best-effort, not a durable workflow engine. Use explicit `/step` calls for the judge path and keep each operation within the configured 55-minute request budget. Firestore preserves completed phase transitions, but the current implementation does not persist an in-flight Vertex polling handle and cannot resume that poll after process loss.

See [Flow.md](Flow.md) for control flow and [Decisions.md](Decisions.md) for the reasons behind the architecture.

## Evidence boundary

Every externally shown artifact has one of three labels:

- `LIVE`: produced by the request currently executing.
- `PRIOR_VERIFIED_RUN`: produced by a real earlier run with hashes and provider job identifiers.
- `EXPLANATION`: fixture or explanatory content that is never presented as measured output.

Gemini may analyze failures and propose hypotheses, repairs, and configurations. It may not grade its own repairs or candidates. Deterministic code enforces replay admission, the two-candidate budget, identical evaluation inputs, and checkpoint promotion. Held-out tasks are excluded from RAG, prompts, repairs, training data, and telemetry.

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

These tests validate schemas, leakage barriers, deterministic orchestration, promotion gates, API behavior, and cloud adapter contracts without calling Gemini, AgentGym, A2A peers, or Google Cloud. Passing them is not evidence that a cloud research run occurred.

Docker Compose is retained only as a three-role image and healthcheck smoke harness:

```bash
docker compose up --build
curl --fail http://localhost:8000/health
curl --fail http://localhost:8001/health
curl --fail http://localhost:8002/health
```

Coordinator, research, and execution listen on ports 8000, 8001, and 8002 respectively. Responses from this harness use local `EXPLANATION` fixtures and must never be shown as model improvement, A2A, or training evidence. The submission flow is the Google Cloud deployment below.

Run repository and infrastructure checks from the repository root:

```bash
cd backend
uv run pytest tests -q
cd ..
backend/.venv/bin/python backend/scripts/check_docs_sync.py
terraform -chdir=infra/terraform fmt -check -recursive
```

## Google Cloud deployment

The Terraform configuration is a deployment skeleton, not proof that a cloud run has occurred. It creates the required APIs, runtime service account, versioned artifact bucket, native Firestore database, empty Secret Manager secret, and three Cloud Run services. Deployed ADK roles use Vertex AI through workload identity (`GOOGLE_GENAI_USE_VERTEXAI=TRUE`); no Gemini API key is stored or injected.

1. Build immutable backend and QLoRA worker images and copy the example variables:

   ```bash
   gcloud builds submit backend --tag REGION-docker.pkg.dev/PROJECT/REPOSITORY/backend:GIT_SHA
   gcloud builds submit TRAINER_CONTEXT --tag REGION-docker.pkg.dev/PROJECT/REPOSITORY/trainer:GIT_SHA
   cp infra/terraform/terraform.tfvars.example infra/terraform/terraform.tfvars
   ```

2. Set `project_id`, `region`, `container_image`, `training_container_image`, `objective_execution_url`, and `rag_corpus_uri`, then apply. The objective URL is a hard prerequisite: it points to a separately deployed, sandboxed FunctionGemma plus AgentGym evidence worker exposing `POST /v1/benchmark`, `/v1/verify-curation`, `/v1/evaluate`, and `/v1/training-evidence`. This Terraform configuration does not provision that worker. Its deployment must grant the runtime service account permission to invoke it.

   The RAG URI is also a hard prerequisite and must reference a readable GCS JSON corpus of allowed `KnowledgeDocument` records. Upload documentation, train-side knowledge, and prior experiment summaries only; held-out and regression scopes fail ingestion. Pin `rag_corpus_sha256` for immutable demo evidence. Terraform does not create or populate this corpus and an external bucket must grant the runtime service account object-read permission.

   ```bash
   terraform -chdir=infra/terraform init
   terraform -chdir=infra/terraform plan -out=deployment.tfplan
   terraform -chdir=infra/terraform apply deployment.tfplan
   ```

3. Terraform creates the research and execution services first, then injects their computed URIs into the coordinator in the same apply. An optional second pass may set `research_service_url` and `execution_service_url` so each team's published A2A Agent Card contains its exact external self URL; coordinator routing does not depend on that pass.

4. If FunctionGemma access requires a Hugging Face token, add it directly to the created Secret Manager resource using stdin or the Cloud Console. Do not place the value in Terraform, `.env`, shell history, logs, or chat. Cloud Run and the Vertex worker receive only `HF_SECRET_ID`; the runtime service account is authorized to read the value.

5. Cloud Run is authenticated by default. Set `allow_unauthenticated = true` only for an intentionally public judge environment; the variable is not a substitute for application authorization.

If the project already has a `(default)` Firestore database, import it into Terraform state rather than attempting to recreate or delete it. Never destroy the database to resolve an ownership conflict.

## Living documentation

All contributors and agents must follow [AGENTS.md](AGENTS.md): read the living documents before work, update `Flow.md` and `Decisions.md` when behavior or choices change, run checks, and append an accurate `ChangeLog.md` entry after verification. Pull-request CI rejects implementation changes without a changelog update.

## Current limitations

- No cloud deployment, Vertex training job, AgentGym run, checkpoint improvement, or model metric is claimed until its real artifact is recorded.
- Credential-free adapters test orchestration contracts only; they are not a replacement for the deployed WebShop, ADK/A2A, and Vertex integration run.
- The sandboxed objective-evidence worker and the GCS RAG corpus are hard deployment prerequisites maintained outside this Terraform configuration.
- A Vertex job handle is not persisted before the coordinator begins polling. A restart during training cannot resume that poll safely, so inspect the provider job before retrying and treat `/auto` as best-effort.
- Authentication, multi-tenancy, billing, continuous training, multiple target models, and a frontend are out of scope.
