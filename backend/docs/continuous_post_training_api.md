# Continuous post-training API contract

The isolated router in `app.api.continuous_post_training` is an integration
boundary for a continuous post-training control plane. It accepts trace
records, creates cycles that begin in `pending_approval`, exposes status,
ordered events, and artifact metadata, and handles approve/reject/cancel
commands. Approval moves a cycle to `queued`; it does not submit a training
job. A worker may later use the service/repository protocols to record status
and content-addressed artifact metadata.

## Application integration hook

The current `app.main` is intentionally not modified. An application factory
can attach a durable implementation or the local repository and include the
router explicitly:

```python
from fastapi import FastAPI

from app.api.continuous_post_training import (
    InMemoryPostTrainingRepository,
    install_post_training_api,
)


def create_app() -> FastAPI:
    app = FastAPI()
    repository = InMemoryPostTrainingRepository()  # replace in production
    install_post_training_api(app, repository)
    return app
```

Alternatively, use `app.include_router(router)` and provide
`app.state.post_training_repository`. Tests can override `get_repository` or
`get_service` with FastAPI dependency overrides. Implementations must satisfy
`PostTrainingRepository` and `PostTrainingService`; repository compare-and-set
plus event append should be atomic in a durable adapter.

## Endpoint summary

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/traces` | Store one trace (`201`) |
| POST | `/api/cycles` | Create a pending-approval cycle (`201`) |
| GET | `/api/cycles/{cycle_id}` | Read cycle state |
| GET | `/api/cycles/{cycle_id}/status` | Alias for cycle state |
| GET | `/api/cycles/{cycle_id}/events?after=0` | Read ordered events |
| GET | `/api/cycles/{cycle_id}/artifacts` | Read artifact metadata |
| POST | `/api/cycles/{cycle_id}/approve` | Queue a pending cycle |
| POST | `/api/cycles/{cycle_id}/reject` | Reject with a nonblank reason |
| POST | `/api/cycles/{cycle_id}/cancel` | Cancel an eligible cycle |

Decision bodies may contain `expected_version` for optimistic concurrency and
an optional `reason`; stale versions return `409`. Missing cycles return `404`,
unknown trace references and invalid state transitions return `422`, and
malformed request bodies are rejected by Pydantic with `422`.

The default in-memory repository is process-local and loses data on restart.
The API does not provide SSE, background execution, authentication, or a
durable artifact store. Those concerns belong to the host application and
worker integration.
