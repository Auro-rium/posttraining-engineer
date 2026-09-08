# SDD ledger — plan: real-posttraining-runs

## Task graph

| Task | Scope | Depends on | Status |
|---|---|---|---|
| 1 | Durable five-run registry and comparison schema/API | none | completed |
| 2 | Deterministic multi-run improvement gate | none | completed |
| 3 | Graph/report artifact generation | 1 | completed |
| 4 | AWS objective/training orchestration integration | 1-3 | partial: guarded provider boundary and fail-closed orchestrator; no live run |
| 5 | Live demo script, docs, and validation | 4 | blocked pending explicit AWS execution approval |

## Conflict scan

| Pair/task | Finding | Ruling |
|---|---|---|
| 1/2 | Registry consumes promotion results; gate produces them. | Use typed promotion results and keep gate pure. |
| 1/3 | Graph consumes persisted run records. | Renderer accepts immutable comparison DTOs and has no AWS dependencies. |
| 3/4 | Orchestrator emits records consumed by graph. | Integration writes records before rendering and never synthesizes metrics. |
| 4/5 | Demo depends on live orchestration contract. | Add CLI only after provider-backed phases are runnable. |
| 1 | Five-run cap must be atomic. | Enforce with repository compare-and-set, not process memory. |
| 2 | Improvement must be real. | Require paired manifest metrics and verified evidence; reject simulations. |
| 3 | User requested graph output. | Produce JSON chart data plus an SVG artifact for the demo. |

## Rulings

- Ruling: Interpret “5 max” as five sequential top-level run IDs, one candidate per run, each starting from the current champion — this makes improvement history comparable and prevents hidden retries.
- Ruling: Keep the current pushed AWS migration intact and build the new capability on a feature branch — this preserves the pushed checkpoint and makes rollback straightforward.
- Ruling: Do not provision or run live AWS resources without a separate explicit execution request; implementation will add guarded code and tests first.
- Ruling: Telemetry is metadata-only, recursively redacted, immutable after emission, and best-effort; it must never turn a successful run into a failed run.
