# SDD ledger — plan: continuous-posttraining implementation

## Task graph

| Task | Scope | Depends on | Shared interfaces | Status |
|---|---|---|---|---|
| 1 | AWS provider adapters | none | ArtifactRef, EvidenceEvent | running |
| 2 | Continuous trace ingestion and trigger | none | TraceEvent, OptimizationCycle | running |
| 3 | Deterministic post-training gates/state | none | OptimizationCycle, PromotionDecision | running |
| 4 | FastAPI continuous API router | 1-3 contracts | repository/service protocols | running |
| 5 | Root integration and validation | 1-4 | app.main | queued |

## Conflict scan

| Pair/task | Finding | Ruling |
|---|---|---|
| 1/2 | Both define artifact/event boundaries but own separate packages. | Provider types must be structural and import-safe; root integration resolves adapters. |
| 1/3 | Provider outputs feed deterministic gates. | Providers return typed evidence/artifact refs; gates never call providers. |
| 2/4 | Ingestion triggers cycles consumed by API. | API depends on protocols only; in-memory implementations remain testable. |
| 3/4 | API exposes approval/rejection state machine. | API delegates all transitions to deterministic domain service. |
| 1 | AWS credentials may be absent locally. | All AWS clients are dependency-injected and lazy; local tests must not call AWS. |
| 2 | Existing main.py uses active_runs. | No worker may edit main.py; root integration replaces it after review. |
| 3 | Existing agents contain random simulation. | New deterministic package is authoritative for gates; live adapter cleanup is root integration work. |

## Rulings

- Preserve all pre-existing dirty worktree changes; do not reset or overwrite them.
- Product remains continuous agentic post-training; service recovery is only a benchmark fixture.
- Fully-live AWS is the target boundary, with honest local fallback for tests.
