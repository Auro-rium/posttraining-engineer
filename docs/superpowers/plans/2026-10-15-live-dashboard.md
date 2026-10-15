# Live Mission-Control Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream a `posttrain-agent` run live to a neo-futuristic Next.js dashboard on Vercel, from a Dockerized read-only SSE API on the operator's machine.

**Architecture:** The agent appends redacted semantic events to an append-only SQLite journal (`state/events.sqlite`). A FastAPI service in Docker tails the journal over SSE (resumable via `Last-Event-ID`) and is exposed to Vercel through a Cloudflare Tunnel. The frontend folds events with a pure reducer shared by golden fixtures with the Python snapshot code.

**Tech Stack:** Python 3.12, FastAPI + uvicorn, SQLite (WAL), Docker Compose + cloudflared; Next.js (App Router) + TypeScript, Zustand, Vitest, Playwright.

**Spec:** `docs/superpowers/specs/2026-10-15-live-dashboard-design.md`

## Global Constraints

- Journal retention: keep the most recent **100,000** rows; `seq` is a global monotonic `INTEGER PRIMARY KEY AUTOINCREMENT`.
- SSE: heartbeat comment every **15 s**; journal poll interval **250 ms**; at most **20** concurrent SSE connections (excess get **429**); frame is `id: <seq>` / `event: <type>` / `data: <json>`; `Last-Event-ID` beats `?since=`.
- `telemetry.snapshot` is emitted at most every **5 s** while active; `job.log.message` truncated to **300** chars.
- Events never contain prompts, completions, message text, tool arguments, held-out contents or secrets; only allow-listed fields (spec 4.3) pass.
- Auth: static read token in `?token=`, constant-time compare; CORS allows only `ALLOWED_ORIGIN`; `POST /v1/runs/{id}/decision` returns **501**.
- A journal failure must never fail or slow a run (best-effort writes, log once).
- UI honesty: `EXPLANATION` evidence is badged **SIMULATED** (amber); sample data is labelled **REPLAY · SAMPLE DATA** and is never mixed with live events; a dropped live connection keeps the last live state labelled **RECONNECTING**.
- Motion off under `prefers-reduced-motion`. Colors: near-black bg, phosphor green primary, amber money/approval, cyan info, red errors; monospace.
- Python code follows the existing package style (`src/posttrain_agent/`, `from __future__ import annotations`, ruff line length 100). Frontend lives in `web/`.
- Frontend tasks (7-11) are built by **Opus 5.5**; backend tasks (1-6) by any implementer.

## Review Focus

1. Client reconnects mid-stream and gets duplicate or missing events (resume exactly after the last `seq`). Test in Task 5.
2. Reader polls while the CLI writes (SQLite lock); the API must neither error nor block the run. Test in Task 5.
3. Event data carries extra or huge unknown fields (dropped; sizes bounded). Test in Task 1.
4. Two runs interleave in one journal (a `run` filter returns only its own events; snapshots do not bleed). Test in Task 5.
5. API started with no `READ_TOKEN` configured (must refuse to start, not run open). Test in Task 5.

## File Structure

- Create `src/posttrain_agent/events.py` (journal + allow-list), `fold.py` (snapshot reducer), `api.py` (FastAPI app), `journal_handler.py` (callback emitter).
- Modify `telemetry.py` (nothing; handler reused), `concurrency.py` (agent events), `replay.py` (run/approval/checkpoint events), `domain.py` (score/gate/budget/job emits), `cli.py` (attach handler), `pyproject.toml` (`api` extra).
- Create `contract/golden.json` (shared reducer fixtures), `scripts/make_sample_journal.py`, `Dockerfile`, `docker-compose.yml`.
- Create `web/` (Next.js app): `lib/events.ts`, `lib/reducer.ts`, `lib/connection.ts`, `components/*`, `public/sample-run.jsonl`.
- Tests: `tests/test_events.py`, `test_fold.py`, `test_journal_wiring.py`, `test_api.py`; `web/**/*.test.ts`, `web/e2e/smoke.spec.ts`.

---

### Task 1: Event journal and redaction allow-list

**Files:** Create `src/posttrain_agent/events.py`; Test `tests/test_events.py`

**Interfaces:**
- Produces: `EVENT_TYPES: frozenset[str]` (the 16 types in spec 4.3); `sanitize(type: str, data: dict) -> dict | None` (None for unknown type; drops non-allow-listed keys; truncates strings >300 chars; drops non-JSON-scalar/list values deeper than 2 levels); `@dataclass Event(seq:int, ts:float, run:str, type:str, data:dict)`; `class Journal(path: Path | str)` with `append(run: str, type: str, data: dict) -> int | None` (best-effort, never raises, returns seq), `read(run: str | None = None, since: int = 0, limit: int = 500) -> list[Event]` (ascending `seq`), `runs() -> list[dict]` (`run`, `started`, `last_seq`, `status`), `last_seq() -> int`.

- [ ] **Step 1: Write failing tests** `test_seq_is_monotonic_and_read_resumes_after_since`, `test_unknown_type_is_rejected`, `test_non_allowlisted_fields_dropped_and_long_strings_truncated`, `test_sentinel_strings_in_every_field_never_reach_the_journal` (call `append` with a prompt-like key `"content": "SECRET"`, tool `"args"`, nested dicts; read the raw sqlite file bytes and assert `SECRET` absent), `test_prune_keeps_latest_100000_rows` (use a small `max_rows` constructor kwarg defaulting to 100_000), `test_append_never_raises_when_db_is_unwritable` (read-only path returns `None`).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_events.py -q` — Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `events.py` with the signatures above. SQLite in WAL mode, one connection per call (`timeout=1`), allow-list as a `dict[str, frozenset[str]]` keyed by type exactly per spec 4.3, prune in `append` every 1,000 writes.
- [ ] **Step 4: Run tests** — Expected: all PASS.
- [ ] **Step 5: Commit** `git add src/posttrain_agent/events.py tests/test_events.py && git commit -m "feat: redacting append-only event journal"`

### Task 2: Snapshot reducer and shared golden fixtures

**Files:** Create `src/posttrain_agent/fold.py`, `contract/golden.json`; Test `tests/test_fold.py`

**Interfaces:**
- Consumes: `Event` from Task 1.
- Produces: `empty_state() -> dict`; `fold(state: dict, ev: Event | dict) -> dict` (pure, never raises, ignores unknown types); `snapshot(events: list[Event]) -> dict`. State keys (the contract the TS reducer must reproduce): `status`, `attempt`, `backend`, `experiment`, `agents` (role → `{active, slot, ok}`), `active_slots` (list), `log` (last 200 `{seq, ts, type, summary}`), `scores` (list), `gate`, `budget`, `job`, `job_log` (last 50), `approval` (`null` or pending object), `telemetry`, `checkpoints` (list), `last_seq`.
- `contract/golden.json`: list of `{name, events: [...], expected: {...state subset...}}` covering: a full promote run, a reject run, two parallel agents then one ends, approval pending then resolved, fallback model call, unknown event type ignored, malformed event ignored.

- [ ] **Step 1: Write failing tests** `test_golden_fixtures` (parametrized over `contract/golden.json`, asserts every key in `expected` equals the folded state), `test_unknown_and_malformed_events_do_not_raise`, `test_snapshot_equals_incremental_fold`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_fold.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement** `fold.py` and author `contract/golden.json` (hand-written minimal events; the expected states are the spec of the reducer).
- [ ] **Step 4: Run tests** — Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: snapshot reducer with shared golden fixtures"`

### Task 3: JournalHandler and subagent events

**Files:** Create `src/posttrain_agent/journal_handler.py`; Modify `concurrency.py`, `cli.py`; Test `tests/test_journal_wiring.py`

**Interfaces:**
- Consumes: `Journal.append`.
- Produces: `JournalHandler(journal: Journal, run: str, primary: str)` (a `BaseCallbackHandler`, `run_inline = True`) emitting `model.call` (with `fallback = name != primary`), `tool.call`, and `telemetry.snapshot` (≤ every 5 s, derived from recent calls). `SubagentConcurrencyMiddleware(limit: int = 2, journal: Journal | None = None, run: str = "")` emits `agent.start`/`agent.end` with `role` = `args["subagent_type"]` and `slot` (1 or 2, lowest free).

- [ ] **Step 1: Write failing tests** `test_model_and_tool_calls_are_journaled_without_prompt_text`, `test_fallback_flag_set_when_model_differs_from_primary`, `test_snapshot_event_throttled_to_5s` (inject a clock), `test_two_parallel_subagents_get_slots_1_and_2_then_reuse` (drive the middleware with fake requests as in `tests/test_parallel.py`).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_journal_wiring.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement** the handler and the middleware changes; in `cli._main` create `Journal(settings.state_dir / "events.sqlite")` and pass the handler in `cbs` plus the journal to `build_agent` (new optional `journal` kwarg forwarded to the middleware).
- [ ] **Step 4: Run** the new tests and `tests/test_parallel.py` — Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: journal model/tool/subagent events"`

### Task 4: Run-controller and domain events

**Files:** Modify `replay.py`, `domain.py`, `agent.py`; Test extend `tests/test_journal_wiring.py`

**Interfaces:**
- Consumes: `Journal`. Produces: `run_with_recovery(..., journal: Journal | None = None)` and `decide(..., journal=None)` emit `run.started`, `run.status`, `checkpoint`, `approval.pending` (from `RunReport.pending`, including `estimate_usd` parsed from the same estimate used in `approval_description`, and `cli_hint`), `approval.resolved`. `make_tools(backend, settings, suite, journal=None, run="")` emits `score` (benchmark/evaluate), `gate.verdict`, `budget.update` (after estimate/launch), `job.update`/`job.log` (status/events tools).

- [ ] **Step 1: Write failing tests** `test_full_fake_run_emits_expected_event_sequence` (reuse `tests/test_approval.py` fixtures: pause for approval, then approve; assert order `run.started`, `agent.start`, `approval.pending`, `run.status(AWAITING_APPROVAL)`, `approval.resolved`, `job.update`, `run.status(COMPLETED)`), `test_score_event_carries_evidence_label`, `test_journal_failure_does_not_fail_the_run` (journal pointing at an unwritable path).
- [ ] **Step 2: Run** the new tests — Expected: FAIL.
- [ ] **Step 3: Implement** the optional `journal` plumbing; keep every emit call best-effort and outside the run's critical path.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q` — Expected: all PASS (existing 60 + new).
- [ ] **Step 5: Commit** `git commit -m "feat: emit run, approval, score, budget and job events"`

### Task 5: FastAPI SSE service

**Files:** Create `src/posttrain_agent/api.py`; Modify `pyproject.toml` (extra `api = ["fastapi>=0.115", "uvicorn>=0.30"]`, script `posttrain-api`); Test `tests/test_api.py`

**Interfaces:**
- Consumes: `Journal`, `fold.snapshot`.
- Produces: `create_app(state_dir: Path, read_token: str, allowed_origin: str, replay_file: Path | None = None, max_streams: int = 20, poll_s: float = 0.25, heartbeat_s: float = 15.0) -> FastAPI`; routes `GET /v1/stream`, `GET /v1/runs`, `GET /v1/runs/{id}/snapshot`, `GET /v1/health`, `POST /v1/runs/{id}/decision` (501). `main()` reads env `READ_TOKEN`, `ALLOWED_ORIGIN`, `STATE_DIR`, `REPLAY_FILE` and **exits non-zero if `READ_TOKEN` is empty**. Replay mode serves a journal `.jsonl` through the same routes, paced by timestamps with optional `?speed=`.

- [ ] **Step 1: Write failing tests** (httpx `ASGITransport`): `test_missing_or_wrong_token_is_401`, `test_cors_only_allows_configured_origin`, `test_stream_emits_frames_with_seq_ids_and_event_names`, `test_resume_with_last_event_id_has_no_gaps_or_duplicates`, `test_last_event_id_beats_since_param`, `test_heartbeat_comment_sent_when_idle` (tiny `heartbeat_s`), `test_run_filter_isolates_interleaved_runs`, `test_snapshot_equals_fold_of_stream`, `test_reader_survives_concurrent_writer` (thread appending while streaming), `test_21st_stream_gets_429`, `test_decision_endpoint_is_501`, `test_main_refuses_to_start_without_read_token`, `test_replay_mode_serves_recorded_journal`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_api.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement** `api.py` using `StreamingResponse` with an async generator that polls `Journal.read(since=last)` every `poll_s` and checks `request.is_disconnected()`.
- [ ] **Step 4: Run** the API tests — Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: read-only SSE API with resume, auth and replay mode"`

### Task 6: Sample journal, Docker and docs

**Files:** Create `scripts/make_sample_journal.py`, `Dockerfile`, `docker-compose.yml`, `web/public/sample-run.jsonl`; Modify `.env.example`, `README.md`

**Interfaces:**
- Produces: `python scripts/make_sample_journal.py --out web/public/sample-run.jsonl` runs the stub-backed fake run (as in `tests/test_approval.py`, approval included) and writes one event per line, evidence `EXPLANATION`. Compose services `api` (uvicorn `posttrain_agent.api`, `./state:/state`) and `tunnel` (`cloudflare/cloudflared`, `TUNNEL_TOKEN`, profile fallback to quick tunnel).

- [ ] **Step 1: Write failing test** `test_sample_journal_has_every_event_type_and_only_simulated_evidence` in `tests/test_journal_wiring.py` (run the generator function, assert all 16 types appear except possibly `job.log`, and every `evidence` is `EXPLANATION`).
- [ ] **Step 2: Run** it — Expected: FAIL.
- [ ] **Step 3: Implement** the generator (importable function `make_sample(out: Path) -> None`), then the Dockerfile/compose (healthcheck on `/v1/health`).
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q` and `docker compose config` — Expected: tests PASS, compose validates.
- [ ] **Step 5: Add** README "Live dashboard" section and `.env.example` entries (`READ_TOKEN`, `ALLOWED_ORIGIN`, `TUNNEL_TOKEN`); commit `git commit -m "feat: sample journal, Docker stack and docs"`

### Task 7: Web scaffold, event types and reducer (Opus 5.5)

**Files:** Create `web/` (Next.js App Router + TS, Vitest), `web/lib/events.ts`, `web/lib/reducer.ts`; Test `web/lib/reducer.test.ts`

**Interfaces:**
- Produces: `type Ev`, `type EventType` (the 16 types), `type RunState` (same keys as `fold.py` state), `foldEvent(state: RunState, ev: Ev): RunState` (pure, never throws), `emptyState(): RunState`.

- [ ] **Step 1: Write failing tests** `golden fixtures match Python expectations` (imports `../../contract/golden.json`), `unknown event types are ignored`, `reducer never throws on malformed events`.
- [ ] **Step 2: Run** `cd web && npx vitest run` — Expected: FAIL.
- [ ] **Step 3: Implement** `reducer.ts` so every golden case passes (the Python `fold.py` is the reference).
- [ ] **Step 4: Run** — Expected: PASS; `npx next build` succeeds.
- [ ] **Step 5: Commit** `git commit -m "feat(web): scaffold, event types, reducer"`

### Task 8: Connection layer and honesty modes (Opus 5.5)

**Files:** Create `web/lib/connection.ts`, `web/lib/store.ts`; Test `web/lib/connection.test.ts`

**Interfaces:**
- Consumes: `foldEvent`. Produces: `useRun(runId): { state: RunState; mode: "LIVE" | "REPLAY" ; conn: "CONNECTING" | "LIVE" | "RECONNECTING" | "OFFLINE" | "AUTH_FAILED" }`. Loads `/v1/runs/{id}/snapshot` then streams `?since=last_seq` via `EventSource` (reconnect with exponential backoff 1 s to 15 s, resume from last `seq`). With no `NEXT_PUBLIC_API_URL` or never-reachable backend: plays `public/sample-run.jsonl` as `mode="REPLAY"`. Never replaces live state with sample data after a live connection existed.

- [ ] **Step 1: Write failing tests** with a fake `EventSource`: `resumes_from_last_seq_after_drop`, `keeps_live_state_and_shows_reconnecting_on_drop`, `falls_back_to_sample_only_when_never_connected`, `401_sets_auth_failed`, `events_are_deduped_by_seq`.
- [ ] **Step 2: Run** `cd web && npx vitest run` — Expected: FAIL.
- [ ] **Step 3: Implement** the hook and store.
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(web): resumable SSE connection and replay fallback"`

### Task 9: Dashboard panels (Opus 5.5)

**Files:** Create `web/app/page.tsx`, `web/app/run/[id]/page.tsx`, `web/components/{StatusBar,ApprovalBanner,AgentBoard,EventStream,ScoreGate,Budget,JobPanel,Telemetry,Timeline}.tsx`; Test `web/components/panels.test.tsx`

**Interfaces:**
- Consumes: `useRun`. Panels are pure components of `RunState` (props only). Layout exactly as the approved mockup (`.superpowers/brainstorm/*/content/layout-b.html`): status bar; approval banner (display-only, shows `cli_hint`); agent board (coordinator + 8 roles, active glow, slot labels); event stream; score/gate/budget/job column; bottom job log, telemetry sparklines (canvas), timeline scrubber (scrubbing folds events up to a `seq` and shows that historical state, with a LIVE button to return).

- [ ] **Step 1: Write failing tests** `approval_banner_shows_only_when_pending_and_has_cli_hint`, `explanation_scores_are_badged_simulated`, `active_agents_show_slot_labels`, `gate_verdict_renders_reasons`, `scrubbing_shows_historical_state_and_live_button_returns`, `replay_mode_shows_sample_data_label`.
- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement** the components and pages.
- [ ] **Step 4: Run** — Expected: PASS; `npx next build` succeeds.
- [ ] **Step 5: Commit** `git commit -m "feat(web): mission-control panels"`

### Task 10: Visual system, motion and accessibility (Opus 5.5)

**Files:** Create `web/styles/terminal.css`, `web/components/Boot.tsx`; Modify panels as needed; Test `web/components/motion.test.tsx`

**Interfaces:** Produces CSS tokens (`--bg`, `--green`, `--amber`, `--cyan`, `--red`), boot typewriter header, scanlines/flicker, count-up scores, event flash-in, PROMOTE pulse / REJECT shake, pulsing active-agent borders, optional muted-by-default terminal blips with a toggle.

- [ ] **Step 1: Write failing tests** `all_animations_disabled_under_prefers_reduced_motion`, `event_log_uses_aria_live_polite_and_is_throttled`, `blips_off_by_default`.
- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement** (use the `frontend-design` skill for the visual polish; honor the Global Constraints colors).
- [ ] **Step 4: Run** tests; check text contrast ≥ 4.5:1 for body text.
- [ ] **Step 5: Commit** `git commit -m "feat(web): terminal visual system and motion"`

### Task 11: E2E smoke and Vercel readiness (Opus 5.5)

**Files:** Create `web/e2e/smoke.spec.ts`, `web/.env.example`, `web/README.md`

- [ ] **Step 1: Write failing test** `smoke.spec.ts`: with no `NEXT_PUBLIC_API_URL`, `/run/sample` shows **REPLAY · SAMPLE DATA**, events flow in, the scrubber works, and under reduced motion no animation classes are active.
- [ ] **Step 2: Run** `cd web && npx playwright test` — Expected: FAIL.
- [ ] **Step 3: Implement** any missing glue; document `NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_READ_TOKEN`, `vercel --prod` (requires the user's `vercel login`), and the tunnel steps.
- [ ] **Step 4: Run** Playwright and `npx next build` — Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(web): e2e smoke test and Vercel docs"`

---

## Self-review

- **Spec coverage:** journal/redaction (T1), reducer contract (T2), events and handler (T3-4), API incl. auth/CORS/429/501/replay/resume (T5), Docker/tunnel/sample/docs (T6), frontend stack/layout/state/engagement/honesty/a11y (T7-11). Gaps: none found. Deploying to Vercel is documented, not performed (needs the user's login).
- **Types:** state keys defined once in T2 and reused in T7; `Journal`, `Event`, `JournalHandler`, `foldEvent` names consistent across tasks.
- **Proportion:** the plan states interfaces and tests, not bodies.
