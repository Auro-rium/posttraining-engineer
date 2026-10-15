# Live Mission-Control Dashboard: Design

Status: draft for review. Date: 2026-10-15.

## 1. Goal and success criteria

A neo-futuristic, highly engaging web page that shows a `posttrain-agent` run unfolding live over SSE,
deployed on Vercel, with the data served from the operator's own machine (Docker).

Success means:
- A visitor opening the Vercel URL during a real run sees agents, events, scores, budget and training
  progress update within about one second, with no refresh.
- With no backend reachable, the page still shows a clearly labelled **REPLAY** of a recorded sample run.
- Nothing simulated is ever presented as measured: every score carries its evidence label.
- Reconnecting after a network drop loses no events.

Non-goals (this iteration): authentication beyond a read token, multi-user accounts, approve/reject
buttons (the contract reserves a slot; see section 9), editing runs, mobile-first layout.

## 2. Decisions already made with the user

| Topic | Decision |
|---|---|
| Interactivity | **Watch-only now**, contract designed so approve/reject can be added later (option C) |
| Backend hosting | **User's own machine, Docker**, exposed to Vercel through a tunnel |
| Visual direction | **B: Mission-Control Terminal** (phosphor green + amber, monospace, scanlines, dense log) |
| Panels | **All eight**: event log, agent board, score/gate, budget, training job, telemetry, approval banner, checkpoint timeline |
| Event source | **B: append-only event journal** written by the agent, tailed by the API over SSE |
| Builder | Frontend built by **Opus 5.5** after the plan is approved |

## 3. Architecture

```
 CLI run (posttrain)         docker compose
 ┌────────────────┐   ┌──────────────────────────────┐        Vercel
 │ agent + tools  │   │  api (FastAPI, read-only)    │      ┌────────────┐
 │ JournalHandler ├──►│  /v1/stream  /v1/runs/...    │◄─SSE─┤ Next.js UI │
 └──────┬─────────┘   └──────────────▲───────────────┘      └────────────┘
        │ writes                      │ reads                      ▲
        ▼                             │                            │ https
   state/events.sqlite  ◄─────────────┘            cloudflared tunnel (public URL)
   (shared volume)
```

The CLI and the API are separate processes sharing `state/` via a Docker volume. The API never writes
to the journal and never runs the agent.

## 4. Event journal and stream contract

### 4.1 Journal
`state/events.sqlite`, WAL mode, one table `events(seq INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL,
run TEXT, type TEXT, data TEXT/JSON)`. `seq` is the global monotonic cursor. Retention: keep the most
recent 100,000 rows (prune oldest on write, never within an active run's window).

Writers: `JournalHandler` (extends `TelemetryHandler`; model/tool callbacks) and explicit
`journal.emit(...)` calls from the run controller (`replay.py`), the domain tools and the approval gate.
A journal failure must never fail a run: writes are best-effort and errors are logged once.

### 4.2 Event envelope
```ts
type Ev = { seq: number; ts: number; run: string; type: EventType; data: Record<string, unknown> };
```

### 4.3 Event types (closed set; unknown types are ignored by the UI)
| type | data (all fields redacted/allow-listed) |
|---|---|
| `run.started` | `goal_len`, `experiment_budget`, `backend` (`nebius`\|`stub`) |
| `run.status` | `status` (RUNNING, AWAITING_APPROVAL, COMPLETED, FAILED), `attempt`, `errors` (type names only) |
| `agent.start` / `agent.end` | `role`, `slot` (1\|2), `ok` (end only) |
| `model.call` | `name` (provider:model), `ms`, `input_tokens`, `output_tokens`, `ok`, `error` (class name), `fallback` (bool) |
| `tool.call` | `name`, `ms`, `ok`, `error` |
| `score` | `split`, `checkpoint`, `value`, `n`, `evidence` (LIVE\|PRIOR_VERIFIED_RUN\|EXPLANATION) |
| `gate.verdict` | `decision`, `reasons`, `champion_score`, `candidate_score`, `relative_improvement` |
| `budget.update` | `spent_usd`, `cap_usd`, `estimate_usd`, `price_known` |
| `job.update` | `job_id`, `status`, `trained_steps`, `total_steps`, `trained_tokens`, `evidence` |
| `job.log` | `level`, `message` (truncated to 300 chars, provider text only) |
| `approval.pending` | `action` (`launch_training`), `base_model`, `dataset_ref`, `hyperparameters`, `estimate_usd`, `cli_hint` |
| `approval.resolved` | `decision` (approve\|reject), `message_len` |
| `checkpoint` | `checkpoint_id`, `step` |
| `telemetry.snapshot` | `p50_ms`, `p95_ms`, `error_rate`, `input_tokens`, `output_tokens` (emitted at most every 5 s while active) |

### 4.4 Redaction (hard requirement)
Events never contain prompts, completions, message text, tool arguments, held-out task contents or
secrets. Only the fields listed above pass; anything else is dropped by an allow-list in `events.py`.
Tested by a property test that runs a full fake run with sentinel strings in every input and asserts
none appear in the journal.

### 4.5 API (FastAPI, read-only, in Docker)
- `GET /v1/stream?run=<thread>&since=<seq>`: SSE. Frame `id: <seq>`, `event: <type>`, `data: <json>`.
  Honors `Last-Event-ID` (takes precedence over `since`). Heartbeat comment every 15 s. Tails the
  journal (poll 250 ms). Without `run`, streams all runs.
- `GET /v1/runs`: runs with status, started, last seq.
- `GET /v1/runs/{id}/snapshot`: current folded state plus `last_seq`; the UI loads this, then streams
  from `last_seq`. The fold logic lives in one shared reducer spec (section 6.3) tested on both sides.
- `GET /v1/health`.
- Auth: static read token in `?token=` (EventSource cannot set headers), compared in constant time.
  CORS allows only `ALLOWED_ORIGIN`. At most 20 concurrent SSE connections; excess get 429.
- Reserved, not built: `POST /v1/runs/{id}/decision` returns 501.
- Replay mode: `REPLAY_FILE=<journal.jsonl>` serves a recorded journal through the same endpoints,
  paced by original timestamps (speed multiplier query param `speed`).

## 5. Deployment

### 5.1 Docker (`docker-compose.yml`)
- `api`: Python 3.12 image running uvicorn; mounts `./state:/state`; opens the journal with SQLite
  read-only semantics and never writes. WAL needs the `-shm` file, so the mount is read-write at the
  filesystem level but the code path is read-only.
- `tunnel`: `cloudflare/cloudflared`. Preferred: a named tunnel via `TUNNEL_TOKEN` (stable URL).
  Fallback: quick tunnel (random URL that changes on restart; the user must update the Vercel env var).
- Env: `READ_TOKEN`, `ALLOWED_ORIGIN`, `REPLAY_FILE` (optional), `STATE_DIR`.

### 5.2 Vercel
`web/` is the project root. Env vars: `NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_READ_TOKEN`. The token is
public by construction (it ships in the browser bundle); it only gates a read-only, redacted stream and
is rotated by changing both env vars. Deploying needs the user's own `vercel login`; Opus prepares the
project and documents `vercel --prod` but cannot deploy without that login.

## 6. Frontend (`web/`, built by Opus 5.5)

### 6.1 Stack
Next.js App Router + TypeScript, Zustand store, native `EventSource`, plain DOM/CSS for panels, a small
canvas for sparklines and the timeline. No heavy chart library. Tests: Vitest (reducer, components) and
a Playwright smoke test in replay mode.

### 6.2 Pages and layout
- `/` run list plus LIVE/REPLAY indicator.
- `/run/[id]` the dashboard: top status bar (thread, state, experiment, connection, seq); amber approval
  banner when `approval.pending` is unresolved (display-only, shows the CLI command); left agent board
  (coordinator + 8 roles, up to 2 active, slot-labelled); center event stream; right score/gate,
  budget, job progress; bottom job-log tail, telemetry sparklines, checkpoint timeline scrubber.
- Visual language (direction B): background near-black, phosphor green primary, amber for money and
  approval, cyan for info, red for errors; monospace; scanlines; CRT flicker. Layout approved via the
  browser mockup at `.superpowers/brainstorm/*/content/layout-b.html`.

### 6.3 State
One pure reducer `fold(state, event) -> state` handles every event type. The server snapshot uses the
same specified semantics; a shared set of golden JSON fixtures (events in, expected state out) is the
contract tested in both Python and TypeScript.

### 6.4 Engagement
Boot sequence with typewriter header; count-up scores; event flash-in; PROMOTE pulse (green) and REJECT
shake (red); blinking approval banner; pulsing borders on active agents; optional terminal blips
(muted by default). All motion is disabled under `prefers-reduced-motion`.

### 6.5 Connection and honesty rules
- States: `CONNECTING`, `LIVE`, `RECONNECTING` (exponential backoff, resume with last `seq`),
  `OFFLINE` (backend unreachable).
- If `NEXT_PUBLIC_API_URL` is unset, or the backend was never reachable, the page shows the bundled
  sample run (`web/public/sample-run.jsonl`), labelled **REPLAY · SAMPLE DATA** in amber, never mixed
  with live events. If a live connection drops mid-run, the last live state stays on screen, labelled
  **RECONNECTING**, and is never replaced by sample data.
- Every score shows its evidence label. `EXPLANATION` is badged **SIMULATED** in amber.
- The sample run is generated from the stub backend by `scripts/make_sample_journal.py`, so it is
  honestly simulated and labelled as such.

## 7. Error handling
- Journal write failure: logged once, run continues, UI shows a gap-free but possibly sparse stream.
- API cannot open the journal: `/v1/health` reports degraded; streams return 503.
- Malformed event: the UI ignores unknown types and fields; the reducer never throws (tested).
- Too many SSE clients: 429; the UI retries with backoff.
- Token wrong: 401; the UI shows `AUTH FAILED` rather than a silent empty page.

## 8. Testing
- Backend (pytest): journal ordering and monotonic `seq`; redaction property test; allow-list drops
  unknown fields; SSE resume via `Last-Event-ID`; heartbeat; 401/429/CORS; snapshot equals fold of the
  stream; replay pacing; pruning; journal failure does not fail a run; shared golden fixtures.
- Integration: a fake run (existing fakes) with the real `JournalHandler`, asserting the expected event
  sequence including two parallel agent slots and an approval pause.
- Frontend: Vitest reducer vs the golden fixtures; component tests for connection states; Playwright
  smoke in replay mode (page renders, events flow, scrubber works, reduced-motion respected); `next
  build` must pass; basic a11y (contrast, `aria-live=polite` on the log, throttled).

## 9. Future (explicitly out of scope now)
Approve/reject buttons via the reserved `POST /v1/runs/{id}/decision` (needs real auth, CSRF-safe
tokens and an audit event), multi-run comparison view, mobile layout.

## 10. Work breakdown
1. **Backend slice** (Python): `events.py` (journal + allow-list), `JournalHandler`, emits in
   `replay.py`/domain tools/approval, `api.py`, replay mode, `Dockerfile`, `docker-compose.yml`,
   sample-journal generator, golden fixtures, tests. Built first because it defines the contract.
2. **Frontend slice** (`web/`, Opus 5.5): everything in section 6, driven by the golden fixtures and the
   sample journal, buildable and testable with no backend running.
3. **Integration and deploy docs**: README section, `.env.example` additions, tunnel and Vercel steps.
