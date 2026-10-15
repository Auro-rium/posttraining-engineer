# posttrain live dashboard (`web/`)

A read-only mission-control view of a posttrain-agent run: status, agents, event stream, scores and
gate, budget, job and a timeline scrubber. It is a Next.js (App Router) app. It reads the run API
described under [Live dashboard](../README.md#live-dashboard) in the root README and never writes
anything.

With no API configured it replays a bundled **simulated** sample (`public/sample-run.jsonl`, a
stub-backed fake run), clearly labelled as such.

## Local development

```bash
cd web
npm ci
npm run dev                 # http://localhost:3000, sample replay (no API configured)
```

Open `/` for the run list or `/run/<id>` for a run. Without an API, any `/run/<id>` plays the
sample (the run list links `/run/sample-simulated-run`).

To use a real backend, start the API from the repo root (see the root README) and point the
dashboard at it:

```bash
# repo root
cp .env.example .env              # set READ_TOKEN, and ALLOWED_ORIGIN=http://localhost:3000
mkdir -p state                    # so Docker does not create ./state root-owned
docker compose up --build         # api on 127.0.0.1:8787 (503 until the first run writes the journal)

# web/
cp .env.example .env.local        # NEXT_PUBLIC_API_URL=http://127.0.0.1:8787
                                  # NEXT_PUBLIC_READ_TOKEN=<same value as READ_TOKEN>
npm run dev
```

`ALLOWED_ORIGIN` must be the exact origin the dashboard is served from (scheme, host and port, no
trailing slash). The API accepts a single origin. Restart `npm run dev` after changing `.env.local`.

## Environment variables

See [`.env.example`](.env.example).

| Variable | Meaning |
|----------|---------|
| `NEXT_PUBLIC_API_URL` | Base URL of the run API, with `http://` or `https://`. Unset or empty: sample replay. |
| `NEXT_PUBLIC_READ_TOKEN` | Must equal `READ_TOKEN` of the API. |

Both are `NEXT_PUBLIC_*`, so Next.js **embeds them in the browser bundle at build time**. Anyone
who can open the dashboard can read them. The read token is therefore not a secret: it only gates
a read-only, redacted event stream. Use a long random value that is used nowhere else, and rotate
it by changing `READ_TOKEN` and `NEXT_PUBLIC_READ_TOKEN` together, then redeploying. Changing either
variable needs a rebuild (on Vercel, a redeploy).

## Honesty modes

The status bar always says where the data comes from. Nothing recorded or simulated is ever called
LIVE.

| Label | Meaning |
|-------|---------|
| `● LIVE` | Streaming the live journal from the API. |
| `REPLAY · SAMPLE DATA` + `OFFLINE` | The bundled simulated sample. The detail in parentheses says why: `no API configured` or `backend unreachable` (the API URL is set but the backend never answered). |
| `REPLAY` + `● STREAMING` | The backend serves a recorded journal (`REPLAY_FILE`), not a live run. |
| `CONNECTING` | First contact with the API. |
| `RECONNECTING` | The connection dropped, or the API answered 429/503. The last received state stays on screen (`last received state shown`) and the dashboard retries with backoff (1 s up to 15 s), resuming from the last seq. It never swaps in sample data once the backend has answered. |
| `AUTH FAILED` | The API rejected the read token (401). Nothing is retried. |
| `OFFLINE` + `invalid API URL` | `NEXT_PUBLIC_API_URL` has no `http://`/`https://` scheme or is not a URL. Nothing is requested. |
| `HISTORICAL · seq N` | The timeline is scrubbed back to seq N. `LATEST ▸` (replay) or `LIVE ▸` returns to the newest state. |
| `SIMULATED` badge | A score, job or gate whose evidence is `EXPLANATION` (stub backend or estimate), not a measurement. |

Under `prefers-reduced-motion: reduce` every animation and transition is off; the content is the
same.

## Tests

```bash
npm test                    # Vitest unit and component tests
npm run typecheck           # tsc on the app, then on the tests (tsconfig.test.json)
npm run test:e2e            # Playwright smoke test (see below)
```

- `npm run typecheck` runs two passes. `tsconfig.json` (used by `next build`) excludes
  `*.test.ts(x)`, `e2e/` and `playwright.config.ts`, because `lib/reducer.test.ts` imports
  `../../contract/golden.json` from outside `web/` (the fixtures shared with the Python reducer).
  This keeps `next build` working when only `web/` is present (Vercel Root Directory = `web`).
  `tsconfig.test.json` type-checks everything, tests included.
- The unit tests need the full repository checkout (they read `contract/golden.json`).
- `npm run test:e2e` runs `e2e/smoke.spec.ts` against a production build: the Playwright config
  starts `next build && next start` on `127.0.0.1:3917` (override with `E2E_PORT`) with
  `NEXT_PUBLIC_API_URL` and `NEXT_PUBLIC_READ_TOKEN` removed from the environment, so it tests the
  sample replay. Do not keep a `.env.local` that sets them while running it. It checks the
  `REPLAY · SAMPLE DATA` label, that events arrive, the `SIMULATED` badges, scrubbing back to
  `HISTORICAL` and `LATEST`, and that no animation runs under reduced motion. It takes about 30 s.
- The e2e test uses Playwright's bundled Chromium, not a system Chrome, so results do not depend
  on the local browser. Install it once with `npx playwright install chromium`.

## Deploy to Vercel

The backend runs on your own machine in Docker and is exposed with a Cloudflare tunnel; only the
dashboard runs on Vercel. Deploying needs **your own Vercel account**: run these yourself (nothing
in this repository deploys automatically).

1. **Expose the API over https** (from the repo root, see the root README):

   ```bash
   mkdir -p state                          # once, before the first `docker compose up`
   docker compose --profile tunnel up -d   # named tunnel, stable hostname (needs TUNNEL_TOKEN)
   docker compose --profile quick up       # or a temporary https://<random>.trycloudflare.com
   ```

   For the named tunnel, route the public hostname to `http://api:8787` in the Cloudflare tunnel
   settings. For the quick tunnel, read the `https://...trycloudflare.com` URL from the
   `cloudflared` log. It changes on every restart, so update `NEXT_PUBLIC_API_URL` and redeploy
   each time. Check it with `curl https://<host>/v1/health`.

2. **Link and configure the project** (in `web/`):

   ```bash
   npm i -g vercel            # or prefix the commands below with npx
   vercel login
   vercel link                # create or pick the project
   ```

   Set the project's **Root Directory** to `web` (Project Settings -> General) if you link from the
   repo root or connect the Git repository; when you run `vercel` inside `web/` it is already the
   root. The framework preset is Next.js; the default build command (`next build`) is right.

3. **Set the environment variables** for Production (and Preview if you use it):

   ```bash
   vercel env add NEXT_PUBLIC_API_URL production       # https://<tunnel host>, no trailing path
   vercel env add NEXT_PUBLIC_READ_TOKEN production    # same value as READ_TOKEN
   ```

4. **Deploy**:

   ```bash
   vercel --prod
   ```

5. **Allow the dashboard's origin on the API.** In the root `.env`, set
   `ALLOWED_ORIGIN=https://<your-project>.vercel.app` (exactly the URL in the browser's address bar,
   no trailing slash) and restart the API (`docker compose up -d api`). The API accepts one origin,
   so a Preview deployment URL is rejected unless you set `ALLOWED_ORIGIN` to it instead.

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `AUTH FAILED (invalid read token (401))` | `NEXT_PUBLIC_READ_TOKEN` differs from `READ_TOKEN`. Make them equal and redeploy (the value is baked in at build time). |
| `REPLAY · SAMPLE DATA (backend unreachable)` on Vercel | The browser never got an answer from the API. Usual causes: the tunnel is down or its quick-tunnel URL changed; `ALLOWED_ORIGIN` is not exactly the Vercel URL (the browser drops the response as a CORS error); the API URL is `http://` (mixed content, see below). Open the browser console to see which. |
| Mixed content / blocked request | Vercel serves https, so the browser blocks an `http://` API. Use the tunnel's `https://` URL in `NEXT_PUBLIC_API_URL`; never a LAN IP or `http://127.0.0.1:8787` in production. |
| `RECONNECTING (too many streams (429))` | The API caps concurrent streams (20). Close other dashboard tabs; it retries by itself. |
| `RECONNECTING (journal unavailable (503))` | Expected on a fresh checkout: the API returns 503 until the first run writes `state/events.sqlite` (start one with `posttrain run`). If it persists after a run, check the `./state` mount and UID/GID. See the root README. |
| `RECONNECTING (backend unreachable / connection lost)` | The tunnel or API went away after a successful connect. The last state stays on screen; it resumes from the last seq when the API is back. |
| `OFFLINE (invalid API URL ...)` | `NEXT_PUBLIC_API_URL` lacks `https://`. Fix it and redeploy. |
| Stuck at `CONNECTING` | The health check has not answered yet; check `curl <API URL>/v1/health`. |
