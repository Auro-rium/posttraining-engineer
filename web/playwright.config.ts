import { defineConfig, devices } from "@playwright/test";

/**
 * E2E smoke test against a production build (`next build && next start`) in sample-replay mode.
 *
 * NEXT_PUBLIC_API_URL / NEXT_PUBLIC_READ_TOKEN are removed from the environment the server inherits,
 * so the build has no API configured and /run/<id> replays the bundled simulated sample. (A
 * web/.env.local that sets them would still be read by Next; keep it absent when running this.)
 *
 * Browser: Playwright's bundled Chromium (`npx playwright install chromium`), not a system Chrome,
 * so the run is the same on any machine. Port: E2E_PORT, default 3917.
 */
delete process.env.NEXT_PUBLIC_API_URL;
delete process.env.NEXT_PUBLIC_READ_TOKEN;

const PORT = Number(process.env.E2E_PORT ?? 3917);
const BASE_URL = `http://127.0.0.1:${PORT}`;

export default defineConfig({
  testDir: "./e2e",
  // The sample replays in about 25 s (gaps capped at 2 s).
  timeout: 90_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: "list",
  use: {
    baseURL: BASE_URL,
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: `npx next build && npx next start -H 127.0.0.1 -p ${PORT}`,
    url: BASE_URL,
    reuseExistingServer: false,
    timeout: 240_000,
    stdout: "pipe",
    stderr: "pipe",
  },
});
