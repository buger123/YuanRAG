import { defineConfig, devices } from "@playwright/test";

/**
 * Layer 5 e2e config — true Chromium loading `dist/` + real WS + real LLM.
 *
 * This is the canonical verification harness for shipping fixes (see
 * `memory/development-methodology.md`). It is NOT collected by pytest; run
 * with `npm run e2e` from `src/frontend/`, or `bash tests/e2e/run.sh`.
 *
 * Pre-conditions (asserted up-front, NOT auto-started — developer runs them
 * explicitly so failures are obvious):
 *   1. FastAPI backend on http://127.0.0.1:8765 with .env containing real
 *      LLM key (otherwise smoke.spec.ts will skip)
 *   2. `npm run build` already produced dist/ in src/frontend/
 *
 * The webServer below spawns `vite preview` which serves the built dist/
 * and proxies /api + /ws back to the backend on :8765.
 */
export default defineConfig({
  testDir: ".",
  testMatch: /(?:smoke|web_search|mobile)\.spec\.ts/,
  // One test at a time — these hit the real backend, parallelism would
  // cross-contaminate sessions.
  workers: 1,
  // Generous timeout — real LLM round-trips are 2–8 s end-to-end.
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  retries: 0,
  reporter: process.env.CI ? "dot" : "list",
  use: {
    baseURL: "http://127.0.0.1:4173",
    trace: "retain-on-failure",
    video: "retain-on-failure",
    screenshot: "only-on-failure",
    actionTimeout: 10_000,
    navigationTimeout: 30_000,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: {
    // `vite preview` reuses the same /api + /ws proxy as `vite dev`,
    // pointing at the FastAPI backend on :8765.
    command: "npm run build && npm run preview -- --port 4173 --strictPort",
    // Run from src/frontend/ so npm finds package.json + vite binary.
    cwd: "../src/frontend",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    stdout: "pipe",
    stderr: "pipe",
  },
  outputDir: "./output",
});