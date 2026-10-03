/**
 * vitest config — frontend unit-test infrastructure (v2.0.26 PR-3).
 *
 * Why vitest + RTL?
 * -----------------
 * The frontend had no unit-test infrastructure before this PR. Every
 * behavioral fix was verified by hand-running the app or by the
 * Playwright smoke harness — both expensive, both prone to regression
 * drift as the codebase grows. Item 6 PR-3 splits the 438-LOC
 * ``handleAgentEvent`` into per-event-type handlers (one file per
 * ``AgentEvent.type``); that split would be pointless without a way
 * to lock the new behavior down with unit tests.
 *
 * Why jsdom (not happy-dom)?
 * ---------------------------
 * jsdom is the most-spec-compliant DOM implementation in the JS
 * ecosystem, with first-class support for ``CustomEvent``,
 * ``AbortController``, and ``crypto.randomUUID`` — all of which the
 * handlers exercise. happy-dom is faster but has known gaps in
 * event dispatch semantics; not worth the risk for a 700ms delta on
 * a test suite that runs once per ship.
 *
 * Why a separate config (not extend vite.config.ts)?
 * --------------------------------------------------
 * vite.config.ts uses ``@vitejs/plugin-react`` which is fine for
 * both, but ``vite build`` would pick up the test config (vitest's
 * presence in the deps tree makes vite try to evaluate it). Keeping
 * the test runner config in its own file means production builds
 * never accidentally import vitest.
 *
 * Glob notes
 * ----------
 * - per-handler unit tests are colocated next to their handler as
 *   test.ts files under src so a code change naturally pulls the
 *   test into the diff.
 * - integration tests (dispatchAgentEvent full lifecycle) live under
 *   the __tests__ folder so the larger suites don't get mixed into
 *   the per-handler test.ts namespace used for unit-test helpers.
 */
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: true,
    include: [
      "src/**/*.test.{ts,tsx}",
      "src/**/__tests__/**/*.test.{ts,tsx}",
    ],
    exclude: ["node_modules/**", "dist/**"],
    // Per-PR-3 plan: no coverage threshold yet — the suite is too
    // small to gate on coverage %. PR-4 (i18n) will add the
    // snapshot tests that make coverage meaningful; we revisit the
    // threshold then.
    coverage: {
      provider: "v8",
      reporter: ["text", "html"],
    },
    setupFiles: ["./src/test-setup.ts"],
  },
});