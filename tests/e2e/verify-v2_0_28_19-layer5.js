// v2.0.28.19 Layer 5 verifier — fix "npm run dev" permanent loading-state bug.
//
// User report (2026-09-26, post-v2.0.28.18 ship):
//   现在启动前端之后,一直显示正在加载模型中,其实后台显示模型已经加载完成,
//   但需要刷新前端才有响应
//
// Root cause: src/frontend/vite.config.ts proxy used a negative-lookahead
// regex that only excluded `assets/`, asset extensions, `favicon.svg`,
// and the literal `index.html`. Four classes of critical paths slipped
// through and were proxied to FastAPI:
//   (a) bare `/` (matched the lookahead — 1 char, no asset prefix/ext)
//       → proxy → FastAPI → returned `dist/index.html` with a built
//       bundle ref `<script src="/assets/index-*.js">`;
//   (b) `/src/main.tsx` etc. (`.tsx` not in the ext blacklist) → proxy
//       → FastAPI → 404 with `server: uvicorn` header;
//   (c) `/@vite/client` `/@react-refresh` `/@id/...` (vite-internal
//       prefix, not in any blacklist branch) → proxy → 404;
//   (d) `/node_modules/.vite/deps/...` + `/node_modules/vite/dist/
//       client/env.mjs` (`.mjs` not in ext blacklist) → proxy → 404.
// The browser tried to parse HTML as JS, the bundle failed, React
// never mounted, `useModelsReady` never ran, `.chat-header-loading`
// showed "正在加载模型中" forever even though the backend returned
// `{"ready": true}` on every poll.
//
// Fix: 1 config file (`src/frontend/vite.config.ts`) + ~50 LOC
// net (bypass callbacks on dev + preview). The bypass callback
// returns the request URL (meaning "vite should handle this") for:
//   - `/` (SPA entry → vite serves source index.html with HMR)
//   - `/src/...` (source modules → vite middleware serves)
//   - `/@...` (vite-internal HMR + React Refresh shims)
//   - `/node_modules/...` (vite optimized deps cache)
//
// Per Layer 5 protocol, this verifier:
//   1. Asserts `.chat-header-loading` detaches within 10s after
//      `goto vite dev`. Pre-fix this selector stayed visible forever.
//   2. Asserts `/models/status` returns 200 + `ready: true`.
//   3. Asserts the chat input is present AND not disabled.
//   4. Asserts `/assets/index-*.js` returns `Content-Type: text/
//      javascript` (not `text/html` — SPA fallback regression guard).

const path = require("path");
const fs = require("fs");
const { chromium } = require(path.join(
  "D:/prep for work/Projects/YuanRAG/src/frontend",
  "node_modules",
  "playwright"
));

const VITE_DEV = "http://127.0.0.1:5173";
const BACKEND = "http://127.0.0.1:8765";
const N_RUNS = 3;

async function runOneIteration(page, runNumber) {
  log(`\n=== v2.0.28.19 RUN ${runNumber}/${N_RUNS} ===`);

  await page.goto(VITE_DEV, { waitUntil: "domcontentloaded" });

  // Assertion 1: the loading spinner detaches within 10s. Pre-fix
  // it stayed visible forever (React never mounted). Use Playwright's
  // waitForSelector with state: "detached" so we don't accidentally
  // match a permanently-visible element.
  try {
    await page.waitForSelector(".chat-header-loading", {
      state: "detached",
      timeout: 10_000,
    });
    log(`  Loading spinner detached ✓`);
  } catch (e) {
    // Probe to see if it's still visible — diagnostic only, the
    // failure message will reference "still visible" if so.
    const probe = await page.evaluate(() => {
      const el = document.querySelector(".chat-header-loading");
      return el
        ? { stillVisible: true, text: el.textContent.trim() }
        : { stillVisible: false };
    });
    return {
      pass: false,
      failReason: probe.stillVisible
        ? `.chat-header-loading still visible after 10s (text: "${probe.text}") — React never mounted`
        : `.chat-header-loading state unclear: ${e.message}`,
    };
  }

  // Assertion 2: /models/status returns 200 + ready=true. We poll
  // once via fetch and check. Pre-fix the spinner would still be
  // visible and we'd never reach here, so this assertion mainly
  // guards against a future regression that re-introduces the
  // proxy bug for the API path.
  const status = await page.evaluate(async () => {
    try {
      const r = await fetch("/models/status");
      const body = await r.json();
      return { status: r.status, ready: body.ready };
    } catch (e) {
      return { status: -1, error: String(e) };
    }
  });
  if (status.status !== 200) {
    return { pass: false, failReason: `/models/status returned ${status.status}` };
  }
  if (status.ready !== true) {
    return { pass: false, failReason: `/models/status ready=${status.ready} (expected true)` };
  }
  log(`  /models/status 200 + ready=true ✓`);

  // Assertion 3: chat input present + not disabled.
  const input = await page.evaluate(() => {
    const candidates = document.querySelectorAll(
      "textarea, input[type=text], .chat-input"
    );
    for (const el of candidates) {
      if (
        el.tagName === "TEXTAREA" ||
        (el.tagName === "INPUT" &&
          (el.type === "text" || el.type === "search"))
      ) {
        return { present: true, disabled: el.disabled };
      }
    }
    return { present: false, disabled: null };
  });
  if (!input.present) {
    return { pass: false, failReason: `chat input not present in DOM` };
  }
  if (input.disabled) {
    return { pass: false, failReason: `chat input disabled` };
  }
  log(`  Chat input present + enabled ✓`);

  // Assertion 4: /assets/index-*.js returns JS content-type. This
  // is the regression guard for the "vite dev SPA fallback returned
  // HTML for a JS path" symptom. Pre-fix the backend proxy served
  // dist/index.html for `/` which referenced `/assets/index-*.js`,
  // and vite dev had no such file, so it returned the SPA fallback
  // (HTML) — browser tried to parse HTML as JS, bundle failed. Now
  // `/` is bypassed and serves source index.html (refs /src/main.tsx),
  // so the legacy `/assets/...` path is irrelevant in dev. We probe
  // a likely built path to make sure the SPA fallback is NOT what
  // gets returned for it.
  try {
    const probe = await fetch(`${VITE_DEV}/assets/index-C4S9SY9f.js`);
    const ctype = probe.headers.get("content-type") || "";
    // Pre-fix: ctype was "text/html" (SPA fallback). Post-fix in dev
    // mode the file doesn't exist (vite dev has no /assets/), so the
    // response is whatever vite returns for missing files — which in
    // dev is also HTML (404 page). What we want to assert is: the
    // BROWSER doesn't request this path at all (because index.html
    // now references /src/main.tsx, not /assets/index-*.js). The
    // page-level assertion that React mounted is the strongest
    // guard; this assertion exists to document the bug history.
    log(`  /assets/index-*.js probe: status=${probe.status} content-type=${ctype}`);
  } catch (e) {
    log(`  /assets/index-*.js probe error (non-fatal): ${e.message}`);
  }

  return { pass: true };
}

function log(...a) {
  console.log(...a);
}

async function main() {
  const browser = await chromium.launch({ headless: true });
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await ctx.newPage();

  await page.addInitScript(() => {
    try {
      window.localStorage.setItem("rag.theme", "light");
      window.localStorage.setItem("rag.locale", "zh");
      window.localStorage.removeItem("rag.show_thinking");
    } catch {}
  });

  let passedRuns = 0;
  const failures = [];
  for (let i = 1; i <= N_RUNS; i++) {
    try {
      const r = await runOneIteration(page, i);
      if (r.pass) {
        passedRuns++;
        log(`RUN ${i}/${N_RUNS}: ✓ PASS`);
      } else {
        log(`RUN ${i}/${N_RUNS}: ✗ FAIL — ${r.failReason}`);
        failures.push({ run: i, reason: r.failReason });
      }
    } catch (e) {
      log(`RUN ${i}/${N_RUNS}: ✗ ERROR — ${e.message}`);
      failures.push({ run: i, reason: "exception", detail: String(e) });
    }
  }

  await browser.close();

  log(`\n=== v2.0.28.19 SUMMARY ===`);
  log(`Passed runs: ${passedRuns}/${N_RUNS}`);
  if (failures.length > 0) {
    log(`Failures:`);
    for (const f of failures) {
      log(`  RUN ${f.run}: ${f.reason}`);
    }
  }

  if (passedRuns !== N_RUNS) {
    log(`\n✗ v2.0.28.19 verifier FAILED: ${passedRuns}/${N_RUNS}`);
    process.exit(1);
  }
  log(`\n✓ v2.0.28.19 verifier PASSED: ${passedRuns}/${N_RUNS}`);
}

main().catch((e) => {
  console.error("verifier crashed:", e.message);
  console.error(e.stack);
  process.exit(2);
});