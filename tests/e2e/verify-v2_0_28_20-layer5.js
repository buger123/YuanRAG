// v2.0.28.20 Layer 5 verifier — visibility-aware polling fix.
//
// User report (2026-09-26, post-v2.0.28.19 ship):
//   是模型加载完之后需要刷新才能使用,后台显示模型已经加载完成,但是前端
//   右上角一直在显示模型加载中,像是卡住了
//
// v2.0.28.19 fixed the bundle-load path. What remained: Chrome/Edge
// throttle ``setTimeout`` in background tabs to once per ~60 s,
// stretching our 2 s poll cadence to a minute. The user opens the
// page, switches tabs while models load, comes back, and the spinner
// stays stuck for up to a minute before the next poll.
//
// Fix: useModelsReady attaches a visibilitychange listener that fires
// an immediate re-poll when the tab returns to the foreground.
//
// This verifier uses Playwright's ``page.route()`` (HTTP-level mock)
// instead of an in-page fetch wrapper — the in-page wrapper didn't
// stick because vite's module loader captures ``fetch`` at import
// time. The route mock works at the network layer so the timing is
// deterministic and independent of how the API client is structured.

const path = require("path");
const { chromium } = require(path.join(
  "D:/prep for work/Projects/YuanRAG/src/frontend",
  "node_modules",
  "playwright"
));

const FRONTEND = "http://127.0.0.1:5173";
const N_RUNS = 3;

async function runOneIteration(page, runNumber) {
  log(`\n=== v2.0.28.20 RUN ${runNumber}/${N_RUNS} ===`);

  // Set up a network-level mock for /models/status: any call while
  // "tab hidden" returns ready=false (spinner stays visible); the
  // first call after "tab visible" returns ready=true (so the
  // visibility-triggered re-poll clears the spinner). This avoids
  // the StrictMode double-effect race where two ticks fire back-to-
  // back and would otherwise both flip to ready=true.
  let modelsStatusCalls = 0;
  let allowReady = false;
  await page.route("**/models/status", async (route) => {
    modelsStatusCalls += 1;
    const body = allowReady
      ? {
          embedding: { name: "BGE-M3", status: "ready" },
          reranker: { name: "BGE Reranker v2-M3", status: "ready" },
          ready: true,
        }
      : {
          embedding: { name: "BGE-M3", status: "downloading" },
          reranker: { name: "BGE Reranker v2-M3", status: "downloading" },
          ready: false,
        };
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(body),
    });
  });

  await page.goto(FRONTEND, { waitUntil: "domcontentloaded" });

  // Wait briefly for the first poll to land — the spinner should be
  // visible because the mock returned ready=false. We deliberately
  // wait < POLL_MS (2000) so the second mock call (which would flip
  // ready=true and clear the spinner) doesn't fire first.
  await page.waitForTimeout(800);

  // Assertion 1: the spinner is currently visible (mock guaranteed this).
  let state = await page.evaluate(() => ({
    loadingVisible: !!document.querySelector(".chat-header-loading"),
    visibilityState: document.visibilityState,
    callCount: (window).__modelsStatusCalls || 0,
  }));
  if (!state.loadingVisible) {
    return {
      pass: false,
      failReason: `spinner NOT visible after first poll (mock should force ready=false — state=${JSON.stringify(state)}, calls=${modelsStatusCalls})`,
    };
  }
  if (modelsStatusCalls < 1) {
    return {
      pass: false,
      failReason: `mock never received /models/status (calls=${modelsStatusCalls})`,
    };
  }
  log(`  Spinner visible after first poll (mock ready=false, calls=${modelsStatusCalls}) ✓`);

  // Simulate the user switching tabs while models are still loading.
  // We explicitly dispatch the event AND override the property —
  // this is exactly what the production hook listens for.
  await page.evaluate(() => {
    Object.defineProperty(document, "visibilityState", {
      value: "hidden",
      configurable: true,
    });
    document.dispatchEvent(new Event("visibilitychange"));
  });

  // Wait while the tab is "background" — in a real browser this is
  // where setTimeout would be throttled.
  await page.waitForTimeout(3000);

  // Verify the spinner is STILL visible during the hidden window.
  state = await page.evaluate(() => ({
    loadingVisible: !!document.querySelector(".chat-header-loading"),
  }));
  if (!state.loadingVisible) {
    return {
      pass: false,
      failReason: `spinner disappeared during hidden window (something else cleared it — calls=${modelsStatusCalls})`,
    };
  }
  log(`  Spinner still visible during hidden window (calls=${modelsStatusCalls}) ✓`);

  const callsBeforeFocus = modelsStatusCalls;

  // Now flip back to visible — the hook should fire an immediate
  // re-poll that returns ready=true and clear the spinner.
  // Flip the mock BEFORE dispatching the event so the very next poll
  // returns ready=true.
  allowReady = true;
  await page.evaluate(() => {
    Object.defineProperty(document, "visibilityState", {
      value: "visible",
      configurable: true,
    });
    document.dispatchEvent(new Event("visibilitychange"));
  });

  // Wait for the hook's re-poll to land + React to render the change.
  await page.waitForTimeout(2000);

  // Assertion 2: the spinner is now GONE (visibility-triggered re-poll
  // delivered the ready=true response).
  state = await page.evaluate(() => ({
    loadingVisible: !!document.querySelector(".chat-header-loading"),
    inputPresent: !!document.querySelector("textarea"),
    inputDisabled: document.querySelector("textarea")?.disabled ?? null,
  }));
  if (state.loadingVisible) {
    return {
      pass: false,
      failReason: `spinner STILL visible after visibilitychange → visible (calls=${modelsStatusCalls}, was ${callsBeforeFocus})`,
    };
  }
  if (modelsStatusCalls <= callsBeforeFocus) {
    return {
      pass: false,
      failReason: `visibilitychange → visible did NOT trigger a re-poll (calls went from ${callsBeforeFocus} to ${modelsStatusCalls})`,
    };
  }
  if (!state.inputPresent) {
    return {
      pass: false,
      failReason: `chat input not present`,
    };
  }
  if (state.inputDisabled) {
    return {
      pass: false,
      failReason: `chat input disabled (modelsReady never propagated to ChatPane)`,
    };
  }
  log(`  Spinner detached after visibilitychange → visible (calls=${modelsStatusCalls}) ✓`);

  // Reset the route for the next iteration by clearing all routes.
  await page.unroute("**/models/status");

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

  log(`\n=== v2.0.28.20 SUMMARY ===`);
  log(`Passed runs: ${passedRuns}/${N_RUNS}`);
  if (failures.length > 0) {
    log(`Failures:`);
    for (const f of failures) {
      log(`  RUN ${f.run}: ${f.reason}`);
    }
  }

  if (passedRuns !== N_RUNS) {
    log(`\n✗ v2.0.28.20 verifier FAILED: ${passedRuns}/${N_RUNS}`);
    process.exit(1);
  }
  log(`\n✓ v2.0.28.20 verifier PASSED: ${passedRuns}/${N_RUNS}`);
}

main().catch((e) => {
  console.error("verifier crashed:", e.message);
  console.error(e.stack);
  process.exit(2);
});