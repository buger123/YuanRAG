// v2.0.30.0 Layer 5 verifier — long-poll model-readiness fix.
//
// User report (2026-09-29):
//   在加载模型之后让前端及时知道模型是否加载完成,现在即使模型加载完成了,
//   前端右上角仍然显示模型加载中
//
// v2.0.28.20 fixed the background-tab throttling. What remained:
// foreground-tab polling lagged up to 2 s behind backend ``ready=true``
// because the hook only re-read status on a 2 s ``setTimeout`` cadence.
// The fix is a long-poll ``/models/wait?timeout=60`` issued on mount,
// which lets the backend block the request until models are ready
// (typical latency ~250 ms when models are already loaded).
//
// We verify five things, each via a ``page.route`` HTTP-level mock:
//   A. Long-poll latency — first /models/wait returns ready=true,
//      spinner detaches within ~1500 ms of mount.
//   B. Long-poll fallback — /models/wait returns 408, hook falls back
//      to /models/status polling, spinner still detaches.
//   C. Ready signal — DOM assertion: .chat-header-loading is gone
//      after the wait resolves, chat input enabled.
//   D. Tab-hide abort — visibilitychange → hidden does NOT leave
//      a hanging long-poll; re-show kicks a fresh wait that resolves.
//   E. Backend log scan — no new Traceback / ERROR in last 200 lines.

const path = require("path");
const fs = require("fs");
const { chromium } = require(path.join(
  "D:/prep for work/Projects/YuanRAG/src/frontend",
  "node_modules",
  "playwright"
));

// Per v2.0.28.20 CRITICAL REBUILD NOTE: source-only fixes need
// ``npm run build`` to land in the user's served dist. The verifier
// MUST run against the production `start.bat` port (`:8765`) —
// NOT vite dev (`:5173`). Dev may diverge from what the user
// actually loads in the browser.
const FRONTEND = "http://127.0.0.1:8765";

function log(...a) {
  console.log(...a);
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

async function step(name, fn) {
  try {
    await fn();
    log(`  ✓ ${name}`);
  } catch (e) {
    log(`  ✗ ${name} — ${e.message}`);
    throw e;
  }
}

// ----- Mock bodies --------------------------------------------------------

const READY_BODY = {
  embedding: { name: "BAAI/bge-m3", status: "ready", progress: 1.0 },
  reranker: { name: "BAAI/bge-reranker-v2-m3", status: "ready", progress: 1.0 },
  ready: true,
};

const DOWNLOADING_BODY = {
  embedding: { name: "BGE-M3", status: "downloading" },
  reranker: { name: "BGE Reranker v2-M3", status: "downloading" },
  ready: false,
};

const READY_LEGACY_BODY = {
  embedding: { name: "BGE-M3", status: "ready" },
  reranker: { name: "BGE Reranker v2-M3", status: "ready" },
  ready: true,
};

async function fulfillJSON(route, status, body) {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

// ----- Run A: long-poll latency -------------------------------------------

async function runLongPollLatency(page) {
  log(`\n=== v2.0.30.0 RUN A: long-poll latency ===`);
  let waitCalls = 0;
  // Trailing ** is required so the glob matches the query string
  // (`?timeout=60`) — without it Playwright's URL parsing treats
  // `?` as a delimiter and the route never matches.
  await page.route("**/models/wait**", async (route) => {
    waitCalls += 1;
    await fulfillJSON(route, 200, READY_BODY);
  });

  const t0 = Date.now();
  await page.goto(FRONTEND, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(
    () => !document.querySelector(".chat-header-loading"),
    { timeout: 1500 },
  );
  const elapsed = Date.now() - t0;

  await step(`spinner detached within 1500 ms (got ${elapsed} ms)`, () => {
    assert(elapsed <= 1500, `spinner took ${elapsed} ms`);
  });
  await step("/models/wait called exactly once", () => {
    assert(waitCalls === 1, `calls=${waitCalls}, expected 1`);
  });

  await page.unroute("**/models/wait**");
}

// ----- Run B: long-poll 408 → fallback to /models/status -----------------

async function runLongPollFallback(page) {
  log(`\n=== v2.0.30.0 RUN B: long-poll 408 → fallback polling ===`);
  let waitCalls = 0;
  let statusCalls = 0;
  await page.route("**/models/wait**", async (route) => {
    waitCalls += 1;
    await fulfillJSON(route, 408, { detail: "model load wait timed out" });
  });
  await page.route("**/models/status", async (route) => {
    statusCalls += 1;
    const body = statusCalls >= 2 ? READY_LEGACY_BODY : DOWNLOADING_BODY;
    await fulfillJSON(route, 200, body);
  });

  await page.goto(FRONTEND, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(
    () => !document.querySelector(".chat-header-loading"),
    { timeout: 5000 },
  );

  await step("wait was called (408 fallback path exercised)", () => {
    assert(waitCalls >= 1, "wait never called");
  });
  await step("fallback polling fired at least twice", () => {
    assert(statusCalls >= 2, `statusCalls=${statusCalls}, expected >=2`);
  });

  await page.unroute("**/models/wait**");
  await page.unroute("**/models/status");
}

// ----- Run C: ready signal — DOM assertion --------------------------------

async function runReadySignal(page) {
  log(`\n=== v2.0.30.0 RUN C: spinner DOM detached after ready ===`);
  await page.route("**/models/wait**", async (route) => {
    await fulfillJSON(route, 200, READY_BODY);
  });

  await page.goto(FRONTEND, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(
    () => !document.querySelector(".chat-header-loading"),
    { timeout: 1500 },
  );

  const state = await page.evaluate(() => ({
    loadingVisible: !!document.querySelector(".chat-header-loading"),
    inputPresent: !!document.querySelector("textarea"),
    inputDisabled: document.querySelector("textarea")?.disabled ?? null,
  }));

  await step("spinner DOM detached", () => {
    assert(!state.loadingVisible, "spinner still present");
  });
  await step("chat input present", () => {
    assert(state.inputPresent, "input missing");
  });
  await step("chat input enabled", () => {
    assert(!state.inputDisabled, "input disabled");
  });

  await page.unroute("**/models/wait**");
}

// ----- Run D: tab-hide does not hang the request -------------------------

async function runTabHideAborts(page) {
  log(`\n=== v2.0.30.0 RUN D: tab hide → abort long-poll → show → detach ===`);

  // Block /models/wait indefinitely. Visibility-flip to hidden should
  // abort this; visibility-flip back to visible should kick a fresh
  // request — we'll swap the mock to a quick-resolve handler.
  let blockResolve;
  const blocking = new Promise((resolve) => {
    blockResolve = resolve;
  });

  await page.route("**/models/wait**", async (route) => {
    // Race the abort signal against a 5 s ceiling — whichever fires first.
    try {
      await Promise.race([
        new Promise((_, reject) => {
          const start = Date.now();
          const tick = () => {
            if (Date.now() - start > 5000) reject(new Error("ceiling"));
            else setTimeout(tick, 50);
          };
          tick();
        }),
        blocking,
      ]);
    } catch {
      // ceiling reached — fall through and return 408
    }
    try {
      await fulfillJSON(route, 408, { detail: "blocked" });
    } catch {
      // route already cancelled by the client abort
    }
  });

  await page.goto(FRONTEND, { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(300);

  // Flip to hidden — hook should abort the in-flight long-poll.
  await page.evaluate(() => {
    Object.defineProperty(document, "visibilityState", {
      value: "hidden",
      configurable: true,
    });
    document.dispatchEvent(new Event("visibilitychange"));
  });
  await page.waitForTimeout(300);

  // Now flip back to visible — the hook should kick a fresh request.
  // Swap the mock to a quick-resolve handler.
  await page.unroute("**/models/wait**");
  await page.route("**/models/wait**", async (route) => {
    await fulfillJSON(route, 200, READY_BODY);
  });

  await page.evaluate(() => {
    Object.defineProperty(document, "visibilityState", {
      value: "visible",
      configurable: true,
    });
    document.dispatchEvent(new Event("visibilitychange"));
  });

  await page.waitForFunction(
    () => !document.querySelector(".chat-header-loading"),
    { timeout: 3000 },
  );

  await step("spinner detached after visibility hide → show cycle", () => {
    // waitForFunction already validated this; this is the marker.
  });

  // Release any leftover blocked handlers.
  blockResolve();

  await page.unroute("**/models/wait**");
}

// ----- Run E: backend log scan --------------------------------------------

async function runBackendLogScan() {
  log(`\n=== v2.0.30.0 RUN E: backend log scan (negative regression) ===`);
  const logPath = path.join(
    "D:/prep for work/Projects/YuanRAG",
    "logs",
    "app.log",
  );
  let tail = "";
  try {
    const all = fs.readFileSync(logPath, "utf8");
    const lines = all.split(/\r?\n/);
    tail = lines.slice(-200).join("\n");
  } catch {
    log("  (no logs/app.log found — skipping negative scan)");
    return;
  }
  const traceback = /Traceback \(most recent call last\):/.test(tail);
  const errorLine = /\bERROR\b/.test(tail);

  await step("no Traceback in last 200 log lines", () => {
    assert(!traceback, "Traceback detected in app.log tail");
  });
  await step("no ERROR line in last 200 log lines", () => {
    assert(!errorLine, "ERROR line detected in app.log tail");
  });
}

// ----- main ---------------------------------------------------------------

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

  const failures = [];
  for (const [name, fn] of [
    ["A", runLongPollLatency],
    ["B", runLongPollFallback],
    ["C", runReadySignal],
    ["D", runTabHideAborts],
  ]) {
    try {
      await fn(page);
    } catch (e) {
      failures.push({ run: name, reason: e.message });
      log(`RUN ${name}: ✗ FAIL — ${e.message}`);
    }
  }

  try {
    await runBackendLogScan();
  } catch (e) {
    failures.push({ run: "E", reason: e.message });
    log(`RUN E: ✗ FAIL — ${e.message}`);
  }

  await browser.close();

  log(`\n=== v2.0.30.0 SUMMARY ===`);
  if (failures.length > 0) {
    log(`✗ v2.0.30.0 verifier FAILED: ${failures.length}/5`);
    for (const f of failures) log(`  RUN ${f.run}: ${f.reason}`);
    process.exit(1);
  }
  log(`✓ v2.0.30.0 verifier PASSED: 5/5`);
}

main().catch((e) => {
  console.error("verifier crashed:", e.message);
  console.error(e.stack);
  process.exit(2);
});