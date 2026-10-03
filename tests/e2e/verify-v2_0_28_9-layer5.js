// v2.0.28.9 — multi-turn replay Layer 5 verifier.
// Regression guard for the bug where each thread only showed its latest
// Q&A pair on page reload (prior turns were clobbered by INSERT OR REPLACE).
//
// Strategy: drive a real chromium session through 2 turns on the same
// thread, then reload the page (simulating browser refresh) and assert
// that BOTH Q&A pairs are still visible. If the runner doesn't hydrate
// prior state, only the latest pair survives the reload.

const path = require("path");
const { chromium } = require(path.join(
  "D:/prep for work/Projects/YuanRAG/src/frontend",
  "node_modules",
  "playwright"
));

const BACKEND = "http://127.0.0.1:8765";

async function backendReady() {
  try {
    const resp = await fetch(`${BACKEND}/models/status`);
    if (!resp.ok) return false;
    const body = await resp.json();
    return body.ready === true;
  } catch {
    return false;
  }
}

function uuidv4() {
  // RFC 4122 v4 — good enough for a thread_id
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

async function main() {
  console.log(">>> Layer 5 v2.0.28.9 multi-turn replay verifier");
  console.log(`>>> Backend: ${BACKEND}`);

  if (!(await backendReady())) {
    console.log("!!! Backend not ready. Layer 5 needs live env — skipping.");
    process.exit(0);
  }
  console.log("  ✓ Backend ready");

  // We use a controlled thread_id (UUID) so we can find this thread in
  // /sessions after the reload. The app auto-creates threads via WS
  // accept when the client connects; we let the UI do that.
  const threadIdHint = uuidv4();
  const browser = await chromium.launch({ headless: true });
  let passed = 0;
  let failed = 0;
  function check(cond, msg) {
    if (cond) {
      console.log(`  ✓ ${msg}`);
      passed++;
    } else {
      console.log(`  ✗ ${msg}`);
      failed++;
    }
  }

  try {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const page = await ctx.newPage();

    // Pre-seed localStorage so the first render picks up our thread_id
    // — this matches the React app's "current thread" pattern where
    // App.tsx generates a thread_id on mount and persists to
    // localStorage.rag.currentThread.
    await page.addInitScript((tid) => {
      try {
        window.localStorage.setItem("rag.theme", "light");
        window.localStorage.setItem("rag.locale", "zh");
      } catch {}
    }, threadIdHint);

    await page.goto(`${BACKEND}/`, { waitUntil: "domcontentloaded" });
    // Wait for models ready.
    try {
      await page.waitForSelector(".chat-header-loading", {
        state: "detached",
        timeout: 60_000,
      });
    } catch {
      check(false, "models ready (chat-header-loading detached within 60s)");
      throw new Error("backend models never reached ready");
    }
    check(true, "models ready (chat-header-loading detached within 60s)");

    // TURN 1 — first user message
    const q1 = "用一句话回答: 1+1 等于几?";
    await page.fill(".chat-textarea", q1);
    await page.click(".btn-send");
    try {
      await page.waitForSelector(
        ".message.assistant .markdown-body >> nth=0",
        { state: "visible", timeout: 30_000 }
      );
    } catch {
      check(false, "TURN 1 assistant bubble appeared");
      throw new Error("TURN 1 never produced assistant bubble");
    }
    check(true, "TURN 1 assistant bubble appeared");
    await page.waitForSelector(".btn-send", {
      state: "visible",
      timeout: 60_000,
    });
    check(true, "TURN 1 send button re-visible (streaming done)");

    // TURN 2 — second user message
    const q2 = "再回答: 2+2 等于几?";
    await page.fill(".chat-textarea", q2);
    await page.click(".btn-send");
    try {
      await page.waitForSelector(
        ".message.assistant .markdown-body >> nth=1",
        { state: "visible", timeout: 30_000 }
      );
    } catch {
      check(false, "TURN 2 second assistant bubble appeared");
      throw new Error("TURN 2 never produced second assistant bubble");
    }
    check(true, "TURN 2 second assistant bubble appeared");
    await page.waitForSelector(".btn-send", {
      state: "visible",
      timeout: 60_000,
    });
    check(true, "TURN 2 send button re-visible (streaming done)");

    // Snapshot the thread_id the app is using (read from the URL
    // hash, the sidebar's active-thread indicator, or fall back to
    // whatever /sessions returns for the most recent thread). Most
    // apps store it in URL or in a data attribute.
    const activeThreadId = await page.evaluate(() => {
      // Try URL hash first (#thread=<id>)
      const hash = window.location.hash || "";
      const m = /thread[=:]([a-zA-Z0-9-]+)/.exec(hash);
      if (m) return m[1];
      // Try localStorage.rag.currentThread
      try {
        return window.localStorage.getItem("rag.currentThread") || "";
      } catch {
        return "";
      }
    });

    // Pre-reload sanity: how many user messages are in the DOM?
    const preReloadUserCount = await page.locator(".message.user").count();
    const preReloadAssistantCount = await page.locator(".message.assistant").count();
    check(
      preReloadUserCount === 2,
      `pre-reload: 2 user messages in DOM (got ${preReloadUserCount})`
    );
    check(
      preReloadAssistantCount === 2,
      `pre-reload: 2 assistant messages in DOM (got ${preReloadAssistantCount})`
    );

    // Reload the page — the frontend will fetch /sessions/{tid}/messages
    // and re-hydrate. If the backend didn't accumulate properly, only
    // TURN 2 survives.
    await page.reload({ waitUntil: "domcontentloaded" });
    try {
      await page.waitForSelector(".chat-header-loading", {
        state: "detached",
        timeout: 60_000,
      });
    } catch {}
    // Give the page a moment to fetch history.
    await page.waitForTimeout(2000);

    // Post-reload: 2 user + 2 assistant messages should still be visible.
    const postReloadUserCount = await page.locator(".message.user").count();
    const postReloadAssistantCount = await page
      .locator(".message.assistant")
      .count();
    check(
      postReloadUserCount === 2,
      `post-reload: 2 user messages in DOM (got ${postReloadUserCount}) — if 1, prior turns were clobbered by INSERT OR REPLACE`
    );
    check(
      postReloadAssistantCount === 2,
      `post-reload: 2 assistant messages in DOM (got ${postReloadAssistantCount}) — if 1, prior turns were clobbered by INSERT OR REPLACE`
    );

    // Verify the actual text content of Q1 + Q2 is still visible
    const allText = await page.locator(".chat-pane").innerText();
    check(
      allText.includes("1+1") || allText.includes("一加一"),
      `post-reload: TURN 1 question text still visible`
    );
    check(
      allText.includes("2+2") || allText.includes("二加二"),
      `post-reload: TURN 2 question text still visible`
    );

    // Wire-level cross-check: GET /sessions and find the thread we
    // just used; GET /sessions/{tid}/messages must return 4 records
    // (2 user + 2 assistant).
    let threadIdForCheck = activeThreadId;
    if (!threadIdForCheck) {
      // Fallback: pick the most recently created thread from /sessions.
      const sessRes = await fetch(`${BACKEND}/sessions`);
      const sessions = await sessRes.json();
      if (Array.isArray(sessions) && sessions.length > 0) {
        sessions.sort((a, b) => {
          const ta = new Date(a.created_at || 0).getTime();
          const tb = new Date(b.created_at || 0).getTime();
          return tb - ta;
        });
        threadIdForCheck = sessions[0].thread_id || sessions[0].id;
      }
    }
    if (threadIdForCheck) {
      const msgRes = await fetch(
        `${BACKEND}/sessions/${encodeURIComponent(threadIdForCheck)}/messages`
      );
      if (msgRes.ok) {
        const msgs = await msgRes.json();
        const userMsgs = msgs.filter((m) => m.role === "user");
        const asstMsgs = msgs.filter((m) => m.role === "assistant");
        check(
          userMsgs.length === 2,
          `GET /sessions/{tid}/messages: 2 user records (got ${userMsgs.length})`
        );
        check(
          asstMsgs.length === 2,
          `GET /sessions/{tid}/messages: 2 assistant records (got ${asstMsgs.length})`
        );
      } else {
        check(
          false,
          `GET /sessions/{tid}/messages returned ${msgRes.status}`
        );
      }
    } else {
      check(
        false,
        "could not determine thread_id for wire-level cross-check"
      );
    }

    await page.screenshot({
      path: "D:/prep for work/Projects/YuanRAG/tests/e2e/output/v2.0.28.9-multiturn.png",
      fullPage: true,
    });
    await ctx.close();
  } finally {
    await browser.close();
  }

  console.log(
    `\n>>> Layer 5 v2.0.28.9 verifier: ${passed} passed, ${failed} failed`
  );
  if (failed > 0) process.exit(1);
}

main().catch((e) => {
  console.error("Layer 5 v2.0.28.9 verifier FAILED:", e.message);
  console.error(e.stack);
  process.exit(1);
});
