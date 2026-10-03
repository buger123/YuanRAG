// v2.0.28.10 — live-vs-reload assistant count parity Layer 5 verifier.
//
// Regression guard for the bug where the live UI's assistant bubble
// count didn't match the reload's assistant bubble count. Pre-fix
// the FSM saved 2 AIMessages per turn (react_agent's text-only
// "draft" + react_generate's "synthesis") but the frontend's
// canonical-overwrite (answerCompleteHandler.ts:116-143) hid the
// draft live. Reload re-rendered the full message list, exposing
// the hidden draft → "extra" assistant bubbles that were never
// seen live.
//
// Strategy: drive 2 turns on the same thread, snapshot live
// assistant count after streaming completes, reload, snapshot
// reload assistant count, assert EQUALITY (live count == reload
// count). Pre-fix this fails (live=2, reload=4); post-fix both
// should be 2.

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

async function snapshotDom(page) {
  return await page.evaluate(() => {
    const userMsgs = Array.from(
      document.querySelectorAll(".message.user .message-text")
    ).map((n) => n.textContent);
    const assistantMsgs = Array.from(
      document.querySelectorAll(".message.assistant")
    ).map((n) => {
      const body = n.querySelector(".markdown-body");
      return body ? body.textContent.trim() : n.textContent.trim();
    });
    const isStreaming = document.querySelector(".thinking-status") !== null;
    const sendBtnVisible = document.querySelector(".btn-send") !== null;
    const stopBtnVisible = document.querySelector(".btn-stop") !== null;
    return {
      userCount: userMsgs.length,
      assistantCount: assistantMsgs.length,
      userContent: userMsgs,
      assistantContent: assistantMsgs,
      isStreaming,
      sendBtnVisible,
      stopBtnVisible,
    };
  });
}

async function main() {
  console.log(">>> Layer 5 v2.0.28.10 live-vs-reload assistant count parity");
  console.log(`>>> Backend: ${BACKEND}`);

  if (!(await backendReady())) {
    console.log("!!! Backend not ready. Layer 5 needs live env — skipping.");
    process.exit(0);
  }
  console.log("  ✓ Backend ready");

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

    await page.addInitScript(() => {
      try {
        window.localStorage.setItem("rag.theme", "light");
        window.localStorage.setItem("rag.locale", "zh");
      } catch {}
    });

    await page.goto(`${BACKEND}/`, { waitUntil: "domcontentloaded" });
    try {
      await page.waitForSelector(".chat-header-loading", {
        state: "detached",
        timeout: 60_000,
      });
    } catch {
      check(false, "models ready (chat-header-loading detached within 60s)");
      throw new Error("backend models never reached ready");
    }
    check(true, "models ready");

    // === TURN 1 ===
    console.log("\n--- TURN 1: send Q1 ---");
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
    check(true, "TURN 1 assistant bubble appeared live");
    await page.waitForSelector(".btn-send", {
      state: "visible",
      timeout: 60_000,
    });
    check(true, "TURN 1 send button re-visible (streaming done)");

    const r1 = await snapshotDom(page);
    console.log("After TURN 1 (live):", JSON.stringify(r1, null, 2));
    check(
      r1.userCount === 1,
      `TURN 1 live: 1 user message (got ${r1.userCount})`
    );
    check(
      r1.assistantCount === 1,
      `TURN 1 live: 1 assistant message (got ${r1.assistantCount})`
    );

    // === TURN 2 ===
    console.log("\n--- TURN 2: send Q2 ---");
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
    check(true, "TURN 2 second assistant bubble appeared live");
    await page.waitForSelector(".btn-send", {
      state: "visible",
      timeout: 60_000,
    });
    check(true, "TURN 2 send button re-visible (streaming done)");

    const r2 = await snapshotDom(page);
    console.log("After TURN 2 (live):", JSON.stringify(r2, null, 2));

    // Pre-reload: 2 users + 2 assistants (1 per turn)
    check(
      r2.userCount === 2,
      `TURN 2 live: 2 user messages (got ${r2.userCount})`
    );
    check(
      r2.assistantCount === 2,
      `TURN 2 live: 2 assistant messages (got ${r2.assistantCount}) — pre-fix would be 2; regression would be more`
    );

    // === Snapshot the live state ===
    const liveAssistantCount = r2.assistantCount;
    const liveAssistantContent = r2.assistantContent;

    // Screenshot live state
    await page.screenshot({
      path: "D:/prep for work/Projects/YuanRAG/tests/e2e/output/v2.0.28.10-live.png",
      fullPage: true,
    });

    // === RELOAD ===
    console.log("\n--- RELOAD PAGE ---");
    await page.reload({ waitUntil: "domcontentloaded" });
    try {
      await page.waitForSelector(".chat-header-loading", {
        state: "detached",
        timeout: 60_000,
      });
    } catch {}
    await page.waitForTimeout(2000);

    const r3 = await snapshotDom(page);
    console.log("After reload:", JSON.stringify(r3, null, 2));

    // Screenshot reload state
    await page.screenshot({
      path: "D:/prep for work/Projects/YuanRAG/tests/e2e/output/v2.0.28.10-reload.png",
      fullPage: true,
    });

    // === THE CRITICAL ASSERTION: live assistant count == reload assistant count ===
    check(
      r3.userCount === 2,
      `post-reload: 2 user messages (got ${r3.userCount}) — if 1, hydration broken (v2.0.28.9 regression)`
    );
    check(
      r3.assistantCount === liveAssistantCount,
      `POST-RELOAD ASSISTANT COUNT == LIVE ASSISTANT COUNT — got live=${liveAssistantCount}, reload=${r3.assistantCount} (pre-fix: live=2 reload=4 due to FSM saving 2 AIMessages per turn)`
    );

    // Both counts must equal 2 (1 per turn, NOT 2 per turn)
    check(
      r3.assistantCount === 2,
      `post-reload: 2 assistant messages (got ${r3.assistantCount}) — if 4, FSM is appending react_agent's draft + react_generate's synthesis (pre-fix shape)`
    );

    // Verify content actually preserved
    const allText = await page.locator(".chat-pane").innerText();
    check(
      allText.includes("1+1") || allText.includes("一加一"),
      `post-reload: TURN 1 question text still visible`
    );
    check(
      allText.includes("2+2") || allText.includes("二加二"),
      `post-reload: TURN 2 question text still visible`
    );

    // === Wire-level cross-check via /sessions/{tid}/messages ===
    // Read the active thread_id from localStorage or URL hash.
    const activeThreadId = await page.evaluate(() => {
      try {
        return window.localStorage.getItem("rag.currentThread") || "";
      } catch {
        return "";
      }
    });
    let threadIdForCheck = activeThreadId;
    if (!threadIdForCheck) {
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
          `GET /sessions/{tid}/messages: 2 assistant records (got ${asstMsgs.length}) — if 4, FSM 2-AIMessage bug present in DB`
        );
      } else {
        check(
          false,
          `GET /sessions/{tid}/messages returned ${msgRes.status}`
        );
      }
    } else {
      check(false, "could not determine thread_id for wire cross-check");
    }

    await ctx.close();
  } finally {
    await browser.close();
  }

  console.log(
    `\n>>> Layer 5 v2.0.28.10 verifier: ${passed} passed, ${failed} failed`
  );
  if (failed > 0) process.exit(1);
}

main().catch((e) => {
  console.error("Layer 5 v2.0.28.10 verifier FAILED:", e.message);
  console.error(e.stack);
  process.exit(1);
});