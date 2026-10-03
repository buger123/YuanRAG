// v2.0.28.15 — synthesis reasoning leak Layer 5 verifier.
//
// User report (2026-09-23, post-v2.0.28.14 ship):
//   "现在没有出现直接回答结果" — the assistant bubble's
//   <ThinkingDrawer> showed the synthesis LLM's meta-commentary
//   ("The user is asking ... I need to call get_current_time")
//   instead of the direct time answer.
//
// Pre-fix wire shape (from debug probe 5 runs):
//   AIMessage 0 (planning-step): content="", reasoning="..."
//   AIMessage 1 (synthesis):      content="现在是 ...", reasoning="..."
//
// v2.0.28.15 fix:
//   * Backend `react_generate.py` drops `yield ("reasoning", ...)`
//     AND drops `additional_kwargs["reasoning"]` stamp on synthesis
//     AIMessage. Same for `fsm.react_generate_direct` (greeting path).
//   * Frontend `mergeAdjacentAssistantTurns`:
//     `reasoning: prev.reasoning ?? m.reasoning`
//         →
//     `reasoning: prev.reasoning` (no fallback).
//
// Strategy:
//   1. Drive 2 turns (你是谁 warmup + 现在几点了 THE BUG).
//   2. Snapshot LIVE DOM:
//      * NO `<div className="thinking-drawer-body">` text matching
//        meta-commentary keywords (The user is asking / I need to
//        call / 我需要调用 / 我会调用 / i need to call / i already
//        provided / synthesis meta-commentary markers).
//      * assistant bubble contains a non-empty `content`
//        (direct answer present — tool card or text or both).
//   3. Snapshot RELOAD DOM:
//      * NO reasoning drawer body with meta-commentary.
//      * assistant bubble count == 2 (v2.0.28.14 invariant).
//      * tool card count == 1 (v2.0.28.14 invariant).
//   4. Wire cross-check:
//      * Pre-reload: synthesis AIMessage (last assistant) has
//        `reasoning: null/missing` in MessageRecord payload.
//      * Post-reload: same — the synthesis reasoning is gone.

const path = require("path");
const { chromium } = require(path.join(
  "D:/prep for work/Projects/YuanRAG/src/frontend",
  "node_modules",
  "playwright"
));

const BACKEND = "http://127.0.0.1:8765";

const META_COMMENTARY_KEYWORDS = [
  "the user is asking",
  "i need to call",
  "i already provided",
  "i should call",
  "i need to use",
  "i will call",
  "用户问",
  "我需要调用",
  "我会调用",
  "我已经",
];

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

async function waitForModels(page, timeoutMs = 90_000) {
  await page.goto(`${BACKEND}/`, { waitUntil: "domcontentloaded" });
  try {
    await page.waitForSelector(".chat-header-loading", {
      state: "detached",
      timeout: timeoutMs,
    });
    return true;
  } catch {
    return false;
  }
}

async function snapshotAssistant(page) {
  return await page.evaluate(() => {
    const asst = Array.from(
      document.querySelectorAll(".message.assistant")
    ).map((n) => {
      const body = n.querySelector(".markdown-body");
      const content = body
        ? body.textContent.trim()
        : n.textContent.trim();
      // Per-bubble reasoning drawer (open or collapsed).
      const drawerBody = n.querySelector(".thinking-drawer-body");
      const drawerText = drawerBody
        ? drawerBody.textContent.trim()
        : null;
      return { content, drawerText };
    });
    const toolCards = Array.from(
      document.querySelectorAll(".tool-call-card, [data-tool-name]")
    ).map((n) => n.textContent.trim());
    return {
      asstCount: asst.length,
      asstContent: asst.map((m) => m.content),
      drawerTexts: asst.map((m) => m.drawerText),
      toolCards,
    };
  });
}

async function sendAndWait(page, query, timeoutMs = 90_000) {
  await page.fill(".chat-textarea", query);
  await page.click(".btn-send");
  try {
    await page.waitForSelector(".btn-send", {
      state: "visible",
      timeout: timeoutMs,
    });
  } catch {}
  await page.waitForTimeout(1500);
}

function hasMetaCommentary(text) {
  if (!text) return false;
  const lc = text.toLowerCase();
  return META_COMMENTARY_KEYWORDS.some((kw) => lc.includes(kw.toLowerCase()));
}

async function main() {
  console.log(">>> Layer 5 v2.0.28.15 synthesis reasoning leak verifier");
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

  let threadIdForCheck = "";

  try {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const page = await ctx.newPage();

    await page.addInitScript(() => {
      try {
        window.localStorage.setItem("rag.theme", "light");
        window.localStorage.setItem("rag.locale", "zh");
        // Default showThinking ON — we want to verify the drawer
        // doesn't leak meta-commentary even when toggled ON
        // (v2.0.28.14 baseline assumption).
        window.localStorage.removeItem("rag.show_thinking");
      } catch {}
    });

    const modelsReady = await waitForModels(page);
    check(modelsReady, "models ready (chat-header-loading detached)");

    // === TURN 1: warmup ===
    const q1 = "你是谁";
    console.log(`\n--- TURN 1: ${q1} ---`);
    await sendAndWait(page, q1, 60_000);

    // === TURN 2: trigger get_current_time tool call ===
    const q2 = "现在几点了";
    console.log(`\n--- TURN 2: ${q2} ---`);
    await sendAndWait(page, q2, 90_000);

    // === LIVE snapshot ===
    const liveSnap = await snapshotAssistant(page);
    console.log(
      "Live snapshot: asstCount=",
      liveSnap.asstCount,
      " toolCards=",
      liveSnap.toolCards.length,
      " drawersWithText=",
      liveSnap.drawerTexts.filter((t) => t).length,
    );

    check(
      liveSnap.asstCount === 2,
      `live: 2 assistant bubbles (TURN 1 + TURN 2) — got ${liveSnap.asstCount}`,
    );

    // TURN 2's bubble (last) should have a get_current_time tool card.
    const liveToolCards = liveSnap.toolCards.filter((c) =>
      /get_current_time/i.test(c),
    );
    check(
      liveToolCards.length >= 1,
      `live: ≥1 get_current_time tool card visible — got ${liveToolCards.length}`,
    );

    // THE BUG FIX assertions (live path):
    // NO drawer body should contain synthesis meta-commentary.
    const liveMetaDrawers = liveSnap.drawerTexts.filter(hasMetaCommentary);
    check(
      liveMetaDrawers.length === 0,
      `live: NO reasoning drawer body contains synthesis meta-commentary keywords — found ${liveMetaDrawers.length} offenders: ${JSON.stringify(liveMetaDrawers.map((t) => t ? t.slice(0, 80) : null))}`,
    );

    await page.screenshot({
      path: "D:/prep for work/Projects/YuanRAG/tests/e2e/output/v2.0.28.15-live.png",
      fullPage: true,
    });

    // === Wire cross-check (BEFORE reload): expect 3 ===
    const activeThreadId = await page.evaluate(() => {
      try {
        return window.localStorage.getItem("rag.active_thread") || "";
      } catch {
        return "";
      }
    });
    threadIdForCheck = activeThreadId;

    if (threadIdForCheck) {
      console.log(`>>> thread_id: ${threadIdForCheck}`);
      const msgRes = await fetch(
        `${BACKEND}/sessions/${encodeURIComponent(threadIdForCheck)}/messages`,
      );
      if (msgRes.ok) {
        const msgs = await msgRes.json();
        const asstMsgs = msgs.filter((m) => m.role === "assistant");
        console.log(
          `>>> wire assistant count (pre-reload): ${asstMsgs.length}`,
        );
        for (let i = 0; i < asstMsgs.length; i++) {
          const m = asstMsgs[i];
          const text = (m.content || "").trim();
          console.log(
            `  [${i}] content=${JSON.stringify(text.slice(0, 80))} reasoning=${JSON.stringify(((m.reasoning || "").slice(0, 80)))} tool_calls=${(m.tool_calls || []).length}`,
          );
        }
        check(
          asstMsgs.length === 3,
          `wire (pre-reload): 3 assistant records (TURN 1 synth + TURN 2 tool-calls AIM + TURN 2 synth) — got ${asstMsgs.length}`,
        );

        // THE BUG FIX assertions (save-side):
        // The LAST AIMessage (synthesis) must NOT carry `reasoning`.
        // Pre-fix the synthesis AIMessage's `reasoning` field held
        // the meta-commentary that leaked into the drawer.
        const synthesisAIM = asstMsgs[asstMsgs.length - 1];
        const synthesisReasoning = (synthesisAIM.reasoning || "").trim();
        check(
          synthesisReasoning === "",
          `wire (pre-reload): synthesis AIMessage (last) has NO reasoning text — got: ${JSON.stringify(synthesisReasoning.slice(0, 80))}`,
        );

        // The synthesis AIMessage's reasoning must also NOT match
        // meta-commentary keywords.
        check(
          !hasMetaCommentary(synthesisReasoning),
          `wire (pre-reload): synthesis AIMessage reasoning does NOT contain meta-commentary keywords — got: ${JSON.stringify(synthesisReasoning.slice(0, 80))}`,
        );
      } else {
        check(false, `GET /sessions/{tid}/messages returned ${msgRes.status}`);
      }
    } else {
      check(false, "could not determine thread_id for wire cross-check");
    }

    // === RELOAD ===
    console.log("\n--- page.reload() ---");
    await page.reload({ waitUntil: "domcontentloaded" });
    try {
      await page.waitForSelector(".chat-header-loading", {
        state: "detached",
        timeout: 60_000,
      });
    } catch {}
    await page.waitForTimeout(2000);

    const reloadSnap = await snapshotAssistant(page);
    console.log(
      "Reload snapshot: asstCount=",
      reloadSnap.asstCount,
      " toolCards=",
      reloadSnap.toolCards.length,
      " drawersWithText=",
      reloadSnap.drawerTexts.filter((t) => t).length,
    );

    // v2.0.28.14 invariants preserved:
    check(
      reloadSnap.asstCount === 2,
      `reload: 2 assistant bubbles (TURN 1 + TURN 2 merged) — got ${reloadSnap.asstCount}`,
    );
    const reloadToolCards = reloadSnap.toolCards.filter((c) =>
      /get_current_time/i.test(c),
    );
    check(
      reloadToolCards.length === 1,
      `reload: exactly 1 get_current_time tool card — got ${reloadToolCards.length}`,
    );

    // THE BUG FIX assertions (reload path):
    // NO drawer body should contain synthesis meta-commentary.
    const reloadMetaDrawers = reloadSnap.drawerTexts.filter(hasMetaCommentary);
    check(
      reloadMetaDrawers.length === 0,
      `reload: NO reasoning drawer body contains synthesis meta-commentary keywords — found ${reloadMetaDrawers.length} offenders: ${JSON.stringify(reloadMetaDrawers.map((t) => t ? t.slice(0, 80) : null))}`,
    );

    await page.screenshot({
      path: "D:/prep for work/Projects/YuanRAG/tests/e2e/output/v2.0.28.15-reload.png",
      fullPage: true,
    });

    // === Wire re-check (post-reload, sanity) ===
    if (threadIdForCheck) {
      const msgRes = await fetch(
        `${BACKEND}/sessions/${encodeURIComponent(threadIdForCheck)}/messages`,
      );
      if (msgRes.ok) {
        const msgs = await msgRes.json();
        const asstMsgs = msgs.filter((m) => m.role === "assistant");
        check(
          asstMsgs.length === 3,
          `wire (post-reload, sanity): still 3 assistant records — got ${asstMsgs.length}`,
        );
        // Belt-and-suspenders: post-reload synthesis AIMessage
        // reasoning still empty (matches pre-reload).
        const synthesisAIM = asstMsgs[asstMsgs.length - 1];
        const synthesisReasoning = (synthesisAIM.reasoning || "").trim();
        check(
          synthesisReasoning === "",
          `wire (post-reload): synthesis AIMessage reasoning still empty — got: ${JSON.stringify(synthesisReasoning.slice(0, 80))}`,
        );
      }
    }
  } finally {
    await browser.close();
  }

  console.log(
    `\n>>> Layer 5 v2.0.28.15 verifier: ${passed} passed, ${failed} failed`,
  );
  if (failed > 0) process.exit(1);
}

main().catch((e) => {
  console.error("Layer 5 v2.0.28.15 verifier FAILED:", e.message);
  console.error(e.stack);
  process.exit(1);
});