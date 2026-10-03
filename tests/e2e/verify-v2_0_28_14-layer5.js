// v2.0.28.14 — double-bubble-on-reload Layer 5 verifier.
//
// User report (2026-09-23, post-v2.0.28.13 ship):
//   "刷新之后变成这样了... 还是不对,出现了两次工具调用"
//
// After reload of a "现在几点了" thread, two assistant bubbles
// appear per ReAct turn:
//   1. Top: planning-step AIMessage bubble (reasoning +
//      get_current_time tool card)
//   2. Bottom: synthesis AIMessage bubble (tool card with timing
//      + final answer)
//
// v2.0.28.13 fixed the ⏳-pending-forever symptom (both cards
// now show ✓ complete). v2.0.28.14 collapses the two bubbles
// into one on reload (frontend-only fix — see plan file).
//
// Live streaming already shows 1 bubble (answerCompleteHandler
// applies canonical-overwrite at line 116-128). Reload previously
// showed 2 because loadHistory mapped wire records 1:1.
//
// Strategy:
//   1. Open real Chromium, drive 2-turn conversation (warmup +
//      "现在几点了" which exercises the v2.0.28.10 2-AIMessage
//      contract).
//   2. Snapshot live DOM: assert .message.assistant count == 2
//      (TURN 1 + TURN 2 each as 1 bubble, live already correct).
//   3. Wire cross-check: GET /sessions/{tid}/messages — assert
//      assistant records == 3 (TURN 1 synthesis + TURN 2
//      planning-step + TURN 2 synthesis; DB shape preserved).
//   4. page.reload() → wait for historyLoaded.
//   5. Snapshot reload DOM: assert .message.assistant count == 2
//      (was 3 pre-fix: TURN 1 + TURN 2 planning-step + TURN 2
//      synthesis now collapsed).
//   6. Reload bubble contains: reasoning drawer (thinking trace)
//      + 1 tool card (rich log with timing) + final answer.
//   7. Wire re-check: still 3 assistant records (DB shape
//      unchanged — fix is display-only).

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
      return body ? body.textContent.trim() : n.textContent.trim();
    });
    const toolCards = Array.from(
      document.querySelectorAll(".tool-call-card, [data-tool-name]")
    ).map((n) => n.textContent.trim());
    // Per-bubble reasoning drawer visibility (planning-step AIMessage
    // carries the LLM's thinking trace; on reload the synthesis's
    // content has the final answer, and the merged reasoning from
    // planning-step sits in the merged bubble's drawer).
    const reasoningDrawers = Array.from(
      document.querySelectorAll(".thinking-drawer")
    ).map((n) => n.textContent.trim().slice(0, 200));
    const isStreaming =
      document.querySelector(".thinking-status") !== null;
    const sendBtnVisible = document.querySelector(".btn-send") !== null;
    return {
      asstCount: asst.length,
      asstContent: asst,
      toolCards,
      reasoningDrawers,
      isStreaming,
      sendBtnVisible,
    };
  });
}

async function sendAndWait(page, query, timeoutMs = 90_000) {
  await page.fill(".chat-textarea", query);
  await page.click(".btn-send");
  // Wait for streaming to finish (send button re-appears).
  try {
    await page.waitForSelector(".btn-send", {
      state: "visible",
      timeout: timeoutMs,
    });
  } catch {}
  await page.waitForTimeout(1500);
}

async function main() {
  console.log(">>> Layer 5 v2.0.28.14 double-bubble-on-reload verifier");
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
      " reasoningDrawers=",
      liveSnap.reasoningDrawers.length,
    );

    // Live: 2 turns -> 2 assistant bubbles (live already correct via
    // answerCompleteHandler canonical-overwrite).
    check(
      liveSnap.asstCount === 2,
      `live: 2 assistant bubbles (TURN 1 + TURN 2) — got ${liveSnap.asstCount}`
    );

    // Live: TURN 2 bubble should have a get_current_time tool card.
    // (1 card on the merged live bubble.)
    const liveToolCards = liveSnap.toolCards.filter((c) =>
      /get_current_time/i.test(c)
    );
    check(
      liveToolCards.length >= 1,
      `live: ≥1 get_current_time tool card visible — got ${liveToolCards.length} (cards: ${JSON.stringify(liveSnap.toolCards)})`
    );

    await page.screenshot({
      path: "D:/prep for work/Projects/YuanRAG/tests/e2e/output/v2.0.28.14-live.png",
      fullPage: true,
    });

    // === Wire cross-check (BEFORE reload): expect 3 ===
    // TURN 1: [HM, AIM_synthesis] -> 1 assistant
    // TURN 2: [HM, AIM_planning_step(tools), AIM_synthesis] -> 2 assistant
    // Total: 3. DB shape unchanged (post-v2.0.28.10 contract).
    const activeThreadId = await page.evaluate(() => {
      try {
        return window.localStorage.getItem("rag.active_thread") || "";
      } catch {
        return "";
      }
    });
    threadIdForCheck = activeThreadId;

    let wireAsstCountLive = 0;
    if (threadIdForCheck) {
      console.log(`>>> thread_id: ${threadIdForCheck}`);
      const msgRes = await fetch(
        `${BACKEND}/sessions/${encodeURIComponent(threadIdForCheck)}/messages`
      );
      if (msgRes.ok) {
        const msgs = await msgRes.json();
        const asstMsgs = msgs.filter((m) => m.role === "assistant");
        wireAsstCountLive = asstMsgs.length;
        console.log(`>>> wire assistant count (pre-reload): ${wireAsstCountLive}`);
        for (let i = 0; i < asstMsgs.length; i++) {
          const m = asstMsgs[i];
          const text = (m.content || "").trim();
          console.log(
            `  [${i}] content=${JSON.stringify(text.slice(0, 80))} tool_calls=${(m.tool_calls || []).length}`
          );
        }
        check(
          wireAsstCountLive === 3,
          `wire (pre-reload): 3 assistant records (TURN 1 synth + TURN 2 tool-calls AIM + TURN 2 synth) — got ${wireAsstCountLive}`
        );

        // The TURN 2 tool-calls AIMessage must carry get_current_time.
        // (v2.0.28.10 contract — planning-step AIMessage from react_agent
        // carries native tool_calls. Its content may be a preamble
        // ("好的,我先查询一下...") OR empty depending on the LLM's
        // drafting behavior; both shapes are valid.)
        const turn2ToolCallsAIM = asstMsgs.find(
          (m) =>
            Array.isArray(m.tool_calls) &&
            m.tool_calls.length > 0 &&
            m.tool_calls.some((tc) => tc.name === "get_current_time")
        );
        check(
          !!turn2ToolCallsAIM,
          `wire (pre-reload): at least one AIMessage carries get_current_time tool_calls (proves intent=qa_complex + react_agent ran) — found ${asstMsgs.filter((m) => Array.isArray(m.tool_calls) && m.tool_calls.length > 0).length} tool-calls AIMessage(s)`
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

    // Wait for chat to render history — models loading detached
    // again is the same signal as first load.
    try {
      await page.waitForSelector(".chat-header-loading", {
        state: "detached",
        timeout: 60_000,
      });
    } catch {}
    await page.waitForTimeout(2000); // let history replay settle

    const reloadSnap = await snapshotAssistant(page);
    console.log(
      "Reload snapshot: asstCount=",
      reloadSnap.asstCount,
      " toolCards=",
      reloadSnap.toolCards.length,
      " reasoningDrawers=",
      reloadSnap.reasoningDrawers.length,
    );

    // ====== THE BUG ASSERTIONS ======

    // 1. Reload shows 2 assistant bubbles (TURN 1 + TURN 2) — NOT 3.
    //    Pre-fix: TURN 1 (1) + TURN 2 planning (1) + TURN 2 synth (1) = 3.
    //    Post-fix: TURN 1 (1) + TURN 2 merged (1) = 2.
    check(
      reloadSnap.asstCount === 2,
      `reload: 2 assistant bubbles (TURN 1 + TURN 2 merged) — got ${reloadSnap.asstCount} (pre-fix was 3)`
    );

    // 2. Reload shows 1 get_current_time tool card (NOT 2).
    //    Pre-fix: 2 cards (planning + synthesis), each with
    //    get_current_time (same tool_use.id).
    //    Post-fix: 1 card (synthesis's rich log, with timing).
    const reloadToolCards = reloadSnap.toolCards.filter((c) =>
      /get_current_time/i.test(c)
    );
    check(
      reloadToolCards.length === 1,
      `reload: exactly 1 get_current_time tool card (NOT 2) — got ${reloadToolCards.length} (cards: ${JSON.stringify(reloadSnap.toolCards)})`
    );

    // 3. Reload's reasoning drawer is still present (planning-step's
    //    thinking trace carried forward into the merged bubble).
    check(
      reloadSnap.reasoningDrawers.length >= 1,
      `reload: ≥1 reasoning drawer visible (planning-step thinking carried into merged bubble) — got ${reloadSnap.reasoningDrawers.length}`
    );

    // 4. Reload's TURN 2 bubble (last) contains SOMETHING — the LLM may
    //    have produced a direct time answer, an error-recovery message
    //    (e.g. "参数必须是有效的 JSON..."), or a preamble; what matters
    //    for v2.0.28.14 is that the bubble exists and is non-empty
    //    (proves the synthesis AIMessage was rehydrated, not dropped).
    const turn2Reload = (
      reloadSnap.asstContent[reloadSnap.asstContent.length - 1] || ""
    ).trim();
    check(
      turn2Reload.length > 0,
      `reload: TURN 2 bubble is non-empty (synthesis AIMessage rehydrated) — got: ${JSON.stringify(turn2Reload.slice(0, 120))}`
    );

    // 4b. Bonus: if the LLM did produce a time answer, note it; if not,
    //     note the error-recovery content. Either is a successful
    //     rehydration of the synthesis AIMessage.
    const directTimeIndicators = [
      "现在时间",
      "当前时间",
      "现在是",
      "上午",
      "下午",
      "星期",
      "周",
      "Asia/Shanghai",
      "北京时间",
    ];
    const hasHHMM = /\d{1,2}:\d{2}/.test(turn2Reload);
    const hasDirectTime =
      hasHHMM ||
      directTimeIndicators.some((kw) => turn2Reload.includes(kw));
    if (hasDirectTime) {
      console.log(`  ✓ reload: TURN 2 bubble contains direct time answer — got: ${JSON.stringify(turn2Reload.slice(0, 80))}`);
    } else {
      console.log(`  ⓘ reload: TURN 2 bubble is a non-time answer (LLM error-recovery or preamble) — got: ${JSON.stringify(turn2Reload.slice(0, 80))}`);
    }

    // 5. No "上一轮" meta-summary prefix (carried from v2.0.28.11).
    check(
      !turn2Reload.startsWith("上一轮"),
      `reload: TURN 2 bubble does NOT start with meta-summary prefix — got: ${JSON.stringify(turn2Reload.slice(0, 30))}`
    );

    await page.screenshot({
      path: "D:/prep for work/Projects/YuanRAG/tests/e2e/output/v2.0.28.14-reload.png",
      fullPage: true,
    });

    // 6. Wire re-check: still 3 records (DB shape unchanged).
    if (threadIdForCheck) {
      const msgRes = await fetch(
        `${BACKEND}/sessions/${encodeURIComponent(threadIdForCheck)}/messages`
      );
      if (msgRes.ok) {
        const msgs = await msgRes.json();
        const asstMsgs = msgs.filter((m) => m.role === "assistant");
        check(
          asstMsgs.length === 3,
          `wire (post-reload, sanity): still 3 assistant records — got ${asstMsgs.length} (DB shape unchanged; v2.0.28.14 is frontend-only)`
        );
      }
    }
  } finally {
    await browser.close();
  }

  console.log(
    `\n>>> Layer 5 v2.0.28.14 verifier: ${passed} passed, ${failed} failed`
  );
  if (failed > 0) process.exit(1);
}

main().catch((e) => {
  console.error("Layer 5 v2.0.28.14 verifier FAILED:", e.message);
  console.error(e.stack);
  process.exit(1);
});