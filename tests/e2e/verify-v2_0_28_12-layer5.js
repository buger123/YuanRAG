// v2.0.28.12 — multi-turn "现在几点了" bug Layer 5 verifier.
//
// Pre-fix bug (2026-09-23, post-v2.0.28.11 polish):
//   TURN 1: "你是谁" → "你好！我是源 RAG..."
//   TURN 2: "现在几点了"
//   → Assistant replied with DIRECT_SYSTEM's verbatim denial language:
//     "我没有获取实时信息的能力，无法告诉你现在的具体时间。你可以查
//      看你的设备屏幕，或者直接问我 "现在几点了"，系统会帮你联网查询
//      ——不过那需要你重新发送一次才能触发检索流程。\n\n你是想问北京
//      时间，还是其他时区？"
//
// Two distinct root causes, one shared symptom:
//   1. Intent misroute: cheap-model classifier sometimes rounds
//      "现在几点了" to intent="greeting" → path went through
//      react_generate_direct (no tools bound) → LLM said verbatim
//      from DIRECT_SYSTEM line 234-237.
//   2. Synthesis-LLM confusion: even when intent=qa_complex →
//      react_agent → react_generate, the synthesis LLM got confused
//      by the old _HISTORY_FRAME_HINT ("every AIMessage is your prior
//      reply") and either produced meta-text or hallucinated the
//      denial.
//
// v2.0.28.12 fix (3 files):
//   1. src/agent/nodes/intent_analysis.py — guard greeting fast-path
//      with `not needs_time` + extend slow-path override to cover
//      "greeting".
//   2. src/agent/nodes/_history_frame_hint.py — clarify the
//      AIMessage taxonomy: text-only = final reply, with tool_calls =
//      planning step.
//   3. src/agent/nodes/react_agent.py — strengthen time-tool hint
//      from "建议先调用" (recommend) to "必须先调用" (must).
//
// Strategy:
//   1. Open real Chromium, drive 2-turn conversation.
//   2. Turn 1: "你是谁" (warmup).
//   3. Turn 2: "现在几点了" — THE BUG.
//   4. Snapshot live assistant bubble.
//   5. Assert the response:
//      a) calls get_current_time tool (tool card visible live).
//      b) does NOT contain DIRECT_SYSTEM denial language
//         ("重新发送", "触发检索", "我没有获取实时信息").
//      c) contains a direct time answer.
//      d) does NOT have the "上一轮" meta-summary prefix.
//   6. Wire cross-check: GET /sessions/{tid}/messages — the saved
//      assistant content also lacks the denial language.

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
    const isStreaming =
      document.querySelector(".thinking-status") !== null;
    const sendBtnVisible = document.querySelector(".btn-send") !== null;
    return {
      asstCount: asst.length,
      asstContent: asst,
      toolCards,
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
  console.log(">>> Layer 5 v2.0.28.12 multi-turn time-question verifier");
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

  // Thread ID captured early so we can wire cross-check after reload.
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

    // === TURN 1: warmup greeting ===
    const q1 = "你是谁";
    console.log(`\n--- TURN 1: ${q1} ---`);
    await sendAndWait(page, q1, 60_000);

    const snap1 = await snapshotAssistant(page);
    check(
      snap1.asstCount === 1,
      `TURN 1: 1 assistant message (got ${snap1.asstCount})`
    );
    check(
      (snap1.asstContent[0] || "").length > 10,
      `TURN 1: assistant reply is non-empty (got len=${(snap1.asstContent[0] || "").length})`
    );

    // === TURN 2: THE BUG ===
    const q2 = "现在几点了";
    console.log(`\n--- TURN 2: ${q2} ---`);
    await sendAndWait(page, q2, 90_000);

    const snap2 = await snapshotAssistant(page);
    console.log("TURN 2 live snapshot:", JSON.stringify(snap2, null, 2));

    // After 2 turns: TURN 1 (1 assistant) + TURN 2 (1+ assistant)
    // = at least 2 assistant messages.
    check(
      snap2.asstCount >= 2,
      `TURN 2: at least 2 assistant messages live (got ${snap2.asstCount})`
    );

    // The TURN 2 assistant text is the LAST non-empty entry.
    const turn2Texts = snap2.asstContent.map((t) => (t || "").trim());
    const turn2Live = (turn2Texts[turn2Texts.length - 1] || "").trim();

    console.log(`TURN 2 live assistant text: ${JSON.stringify(turn2Live.slice(0, 200))}`);

    // ====== THE BUG ASSERTIONS ======

    // 1. get_current_time tool card visible live — proves the LLM
    //    actually called the tool (vs hallucinating a denial).
    const hasToolCard = snap2.toolCards.some((c) =>
      /get_current_time|当前时间/i.test(c)
    );
    check(
      hasToolCard,
      `TURN 2: get_current_time tool card visible live — cards: ${JSON.stringify(snap2.toolCards)}`
    );

    // 2. NO DIRECT_SYSTEM denial language — these exact phrases appear
    //    in DIRECT_SYSTEM line 234-237 and were verbatim in the bug.
    const denialPhrases = [
      "重新发送",
      "触发检索",
      "我没有获取实时信息",
      "无法获取实时",
      "联网查询",
    ];
    const denialMatches = denialPhrases.filter((p) => turn2Live.includes(p));
    check(
      denialMatches.length === 0,
      `TURN 2: NO DIRECT_SYSTEM denial language (matched: ${JSON.stringify(denialMatches)}) — got: ${JSON.stringify(turn2Live.slice(0, 80))}`
    );

    // 3. Contains a direct time answer. The synthesis LLM has multiple
    //    acceptable phrasings ("现在时间是", "现在是", "当前时间",
    //    "2026年9月23日", "星期三", "周三", "Asia/Shanghai", "北京时间",
    //    explicit HH:MM). Match broadly — the v2.0.28.12 fix is about
    //    using the TM data, not about exact phrasing.
    const directTimeIndicators = [
      "现在时间",
      "当前时间",
      "现在是",
      "上午",
      "下午",
      "星期",
      "周",          // 周三 / 周四
      "Asia/Shanghai",
      "北京时间",
      "Beijing",
    ];
    // Also accept HH:MM pattern (e.g. "13:03").
    const hasHHMM = /\d{1,2}:\d{2}/.test(turn2Live);
    const hasDirectTime =
      hasHHMM ||
      directTimeIndicators.some((kw) => turn2Live.includes(kw));
    check(
      hasDirectTime,
      `TURN 2: response contains direct time answer (one of ${directTimeIndicators.join(", ")} OR HH:MM pattern) — got: ${JSON.stringify(turn2Live.slice(0, 80))}`
    );

    // 4. No "上一轮" meta-summary prefix (the v2.0.28.11 bug also
    //    gated this; v2.0.28.12 keeps the fix).
    check(
      !turn2Live.startsWith("上一轮"),
      `TURN 2: response does NOT start with meta-summary prefix — got: ${JSON.stringify(turn2Live.slice(0, 30))}`
    );

    await page.screenshot({
      path: "D:/prep for work/Projects/YuanRAG/tests/e2e/output/v2.0.28.12-live.png",
      fullPage: true,
    });

    // === RELOAD — assert the saved assistant also lacks denial ===
    console.log("\n--- RELOAD ---");
    await page.reload({ waitUntil: "domcontentloaded" });
    try {
      await page.waitForSelector(".chat-header-loading", {
        state: "detached",
        timeout: 60_000,
      });
    } catch {}
    await page.waitForTimeout(2000);

    const snapReload = await snapshotAssistant(page);
    console.log("Reload snapshot:", JSON.stringify(snapReload, null, 2));

    const reloadTexts = snapReload.asstContent.map((t) => (t || "").trim());
    // The TURN 2 synthesis is the LAST non-empty text.
    const reloadSynthesis =
      [...reloadTexts].reverse().find((t) => t.length > 0) || "";
    console.log(`Reload synthesis: ${JSON.stringify(reloadSynthesis.slice(0, 200))}`);

    check(
      !denialPhrases.some((p) => reloadSynthesis.includes(p)),
      `post-reload: synthesis response does NOT contain any denial phrase — got: ${JSON.stringify(reloadSynthesis.slice(0, 80))}`
    );
    const reloadHasHHMM = /\d{1,2}:\d{2}/.test(reloadSynthesis);
    check(
      reloadHasHHMM || directTimeIndicators.some((kw) => reloadSynthesis.includes(kw)),
      `post-reload: synthesis response contains a direct time indicator — got: ${JSON.stringify(reloadSynthesis.slice(0, 80))}`
    );

    // === Wire cross-check via /sessions/{tid}/messages ===
    const activeThreadId = await page.evaluate(() => {
      try {
        return window.localStorage.getItem("rag.currentThread") || "";
      } catch {
        return "";
      }
    });
    threadIdForCheck = activeThreadId;
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
      console.log(`>>> thread_id: ${threadIdForCheck}`);
      const msgRes = await fetch(
        `${BACKEND}/sessions/${encodeURIComponent(threadIdForCheck)}/messages`
      );
      if (msgRes.ok) {
        const msgs = await msgRes.json();
        const asstMsgs = msgs.filter((m) => m.role === "assistant");
        console.log(`Saved assistant count: ${asstMsgs.length}`);
        for (let i = 0; i < asstMsgs.length; i++) {
          const m = asstMsgs[i];
          const text = (m.content || "").trim();
          console.log(
            `  [${i}] content=${JSON.stringify(text.slice(0, 120))} tool_calls=${JSON.stringify(m.tool_calls)}`
          );
        }
        // Per v2.0.28.10 + v2.0.28.11:
        //   TURN 1 = [HM, AIM_synthesis] = 1 assistant.
        //   TURN 2 = [HM, AIM_tool_calls(get_current_time), AIM_synthesis] = 2 assistant records
        //     (the tool-calls AIM has empty content but legitimate role=assistant metadata).
        // Total saved assistant = 3.
        check(
          asstMsgs.length === 3,
          `wire /sessions: 3 assistant records (got ${asstMsgs.length}) — TURN 1 synthesis + TURN 2 tool-calls AIM + TURN 2 synthesis`
        );

        // The TURN 2 synthesis is the LAST assistant record with non-empty content.
        const turn2Saved =
          [...asstMsgs].reverse().find(
            (m) => (m.content || "").trim().length > 0
          ) || {};
        const savedAssistant = (turn2Saved.content || "").trim();
        check(
          !denialPhrases.some((p) => savedAssistant.includes(p)),
          `wire /sessions saved TURN 2 synthesis does NOT contain any denial phrase — got: ${JSON.stringify(savedAssistant.slice(0, 80))}`
        );
        const savedHasHHMM = /\d{1,2}:\d{2}/.test(savedAssistant);
        check(
          savedHasHHMM || directTimeIndicators.some((kw) => savedAssistant.includes(kw)),
          `wire /sessions saved TURN 2 synthesis contains a direct time indicator — got: ${JSON.stringify(savedAssistant.slice(0, 80))}`
        );

        // The TURN 2 tool-calls AIM MUST carry get_current_time
        // (proves intent was qa_complex and react_agent ran with tools
        // bound — the intent_analysis override worked).
        const turn2ToolCallsAIM = asstMsgs.find(
          (m) =>
            Array.isArray(m.tool_calls) &&
            m.tool_calls.length > 0 &&
            (m.content || "").trim().length === 0
        );
        check(
          turn2ToolCallsAIM &&
            (turn2ToolCallsAIM.tool_calls || []).some(
              (tc) => tc.name === "get_current_time"
            ),
          `wire /sessions: TURN 2 has a tool-calls AIMessage calling get_current_time (proves intent=qa_complex + react_agent ran)`
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
    `\n>>> Layer 5 v2.0.28.12 verifier: ${passed} passed, ${failed} failed`
  );
  if (failed > 0) process.exit(1);
}

main().catch((e) => {
  console.error("Layer 5 v2.0.28.12 verifier FAILED:", e.message);
  console.error(e.stack);
  process.exit(1);
});