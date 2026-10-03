// v2.0.28.11 — synthesis LLM meta-text bug Layer 5 verifier.
//
// Pre-fix bug:
//   Q: "现在是几点"
//   A: "上一轮的回复简要总结:\n\n- **当前时间**:2026 年 9 月 23 日..."
//
// The synthesis LLM was reading react_agent's text-only draft AIMessage
// (via _HISTORY_FRAME_HINT mislabeling it as "your previous reply")
// and producing a meta-summary of its own prior reply.
//
// v2.0.28.11 strips the trailing text-only AIMessage in
// _sanitize_history_for_generate so the synthesis LLM sees a fresh
// prompt — the user's question + the time ToolMessage — and produces
// a direct answer.
//
// Strategy:
//   1. Drive a real Chromium session, ask "现在是几点".
//   2. Wait for streaming to complete.
//   3. Snapshot live assistant bubble.
//   4. Assert the live response does NOT start with the meta-summary
//      prefix "上一轮".
//   5. Assert the live response contains a direct time answer
//      ("当前时间" or "现在是" or just the date string).
//   6. Wire cross-check: GET /sessions/{tid}/messages → saved assistant
//      content also lacks the meta-summary prefix.

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

async function main() {
  console.log(">>> Layer 5 v2.0.28.11 synthesis LLM direct-answer verifier");
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

    const modelsReady = await waitForModels(page);
    check(modelsReady, "models ready (chat-header-loading detached)");

    // === SEND the time question ===
    const q = "现在是几点";
    console.log(`\n--- SEND: ${q} ---`);
    await page.fill(".chat-textarea", q);
    await page.click(".btn-send");

    // Wait for the assistant bubble to appear.
    let assistantAppeared = false;
    try {
      await page.waitForSelector(
        ".message.assistant .markdown-body >> nth=0",
        { state: "visible", timeout: 30_000 }
      );
      assistantAppeared = true;
    } catch {
      // Try with longer timeout (BGE embedding first).
    }
    if (!assistantAppeared) {
      try {
        await page.waitForSelector(
          ".message.assistant .markdown-body >> nth=0",
          { state: "visible", timeout: 60_000 }
        );
        assistantAppeared = true;
      } catch {}
    }
    check(assistantAppeared, "TURN 1 assistant bubble appeared live");

    // Wait for streaming to complete (send button re-appears).
    let streamDone = false;
    try {
      await page.waitForSelector(".btn-send", {
        state: "visible",
        timeout: 60_000,
      });
      streamDone = true;
    } catch {}
    check(streamDone, "streaming complete (send button re-visible)");

    // Small settle wait for any final overwrite to land.
    await page.waitForTimeout(1500);

    // === Snapshot live state ===
    const snap = await snapshotAssistant(page);
    console.log("Live snapshot:", JSON.stringify(snap, null, 2));

    const assistantText = (snap.asstContent[0] || "").trim();

    check(
      snap.asstCount === 1,
      `TURN 1 live: 1 assistant message (got ${snap.asstCount})`
    );

    // === THE BUG ASSERTIONS ===

    // 1. The response must NOT start with the meta-summary prefix.
    check(
      !assistantText.startsWith("上一轮"),
      `response does NOT start with meta-summary prefix '上一轮' — got: ${JSON.stringify(assistantText.slice(0, 30))}`
    );

    // 2. The response must NOT contain the "brief summary" phrase
    //    that the buggy LLM emitted.
    check(
      !assistantText.includes("上一轮"),
      `response does NOT contain '上一轮' anywhere — got: ${JSON.stringify(assistantText.slice(0, 60))}`
    );

    // 3. The response must contain a direct time answer.
    //    Acceptable indicators: 当前时间 / 现在是 / the date string /
    //    上午 / 下午 / 周三.
    const directTimeIndicators = [
      "当前时间",
      "现在是",
      "上午",
      "下午",
      "周",        // 周三 / 周四
      "Asia/Shanghai",  // timezone (sometimes in technical answer)
    ];
    const hasDirectTime = directTimeIndicators.some((kw) =>
      assistantText.includes(kw)
    );
    check(
      hasDirectTime,
      `response contains direct time answer (one of ${directTimeIndicators.join(", ")}) — got: ${JSON.stringify(assistantText.slice(0, 80))}`
    );

    // 4. The response should NOT be the "抱歉,我无法确定..." fallback
    //    (which would happen if v2.0.9 TM preservation regressed).
    check(
      !assistantText.includes("无法确定"),
      `response does NOT use the 'unable to determine time' fallback — got: ${JSON.stringify(assistantText.slice(0, 80))}`
    );

    // 5. The ToolMessage card should appear live (get_current_time was called).
    const hasToolCard = snap.toolCards.some((c) => /get_current_time|当前时间/i.test(c));
    check(
      hasToolCard,
      `get_current_time tool card visible live — cards: ${JSON.stringify(snap.toolCards)}`
    );

    await page.screenshot({
      path: "D:/prep for work/Projects/YuanRAG/tests/e2e/output/v2.0.28.11-live.png",
      fullPage: true,
    });

    // === RELOAD — assert the saved assistant also lacks meta-summary ===
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

    // Find the NON-EMPTY assistant content (the synthesis). The
    // empty one is the react_agent's tool_calls AIMessage (no text
    // — only tool_calls metadata). That's legitimate per v2.0.10.1
    // (parent-of-preserved survives sanitizer). The frontend
    // renders it as an empty bubble; the actual answer is the
    // non-empty entry.
    const reloadTexts = snapReload.asstContent.map((t) => (t || "").trim());
    const reloadSynthesis = reloadTexts.find((t) => t.length > 0) || "";
    // 1 turn with 1 tool-call round = [HM, AIM_tool_calls, AIM_synthesis]
    // → 2 assistant records on reload (the tool-calls AIM has empty
    // content but is rendered as a bubble; the synthesis is the
    // real answer).
    check(
      snapReload.asstCount === 2,
      `post-reload: 2 assistant messages (HM + tool-calls AIM + synthesis → 2 assistant records; got ${snapReload.asstCount})`
    );
    check(
      !reloadSynthesis.startsWith("上一轮"),
      `post-reload: synthesis response does NOT start with meta-summary prefix — got: ${JSON.stringify(reloadSynthesis.slice(0, 30))}`
    );
    check(
      !reloadSynthesis.includes("上一轮"),
      `post-reload: synthesis response does NOT contain '上一轮' anywhere — got: ${JSON.stringify(reloadSynthesis.slice(0, 60))}`
    );

    // === Wire cross-check via /sessions/{tid}/messages ===
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
        const asstMsgs = msgs.filter((m) => m.role === "assistant");
        // 1 turn with 1 tool-call round = [HM, AIM_tool_calls, AIM_synthesis]
        // → 2 assistant records on the wire (the tool-calls AIM has
        // empty content but legitimate role=assistant metadata). The
        // synthesis is the one with non-empty content.
        check(
          asstMsgs.length === 2,
          `GET /sessions/{tid}/messages: 2 assistant records (got ${asstMsgs.length}) — 1 tool-call AIM + 1 synthesis`
        );
        const synthRecord = asstMsgs.find(
          (m) => (m.content || "").trim().length > 0
        );
        const savedAssistant = (synthRecord?.content || "").trim();
        check(
          !savedAssistant.startsWith("上一轮"),
          `wire /sessions saved synthesis does NOT start with '上一轮' — got: ${JSON.stringify(savedAssistant.slice(0, 30))}`
        );
        check(
          !savedAssistant.includes("上一轮"),
          `wire /sessions saved synthesis does NOT contain '上一轮' — got: ${JSON.stringify(savedAssistant.slice(0, 60))}`
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
    `\n>>> Layer 5 v2.0.28.11 verifier: ${passed} passed, ${failed} failed`
  );
  if (failed > 0) process.exit(1);
}

main().catch((e) => {
  console.error("Layer 5 v2.0.28.11 verifier FAILED:", e.message);
  console.error(e.stack);
  process.exit(1);
});