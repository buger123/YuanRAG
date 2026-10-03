// Debug probe for new bug reported 2026-09-23 23:44:
// User said "现在是几点", tool ran successfully returning 23:44,
// but synthesis LLM responded "好的,那就早点休息吧,晚安!"
// instead of stating the time.
//
// Hypothesis: prior turn ended with casual chat wrap-up
// ("请问你想聊些什么?") and the synthesis LLM is being confused
// into continuing that conversational mode instead of answering
// the fresh question.
//
// Pattern to reproduce:
//   TURN 1: 你是谁
//   TURN 2: 我是谁
//   TURN 3: 现在是几点   <-- THE BUG
//
// Run 5 times to confirm deterministic.

const path = require("path");
const { chromium } = require(path.join(
  "D:/prep for work/Projects/YuanRAG/src/frontend",
  "node_modules",
  "playwright"
));

const BACKEND = "http://127.0.0.1:8765";

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

  await page.goto(BACKEND, { waitUntil: "domcontentloaded" });
  await page.waitForSelector(".chat-header-loading", {
    state: "detached",
    timeout: 90_000,
  });

  async function send(q) {
    await page.fill(".chat-textarea", q);
    await page.click(".btn-send");
    await page.waitForTimeout(2000);
    // Wait for the streaming to complete (send button visible).
    let lastText = "";
    for (let i = 0; i < 60; i++) {
      await page.waitForTimeout(2000);
      const t = await page.evaluate(() => {
        const asst = document.querySelectorAll(".message.assistant");
        const last = asst[asst.length - 1];
        return last ? last.textContent : "";
      });
      if (t === lastText && i > 3) break;
      lastText = t;
    }
  }

  console.log(">>> NEW THREAD (no prior context)");
  await send("你是谁");
  await send("我是谁");
  console.log(">>> TURN 3 (THE BUG): 现在是几点");
  await send("现在是几点");

  // Capture the synthesis answer for TURN 3
  const result = await page.evaluate(() => {
    const asst = Array.from(document.querySelectorAll(".message.assistant"));
    const last = asst[asst.length - 1];
    if (!last) return null;
    const body = last.querySelector(".markdown-body");
    const text = body ? body.textContent.trim() : last.textContent.trim();
    return { text, has_time: /\d{1,2}:\d{2}|23:44|\d{4}年\d{1,2}月\d{1,2}日/.test(text) };
  });

  console.log("TURN 3 synthesis answer:", JSON.stringify(result));
  if (!result.has_time) {
    console.log("!!! BUG REPRODUCED — synthesis answer lacks the time");
  } else {
    console.log("✓ Synthesis answer contains the time — no bug in this run");
  }

  await browser.close();
}

main().catch((e) => {
  console.error("probe failed:", e.message);
  console.error(e.stack);
  process.exit(1);
});