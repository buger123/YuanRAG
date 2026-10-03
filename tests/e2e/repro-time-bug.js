// Reproduce: "现在是几点" doesn't answer correctly
// Drive a chromium session, ask "现在是几点", capture live response.
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

(async () => {
  console.log(">>> Repro: '现在是几点' should return the current time");
  if (!(await backendReady())) {
    console.log("!!! Backend not ready — skipping");
    process.exit(0);
  }
  console.log("  ✓ Backend ready");

  const browser = await chromium.launch({ headless: true });
  try {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const page = await ctx.newPage();

    // Capture console for debugging
    page.on("console", (msg) => console.log(`  [browser ${msg.type()}]`, msg.text()));

    await page.addInitScript(() => {
      try {
        window.localStorage.setItem("rag.theme", "light");
        window.localStorage.setItem("rag.locale", "zh");
      } catch {}
    });

    await page.goto(`${BACKEND}/`, { waitUntil: "domcontentloaded" });
    await page.waitForSelector(".chat-header-loading", { state: "detached", timeout: 60_000 });
    console.log("  ✓ Models ready");

    // Ask the time question
    const q = "现在是几点";
    console.log(`\n>>> Sending: ${q}`);
    await page.fill(".chat-textarea", q);
    await page.click(".btn-send");

    // Wait for streaming to complete
    await page.waitForSelector(".btn-send", { state: "visible", timeout: 60_000 });
    console.log("  ✓ Streaming complete");

    // Snapshot
    const result = await page.evaluate(() => {
      const userMsgs = Array.from(
        document.querySelectorAll(".message.user .message-text")
      ).map((n) => n.textContent);
      const assistantMsgs = Array.from(
        document.querySelectorAll(".message.assistant")
      ).map((n) => {
        const body = n.querySelector(".markdown-body");
        return body ? body.textContent.trim() : n.textContent.trim();
      });
      const toolCalls = Array.from(
        document.querySelectorAll(".tool-call-card, [data-tool-name]")
      ).map((n) => n.textContent.trim());
      return { userMsgs, assistantMsgs, toolCalls };
    });
    console.log("\n>>> Result:");
    console.log(JSON.stringify(result, null, 2));

    await page.screenshot({
      path: "D:/prep for work/Projects/YuanRAG/tests/e2e/output/repro-time-bug.png",
      fullPage: true,
    });

    // Also check wire-level messages
    const sessRes = await fetch(`${BACKEND}/sessions`);
    const sessions = await sessRes.json();
    if (Array.isArray(sessions) && sessions.length > 0) {
      const threadId = sessions[0].thread_id || sessions[0].id;
      const msgRes = await fetch(`${BACKEND}/sessions/${encodeURIComponent(threadId)}/messages`);
      const msgs = await msgRes.json();
      console.log(`\n>>> Wire messages (thread ${threadId.slice(0, 12)}...):`);
      console.log(JSON.stringify(msgs.map((m) => ({ role: m.role, content: m.content?.slice(0, 80) })), null, 2));
    }

    await ctx.close();
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error("Error:", e.message);
  console.error(e.stack);
  process.exit(1);
});