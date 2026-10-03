// v2.0.28.15 PROBE — user report 2026-09-23:
// "现在没有出现直接回答结果" + shows reasoning-trace leaking into
// assistant bubble:
//   "The user is asking '现在几点了' (What time is it now?).
//    This is a time-sensitive question, so I need to call
//    get_current_time to get the accurate current time."
//
// Same family as v2.0.28.12/v2.0.28.13 — planning-step AIMessage
// preamble leaks into visible content instead of being filtered.
//
// PROBE only — captures the wire shape + asks "现在几点了" so we
// can see exactly what each AIMessage's content looks like.

const path = require("path");
const { chromium } = require(path.join(
  "D:/prep for work/Projects/YuanRAG/src/frontend",
  "node_modules",
  "playwright"
));

const BACKEND = "http://127.0.0.1:8765";

async function main() {
  console.log(">>> v2.0.28.15 PROBE — capture wire shape for 现在几点了");

  if (
    !(await fetch(`${BACKEND}/models/status`)
      .then((r) => r.json())
      .then((b) => b.ready))
  ) {
    console.log("!!! Backend not ready. Exiting.");
    process.exit(0);
  }

  const browser = await chromium.launch({ headless: true });
  try {
    const ctx = await browser.newContext({
      viewport: { width: 1440, height: 900 },
    });
    const page = await ctx.newPage();
    await page.addInitScript(() => {
      try {
        window.localStorage.setItem("rag.theme", "light");
        window.localStorage.setItem("rag.locale", "zh");
      } catch {}
    });

    await page.goto(BACKEND, { waitUntil: "domcontentloaded" });
    try {
      await page.waitForSelector(".chat-header-loading", {
        state: "detached",
        timeout: 90_000,
      });
    } catch {}

    // Ask "现在几点了" directly (no warmup — closest to user's report)
    await page.fill(".chat-textarea", "现在几点了");
    await page.click(".btn-send");
    try {
      await page.waitForSelector(".btn-send", {
        state: "visible",
        timeout: 90_000,
      });
    } catch {}
    await page.waitForTimeout(2000);

    // Snapshot live
    const liveSnap = await page.evaluate(() => {
      const asst = Array.from(
        document.querySelectorAll(".message.assistant")
      ).map((n) => {
        const body = n.querySelector(".markdown-body");
        const drawer = n.querySelector(".thinking-drawer");
        return {
          content: body ? body.textContent.trim() : n.textContent.trim(),
          reasoningDrawerText: drawer
            ? drawer.textContent.trim().slice(0, 300)
            : null,
          fullText: n.textContent.trim(),
        };
      });
      return { asstCount: asst.length, messages: asst };
    });
    console.log(
      `\nLive: ${liveSnap.asstCount} bubbles`
    );
    for (let i = 0; i < liveSnap.messages.length; i++) {
      const m = liveSnap.messages[i];
      console.log(`\n=== LIVE BUBBLE ${i} ===`);
      console.log(`content: ${JSON.stringify(m.content.slice(0, 200))}`);
      console.log(`reasoningDrawer: ${JSON.stringify(m.reasoningDrawerText)}`);
    }

    // Snapshot reload
    await page.reload({ waitUntil: "domcontentloaded" });
    try {
      await page.waitForSelector(".chat-header-loading", {
        state: "detached",
        timeout: 60_000,
      });
    } catch {}
    await page.waitForTimeout(2000);

    const reloadSnap = await page.evaluate(() => {
      const asst = Array.from(
        document.querySelectorAll(".message.assistant")
      ).map((n) => {
        const body = n.querySelector(".markdown-body");
        const drawer = n.querySelector(".thinking-drawer");
        return {
          content: body ? body.textContent.trim() : n.textContent.trim(),
          reasoningDrawerText: drawer
            ? drawer.textContent.trim().slice(0, 300)
            : null,
          fullText: n.textContent.trim(),
        };
      });
      return { asstCount: asst.length, messages: asst };
    });
    console.log(
      `\n\nReload: ${reloadSnap.asstCount} bubbles`
    );
    for (let i = 0; i < reloadSnap.messages.length; i++) {
      const m = reloadSnap.messages[i];
      console.log(`\n=== RELOAD BUBBLE ${i} ===`);
      console.log(`content: ${JSON.stringify(m.content.slice(0, 200))}`);
      console.log(`reasoningDrawer: ${JSON.stringify(m.reasoningDrawerText)}`);
    }

    // Wire cross-check
    const threadId = await page.evaluate(() => {
      return window.localStorage.getItem("rag.active_thread") || "";
    });
    if (threadId) {
      const r = await fetch(
        `${BACKEND}/sessions/${encodeURIComponent(threadId)}/messages`
      );
      if (r.ok) {
        const msgs = await r.json();
        const asst = msgs.filter((m) => m.role === "assistant");
        console.log(
          `\n\nWire: ${asst.length} assistant records`
        );
        for (let i = 0; i < asst.length; i++) {
          const m = asst[i];
          console.log(`\n=== WIRE AIMessage ${i} ===`);
          console.log(
            `content: ${JSON.stringify((m.content || "").slice(0, 200))}`
          );
          console.log(
            `reasoning: ${JSON.stringify(
              (m.reasoning || "").slice(0, 200)
            )}`
          );
          console.log(`tool_calls: ${(m.tool_calls || []).length}`);
          for (const tc of m.tool_calls || []) {
            console.log(
              `  - id=${tc.id} name=${tc.name} ok=${tc.ok} result=${(tc.result || "").slice(0, 60)}`
            );
          }
        }
      }
    }
  } finally {
    await browser.close();
  }
}

main().catch((e) => {
  console.error("PROBE FAILED:", e.message);
  console.error(e.stack);
  process.exit(1);
});