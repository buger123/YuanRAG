// Reproduce the bug: "answer doesn't display initially but shows on reload/re-click"
// Drive a chromium session, send Q1, wait for streaming to finish, then check
// DOM state WITHOUT reloading. Compare with state AFTER reload.
const path = require("path");
const { chromium } = require(path.join(
  "D:/prep for work/Projects/YuanRAG/src/frontend",
  "node_modules",
  "playwright",
));

const BACKEND = "http://127.0.0.1:8765";

(async () => {
  const browser = await chromium.launch({ headless: true });
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
    await page.waitForSelector(".chat-header-loading", { state: "detached", timeout: 60_000 });
  } catch (e) {
    console.log("backend not ready, skipping");
    await browser.close();
    return;
  }

  // === ROUND 1: First message ===
  console.log("\n--- ROUND 1: send Q1, watch for answer ---");
  const q1 = "用一句话回答: 中国的首都是哪里?";
  await page.fill(".chat-textarea", q1);
  await page.click(".btn-send");

  // Wait for streaming to complete (send button visible + 1 assistant bubble)
  await page.waitForSelector(".btn-send", { state: "visible", timeout: 60_000 });

  // Snapshot DOM
  const r1 = await page.evaluate(() => {
    const userMsgs = Array.from(document.querySelectorAll(".message.user .message-text"))
      .map((n) => n.textContent);
    const assistantMsgs = Array.from(document.querySelectorAll(".message.assistant"))
      .map((n) => {
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
  console.log("After Q1 streaming completes:", JSON.stringify(r1, null, 2));

  // Screenshot before reload
  await page.screenshot({
    path: "D:/prep for work/Projects/YuanRAG/tests/e2e/output/repro-bug-before-reload-r1.png",
    fullPage: true,
  });

  // === Reload to see if the answer shows up ===
  console.log("\n--- RELOAD PAGE ---");
  await page.reload({ waitUntil: "domcontentloaded" });
  try {
    await page.waitForSelector(".chat-header-loading", { state: "detached", timeout: 60_000 });
  } catch {}
  await page.waitForTimeout(2000);

  const r2 = await page.evaluate(() => {
    const userMsgs = Array.from(document.querySelectorAll(".message.user .message-text"))
      .map((n) => n.textContent);
    const assistantMsgs = Array.from(document.querySelectorAll(".message.assistant"))
      .map((n) => {
        const body = n.querySelector(".markdown-body");
        return body ? body.textContent.trim() : n.textContent.trim();
      });
    return {
      userCount: userMsgs.length,
      assistantCount: assistantMsgs.length,
      userContent: userMsgs,
      assistantContent: assistantMsgs,
    };
  });
  console.log("After reload:", JSON.stringify(r2, null, 2));

  // Screenshot after reload
  await page.screenshot({
    path: "D:/prep for work/Projects/YuanRAG/tests/e2e/output/repro-bug-after-reload-r1.png",
    fullPage: true,
  });

  // === ROUND 2: Send Q2 in same thread ===
  console.log("\n--- ROUND 2: send Q2 on same thread ---");
  const q2 = "再回答: 美国呢?";
  await page.fill(".chat-textarea", q2);
  await page.click(".btn-send");
  await page.waitForSelector(".btn-send", { state: "visible", timeout: 60_000 });

  const r3 = await page.evaluate(() => {
    const userMsgs = Array.from(document.querySelectorAll(".message.user .message-text"))
      .map((n) => n.textContent);
    const assistantMsgs = Array.from(document.querySelectorAll(".message.assistant"))
      .map((n) => {
        const body = n.querySelector(".markdown-body");
        return body ? body.textContent.trim() : n.textContent.trim();
      });
    const isStreaming = document.querySelector(".thinking-status") !== null;
    const sendBtnVisible = document.querySelector(".btn-send") !== null;
    return {
      userCount: userMsgs.length,
      assistantCount: assistantMsgs.length,
      userContent: userMsgs,
      assistantContent: assistantMsgs,
      isStreaming,
      sendBtnVisible,
    };
  });
  console.log("After Q2 streaming completes:", JSON.stringify(r3, null, 2));

  await page.screenshot({
    path: "D:/prep for work/Projects/YuanRAG/tests/e2e/output/repro-bug-before-reload-r2.png",
    fullPage: true,
  });

  // === Reload again ===
  console.log("\n--- RELOAD AGAIN ---");
  await page.reload({ waitUntil: "domcontentloaded" });
  try {
    await page.waitForSelector(".chat-header-loading", { state: "detached", timeout: 60_000 });
  } catch {}
  await page.waitForTimeout(2000);

  const r4 = await page.evaluate(() => {
    const userMsgs = Array.from(document.querySelectorAll(".message.user .message-text"))
      .map((n) => n.textContent);
    const assistantMsgs = Array.from(document.querySelectorAll(".message.assistant"))
      .map((n) => {
        const body = n.querySelector(".markdown-body");
        return body ? body.textContent.trim() : n.textContent.trim();
      });
    return {
      userCount: userMsgs.length,
      assistantCount: assistantMsgs.length,
      userContent: userMsgs,
      assistantContent: assistantMsgs,
    };
  });
  console.log("After 2nd reload:", JSON.stringify(r4, null, 2));

  await page.screenshot({
    path: "D:/prep for work/Projects/YuanRAG/tests/e2e/output/repro-bug-after-reload-r2.png",
    fullPage: true,
  });

  await browser.close();
  console.log("\n=== Done ===");
})().catch((e) => {
  console.error("Error:", e.message);
  console.error(e.stack);
  process.exit(1);
});