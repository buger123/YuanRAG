// Take a screenshot of the frontend to diagnose layout issue.
const path = require("path");
const { chromium } = require(path.join(
  "D:/prep for work/Projects/YuanRAG/src/frontend",
  "node_modules",
  "playwright"
));

const BACKEND = "http://127.0.0.1:8765";

(async () => {
  const browser = await chromium.launch({ headless: true });

  // Desktop viewport
  const ctx = await browser.newContext({
    viewport: { width: 1440, height: 900 },
  });
  const page = await ctx.newPage();
  await page.goto(`${BACKEND}/`, { waitUntil: "domcontentloaded" });

  // Wait for models ready
  try {
    await page.waitForSelector(".chat-header-loading", {
      state: "detached",
      timeout: 60_000,
    });
  } catch {
    /* may not be loaded */
  }

  await page.screenshot({
    path: "D:/prep for work/Projects/YuanRAG/tests/e2e/output/desktop.png",
    fullPage: false,
  });
  console.log("Desktop screenshot saved");

  // Mobile viewport
  const ctxM = await browser.newContext({
    viewport: { width: 390, height: 844 },
  });
  const pageM = await ctxM.newPage();
  await pageM.goto(`${BACKEND}/`, { waitUntil: "domcontentloaded" });
  try {
    await pageM.waitForSelector(".chat-header-loading", {
      state: "detached",
      timeout: 60_000,
    });
  } catch {}
  await pageM.screenshot({
    path: "D:/prep for work/Projects/YuanRAG/tests/e2e/output/mobile.png",
    fullPage: false,
  });
  console.log("Mobile screenshot saved");

  // Inspect the layout to find issues
  const layout = await page.evaluate(() => {
    const app = document.querySelector(".app");
    const sidebar = document.querySelector(".sidebar");
    const chatPane = document.querySelector(".chat-pane") || document.querySelector(".chat");
    const sidebarBrand = document.querySelector(".sidebar-brand");
    return {
      appBox: app ? app.getBoundingClientRect().toJSON() : null,
      appStyle: app ? window.getComputedStyle(app).gridTemplateColumns : null,
      sidebarBox: sidebar ? sidebar.getBoundingClientRect().toJSON() : null,
      sidebarStyle: sidebar ? {
        position: window.getComputedStyle(sidebar).position,
        width: window.getComputedStyle(sidebar).width,
        transform: window.getComputedStyle(sidebar).transform,
      } : null,
      sidebarBrandBox: sidebarBrand ? sidebarBrand.getBoundingClientRect().toJSON() : null,
      sidebarBrandWidth: sidebarBrand ? window.getComputedStyle(sidebarBrand).width : null,
      chatPaneBox: chatPane ? chatPane.getBoundingClientRect().toJSON() : null,
      windowWidth: window.innerWidth,
      windowHeight: window.innerHeight,
      mediaMatches: {
        mobile: window.matchMedia("(max-width: 768px)").matches,
        dark: window.matchMedia("(prefers-color-scheme: dark)").matches,
      },
      theme: document.documentElement.getAttribute("data-theme"),
    };
  });
  console.log("Desktop layout:", JSON.stringify(layout, null, 2));

  await browser.close();
})().catch((e) => {
  console.error("Error:", e.message);
  process.exit(1);
});