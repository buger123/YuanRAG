// v2.0.28.7 — desktop layout regression guard.
// Asserts that the sidebar lands at x=0..280 and the chat-pane fills
// the rest of the row, BOTH at full viewport height. If the
// ``sidebar-backdrop`` ever regresses into a grid-occupier (e.g. its
// base rule is moved back inside ``@media (max-width: 768px)``), the
// backdrop would steal row 1 col 1, pushing the sidebar into col 2
// and the chat-pane into row 2 — this script catches that.

const path = require("path");
const { chromium } = require(path.join(
  "D:/prep for work/Projects/YuanRAG/src/frontend",
  "node_modules",
  "playwright"
));

const BACKEND = "http://127.0.0.1:8765";
const VIEWPORTS = [
  { name: "desktop-1440", width: 1440, height: 900 },
  { name: "desktop-1024", width: 1024, height: 768 },
  { name: "tablet-800", width: 800, height: 1024 },
];

(async () => {
  const browser = await chromium.launch({ headless: true });
  let passed = 0;
  let failed = 0;

  for (const vp of VIEWPORTS) {
    const ctx = await browser.newContext({ viewport: { width: vp.width, height: vp.height } });
    const page = await ctx.newPage();
    await page.goto(`${BACKEND}/`, { waitUntil: "domcontentloaded" });
    // Wait for models ready
    try {
      await page.waitForSelector(".chat-header-loading", {
        state: "detached",
        timeout: 60_000,
      });
    } catch {
      /* may not load in CI; layout assertions still meaningful */
    }
    const layout = await page.evaluate(() => {
      const sidebar = document.querySelector(".sidebar");
      const chatPane = document.querySelector(".chat-pane");
      return {
        sidebar: sidebar ? sidebar.getBoundingClientRect().toJSON() : null,
        chatPane: chatPane ? chatPane.getBoundingClientRect().toJSON() : null,
        sidebarComputed: sidebar
          ? {
              position: window.getComputedStyle(sidebar).position,
              width: window.getComputedStyle(sidebar).width,
              height: window.getComputedStyle(sidebar).height,
              transform: window.getComputedStyle(sidebar).transform,
            }
          : null,
        chatPaneComputed: chatPane
          ? {
              position: window.getComputedStyle(chatPane).position,
              width: window.getComputedStyle(chatPane).width,
              height: window.getComputedStyle(chatPane).height,
            }
          : null,
        backdropComputed: (() => {
          const bd = document.querySelector(".sidebar-backdrop");
          return bd
            ? {
                position: window.getComputedStyle(bd).position,
                opacity: window.getComputedStyle(bd).opacity,
                pointerEvents: window.getComputedStyle(bd).pointerEvents,
                // If position != fixed at desktop, the bug is back
                isFixed: window.getComputedStyle(bd).position === "fixed",
              }
            : null;
        })(),
      };
    });

    const checks = [];
    // Sidebar at left, full height (on desktop).
    checks.push({
      name: `${vp.name}: sidebar.x === 0`,
      ok: layout.sidebar && layout.sidebar.x === 0,
      detail: `x=${layout.sidebar?.x}`,
    });
    checks.push({
      name: `${vp.name}: sidebar.width === 280`,
      ok: layout.sidebar && Math.round(layout.sidebar.width) === 280,
      detail: `width=${layout.sidebar?.width}`,
    });
    checks.push({
      name: `${vp.name}: sidebar.height === viewport.height (full row)`,
      ok: layout.sidebar && Math.round(layout.sidebar.height) === vp.height,
      detail: `height=${layout.sidebar?.height}`,
    });
    checks.push({
      name: `${vp.name}: chatPane.x === 280 (sidebar + 280)`,
      ok: layout.chatPane && layout.chatPane.x === 280,
      detail: `x=${layout.chatPane?.x}`,
    });
    checks.push({
      name: `${vp.name}: chatPane.width === viewport.width - 280`,
      ok:
        layout.chatPane &&
        Math.round(layout.chatPane.width) === vp.width - 280,
      detail: `width=${layout.chatPane?.width}`,
    });
    checks.push({
      name: `${vp.name}: chatPane.height === viewport.height (full row)`,
      ok:
        layout.chatPane &&
        Math.round(layout.chatPane.height) === vp.height,
      detail: `height=${layout.chatPane?.height}`,
    });
    // Backdrop must be position:fixed at desktop — if it isn't, the
    // ``.sidebar-backdrop`` rule has been moved back inside @media
    // (the exact bug this PR fixes).
    checks.push({
      name: `${vp.name}: .sidebar-backdrop is position:fixed`,
      ok: layout.backdropComputed?.isFixed === true,
      detail: `position=${layout.backdropComputed?.position}`,
    });

    for (const c of checks) {
      if (c.ok) {
        console.log(`  ✓ ${c.name}`);
        passed++;
      } else {
        console.log(`  ✗ ${c.name} — ${c.detail}`);
        failed++;
      }
    }

    await ctx.close();
  }

  // Screenshot for the record
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await ctx.newPage();
  await page.goto(`${BACKEND}/`, { waitUntil: "domcontentloaded" });
  try {
    await page.waitForSelector(".chat-header-loading", {
      state: "detached",
      timeout: 60_000,
    });
  } catch {}
  await page.screenshot({
    path: "D:/prep for work/Projects/YuanRAG/tests/e2e/output/v2.0.28.7-desktop.png",
  });

  // v2.0.28.8 — default theme assertion. A fresh-context visitor
  // (no localStorage `rag.theme` set) MUST resolve to ``light``
  // regardless of OS prefers-color-scheme. Pin this so a future
  // PR can't silently flip the default back to "system" without
  // the test catching it.
  const themeInfo = await page.evaluate(() => ({
    datasetTheme: document.documentElement.dataset.theme,
    stored: window.localStorage.getItem("rag.theme"),
    prefersDark: window.matchMedia("(prefers-color-scheme: dark)").matches,
    bodyBg: window
      .getComputedStyle(document.body)
      .getPropertyValue("background-color"),
  }));
  const themeChecks = [
    {
      name: "desktop-1440: <html data-theme=\"light\"> (v2.0.28.8 default)",
      ok: themeInfo.datasetTheme === "light",
      detail: `data-theme=${themeInfo.datasetTheme}`,
    },
    {
      name: "desktop-1440: body background is NOT a dark surface",
      // Heuristic: rgb sum > 600 means a light surface (mostly
      // high-channel values like #ffffff). Dark themes bottom out
      // around rgb(20-30) per channel, sum < 100. We don't pin a
      // specific color because the design tokens are independent
      // of this PR — we only assert the theme actually applied.
      ok: (() => {
        const m = /rgb\((\d+),\s*(\d+),\s*(\d+)\)/.exec(themeInfo.bodyBg);
        if (!m) return false;
        const [r, g, b] = [m[1], m[2], m[3]].map(Number);
        return r + g + b > 600;
      })(),
      detail: `body bg=${themeInfo.bodyBg}`,
    },
  ];
  for (const c of themeChecks) {
    if (c.ok) {
      console.log(`  ✓ ${c.name}`);
      passed++;
    } else {
      console.log(`  ✗ ${c.name} — ${c.detail}`);
      failed++;
    }
  }
  await ctx.close();

  await browser.close();
  console.log(
    `\n>>> Layer 5 v2.0.28.7 verifier: ${passed} passed, ${failed} failed`
  );
  if (failed > 0) process.exit(1);
})().catch((e) => {
  console.error("Error:", e.message);
  process.exit(1);
});
