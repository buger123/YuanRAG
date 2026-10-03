// Deep layout diagnostic
const path = require("path");
const { chromium } = require(path.join(
  "D:/prep for work/Projects/YuanRAG/src/frontend",
  "node_modules",
  "playwright"
));

const BACKEND = "http://127.0.0.1:8765";

(async () => {
  const browser = await chromium.launch({ headless: true });
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await ctx.newPage();
  await page.goto(`${BACKEND}/`, { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(2000);

  const dump = await page.evaluate(() => {
    function cs(el) {
      if (!el) return null;
      const c = window.getComputedStyle(el);
      return {
        display: c.display,
        gridTemplateColumns: c.gridTemplateColumns,
        gridTemplateRows: c.gridTemplateRows,
        gridAutoFlow: c.gridAutoFlow,
        gridAutoRows: c.gridAutoRows,
        gridColumn: c.gridColumn,
        gridRow: c.gridRow,
        width: c.width,
        height: c.height,
        box: el.getBoundingClientRect().toJSON(),
        order: c.order,
        alignSelf: c.alignSelf,
        justifySelf: c.justifySelf,
        position: c.position,
      };
    }
    const allApps = Array.from(document.querySelectorAll(".app"));
    const out = { numApps: allApps.length, apps: allApps.map(cs) };
    out.sidebar = cs(document.querySelector(".sidebar"));
    out.chatPane = cs(document.querySelector(".chat-pane"));
    out.chatMain = cs(document.querySelector(".chat, main"));
    out.sidebarParent = (() => {
      const s = document.querySelector(".sidebar");
      return s ? { tag: s.parentElement?.tagName, class: s.parentElement?.className, cs: cs(s.parentElement) } : null;
    })();
    out.chatPaneParent = (() => {
      const c = document.querySelector(".chat-pane");
      return c ? { tag: c.parentElement?.tagName, class: c.parentElement?.className, cs: cs(c.parentElement) } : null;
    })();
    // Get raw DOM order of children inside .app
    const appEl = allApps[0];
    out.appChildren = appEl
      ? Array.from(appEl.children).map((el) => ({
          tag: el.tagName,
          class: el.className,
          cs: {
            gridColumn: window.getComputedStyle(el).gridColumn,
            gridRow: window.getComputedStyle(el).gridRow,
            order: window.getComputedStyle(el).order,
            display: window.getComputedStyle(el).display,
            box: el.getBoundingClientRect().toJSON(),
          },
        }))
      : [];
    return out;
  });
  console.log(JSON.stringify(dump, null, 2));
  await browser.close();
})().catch((e) => {
  console.error("Error:", e.message);
  process.exit(1);
});
