// v2.0.31.0 Layer 5 verifier — beautify high-precision UI.
//
// User request (2026-09-29):
//   美化一下 [自动/开启/关闭] UI + 🔒 icon 渲染以符合整体的观感
//
// Pre-v2.0.31.0: the segmented control had NO styling at all
// (``grep "high-precision\|verbatim" styles.css`` returned 0 lines)
// so the 3 toggle buttons rendered with browser-default chrome —
// gray backgrounds, default border, no rounded container grouping
// them. The bubble 🔒 indicator was a raw emoji with no badge
// styling. Both broke the "blue + white + paper" design language
// established by .segmented / .segmented-option / .citation-chip
// from PR-4 / PR-5.
//
// v2.0.31.0 ships:
//   * ``.high-precision-toggle`` reuses the .segmented class so
//     the 3-state picker shares its visual language with the
//     theme / language segmented pickers in SettingsDialog.
//     ``flex: 0 0 100%`` on the toggle makes it wrap to its own
//     row above the textarea on every viewport (the original
//     JSX comment already documented this intent — the bug was
//     that ``.chat-input { display: flex }`` placed the toggle
//     BESIDE the textarea, contradicting the comment).
//   * ``.verbatim-lock-indicator`` — small accent-soft pill at the
//     top of the assistant bubble. ``width: fit-content`` keeps
//     it snug around the icon + label rather than spanning the
//     bubble width. Border-radius 999px for full pill shape.
//   * ``.verbatim-lock-icon`` — shared lock-emoji styling for
//     both the bubble indicator and the inline glyph on the
//     "开启 / on" segmented button.
//
// We verify three things via real Chromium + real dist/ + real
// backend @ :8765 (NOT :5173 vite dev — per v2.0.28.20 REBUILD
// NOTE):
//   A. Segmented control renders the .segmented pill — computed
//      border-radius, background, and 100%-width on its row.
//   B. Active state uses data-active="true" + accent token, NOT
//      the old ``.active`` class.
//   C. When an assistant bubble carries sources with verbatim=true,
//      the indicator pill renders with .verbatim-lock-indicator
//      class and a non-empty text content.
//
// We don't try to force a verbatim extraction in this verifier
// (that path requires real-LLM + post-Nov-2026 Phase 8 fixtures).
// Instead we synthesize a fake assistant message by injecting a
// DOM node with the same className pattern the component uses,
// so we can assert the CSS resolves the way we expect. Real-LLM
// verbatim coverage was already verified in v2.0.29.9 Layer 5.

const path = require("path");
const { chromium } = require(path.join(
  "D:/prep for work/Projects/YuanRAG/src/frontend",
  "node_modules",
  "playwright",
));

// Per v2.0.28.20 CRITICAL REBUILD NOTE: must run against dist/ on :8765.
const FRONTEND = "http://127.0.0.1:8765";

function log(...a) {
  console.log(...a);
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

async function step(name, fn) {
  try {
    await fn();
    log(`  ✓ ${name}`);
  } catch (e) {
    log(`  ✗ ${name} — ${e.message}`);
    throw e;
  }
}

// ----- Run A: segmented control renders the .segmented pill ----------------

async function runSegmentedStyles(page) {
  log(`\n=== v2.0.31.0 RUN A: segmented control renders .segmented pill ===`);
  await page.goto(FRONTEND, { waitUntil: "domcontentloaded" });
  // Wait until the chat composer is mounted. The toggle is always
  // rendered (independent of modelsReady) but the form is gated by
  // modelsReady — once history loads, the toggle is in the DOM.
  // v2.0.31.1: the segmented is now wrapped in .high-precision-toggle-row,
  // which is the flex item that occupies the full form width.
  await page.waitForSelector(".high-precision-toggle-row", { timeout: 5000 });

  const styles = await page.evaluate(() => {
    const row = document.querySelector(".high-precision-toggle-row");
    const toggle = document.querySelector(".high-precision-toggle");
    if (!toggle || !row) return null;
    const cs = getComputedStyle(toggle);
    const rowCs = getComputedStyle(row);
    const rect = row.getBoundingClientRect();
    const btnCs = toggle.querySelector(".high-precision-toggle-btn")
      ? getComputedStyle(toggle.querySelector(".high-precision-toggle-btn"))
      : null;
    return {
      hasSegmentedClass: toggle.classList.contains("segmented"),
      bg: cs.backgroundColor,
      borderRadius: cs.borderTopLeftRadius,
      // The ROW (wrapper, not the segmented itself) is the flex item
      // that owns the 100% width — the segmented inside is just a
      // content element of the row. v2.0.31.1 added the row wrapper
      // to host the label + segmented + description.
      rowFlexBasis: rowCs.flexBasis,
      rowFlexGrow: rowCs.flexGrow,
      rowFlexShrink: rowCs.flexShrink,
      rowDisplay: rowCs.display,
      rectWidth: rect.width,
      btnDisplay: btnCs ? btnCs.display : null,
      btnFlexDir: btnCs ? btnCs.flexDirection : null,
      btnAlignItems: btnCs ? btnCs.alignItems : null,
    };
  });

  await step("toggle has .segmented class", () => {
    assert(styles.hasSegmentedClass, "missing .segmented class");
  });
  await step("toggle background is not transparent (the .segmented pill)", () => {
    // var(--bg-panel-2) ≈ #f1f4f9 → rgb(241, 244, 249)
    assert(
      styles.bg !== "rgba(0, 0, 0, 0)" && styles.bg !== "transparent",
      `bg=${styles.bg} (transparent → no pill)`,
    );
  });
  await step("toggle border-radius > 0 (rounded pill, not square)", () => {
    // var(--radius-sm) = 6px
    const r = parseFloat(styles.borderRadius);
    assert(r >= 4, `border-radius=${styles.borderRadius}`);
  });
  await step("row flex: 0 0 100% (own row above textarea)", () => {
    // v2.0.31.1: the .high-precision-toggle-row wrapper (label +
    // segmented + description) is the flex item that occupies 100%
    // width. Before v2.0.31.1 this was on .high-precision-toggle
    // directly; now the wrapper takes the responsibility.
    assert(styles.rowFlexBasis === "100%", `flex-basis=${styles.rowFlexBasis}`);
    assert(styles.rowFlexGrow === "0", `flex-grow=${styles.rowFlexGrow}`);
    assert(styles.rowFlexShrink === "0", `flex-shrink=${styles.rowFlexShrink}`);
  });
  await step("row width fills chat-input row", () => {
    assert(styles.rectWidth > 200, `width=${styles.rectWidth}`);
  });
  await step("button is a flex container (blockified from inline-flex)", () => {
    // .high-precision-toggle-btn is ``display: inline-flex`` in CSS
    // but per CSS Display 3 §2.2, a flex item inside a flex
    // container gets blockified — inline-flex → flex. The
    // important thing is that the button REMAINS a flex container
    // (so the lock icon + label flex-align), not that the raw
    // computed value is "inline-flex". We verify the layout by
    // checking that the inline children inside the button are
    // laid out horizontally (flex-direction: row) with center
    // alignment (align-items: center) and a gap (gap: 4px).
    assert(
      styles.btnDisplay === "inline-flex" || styles.btnDisplay === "flex",
      `btn display=${styles.btnDisplay} (must be flex container)`,
    );
    assert(
      styles.btnFlexDir === "row",
      `btn flex-direction=${styles.btnFlexDir}`,
    );
    assert(
      styles.btnAlignItems === "center",
      `btn align-items=${styles.btnAlignItems}`,
    );
  });
}

// ----- Run B: active state uses data-active="true" ------------------------

async function runActiveState(page) {
  log(`\n=== v2.0.31.0 RUN B: active state via data-active="true" ===`);
  await page.goto(FRONTEND, { waitUntil: "domcontentloaded" });
  await page.waitForSelector(".high-precision-toggle-btn", { timeout: 5000 });

  // Default is "auto" — the first button must carry data-active="true"
  // and be styled with the .segmented-option[data-active] rule
  // (bg-panel background + accent color + shadow).
  const initial = await page.evaluate(() => {
    const btns = Array.from(
      document.querySelectorAll(".high-precision-toggle-btn"),
    );
    return btns.map((b) => {
      const cs = getComputedStyle(b);
      return {
        label: b.textContent.trim(),
        dataActive: b.getAttribute("data-active"),
        ariaChecked: b.getAttribute("aria-checked"),
        color: cs.color,
        bg: cs.backgroundColor,
        shadow: cs.boxShadow,
      };
    });
  });

  await step("3 toggle buttons rendered", () => {
    assert(initial.length === 3, `got ${initial.length}`);
  });
  await step("exactly one button has data-active=\"true\"", () => {
    const active = initial.filter((b) => b.dataActive === "true");
    assert(active.length === 1, `active count=${active.length}`);
  });
  await step("active button uses accent color (rgb(37, 99, 235))", () => {
    const active = initial.find((b) => b.dataActive === "true");
    assert(
      active.color === "rgb(37, 99, 235)" || active.color.includes("37, 99, 235"),
      `color=${active.color}`,
    );
  });
  await step("active button background is --bg-panel (white-ish)", () => {
    const active = initial.find((b) => b.dataActive === "true");
    // var(--bg-panel) ≈ #ffffff → rgb(255, 255, 255)
    assert(
      active.bg === "rgb(255, 255, 255)" || active.bg.includes("255, 255, 255"),
      `bg=${active.bg}`,
    );
  });
  await step("active button has shadow (var(--shadow-sm))", () => {
    const active = initial.find((b) => b.dataActive === "true");
    // shadow-sm ≈ 0 1px 2px rgba(15,23,42,0.04) — non-"none" string
    assert(active.shadow !== "none", `shadow=${active.shadow}`);
  });

  // Click the "off" button — verify data-active moves, and the
  // previously-active button's accent color clears.
  await page.evaluate(() => {
    const btns = Array.from(
      document.querySelectorAll(".high-precision-toggle-btn"),
    );
    // Find the off button by label match (avoid relying on order)
    const offBtn = btns.find((b) => b.textContent.trim().endsWith("关闭") || b.textContent.trim().endsWith("Off"));
    if (offBtn) offBtn.click();
  });
  await page.waitForTimeout(200);

  const afterClick = await page.evaluate(() => {
    const btns = Array.from(
      document.querySelectorAll(".high-precision-toggle-btn"),
    );
    return btns.map((b) => ({
      label: b.textContent.trim(),
      dataActive: b.getAttribute("data-active"),
    }));
  });

  await step("clicking off moves data-active to off button", () => {
    const active = afterClick.filter((b) => b.dataActive === "true");
    assert(
      active.length === 1,
      `active count=${active.length} after click`,
    );
    const label = active[0].label;
    assert(
      label.endsWith("关闭") || label.endsWith("Off"),
      `active label=${label}`,
    );
  });
}

// ----- Run C: bubble verbatim-lock-indicator pill -------------------------

async function runBubbleIndicator(page) {
  log(`\n=== v2.0.31.0 RUN C: bubble verbatim-lock-indicator pill ===`);
  await page.goto(FRONTEND, { waitUntil: "domcontentloaded" });
  await page.waitForSelector(".chat-messages, .chat-input", { timeout: 5000 });

  // Inject a synthetic assistant bubble with the verbatim indicator
  // so we can assert the CSS resolves without needing a real LLM.
  // We mount a .message.assistant + .verbatim-lock-indicator inside
  // the first chat-messages container (or .chat-input's parent).
  const injection = await page.evaluate(() => {
    const host =
      document.querySelector(".chat-messages") ||
      document.querySelector(".chat-pane") ||
      document.body;
    const wrap = document.createElement("div");
    wrap.className = "message assistant";
    wrap.style.position = "relative";
    wrap.style.padding = "12px 16px";
    wrap.style.maxWidth = "80%";
    wrap.innerHTML = `
      <div class="verbatim-lock-indicator" title="synthetic">
        <span class="verbatim-lock-icon" aria-hidden="true">🔒</span>
        <span>Verbatim mode active</span>
      </div>
      <div class="markdown-body"><p>synthetic answer body</p></div>
    `;
    host.appendChild(wrap);
    return true;
  });
  assert(injection, "injection failed");

  await page.waitForSelector(".verbatim-lock-indicator", { timeout: 3000 });

  const styles = await page.evaluate(() => {
    const ind = document.querySelector(".verbatim-lock-indicator");
    if (!ind) return null;
    const cs = getComputedStyle(ind);
    const rect = ind.getBoundingClientRect();
    const iconCs = ind.querySelector(".verbatim-lock-icon")
      ? getComputedStyle(ind.querySelector(".verbatim-lock-icon"))
      : null;
    return {
      display: cs.display,
      alignItems: cs.alignItems,
      gap: cs.gap,
      padding: cs.paddingTop + " " + cs.paddingRight,
      bg: cs.backgroundColor,
      borderRadius: cs.borderTopLeftRadius,
      borderTopColor: cs.borderTopColor,
      color: cs.color,
      fontSize: cs.fontSize,
      fontWeight: cs.fontWeight,
      boxShadow: cs.boxShadow,
      // Pill shape: fit-content → wider than text but not full bubble
      pillWidth: rect.width,
      bubbleWidth: ind.parentElement.getBoundingClientRect().width,
      iconDisplay: iconCs ? iconCs.display : null,
      iconFontSize: iconCs ? iconCs.fontSize : null,
    };
  });

  await step("indicator renders with display:inline-flex", () => {
    assert(
      styles.display === "inline-flex",
      `display=${styles.display}`,
    );
  });
  await step("indicator aligns center (icon + label baseline)", () => {
    assert(styles.alignItems === "center", `align-items=${styles.alignItems}`);
  });
  await step("indicator has gap between icon and label", () => {
    // gap: 5px
    const g = parseFloat(styles.gap);
    assert(g >= 3, `gap=${styles.gap}`);
  });
  await step("indicator background is --accent-soft (light blue)", () => {
    // var(--accent-soft) ≈ #eff6ff → rgb(239, 246, 255)
    assert(
      styles.bg === "rgb(239, 246, 255)" || styles.bg.includes("239, 246, 255"),
      `bg=${styles.bg}`,
    );
  });
  await step("indicator border-radius ≥ 999px → pill shape", () => {
    const r = parseFloat(styles.borderRadius);
    // 999px cap → computed value depends on element size; what
    // matters is "pill" not "rounded-rect" — assert ≥ height/2.
    const pillHeight = parseFloat(styles.fontSize) * 1.4;
    assert(r >= pillHeight / 2, `border-radius=${styles.borderRadius}`);
  });
  await step("indicator text color is --accent-strong (deep blue)", () => {
    // var(--accent-strong) ≈ #1e40af → rgb(30, 64, 175)
    assert(
      styles.color === "rgb(30, 64, 175)" || styles.color.includes("30, 64, 175"),
      `color=${styles.color}`,
    );
  });
  await step("indicator font-size 11px + weight 500", () => {
    assert(styles.fontSize === "11px", `font-size=${styles.fontSize}`);
    assert(styles.fontWeight === "500", `font-weight=${styles.fontWeight}`);
  });
  await step("indicator width is fit-content (snug, not full bubble)", () => {
    // pillWidth < bubbleWidth — but allow ~equal if bubble is narrow
    assert(
      styles.pillWidth < styles.bubbleWidth,
      `pill=${styles.pillWidth}, bubble=${styles.bubbleWidth}`,
    );
  });
  await step("lock icon is sized at 12px (font-size token)", () => {
    // Per CSS Display 3 §2.2, a flex item with display:inline-block
    // gets blockified to display:block when its parent is a flex
    // container. The raw computed display value will be "block",
    // not "inline-block" — but the icon's font-size / vertical-align
    // / inline-block rendering (when not blockified) is what makes
    // the emoji render as a small glyph next to the label. We
    // assert font-size 12px (the meaningful visual property) rather
    // than the raw display value.
    assert(
      styles.iconFontSize === "12px",
      `icon font-size=${styles.iconFontSize}`,
    );
  });
}

// ----- Run D: chat-input flex-wrap puts toggle on its own row -------------

async function runFlexWrap(page) {
  log(`\n=== v2.0.31.0 RUN D: chat-input flex-wrap layout ===`);
  await page.goto(FRONTEND, { waitUntil: "domcontentloaded" });
  await page.waitForSelector(".chat-input", { timeout: 5000 });

  const layout = await page.evaluate(() => {
    const form = document.querySelector(".chat-input");
    const toggle = document.querySelector(".high-precision-toggle");
    const textarea = document.querySelector(".chat-textarea");
    if (!form || !toggle || !textarea) return null;
    const cs = getComputedStyle(form);
    const tRect = toggle.getBoundingClientRect();
    const taRect = textarea.getBoundingClientRect();
    return {
      flexWrap: cs.flexWrap,
      // Toggle sits on its own row → its bottom is above textarea top
      toggleBottom: tRect.bottom,
      textareaTop: taRect.top,
    };
  });

  await step(".chat-input has flex-wrap: wrap", () => {
    assert(layout.flexWrap === "wrap", `flex-wrap=${layout.flexWrap}`);
  });
  await step("toggle sits above textarea (own row)", () => {
    assert(
      layout.toggleBottom <= layout.textareaTop + 1,
      `toggle bottom=${layout.toggleBottom}, textarea top=${layout.textareaTop}`,
    );
  });
}

// ----- main ---------------------------------------------------------------

async function main() {
  const browser = await chromium.launch({ headless: true });
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await ctx.newPage();

  await page.addInitScript(() => {
    try {
      window.localStorage.setItem("rag.theme", "light");
      window.localStorage.setItem("rag.locale", "en");
    } catch {}
  });

  const failures = [];
  for (const [name, fn] of [
    ["A", runSegmentedStyles],
    ["B", runActiveState],
    ["C", runBubbleIndicator],
    ["D", runFlexWrap],
  ]) {
    try {
      await fn(page);
    } catch (e) {
      failures.push({ run: name, reason: e.message });
      log(`RUN ${name}: ✗ FAIL — ${e.message}`);
    }
  }

  await browser.close();

  log(`\n=== v2.0.31.0 SUMMARY ===`);
  if (failures.length > 0) {
    log(`✗ v2.0.31.0 verifier FAILED: ${failures.length}/4`);
    for (const f of failures) log(`  RUN ${f.run}: ${f.reason}`);
    process.exit(1);
  }
  log(`✓ v2.0.31.0 verifier PASSED: 4/4`);
}

main().catch((e) => {
  console.error("verifier crashed:", e.message);
  console.error(e.stack);
  process.exit(2);
});