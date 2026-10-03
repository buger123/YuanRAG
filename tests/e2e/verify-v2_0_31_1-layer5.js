// v2.0.31.1 Layer 5 verifier — first-time discoverability for
// the high-precision toggle.
//
// User request (2026-09-29):
//   想办法需要让用户第一次使用时就知道这个自动/开启/关闭指的是什么,
//   优化用户体验
//
// Pre-v2.0.31.1 problem:
//   The 3-state [自动/开启/关闭] segmented control had no visible
//   label and no inline description. The only explanation was the
//   ``title`` attribute on each button (browser tooltip), which is
//     (1) hover-only — completely invisible on touch devices
//     (2) not screen-reader-friendly (titles are inconsistent)
//     (3) zero visual affordance — a first-time user sees "自动 | 开启 | 关闭"
//         and has no idea what each option means.
//
// v2.0.31.1 ships:
//     * ``.high-precision-toggle-label`` — small label "🔒 高精确模式"
//       above the segmented so the user knows what the toggle IS.
//     * ``.high-precision-toggle-description`` — dynamic text below
//       the segmented that updates when the user clicks each mode,
//       explaining what the CURRENT mode does (no hover required).
//     * ``aria-live="polite"`` on the description so screen readers
//       announce the new description on selection.
//     * Existing per-button ``title`` tooltips remain for desktop
//       hover users (no regression).
//
// We verify 4 things via real Chromium + real dist/ + real backend
// (NOT :5173 vite dev per v2.0.28.20 REBUILD NOTE):
//   A. Label is visible above segmented — user sees "🔒 高精确模式"
//      even on first load (no hover / no interaction required).
//   B. Description is visible below segmented — shows the CURRENT
//      mode's hint text (initially the auto-mode hint).
//   C. Clicking each mode updates the description in place — proves
//      the dynamic update works.
//   D. Label + description + segmented are all in the DOM (not just
//      tooltips) so first-time mobile / touch / screen-reader users
//      get the same information.

const path = require("path");
const { chromium } = require(path.join(
  "D:/prep for work/Projects/YuanRAG/src/frontend",
  "node_modules",
  "playwright",
));

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

// ----- Run A: label is visible above segmented ----------------------------

async function runLabelVisible(page) {
  log(`\n=== v2.0.31.1 RUN A: label is visible above segmented ===`);
  await page.goto(FRONTEND, { waitUntil: "domcontentloaded" });
  await page.waitForSelector(".high-precision-toggle-row", { timeout: 5000 });

  const layout = await page.evaluate(() => {
    const row = document.querySelector(".high-precision-toggle-row");
    if (!row) return null;
    const label = row.querySelector(".high-precision-toggle-label");
    const segmented = row.querySelector(".high-precision-toggle");
    if (!label || !segmented) return null;
    const labelRect = label.getBoundingClientRect();
    const segRect = segmented.getBoundingClientRect();
    const labelCs = getComputedStyle(label);
    return {
      labelText: label.textContent.trim(),
      // Strip the lock emoji glyph for the "non-empty" check.
      labelVisible: labelRect.width > 0 && labelRect.height > 0,
      labelDisplay: labelCs.display,
      // Label must sit ABOVE the segmented (smaller `top`).
      labelTop: labelRect.top,
      segTop: segRect.top,
      // Label has visible icon + text (not empty).
      hasIcon: !!label.querySelector(".verbatim-lock-icon"),
    };
  });

  await step("label element is visible (non-zero size)", () => {
    assert(layout.labelVisible, "label has zero size");
  });
  await step("label has a lock icon", () => {
    assert(layout.hasIcon, "label missing .verbatim-lock-icon");
  });
  await step("label has visible text (not just icon)", () => {
    // Strip the icon glyph — the rest must be non-empty.
    const textOnly = layout.labelText.replace(/🔒/g, "").trim();
    assert(textOnly.length > 0, `label text='${layout.labelText}'`);
  });
  await step("label sits ABOVE the segmented control", () => {
    assert(
      layout.labelTop < layout.segTop,
      `label.top=${layout.labelTop}, segmented.top=${layout.segTop}`,
    );
  });
  await step("label uses inline-flex (icon + text align)", () => {
    assert(
      layout.labelDisplay === "inline-flex" || layout.labelDisplay === "flex",
      `label display=${layout.labelDisplay}`,
    );
  });
}

// ----- Run B: description is visible below segmented ---------------------

async function runDescriptionVisible(page) {
  log(`\n=== v2.0.31.1 RUN B: description is visible below segmented ===`);
  await page.goto(FRONTEND, { waitUntil: "domcontentloaded" });
  await page.waitForSelector(".high-precision-toggle-description", {
    timeout: 5000,
  });

  const desc = await page.evaluate(() => {
    const d = document.querySelector(".high-precision-toggle-description");
    if (!d) return null;
    const seg = document.querySelector(".high-precision-toggle");
    const label = document.querySelector(".high-precision-toggle-label");
    const rect = d.getBoundingClientRect();
    const segRect = seg.getBoundingClientRect();
    const labelRect = label.getBoundingClientRect();
    const cs = getComputedStyle(d);
    return {
      text: d.textContent.trim(),
      // Strip the optional aria-live prefix.
      visible: rect.width > 0 && rect.height > 0,
      // Description must sit BELOW the segmented.
      top: rect.top,
      segBottom: segRect.bottom,
      // Must NOT overlap the segmented.
      ariaLive: d.getAttribute("aria-live"),
      fontSize: cs.fontSize,
      color: cs.color,
      // Has min-height to prevent layout shift between modes.
      minHeight: cs.minHeight,
    };
  });

  await step("description element exists with non-empty text", () => {
    assert(desc.text.length > 0, `description text='${desc.text}'`);
  });
  await step("description is visible (non-zero size)", () => {
    assert(desc.visible, "description has zero size");
  });
  await step("description sits BELOW the segmented control", () => {
    assert(
      desc.top >= desc.segBottom - 1,
      `description.top=${desc.top}, segmented.bottom=${desc.segBottom}`,
    );
  });
  await step("description has aria-live=polite (screen-reader announce)", () => {
    assert(desc.ariaLive === "polite", `aria-live=${desc.ariaLive}`);
  });
  await step("description font-size is 11px (subtle, dim helper text)", () => {
    assert(desc.fontSize === "11px", `font-size=${desc.fontSize}`);
  });
  await step("description uses --text-dim color (not loud)", () => {
    // var(--text-dim) ≈ #6b7280 → rgb(107, 114, 128)
    assert(
      desc.color === "rgb(107, 114, 128)" || desc.color.includes("107, 114, 128"),
      `color=${desc.color}`,
    );
  });
  await step("description has min-height (no layout shift on change)", () => {
    // 1.5em on 11px = 16.5px. Allow any non-zero min-height.
    const mh = parseFloat(desc.minHeight);
    assert(mh > 0, `min-height=${desc.minHeight}`);
  });
}

// ----- Run C: clicking each mode updates the description ------------------

async function runDescriptionUpdates(page) {
  log(`\n=== v2.0.31.1 RUN C: description updates on mode click ===`);
  await page.goto(FRONTEND, { waitUntil: "domcontentloaded" });
  await page.waitForSelector(".high-precision-toggle-description", {
    timeout: 5000,
  });

  // Read the initial description (should be the auto-mode hint).
  const initial = await page.evaluate(() => {
    return document.querySelector(".high-precision-toggle-description")
      .textContent.trim();
  });

  await step("initial description is non-empty", () => {
    assert(initial.length > 0, `text='${initial}'`);
  });

  // Click each mode in turn and verify the description CHANGES.
  // We capture both the new text and the active button label so we
  // can assert the mapping is correct (not just that the text changed).
  //
  // React state updates are async (batched), so we wait for the DOM
  // to settle after each click before reading the description.
  const modeResults = [];
  for (const mode of ["auto", "on", "off"]) {
    // Click + wait for re-render. We poll until the active button's
    // data-active matches the requested mode (with a 1s ceiling).
    const clicked = await page.evaluate((mode) => {
      const btns = Array.from(
        document.querySelectorAll(".high-precision-toggle-btn"),
      );
      const labels = mode === "auto"
        ? ["自动", "Auto"]
        : mode === "on"
          ? ["开启", "On"]
          : ["关闭", "Off"];
      const btn = btns.find((b) =>
        labels.some((l) => b.textContent.trim().endsWith(l)),
      );
      if (!btn) return null;
      btn.click();
      return btn.textContent.trim();
    }, mode);
    assert(clicked, `button for mode=${mode} not found`);

    // Wait for React to re-render with the new active state.
    await page.waitForFunction(
      (mode) => {
        const btns = Array.from(
          document.querySelectorAll(".high-precision-toggle-btn"),
        );
        // After click, exactly one button must have data-active="true"
        // and it must be the requested mode's button.
        const active = btns.filter((b) => b.getAttribute("data-active") === "true");
        if (active.length !== 1) return false;
        const expected = mode === "auto" ? ["自动", "Auto"]
          : mode === "on" ? ["开启", "On"]
          : ["关闭", "Off"];
        return expected.some((l) => active[0].textContent.trim().endsWith(l));
      },
      mode,
      { timeout: 1000 },
    );

    const description = await page.evaluate(() =>
      document
        .querySelector(".high-precision-toggle-description")
        .textContent.trim(),
    );
    modeResults.push({ mode, clicked, description });
  }

  // Assert the three descriptions are all DISTINCT — proves the
  // dynamic update is in place, not just a static label.
  const texts = modeResults.map((r) => r.description);
  await step("3 mode clicks produced 3 distinct descriptions", () => {
    const unique = new Set(texts);
    assert(
      unique.size === 3,
      `got ${unique.size} unique: ${JSON.stringify(texts)}`,
    );
  });

  // Assert each description is a meaningful length (>10 chars).
  await step("each description has substantive content (>10 chars)", () => {
    for (const r of modeResults) {
      assert(
        r.description.length > 10,
        `mode=${r.mode} description='${r.description}'`,
      );
    }
  });

  // Assert the initial (auto) description matches what we got on
  // the auto click — i.e. clicking the SAME mode we started on is a
  // no-op (regression guard).
  await step("re-clicking active mode is a no-op (text stable)", () => {
    // Click auto again — text should be the same as initial.
    const afterAuto = modeResults[0].description;
    assert(
      afterAuto === initial,
      `initial='${initial}', after-auto='${afterAuto}'`,
    );
  });
}

// ----- Run D: discoverability is non-tooltip-only ------------------------

async function runDiscoverabilityNotTooltip(page) {
  log(`\n=== v2.0.31.1 RUN D: discoverability NOT tooltip-only ===`);
  await page.goto(FRONTEND, { waitUntil: "domcontentloaded" });
  await page.waitForSelector(".high-precision-toggle-row", { timeout: 5000 });

  // First-time-user simulation: zero interaction (no click, no hover,
  // no keyboard focus). The user should STILL see enough info to
  // understand the toggle.
  const visible = await page.evaluate(() => {
    const row = document.querySelector(".high-precision-toggle-row");
    const label = row.querySelector(".high-precision-toggle-label");
    const desc = row.querySelector(".high-precision-toggle-description");
    const seg = row.querySelector(".high-precision-toggle");
    return {
      labelText: label ? label.textContent.trim() : "",
      descText: desc ? desc.textContent.trim() : "",
      segButtonCount: seg
        ? seg.querySelectorAll(".high-precision-toggle-btn").length
        : 0,
    };
  });

  await step("first-time user sees the feature LABEL without hover", () => {
    assert(visible.labelText.length > 0, `label='${visible.labelText}'`);
  });
  await step("first-time user sees the CURRENT mode DESCRIPTION without hover", () => {
    assert(visible.descText.length > 0, `desc='${visible.descText}'`);
  });
  await step("first-time user sees the 3 segmented BUTTONS without hover", () => {
    assert(visible.segButtonCount === 3, `count=${visible.segButtonCount}`);
  });

  // Now simulate the keyboard-only / screen-reader case: focus the
  // first button and read aria-checked + role. The radiogroup
  // aria-label is the same as the label text, so SR users hear it.
  const a11y = await page.evaluate(async () => {
    const seg = document.querySelector(".high-precision-toggle");
    const firstBtn = seg.querySelector(".high-precision-toggle-btn");
    firstBtn.focus();
    return {
      role: firstBtn.getAttribute("role"),
      ariaChecked: firstBtn.getAttribute("aria-checked"),
      radiogroupRole: seg.getAttribute("role"),
      radiogroupLabel: seg.getAttribute("aria-label"),
    };
  });

  await step("button has role=radio", () => {
    assert(a11y.role === "radio", `role=${a11y.role}`);
  });
  await step("button has aria-checked (active state to SR)", () => {
    assert(
      a11y.ariaChecked === "true" || a11y.ariaChecked === "false",
      `aria-checked=${a11y.ariaChecked}`,
    );
  });
  await step("container has role=radiogroup", () => {
    assert(a11y.radiogroupRole === "radiogroup", `role=${a11y.radiogroupRole}`);
  });
  await step("container has aria-label describing the group", () => {
    assert(
      a11y.radiogroupLabel && a11y.radiogroupLabel.length > 0,
      `aria-label='${a11y.radiogroupLabel}'`,
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
      window.localStorage.setItem("rag.locale", "zh");
    } catch {}
  });

  const failures = [];
  for (const [name, fn] of [
    ["A", runLabelVisible],
    ["B", runDescriptionVisible],
    ["C", runDescriptionUpdates],
    ["D", runDiscoverabilityNotTooltip],
  ]) {
    try {
      await fn(page);
    } catch (e) {
      failures.push({ run: name, reason: e.message });
      log(`RUN ${name}: ✗ FAIL — ${e.message}`);
    }
  }

  await browser.close();

  log(`\n=== v2.0.31.1 SUMMARY ===`);
  if (failures.length > 0) {
    log(`✗ v2.0.31.1 verifier FAILED: ${failures.length}/4`);
    for (const f of failures) log(`  RUN ${f.run}: ${f.reason}`);
    process.exit(1);
  }
  log(`✓ v2.0.31.1 verifier PASSED: 4/4`);
}

main().catch((e) => {
  console.error("verifier crashed:", e.message);
  console.error(e.stack);
  process.exit(2);
});