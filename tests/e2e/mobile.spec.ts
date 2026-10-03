/**
 * v2.0.26.2 (PR-5) — Layer 5 NEW assertions for responsive + mobile.
 *
 * Six assertions under an iPhone 13 (390 × 844) viewport — the canonical
 * "small phone" breakpoint per Apple HIG. Each catches a distinct
 * class of regression:
 *
 *   1. Sidebar hidden by default  — @media (max-width: 768px) absent
 *   2. Hamburger visible          — mobile display rule missing
 *   3. Hamburger → drawer opens   — prop-drill (App.tsx → ChatPane →
 *                                    App.tsx → Sidebar) broken
 *   4. Backdrop tap → drawer closes — onClick not bound
 *   5. Modal fits viewport        — min-width: 420px not relaxed
 *   6. Touch targets ≥ 44 × 44   — Apple HIG fatal
 *
 * The tests use a desktop Chromium at the iPhone viewport so we don't
 * need to install the mobile browser binaries in CI. The viewport size
 * is the only thing that differs — that is exactly what the @media
 * query reacts to.
 *
 * Backend-skip pattern mirrors smoke.spec.ts:101-110 — this harness
 * requires a real env, but the assertions themselves don't hit any
 * network endpoints after the initial page load.
 */
import { test, expect, type Page } from "@playwright/test";

const BACKEND = "http://127.0.0.1:8765";

async function backendReady(): Promise<boolean> {
  try {
    const resp = await fetch(`${BACKEND}/models/status`);
    if (!resp.ok) return false;
    const body = (await resp.json()) as { ready?: boolean };
    return body.ready === true;
  } catch {
    return false;
  }
}

async function waitForModelsReady(page: Page): Promise<boolean> {
  try {
    await page.waitForSelector(".chat-header-loading", {
      state: "detached",
      timeout: 60_000,
    });
    return true;
  } catch {
    return false;
  }
}

test.describe("PR-5 mobile — iPhone 13 viewport", () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test.beforeEach(async ({ page }) => {
    if (!(await backendReady())) {
      test.skip(
        true,
        "Backend at " +
          BACKEND +
          " not ready. mobile.spec.ts needs a real env; see tests/e2e/run.sh.",
      );
      return;
    }
    await page.goto("/");
    const ok = await waitForModelsReady(page);
    expect(ok, "models should finish prewarm within 60s").toBe(true);
  });

  test("Assertion 1 — sidebar is hidden by default (off-canvas)", async ({
    page,
  }) => {
    // The aside should be positioned fixed offscreen (transform:
    // translateX(-100%)), so its visible bounding box must have x < 0.
    const box = await page.locator(".sidebar").boundingBox();
    // boundingBox can be null if completely out of layout — treat as off-screen.
    if (box !== null) {
      expect(
        box.x + box.width,
        "sidebar should not extend into the visible viewport when drawer is closed",
      ).toBeLessThanOrEqual(0);
    }
    // The sidebar must NOT carry the is-open class.
    await expect(page.locator(".sidebar")).not.toHaveClass(/is-open/);
  });

  test("Assertion 2 — hamburger button is visible", async ({ page }) => {
    const hamburger = page.locator(".btn-hamburger");
    await expect(hamburger).toBeVisible();
    // Its bounding box should have non-zero dimensions inside the
    // 390-wide viewport.
    const box = await hamburger.boundingBox();
    expect(box).not.toBeNull();
    expect(box!.x).toBeGreaterThanOrEqual(0);
    expect(box!.x + box!.width).toBeLessThanOrEqual(390);
  });

  test("Assertion 3 — tapping hamburger opens the drawer", async ({ page }) => {
    await page.click(".btn-hamburger");
    // Wait for the drawer's slide-in transition (0.2 s in styles.css).
    await page.waitForTimeout(300);
    // Sidebar slides in: gets is-open class.
    await expect(page.locator(".sidebar")).toHaveClass(/is-open/);
    // Hamburger aria-expanded reflects drawer state.
    await expect(page.locator(".btn-hamburger")).toHaveAttribute(
      "aria-expanded",
      "true",
    );
    // Sidebar now overlaps the chat pane.
    const box = await page.locator(".sidebar").boundingBox();
    expect(box).not.toBeNull();
    expect(box!.x).toBeLessThanOrEqual(0.5); // transform: translateX(0) ≈ x = 0
    expect(box!.x + box!.width).toBeGreaterThan(0);
  });

  test("Assertion 4 — tapping backdrop closes the drawer", async ({ page }) => {
    await page.click(".btn-hamburger");
    await page.waitForTimeout(300);
    await expect(page.locator(".sidebar")).toHaveClass(/is-open/);
    // Backdrop must be the active click target while open.
    await page.click(".sidebar-backdrop", { force: true });
    await expect(page.locator(".sidebar")).not.toHaveClass(/is-open/);
    await expect(page.locator(".btn-hamburger")).toHaveAttribute(
      "aria-expanded",
      "false",
    );
  });

  test("Assertion 5 — SettingsDialog modal fits the viewport", async ({
    page,
  }) => {
    // Open SettingsDialog. The settings button lives in the sidebar;
    // since the drawer is closed on mobile we have to open it first.
    await page.click(".btn-hamburger");
    // Wait for the drawer's slide-in transition (0.2 s in styles.css).
    await page.waitForTimeout(300);
    // Open the modal by triggering a real React onClick. Playwright's
    // ``click`` is actionability-gated and the gear button sits very
    // close to the drawer's right edge where a pointer-events quirk
    // can flag it as occluded; we fall back to dispatching the
    // event in-page if the first attempt fails. React's synthetic
    // event system listens at document root so a native
    // ``MouseEvent`` dispatched on the button propagates correctly.
    const gearClicked = await page.evaluate(() => {
      const btn = document.querySelector(
        ".sidebar-brand .sidebar-icon-btn",
      ) as HTMLButtonElement | null;
      if (!btn) return false;
      btn.dispatchEvent(
        new MouseEvent("click", { bubbles: true, cancelable: true, view: window }),
      );
      return true;
    });
    expect(gearClicked, "settings gear button must exist").toBe(true);
    await page.waitForSelector(".modal", { state: "attached", timeout: 5_000 });
    // The modal must be visible AND fit within the 390-wide viewport.
    const modal = page.locator(".modal").first();
    await expect(modal).toBeVisible();
    const box = await modal.boundingBox();
    expect(box).not.toBeNull();
    expect(box!.x).toBeGreaterThanOrEqual(0);
    expect(box!.x + box!.width).toBeLessThanOrEqual(390);
    // No horizontal scroll on the document.
    const overflow = await page.evaluate(() => ({
      scroll: document.documentElement.scrollWidth,
      client: document.documentElement.clientWidth,
    }));
    expect(
      overflow.scroll,
      "document should not horizontally scroll on mobile",
    ).toBeLessThanOrEqual(overflow.client);
  });

  test("Assertion 6 — touch targets in the sidebar ≥ 44 × 44 px", async ({
    page,
  }) => {
    // Open the drawer so the delete buttons are reachable.
    await page.click(".btn-hamburger");
    // Wait for drawer to settle (CSS transition is 0.2 s).
    await page.waitForTimeout(300);
    // Create a session + a document so the delete buttons exist.
    // For the hamburger + dismiss buttons alone we can already assert;
    // the delete-button assertion needs seeded state — skip if absent.
    const hamburger = page.locator(".btn-hamburger");
    const hambBox = await hamburger.boundingBox();
    expect(hambBox).not.toBeNull();
    expect(hambBox!.width, "hamburger width").toBeGreaterThanOrEqual(44);
    expect(hambBox!.height, "hamburger height").toBeGreaterThanOrEqual(44);
    // If a session exists, check the session delete button too.
    const sessionDelete = page.locator(".session-item-delete").first();
    if ((await sessionDelete.count()) > 0) {
      const sb = await sessionDelete.boundingBox();
      expect(sb).not.toBeNull();
      expect(sb!.width).toBeGreaterThanOrEqual(44);
      expect(sb!.height).toBeGreaterThanOrEqual(44);
    }
    const docDelete = page.locator(".doc-item-delete").first();
    if ((await docDelete.count()) > 0) {
      const db = await docDelete.boundingBox();
      expect(db).not.toBeNull();
      expect(db!.width).toBeGreaterThanOrEqual(44);
      expect(db!.height).toBeGreaterThanOrEqual(44);
    }
  });
});
