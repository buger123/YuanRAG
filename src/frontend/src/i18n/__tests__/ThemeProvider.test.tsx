/**
 * v2.0.26.2 (PR-5) — ThemeProvider tests.
 *
 * Covers the 3-state toggle (light / dark / system) and the data-theme
 * side-effect that drives the design tokens in styles.css. The
 * matchMedia stub lives in test-setup.ts (jsdom doesn't implement it).
 *
 * Why these tests matter: a regression in the ``useEffect`` that
 * flips ``document.documentElement.dataset.theme`` would not throw
 * any error in jsdom, but every dark-mode user would suddenly see
 * light colors (or vice versa). Pinning the side-effect at the
 * storage layer is cheap insurance.
 */
import { act, render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useEffect, useState, type ReactNode } from "react";
import { ThemeProvider, useTheme } from "../ThemeProvider";

function setupMatchMedia(matches: boolean) {
  // Override the default matchMedia stub (matches: false) for tests
  // that need to simulate a dark OS preference. The ThemeProvider's
  // initial useState initializer reads ``matches`` synchronously,
  // so we must install the override BEFORE render.
  vi.spyOn(window, "matchMedia").mockImplementation(
    (query: string) =>
      ({
        matches,
        media: query,
        onchange: null,
        addEventListener: () => undefined,
        removeEventListener: () => undefined,
        addListener: () => undefined,
        removeListener: () => undefined,
        dispatchEvent: () => false,
      }) as MediaQueryList,
  );
}

/** Capture the current ThemeContext value for assertions. Exposes
 *  the theme setter via ``window.__setTheme`` so test code can
 *  invoke it without re-rendering the component tree. */
function Probe({ children }: { children?: ReactNode }) {
  const ctx = useTheme();
  useEffect(() => {
    (window as unknown as { __setTheme: typeof ctx.setTheme }).__setTheme =
      ctx.setTheme;
  }, [ctx.setTheme]);
  return (
    <div
      data-theme-intent={ctx.theme}
      data-theme-resolved={ctx.resolved}
    >
      {children}
    </div>
  );
}

function setThemeFromWindow(value: "light" | "dark" | "system") {
  const fn = (window as unknown as {
    __setTheme?: (v: "light" | "dark" | "system") => void;
  }).__setTheme;
  if (!fn) throw new Error("ThemeProvider did not register window.__setTheme");
  fn(value);
}

describe("ThemeProvider", () => {
  afterEach(() => {
    localStorage.clear();
    document.documentElement.removeAttribute("data-theme");
    vi.restoreAllMocks();
  });

  it("flips data-theme when setTheme('dark') is called", () => {
    setupMatchMedia(false);
    render(
      <ThemeProvider>
        <Probe />
      </ThemeProvider>,
    );
    act(() => setThemeFromWindow("dark"));
    expect(document.documentElement.dataset.theme).toBe("dark");
  });

  it("flips data-theme back to light when setTheme('light')", () => {
    setupMatchMedia(false);
    render(
      <ThemeProvider>
        <Probe />
      </ThemeProvider>,
    );
    act(() => setThemeFromWindow("dark"));
    act(() => setThemeFromWindow("light"));
    expect(document.documentElement.dataset.theme).toBe("light");
  });

  it("'system' resolves to 'dark' when the OS prefers dark", () => {
    setupMatchMedia(true);
    render(
      <ThemeProvider>
        <Probe />
      </ThemeProvider>,
    );
    // v2.0.28.8 — the default changed from "system" to "light", so we
    // have to explicitly opt into "system" before asserting that the
    // OS preference drives the resolved value.
    act(() => setThemeFromWindow("system"));
    expect(document.documentElement.dataset.theme).toBe("dark");
  });

  it("'system' resolves to 'light' when the OS prefers light", () => {
    setupMatchMedia(false);
    render(
      <ThemeProvider>
        <Probe />
      </ThemeProvider>,
    );
    // v2.0.28.8 — explicit "system" opt-in (default is now "light",
    // which already resolves to "light" without this act, so this
    // test specifically pins the system-auto path).
    act(() => setThemeFromWindow("system"));
    expect(document.documentElement.dataset.theme).toBe("light");
  });

  it("v2.0.28.8 — default theme resolves to 'light' regardless of OS preference", () => {
    // Pin the new contract: a fresh user lands in light mode even
    // when their OS prefers dark. They can still pick "dark" or
    // "system" from Settings → Theme, but the *default* is light.
    setupMatchMedia(true);
    render(
      <ThemeProvider>
        <Probe />
      </ThemeProvider>,
    );
    expect(document.documentElement.dataset.theme).toBe("light");
  });

  it("v2.0.28.8 — default theme persists as 'light' to localStorage on first set", () => {
    // The first time the user changes the theme, ``useLocalStorage``
    // writes the new value to localStorage. Before that, no entry
    // exists — but the resolved value is still light.
    setupMatchMedia(true);
    render(
      <ThemeProvider>
        <Probe />
      </ThemeProvider>,
    );
    expect(document.documentElement.dataset.theme).toBe("light");
    // Persist an explicit choice and re-render to confirm the
    // previously-default light behavior still works after a write.
    act(() => setThemeFromWindow("dark"));
    act(() => setThemeFromWindow("light"));
    expect(localStorage.getItem("rag.theme")).toBe("light");
    expect(document.documentElement.dataset.theme).toBe("light");
  });

  it("persists user choice to localStorage", () => {
    setupMatchMedia(false);
    render(
      <ThemeProvider>
        <Probe />
      </ThemeProvider>,
    );
    act(() => setThemeFromWindow("dark"));
    expect(localStorage.getItem("rag.theme")).toBe("dark");
  });

  it("throws if useTheme is called outside the provider", () => {
    // Capture console.error so the React "uncaught error" log
    // doesn't fail the test.
    const errSpy = vi.spyOn(console, "error").mockImplementation(() => undefined);
    expect(() => render(<Probe />)).toThrow(/useTheme must be used inside/);
    errSpy.mockRestore();
  });

  it("mirrors resolved theme into the theme-color meta tag", () => {
    setupMatchMedia(false);
    // jsdom does not parse <head> contents, so we inject the meta
    // tag manually before render.
    const meta = document.createElement("meta");
    meta.setAttribute("name", "theme-color");
    document.head.appendChild(meta);
    render(
      <ThemeProvider>
        <Probe />
      </ThemeProvider>,
    );
    expect(meta.getAttribute("content")).toMatch(/^#/);
    // Flip to dark — content should change.
    act(() => setThemeFromWindow("dark"));
    expect(meta.getAttribute("content")).toBe("#161a21");
    document.head.removeChild(meta);
  });
});

/** ``useState`` is intentionally unused at runtime in the probe
 *  but we re-export it so the linter doesn't complain about the
 *  import — keeping it makes it trivial to extend the probe in
 *  future tests (e.g. assert re-render counts). */
export const _useState = useState;
