/**
 * Vitest setup file — runs before each test file. Imports
 * @testing-library/jest-dom so its matchers (``.toBeInTheDocument``
 * etc.) are globally available, and stubs the few browser globals
 * the handlers reach for outside of jsdom's stock surface.
 */
import "@testing-library/jest-dom/vitest";

// jsdom provides ``crypto.randomUUID`` in v22+; if a future jsdom
// release drops it, fail loud here rather than letting handlers
// produce identical ``id``s in tests.
if (typeof crypto === "undefined" || !crypto.randomUUID) {
  throw new Error(
    "test-setup.ts: crypto.randomUUID missing — jsdom version too old",
  );
}

// v2.0.26.2 (PR-5) — jsdom does not implement ``matchMedia``. The
// ThemeProvider calls it on mount to read ``prefers-color-scheme``,
// so without a stub any component test that transitively renders
// ThemeProvider throws ``TypeError: window.matchMedia is not a
// function``. The stub returns a MediaQueryList with ``matches:
// false`` (light mode) by default; tests that need a dark system
// preference override it per-test. ``addEventListener`` /
// ``removeEventListener`` are no-ops — no test relies on the OS
// preference flipping mid-test, and the stub satisfies the
// cleanup path.
if (typeof window !== "undefined" && !window.matchMedia) {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      addListener: (_cb: any) => undefined,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      removeListener: (_cb: any) => undefined,
      dispatchEvent: () => false,
    }),
  });
}