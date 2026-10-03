// v2.0.26.2 (PR-5) — ThemeProvider: 3-state theme toggle (light /
// dark / system) with `prefers-color-scheme` auto-detect and
// `useLocalStorage` persistence.
//
// v2.0.28.8 — default flipped from `"system"` to `"light"`. Reasoning:
// YuanRAG is a long-form Q&A tool with citation-heavy reading, and
// light surfaces score better on sustained readability for the
// majority of documents users query against. A user whose OS
// prefers dark and never opens Settings would otherwise land in
// dark mode and not know there's a toggle. They can still pick
// `"dark"` or `"system"` (auto) from Settings → Theme; we just stop
// *forcing* every fresh visitor through the OS-preference path.
//
// Existing users with a persisted `rag.theme` value (set via the
// Settings segmented control in a prior session) are unaffected —
// their saved choice wins over the new default. Only fresh visitors
// (or users who clear `localStorage.rag.theme`) see the new default.
//
// Why colocated with LocaleProvider rather than merged into
// `LocaleContext`:
//   - Theme has zero translation coupling (no `t()` calls), so
//     keeping it in a separate Context avoids needless re-renders
//     when the locale flips (which already mutates the i18n
//     strings on every component).
//   - `useTheme()` can be called from any descendant without the
//     i18n dependency, which matters for the upcoming PR-6 toast
//     system that wants to surface theme-aware icons.
//
// The `resolved` field is what the UI should read for actual
// styling decisions (the `theme` field is the user's *intent*,
// which may be `"system"`); `resolved` is always `"light"` or
// `"dark"`.
//
// Dark variants of all design tokens live in `styles.css` under
// `@media (prefers-color-scheme: dark)` and `:root[data-theme="dark"]`.
// This provider only flips the `data-theme` attribute on
// `<html>` and mirrors the choice into the `<meta name="theme-color">`
// tag so the mobile browser chrome (status bar tint, address bar)
// matches the app surface.
import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { useLocalStorage } from "../hooks/useLocalStorage";

export type Theme = "light" | "dark" | "system";

export interface ThemeContextValue {
  /** The user's persisted intent — may be `"system"`. */
  theme: Theme;
  /** Stable setter from `useLocalStorage`. */
  setTheme: (next: Theme | ((prev: Theme) => Theme)) => void;
  /** What the UI should actually paint — always `"light"` or `"dark"`. */
  resolved: "light" | "dark";
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

function parseTheme(raw: string): Theme {
  if (raw === "light" || raw === "dark" || raw === "system") return raw;
  return "system";
}

function serializeTheme(value: Theme): string {
  return value;
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useLocalStorage<Theme>(
    "rag.theme",
    "light",
    parseTheme,
    serializeTheme,
  );

  // Track the OS preference so `theme === "system"` can resolve
  // to the right value and react to the user's dark-mode toggle in
  // their OS settings without a reload.
  const [systemDark, setSystemDark] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    try {
      return window.matchMedia("(prefers-color-scheme: dark)").matches;
    } catch {
      return false;
    }
  });

  useEffect(() => {
    if (typeof window === "undefined") return undefined;
    let mql: MediaQueryList;
    try {
      mql = window.matchMedia("(prefers-color-scheme: dark)");
    } catch {
      return undefined;
    }
    const onChange = (e: MediaQueryListEvent) => setSystemDark(e.matches);
    // Newer browsers (Safari 14+, Chrome 39+) use addEventListener;
    // older WebKit only had addListener. Cover both for safety.
    if (typeof mql.addEventListener === "function") {
      mql.addEventListener("change", onChange);
      return () => mql.removeEventListener("change", onChange);
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (mql as any).addListener(onChange);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return () => (mql as any).removeListener(onChange);
  }, []);

  const resolved: "light" | "dark" =
    theme === "system" ? (systemDark ? "dark" : "light") : theme;

  // Flip `data-theme` and the mobile browser chrome tint whenever
  // the resolved value changes.
  useEffect(() => {
    if (typeof document === "undefined") return;
    const root = document.documentElement;
    root.dataset.theme = resolved === "dark" ? "dark" : "light";
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) {
      meta.setAttribute(
        "content",
        resolved === "dark" ? "#161a21" : "#5b8def",
      );
    }
  }, [resolved]);

  const value = useMemo<ThemeContextValue>(
    () => ({ theme, setTheme, resolved }),
    [theme, setTheme, resolved],
  );

  return (
    <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
  );
}

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) {
    throw new Error("useTheme must be used inside a <ThemeProvider>");
  }
  return ctx;
}