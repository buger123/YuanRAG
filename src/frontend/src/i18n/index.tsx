// v2.0.6 P2-4 — first React Context in the app. Bundles the
// active locale, a ``setLocale`` setter, and a memoized ``t`` lookup
// function. Consumers do ``const { t, locale } = useLocale()`` rather
// than importing the dictionaries directly, so the locale switch is
// purely a state change and React re-renders every consumer in one
// pass (no global event bus needed).
//
// Why a custom hook instead of a third-party library
// --------------------------------------------------
// ``react-intl`` / ``i18next`` are overkill for a hardcoded 60-key
// dictionary. They also ship formatting helpers (plural rules,
// number formatting) that we don't need and that would inflate the
// bundle. The plain ``useLocale`` below gives us the same ergonomic
// API at ~1 KB of source.

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  type ReactNode,
} from "react";
import zh, { type ZhKey } from "./zh";
import en, { type EnKey } from "./en";
import { useLocalStorage } from "../hooks/useLocalStorage";
import { setLocaleT } from "./errors";

// Union of every key across both dictionaries. The dictionaries are
// pinned to ``as const`` so TypeScript narrows each entry to its
// string literal — adding a new key to one side without the other
// surfaces as a compile error.
export type LocaleKey = ZhKey | EnKey;

export type Locale = "zh" | "en";

interface LocaleContextValue {
  locale: Locale;
  setLocale: (next: Locale | ((prev: Locale) => Locale)) => void;
  /**
   * Look up a translation key in the active locale. Missing keys
   * fall back to the Chinese value (the original language of the
   * app), then to the key itself — so a partially translated UI
   * never renders blank, and a typo'd key still gives a visible
   * breadcrumb to the developer.
   *
   * Placeholders: pass an object whose keys match ``{name}`` tokens
   * in the dictionary entry, e.g. ``t("sidebar.embedding", { count: 12 })``.
   * No ICU plural rules; if you need them, do the branching in the
   * caller (``count === 1 ? t("…singular") : t("…plural")``).
   */
  t: (key: LocaleKey, vars?: Record<string, string | number>) => string;
}

const LocaleContext = createContext<LocaleContextValue | null>(null);

interface ProviderProps {
  children: ReactNode;
}

export function LocaleProvider({ children }: ProviderProps) {
  const [locale, setLocale] = useLocalStorage<Locale>(
    "rag.locale",
    "zh",
    // Manual parse so an unexpected value (e.g. "fr" left over from
    // an earlier test) falls back to the default instead of
    // crashing. Same idea for the serializer: keep it as a plain
    // string rather than a JSON-quoted value so the storage layer
    // is human-readable in DevTools.
    (raw) => (raw === "zh" || raw === "en" ? raw : "zh"),
    (value) => value,
  );

  const t = useCallback<LocaleContextValue["t"]>(
    (key, vars) => {
      const dict = locale === "en" ? en : zh;
      // Lookup with explicit ``any`` index because the two
      // dictionaries have different (yet compatible via the union)
      // shapes — narrowing ``LocaleKey`` to either dict's keyof
      // requires a type assertion that's noisier than the index.
      const template = (dict as Record<string, string>)[key as string];
      const fallback = (zh as Record<string, string>)[key as string] ?? key;
      const resolved = template ?? fallback;
      if (!vars) return resolved;
      return resolved.replace(/\{(\w+)\}/g, (_, name) => {
        const v = vars[name];
        return v === undefined ? `{${name}}` : String(v);
      });
    },
    [locale],
  );

  const value = useMemo<LocaleContextValue>(
    () => ({ locale, setLocale, t }),
    [locale, setLocale, t],
  );

  // v2.0.26.1 (PR-4) — register the live translator with
  // ``errors.ts`` so ``apiError(...)`` calls made OUTSIDE the React
  // tree (notably from ``api/client.ts``'s dozen module-scope
  // functions) still produce the right language when the user
  // toggles SettingsDialog. Without this, ``apiError`` would freeze
  // to whatever the default module-import saw, which on first load
  // is Chinese — so a fresh English user would see Chinese error
  // copy until they reloaded. The setter is idempotent and cheap
  // (just a module-level variable assignment), so the effect runs
  // on every locale change without concern.
  useEffect(() => {
    setLocaleT(t);
  }, [t]);

  return (
    <LocaleContext.Provider value={value}>{children}</LocaleContext.Provider>
  );
}

export function useLocale(): LocaleContextValue {
  const ctx = useContext(LocaleContext);
  if (!ctx) {
    // Throwing here (rather than returning a default) surfaces the
    // bug at the first render of a component that forgot to wrap
    // itself in ``LocaleProvider`` — far better than a silent
    // blank UI.
    throw new Error("useLocale must be used inside a <LocaleProvider>");
  }
  return ctx;
}
