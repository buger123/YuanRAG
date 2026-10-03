// v2.0.6 P2-3 / P2-4 — shared localStorage-backed ``useState``
// hook. Earlier persistence calls (the active-thread id in App.tsx,
// the show-thinking toggle in ChatPane.tsx) were inlined and
// inconsistent: each one wrote the raw value to localStorage on
// every change, swallowed JSON parse errors silently, and never
// told the user what happened when storage was unavailable (Safari
// private mode, quota exhausted, blocked by an extension). Central-
// izing here gives every preference the same safe semantics and
// keeps the call sites one line each.
//
// Why not just write ``useState`` + an effect
// --------------------------------------------
// The naive shape leaks storage semantics into every consumer:
//
//   const [v, setV] = useState(default);
//   useEffect(() => { try { localStorage.setItem(key, JSON.stringify(v)) }
//     catch {} }, [v]);
//
// — and on first render reads ``null`` then a real value, causing
// a flicker for UI like the theme switch that depends on the
// persisted value being available synchronously. This hook
// initializes state from storage (so the first paint already has
// the persisted value), accepts a JSON serializer per call site
// (so theme can store ``"light"`` as a plain string while richer
// state could store ``{ ... }``), and tolerates any storage
// failure (bad JSON, missing key, denied access) by falling back
// to the supplied default. The setter also catches write errors
// so a failing storage layer never crashes the React tree.

import { useCallback, useEffect, useRef, useState } from "react";

type Setter<T> = (value: T | ((prev: T) => T)) => void;

export function useLocalStorage<T>(
  key: string,
  defaultValue: T,
  parse: (raw: string) => T = JSON.parse as (raw: string) => T,
  serialize: (value: T) => string = JSON.stringify,
): [T, Setter<T>] {
  // Refs keep the serializer functions reachable for the storage
  // listener below without forcing the listener to be re-bound on
  // every render (which would, in turn, re-run the effect and
  // cause every change to also fire a manual write).
  const parseRef = useRef(parse);
  const serializeRef = useRef(serialize);
  parseRef.current = parse;
  serializeRef.current = serialize;

  const [value, setValue] = useState<T>(() => {
    if (typeof window === "undefined") return defaultValue;
    try {
      const raw = window.localStorage.getItem(key);
      if (raw === null) return defaultValue;
      return parseRef.current(raw);
    } catch {
      // Bad JSON, denied storage, private mode — fall back to
      // default. We deliberately don't surface the error to the
      // console: a corrupted preference should not spam the
      // developer console every render.
      return defaultValue;
    }
  });

  // Persist on every change. Writing inside an effect (rather than
  // inside the setter) keeps the setter cheap and ensures the
  // write happens AFTER React commits the new state, so an
  // unrelated re-render can never observe a stored value that
  // doesn't match the rendered state.
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      window.localStorage.setItem(key, serializeRef.current(value));
    } catch {
      // Quota / disabled storage — same swallow as the read path.
      // The in-memory state is still authoritative for this tab.
    }
  }, [key, value]);

  // Cross-tab sync — when the user has the app open in two tabs
  // and flips the theme in one, the other picks it up without a
  // reload. We intentionally listen on ``window`` (not the
  // element) because the storage event only fires on other
  // documents, not the one that wrote the value.
  const valueRef = useRef(value);
  valueRef.current = value;
  const setValueStable = useCallback<Setter<T>>((next) => {
    setValue((prev) => {
      const resolved =
        typeof next === "function"
          ? (next as (p: T) => T)(prev)
          : next;
      return resolved;
    });
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    function onStorage(ev: StorageEvent) {
      if (ev.key !== key) return;
      if (ev.newValue === null) {
        setValue(defaultValue);
        return;
      }
      try {
        setValue(parseRef.current(ev.newValue));
      } catch {
        // Ignore malformed cross-tab writes — the listener's
        // purpose is convenience, not correctness, and the next
        // user-driven change will overwrite any garbage.
      }
    }
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, [key, defaultValue]);

  return [value, setValueStable];
}
