/**
 * Toast notification system — v2.0.26.3 (PR-6).
 *
 * Architecture mirrors the PR-4 i18n singleton pattern in
 * ``src/frontend/src/i18n/errors.ts``: a module-level dispatcher
 * (``setToastDispatcher`` / ``getCurrentDispatcher``) is registered
 * by the ``ToastProvider`` on mount; non-React call sites (notably
 * ``useThreadSocket.ts``'s WebSocket catch blocks) can fire toasts
 * without depending on React context.
 *
 * Why a singleton dispatcher (not just a custom event bus):
 *   - The hook is used by code outside React's render path. A
 *     window-level event bus would need a second listener pattern,
 *     and would not give us a stable reference for the dispatcher
 *     itself.
 *   - The pattern is already proven by ``setLocaleT`` for the
 *     apiError singleton; reusing the shape keeps the codebase
 *     symmetric.
 *   - Tests can swap the dispatcher via the setter without
 *     rendering a provider (helpful for unit-testing the hook).
 *
 * Why a stack cap (max 3 visible):
 *   - Production user-complaints audit showed toast floods in
 *     some apps where the developer wired every ``console.error``
 *     to a toast. Capping the visible stack forces the most
 *     recent 3 to show and silently drops older ones (they've
 *     already been logged via ``console.error`` so the developer
 *     still has the trace). This is the same convention Slack
 *     / GitHub use.
 *
 * Why an opt-in ``duration: 0`` for sticky toasts:
 *   - Some failures (e.g. "API key required" surfaced via toast)
 *     must stay until the user explicitly dismisses. Setting
 *     ``duration: 0`` disables the auto-dismiss timer.
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useId,
  useMemo,
  useRef,
  useSyncExternalStore,
} from "react";
import type { ReactNode } from "react";
import { useLocale } from "../i18n";

export type ToastKind = "error" | "success" | "warning" | "info";

export interface ToastOptions {
  kind: ToastKind;
  message: string;
  /** Auto-dismiss in milliseconds. Default 5000; pass 0 for sticky. */
  duration?: number;
}

export type ShowToast = (options: ToastOptions) => void;

interface ToastRecord {
  id: string;
  kind: ToastKind;
  message: string;
  /** Resolved duration (0 = sticky). */
  duration: number;
  /** Epoch ms when the toast was created — drives the auto-dismiss timer. */
  createdAt: number;
}

const DEFAULT_DURATION_MS = 5000;
/** Hard cap on visible toasts — older entries are dropped silently. */
const MAX_VISIBLE = 3;
/** Subscriber list for the module-level store. Each subscriber is
 *  a `useSyncExternalStore` snapshot listener. */
const subscribers = new Set<() => void>();
/** Live toast records (mutable in module scope). Reads via
 *  ``useSyncExternalStore`` always read the latest reference. */
let toasts: ToastRecord[] = [];

/** Counter for stable id generation. Not crypto-quality — just unique
 *  enough for React keys. */
let nextId = 1;

function emitChange() {
  for (const sub of subscribers) sub();
}

function addToast(options: ToastOptions): ToastRecord {
  const record: ToastRecord = {
    id: `toast-${nextId++}`,
    kind: options.kind,
    message: options.message,
    duration: options.duration ?? DEFAULT_DURATION_MS,
    createdAt: Date.now(),
  };
  // If we are already at the cap, drop the oldest entry. We keep
  // ``toasts.length <= MAX_VISIBLE`` invariant for predictable layout.
  const next = [...toasts, record];
  if (next.length > MAX_VISIBLE) {
    next.splice(0, next.length - MAX_VISIBLE);
  }
  toasts = next;
  emitChange();
  return record;
}

function removeToast(id: string): void {
  const before = toasts.length;
  toasts = toasts.filter((t) => t.id !== id);
  if (toasts.length !== before) emitChange();
}

/**
 * Module-level dispatcher singleton. ``ToastProvider`` registers
 * itself on mount via ``setToastDispatcher``; non-React code can
 * fire toasts via ``getCurrentDispatcher()?.(...)`` without
 * coupling to the React tree.
 */
let _dispatcher: ShowToast | null = null;

export function setToastDispatcher(show: ShowToast | null): void {
  _dispatcher = show;
}

export function getCurrentDispatcher(): ShowToast | null {
  return _dispatcher;
}

/**
 * Test-only helper — clear the module-level toast store. Production
 * code should NEVER call this; toasts are user-visible by design.
 * Lives next to the dispatcher singleton so test files can import
 * both from the same module without a second import line.
 */
export function __resetToastsForTests(): void {
  toasts = [];
  emitChange();
}

interface ToastContextValue {
  showToast: ShowToast;
}

const ToastContext = createContext<ToastContextValue | null>(null);

/**
 * Programmatic toast trigger for React code. Throws when used
 * outside a ``ToastProvider`` — callers in non-React paths should
 * use :func:`getCurrentDispatcher` directly.
 *
 * Returns the same ``{ showToast }`` object shape as the other
 * hooks in this codebase (``useTheme``, ``useLocale``) so call
 * sites read consistently:
 *
 *   const { showToast } = useToast();
 *   const { theme, setTheme } = useTheme();
 */
export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (!ctx) {
    throw new Error("useToast must be used inside a <ToastProvider>");
  }
  return ctx;
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const showToast = useCallback<ShowToast>((options) => {
    addToast(options);
  }, []);
  const value = useMemo<ToastContextValue>(() => ({ showToast }), [showToast]);
  // Register the dispatcher for non-React callers. Unregister on
  // unmount so a stale provider (e.g. test cleanup) doesn't leak.
  useEffect(() => {
    setToastDispatcher(showToast);
    return () => {
      // Only clear if WE are still the registered dispatcher —
      // avoid stomping a newer provider that replaced us.
      if (_dispatcher === showToast) setToastDispatcher(null);
    };
  }, [showToast]);
  return (
    <ToastContext.Provider value={value}>
      {children}
    </ToastContext.Provider>
  );
}

function subscribe(notify: () => void): () => void {
  subscribers.add(notify);
  return () => {
    subscribers.delete(notify);
  };
}

function getSnapshot(): ToastRecord[] {
  return toasts;
}

function getServerSnapshot(): ToastRecord[] {
  return [];
}

/**
 * Render the toast container. Mount once at the root of the app
 * (sibling of the App tree inside ``ToastProvider``). The container
 * is fixed bottom-right on desktop, top-center on mobile (via the
 * ``@media (max-width: 768px)`` block in ``styles.css``).
 */
export function ToastViewport() {
  const { t } = useLocale();
  const live = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
  if (live.length === 0) return null;
  return (
    <div className="toast-container" role="region" aria-label={t("toast.regionAria")}>
      {live.map((toast) => (
        <ToastItem key={toast.id} toast={toast} />
      ))}
    </div>
  );
}

const ICON_FOR_KIND: Record<ToastKind, string> = {
  error: "✕",
  success: "✓",
  warning: "⚠",
  info: "ℹ",
};

function ToastItem({ toast }: { toast: ToastRecord }) {
  const { t } = useLocale();
  // Stable id for aria-controls pairing between the dismiss button
  // and the toast body (the body has no separate focusable element,
  // so the id is mostly for test selectors — but matching the
  // pattern from PR-5 keeps the codebase consistent).
  const bodyId = useId();
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const dismiss = useCallback(() => removeToast(toast.id), [toast.id]);

  useEffect(() => {
    if (toast.duration <= 0) return undefined;
    timerRef.current = setTimeout(dismiss, toast.duration);
    return () => {
      if (timerRef.current !== null) clearTimeout(timerRef.current);
    };
  }, [toast.duration, dismiss]);

  return (
    <div
      className={`toast toast-${toast.kind}`}
      id={bodyId}
      role={toast.kind === "error" ? "alert" : "status"}
      aria-live={toast.kind === "error" ? "assertive" : "polite"}
    >
      <span className="toast-icon" aria-hidden="true">
        {ICON_FOR_KIND[toast.kind]}
      </span>
      <span className="toast-message">{toast.message}</span>
      <button
        type="button"
        className="toast-dismiss"
        onClick={dismiss}
        aria-label={t("toast.dismissAria")}
      >
        ×
      </button>
    </div>
  );
}