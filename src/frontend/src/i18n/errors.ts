/**
 * Localized error messages for the HTTP client.
 *
 * v2.0.26.1 (PR-4) — the previous file shipped Chinese-only labels
 * (``STATUS_LABEL_ZH``, ``NETWORK_ERROR_ZH``). When the user toggled
 * SettingsDialog to English, the rest of the UI flipped to English
 * but the API error stayed in Chinese — the worst-of-both-worlds
 * visual. This version threads ``t: TFunction`` (from
 * ``useLocale()``) through :func:`apiError` so the labels track the
 * UI language.
 *
 * QW #8: before this, a failed upload showed "Failed to fetch" (or
 * the raw ``413 Payload Too Large: {"detail":"File too large"}`` text)
 * — the user saw English while the rest of the UI was Chinese, and
 * the status-code prefix looked like a debug log line.
 *
 * Strategy: every fetch site throws via ``apiError()`` (below) which
 * maps status / network error → a short string suitable for an
 * inline banner. The original ``Error`` carries the localized message
 * so callers can show it directly OR re-throw with custom context.
 */

import en from "./en";
import zh from "./zh";

/**
 * Function-form i18n signature — same shape ``useLocale().t``
 * returns. Re-declared (vs imported) to avoid a circular dep on
 * the i18n context module.
 */
export type TFunction = (
  key: keyof typeof en | keyof typeof zh,
  params?: Record<string, string | number>,
) => string;

/**
 * Default translator used when a caller doesn't pass ``t`` — falls
 * back to the Chinese catalog. This keeps the legacy 3-arg
 * ``apiError(status, body, cause)`` call sites working while new
 * code threads the locale through. Tests and SSR-style callers
 * that don't have a React tree can use the default and get the
 * project-default-locale copy.
 */
const DEFAULT_T: TFunction = (
  key,
  params,
): string => {
  const entry = (zh as Record<string, string>)[key as string];
  if (typeof entry !== "string") return String(key);
  if (!params) return entry;
  return entry.replace(/\{(\w+)\}/g, (_, name) =>
    name in params ? String(params[name]) : `{${name}}`,
  );
};

/**
 * Module-level locale pointer. Set by ``i18n/index.tsx``'s
 * ``LocaleProvider`` on mount and on every locale change so any
 * ``apiError(...)`` call without an explicit ``t`` argument
 * still produces the right language for the active UI.
 *
 * This is the canonical solution to "errors.ts has no React
 * tree" — the i18n context lives in a module, the provider
 * registers itself with this setter, and ``apiError`` picks up
 * the live translator via :func:`getCurrentT`. No prop drilling
 * needed through ``api/client.ts``'s dozen functions.
 */
let _currentT: TFunction = DEFAULT_T;

export function setLocaleT(t: TFunction): void {
  _currentT = t;
}

export function getCurrentT(): TFunction {
  return _currentT;
}

/**
 * HTTP status → i18n key. Numeric codes are stable; the keys map to
 * en.ts / zh.ts so the user sees the localized label.
 */
const STATUS_LABEL_KEYS: Record<number, keyof typeof en> = {
  400: "errors.status.400",
  401: "errors.status.401",
  403: "errors.status.403",
  404: "errors.status.404",
  408: "errors.status.408",
  409: "errors.status.409",
  413: "errors.status.413",
  415: "errors.status.415",
  422: "errors.status.422",
  429: "errors.status.429",
  500: "errors.status.500",
  502: "errors.status.502",
  503: "errors.status.503",
  504: "errors.status.504",
};

/**
 * Common network-layer failures (TypeError from fetch) — the
 * English message text is the wire-format identifier; we map each
 * to the corresponding ``errors.network.*`` key.
 */
const NETWORK_ERROR_KEYS: Record<string, keyof typeof en> = {
  "Failed to fetch": "errors.network.failedToFetch",
  "NetworkError when attempting to fetch resource":
    "errors.network.networkError",
  "Load failed": "errors.network.loadFailed",
};

/**
 * v2.0.26.1 backward-compat: the previous API returned Chinese
 * labels directly (no t() parameter). Tests and legacy callers
 * that didn't pick up the new signature still work via this
 * record — which mirrors the existing Chinese copy. New code MUST
 * use :func:`apiError` with ``t`` from ``useLocale()``.
 *
 * @deprecated Use :func:`apiError` with ``t`` from ``useLocale()``.
 */
export const STATUS_LABEL_ZH: Record<number, string> = {
  400: "请求格式错误",
  401: "未授权,请检查 API 密钥",
  403: "没有访问权限",
  404: "资源不存在",
  408: "请求超时",
  409: "资源冲突",
  413: "文件过大",
  415: "不支持的文件格式",
  422: "请求参数无效",
  429: "请求过于频繁,请稍后重试",
  500: "服务器内部错误",
  502: "网关错误",
  503: "服务暂时不可用",
  504: "网关超时",
};

export interface ApiError extends Error {
  /** HTTP status if the response reached the server, else 0. */
  status: number;
  /** Raw response body text (truncated to 500 chars). */
  body?: string;
  /** Original underlying error if this wraps a network failure. */
  cause?: unknown;
}

/**
 * Construct a localized ApiError from a failed ``fetch`` response or
 * thrown TypeError.
 *
 * Signature is the original 3-arg form
 * ``apiError(status, body, cause?)`` with an OPTIONAL ``t`` as the
 * 4th arg. The old call sites in ``api/client.ts`` keep working
 * (they get the Chinese default via :data:`DEFAULT_T`); new
 * callers can pass ``t`` from ``useLocale()`` so the label flips
 * when the user toggles SettingsDialog.
 *
 * @param status HTTP status (0 if the request never reached the server)
 * @param body   raw response body text, if any
 * @param cause  the original thrown error (kept for debugging)
 * @param t      locale function from ``useLocale()``. Optional —
 *               defaults to Chinese (the project's default locale)
 */
export function apiError(
  status: number,
  body: string | undefined,
  cause?: unknown,
  t: TFunction = _currentT,
): ApiError {
  let label: string;
  let detail: string | null = null;

  if (status === 0 && cause instanceof Error) {
    // Network-layer failure: TypeError from fetch with no status.
    const original = cause.message || "Unknown network error";
    const key = NETWORK_ERROR_KEYS[original];
    label = key ? t(key) : `${t("errors.network.generic")}:${original}`;
  } else if (status > 0) {
    // Try to extract a JSON ``detail`` from the body (FastAPI default
    // error shape). If it's there and is a meaningful string, surface
    // it directly — otherwise fall back to the status label.
    detail = extractDetail(body);
    const key = STATUS_LABEL_KEYS[status];
    label = key ? t(key) : `${t("errors.fallback")} (HTTP ${status})`;
  } else {
    label = t("errors.fallback");
  }

  const message = detail ? `${label}:${detail}` : label;
  const err = new Error(message) as ApiError;
  err.status = status;
  err.body = body?.slice(0, 500);
  err.cause = cause;
  // Naming the error class helps debugging — `err.name === "ApiError"`
  // is a stable handle for call-site filtering.
  err.name = "ApiError";
  return err;
}

/**
 * Extract ``detail`` from a FastAPI-shaped error body. The backend
 * wraps errors as ``{"detail": "..."}`` (string) or ``{"detail":
 * [{"loc": [...], "msg": "..."}]}`` (validation list). We surface
 * the first as-is; for the list we join the messages so the user
 * sees something coherent rather than JSON braces.
 */
function extractDetail(body: string | undefined): string | null {
  if (!body) return null;
  try {
    const parsed = JSON.parse(body);
    if (typeof parsed?.detail === "string") {
      return parsed.detail;
    }
    if (Array.isArray(parsed?.detail)) {
      return parsed.detail
        .map((e: { msg?: string }) => e?.msg)
        .filter(Boolean)
        .join("; ");
    }
  } catch {
    /* not JSON — fall through */
  }
  // Non-JSON body: surface the raw text but cap at 200 chars so a
  // stack trace doesn't blow up the banner.
  return body.length > 200 ? body.slice(0, 200) + "…" : body;
}

// Re-export so consumers don't need a second import for ModelStatus.
export type { ModelStatus } from "./status";
