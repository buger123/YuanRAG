/**
 * Model-status vocabulary + display labels.
 *
 * v2.0.26.1 (PR-4) — switched from a hardcoded ``Record<ModelStatus,
 * string>`` (Chinese-only) to a function that takes ``t: TFunction``
 * from ``useLocale()``. This is the same shape ``errors.ts`` below
 * uses, so the two i18n helpers stay consistent.
 *
 * Callers MUST pass the locale function at render time (not at
 * module load) — otherwise the labels freeze to whatever locale
 * was active when the module first imported. See
 * ``src/frontend/src/components/SettingsDialog.tsx`` and
 * ``src/frontend/src/components/ModelDownloadProgress.tsx`` for
 * the consumption pattern.
 */
import type en from "./en";
import type zh from "./zh";

export type ModelStatus = "ready" | "downloading" | "missing" | "pending";

/**
 * Function-form i18n signature — the same shape ``useLocale().t``
 * returns. Re-declared (vs imported) to avoid a circular dep on
 * the i18n context module.
 */
export type TFunction = (key: keyof typeof en | keyof typeof zh, params?: Record<string, string | number>) => string;

export function statusLabel(t: TFunction, status: ModelStatus): string {
  switch (status) {
    case "ready":
      return t("status.ready");
    case "downloading":
      return t("status.downloading");
    case "missing":
      return t("status.missing");
    case "pending":
      return t("status.pending");
  }
}

/**
 * v2.0.26.1 backward-compat: the previous API exposed
 * ``STATUS_LABEL: Record<ModelStatus, string>``. Some legacy
 * callers (tests, possibly a stray consumer) read it as a
 * record. Provide a deprecated accessor that returns the
 * Chinese labels (the project's default locale) — using this
 * means you bypass locale-aware rendering. New code MUST use
 * :func:`statusLabel` instead.
 *
 * @deprecated Use :func:`statusLabel` with ``t`` from ``useLocale()``.
 */
export const STATUS_LABEL: Record<ModelStatus, string> = {
  ready: "已就绪",
  downloading: "下载中…",
  missing: "未开始",
  pending: "等待中",
};
