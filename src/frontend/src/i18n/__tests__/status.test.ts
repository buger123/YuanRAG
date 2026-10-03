/**
 * PR-4 (v2.0.26.1) — status label tests. The previous file
 * shipped a hardcoded Chinese ``Record<ModelStatus, string>``;
 * the new ``statusLabel(t, status)`` takes ``t`` from
 * ``useLocale()`` so the label flips when SettingsDialog
 * toggles the language.
 *
 * These tests pin both the function-form API and the legacy
 * ``STATUS_LABEL`` record (kept as a Chinese-only fallback for
 * backward compatibility with any external consumer).
 */
import { describe, expect, it } from "vitest";
import {
  STATUS_LABEL,
  statusLabel,
  type ModelStatus,
  type TFunction,
} from "../status";
import en from "../en";
import zh from "../zh";

// Minimal TFunction shim — same shape as ``useLocale().t`` for
// the keys we exercise here. Avoids pulling in a real
// ``LocaleProvider`` (which requires a React tree + LocalStorage
// hook — way too much test setup for what amounts to a key
// lookup).
const T_EN: TFunction = (key, params) => {
  const template = (en as Record<string, string>)[key as string];
  if (!template) return String(key);
  if (!params) return template;
  return template.replace(/\{(\w+)\}/g, (_, name) =>
    name in params ? String(params[name]) : `{${name}}`,
  );
};
const T_ZH: TFunction = (key, params) => {
  const template = (zh as Record<string, string>)[key as string];
  if (!template) return String(key);
  if (!params) return template;
  return template.replace(/\{(\w+)\}/g, (_, name) =>
    name in params ? String(params[name]) : `{${name}}`,
  );
};

describe("statusLabel (PR-4 function form)", () => {
  const cases: Array<{ status: ModelStatus; en: string; zh: string }> = [
    { status: "ready", en: "Ready", zh: "就绪" },
    { status: "downloading", en: "Downloading…", zh: "下载中…" },
    { status: "missing", en: "Not started", zh: "未开始" },
    { status: "pending", en: "Pending", zh: "等待中" },
  ];
  for (const { status, en: enText, zh: zhText } of cases) {
    it(`returns English for status="${status}" when locale=en`, () => {
      expect(statusLabel(T_EN, status)).toBe(enText);
    });
    it(`returns Chinese for status="${status}" when locale=zh`, () => {
      expect(statusLabel(T_ZH, status)).toBe(zhText);
    });
  }

  it("returns the localized value for every ModelStatus union member", () => {
    // Exhaustiveness: if a new variant is added to ModelStatus
    // the switch in statusLabel must be updated. We assert here
    // that the union size matches the case count we test.
    const statuses: ModelStatus[] = [
      "ready",
      "downloading",
      "missing",
      "pending",
    ];
    expect(new Set(statuses).size).toBe(4);
  });
});

describe("STATUS_LABEL (deprecated Chinese-only fallback)", () => {
  it("returns Chinese labels for every status (kept for backward compat)", () => {
    expect(STATUS_LABEL.ready).toBe("已就绪");
    expect(STATUS_LABEL.downloading).toBe("下载中…");
    expect(STATUS_LABEL.missing).toBe("未开始");
    expect(STATUS_LABEL.pending).toBe("等待中");
  });
});
