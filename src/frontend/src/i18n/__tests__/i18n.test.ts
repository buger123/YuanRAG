/**
 * PR-4 (v2.0.26.1) — i18n layer tests. These pin the catalog
 * shape so a future contributor adding a key to en.ts without
 * adding the matching zh.ts entry fails at test time, not at
 * runtime. The TypeScript ``keyof typeof en | keyof typeof zh``
 * union on ``LocaleKey`` already catches a missing key at compile
 * time, but this test also asserts the runtime invariants:
 *
 *   - Every key has a non-empty string value in BOTH locales.
 *   - The placeholders in the two locales MATCH — a key like
 *     ``"chat.step": "Step {n}"`` and ``"chat.step": "第 {n} 步"``
 *     share the same set of ``{name}`` tokens, otherwise the
 *     ``t(key, { n: 1 })`` call would silently produce a broken
 *     output (the side without the placeholder returns the raw
 *     ``{n}``).
 */
import { describe, expect, it } from "vitest";
import en from "../en";
import zh from "../zh";

const PLACEHOLDER_RE = /\{(\w+)\}/g;

function extractPlaceholders(template: string): Set<string> {
  const out = new Set<string>();
  let m: RegExpExecArray | null;
  PLACEHOLDER_RE.lastIndex = 0;
  while ((m = PLACEHOLDER_RE.exec(template)) !== null) {
    out.add(m[1]);
  }
  return out;
}

describe("PR-4 i18n catalog parity", () => {
  it("every zh.ts key is also in en.ts", () => {
    const zhKeys = Object.keys(zh);
    const enKeys = new Set(Object.keys(en));
    const missing = zhKeys.filter((k) => !enKeys.has(k));
    expect(missing).toEqual([]);
  });

  it("every en.ts key is also in zh.ts", () => {
    const enKeys = Object.keys(en);
    const zhKeys = new Set(Object.keys(zh));
    const missing = enKeys.filter((k) => !zhKeys.has(k));
    expect(missing).toEqual([]);
  });

  it("every entry has a non-empty string in both locales", () => {
    const enRecord = en as Record<string, string>;
    const zhRecord = zh as Record<string, string>;
    for (const key of Object.keys(en)) {
      expect(typeof enRecord[key]).toBe("string");
      expect(enRecord[key].length).toBeGreaterThan(0);
      expect(typeof zhRecord[key]).toBe("string");
      expect(zhRecord[key].length).toBeGreaterThan(0);
    }
  });

  it("placeholders match between locales (no key has {x} in one and not the other)", () => {
    const enRecord = en as Record<string, string>;
    const zhRecord = zh as Record<string, string>;
    const drift: string[] = [];
    for (const key of Object.keys(en)) {
      const enPh = extractPlaceholders(enRecord[key]);
      const zhPh = extractPlaceholders(zhRecord[key]);
      const symmetric =
        enPh.size === zhPh.size &&
        [...enPh].every((p) => zhPh.has(p));
      if (!symmetric) {
        drift.push(
          `${key}: en=[${[...enPh].join(",")}] zh=[${[...zhPh].join(",")}]`,
        );
      }
    }
    expect(drift).toEqual([]);
  });
});
