/**
 * PR-4 (v2.0.26.1) — errors.ts tests. The previous file was
 * Chinese-only. The new ``apiError(status, body, cause?, t?)``
 * resolves labels via the i18n catalog and falls back to the
 * default (Chinese) translator when ``t`` is omitted — keeping
 * the legacy 3-arg call sites in ``api/client.ts`` working.
 *
 * Tests below verify:
 *   - HTTP status → localized label in both locales.
 *   - Network-layer TypeError messages map to the right keys.
 *   - Unknown status / unknown network message fall back to the
 *     ``errors.fallback`` key (or ``errors.network.generic`` for
 *     the network branch).
 *   - FastAPI ``{"detail": "..."}`` JSON shape is extracted and
 *     appended to the label.
 *   - ``setLocaleT`` / ``getCurrentT`` swap the default
 *     translator so module-scope callers (``api/client.ts``)
 *     track the active UI locale.
 */
import { afterEach, describe, expect, it } from "vitest";
import {
  apiError,
  getCurrentT,
  setLocaleT,
  type TFunction,
} from "../errors";
import en from "../en";
import zh from "../zh";

// Minimal TFunction shims — same shape as ``useLocale().t``.
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

describe("apiError — HTTP status → localized label", () => {
  it("returns English label for status 413 (File too large) with t=T_EN", () => {
    const err = apiError(413, undefined, undefined, T_EN);
    expect(err.message).toBe("File too large");
    expect(err.status).toBe(413);
  });

  it("returns Chinese label for status 413 with t=T_ZH", () => {
    const err = apiError(413, undefined, undefined, T_ZH);
    expect(err.message).toBe("文件过大");
  });

  it("returns English 401 (Unauthorized) with t=T_EN", () => {
    expect(apiError(401, undefined, undefined, T_EN).message).toBe(
      "Unauthorized — please check your API key",
    );
  });

  it("returns English 429 (Too many requests) with t=T_EN", () => {
    expect(apiError(429, undefined, undefined, T_EN).message).toBe(
      "Too many requests — please slow down",
    );
  });

  it("appends the FastAPI detail to the label when the body parses", () => {
    const err = apiError(
      413,
      JSON.stringify({ detail: "Empty upload" }),
      undefined,
      T_EN,
    );
    // PR-4 backend localized copy lands in detail; the frontend
    // appends it after the status label so the user sees both.
    expect(err.message).toContain("File too large");
    expect(err.message).toContain("Empty upload");
    expect(err.status).toBe(413);
  });

  it("falls back to errors.fallback for unknown status codes", () => {
    const err = apiError(599, undefined, undefined, T_EN);
    expect(err.message).toContain("Unknown error");
    expect(err.message).toContain("HTTP 599");
  });

  it("uses the default translator (Chinese) when t is omitted", () => {
    const err = apiError(413, undefined, undefined);
    expect(err.message).toBe("文件过大");
  });
});

describe("apiError — network-layer TypeError", () => {
  it("maps 'Failed to fetch' to the Chinese network-error label by default", () => {
    const err = apiError(0, undefined, new TypeError("Failed to fetch"));
    expect(err.message).toBe("请求失败");
    expect(err.status).toBe(0);
  });

  it("maps 'Failed to fetch' to the English network-error label when t=T_EN", () => {
    const err = apiError(
      0,
      undefined,
      new TypeError("Failed to fetch"),
      T_EN,
    );
    expect(err.message).toBe("Failed to fetch");
  });

  it("maps 'NetworkError when attempting to fetch resource' to the localized key", () => {
    const errEn = apiError(
      0,
      undefined,
      new TypeError(
        "NetworkError when attempting to fetch resource",
      ),
      T_EN,
    );
    expect(errEn.message).toBe(
      "NetworkError when attempting to fetch resource",
    );
    const errZh = apiError(
      0,
      undefined,
      new TypeError(
        "NetworkError when attempting to fetch resource",
      ),
      T_ZH,
    );
    expect(errZh.message).toBe("网络请求出错");
  });

  it("falls back to errors.network.generic + raw text for unrecognized network errors", () => {
    const err = apiError(
      0,
      undefined,
      new TypeError("Some weird browser-specific message"),
      T_EN,
    );
    expect(err.message).toContain("Network error");
    expect(err.message).toContain("Some weird browser-specific message");
  });
});

describe("setLocaleT / getCurrentT", () => {
  afterEach(() => {
    // Reset to default so test order doesn't matter.
    setLocaleT(
      (key, params) => {
        const template = (zh as Record<string, string>)[
          key as string
        ];
        if (!template) return String(key);
        if (!params) return template;
        return template.replace(/\{(\w+)\}/g, (_, name) =>
          name in params ? String(params[name]) : `{${name}}`,
        );
      },
    );
  });

  it("default translator is Chinese (project default locale)", () => {
    // The default translator lives in the module; verify it
    // returns Chinese for a known key without any setter call.
    const err = apiError(413, undefined, undefined);
    expect(err.message).toBe("文件过大");
  });

  it("setLocaleT swaps the default translator for module-scope callers", () => {
    setLocaleT(T_EN);
    // No explicit t arg — picks up T_EN via getCurrentT.
    const err = apiError(413, undefined, undefined);
    expect(err.message).toBe("File too large");
  });

  it("getCurrentT returns the most-recently-set translator", () => {
    setLocaleT(T_EN);
    expect(getCurrentT()).toBe(T_EN);
  });
});
