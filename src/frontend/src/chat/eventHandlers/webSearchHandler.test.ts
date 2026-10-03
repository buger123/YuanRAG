/**
 * webSearchHandler — covers:
 *  1. attempted=true flips webSearching + records ref.
 *  2. attempted=false flips webSearching but does NOT record ref.
 *  3. answer_complete will later read the ref as a fallback.
 */
import { describe, expect, it } from "vitest";
import { webSearchHandler } from "./webSearchHandler";
import { makeCtx, makeEvent } from "./__tests__/testHelpers";

describe("webSearchHandler", () => {
  it("attempted=true flips webSearching + records ref", () => {
    const ctx = makeCtx("t1");
    webSearchHandler(ctx, makeEvent("web_search", { attempted: true }));
    expect(ctx.state.t1.webSearching).toBe(true);
    expect(ctx.webSearchedByThreadRef.current.get("t1")).toBe(true);
  });

  it("attempted=false flips webSearching but does NOT record ref", () => {
    const ctx = makeCtx("t1");
    webSearchHandler(ctx, makeEvent("web_search", { attempted: false }));
    expect(ctx.state.t1.webSearching).toBe(true);
    expect(ctx.webSearchedByThreadRef.current.get("t1")).toBeUndefined();
  });
});