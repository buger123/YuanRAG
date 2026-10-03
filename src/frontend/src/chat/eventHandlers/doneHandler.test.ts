/**
 * doneHandler — covers:
 *  1. done event clears the per-thread web-search ref (next turn
 *     won't inherit the banner from this turn).
 *  2. done event clears the per-thread step counter.
 *  3. done event resets isStreaming / webSearching / currentStep.
 *  4. done event dispatches rag:session-updated window event.
 *  5. done event calls closeWebSocket(tid) (no reason).
 */
import { describe, expect, it } from "vitest";
import { doneHandler } from "./doneHandler";
import { makeCtx, makeEvent } from "./__tests__/testHelpers";

describe("doneHandler", () => {
  it("clears the per-thread webSearched ref so next turn doesn't inherit", () => {
    const ctx = makeCtx("t1");
    ctx.webSearchedByThreadRef.current.set("t1", true);
    doneHandler(ctx, makeEvent("done", {}));
    expect(ctx.webSearchedByThreadRef.current.has("t1")).toBe(false);
  });

  it("clears the per-thread step counter", () => {
    const ctx = makeCtx("t1");
    ctx.stepByThreadRef.current.set("t1", 4);
    doneHandler(ctx, makeEvent("done", {}));
    expect(ctx.stepByThreadRef.current.has("t1")).toBe(false);
  });

  it("resets isStreaming + webSearching + currentStep", () => {
    const ctx = makeCtx("t1", {
      isStreaming: true,
      webSearching: true,
      currentStep: 5,
    });
    doneHandler(ctx, makeEvent("done", {}));
    expect(ctx.state.t1.isStreaming).toBe(false);
    expect(ctx.state.t1.webSearching).toBe(false);
    expect(ctx.state.t1.currentStep).toBe(0);
  });

  it("dispatches rag:session-updated window event", () => {
    const ctx = makeCtx("t1");
    doneHandler(ctx, makeEvent("done", {}));
    expect(ctx.spy.windowEvents).toContainEqual({
      type: "rag:session-updated",
      detail: undefined,
    });
  });

  it("closes the WS (no reason — normal close path)", () => {
    const ctx = makeCtx("t1");
    doneHandler(ctx, makeEvent("done", {}));
    expect(ctx.spy.closedThreads).toContainEqual({
      threadId: "t1",
      reason: undefined,
    });
  });
});