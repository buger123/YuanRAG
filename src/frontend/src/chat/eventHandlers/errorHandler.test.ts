/**
 * errorHandler — covers:
 *  1. error event appends a role:"error" bubble.
 *  2. error event clears isStreaming + webSearching.
 *  3. error event calls closeWebSocket(tid, "error") — P0-F3.
 */
import { describe, expect, it } from "vitest";
import { errorHandler } from "./errorHandler";
import { makeCtx, makeEvent, messages } from "./__tests__/testHelpers";

describe("errorHandler", () => {
  it("appends a role:error bubble", () => {
    const ctx = makeCtx("t1", { messages: [] });
    errorHandler(ctx, makeEvent("error", { message: "服务器超时" }));
    const msgs = messages(ctx);
    expect(msgs).toHaveLength(1);
    expect((msgs[0] as { role: string; content: string }).role).toBe("error");
    expect((msgs[0] as { content: string }).content).toBe("服务器超时");
  });

  it("clears isStreaming + webSearching", () => {
    const ctx = makeCtx("t1", { isStreaming: true, webSearching: true });
    errorHandler(ctx, makeEvent("error", { message: "x" }));
    expect(ctx.state.t1.isStreaming).toBe(false);
    expect(ctx.state.t1.webSearching).toBe(false);
  });

  it("P0-F3 — explicitly closes the WS with reason 'error'", () => {
    const ctx = makeCtx("t1");
    errorHandler(ctx, makeEvent("error", { message: "x" }));
    expect(ctx.spy.closedThreads).toContainEqual({
      threadId: "t1",
      reason: "error",
    });
  });
});