/**
 * reasoningHandler — covers:
 *  1. Reasoning appends to the last assistant's reasoning field.
 *  2. Reasoning seeds an empty assistant bubble if none exists.
 *  3. Multiple reasoning events accumulate (concatenate).
 *  4. Does NOT flip isStreaming (only tokenHandler does).
 */
import { describe, expect, it } from "vitest";
import { reasoningHandler } from "./reasoningHandler";
import { makeCtx, makeEvent, messages } from "./__tests__/testHelpers";

describe("reasoningHandler", () => {
  it("appends to the last assistant's reasoning field", () => {
    const ctx = makeCtx("t1", {
      messages: [{ id: "a1", role: "assistant", content: "X", reasoning: "step 1 " }],
    });
    reasoningHandler(ctx, makeEvent("reasoning", { content: "step 2" }));
    const m = messages(ctx)[0] as { reasoning?: string; content: string };
    expect(m.reasoning).toBe("step 1 step 2");
    expect(m.content).toBe("X"); // visible content untouched
  });

  it("seeds an empty assistant bubble if none exists", () => {
    const ctx = makeCtx("t1", { messages: [] });
    reasoningHandler(ctx, makeEvent("reasoning", { content: "thinking..." }));
    const msgs = messages(ctx);
    expect(msgs).toHaveLength(1);
    expect((msgs[0] as { role: string }).role).toBe("assistant");
    expect((msgs[0] as { content: string }).content).toBe("");
    expect((msgs[0] as { reasoning?: string }).reasoning).toBe("thinking...");
  });

  it("multiple reasoning events accumulate", () => {
    const ctx = makeCtx("t1", { messages: [] });
    reasoningHandler(ctx, makeEvent("reasoning", { content: "A" }));
    reasoningHandler(ctx, makeEvent("reasoning", { content: "B" }));
    reasoningHandler(ctx, makeEvent("reasoning", { content: "C" }));
    expect((messages(ctx)[0] as { reasoning?: string }).reasoning).toBe("ABC");
  });

  it("does NOT flip isStreaming", () => {
    const ctx = makeCtx("t1", { messages: [], isStreaming: false });
    reasoningHandler(ctx, makeEvent("reasoning", { content: "thinking" }));
    expect(ctx.state.t1.isStreaming).toBe(false);
  });
});