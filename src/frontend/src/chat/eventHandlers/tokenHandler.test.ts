/**
 * tokenHandler — covers:
 *  1. Token appends to the last assistant message.
 *  2. Token seeds a new assistant message when none exists.
 *  3. Empty / whitespace-only chunks are no-ops.
 *  4. Structured content (string[]) gets normalized to a string.
 *  5. Structured content (object with .text) gets normalized.
 *  6. After a token, ``isStreaming`` flips to true.
 */
import { describe, expect, it } from "vitest";
import { tokenHandler } from "./tokenHandler";
import { makeCtx, makeEvent, messages } from "./__tests__/testHelpers";

describe("tokenHandler", () => {
  it("appends to the last assistant message", () => {
    const ctx = makeCtx("t1", {
      messages: [
        { id: "a1", role: "assistant", content: "Hello" },
      ],
    });
    tokenHandler(ctx, makeEvent("token", { content: " world" }));
    expect(messages(ctx)).toHaveLength(1);
    expect((messages(ctx)[0] as { content: string }).content).toBe("Hello world");
    expect(ctx.state.t1.isStreaming).toBe(true);
  });

  it("seeds a new assistant message when none exists", () => {
    const ctx = makeCtx("t1", { messages: [] });
    tokenHandler(ctx, makeEvent("token", { content: "First!" }));
    const msgs = messages(ctx);
    expect(msgs).toHaveLength(1);
    expect((msgs[0] as { role: string; content: string }).role).toBe("assistant");
    expect((msgs[0] as { content: string }).content).toBe("First!");
  });

  it("does nothing for empty chunks", () => {
    const ctx = makeCtx("t1", { messages: [] });
    tokenHandler(ctx, makeEvent("token", { content: "" }));
    expect(messages(ctx)).toHaveLength(0);
    expect(ctx.state.t1.isStreaming).toBe(false);
  });

  it("normalizes string[] content blocks", () => {
    const ctx = makeCtx("t1", {
      messages: [{ id: "a1", role: "assistant", content: "" }],
    });
    tokenHandler(
      ctx,
      makeEvent("token", { content: ["Hello", " ", "world"] as unknown as string }),
    );
    expect((messages(ctx)[0] as { content: string }).content).toBe("Hello world");
  });

  it("normalizes object[] content blocks with .text", () => {
    const ctx = makeCtx("t1", {
      messages: [{ id: "a1", role: "assistant", content: "" }],
    });
    tokenHandler(
      ctx,
      makeEvent("token", {
        content: [{ text: "Hi" }, { text: " there" }] as unknown as string,
      }),
    );
    expect((messages(ctx)[0] as { content: string }).content).toBe("Hi there");
  });
});