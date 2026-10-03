/**
 * toolCallStartHandler — covers:
 *  1. New tool_call_start appends to the last assistant's tool_calls.
 *  2. tool_call_id empty is a silent no-op.
 *  3. Duplicate id (back-to-back re-delivery) is idempotent.
 *  4. No assistant bubble exists → seeds one with empty content.
 *  5. step / args / name are normalized defensively.
 */
import { describe, expect, it } from "vitest";
import { toolCallStartHandler } from "./toolCallStartHandler";
import { makeCtx, makeEvent, messages } from "./__tests__/testHelpers";
import type { ToolCall } from "../../chat-utils";

function lastToolCalls(ctx: ReturnType<typeof makeCtx>): ToolCall[] {
  const msgs = messages(ctx);
  const last = msgs[msgs.length - 1] as { tool_calls?: ToolCall[] };
  return last.tool_calls || [];
}

describe("toolCallStartHandler", () => {
  it("appends a new tool_call card to the last assistant", () => {
    const ctx = makeCtx("t1", {
      messages: [{ id: "a1", role: "assistant", content: "thinking..." }],
    });
    toolCallStartHandler(
      ctx,
      makeEvent("tool_call_start", {
        tool_call_id: "tc1",
        name: "search_docs",
        args: { query: "hi" },
        step: 1,
      }),
    );
    const cards = lastToolCalls(ctx);
    expect(cards).toHaveLength(1);
    expect(cards[0].id).toBe("tc1");
    expect(cards[0].name).toBe("search_docs");
    expect(cards[0].args).toEqual({ query: "hi" });
    expect(cards[0].step).toBe(1);
    expect(cards[0].ok).toBeUndefined(); // running
    expect(typeof cards[0].started_at).toBe("string");
  });

  it("empty tool_call_id is a no-op", () => {
    const ctx = makeCtx("t1", {
      messages: [{ id: "a1", role: "assistant", content: "" }],
    });
    toolCallStartHandler(
      ctx,
      makeEvent("tool_call_start", { tool_call_id: "", name: "x" }),
    );
    expect(lastToolCalls(ctx)).toHaveLength(0);
  });

  it("duplicate id is idempotent (no double-card)", () => {
    const ctx = makeCtx("t1", {
      messages: [{ id: "a1", role: "assistant", content: "" }],
    });
    toolCallStartHandler(
      ctx,
      makeEvent("tool_call_start", { tool_call_id: "tc1", name: "search_docs" }),
    );
    toolCallStartHandler(
      ctx,
      makeEvent("tool_call_start", { tool_call_id: "tc1", name: "search_docs" }),
    );
    expect(lastToolCalls(ctx)).toHaveLength(1);
  });

  it("seeds an empty assistant bubble if none exists", () => {
    const ctx = makeCtx("t1", { messages: [] });
    toolCallStartHandler(
      ctx,
      makeEvent("tool_call_start", { tool_call_id: "tc1", name: "lookup" }),
    );
    const msgs = messages(ctx);
    expect(msgs).toHaveLength(1);
    expect((msgs[0] as { role: string; content: string }).role).toBe("assistant");
    expect((msgs[0] as { content: string }).content).toBe("");
    expect(lastToolCalls(ctx)).toHaveLength(1);
  });

  it("defensive normalization: name string-coerced, args coerced to object", () => {
    const ctx = makeCtx("t1", {
      messages: [{ id: "a1", role: "assistant", content: "" }],
    });
    toolCallStartHandler(
      ctx,
      makeEvent("tool_call_start", {
        tool_call_id: "tc1",
        name: 42 as unknown as string,
        args: "not an object" as unknown as Record<string, unknown>,
      }),
    );
    const card = lastToolCalls(ctx)[0];
    expect(card.name).toBe("42");
    expect(card.args).toEqual({});
  });
});