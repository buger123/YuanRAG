/**
 * toolCallEndHandler — covers:
 *  1. Matches by id, patches result / ok / elapsed_ms / ended_at.
 *  2. Falls back to existing card's step if event step is undefined.
 *  3. Synthesizes a closed card when start never arrived.
 *  4. Empty tool_call_id is a no-op.
 *  5. ok defaults to true when event omits it (defensive).
 */
import { describe, expect, it } from "vitest";
import { toolCallEndHandler } from "./toolCallEndHandler";
import { toolCallStartHandler } from "./toolCallStartHandler";
import { makeCtx, makeEvent, messages } from "./__tests__/testHelpers";
import type { ToolCall } from "../../chat-utils";

function lastToolCalls(ctx: ReturnType<typeof makeCtx>): ToolCall[] {
  const msgs = messages(ctx);
  const last = msgs[msgs.length - 1] as { tool_calls?: ToolCall[] };
  return last.tool_calls || [];
}

describe("toolCallEndHandler", () => {
  it("matches by id and patches result / ok / elapsed_ms", () => {
    const ctx = makeCtx("t1", {
      messages: [{ id: "a1", role: "assistant", content: "" }],
    });
    toolCallStartHandler(
      ctx,
      makeEvent("tool_call_start", {
        tool_call_id: "tc1",
        name: "search_docs",
        args: { q: "x" },
        step: 2,
      }),
    );
    toolCallEndHandler(
      ctx,
      makeEvent("tool_call_end", {
        tool_call_id: "tc1",
        result_summary: "found 3 results",
        ok: true,
        elapsed_ms: 123,
      }),
    );
    const card = lastToolCalls(ctx)[0];
    expect(card.id).toBe("tc1");
    expect(card.result).toBe("found 3 results");
    expect(card.ok).toBe(true);
    expect(card.elapsed_ms).toBe(123);
    expect(card.step).toBe(2); // preserved from start
    expect(typeof card.ended_at).toBe("string");
  });

  it("synthesizes a closed card when start never arrived", () => {
    const ctx = makeCtx("t1", {
      messages: [{ id: "a1", role: "assistant", content: "" }],
    });
    toolCallEndHandler(
      ctx,
      makeEvent("tool_call_end", {
        tool_call_id: "tc-orphan",
        result_summary: "synthesized",
        ok: false,
        elapsed_ms: 50,
      }),
    );
    const cards = lastToolCalls(ctx);
    expect(cards).toHaveLength(1);
    expect(cards[0].id).toBe("tc-orphan");
    expect(cards[0].name).toBe("(unknown)");
    expect(cards[0].ok).toBe(false);
    expect(cards[0].elapsed_ms).toBe(50);
  });

  it("empty tool_call_id is a no-op", () => {
    const ctx = makeCtx("t1", {
      messages: [{ id: "a1", role: "assistant", content: "" }],
    });
    toolCallEndHandler(
      ctx,
      makeEvent("tool_call_end", { tool_call_id: "", result_summary: "x" }),
    );
    expect(lastToolCalls(ctx)).toHaveLength(0);
  });

  it("ok defaults to true when event omits the field", () => {
    const ctx = makeCtx("t1", {
      messages: [{ id: "a1", role: "assistant", content: "" }],
    });
    toolCallStartHandler(
      ctx,
      makeEvent("tool_call_start", { tool_call_id: "tc1", name: "noop" }),
    );
    toolCallEndHandler(
      ctx,
      makeEvent("tool_call_end", {
        tool_call_id: "tc1",
        result_summary: "ok",
        // ok omitted on purpose
      }),
    );
    expect(lastToolCalls(ctx)[0].ok).toBe(true);
  });
});