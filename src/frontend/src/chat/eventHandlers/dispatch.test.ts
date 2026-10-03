/**
 * dispatchAgentEvent — integration test that walks a full turn
 * lifecycle through the dispatcher and asserts the final state.
 *
 * This is the SINGLE integration test that exercises every event
 * type in a single thread, in the order the wire would send them.
 * If the dispatcher ever stops calling a handler, or if a handler
 * forgets to update state, this test catches it.
 *
 * Coverage:
 *   - intent → step_start → reasoning → token* → tool_call_start
 *     → tool_call_end → web_search → answer_complete → grounding
 *     → done
 *   - Final state: isStreaming=false, webSearching=false,
 *     currentStep=0, assistant has content + sources + tool_calls
 *     + reasoning + grounding + intent.
 */
import { describe, expect, it } from "vitest";
import { dispatchAgentEvent } from "./dispatch";
import { makeCtx, makeEvent, messages } from "./__tests__/testHelpers";
import type { Source } from "../../chat-utils";

describe("dispatchAgentEvent — integration", () => {
  it("drives a full turn lifecycle end-to-end", () => {
    const ctx = makeCtx("t1", { messages: [] });

    // intent arrives before tokens (cheap-model classification).
    dispatchAgentEvent(
      ctx,
      makeEvent("intent", {
        intent: "qa_complex",
        corrected_query: "tell me about the transformer",
      }),
    );
    // step_start bumps the counter.
    dispatchAgentEvent(ctx, makeEvent("step_start", { step: 1 }));
    // reasoning accumulates on the seeded assistant.
    dispatchAgentEvent(ctx, makeEvent("reasoning", { content: "User asks about " }));
    dispatchAgentEvent(ctx, makeEvent("reasoning", { content: "transformers." }));
    // tokens append to the assistant.
    dispatchAgentEvent(ctx, makeEvent("token", { content: "A transformer " }));
    dispatchAgentEvent(ctx, makeEvent("token", { content: "is a neural " }));
    dispatchAgentEvent(ctx, makeEvent("token", { content: "network. [1]" }));
    // tool call.
    dispatchAgentEvent(
      ctx,
      makeEvent("tool_call_start", {
        tool_call_id: "tc1",
        name: "search_docs",
        args: { q: "transformer" },
        step: 1,
      }),
    );
    dispatchAgentEvent(
      ctx,
      makeEvent("tool_call_end", {
        tool_call_id: "tc1",
        result_summary: "found 2",
        ok: true,
        elapsed_ms: 80,
      }),
    );
    // web_search attempt.
    dispatchAgentEvent(ctx, makeEvent("web_search", { attempted: false }));
    // answer_complete lands the canonical.
    const sources: Source[] = [
      { id: "s1", filename: "a.pdf", source_kind: "local" } as unknown as Source,
    ];
    dispatchAgentEvent(
      ctx,
      makeEvent("answer_complete", {
        answer: "A transformer is a neural network. [1]",
        sources,
        source_kinds: ["local"],
      }),
    );
    // grounding post-check.
    dispatchAgentEvent(ctx, makeEvent("grounding", { status: "verified" }));
    // done closes the turn.
    dispatchAgentEvent(ctx, makeEvent("done", {}));

    // --- final state assertions ---
    const state = ctx.state.t1;
    expect(state.isStreaming).toBe(false);
    expect(state.webSearching).toBe(false);
    expect(state.currentStep).toBe(0);

    const msgs = messages(ctx);
    // We expect: one assistant bubble (intent seeded it, tokens filled it,
    // tool_call_start preserved cards, answer_complete overwrote content
    // from the canonical, grounding stamped).
    const assistants = msgs.filter(
      (m) => (m as { role: string }).role === "assistant",
    );
    expect(assistants).toHaveLength(1);
    const a = assistants[0] as {
      content: string;
      intent?: string;
      reasoning?: string;
      sources?: Source[];
      tool_calls?: { id: string; ok?: boolean }[];
      grounding?: string;
    };
    expect(a.content).toBe("A transformer is a neural network. [1]");
    expect(a.intent).toBe("qa_complex");
    expect(a.reasoning).toBe("User asks about transformers.");
    expect(a.sources).toEqual(sources);
    expect(a.tool_calls?.[0].id).toBe("tc1");
    expect(a.tool_calls?.[0].ok).toBe(true);
    expect(a.grounding).toBe("verified");

    // Side effects.
    expect(ctx.webSearchedByThreadRef.current.has("t1")).toBe(false);
    expect(ctx.stepByThreadRef.current.has("t1")).toBe(false);
    // window events fired: rag:sources, rag:session-updated.
    expect(ctx.spy.windowEvents.map((e) => e.type)).toEqual([
      "rag:sources",
      "rag:session-updated",
    ]);
    // WS closed by done (no reason) — error did NOT fire.
    expect(ctx.spy.closedThreads).toEqual([
      { threadId: "t1", reason: undefined },
    ]);
  });

  it("error path: error event closes WS with reason 'error'", () => {
    const ctx = makeCtx("t1", {
      messages: [{ id: "a1", role: "assistant", content: "partial" }],
      isStreaming: true,
      webSearching: true,
    });
    dispatchAgentEvent(
      ctx,
      makeEvent("error", { message: "网络中断" }),
    );
    const msgs = messages(ctx);
    expect(msgs).toHaveLength(2);
    expect((msgs[1] as { role: string; content: string }).role).toBe("error");
    expect((msgs[1] as { content: string }).content).toBe("网络中断");
    expect(ctx.state.t1.isStreaming).toBe(false);
    expect(ctx.state.t1.webSearching).toBe(false);
    expect(ctx.spy.closedThreads).toContainEqual({
      threadId: "t1",
      reason: "error",
    });
  });
});