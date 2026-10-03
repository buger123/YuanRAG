/**
 * answerCompleteHandler — covers the most complex handler:
 *
 *  1. Sources default source_kind to "local" when missing.
 *  2. Canonical content overwrite overwrites the streamed accumulation.
 *  3. v2.0.13 — whitespace-only canonical FALLS BACK to last.content
 *     (the streamed accumulation), preserving the answer when the
 *     server stripped visible text.
 *  4. tool_calls accumulated during ReAct are preserved on the
 *     canonical overwrite.
 *  5. window-level ``rag:sources`` event is dispatched.
 *  6. step counter resets to 0; webSearching flips to false.
 *  7. webSearched derived from source_kinds (preferred over ref).
 *  8. webSearched derived from ref when source_kinds is empty (legacy).
 *  9. Seeds a new assistant bubble if none exists.
 * 10. Empty answer field is OK (handler still seeds / overwrites).
 */
import { describe, expect, it } from "vitest";
import { answerCompleteHandler } from "./answerCompleteHandler";
import { toolCallStartHandler } from "./toolCallStartHandler";
import { makeCtx, makeEvent, messages } from "./__tests__/testHelpers";
import type { Source, ToolCall } from "../../chat-utils";

function lastAssistant(ctx: ReturnType<typeof makeCtx>) {
  const msgs = messages(ctx);
  return msgs[msgs.length - 1] as {
    content: string;
    sources?: Source[];
    tool_calls?: ToolCall[];
    webSearched?: boolean;
    source_kinds?: string[];
    tokens?: unknown[];
    created_at?: string;
  };
}

describe("answerCompleteHandler", () => {
  it("overwrites streamed accumulation with server canonical answer", () => {
    const ctx = makeCtx("t1", {
      messages: [{ id: "a1", role: "assistant", content: "Hello wo" }],
    });
    answerCompleteHandler(
      ctx,
      makeEvent("answer_complete", {
        answer: "Hello world",
        sources: [],
        source_kinds: [],
      }),
    );
    expect(lastAssistant(ctx).content).toBe("Hello world");
  });

  it("v2.0.13 — whitespace-only canonical falls back to last.content", () => {
    const ctx = makeCtx("t1", {
      messages: [{ id: "a1", role: "assistant", content: "streamed text" }],
    });
    answerCompleteHandler(
      ctx,
      makeEvent("answer_complete", {
        answer: "   \n  ", // whitespace only
        sources: [],
        source_kinds: [],
      }),
    );
    // The streamed text wins because the canonical is empty after trim.
    expect(lastAssistant(ctx).content).toBe("streamed text");
  });

  it("preserves tool_calls accumulated during ReAct loop", () => {
    const ctx = makeCtx("t1", {
      messages: [{ id: "a1", role: "assistant", content: "streamed" }],
    });
    toolCallStartHandler(
      ctx,
      makeEvent("tool_call_start", { tool_call_id: "tc1", name: "lookup" }),
    );
    answerCompleteHandler(
      ctx,
      makeEvent("answer_complete", {
        answer: "final answer",
        sources: [],
        source_kinds: [],
      }),
    );
    const a = lastAssistant(ctx);
    expect(a.content).toBe("final answer");
    expect(a.tool_calls?.map((c) => c.id)).toEqual(["tc1"]);
  });

  it("dispatches rag:sources window event", () => {
    const ctx = makeCtx("t1", {
      messages: [{ id: "a1", role: "assistant", content: "" }],
    });
    const sources: Source[] = [
      { id: "s1", filename: "a.pdf", source_kind: "local" } as unknown as Source,
    ];
    answerCompleteHandler(
      ctx,
      makeEvent("answer_complete", {
        answer: "ok",
        sources,
        source_kinds: ["local"],
      }),
    );
    expect(ctx.spy.windowEvents).toContainEqual({
      type: "rag:sources",
      detail: sources,
    });
  });

  it("resets step counter + webSearching", () => {
    const ctx = makeCtx("t1", {
      currentStep: 3,
      webSearching: true,
    });
    answerCompleteHandler(
      ctx,
      makeEvent("answer_complete", {
        answer: "ok",
        sources: [],
        source_kinds: [],
      }),
    );
    expect(ctx.state.t1.currentStep).toBe(0);
    expect(ctx.state.t1.webSearching).toBe(false);
    expect(ctx.stepByThreadRef.current.get("t1")).toBe(0);
  });

  it("webSearched prefers source_kinds.includes('web') over the ref", () => {
    const ctx = makeCtx("t1", {
      messages: [{ id: "a1", role: "assistant", content: "" }],
    });
    // ref is true but wire says source_kinds=[local] — wire wins.
    ctx.webSearchedByThreadRef.current.set("t1", true);
    answerCompleteHandler(
      ctx,
      makeEvent("answer_complete", {
        answer: "ok",
        sources: [],
        source_kinds: ["local"],
      }),
    );
    expect(lastAssistant(ctx).webSearched).toBe(false);
    expect(lastAssistant(ctx).source_kinds).toEqual(["local"]);
  });

  it("webSearched falls back to ref when source_kinds is empty (legacy)", () => {
    const ctx = makeCtx("t1", {
      messages: [{ id: "a1", role: "assistant", content: "" }],
    });
    ctx.webSearchedByThreadRef.current.set("t1", true);
    answerCompleteHandler(
      ctx,
      makeEvent("answer_complete", {
        answer: "ok",
        sources: [],
        // source_kinds intentionally absent
      }),
    );
    expect(lastAssistant(ctx).webSearched).toBe(true);
  });

  it("sources default source_kind to 'local' when missing", () => {
    const ctx = makeCtx("t1", {
      messages: [{ id: "a1", role: "assistant", content: "" }],
    });
    answerCompleteHandler(
      ctx,
      makeEvent("answer_complete", {
        answer: "ok",
        sources: [{ id: "s1", filename: "legacy.pdf" } as unknown as Source],
        source_kinds: ["local"],
      }),
    );
    expect(lastAssistant(ctx).sources?.[0].source_kind).toBe("local");
  });

  it("seeds a new assistant bubble if none exists", () => {
    const ctx = makeCtx("t1", { messages: [] });
    answerCompleteHandler(
      ctx,
      makeEvent("answer_complete", {
        answer: "answer",
        sources: [],
        source_kinds: [],
      }),
    );
    const msgs = messages(ctx);
    expect(msgs).toHaveLength(1);
    expect(lastAssistant(ctx).content).toBe("answer");
  });

  it("created_at forwarded from wire if string", () => {
    const ctx = makeCtx("t1", {
      messages: [
        { id: "a1", role: "assistant", content: "x", created_at: "old" },
      ],
    });
    answerCompleteHandler(
      ctx,
      makeEvent("answer_complete", {
        answer: "ok",
        sources: [],
        source_kinds: [],
        created_at: "2026-09-19T10:00:00Z",
      }),
    );
    expect(lastAssistant(ctx).created_at).toBe("2026-09-19T10:00:00Z");
  });
});