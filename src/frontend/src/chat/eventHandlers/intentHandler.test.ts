/**
 * intentHandler — covers:
 *  1. intent + corrected_query patch the last assistant.
 *  2. intent without corrected_query still patches the assistant.
 *  3. Empty intent is a silent no-op.
 *  4. Intent seeds an empty assistant bubble if none exists.
 */
import { describe, expect, it } from "vitest";
import { intentHandler } from "./intentHandler";
import { makeCtx, makeEvent, messages } from "./__tests__/testHelpers";

describe("intentHandler", () => {
  it("patches intent + corrected_query onto the last assistant", () => {
    const ctx = makeCtx("t1", {
      messages: [{ id: "a1", role: "assistant", content: "Hi" }],
    });
    intentHandler(
      ctx,
      makeEvent("intent", {
        intent: "qa_complex",
        corrected_query: "tell me about the transformers architecture",
      }),
    );
    const m = messages(ctx)[0] as {
      intent?: string;
      corrected_query?: string;
    };
    expect(m.intent).toBe("qa_complex");
    expect(m.corrected_query).toBe("tell me about the transformers architecture");
  });

  it("patches intent without corrected_query", () => {
    const ctx = makeCtx("t1", {
      messages: [{ id: "a1", role: "assistant", content: "Hi" }],
    });
    intentHandler(ctx, makeEvent("intent", { intent: "greeting" }));
    const m = messages(ctx)[0] as {
      intent?: string;
      corrected_query?: string;
    };
    expect(m.intent).toBe("greeting");
    expect(m.corrected_query).toBeUndefined();
  });

  it("empty intent is a no-op", () => {
    const ctx = makeCtx("t1", {
      messages: [{ id: "a1", role: "assistant", content: "Hi" }],
    });
    intentHandler(ctx, makeEvent("intent", { intent: "" }));
    const m = messages(ctx)[0] as { intent?: string };
    expect(m.intent).toBeUndefined();
  });

  it("seeds an empty assistant bubble if none exists", () => {
    const ctx = makeCtx("t1", { messages: [] });
    intentHandler(
      ctx,
      makeEvent("intent", {
        intent: "qa_complex",
        corrected_query: "fix typo",
      }),
    );
    const msgs = messages(ctx);
    expect(msgs).toHaveLength(1);
    const m = msgs[0] as {
      role: string;
      content: string;
      intent?: string;
      corrected_query?: string;
    };
    expect(m.role).toBe("assistant");
    expect(m.content).toBe("");
    expect(m.intent).toBe("qa_complex");
    expect(m.corrected_query).toBe("fix typo");
  });
});