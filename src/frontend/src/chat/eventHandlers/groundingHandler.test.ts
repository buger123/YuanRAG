/**
 * groundingHandler — covers:
 *  1. grounding sets the last assistant's grounding field.
 *  2. grounding is a no-op if no assistant message exists yet.
 */
import { describe, expect, it } from "vitest";
import { groundingHandler } from "./groundingHandler";
import { makeCtx, makeEvent, messages } from "./__tests__/testHelpers";

describe("groundingHandler", () => {
  it("sets the last assistant's grounding field", () => {
    const ctx = makeCtx("t1", {
      messages: [{ id: "a1", role: "assistant", content: "Hi" }],
    });
    groundingHandler(ctx, makeEvent("grounding", { status: "verified" }));
    expect((messages(ctx)[0] as { grounding?: string }).grounding).toBe(
      "verified",
    );
  });

  it("no-op when no assistant exists", () => {
    const ctx = makeCtx("t1", { messages: [] });
    groundingHandler(ctx, makeEvent("grounding", { status: "verified" }));
    expect(messages(ctx)).toHaveLength(0);
  });
});