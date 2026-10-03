/**
 * stepStartHandler — covers:
 *  1. step_start sets currentStep + writes to ref.
 *  2. step <= 0 clamps to 1 (defensive).
 *  3. step_end is a no-op (no state change).
 */
import { describe, expect, it } from "vitest";
import { stepEndHandler } from "./stepEndHandler";
import { stepStartHandler } from "./stepStartHandler";
import { makeCtx, makeEvent } from "./__tests__/testHelpers";

describe("stepStartHandler / stepEndHandler", () => {
  it("step_start sets currentStep + writes to ref", () => {
    const ctx = makeCtx("t1");
    stepStartHandler(ctx, makeEvent("step_start", { step: 2 }));
    expect(ctx.state.t1.currentStep).toBe(2);
    expect(ctx.stepByThreadRef.current.get("t1")).toBe(2);
  });

  it("step_start clamps step <= 0 to 1", () => {
    const ctx = makeCtx("t1");
    stepStartHandler(ctx, makeEvent("step_start", { step: 0 }));
    expect(ctx.state.t1.currentStep).toBe(1);
    stepStartHandler(
      ctx,
      makeEvent("step_start", { step: -5 as unknown as number }),
    );
    expect(ctx.state.t1.currentStep).toBe(1);
  });

  it("step_end is a no-op", () => {
    const ctx = makeCtx("t1", { currentStep: 3 });
    stepEndHandler(ctx, makeEvent("step_end", {}));
    expect(ctx.state.t1.currentStep).toBe(3);
  });
});