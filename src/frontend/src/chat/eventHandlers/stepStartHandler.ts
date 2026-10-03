/**
 * stepStartHandler — bump the per-thread ReAct step counter.
 *
 * Behavior:
 * 1. Normalize ``evt.step`` to a positive integer (defensive —
 *    backend sends 1-based but a future refactor could send 0 or
 *    undefined; we clamp to 1 to avoid rendering "Step 0").
 * 2. Write to ``stepByThreadRef`` (used by answer_complete as a
 *    belt-and-suspenders reset, and by future history-replay
 *    features).
 * 3. Flip ``currentStep`` on the thread state — ChatPane renders
 *    "Step N" next to the thinking pill while this is >0.
 *
 * The ref AND the state-field are kept in sync because the state
 * drives the visible UI and the ref drives history-replay that
 * doesn't necessarily re-trigger renders.
 */
import type { AgentEvent } from "../../chat-utils";
import { patchThread } from "./context";
import type { HandlerContext } from "./context";

export function stepStartHandler(
  ctx: HandlerContext,
  evt: Extract<AgentEvent, { type: "step_start" }>,
): void {
  const step = typeof evt.step === "number" && evt.step > 0 ? evt.step : 1;
  ctx.stepByThreadRef.current.set(ctx.threadId, step);
  ctx.applyThreads((all) => patchThread(all, ctx.threadId, { currentStep: step }));
}