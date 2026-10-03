/**
 * stepEndHandler — signals that the current ReAct step closed.
 *
 * Behavior: NONE. step_end doesn't drive any separate UI state —
 * the NEXT ``step_start`` flips ``currentStep`` back up, and
 * ``answer_complete`` / ``done`` reset it to 0 when the turn
 * ends. The ref is kept (see ``stepStartHandler``) so a future
 * history-replay feature could expose the max-step-reached.
 *
 * Why keep the handler at all?
 * ----------------------------
 * Two reasons:
 *   1. The dispatch table (``dispatch.ts``) is exhaustive over
 *      ``AgentEvent["type"]`` — removing this entry would force a
 *      ``default: never`` branch and silently break if a future
 *      wire version added a new event type.
 *   2. A no-op handler is a clear marker in the test suite that
 *      "this event is expected to fire and produce no state change"
 *      — if it ever DOES produce a change, the test will fail and
 *      the contributor will know to revisit.
 */
import type { AgentEvent } from "../../chat-utils";
import type { HandlerContext } from "./context";

export function stepEndHandler(
  _ctx: HandlerContext,
  _evt: Extract<AgentEvent, { type: "step_end" }>,
): void {
  // Intentionally empty — see file header.
}