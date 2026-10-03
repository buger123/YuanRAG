/**
 * groundingHandler — set the per-message grounding status used by
 * ChatPane to render the "引用核查" banner.
 *
 * Behavior:
 * 1. Patch the LAST assistant message's ``grounding`` field with
 *    ``evt.status``. If no assistant message exists, the patch is
 *    a silent no-op (no seeded bubble — grounding is meaningless
 *    without an answer).
 *
 * The backend emits ``grounding`` events from the post-generation
 * hallucination check (Item 7's ``check_hallucination`` node).
 * Status values are free-form strings; ChatPane maps them to
 * colors via a CSS lookup table.
 */
import type { AgentEvent } from "../../chat-utils";
import { patchThreadMessages } from "./context";
import type { HandlerContext } from "./context";

export function groundingHandler(
  ctx: HandlerContext,
  evt: Extract<AgentEvent, { type: "grounding" }>,
): void {
  ctx.applyThreads((all) =>
    patchThreadMessages(all, ctx.threadId, (msgs) => {
      const next = [...msgs];
      const last = next[next.length - 1];
      if (last && last.role === "assistant") {
        next[next.length - 1] = { ...last, grounding: evt.status };
      }
      return next;
    }),
  );
}