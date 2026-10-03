/**
 * reasoningHandler — append an LLM reasoning chunk (Anthropic
 * extended-thinking blocks) to the active assistant message.
 *
 * Behavior:
 * 1. Append ``evt.content`` to the last assistant message's
 *    ``reasoning`` field (accumulating, not replacing). Reasoning
 *    never appears in the chat body; ChatPane's ThinkingDrawer
 *    renders it as a collapsible section.
 * 2. If no assistant message exists yet (very first event of a turn
 *    is a thinking block), seed an empty assistant so the reasoning
 *    has somewhere to live. Subsequent tokens append to the
 *    assistant's ``content`` field on the same message.
 *
 * Unlike ``tokenHandler``, this handler does NOT flip
 * ``isStreaming`` — reasoning alone doesn't mean we're streaming
 * the visible answer yet. ``doneHandler`` flips it off; the next
 * ``tokenHandler`` flips it back on.
 */
import type { AgentEvent } from "../../chat-utils";
import { patchThreadMessages } from "./context";
import type { HandlerContext } from "./context";

export function reasoningHandler(
  ctx: HandlerContext,
  evt: Extract<AgentEvent, { type: "reasoning" }>,
): void {
  ctx.applyThreads((all) =>
    patchThreadMessages(all, ctx.threadId, (msgs) => {
      const next = [...msgs];
      const last = next[next.length - 1];
      if (last && last.role === "assistant") {
        next[next.length - 1] = {
          ...last,
          reasoning: (last.reasoning || "") + evt.content,
        };
      } else {
        // No assistant yet — seed an empty one for reasoning to attach.
        next.push({
          id: crypto.randomUUID(),
          role: "assistant",
          content: "",
          reasoning: evt.content,
        });
      }
      return next;
    }),
  );
}