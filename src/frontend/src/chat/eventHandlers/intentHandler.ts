/**
 * intentHandler — store the cheap-model pre-ReAct classification
 * on the current assistant message.
 *
 * Behavior:
 * 1. Normalize ``evt.intent`` to a string (defensive — older wire
 *    payloads sent undefined for greeting turns). Drop empty
 *    intents silently.
 * 2. Optional ``evt.corrected_query`` is the cheap-model typo
 *    rewrite. ChatPane shows "已纠错: …" only when both are
 *    present and the corrected form differs from the user input.
 * 3. If no assistant bubble exists yet (intent can arrive before
 *    the first token), seed an empty assistant so the intent
 *    metadata has a place to live. Subsequent tokens fill the
 *    content.
 */
import type { AgentEvent } from "../../chat-utils";
import { patchThreadMessages } from "./context";
import type { HandlerContext } from "./context";

export function intentHandler(
  ctx: HandlerContext,
  evt: Extract<AgentEvent, { type: "intent" }>,
): void {
  const intent = typeof evt.intent === "string" ? evt.intent : "";
  const corrected =
    typeof evt.corrected_query === "string"
      ? evt.corrected_query
      : undefined;
  if (!intent) return;

  ctx.applyThreads((all) =>
    patchThreadMessages(all, ctx.threadId, (msgs) => {
      const next = [...msgs];
      const last = next[next.length - 1];
      if (last && last.role === "assistant") {
        next[next.length - 1] = {
          ...last,
          intent,
          corrected_query: corrected,
        };
      } else {
        // Seed an empty assistant bubble for intent metadata.
        next.push({
          id: crypto.randomUUID(),
          role: "assistant",
          content: "",
          intent,
          corrected_query: corrected,
        });
      }
      return next;
    }),
  );
}