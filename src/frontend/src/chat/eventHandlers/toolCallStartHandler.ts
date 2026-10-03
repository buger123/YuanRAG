/**
 * toolCallStartHandler — append a new ToolCall record to the
 * current assistant message.
 *
 * Behavior:
 * 1. Normalize ``evt.tool_call_id`` to a non-empty string. Drop
 *    silently if the wire somehow sent an empty id — a card
 *    without an id can't be cross-referenced with the matching
 *    ``tool_call_end`` event.
 * 2. Build the ToolCall record. ``name``, ``args`` are best-effort
 *    (string coerce for name; object literal coerce for args).
 *    ``ok: undefined`` is the running-card signal ChatPane keys
 *    off — once ``tool_call_end`` lands and fills ``ok``, the
 *    card flips to the success/failure visual.
 * 3. ``started_at`` is stamped client-side so ToolCallCard can
 *    show "X.Xs" elapsed even before ``tool_call_end`` arrives
 *    (refresh during long-running tool calls).
 * 4. Idempotent: if a card with the same id already exists
 *    (back-to-back re-delivery from the WS layer), the patch is a
 *    no-op — LangChain guarantees no duplicates within a turn, but
 *    a defensive check costs nothing and prevents UI double-cards.
 * 5. If no assistant bubble exists yet, seed one with an empty
 *    content + the tool_calls array so the card has somewhere to
 *    render.
 */
import type { AgentEvent, ToolCall } from "../../chat-utils";
import { patchThreadMessages } from "./context";
import type { HandlerContext } from "./context";

export function toolCallStartHandler(
  ctx: HandlerContext,
  evt: Extract<AgentEvent, { type: "tool_call_start" }>,
): void {
  const toolCallId = String(evt.tool_call_id || "");
  if (!toolCallId) return;

  const call: ToolCall = {
    id: toolCallId,
    name: String(evt.name || ""),
    args:
      evt.args && typeof evt.args === "object"
        ? (evt.args as Record<string, unknown>)
        : {},
    ok: undefined,
    step: typeof evt.step === "number" ? evt.step : undefined,
    started_at: new Date().toISOString(),
  };

  ctx.applyThreads((all) =>
    patchThreadMessages(all, ctx.threadId, (msgs) => {
      const next = [...msgs];
      const last = next[next.length - 1];
      const existing = last && last.role === "assistant"
        ? last.tool_calls || []
        : [];

      // Idempotent — duplicate id is a no-op.
      if (existing.some((c: ToolCall) => c.id === toolCallId)) {
        return msgs;
      }

      const tool_calls = [...existing, call];
      if (last && last.role === "assistant") {
        next[next.length - 1] = { ...last, tool_calls };
      } else {
        next.push({
          id: crypto.randomUUID(),
          role: "assistant",
          content: "",
          tool_calls,
        });
      }
      return next;
    }),
  );
}