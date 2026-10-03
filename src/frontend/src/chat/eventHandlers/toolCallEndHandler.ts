/**
 * toolCallEndHandler — close out a running ToolCall with its
 * result / status / elapsed_ms.
 *
 * Behavior:
 * 1. Normalize ``evt.tool_call_id`` to a non-empty string. Drop
 *    silently on empty id — same rationale as
 *    ``toolCallStartHandler``.
 * 2. Normalize ``evt.ok`` to a boolean (default ``true``). The
 *    backend should always send the field, but the default of
 *    true is the historically-prevailing case and matches the
 *    "fill in on success" behavior the renderer relies on.
 * 3. Normalize ``evt.elapsed_ms`` to a non-negative number
 *    (default 0). ``evt.step`` falls back to whatever was on the
 *    start card (so a card with no step field keeps its
 *    undefined step), and only swaps to a number if the wire
 *    sent one.
 * 4. Find the matching start card by id:
 *
 *    a) If found, patch the card in place by index. The spread
 *       preserves ``started_at`` and ``args`` from the start event.
 *    b) If NOT found AND an assistant bubble exists, synthesize a
 *       closed card so the UI shows something rather than a
 *       dangling "running" entry. Name falls back to ``(unknown)``.
 *
 * 5. ``ended_at`` is stamped client-side at the moment the event
 *    arrives (not from the wire) — small clock skew vs the server
 *    is acceptable for a per-card timing display.
 */
import type { AgentEvent, ToolCall } from "../../chat-utils";
import { patchThreadMessages } from "./context";
import type { HandlerContext } from "./context";

export function toolCallEndHandler(
  ctx: HandlerContext,
  evt: Extract<AgentEvent, { type: "tool_call_end" }>,
): void {
  const toolCallId = String(evt.tool_call_id || "");
  if (!toolCallId) return;

  const summary = String(evt.result_summary || "");
  const ok = typeof evt.ok === "boolean" ? evt.ok : true;
  const elapsed = typeof evt.elapsed_ms === "number" ? evt.elapsed_ms : 0;
  const step = typeof evt.step === "number" ? evt.step : undefined;
  const ended_at = new Date().toISOString();

  ctx.applyThreads((all) =>
    patchThreadMessages(all, ctx.threadId, (msgs) => {
      const next = [...msgs];
      const last = next[next.length - 1];
      const existing = last && last.role === "assistant"
        ? last.tool_calls || []
        : [];

      const idx = existing.findIndex((c: ToolCall) => c.id === toolCallId);

      if (idx >= 0) {
        // Path (a) — patch the running card in place.
        const updated = [...existing];
        updated[idx] = {
          ...updated[idx],
          result: summary,
          ok,
          elapsed_ms: elapsed,
          step: step ?? updated[idx].step,
          ended_at,
        };
        if (last && last.role === "assistant") {
          next[next.length - 1] = { ...last, tool_calls: updated };
        }
      } else if (last && last.role === "assistant") {
        // Path (b) — synthesize a closed card for a missed start.
        next[next.length - 1] = {
          ...last,
          tool_calls: [
            ...existing,
            {
              id: toolCallId,
              name: "(unknown)",
              result: summary,
              ok,
              elapsed_ms: elapsed,
              step,
              ended_at,
            },
          ],
        };
      }
      return next;
    }),
  );
}