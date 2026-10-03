/**
 * tokenHandler — append an LLM-streamed text chunk to the active
 * assistant message.
 *
 * Behavior (from App.tsx's inline ``handleAgentEvent``):
 *
 * 1. Normalize ``evt.content`` to a string. The backend should
 *    always send strings, but structured content blocks have appeared
 *    from some providers in the past (Anthropic ``[{"text":"..."}]``
 *    arrays). We walk the array, pull out string entries and the
 *    ``text`` field of object entries, join with no separator.
 * 2. Drop empty chunks silently — they trigger no state change.
 *    This handles the "first event is an empty token" edge case some
 *    providers emit.
 * 3. Append the chunk to the LAST message if it's an assistant
 *    message; otherwise seed a new assistant message. This is what
 *    makes the streamed answer appear as one growing bubble.
 * 4. Flip ``isStreaming: true`` so ChatPane shows the streaming
 *    indicator while tokens are arriving.
 *
 * Tests in ``tokenHandler.test.ts`` cover each of these.
 */
import type { AgentEvent } from "../../chat-utils";
import { patchThread, patchThreadMessages } from "./context";
import type { HandlerContext } from "./context";

export function tokenHandler(
  ctx: HandlerContext,
  evt: Extract<AgentEvent, { type: "token" }>,
): void {
  // 1. Normalize content to a string.
  let chunk = "";
  const raw: unknown = (evt as { content: unknown }).content;
  if (typeof raw === "string") {
    chunk = raw;
  } else if (Array.isArray(raw)) {
    chunk = (raw as unknown[])
      .map((b) =>
        typeof b === "string"
          ? b
          : b && typeof (b as { text?: unknown }).text === "string"
          ? (b as { text: string }).text
          : "",
      )
      .join("");
  }
  // 2. Drop empty chunks.
  if (!chunk) return;

  // 3. Append to last assistant, or seed.
  ctx.applyThreads((all) =>
    patchThreadMessages(all, ctx.threadId, (msgs) => {
      const next = [...msgs];
      const last = next[next.length - 1];
      if (last && last.role === "assistant") {
        next[next.length - 1] = {
          ...last,
          content: last.content + chunk,
        };
      } else {
        next.push({
          id: crypto.randomUUID(),
          role: "assistant",
          content: chunk,
        });
      }
      return next;
    }),
  );
  // 4. Mark as streaming.
  ctx.applyThreads((all) => patchThread(all, ctx.threadId, { isStreaming: true }));
}