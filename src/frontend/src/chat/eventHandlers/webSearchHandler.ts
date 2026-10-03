/**
 * webSearchHandler — emit the "🔎 正在联网搜索…" thinking pill.
 *
 * Behavior:
 * 1. If ``evt.attempted`` is true, mark the per-thread web-searched
 *    ref so the FOLLOWING ``answer_complete`` can compute its
 *    ``source_kinds`` fallback. (The banner now primarily reads
 *    ``evt.source_kinds`` from answer_complete directly; the ref is
 *    only a fallback for older wire payloads.)
 * 2. Flip ``webSearching: true`` on the thread state — ChatPane
 *    uses this to render the pill. ``answer_complete`` flips it
 *    back to false when the search returned.
 *
 * v1.1.15 history: the original code used the ``webSearched``
 * ref to drive the entire "联网搜索结果" footer banner. v1.1.15
 * moved the banner onto ``answer_complete.source_kinds`` (more
 * accurate — only shown when the answer actually used web content).
 * This handler kept the ref-only flip because ChatPane's pill UI
 * still keys off ``webSearching`` (the "正在联网搜索…" indicator
 * during the search itself, before the answer is ready).
 */
import type { AgentEvent } from "../../chat-utils";
import { patchThread } from "./context";
import type { HandlerContext } from "./context";

export function webSearchHandler(
  ctx: HandlerContext,
  evt: Extract<AgentEvent, { type: "web_search" }>,
): void {
  if (evt.attempted) {
    ctx.webSearchedByThreadRef.current.set(ctx.threadId, true);
  }
  ctx.applyThreads((all) =>
    patchThread(all, ctx.threadId, { webSearching: true }),
  );
}