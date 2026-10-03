/**
 * doneHandler — terminal event of a turn. Cleans up per-turn
 * refs, resets streaming state, tells the sidebar the session
 * might have changed, and closes the WS so the next request
 * lazily reopens a fresh one.
 *
 * Behavior:
 *
 * 1. **Clear per-thread web-search tracker**: ``done`` fires after
 *    ``answer_complete`` has already stamped the banner onto the
 *    final message, so the ref isn't needed for the current turn.
 *    Clearing prevents the NEXT turn's ``answer_complete`` from
 *    inheriting the flag from a previous turn's web fallback. A
 *    no-web turn following a web turn would otherwise incorrectly
 *    show the "本回答参考了联网搜索结果" footer.
 *
 * 2. **Clear ReAct step counter**: belt-and-suspenders against a
 *    future event-order swap. ``answer_complete`` already resets
 *    it to 0; ``done`` resets again for safety.
 *
 * 3. **Reset streaming state**: ``webSearching``, ``isStreaming``,
 *    ``currentStep`` all go false/0. ``isStreaming: false`` stops
 *    the typing-dots indicator; ``currentStep: 0`` removes the
 *    "Step N" pill.
 *
 * 4. **Tell the Sidebar the session list / message counts may have
 *    changed**: dispatch a window-level ``rag:session-updated``
 *    CustomEvent. A new thread may have just been created, or an
 *    existing thread's message_count went up. The Sidebar listens
 *    for this and triggers its listSessions refetch.
 *
 * 5. **Close the socket** so the next request on this thread
 *    lazily reopens a fresh one. The WS is kept alive until
 *    ``done`` so any trailing events (sources, grounding) all
 *    arrive on the same connection. ``closeWebSocket`` is a
 *    no-op if the socket isn't open — the inline version had a
 *    manual readyState check which the hook now owns.
 *
 *    No ``reason`` is passed here (only ``error`` passes one) —
 *    ``done`` is the normal-close path.
 */
import type { AgentEvent } from "../../chat-utils";
import { patchThread } from "./context";
import type { HandlerContext } from "./context";

export function doneHandler(
  ctx: HandlerContext,
  _evt: Extract<AgentEvent, { type: "done" }>,
): void {
  // 1. Per-thread ref cleanup.
  ctx.webSearchedByThreadRef.current.delete(ctx.threadId);
  // 2. Step counter cleanup.
  ctx.stepByThreadRef.current.delete(ctx.threadId);
  // 3. Reset streaming state.
  ctx.applyThreads((all) =>
    patchThread(all, ctx.threadId, {
      webSearching: false,
      isStreaming: false,
      currentStep: 0,
    }),
  );
  // 4. Sidebar refresh signal.
  ctx.dispatchWindowEvent("rag:session-updated");
  // 5. Close the WS so the next request lazily reopens a fresh one.
  ctx.closeWebSocket(ctx.threadId);
}