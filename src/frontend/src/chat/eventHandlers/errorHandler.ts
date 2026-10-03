/**
 * errorHandler — append a visible error bubble and tear down the
 * stream + socket for the thread.
 *
 * Behavior:
 *
 * 1. Append a ``role: "error"`` message bubble with the server's
 *    error message. ChatPane renders this with the red-tinted
 *    error styling so the user has a clear signal something went
 *    wrong.
 *
 * 2. Clear the per-thread streaming state: ``webSearching`` AND
 *    ``isStreaming`` both go false. The thinking pill must NOT
 *    keep showing after an error — the user reads "网络错误" +
 *    "正在联网搜索…" at the same time otherwise.
 *
 * 3. **P0-F3 — explicitly close the WS on the error branch** so
 *    the ``onclose`` that fires (with code 1006 / abnormal)
 *    doesn't re-schedule a reconnect for a thread the agent has
 *    already declared dead. Without this,
 *    ``shouldReconnect(tid)`` would see ``isStreaming === true``
 *    (we just cleared it but middleware may read pre-clear state)
 *    and try to dial home again, spinning through the 1k/2k/5k
 *    backoff ladder for a thread the user has already seen error
 *    out.
 *
 *    The ``done`` branch above does the same — this closes the
 *    symmetric gap for the error path.
 *
 *    The ``closeWebSocket`` reason "error" is forwarded to QW-13
 *    reconnect-with-backoff logging so post-mortem logs can
 *    distinguish "user navigated away" closes from "server error"
 *    closes from "abnormal code 1006" closes.
 */
import type { AgentEvent } from "../../chat-utils";
import { patchThread, patchThreadMessages } from "./context";
import type { HandlerContext } from "./context";

export function errorHandler(
  ctx: HandlerContext,
  evt: Extract<AgentEvent, { type: "error" }>,
): void {
  // 1. Append error bubble.
  ctx.applyThreads((all) =>
    patchThreadMessages(all, ctx.threadId, (msgs) => [
      ...msgs,
      { id: crypto.randomUUID(), role: "error", content: evt.message },
    ]),
  );
  // 2. Clear streaming state.
  ctx.applyThreads((all) =>
    patchThread(all, ctx.threadId, {
      webSearching: false,
      isStreaming: false,
    }),
  );
  // 3. Close the WS so onclose can't re-trigger reconnect for a dead thread.
  ctx.closeWebSocket(ctx.threadId, "error");
}