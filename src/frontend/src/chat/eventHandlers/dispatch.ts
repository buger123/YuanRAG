/**
 * dispatchAgentEvent — single entry point that routes an
 * ``AgentEvent`` to its per-type handler module. Replaces the
 * 438-LOC ``handleAgentEvent`` previously inlined in App.tsx.
 *
 * Design notes
 * ------------
 *
 * 1. **Exhaustive switch**: every member of
 *    ``AgentEvent["type"]`` has a case here. The TypeScript
 *    ``never`` check on ``default`` will fail to compile if a new
 *    event type is added to ``AgentEvent`` and this dispatch isn't
 *    updated — turning "missing handler" from a silent runtime
 *    no-op into a compile-time error.
 *
 * 2. **Order matters for some pairings** but the dispatcher itself
 *    is order-agnostic — handlers run in the order the wire sends
 *    events. The canonical content overwrite in
 *    ``answerCompleteHandler`` runs after all ``token`` events for
 *    the same turn have already mutated ``last.content``, which is
 *    the desired behavior.
 *
 * 3. **The ctx arg carries all side effects**: state patches,
 *    ref writes, window-event dispatches, and WS close calls. The
 *    dispatcher itself is pure dispatch — no business logic.
 *
 * 4. **The error path returns silently**: a thrown handler
 *    bubbles up to the caller (the ``useThreadSocket`` onMessage
 *    callback), where it lands in the hook's catch and is logged.
 *    We do NOT swallow handler errors — they're a bug worth
 *    surfacing during development. Production runs keep the WS
 *    open through normal flow so the user just sees a missing
 *    update for that event.
 *
 * Handler contract:
 *   ``(ctx: HandlerContext, evt: AgentEventX) => void``
 *
 * No return value — handlers mutate ``ctx.threads`` via
 * ``ctx.applyThreads`` (the React setState-shaped updater).
 */
import type { AgentEvent } from "../../chat-utils";
import type { HandlerContext } from "./context";
import { tokenHandler } from "./tokenHandler";
import { reasoningHandler } from "./reasoningHandler";
import { webSearchHandler } from "./webSearchHandler";
import { answerCompleteHandler } from "./answerCompleteHandler";
import { groundingHandler } from "./groundingHandler";
import { intentHandler } from "./intentHandler";
import { stepStartHandler } from "./stepStartHandler";
import { stepEndHandler } from "./stepEndHandler";
import { toolCallStartHandler } from "./toolCallStartHandler";
import { toolCallEndHandler } from "./toolCallEndHandler";
import { errorHandler } from "./errorHandler";
import { doneHandler } from "./doneHandler";

export function dispatchAgentEvent(
  ctx: HandlerContext,
  evt: AgentEvent,
): void {
  switch (evt.type) {
    case "token":
      tokenHandler(ctx, evt);
      break;
    case "reasoning":
      reasoningHandler(ctx, evt);
      break;
    case "web_search":
      webSearchHandler(ctx, evt);
      break;
    case "answer_complete":
      answerCompleteHandler(ctx, evt);
      break;
    case "grounding":
      groundingHandler(ctx, evt);
      break;
    case "intent":
      intentHandler(ctx, evt);
      break;
    case "step_start":
      stepStartHandler(ctx, evt);
      break;
    case "step_end":
      stepEndHandler(ctx, evt);
      break;
    case "tool_call_start":
      toolCallStartHandler(ctx, evt);
      break;
    case "tool_call_end":
      toolCallEndHandler(ctx, evt);
      break;
    case "error":
      errorHandler(ctx, evt);
      break;
    case "done":
      doneHandler(ctx, evt);
      break;
    default: {
      // Exhaustiveness check. Adding a new AgentEvent type without
      // adding a case here fails TypeScript compilation.
      const _exhaustive: never = evt;
      void _exhaustive;
    }
  }
}