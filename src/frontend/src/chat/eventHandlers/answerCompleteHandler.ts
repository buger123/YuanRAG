/**
 * answerCompleteHandler — the most complex handler. Lands the
 * authoritative final answer from the server, sources, and the
 * web-search footer banner. Owns the canonical content overwrite
 * that defends against streamed-body duplication.
 *
 * Behavior:
 *
 * 1. **Sources default**: ``Source.source_kind`` is required post
 *    SoT-1, but legacy persisted messages (pre-v1.1.13) won't
 *    carry it. Default to ``"local"`` — historical sources were
 *    all local before v1.1.13, so the legacy default is correct.
 *
 * 2. **Banner kinds from the wire**: v1.1.15+ reads ``source_kinds``
 *    from the wire (the backend's ``_derive_source_kinds``
 *    aggregates across the merged local + web doc list). The
 *    per-thread ``webSearchedByThreadRef`` is only a fallback for
 *    legacy messages that still key banner rendering off the bool.
 *    Prefer ``source_kinds`` if non-empty; otherwise fall back.
 *
 * 3. **Dispatch a window-level ``rag:sources`` event** so the
 *    Sidebar (which owns the source-list panel for the latest
 *    turn) can update without us threading the sources array
 *    through React state — same wire the inline App.tsx used.
 *
 * 4. **Canonical content overwrite** — THE ONLY defense against
 *    streamed-body duplication:
 *
 *    Use the server's ``evt.answer`` as the single source of
 *    truth instead of trusting the streamed accumulation in
 *    ``last.content``. ``last.content`` is only useful as a
 *    fallback if ``answerText`` is empty (defensive against a
 *    malformed event).
 *
 *    v2.0.13 — guard the canonical overwrite with
 *    ``trim().length > 0`` instead of bare ``length > 0``.
 *    The previous check accepted whitespace-only canonicals
 *    (``" "``, ``"\n"``) as "real content" and clobbered the
 *    accumulated streamed text with empty space. This happens
 *    when the LLM streams only ``<tool_call>...</tool_call>``
 *    XML (the backend's ``_strip_tool_blocks`` collapses those
 *    blocks to whitespace) or only thinking blocks (no visible
 *    text reaches ``answer_text``). Falling through to
 *    ``last?.content ?? ""`` preserves the streamed
 *    accumulation in those degenerate paths.
 *
 * 5. **Preserve the tool_calls log**: the server's canonical
 *    answerComplete event doesn't echo the cards back (they're
 *    already wired through their own ``tool_call_*`` events);
 *    without this spread the canonical overwrite would clobber
 *    them. Same for ``created_at`` (forwarded verbatim from the
 *    wire so history replay sees the same timestamp as the
 *    AIMessage in the checkpointer).
 *
 * 6. **Tokenize citations**: ``tokenizeCitations(answerText)``
 *    turns ``[1]`` markers into clickable chips. Re-tokenized
 *    from the canonical text (not ``last.content``) so the
 *    chips match the visible answer.
 *
 * 7. **Reset step counter**: the ReAct loop is closed. Reset
 *    ``stepByThreadRef`` AND ``currentStep`` to 0. ``done`` is
 *    the canonical "I'm done" signal but a partial-response
 *    error path can close out without ``done`` ever firing, so
 *    resetting here is the defensive choice. ``webSearching``
 *    also flips back to false (the search returned).
 *
 * History of attempts to dedup at the backend (FELL BACK TO
 * THE FRONTEND OVERWRITE):
 *
 * - v1.1.4 tried to dedup in ``src/agent/runner.py``
 *   ``stream_agent`` and ``src/agent/nodes/generate.py``
 *   ``generate_answer_async`` by tracking ``seen_chunk_ids``.
 *   REVERTED — ``AIMessageChunk.id`` is the message/run id, NOT
 *   a per-chunk unique id; all chunks in a single astream share
 *   one id, so the dedup skipped every chunk after the first
 *   and broke every response with ``(no answer text generated)``.
 *   See ``tests/test_runner_dedup.py`` module docstring for the
 *   full post-mortem.
 */
import type { AgentEvent, Source } from "../../chat-utils";
import {
  patchThread,
  patchThreadMessages,
  withDefaultSourceKind,
} from "./context";
import type { HandlerContext } from "./context";

export function answerCompleteHandler(
  ctx: HandlerContext,
  evt: Extract<AgentEvent, { type: "answer_complete" }>,
): void {
  // 1. Sources with default source_kind.
  const sources: Source[] = withDefaultSourceKind(evt.sources);

  // 2. Banner kinds from the wire (with ref fallback).
  const sourceKinds = Array.isArray(evt.source_kinds) ? evt.source_kinds : [];
  const webSearched =
    sourceKinds.length > 0
      ? sourceKinds.includes("web")
      : ctx.webSearchedByThreadRef.current.get(ctx.threadId) === true;

  // 3. Dispatch window event for the Sidebar source-list panel.
  ctx.dispatchWindowEvent("rag:sources", sources);

  // 4-7. Canonical overwrite + tool_calls preservation + step reset.
  ctx.applyThreads((all) => {
    const updated = patchThreadMessages(all, ctx.threadId, (msgs) => {
      const next = [...msgs];
      const last = next[next.length - 1];
      const answerText = evt.answer ?? "";
      const canonicalContent =
        answerText.trim().length > 0
          ? answerText
          : last?.content ?? "";

      if (last && last.role === "assistant") {
        next[next.length - 1] = {
          ...last,
          content: canonicalContent,
          sources,
          tokens: ctx.tokenizeCitations(canonicalContent),
          webSearched,
          source_kinds: sourceKinds,
          created_at:
            typeof evt.created_at === "string"
              ? evt.created_at
              : last.created_at,
        };
      } else {
        next.push({
          id: crypto.randomUUID(),
          role: "assistant",
          content: canonicalContent,
          sources,
          tokens: ctx.tokenizeCitations(canonicalContent),
          webSearched,
          source_kinds: sourceKinds,
          created_at:
            typeof evt.created_at === "string"
              ? evt.created_at
              : undefined,
        });
      }
      return next;
    });
    ctx.stepByThreadRef.current.set(ctx.threadId, 0);
    return patchThread(updated, ctx.threadId, {
      webSearching: false,
      currentStep: 0,
    });
  });
}