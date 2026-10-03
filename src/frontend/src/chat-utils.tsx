import type React from "react";
import { CitationChip } from "./components/CitationChip";
import type { MessageRecord } from "./api/client";

// Shared chat types used by both App.tsx (which owns the per-thread
// WebSocket + message buffers) and ChatPane.tsx (which renders them).
// Kept in a separate file so App can construct messages on agent
// events without ChatPane pulling in the entire WS lifecycle.

export interface Source {
  index: number;
  filename: string;
  page?: number;
  section?: string;
  sheet?: string;
  text: string;
  score: number;
  url?: string;
  domain?: string;
  /** v2.0.7 SoT-4: required (was optional). The backend always
   *  emits ``"local"`` or ``"web"`` — see
   *  ``src.api.schemas.Source.source_kind``. Older history rows
   *  persisted before this contract still decode (``source_kind``
   *  defaults to ``"local"`` in the consumer), but new wire
   *  payloads always carry it. */
  source_kind: "local" | "web";
  // v2.0.29.4 (Phase 4 PR-1) — wire payload always carries
  // ``chunk_id`` + ``doc_id`` (see ``src.api.schemas.Source``).
  // CitationChip dispatches these on click so debugging tooling
  // can identify the exact source the chip points at. Backward
  // compat: older wire payloads pre-PR-1 carry empty strings,
  // which still satisfy the optional type.
  chunk_id?: string;
  doc_id?: string;
  /** v2.0.29.9 (Phase 8) — verbatim extraction mode marker.
   *  When ``true``, this source was used in extractive mode
   *  (verbatim quote, no rewriting). ``ChatPane`` reads this
   *  to render a 🔒 icon on the bubble. Additive — old wire
   *  payloads carry ``undefined``, treated as ``false``. */
  verbatim?: boolean;
}

/** v2.0 — per-tool-call record rendered as a ``ToolCallCard``.
 *
 * Built up live as ``tool_call_start`` / ``tool_call_end`` events
 * arrive. Persisted server-side via AIMessage.additional_kwargs
 * (v2.0 schema) and rehydrated on history replay.
 *
 * Optional fields: many are filled in only by the matching end
 * event. While a card is "running" (end not yet arrived), the UI
 * uses ``call.ok === undefined`` as the in-flight signal — the
 * runner sends ``tool_call_start`` immediately after the AIMessage
 * is produced and ``tool_call_end`` once the ToolNode returns. */
export interface ToolCall {
  id: string;
  name: string;
  args?: Record<string, unknown>;
  /** Truncated tool result (first 800 chars) — full content lives
   *  on the persisted ToolMessage in the checkpointer. */
  result?: string;
  ok?: boolean;
  step?: number;
  /** ISO 8601 from the runner's ``_ToolCallTracker.start``. */
  started_at?: string;
  ended_at?: string;
  elapsed_ms?: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "error";
  content: string;
  sources?: Source[];
  grounding?: string;
  /** True if this turn triggered a web search fallback.
   *
   * v1.1.15 — deprecated for banner rendering: it was set from a
   * per-thread ref keyed off ``events.web_search(attempted)``, which
   * was both (a) brittle across reconnects and (b) unable to
   * distinguish doc-only vs web-only answers. The frontend now reads
   * ``source_kinds`` for banner logic and falls back to ``webSearched``
   * only as a last-resort hint for messages that pre-date the
   * ``source_kinds`` field. */
  webSearched?: boolean;
  /** v1.1.15 — which kinds of sources shaped this answer.
   *  Drives the banner:
   *    ["local"]           → "本回答参考了文档内容"
   *    ["web"]             → "🔎 本回答参考了联网搜索结果"
   *    ["local", "web"]    → both banners
   *    [] / undefined      → no banner (greeting / direct / legacy)
   *  Comes from the live ``events.answer_complete.source_kinds``
   *  field and from ``MessageRecord.source_kinds`` on history replay
   *  (see ``src/api/schemas.py``). */
  source_kinds?: string[];
  /** Accumulated LLM reasoning (Anthropic extended-thinking blocks). */
  reasoning?: string;
  /** Tokenized message content (post answer_complete) for inline
   *  citation chips. Null until answer_complete fires (or until
   *  history replay populates them for old messages). */
  tokens?: React.ReactNode[];
  /** v2.0 — tool-call log emitted by the ReAct loop. Rendered
   *  as one ``ToolCallCard`` per element, BEFORE the prose body
   *  (so the user sees the agent's reasoning/actions before the
   *  final answer). Persists to history via AIMessage additional
   *  kwargs (``tool_calls``); rehydrated on page refresh. */
  tool_calls?: ToolCall[];
  /** v2.0 — ISO 8601 server-side timestamp on the assistant turn.
   *  Rehydrated from ``MessageRecord.created_at`` on history replay;
   *  live from ``events.answer_complete.created_at``. Renders as a
   *  small dim timestamp under the message (UTC; not localized in
   *  the UI yet). */
  created_at?: string;
  /** v2.0 — pre-ReAct classification result (greeting / summary /
   *  qa_complex). Surfaceable as a subtle "已分析" hint under the
   *  message; the frontend currently uses this only for routing
   *  displays, not for direct rendering. */
  intent?: string;
  /** v2.0 — cheap-model typo-correction of the original query.
   *  Surfaces as "已纠错: <corrected_query>" only when the corrected
   *  form differs from the user's actual input. */
  corrected_query?: string;
}

export type AgentEvent =
  | { type: "token"; content: string }
  | { type: "reasoning"; content: string }
  | { type: "web_search"; attempted: boolean }
  | {
      type: "answer_complete";
      answer: string;
      sources: Source[];
      /** v1.1.15 — aggregated kinds of sources the answer used.
       *  Mirrors ``MessageRecord.source_kinds`` on history replay. */
      source_kinds?: string[];
      /** v2.0 — server-side ISO 8601 timestamp on the assistant
       *  turn. Forwarded as-is into ``m.created_at`` on the live
       *  message. ``undefined`` for older events / older messages. */
      created_at?: string;
    }
  | { type: "grounding"; status: string }
  | { type: "error"; message: string }
  | { type: "done" }
  // ---- v2.0 ReAct events ----
  /** Pre-ReAct classification. Emitted once at the start of every
   *  qa_complex / summary / greeting turn. ``corrected_query`` is the
   *  cheap-model typo rewrite (if any) — the frontend can surface
   *  it as a subtle "已纠错" hint but the prompt the LLM actually
   *  sees is what the agent decides. */
  | { type: "intent"; intent: string; corrected_query?: string }
  /** ReAct step start. ``step`` is 1-based — step 1 is the first
   *  ``react_agent`` call, step 2 is the post-tool re-entry, etc.
   *  Drives the "Step N" pill in ChatPane. ``step`` is optional in
   *  the type because handlers clamp to 1 when missing — the wire
   *  has been observed to emit undefined in legacy payloads. */
  | { type: "step_start"; step?: number }
  | { type: "step_end"; step?: number; ok?: boolean }
  /** Tool-call dispatch. ``args`` is the LLM's raw call payload
   *  (NOT the validated Pydantic args — that's surfaced on the
   *  matching ``tool_call_end`` once the ToolNode runs). ``name``,
   *  ``args``, and ``step`` are all optional because the handler
   *  normalizes each (empty name → "", missing args → {}, missing
   *  step → undefined). The tool_call_id is the only required
   *  field — without it the matching tool_call_end can't find the
   *  card. */
  | {
      type: "tool_call_start";
      name?: string;
      args?: Record<string, unknown>;
      tool_call_id: string;
      step?: number;
    }
  | {
      type: "tool_call_end";
      tool_call_id: string;
      result_summary?: string;
      ok?: boolean;
      step?: number;
      elapsed_ms?: number;
    };

// Split "Hello [1] world [2] foo" into [text, chip(1), text, chip(2), text].
//
// PR-4 (v2.0.26.1): the chip itself moved to
// ``src/frontend/src/components/CitationChip.tsx`` so the
// locale-aware aria-label / title logic lives in exactly one
// place. ``tokenizeCitations`` now just decides WHERE the chips
// go; ``CitationChip`` decides WHAT they say.
export function tokenizeCitations(text: string): React.ReactNode[] {
  const re = /\[(\d+)\]/g;
  const out: React.ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const idx = Number(m[1]);
    out.push(
      <CitationChip
        key={`cite-${m.index}-${idx}`}
        idx={idx}
      />,
    );
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

// v2.0.28.14 — restore MessagesRecord[] → ChatMessage[] for history
// replay. Extracted from the App.tsx ``loadHistory`` mapping chain
// (was inline at lines 679-770) so the transform is pure and
// unit-testable without mocking getSessionMessages or crypto.
// Defensive defaults match the original inline chain 1:1: sources
// default ``source_kind: "local"`` (v2.0.7 SoT-4), tool_calls
// entries without string id/name are dropped (defense against
// malformed historical rows), each ChatMessage gets a fresh
// ``crypto.randomUUID()``. tokens are only computed for assistant
// messages with non-empty sources (mirrors inline behavior).
export function restoreMessages(records: MessageRecord[]): ChatMessage[] {
  return records
    .filter((r) => r.role === "user" || r.role === "assistant")
    .map((r) => {
      // v2.0.7 SoT-4: defensive default for legacy persisted rows
      // that pre-date the required ``source_kind`` field.
      const sources: Source[] | undefined = r.sources
        ? r.sources.map((s) => ({ ...s, source_kind: s.source_kind ?? "local" }))
        : undefined;

      // v1.1.15 — banner kinds survive a page refresh.
      const sourceKinds = Array.isArray(r.source_kinds)
        ? r.source_kinds
        : undefined;

      // v2.0 — rehydrate the tool-call log into ChatPane-shaped
      // records. Drop entries that don't have the required id /
      // name fields rather than coerce (defense against malformed
      // historical rows).
      const toolCalls: ToolCall[] | undefined =
        Array.isArray(r.tool_calls)
          ? (r.tool_calls
              .filter(
                (tc): tc is ToolCall =>
                  tc != null &&
                  typeof tc === "object" &&
                  typeof (tc as { id?: unknown }).id === "string" &&
                  typeof (tc as { name?: unknown }).name === "string",
              )
              .map((tc) => ({
                id: String(tc.id || ""),
                name: String(tc.name || ""),
                args:
                  tc.args && typeof tc.args === "object"
                    ? (tc.args as Record<string, unknown>)
                    : {},
                result:
                  typeof tc.result === "string" ? tc.result : undefined,
                ok: typeof tc.ok === "boolean" ? tc.ok : undefined,
                step: typeof tc.step === "number" ? tc.step : undefined,
                started_at:
                  typeof tc.started_at === "string"
                    ? tc.started_at
                    : undefined,
                ended_at:
                  typeof tc.ended_at === "string"
                    ? tc.ended_at
                    : undefined,
                elapsed_ms:
                  typeof tc.elapsed_ms === "number" ? tc.elapsed_ms : undefined,
              })))
          : undefined;

      // v2.0 — server-side timestamp.
      const createdAt: string | undefined =
        typeof r.created_at === "string" ? r.created_at : undefined;

      return {
        id: crypto.randomUUID(),
        role: r.role as "user" | "assistant",
        content: r.content,
        sources,
        reasoning: r.reasoning,
        source_kinds: sourceKinds,
        tool_calls: toolCalls,
        created_at: createdAt,
        tokens:
          r.role === "assistant" && sources && sources.length > 0
            ? tokenizeCitations(r.content)
            : undefined,
      };
    });
}

// v2.0.28.14 — collapse adjacent assistant ChatMessages that
// belong to the same ReAct turn.
//
// Background (per the v2.0.28.10 contract + v2.0.28.13 wire
// enrichment):
//   * Each tool-using turn persists 2 AIMessages on the wire:
//     (1) the planning-step AIMessage from ``react_agent`` (native
//     ``tool_calls`` attribute, no timing fields), and (2) the
//     synthesis AIMessage from ``react_generate`` (rich log via
//     ``_build_tool_call_log`` with timing fields).
//   * Both AIMessages' ``tool_calls[i].id`` reference the SAME
//     Anthropic ``tool_use.id`` — they are NOT independently
//     identifiable by id alone (see v2.0.28.14 memory trap #2).
//   * Live streaming already handles this via ``answerCompleteHandler``
//     canonical-overwrite (line 116-128): when ``answer_complete``
//     fires, the LAST assistant bubble's ``content``/``sources``/
//     ``tokens`` are replaced in-place, and React reconciles by
//     ``key={m.id}`` so the user sees 1 bubble growing.
//   * On history replay, ``restoreMessages`` produces 2 ChatMessages
//     per turn (1:1 from the wire records). Without this collapse
//     pass the user would see 2 assistant bubbles on reload — the
//     "double bubble" bug fixed here.
//
// Rule: two consecutive ChatMessages with ``role === "assistant"``
// AND both have non-empty ``tool_calls`` merge into one. Otherwise
// pass through (covers single-turn Q&A, single-tool single-round,
// turns separated by a user message, etc.).
//
// Merge strategy:
//   * Synthesis's fields are authoritative for everything EXCEPT
//     ``reasoning``: it has the final answer (content), the rich
//     log (tool_calls with timing), citations (sources), tokens,
//     source_kinds, grounding, created_at, webSearched.
//   * Planning-step's ``reasoning`` is preserved (the LLM's
//     thinking trace — useful for transparency). Fall back to
//     synthesis's reasoning if planning-step lacks it.
//   * ``intent`` / ``corrected_query`` fall back to planning-step
//     if synthesis lacks them.
//   * ``tool_calls`` is dropped from prev (planning-step) entirely
//     and only synthesis's is kept. Concatenating them would produce
//     React ``duplicate key`` warnings in ``ToolCallCard`` (same
//     ``tool_call.id`` in the same ``tool-call-list``); see trap #2.
export function mergeAdjacentAssistantTurns(
  messages: ChatMessage[],
): ChatMessage[] {
  const merged: ChatMessage[] = [];
  for (const m of messages) {
    const prev = merged[merged.length - 1];
    if (
      prev &&
      prev.role === "assistant" &&
      m.role === "assistant" &&
      prev.tool_calls &&
      prev.tool_calls.length > 0 &&
      m.tool_calls &&
      m.tool_calls.length > 0
    ) {
      // Collapse prev (planning-step) + m (synthesis) → m's fields
      // are authoritative; prev's reasoning carries forward.
      // v2.0.28.15 — DO NOT fall back to synthesis reasoning. Pre-fix
      // the ``prev.reasoning ?? m.reasoning`` rule surfaced the
      // synthesis LLM's "I'm about to write the answer" meta-
      // commentary (probe Run 1 saw "I already provided the answer in
      // my previous response with the time 2026年9月23日 19:09" — false
      // hallucination from the LLM treating the planning-step
      // AIMessage as a completed previous turn). Planning-step
      // reasoning is the LLM's tool-selection rationale which has
      // transparency value; synthesis reasoning is meta-commentary
      // that misleads. Force prev's reasoning; never fall back.
      merged[merged.length - 1] = {
        ...m,
        reasoning: prev.reasoning,
        intent: m.intent ?? prev.intent,
        corrected_query: m.corrected_query ?? prev.corrected_query,
      };
    } else {
      merged.push(m);
    }
  }
  return merged;
}