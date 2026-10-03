/**
 * HandlerContext — the read-only side of the world each per-event-type
 * handler needs to do its job.
 *
 * v2.0.26 PR-3 split: ``handleAgentEvent`` (438 LOC, 13 event types)
 * used to live as a single ``useCallback`` inside ``App.tsx`` with
 * closures over local state + refs. That shape was un-testable in
 * isolation (the only way to exercise it was to wire the whole app,
 * fire a real WS message, and observe the DOM).
 *
 * The new shape: every handler is a pure function
 * ``(ctx: HandlerContext, evt: AgentEvent) => void`` that mutates
 * state by calling ``ctx.applyThreads``. Tests pass a synchronous
 * stub ``applyThreads`` and observe the resulting state directly,
 * without React or the WS plumbing.
 *
 * Why these specific fields
 * -------------------------
 * - ``threadId``: the handler can't infer which thread it's serving
 *   from the event payload (events don't carry their target). The
 *   dispatch wrapper in ``dispatch.ts`` threads the current active
 *   thread through ``ctx``.
 * - ``threads``: read-only snapshot of the latest state, used by
 *   handlers that need to know what the current state is before
 *   applying their update (most do, since they often read the last
 *   message).
 * - ``applyThreads``: the only way handlers mutate state. It takes
 *   a functional updater identical in shape to React's
 *   ``setState(updater)`` so the handler code is unchanged from
 *   the inline version — just substitute ``setThreads`` with
 *   ``applyThreads`` and the closure semantics carry over. Tests
 *   use a stub that records + applies each updater synchronously.
 * - ``webSearchedByThreadRef`` + ``stepByThreadRef``: mutable refs
 *   that cross multiple events (web_search sets, answer_complete
 *   reads, done clears). Held outside React state because their
 *   updates don't drive renders — the corresponding ``thread.state``
 *   field is what re-renders, and the ref is read at the moment
 *   answer_complete needs it.
 * - ``tokenizeCitations``: shared helper that turns ``[1]`` / ``[2]``
 *   markers in the answer text into clickable citation chips. Used
 *   by answer_complete to populate ``ChatMessage.tokens``.
 * - ``dispatchWindowEvent`` + ``closeWebSocket``: side-effects that
 *   live outside the React tree. ``dispatchWindowEvent`` fires
 *   ``rag:sources`` and ``rag:session-updated``; ``closeWebSocket``
 *   is the QW #13 hook for explicit close on error/done. Both are
 *   stubbed in tests with a spy.
 *
 * Why no ``setThreads`` directly
 * ------------------------------
 * Passing React's setter through would force every test to mount a
 * component (because useState setters only exist inside a render).
 * The functional-update pattern (``applyThreads((prev) => ...)``)
 * is the same shape React uses internally, so the handler code can
 * stay identical and tests can stub it with a plain function. The
 * dispatch wrapper in App.tsx bridges the two: it constructs a
 * real ``applyThreads`` that calls the React ``setThreads``.
 */
import type { ReactNode } from "react";
import type { AgentEvent, ChatMessage, Source } from "../../chat-utils";

export interface ThreadState {
  messages: ChatMessage[];
  isStreaming: boolean;
  historyLoaded: boolean;
  webSearching: boolean;
  currentStep: number;
  /** v2.0.29.9 (Phase 8) — per-thread verbatim extraction mode.
   * Mirrors App.tsx's canonical ThreadState so the two are shape-
   * compatible (TS records must match exactly across the union).
   *   "auto" — run hybrid detect, may or may not trigger
   *   "on"   — force extractive (user explicitly enabled)
   *   "off"  — force normal mode (user explicitly disabled)
   * Persisted via ChatRequest.high_precision (tristate Literal). */
  highPrecision: "auto" | "on" | "off";
}

/** Functional state updater; identical shape to ``setState``. */
export type ThreadsUpdater = (
  prev: Record<string, ThreadState>,
) => Record<string, ThreadState>;

export interface HandlerContext {
  threadId: string;
  threads: Record<string, ThreadState>;
  applyThreads: (updater: ThreadsUpdater) => void;
  webSearchedByThreadRef: { current: Map<string, boolean> };
  stepByThreadRef: { current: Map<string, number> };
  tokenizeCitations: (text: string) => ReactNode[];
  dispatchWindowEvent: (type: string, detail?: unknown) => void;
  closeWebSocket: (threadId: string, reason?: string) => void;
}

/**
 * Empty/default thread state. Matches the one in App.tsx (frozen
 * object) so handlers that read ``threads[tid] ?? EMPTY_THREAD_STATE``
 * behave identically outside React.
 */
export const EMPTY_THREAD_STATE: ThreadState = Object.freeze({
  messages: [],
  isStreaming: false,
  historyLoaded: false,
  webSearching: false,
  currentStep: 0,
  highPrecision: "auto",
}) as ThreadState;

/**
 * Patch a single thread's state. Same shape as the inline helper
 * in App.tsx — moved here so handlers can use it without importing
 * the whole App module (which would pull in React + i18n + WS, all
 * unnecessary for the handler's job).
 */
export function patchThread(
  threads: Record<string, ThreadState>,
  tid: string,
  patch: Partial<ThreadState>,
): Record<string, ThreadState> {
  const current = threads[tid] ?? EMPTY_THREAD_STATE;
  return { ...threads, [tid]: { ...current, ...patch } };
}

/**
 * Patch a single thread's messages array via a transform function.
 * Mirrors App.tsx's ``patchThreadMessages`` helper for the same
 * reason as ``patchThread`` above (avoid pulling in App.tsx).
 */
export function patchThreadMessages(
  threads: Record<string, ThreadState>,
  tid: string,
  fn: (messages: ChatMessage[]) => ChatMessage[],
): Record<string, ThreadState> {
  const current = threads[tid] ?? EMPTY_THREAD_STATE;
  return {
    ...threads,
    [tid]: { ...current, messages: fn(current.messages) },
  };
}

/**
 * Default ``source_kind`` fallback for legacy persisted messages.
 * Same default App.tsx used inline — moved here so handlers don't
 * import the App-level constant.
 */
export function withDefaultSourceKind(sources: Source[] | undefined): Source[] {
  return (sources ?? []).map((s) => ({
    ...s,
    source_kind: s.source_kind ?? "local",
  }));
}

/**
 * Type guard: an AgentEvent's payload type. Used by the dispatch
 * switch for narrowing without ``as`` casts.
 */
export type EventByType<K extends AgentEvent["type"]> = Extract<
  AgentEvent,
  { type: K }
>;