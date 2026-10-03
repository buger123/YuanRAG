/**
 * Shared test infrastructure for the eventHandlers/* unit tests.
 *
 * Each handler test needs a fake ``HandlerContext`` with:
 *
 * 1. **State holder**: a plain ``Record<threadId, ThreadState>``
 *    the tests can read after the handler runs.
 * 2. **applyThreads**: a simple imperative ``(updater) => void``
 *    that runs the functional updater against the in-memory map.
 *    Mirrors React's ``setState(updater)`` semantics.
 * 3. **refs**: the per-thread ref Maps wrapped as
 *    ``{ current: Map<...> }`` so handlers can read/write.
 * 4. **tokenizeCitations**: a trivial pass-through that splits the
 *    text on ``[N]`` markers and returns plain strings. Real
 *    ChatPane replaces them with chips, but handlers don't need
 *    that fidelity — they just need the helper to be callable.
 * 5. **dispatchWindowEvent / closeWebSocket**: jest-style spies
 *    so the tests can assert which window events fired and which
 *    threads got closed (and with what reason).
 *
 * Why not @testing-library/react + render?
 *
 * Handler functions are pure state mutators — they don't render
 * DOM. Rendering App.tsx for each test would couple the tests to
 * the entire app shell (Sidebar / ChatPane / SettingsDialog
 * dependencies) and make the test surface brittle. Driving the
 * handlers with a fake ctx keeps each test scoped to the
 * handler's actual contract: "given evt X, threads[threadId]
 * mutates like Y".
 */
import type { ReactNode } from "react";
import type { AgentEvent } from "../../../chat-utils";
import type {
  HandlerContext,
  ThreadsUpdater,
  ThreadState,
} from "../context";

/**
 * Spy record — accumulating assertions about side effects the
 * handlers caused. Tests inspect this after running the handler.
 */
export interface HandlerSpy {
  windowEvents: Array<{ type: string; detail: unknown }>;
  closedThreads: Array<{ threadId: string; reason: string | undefined }>;
}

/**
 * Functional ``applyThreads`` backing store. Implements
 * React's ``setState(updater)`` shape:
 *
 *   applyThreads((prev) => patchThread(prev, tid, { isStreaming: true }))
 *
 * Reads from ``state`` so handlers see the latest after each call
 * (mirrors the React-state-as-source-of-truth pattern in App.tsx).
 */
export type ApplyThreadsFn = (updater: ThreadsUpdater) => void;

export interface TestCtx extends HandlerContext {
  state: Record<string, ThreadState>;
  apply: ApplyThreadsFn;
  spy: HandlerSpy;
}

/**
 * Trivial tokenizeCitations: splits on ``[N]`` markers and
 * returns an array of plain strings (no chip wrapping). Tests
 * assert the COUNT or the marker alignment, not the chip DOM.
 */
export function stubTokenizeCitations(text: string): ReactNode[] {
  const parts: string[] = [];
  let rest = text;
  // Match literal [1], [12], etc.
  const re = /\[(\d+)\]/g;
  let lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(rest)) !== null) {
    if (m.index > lastIndex) parts.push(rest.slice(lastIndex, m.index));
    parts.push(`[${m[1]}]`);
    lastIndex = m.index + m[0].length;
  }
  if (lastIndex < rest.length) parts.push(rest.slice(lastIndex));
  return parts;
}

/**
 * Build a fresh test context for one handler invocation.
 *
 *  - ``threadId``: the thread the handler should operate on.
 *  - ``initialState``: optional initial ThreadState for the thread
 *    (defaults to ``EMPTY_THREAD_STATE``-equivalent — messages=[],
 *    isStreaming=false, etc.).
 */
export function makeCtx(
  threadId = "t1",
  initialState: Partial<ThreadState> = {},
): TestCtx {
  const state: Record<string, ThreadState> = {
    [threadId]: {
      messages: [],
      isStreaming: false,
      historyLoaded: false,
      webSearching: false,
      currentStep: 0,
      highPrecision: "auto",
      ...initialState,
    },
  };

  const spy: HandlerSpy = {
    windowEvents: [],
    closedThreads: [],
  };

  // applyThreads uses the functional pattern — the closure
  // captures `state` by reference, so each call sees the latest
  // patches from previous calls (just like React's setState).
  const apply: ApplyThreadsFn = (updater) => {
    state[threadId] = Object.keys(state[threadId]).length
      ? state[threadId]
      : state[threadId];
    const next = updater(state);
    // Replace the whole map (handler returns the new state map).
    for (const k of Object.keys(state)) delete state[k];
    Object.assign(state, next);
  };

  const ctx: TestCtx = {
    threadId,
    threads: state,
    state,
    apply,
    applyThreads: apply,
    webSearchedByThreadRef: { current: new Map() },
    stepByThreadRef: { current: new Map() },
    tokenizeCitations: stubTokenizeCitations,
    dispatchWindowEvent: (type, detail) => {
      spy.windowEvents.push({ type, detail });
    },
    closeWebSocket: (tid, reason) => {
      spy.closedThreads.push({ threadId: tid, reason });
    },
    spy,
  };

  return ctx;
}

/**
 * Convenience: assert the thread has a specific number of
 * messages.
 */
export function messages(ctx: TestCtx): unknown[] {
  return ctx.state[ctx.threadId].messages;
}

/**
 * Type helper for tests that build an AgentEvent literal — saves
 * the type-narrowing boilerplate.
 */
export function makeEvent<K extends AgentEvent["type"]>(
  ...args: Extract<AgentEvent, { type: K }> extends infer E
    ? E extends { type: K }
      ? [K, Omit<E, "type">]
      : never
    : never
): Extract<AgentEvent, { type: K }> {
  const [type, rest] = args as [K, Omit<Extract<AgentEvent, { type: K }>, "type">];
  return { type, ...rest } as Extract<AgentEvent, { type: K }>;
}