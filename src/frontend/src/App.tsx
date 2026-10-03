import { useCallback, useEffect, useRef, useState } from "react";
import { useLocale } from "./i18n";
import { Sidebar } from "./components/Sidebar";
import { ChatPane } from "./components/ChatPane";
import { SettingsDialog } from "./components/SettingsDialog";
import { ModelDownloadProgress } from "./components/ModelDownloadProgress";
import { useToast } from "./components/Toast";
import {
  getSessionMessages,
  type MessageRecord,
  type Settings,
} from "./api/client";
import { useModelsReady } from "./hooks/useModelsReady";
import { useThreadSocket } from "./hooks/useThreadSocket";
import {
  type AgentEvent,
  type ChatMessage,
  tokenizeCitations,
  restoreMessages,
  mergeAdjacentAssistantTurns,
} from "./chat-utils";
import { dispatchAgentEvent } from "./chat/eventHandlers/dispatch";
import type { HandlerContext } from "./chat/eventHandlers/context";
import { apiError } from "./i18n/errors";

// Stable empty-array constant for ChatPane's `messages` prop. Without
// it, the `messagesByThread[threadId] || []` fallback allocates a new
// `[]` on every App render (i.e. every streamed token), which defeats
// `React.memo` on ChatPane by giving the prop a fresh identity each
// time. Hoisting to module scope makes the reference stable forever.
const EMPTY_MESSAGES: ChatMessage[] = [];

// v1.1.16 — map RFC-6455 close codes to a user-facing banner
// message. The banner IIFE in the JSX (``connectionError.message``)
// renders this verbatim, so the wording lives in one place. Codes
// that don't represent an actionable failure for the user (1000
// Normal Closure, 1005 No Status Received) are filtered out before
// we even reach this function — see ``useThreadSocket``'s onclose
// code-filter (it gates the ``onAbnormalClose`` callback).
//
// v2.0.6 P2-4 — wraps ``useLocale().t`` so the banner re-renders
// when the user flips the language. The translator function lives
// outside React's render path so the lookup table doesn't get
// recreated on every keystroke.
//
// Reference: https://datatracker.ietf.org/doc/html/rfc6455#section-7.4.1
function makeMessageForCloseCode(
  t: (key:
    | "connection.close1001"
    | "connection.close1006"
    | "connection.close1008"
    | "connection.close1009"
    | "connection.close1011"
    | "connection.close1014"
    | "connection.closeGeneric") => string,
): (code: number) => string {
  return (code) => {
    switch (code) {
      case 1001:
        return t("connection.close1001");
      case 1006:
        return t("connection.close1006");
      case 1008:
        return t("connection.close1008");
      case 1009:
        return t("connection.close1009");
      case 1011:
        return t("connection.close1011");
      case 1014:
        return t("connection.close1014");
      default:
        return t("connection.closeGeneric");
    }
  };
}

// Per-thread state schema (Phase E refactor). One record per thread
// instead of four parallel ``Record<threadId, X>`` maps — adding a new
// per-thread flag is now a single field on this type rather than a new
// useState + four lines of prop-drilling updates.
//
// The default ``Object.freeze``d value is reused for every thread that
// hasn't been touched yet, so reads (``threads[tid] ?? EMPTY_THREAD_STATE``)
// give a STABLE reference until the thread becomes active. This is the
// same trick the pre-refactor code used for ``EMPTY_MESSAGES`` — without
// it, ChatPane would re-render on every App render because its
// messages prop would be a fresh ``[]`` each time.
export interface ThreadState {
  messages: ChatMessage[];
  isStreaming: boolean;
  historyLoaded: boolean;
  webSearching: boolean;
  /** v2.0 — current ReAct step (1-based). 0 = idle / no step
   *  in flight. Bumped by ``step_start`` events; reset to 0 when
   *  ``done`` fires. The frontend renders "Step N" next to the
   *  thinking pill while this is >0. */
  currentStep: number;
  /** v2.0.29.9 (Phase 8) — per-thread verbatim extraction mode.
   *  - ``"auto"`` (default) → backend runs hybrid detect (regex +
   *    cheap-LLM); extractive mode may or may not trigger.
   *  - ``"on"``  → force extractive (verbatim quote, no rewriting).
   *  - ``"off"`` → force normal synthesis EVEN if backend detect
   *    fires (per user decision 2026-09-28, user OFF always wins).
   *  Set via the segmented control in ``ChatInput``; forwarded to
   *  backend via the WS payload / ChatRequest. Persists per-thread
   *  in this in-memory map (NOT in the DB — a reload reverts to
   *  "auto"; per-session is sufficient for Phase 8 PR-1). */
  highPrecision: "auto" | "on" | "off";
}

const EMPTY_THREAD_STATE: ThreadState = Object.freeze({
  messages: EMPTY_MESSAGES,
  isStreaming: false,
  historyLoaded: false,
  webSearching: false,
  currentStep: 0,
  highPrecision: "auto",
}) as ThreadState;

type ThreadPatch = Partial<ThreadState>;

/** Return a new threads map with ``tid``'s state merged with ``patch``.
 *  Equivalent to: ``{ ...threads, [tid]: { ...current, ...patch } }``,
 *  but readable at call sites and impossible to typo the field name. */
function patchThread(
  threads: Record<string, ThreadState>,
  tid: string,
  patch: ThreadPatch,
): Record<string, ThreadState> {
  const current = threads[tid] ?? EMPTY_THREAD_STATE;
  return { ...threads, [tid]: { ...current, ...patch } };
}

/** Like :func:`patchThread` but also runs ``fn`` over the messages array
 *  so the caller can write ``prev -> next`` instead of manually
 *  building a new array + spreading siblings. Avoids the noisy
 *  ``setMessagesByThread((all) => { const t = all[tid] || []; ... })``
 *  boilerplate that was scattered through ``handleAgentEvent``. */
function patchThreadMessages(
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

export function App() {
  // v2.0.6 P2-4 — pull the active locale and lookup function so the
  // connection-error banner and other module-scoped text can re-
  // render on language flip.
  const { t } = useLocale();
  // v2.0.26.3 (PR-6) — toast dispatcher for user-visible failure
  // notifications (replaces PR-2 inline banner + raw console.error
  // for active user-triggered actions). Background refetches
  // (listSessions / listDocuments) stay on console.error so we
  // don't spam toasts on transient network blips.
  const { showToast } = useToast();
  // UUIDs that were minted LOCALLY in this session — either by the
  // initial 'navigate' branch below, by clicking "+ 新对话", or by
  // deleting the currently-active session (which also rolls a fresh
  // UUID for the new "empty" chat). These threads do not (yet) exist
  // on the server, so ``loadHistory`` must skip the GET
  // ``/sessions/{tid}/messages`` fetch and pre-mark them as
  // ``historyLoaded: true`` — otherwise ChatPane briefly shows the
  // "正在加载对话历史…" spinner while the request round-trips only to
  // return 404, then snaps back to the empty state. Consumed-on-read
  // (the ref ``delete``s the entry once ``loadHistory`` has acted on
  // it), so the same UUID is never accidentally marked twice if the
  // user re-opens the same freshly-minted thread via some other path.
  //
  // Declared at the top of the component (BEFORE the ``useState``
  // initializer that may register the cold-launch UUID) so the
  // initializer's closure can capture it without hitting the TDZ.
  const freshUuidsRef = useRef<Set<string>>(new Set());

  // ``threadId`` policy — distinguish cold launches from in-tab refresh:
  //   * typed URL / new tab / browser reopen  → brand-new UUID
  //     (the user wants every fresh open to start as a new chat)
  //   * F5 / refresh / location.reload()     → restore the last
  //     "real" thread from localStorage so refresh doesn't lose the
  //     conversation in progress
  //   * browser back/forward                 → same as refresh
  //
  // ``PerformanceNavigationTiming.type`` distinguishes the two cases:
  //   'navigate'    = fresh open → do NOT restore
  //   'reload'      = F5 refresh → restore
  //   'back_forward'= back/forward → restore
  //
  // A thread becomes "real" (and gets committed to storage) on the
  // first meaningful action (document upload, message sent, or sidebar
  // session clicked). Pure UI navigations like "+ New chat" or
  // deleting the active session wipe the commit so a refresh after
  // those actions starts a fresh chat rather than restoring the
  // just-discarded thread — see ``onNewChat`` for the wipe.
  const [threadId, setThreadId] = useState<string>(() => {
    const navType = (
      performance.getEntriesByType("navigation")[0] as
        | PerformanceNavigationTiming
        | undefined
    )?.type;
    if (navType === "reload" || navType === "back_forward") {
      const committed = localStorage.getItem("rag.active_thread");
      if (committed) return committed;
    }
    const fresh = crypto.randomUUID();
    // Brand-new cold launch ('navigate' branch). Register so
    // ``loadHistory`` skips the wasted GET that would 404 anyway.
    freshUuidsRef.current.add(fresh);
    return fresh;
  });
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState<Settings | null>(null);
  // v2.0.26.2 (PR-5) — mobile sidebar drawer state. Hoisted here (not
  // inside Sidebar) because ChatPane also needs to render the
  // hamburger button that opens it. Auto-closes when the viewport
  // crosses 768 px back to desktop so a desktop-style horizontal
  // resize never leaves a drawer stuck open. See styles.css
  // @media (max-width: 768px) for the off-canvas transition.
  const [drawerOpen, setDrawerOpen] = useState(false);
  // Model readiness + API key state. Replaces 3 useState + 2 effects
  // (one-shot checkModels + 2s poll) with a single hook that polls
  // until terminal (ready / missing / failed) instead of polling
  // forever. See useModelsReady.ts for terminal-state rationale.
  // ``modelsStatus`` is intentionally consumed by future PR-2
  // (terminal-state banner); for PR-1 we just need the boolean flag.
  const {
    ready: modelsReady,
    checked: modelsChecked,
    status: _modelsStatus,
    apiKeyConfigured,
    embedding: modelsEmbedding,
    reranker: modelsReranker,
    refresh: refreshModels,
  } = useModelsReady();

  // QW #10: surfaced WS error state. Previously ws.onerror /
  // ws.onclose only logged to console and dropped the map entry —
  // users saw a frozen stream with no indication that the connection
  // had died. We now set a structured error that the banner reads;
  // the banner auto-dismisses when the user reopens the thread
  // (handled in the listener below).
  const [connectionError, setConnectionError] = useState<{
    message: string;
  } | null>(null);

  // Commit the current ``threadId`` to localStorage whenever the
  // session becomes "real". ChatPane fires ``rag:session-updated`` on
  // every WS ``done`` (the LLM turn is the canonical "real" signal);
  // Sidebar fires it after a successful document upload or after the
  // user clicks an existing session in the sidebar list. The handler
  // re-binds on every ``threadId`` change so it always writes the
  // current value, never a stale closure.
  useEffect(() => {
    function commitActiveThread() {
      localStorage.setItem("rag.active_thread", threadId);
    }
    window.addEventListener("rag:session-updated", commitActiveThread);
    return () =>
      window.removeEventListener("rag:session-updated", commitActiveThread);
  }, [threadId]);

  // QW #10: turn the WS lifecycle events that ``useThreadSocket``
  // dispatches (``onAbnormalClose`` / ``onError``) into a visible
  // banner. The events carry a small payload (kind + message) so the
  // banner can render something the user actually understands —
  // "连接已断开,正在后台重试..." beats a silent stuck spinner.
  //
  // We clear the banner automatically when the active thread
  // changes: a fresh thread can't have inherited the error, and a
  // user-driven switch is the cleanest way to say "I see it, move on".
  useEffect(() => {
    function onWsEvent(e: Event) {
      const ce = e as CustomEvent<{ kind: string; message: string; code?: number }>;
      const detail = ce.detail || { kind: "closed" };
      // v2.0.6 P2-4 — translate at receive time, not at dispatch
      // time. The dispatch (in useThreadSocket's onAbnormalClose)
      // ships the raw close code; we map it through the i18n table
      // HERE so flipping the language re-renders the banner without
      // a new WS event.
      const fallback = makeMessageForCloseCode(t);
      let message: string;
      if (typeof detail.code === "number" && detail.code !== 0) {
        message = fallback(detail.code);
      } else if (detail.message) {
        message = detail.message;
      } else {
        message = fallback(0);
      }
      setConnectionError({ message });
    }
    window.addEventListener("rag:ws-error", onWsEvent as EventListener);
    window.addEventListener("rag:ws-closed", onWsEvent as EventListener);
    return () => {
      window.removeEventListener("rag:ws-error", onWsEvent as EventListener);
      window.removeEventListener("rag:ws-closed", onWsEvent as EventListener);
    };
  }, [t]);
  useEffect(() => {
    // Clear when the user switches to a different thread; the error
    // belonged to the prior socket, not the new one.
    setConnectionError(null);
  }, [threadId]);

  // "+ 新对话" handler. The Sidebar used to do this inline
  // (``onSelectThread(crypto.randomUUID())`` + ``onClearActiveThread``),
  // but that routed the new UUID through ``loadHistory`` which fires a
  // GET that always 404s, leaving ChatPane showing the
  // "正在加载对话历史…" spinner for the round-trip before snapping back
  // to the empty state. Centralizing here lets us register the fresh
  // UUID in ``freshUuidsRef`` so ``loadHistory`` skips the fetch AND
  // pre-marks the new thread as ``historyLoaded: true`` — the empty
  // state renders on the very first frame.
  const onNewChat = useCallback(() => {
    const fresh = crypto.randomUUID();
    freshUuidsRef.current.add(fresh);
    setThreadId(fresh);
    localStorage.removeItem("rag.active_thread");
  }, []);

  // ─── Per-thread chat state (parallel answering) ────────────────────
  //
  // Hoisted from ChatPane to App so multiple threads can have in-flight
  // streams running simultaneously. Each thread's WebSocket is opened
  // lazily (on first send) and torn down when its run completes
  // (``done`` event). Switching the active ``threadId`` does NOT close
  // other threads' sockets — they keep streaming in the background and
  // the user can switch back to see their final state. This is what
  // makes "Q1 in thread A, switch to thread B, ask Q2, both answers
  // stream in parallel" work.
  //
  // Backend-side, every agent run completes and writes its checkpoint
  // regardless of whether its client is still connected (see
  // ``src/api/websocket.py``'s ``_drain_generator``). So even if a WS
  // dies from a network blip rather than a clean thread switch, the
  // answer is still preserved server-side and the user sees it the
  // next time they open that thread.
  //
  // Phase E consolidation: previously four parallel
  // ``Record<threadId, X>`` maps (``messagesByThread`` /
  // ``streamingByThread`` / ``historyLoadedByThread`` /
  // ``webSearchingByThread``). Each per-thread flag was its own useState
  // and had to be kept in sync at every update site — adding a new flag
  // meant four new lines of boilerplate and a real risk of forgetting
  // one. Now a single ``Record<threadId, ThreadState>`` map; mutations
  // go through ``patchThread`` / ``patchThreadMessages`` helpers (defined
  // above) so the field set is type-checked and impossible to typo.
  const [threads, setThreads] = useState<Record<string, ThreadState>>({});

  // v1.1.3 — per-thread "did this turn actually hit the web search?"
  // tracker. The backend emits a ``web_search`` event from the
  // ``search_web`` node, but only when that node actually runs. For
  // every other turn (greetings / direct answers / successful
  // retrieval that never had to fall back) the event is absent — yet
  // ``answer_complete`` fires regardless, and the message used to
  // hardcode ``webSearched: true`` here. The result was a spurious
  // "🔎 本回答参考了联网搜索结果" footer on every assistant message,
  // including a "你好" / "你是谁" greeting that obviously never went
  // anywhere near the web.
  //
  // We track this per-thread (not globally) so parallel streams on
  // different threads can't poison each other's footer. Cleared when
  // the thread's ``done`` event arrives so the next turn starts
  // fresh.
  const webSearchedByThreadRef = useRef<Map<string, boolean>>(new Map());

  // Per-thread WS lifecycle — see ``useThreadSocket`` for the full
  // contract (lazy open / re-use open socket / 1000/1005 code
  // filter / QW #13 reconnect with backoff / per-socket de-dup of
  // ``onerror`` vs ``onclose`` banner events). The hook owns its
  // own internal ``wsByThreadRef`` (not in App's scope) so we don't
  // double-track. Hook call is placed AFTER ``handleAgentEvent`` so
  // the ``onMessage`` callback can close over it without a
  // forward-reference hack.

  // Route an agent event into the right thread's state slice. The
  // 438-LOC ``if (evt.type === ...)`` cascade that used to live here
  // was split into per-event-type modules under
  // ``src/chat/eventHandlers/`` (Item 6 PR-3, v2.0.26) — see
  // ``dispatchAgentEvent`` for the dispatch table and each
  // ``*Handler.ts`` for the per-event logic. Handlers all use
  // functional setState so they always see the latest thread's
  // buffer, even when several WS messages are interleaved (e.g. a
  // token arrives for thread A while thread B is mid-token).
  const handleAgentEvent = useCallback(
    (tid: string, evt: AgentEvent) => {
      const ctx: HandlerContext = {
        threadId: tid,
        threads,
        applyThreads: setThreads,
        webSearchedByThreadRef,
        stepByThreadRef,
        tokenizeCitations,
        dispatchWindowEvent: (type, detail) =>
          window.dispatchEvent(new CustomEvent(type, { detail })),
        closeWebSocket: (id, reason) => threadSocket.close(id, reason),
      };
      dispatchAgentEvent(ctx, evt);
    },
    // All bindings above are stable identities:
    //  - setThreads / threads come from useState (stable)
    //  - the two refs hold their {current: Map<...>} wrappers (stable)
    //  - tokenizeCitations is module-scoped (stable)
    //  - threadSocket is the stable return of useThreadSocket below
    // The original handleAgentEvent used [] deps with the same
    // lazy-ref-read pattern; the split handlers all read state via
    // functional setState, so adding these to deps would only create
    // a rebuild loop without changing semantics.
    [],
  );

  // Per-thread WebSocket lifecycle. The hook owns its own internal
  // ``wsByThreadRef`` ref and the per-socket de-dup of
  // ``onerror`` / ``onclose`` banner events (the latter was inlined
  // inside the previous ``getOrCreateWS`` — moved here so the
  // lifecycle logic stays single-sourced). ``onMessage`` routes
  // every parsed agent event into ``handleAgentEvent`` above;
  // ``shouldReconnect`` only authorizes reconnect for threads that
  // are still mid-stream so an idle thread that lost its socket
  // doesn't keep trying to dial home (QW #13).
  const threadSocket = useThreadSocket({
    // Hook is payload-agnostic (it parses JSON and hands back
    // ``unknown``). App owns the AgentEvent shape, so cast at the
    // boundary — same place the inline version did.
    onMessage: useCallback(
      (tid: string, evt: unknown) =>
        handleAgentEvent(tid, evt as AgentEvent),
      // handleAgentEvent has [] deps and reads state via functional
      // setState, so this wrapper can stay stable too.
      [],
    ),
    shouldReconnect: useCallback(
      (tid: string) => threads[tid]?.isStreaming === true,
      // threads is a stable useState identity; the ref-read stays
      // current because the closure captures the threads map by
      // reference and reads it at call time (WS events fire async).
      [threads],
    ),
    onAbnormalClose: useCallback(
      (_tid: string, code: number) => {
        window.dispatchEvent(
          new CustomEvent("rag:ws-closed", {
            detail: {
              kind: "closed",
              code,
              // v2.0.6 P2-4 — translate at receive time, not at
              // dispatch time. The receiver (onWsEvent above)
              // re-renders on locale change and re-translates the
              // raw code; baking the Chinese string in here would
              // freeze the banner in the locale the user was in
              // when the socket opened.
              message: "",
            },
          }),
        );
      },
      [],
    ),
    onError: useCallback(
      (_tid: string) => {
        window.dispatchEvent(
          new CustomEvent("rag:ws-error", {
            detail: {
              kind: "error",
              code: 1006,
              // v2.0.8 i18n-2: resolve the fallback banner through
              // ``t`` so it re-renders on locale flip. The
              // translator is captured from the closure that built
              // the WS (App's render scope).
              message: t("app.wsErrorFallback"),
            },
          }),
        );
      },
      [t],
    ),
    // P0-F2: the hook already designed these callbacks in v2.0.x but
    // App never wired them. Wire them now so the user sees the
    // CURRENT reconnect state at all times instead of a stale error.
    onReconnectAttempt: useCallback(
      (_tid: string, attempt: number) => {
        setConnectionError({
          message: t("connection.reconnecting", { attempt }),
        });
      },
      [t],
    ),
    onReconnectExhausted: useCallback(
      (_tid: string) => {
        setConnectionError({ message: t("connection.failed") });
      },
      [t],
    ),
    onReconnected: useCallback(
      (_tid: string) => {
        setConnectionError(null);
      },
      [],
    ),
  });

  // v2.0 — current ReAct step counter per thread. Drives the
  // "Step N" pill in ChatPane (bumped on every ``step_start`` event).
  // Held in a ref (not state) because it's purely visual; ChatPane
  // reads it through the existing per-thread ``ThreadState`` mirror
  // below.
  const stepByThreadRef = useRef<Map<string, number>>(new Map());
  // P0-F2: track the last user message sent per thread so the
  // connection-error banner's "重试上一条消息" button can resend it.
  // Ref instead of state because the value is only consumed when
  // the user clicks retry (event handler) — no need to re-render.
  const lastUserMessageByThreadRef = useRef<Map<string, string>>(
    new Map(),
  );
  // P0-extension (PR-2): per-thread AbortController + monotonic seq
  // counter so the previous thread's history fetch is cancelled (and
  // its result discarded if the network already finished) the moment
  // the user switches threads. Without this, switching A → B → A in
  // rapid succession could leave thread A's pane showing thread B's
  // history — the second GET for A races the first GET for A (and
  // wins), but the first GET's slower response lands last and
  // overwrites with B's history. Two guards are used together:
  //   - ``seq``: monotonic per-thread counter, incremented on every
  //     call; post-await only mutate state if seq is still current.
  //   - ``controller``: abort the in-flight ``fetch`` so the network
  //     request is cancelled, not just ignored when it lands.
  // Both guards live in refs (not state) because they're keyed by
  // thread id and only read inside the async callback — no re-render
  // hook needed.
  const loadHistoryControllersRef = useRef<Map<string, AbortController>>(
    new Map(),
  );
  const loadHistorySeqRef = useRef<Map<string, number>>(new Map());

  // Public: send a user message into a thread. Called by ChatPane's
  // submit handler. Optimistically appends the user message to the
  // thread's buffer, opens (or reuses) the thread's WebSocket, and
  // sends the payload. If the WS is still in CONNECTING (cold open),
  // the send is queued for the open event so the first message on a
  // freshly-mounted thread isn't dropped.
  const sendToThread = useCallback(
    (tid: string, text: string, options?: { highPrecision?: "auto" | "on" | "off" }) => {
      // P0-F2: remember the last user message per thread for the
      // connection-error retry button. Trimmed so an empty/whitespace
      // retry doesn't get queued; ``sendToThread`` itself still
      // guards on text length at the ChatPane level.
      const trimmed = text.trim();
      if (trimmed) {
        lastUserMessageByThreadRef.current.set(tid, trimmed);
      }
      setThreads((all) =>
        patchThreadMessages(all, tid, (msgs) => [
          ...msgs,
          {
            id: crypto.randomUUID(),
            role: "user",
            content: text,
            // v2.0 — stamp the user message locally with the
            // ISO 8601 timestamp the runner also stamps server-side
            // on the HumanMessage (defensive — the optimistic
            // timestamp may drift from the server's by ms; the
            // history replay uses the server's value so the
            // divergence is invisible to the user).
            created_at: new Date().toISOString(),
          },
        ]),
      );
      setThreads((all) => patchThread(all, tid, { isStreaming: true }));
      const ws = threadSocket.getOrCreate(tid);
      // v2.0.29.9 (Phase 8) — forward per-thread verbatim toggle to
      // the backend via the WS payload. ``options.highPrecision``
      // overrides the ThreadState value (used by retry from the
      // connection-error toast which doesn't have the toggle's
      // current value at hand). Default = read from ThreadState so
      // the segmented control in ChatInput doesn't have to thread
      // the value through every callback.
      const hp =
        options?.highPrecision ??
        threads[tid]?.highPrecision ??
        "auto";
      const payload = JSON.stringify({ message: text, high_precision: hp });
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(payload);
      } else if (ws.readyState === WebSocket.CONNECTING) {
        ws.addEventListener(
          "open",
          () => {
            ws.send(payload);
          },
          { once: true },
        );
      } else {
        // CLOSING / CLOSED — replace and queue. Shouldn't happen in
        // normal flow (we close on ``done`` and replace on next send)
        // but defensive code costs nothing. ``threadSocket.close``
        // both closes any half-closed socket and drops the map
        // entry; the next ``getOrCreate`` then opens a fresh one.
        threadSocket.close(tid);
        const fresh = threadSocket.getOrCreate(tid);
        fresh.addEventListener(
          "open",
          () => {
            fresh.send(payload);
          },
          { once: true },
        );
      }
    },
    [threadSocket],
  );

  // Public: load (or re-load) a thread's persisted history. Called by
  // ChatPane whenever the active ``threadId`` changes. Replaces any
  // in-memory buffer for the thread with the server's view — important
  // because the in-memory buffer may be empty (cold switch to an old
  // thread) or stale (after a reload). After the load completes,
  // ``threads[tid].historyLoaded`` is true and ChatPane re-enables the
  // input.
  //
  // Skip-if-populated: if we already have an in-memory buffer for this
  // thread (user just bounced A → B → A), the buffer IS the source of
  // truth — we've been mutating it via ``handleAgentEvent`` the whole
  // time, the server hasn't seen anything newer. Refetching would just
  // burn a network round-trip + two renders (loading→ready flicker).
  const loadHistory = useCallback(
    async (tid: string) => {
      // Brand-new UUID minted by THIS client (cold 'navigate' launch,
      // "+ 新对话", or deleting the active session). The thread has
      // never existed on the server — skip the GET that would 404
      // anyway, and pre-mark ``historyLoaded: true`` so ChatPane
      // renders the empty state on the first frame instead of
      // flashing the "正在加载对话历史…" spinner for the round-trip.
      if (freshUuidsRef.current.has(tid)) {
        freshUuidsRef.current.delete(tid);
        setThreads((all) => {
          const existing = all[tid];
          if (existing) {
            // Already has an in-memory buffer (e.g. typed before the
            // threadId flip finished registering). Just flip the flag.
            if (existing.historyLoaded) return all;
            return patchThread(all, tid, { historyLoaded: true });
          }
          return patchThread(all, tid, {
            messages: EMPTY_MESSAGES,
            historyLoaded: true,
          });
        });
        return;
      }

      // Read fresh state inside the callback (not from a closure capture)
      // so concurrent switches between threads see the right baseline.
      let alreadyInMemory = false;
      setThreads((all) => {
        alreadyInMemory = all[tid] !== undefined;
        if (alreadyInMemory) {
          // Flip historyLoaded only if it wasn't already true.
          if (all[tid]?.historyLoaded) return all;
          return patchThread(all, tid, { historyLoaded: true });
        }
        return patchThread(all, tid, { historyLoaded: false });
      });
      if (alreadyInMemory) return;

      // v2.0.25.1 P0-extension — claim the seq + abort any previous
      // fetch for this thread BEFORE awaiting. By the time the
      // getSessionMessages Promise settles, we know whether we still
      // own the slot (seq still current) and whether the network was
      // cancelled (signal.aborted). Both must be false for the
      // response to be applied. Note: a freshly-minted UUID won't
      // hit this branch because of the ``freshUuidsRef`` early-return
      // above — only existing sessions that need the GET exercise
      // the seq guard.
      const mySeq = (loadHistorySeqRef.current.get(tid) ?? 0) + 1;
      loadHistorySeqRef.current.set(tid, mySeq);
      const previousController = loadHistoryControllersRef.current.get(tid);
      if (previousController) {
        // Don't pass a reason — the catch block distinguishes the
        // AbortError from real failures by checking ``signal.aborted``
        // after the fact.
        previousController.abort();
      }
      const controller = new AbortController();
      loadHistoryControllersRef.current.set(tid, controller);

      try {
        const records: MessageRecord[] = await getSessionMessages(
          tid,
          controller.signal,
        );
        // Race loser / cancelled: a newer call for this thread (or
        // an abort) has superseded us. Drop the response silently —
        // the winning call will set messages + historyLoaded.
        if (
          loadHistorySeqRef.current.get(tid) !== mySeq ||
          controller.signal.aborted
        ) {
          return;
        }
        // v2.0.28.14 - restoreMessages wraps the wire
        // record -> ChatMessage mapping as a pure transform so it
        // can be unit-tested without mocking getSessionMessages.
        // mergeAdjacentAssistantTurns then collapses the 2
        // AIMessages that the v2.0.28.10 contract persists per
        // ReAct turn (planning-step + synthesis) into a single
        // bubble - fixes the double-bubble reload UX bug where
        // the user saw two get_current_time cards.
        const restored: ChatMessage[] = mergeAdjacentAssistantTurns(
          restoreMessages(records),
        );
        setThreads((all) =>
          patchThread(all, tid, {
            messages: restored,
            historyLoaded: true,
          }),
        );
      } catch (e) {
        // v2.0.25.1 — an abort is an EXPECTED outcome when the user
        // switches threads mid-load: the previous call's fetch was
        // cancelled by our own controller. Log it at ``debug`` (not
        // ``error``) and skip the empty-pane fallback so we don't
        // overwrite the new thread's already-loading state. A real
        // network failure still logs at error + falls through to an
        // empty pane so the user sees an empty chat (matching today's
        // behavior).
        if (
          controller.signal.aborted ||
          (e instanceof Error && e.name === "AbortError")
        ) {
          // eslint-disable-next-line no-console
          console.debug("loadHistory aborted (newer call won):", tid);
          return;
        }
        console.error("load session history failed", e);
        // v2.0.26.3 (PR-6) — toast instead of silent console.error
        // so the user sees the failure (the user just clicked on a
        // session in the sidebar; this is an active user-triggered
        // load, not a background refetch).
        showToast({ kind: "error", message: apiError(0, undefined, e, t).message });
        // Don't block the UI forever — fall through to an empty pane.
        // Re-check the seq guard here too — if a newer call landed
        // during the await, applying the empty pane would clobber it.
        if (loadHistorySeqRef.current.get(tid) !== mySeq) return;
        setThreads((all) =>
          patchThread(all, tid, {
            messages: all[tid]?.messages ?? [],
            historyLoaded: true,
          }),
        );
      }
    },
    [],
  );

  // When the active thread changes, ensure its history is loaded and
  // jump its scroll position to the bottom (handled inside ChatPane
  // by reading its messages prop). Loading is idempotent: if the
  // thread already has a buffer (e.g. user typed into it before
  // switching away), we still refetch — the server is the source of
  // truth, and any in-flight stream is preserved by handleAgentEvent
  // regardless of which thread is active.
  useEffect(() => {
    loadHistory(threadId);
  }, [threadId, loadHistory]);

  // Auto-open settings when there's a real problem the user can fix —
  // no API key configured. Cold start warmup of cached weights is NOT
  // a problem and should not interrupt the user with a modal; the
  // spinner in the chat header covers that case. Mirrors the previous
  // inline checkModels behaviour now that the hook owns the polling.
  //
  // Single-shot on first checked=true && !apiKeyConfigured transition
  // to avoid re-opening the modal if the user closes it manually and
  // the hook re-checks settings on the next refresh.
  const openedForMissingKeyRef = useRef(false);
  useEffect(() => {
    if (!modelsChecked) return;
    if (apiKeyConfigured) {
      openedForMissingKeyRef.current = false;
      return;
    }
    if (openedForMissingKeyRef.current) return;
    openedForMissingKeyRef.current = true;
    setSettingsOpen(true);
  }, [modelsChecked, apiKeyConfigured]);

  // Stable child callbacks. ``setThreadId`` from useState is already
  // referentially stable; the others wrap a state-setter in a useCallback
  // so ChatPane / Sidebar (both wrapped in React.memo) can actually skip
  // re-renders when their other props are unchanged. Without this the
  // inline `(text) => sendToThread(threadId, text)` etc. are fresh
  // lambdas every render and the memo gate is permanently open.
  const openSettings = useCallback(() => setSettingsOpen(true), []);
  const closeSettings = useCallback(() => setSettingsOpen(false), []);
  // v2.0.26.2 (PR-5) — mobile drawer open/close setters. Stable
  // identity so memo'd Sidebar / ChatPane can skip re-renders.
  const openDrawer = useCallback(() => setDrawerOpen(true), []);
  const closeDrawer = useCallback(() => setDrawerOpen(false), []);

  // Auto-close the drawer when the viewport crosses 768 px back to
  // desktop — otherwise a user who rotated from landscape to
  // portrait with the drawer open would see a permanently-shifted
  // sidebar in landscape mode. The matchMedia listener also catches
  // cases where the resize crosses the breakpoint without a single
  // window.resize event firing (e.g. devtools docked at 769 px).
  useEffect(() => {
    if (typeof window === "undefined") return undefined;
    let mql: MediaQueryList;
    try {
      mql = window.matchMedia("(max-width: 768px)");
    } catch {
      return undefined;
    }
    const onChange = (e: MediaQueryListEvent) => {
      if (!e.matches) setDrawerOpen(false);
    };
    if (typeof mql.addEventListener === "function") {
      mql.addEventListener("change", onChange);
      return () => mql.removeEventListener("change", onChange);
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (mql as any).addListener(onChange);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return () => (mql as any).removeListener(onChange);
  }, []);
  const onSend = useCallback(
    (text: string) => sendToThread(threadId, text),
    [threadId, sendToThread],
  );
  // QW #4: mid-stream stop. Closing the WS triggers the existing
  // drain handler in ``useThreadSocket`` (P1-3) — the backend keeps
  // running via ``_background_runs`` and the partial answer lands
  // in the thread's history when the user reopens it. We also flip
  // the local ``isStreaming`` flag off so the UI immediately
  // re-enables the send button (otherwise the user would stare at
  // a frozen composer until the run's natural ``done`` event — by
  // definition, never, because we closed the socket).
  const onStop = useCallback(() => {
    // The hook's ``close`` is a no-op if the socket isn't open and
    // always drops the map entry — replaces the manual get + close
    // + delete triplet the inline version had to write.
    threadSocket.close(threadId);
    setThreads((all) =>
      patchThread(all, threadId, { isStreaming: false, webSearching: false }),
    );
    // v2.0.26.3 (PR-6) — confirm to the user that the cancel
    // landed. Without this the user has to infer "the composer
    // re-enabled, so it must have worked" — the explicit feedback
    // is friendlier. Success kind + 3 s duration: shorter than
    // error toasts (5 s default) because success messages don't
    // need to be dwelled on.
    showToast({
      kind: "success",
      message: t("toast.stoppedBody"),
      duration: 3000,
    });
  }, [threadId, patchThread, threadSocket, showToast, t]);

  // ─── Citation click feedback ─────────────────────────────────────
  //
  // Inline citation chips (rendered inside the assistant answer via the
  // ``remarkCitations`` plugin) and the per-message source chips
  // (rendered below the answer) BOTH dispatch a ``rag:citation-click``
  // CustomEvent with ``detail.index``. Without this listener the click
  // is a no-op — the user sees the chip highlight on hover but nothing
  // happens when they actually click it.
  //
  // Behaviour on click:
  // 1. Find every chip in the DOM whose ``data-citation-index`` matches
  //    the clicked index — this covers the inline chip AND the
  //    matching chip in the bottom citation row of the same message,
  //    so a click on either side flashes BOTH for visual continuity.
  // 2. Add the ``is-highlighted`` class for ~2 s. The class carries a
  //    keyframe animation in styles.css that gives a brief blue ring +
  //    bg pulse — clearly visible feedback that the click registered.
  // 3. Scroll the FIRST matching chip into view inside the chat
  //    scrollable container (not the page). ``block: "nearest"`` keeps
  //    the scroll surgical: we only move if the chip is offscreen.
  //
  // Why DOM-driven rather than state-driven: the chips are rendered
  // inside ``Markdown`` (deeply nested) and a state round-trip would
  // re-render the entire message tree on every click. A direct DOM
  // mutation is cheap, scoped, and doesn't disturb React's render
  // cycle. We clean up the class via a 2 s timeout so it self-heals.
  useEffect(() => {
    function onCitationClick(ev: Event) {
      const e = ev as CustomEvent<{ index: number }>;
      const idx = e.detail?.index;
      if (typeof idx !== "number") return;
      // Query the entire document — chips live inside the chat pane
      // and (after filtering) inside the citation-row below it.
      const matches = document.querySelectorAll<HTMLElement>(
        `.citation-chip[data-citation-index="${idx}"]`,
      );
      if (!matches.length) return;
      // Scroll the first match into view. ``nearest`` is the
      // least-intrusive option: no scroll if already in viewport,
      // minimum scroll if not.
      matches[0].scrollIntoView({ behavior: "smooth", block: "nearest" });
      for (const el of Array.from(matches)) {
        // Re-trigger the animation even if the class is still
        // applied from a previous click — without this, a second
        // click within 2 s does nothing visible.
        el.classList.remove("is-highlighted");
        // Force reflow so the next class addition restarts the
        // animation rather than being coalesced by the browser.
        void el.offsetWidth;
        el.classList.add("is-highlighted");
        window.setTimeout(() => {
          el.classList.remove("is-highlighted");
        }, 2200);
      }
    }
    window.addEventListener("rag:citation-click", onCitationClick);
    return () =>
      window.removeEventListener("rag:citation-click", onCitationClick);
  }, []);

  return (
    <div className="app">
      {/* QW #10: inline banner shown when the WebSocket dies
          mid-stream. Sits above the sidebar/chat split so it's
          visible regardless of which view the user is on.
          ``role="status"`` (not ``alert``) is correct here — the
          error is non-urgent; interrupting screen-reader
          announcement would be more annoying than the frozen
          stream itself. The banner is dismissable; auto-clear on
          thread switch is handled by the effect above. */}
      {connectionError && (
        <div className="connection-error-banner" role="status">
          <span className="connection-error-icon" aria-hidden="true">
            ⚠
          </span>
          <span className="connection-error-message">
            {connectionError.message}
          </span>
          {/* P0-F2 — retry button: re-sends the last user message on
              this thread so the user doesn't have to scroll back to
              the input, retype, and re-hit send. Only shown when
              we've actually recorded a send (the ref is set inside
              ``sendToThread``); for code-driven errors with no
              user input behind them (e.g. mid-stream crash), the
              banner shows the dismiss button alone. Clearing
              ``connectionError`` here so the banner disappears
              during the retry attempt — the new attempt will either
              succeed (``onReconnected`` no-op because it's a new
              WS, not a reconnect) or fail (a fresh
              ``rag:ws-closed`` event re-shows it). */}
          {lastUserMessageByThreadRef.current.has(threadId) && (
            <button
              type="button"
              className="connection-error-retry"
              onClick={() => {
                const last = lastUserMessageByThreadRef.current.get(threadId);
                if (last) {
                  setConnectionError(null);
                  sendToThread(threadId, last);
                }
              }}
              aria-label={t("connection.retryAria")}
            >
              {t("connection.retry")}
            </button>
          )}
          <button
            type="button"
            className="connection-error-dismiss"
            onClick={() => setConnectionError(null)}
            aria-label={t("connection.dismissAria")}
          >
            ×
          </button>
        </div>
      )}
      <Sidebar
        threadId={threadId}
        onSelectThread={setThreadId}
        onOpenSettings={openSettings}
        onNewChat={onNewChat}
        drawerOpen={drawerOpen}
        onCloseDrawer={closeDrawer}
      />
      <ChatPane
        threadId={threadId}
        messages={threads[threadId]?.messages ?? EMPTY_MESSAGES}
        isStreaming={threads[threadId]?.isStreaming ?? false}
        historyLoaded={threads[threadId]?.historyLoaded ?? false}
        webSearching={threads[threadId]?.webSearching ?? false}
        currentStep={threads[threadId]?.currentStep ?? 0}
        modelsReady={modelsReady}
        onSend={onSend}
        onStop={onStop}
        drawerOpen={drawerOpen}
        onOpenDrawer={openDrawer}
        // v2.0.29.9 (Phase 8) — per-thread verbatim-extraction
        // toggle (default "auto"). The segmented control in ChatInput
        // writes through ``setThreadHighPrecision`` (defined inline
        // below); the value is forwarded via the WS payload's
        // ``high_precision`` field on the next send.
        highPrecision={threads[threadId]?.highPrecision ?? "auto"}
        onHighPrecisionChange={(v) =>
          setThreads((all) => patchThread(all, threadId, { highPrecision: v }))
        }
      />

      {settingsOpen && (
        <SettingsDialog
          current={settings}
          modelsReady={modelsReady}
          embedding={modelsEmbedding}
          reranker={modelsReranker}
          onClose={closeSettings}
          onSaved={(s) => {
            setSettings(s);
            // Don't close on Save — user may still need to trigger model
            // download. Only close when models become ready.
          }}
          onModelsReady={() => {
            // Models finished downloading inside the dialog. Refresh
            // the hook so the next tick is immediate (instead of
            // waiting up to 2 s) and close the dialog.
            refreshModels();
            setSettingsOpen(false);
          }}
        />
      )}

      {/* During the very first status poll we show the progress modal
          with the per-model rows so the user has something to look at.
          After that, if the API key is fine and we're just waiting on
          the in-RAM warmup, the chat header shows a quiet inline hint
          instead — no modal. The progress modal re-appears only if
          the API key isn't configured (handled by the conditional
          SettingsDialog open above + the keyring check below). */}
      {!modelsChecked && (
        <ModelDownloadProgress
          modelsReady={modelsReady}
          embedding={modelsEmbedding}
          reranker={modelsReranker}
          onReady={() => {
            refreshModels();
          }}
        />
      )}

      {modelsChecked && !apiKeyConfigured && !settingsOpen && (
        <div className="modal-backdrop">
          <div className="modal modal-key-required">
            <div className="modal-key-required-icon" aria-hidden="true">🔑</div>
            <h2>{t("app.apiKeyTitle")}</h2>
            <p>{t("app.apiKeyBody")}</p>
            <div className="modal-actions">
              <button
                className="btn-primary"
                onClick={() => setSettingsOpen(true)}
              >
                {t("app.apiKeyOpenSettings")}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}