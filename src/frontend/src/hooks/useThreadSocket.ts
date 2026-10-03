/**
 * useThreadSocket — per-thread WebSocket lifecycle management.
 *
 * P2-1 split: the WebSocket plumbing inside ``App.tsx`` used to live as
 * ~50 lines of inline ``useRef`` + ``useCallback`` that were easy to
 * mis-read in a 770-line file. Hoisting them into a dedicated hook
 * keeps the lifecycle concerns (lazy open / re-use open socket /
 * onclose cleanup / QW #13 reconnect with backoff) in one place and
 * lets ``App.tsx`` describe WHAT it wants ("give me the socket for
 * thread X, dispatch events to Y") without re-implementing HOW.
 *
 * Why a ref, not state
 * --------------------
 * The sockets themselves don't drive rendering — the events they
 * produce do, and those flow through the thread-state map held by
 * ``App.tsx``. Putting sockets in state would:
 *  - Trigger a re-render every time a socket opens / closes, with no
 *    visible UI change to show for it.
 *  - Fall into the React 18 StrictMode double-mount trap where
 *    state-based sockets would get re-created mid-handshake and leak
 *    the first connection.
 *
 * Why a Map (not a Record)
 * ------------------------
 * Maps have O(1) add/remove that doesn't allocate per thread the way
 * `{ ...obj, [tid]: ws }` does. We expect ~tens of threads in the
 * sidebar over a session — not millions — so the perf delta is
 * negligible, but the API ergonomics (typed ``.get(tid)`` returning
 * ``T | undefined``) beat ``Record``'s string-index signature.
 *
 * Banner hooks (onAbnormalClose / onError)
 * ----------------------------------------
 * These were previously inlined inside App.tsx's ``getOrCreateWS``.
 * They belong in this hook because they fire from the same socket
 * lifecycle that owns the de-dup flag (browsers fire BOTH ``onerror``
 * and ``onclose`` for abnormal terminations — without the per-socket
 * guard, callers would see double banners). The de-dup flag is
 * closure-local per socket; a fresh WS (after reconnect) gets its
 * own slot.
 *
 * Why 1000/1005 codes are filtered here
 * --------------------------------------
 * 1000 (Normal Closure) and 1005 (No Status Received) are NOT
 * user-actionable — every successful turn fires one of them via the
 * explicit ``ws.close()`` from the ``done`` handler. Surfacing them
 * as "⚠ 聊天连接已断开" was the user-reported v1.1.16 regression.
 * Codes that DO mean something (1001 / 1006 / 1008 / 1009 / 1011 /
 * 1014) are passed to ``onAbnormalClose`` so the banner can map them
 * to a localized message.
 */
import { useCallback, useEffect, useRef } from "react";
// v2.0.26.3 (PR-6) — toast dispatcher singleton so the hook can
// fire user-visible notifications without coupling to React
// context. The Provider registers itself on mount via
// ``setToastDispatcher``; we read it via ``getCurrentDispatcher``.
// Falls back to ``console.error`` if no Provider is mounted (e.g.
// in isolated unit tests that render the hook directly).
import { getCurrentDispatcher } from "../components/Toast";
import { apiError, getCurrentT } from "../i18n/errors";

/**
 * Construct the chat WebSocket URL for a thread.
 *
 * Pulled out of the hook body so tests / Storybook can call it without
 * a DOM. Mirrors the backend's ``/ws/chat?thread_id=<tid>`` contract;
 * any change here must land in tandem with ``src/api/websocket.py``.
 */
export function chatSocketUrl(threadId: string): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/chat?thread_id=${encodeURIComponent(threadId)}`;
}

/** QW #13: per-thread reconnect attempt counter, paired with the
 *  backoff schedule below. Keyed by thread id; cleared when the
 *  thread successfully opens OR the user explicitly gives up. */
const RECONNECT_BACKOFF_MS = [1000, 2000, 5000] as const;
const RECONNECT_MAX_ATTEMPTS = RECONNECT_BACKOFF_MS.length;

export interface UseThreadSocketOptions {
  /** Called for every parsed agent event on any managed socket. */
  onMessage: (threadId: string, event: unknown) => void;
  /**
   * Decide whether a given thread is eligible for an automatic
   * reconnect after an unexpected close. ``App.tsx`` returns
   * ``true`` only for threads that are mid-stream (so an idle
   * thread that lost its socket doesn't keep trying to dial home).
   * Return ``false`` to skip reconnect entirely for the thread.
   */
  shouldReconnect: (threadId: string) => boolean;
  /**
   * Called after each reconnect attempt — useful for surfacing a
   * "正在重连 (attempt 2/3)..." hint. Defaults to a no-op so
   * callers that don't care can omit it.
   */
  onReconnectAttempt?: (threadId: string, attempt: number) => void;
  /**
   * Called once we've exhausted retries. The parent should
   * surface a permanent error (the QW #10 banner stays up).
   */
  onReconnectExhausted?: (threadId: string) => void;
  /**
   * Called on abnormal close (code !== 1000 / 1005). Used by App
   * to fire the ``rag:ws-closed`` window event that drives the
   * connection-error banner. Fires at most ONCE per socket
   * lifecycle — the per-socket ``alreadyDispatched`` flag de-dups
   * against ``onError`` (browsers fire both for abnormal
   * terminations).
   */
  onAbnormalClose?: (threadId: string, code: number) => void;
  /**
   * Called on ``onerror`` (always abnormal). Fires at most ONCE
   * per socket lifecycle; subsequent ``onclose`` for the same
   * termination will NOT also fire ``onAbnormalClose``.
   */
  onError?: (threadId: string) => void;
  /**
   * P0-F2: Called when a socket successfully opens AFTER a previous
   * abnormal close for the same thread. Used by App to clear the
   * connection-error banner. Does NOT fire on the initial
   * ``getOrCreate`` open — only on genuine reconnects, so the banner
   * doesn't flash off on first page load. Fires once per successful
   * reconnect; the next close clears the ``wasReconnect`` flag again.
   */
  onReconnected?: (threadId: string) => void;
}

export interface UseThreadSocketApi {
  /**
   * Return the OPEN socket for ``threadId`` or open a fresh one. Safe
   * to call multiple times for the same thread — second-and-later
   * calls reuse the existing socket as long as it's still OPEN.
   */
  getOrCreate: (threadId: string) => WebSocket;
  /**
   * Close the socket for ``threadId`` (no-op if not present or not
   * OPEN). Does NOT cancel a pending reconnect — use
   * ``cancelReconnect`` for that. Called by App on the LLM's
   * ``done`` event (to release the connection cleanly) and from
   * the onStop button (user cancels mid-stream).
   */
  close: (threadId: string, reason?: string) => void;
  /**
   * Test/cleanup helper. Closes every managed socket and forgets the
   * map. Not used in production — React's normal lifecycle doesn't
   * expose a teardown hook here — but handy for Jest teardown.
   */
  closeAll: () => void;
  /**
   * Cancel any pending reconnect for ``threadId``. Called by the
   * parent when the user clicks "停止生成" or otherwise signals
   * they no longer care about the in-flight stream.
   */
  cancelReconnect: (threadId: string) => void;
}

export function useThreadSocket(opts: UseThreadSocketOptions): UseThreadSocketApi {
  const {
    onMessage,
    shouldReconnect,
    onReconnectAttempt,
    onReconnectExhausted,
    onAbnormalClose,
    onError,
    onReconnected,
  } = opts;
  // Hold the map in a ref so the callback identity is stable across
  // renders. Without this, every App render would produce a new
  // ``getOrCreate`` reference, which propagates to ChatPane / Sidebar
  // and re-opens their memo gate.
  const wsByThreadRef = useRef<Map<string, WebSocket>>(new Map());
  // QW #13: pending reconnect timers per thread. Stored outside
  // ``wsByThreadRef`` because the timer is independent of the WS
  // object — the WS may be GC'd, the timer must outlive it.
  const reconnectTimersRef = useRef<Map<string, number>>(new Map());
  // Per-thread attempt counter. Bumped on every scheduled
  // reconnect; cleared on a successful next open or on cancel.
  const reconnectAttemptRef = useRef<Map<string, number>>(new Map());

  // QW #13: attach handlers to a freshly-created socket. Shared
  // between the initial open (via ``getOrCreate``) and the
  // reconnect timer so the lifecycle logic stays single-sourced.
  const attachHandlers = useCallback(
    (ws: WebSocket, threadId: string) => {
      // v1.1.16 — closure-local de-dup flag. Browsers fire BOTH
      // ``onerror`` and ``onclose`` for abnormal terminations
      // (in that order). Without this flag the user would see the
      // disconnect banner flash twice — once from the network-level
      // error, once from the close-frame follow-up. Closure-local
      // means a fresh WS (after reconnect) gets its own slot.
      let alreadyDispatched = false;
      // P0-F2: closure-local flag set by ``onclose`` when this
      // socket will be reconnected (parent still cares + reconnect
      // scheduled). The next ``onopen`` consults this to decide
      // whether to fire ``onReconnected`` — the initial page-load
      // open does NOT count, so the banner doesn't flash off on
      // first paint. Each fresh WS gets its own slot, matching the
      // ``alreadyDispatched`` pattern above.
      let wasReconnect = false;

      ws.onopen = () => {
        // Reset the attempt counter on the OPEN side, not the CLOSE
        // side — that way "attempts that actually dialed" is what
        // the counter measures (instead of "attempts that ended in
        // a close"). An attempt that opened but failed immediately
        // before close still counted, which is the correct semantic
        // for the reconnect backoff ladder.
        reconnectAttemptRef.current.delete(threadId);
        if (wasReconnect) {
          onReconnected?.(threadId);
        }
      };
      ws.onmessage = (ev) => {
        // Caller is responsible for parsing / dispatching. We hand off
        // the raw parsed payload rather than re-serializing, so the
        // shape stays in one place (App.tsx's handleAgentEvent).
        try {
          const data = JSON.parse(ev.data);
          onMessage(threadId, data);
        } catch (e) {
          // v2.0.26.3 (PR-6) — toast in addition to the console log.
          // Warning kind (not error) because a single malformed frame
          // from the server is non-fatal — the rest of the stream
          // may still be fine. The next valid ``done`` will reset
          // the user's mental model.
          console.error("Failed to parse WS message", e);
          getCurrentDispatcher()?.({
            kind: "warning",
            message: getCurrentT()("toast.wsParseError"),
          });
        }
      };
      ws.onclose = (ev: CloseEvent) => {
        // Clean up the map entry. We do NOT clear streaming state
        // here — the backend may still be running the agent run
        // (the ``_drain_generator`` handoff keeps it alive even
        // though the client is gone). The user will see the answer
        // the next time they load this thread's history. The only
        // way to know the run is truly done is the ``done`` event.
        wsByThreadRef.current.delete(threadId);
        const pendingTimer = reconnectTimersRef.current.get(threadId);
        if (pendingTimer !== undefined) {
          window.clearTimeout(pendingTimer);
          reconnectTimersRef.current.delete(threadId);
        }
        // Codes that DO warrant a banner:
        //   1001 Going Away              — server shutdown / page nav
        //   1006 Abnormal Closure        — network drop, server crash
        //   1008 Policy Violation        — auth / origin rejection
        //   1009 Message Too Big         — payload exceeded size cap
        //   1011 Internal Error          — server-side unhandled error
        //   1014 Bad Gateway             — proxy / load-balancer issue
        //   anything else                — be defensive
        // Codes that DON'T (Normal Closure 1000 fires on every
        // successful ``done``; 1005 No Status Received fires when
        // the server crashes without a close frame — but the
        // matching ``onerror`` already showed the banner).
        const isNormal = ev.code === 1000 || ev.code === 1005;
        if (!isNormal && !alreadyDispatched) {
          alreadyDispatched = true;
          onAbnormalClose?.(threadId, ev.code);
        }
        // QW #13: only schedule a reconnect for threads the parent
        // still cares about (mid-stream). Idle threads that lose
        // their socket just stay idle — no surprise traffic.
        if (shouldReconnect(threadId)) {
          wasReconnect = true;
          scheduleReconnectFor(threadId);
        }
      };
      ws.onerror = () => {
        // Mirror onclose's cleanup; browsers fire onerror before
        // onclose for abnormal terminations. Don't reconnect here —
        // ``onclose`` will fire next and own the scheduling.
        wsByThreadRef.current.delete(threadId);
        if (!alreadyDispatched) {
          alreadyDispatched = true;
          onError?.(threadId);
        }
      };
    },
    [onMessage, shouldReconnect, onAbnormalClose, onError, onReconnected],
  );

  // QW #13: schedule the next reconnect attempt. Reads the current
  // attempt counter, picks the matching backoff delay, and opens a
  // fresh socket when the timer fires. Each schedule bumps the
  // counter; the cap (RECONNECT_MAX_ATTEMPTS) is enforced here.
  const scheduleReconnectFor = useCallback(
    (threadId: string) => {
      const attempt = reconnectAttemptRef.current.get(threadId) ?? 0;
      if (attempt >= RECONNECT_MAX_ATTEMPTS) {
        reconnectAttemptRef.current.delete(threadId);
        onReconnectExhausted?.(threadId);
        return;
      }
      const delay = RECONNECT_BACKOFF_MS[attempt];
      reconnectAttemptRef.current.set(threadId, attempt + 1);
      onReconnectAttempt?.(threadId, attempt + 1);
      const timer = window.setTimeout(() => {
        // The user may have stopped the stream while we were
        // waiting — re-check the predicate at fire time so we
        // don't open a brand-new socket for a thread the parent
        // has marked dead.
        if (!shouldReconnect(threadId)) {
          reconnectAttemptRef.current.delete(threadId);
          return;
        }
        try {
          const existing = wsByThreadRef.current.get(threadId);
          if (existing && existing.readyState === WebSocket.OPEN) {
            return;  // somebody already opened it (e.g. user retried)
          }
          if (existing) {
            try {
              existing.close();
            } catch {
              /* ignore */
            }
            wsByThreadRef.current.delete(threadId);
          }
          const ws = new WebSocket(chatSocketUrl(threadId));
          wsByThreadRef.current.set(threadId, ws);
          attachHandlers(ws, threadId);
        } catch (e) {
          // v2.0.26.3 (PR-6) — toast alongside the console log so
          // the user knows the next reconnect attempt failed (the
          // banner they had was for the original disconnect; this
          // is the follow-up dial failure which previously was
          // invisible).
          console.error("reconnect failed:", e);
          getCurrentDispatcher()?.({
            kind: "error",
            message: apiError(0, undefined, e, getCurrentT()).message,
          });
        }
      }, delay);
      reconnectTimersRef.current.set(threadId, timer);
    },
    [shouldReconnect, onReconnectAttempt, onReconnectExhausted, attachHandlers],
  );

  const getOrCreate = useCallback(
    (threadId: string): WebSocket => {
      const existing = wsByThreadRef.current.get(threadId);
      if (existing && existing.readyState === WebSocket.OPEN) {
        return existing;
      }
      if (existing) {
        // CONNECTING or CLOSING — discard and replace. We don't await
        // close; the old socket will be GC'd when the new one binds.
        try {
          existing.close();
        } catch {
          /* already closed */
        }
        wsByThreadRef.current.delete(threadId);
      }
      const ws = new WebSocket(chatSocketUrl(threadId));
      wsByThreadRef.current.set(threadId, ws);
      attachHandlers(ws, threadId);
      return ws;
    },
    [attachHandlers],
  );

  const close = useCallback((threadId: string, reason?: string) => {
    const ws = wsByThreadRef.current.get(threadId);
    if (!ws) return;
    if (ws.readyState === WebSocket.OPEN) {
      try {
        ws.close();
      } catch {
        /* already closing */
      }
    }
    // ``reason`` is informational (logged for QW-13 debugging); the
    // WebSocket frame doesn't carry caller-supplied close reasons
    // cross-browser without a code, so we just record the why in
    // the console for grep-after-incident triage.
    if (reason) {
      // eslint-disable-next-line no-console
      console.debug(`[ws] close(${threadId}) reason=${reason}`);
    }
    // Drop the map entry eagerly — ``onclose`` will fire next and
    // also delete, but we don't want a follow-up getOrCreate to
    // briefly see a half-closed socket and try to send on it.
    wsByThreadRef.current.delete(threadId);
  }, []);

  const closeAll = useCallback(() => {
    for (const ws of wsByThreadRef.current.values()) {
      try {
        ws.close();
      } catch {
        /* ignore */
      }
    }
    wsByThreadRef.current.clear();
    // QW #13: also cancel any in-flight reconnect timers so a
    // hard remount doesn't keep firing in the background.
    for (const timer of reconnectTimersRef.current.values()) {
      window.clearTimeout(timer);
    }
    reconnectTimersRef.current.clear();
    reconnectAttemptRef.current.clear();
  }, []);

  const cancelReconnect = useCallback((threadId: string) => {
    const timer = reconnectTimersRef.current.get(threadId);
    if (timer !== undefined) {
      window.clearTimeout(timer);
      reconnectTimersRef.current.delete(threadId);
    }
    reconnectAttemptRef.current.delete(threadId);
  }, []);

  // Defensive cleanup: if the hook itself unmounts (rare — App.tsx
  // doesn't unmount during normal use, only in tests), drop
  // everything so no timers fire into the void.
  useEffect(() => {
    return () => {
      closeAll();
    };
  }, [closeAll]);

  return { getOrCreate, close, closeAll, cancelReconnect };
}