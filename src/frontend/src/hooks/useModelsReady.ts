/**
 * useModelsReady — single-source-of-truth poll for backend model warmup.
 *
 * Replaces three independent polling loops that lived in App.tsx,
 * SettingsDialog.tsx, and ModelDownloadProgress.tsx (each with its own
 * cadence — 2s, 5s, 5s respectively — and its own cancellation
 * pattern). The hook collapses them to ONE poll of `/models/status`,
 * exits cleanly on terminal state (ready / missing / failed) instead
 * of polling forever, and exposes both the boolean `ready` flag and a
 * higher-level `status` so consumers can render banners for non-ready
 * terminal states (P0-F1's "dead poller never exits" complaint).
 *
 * Terminal-state rationale
 * ------------------------
 * The previous inline polling only short-circuited on ``ready === true``
 * (App.tsx:1142-1161). If the backend returns ``status: "missing"``
 * (no weights configured) or ``status: "failed"`` (download errored),
 * the poll fired every 2s forever — wasted traffic AND a confusing
 * "spinner that never goes away" UX. The hook stops on any terminal
 * value and surfaces the status to the caller so they can render
 * "models not configured — open settings" instead.
 *
 * Settings fetch is one-shot
 * --------------------------
 * The backend's /settings endpoint is expensive-ish (constructs the
 * full ``Settings`` Pydantic model + env resolution). The hook fetches
 * it on the first tick only to populate ``apiKeyConfigured``; the
 * ``useEffect`` in App that needs to auto-open settings on cold launch
 * keys off ``apiKeyConfigured`` (a single render), so subsequent
 * settings changes (e.g. user toggles provider) are picked up via the
 * regular SettingsDialog save flow, not via this hook.
 *
 * Visibility-aware polling
 * ------------------------
 * v2.0.28.20 — Chrome/Edge throttle ``setTimeout`` in background tabs
 * to once per ~60 s, which stretches our 2 s poll cadence to a minute
 * and leaves the chat-header spinner stuck for the entire interval
 * even after the user comes back. The hook attaches a
 * ``visibilitychange`` listener that fires an immediate re-poll the
 * moment the tab returns to the foreground, so the spinner clears
 * without the user needing to refresh.
 *
 * Long-poll first, fall back to polling
 * --------------------------------------
 * v2.0.30.0 — pre-fix, the foregrounded-tab spinner could lag up to
 * 2 s behind backend ``ready=true`` because the hook only re-read
 * status on a 2 s ``setTimeout`` cadence. We now issue one
 * ``GET /models/wait?timeout=60`` on mount; the backend blocks the
 * request for up to 60 s and returns as soon as both models report
 * ``loaded=True`` (latency ~250 ms when models are already ready,
 * which is the cold-restart-with-warm-cache case). On timeout (408)
 * the hook falls back to the existing 2 s polling cadence. On tab
 * hide the long-poll is aborted via ``AbortController``; the existing
 * visibilitychange listener still drives an immediate re-poll on
 * focus, so the worst-case is bounded by the network roundtrip.
 */
import { useCallback, useEffect, useState } from "react";
import { getModelsStatus, getSettings, waitForModelsReady } from "../api/client";

const POLL_MS = 2000;

export type OverallModelsStatus = "pending" | "ready" | "missing" | "failed";

/** Per-model status payload as returned by ``/models/status``. */
export interface ModelState {
  name: string;
  status: string;
}

export interface UseModelsReadyResult {
  /** Backend reports both embedding + reranker ready (caller can gate input). */
  ready: boolean;
  /** First poll completed (caller can hide initial-load spinner). */
  checked: boolean;
  /** Derives an overall status from per-model statuses. */
  status: OverallModelsStatus;
  /** Backend's ``api_key_source`` is not "none" (caller can suppress api-key modal). */
  apiKeyConfigured: boolean;
  // v2.0.25 P0-F1 — per-model granularity for components that render
  // a row per model (SettingsDialog's "embedding / reranker status"
  // panel + ModelDownloadProgress's first-run modal). Without these,
  // those components couldn't migrate to the hook without losing
  // their existing UI labels.
  embedding: ModelState | null;
  reranker: ModelState | null;
  /**
   * Force an immediate re-poll. Called after the user successfully
   * downloads models via SettingsDialog (which previously called
   * ``setModelsReady(true)`` directly). The hook re-runs ``tick``
   * synchronously; if the backend now reports ``ready``, the hook
   * flips state and exits the poll loop. If still not ready, the
   * regular 2 s cadence resumes.
   */
  refresh: () => void;
}

/**
 * Derive an overall status from the per-model status map.
 *
 * - "missing": either model is missing → user must download / configure
 * - "failed": either model failed → user must retry
 * - "ready": both ready (matches backend's ``s.ready``)
 * - "pending": warmup/download in flight, or both pending
 */
function deriveStatus(s: {
  embedding: { status: string };
  reranker: { status: string };
}): OverallModelsStatus {
  if (s.embedding.status === "missing" || s.reranker.status === "missing") {
    return "missing";
  }
  if (s.embedding.status === "failed" || s.reranker.status === "failed") {
    return "failed";
  }
  if (s.embedding.status === "ready" && s.reranker.status === "ready") {
    return "ready";
  }
  return "pending";
}

export function useModelsReady(): UseModelsReadyResult {
  const [ready, setReady] = useState(false);
  const [checked, setChecked] = useState(false);
  const [status, setStatus] = useState<OverallModelsStatus>("pending");
  const [apiKeyConfigured, setApiKeyConfigured] = useState(true);
  const [embedding, setEmbedding] = useState<ModelState | null>(null);
  const [reranker, setReranker] = useState<ModelState | null>(null);
  // Bumped by ``refresh()`` to force the poll effect to re-run
  // immediately. Used by SettingsDialog after a successful download.
  const [refreshTick, setRefreshTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let settingsFetched = false;
    // v2.0.30.0 — single AbortController for the long-poll on mount
    // and any subsequent long-polls kicked by ``onVisibility`` after
    // a tab-hide abort. Cancelled in the effect cleanup so an
    // unmount-mid-wait leaves no orphan request.
    let waitController: AbortController | null = null;

    /** Apply a status snapshot to state, plus the one-shot settings fetch. */
    async function apply(s: Awaited<ReturnType<typeof getModelsStatus>>) {
      const next = deriveStatus(s);
      setStatus(next);
      setReady(s.ready);
      setChecked(true);
      setEmbedding(s.embedding ?? null);
      setReranker(s.reranker ?? null);

      if (!settingsFetched) {
        settingsFetched = true;
        try {
          const settings = await getSettings();
          if (cancelled) return;
          setApiKeyConfigured(settings.api_key_source !== "none");
        } catch {
          /* settings fetch failure is non-fatal; apiKeyConfigured
             stays at its default (true) so the modal doesn't open
             on a transient backend hiccup */
        }
      }
      return next;
    }

    /** Long-poll wrapper. Returns the snapshot on success, ``null`` on
     *  timeout / abort / network error so the caller can fall back to
     *  the polling path.
     */
    async function longPoll(signal: AbortSignal) {
      try {
        return await waitForModelsReady(60, signal);
      } catch (err) {
        // AbortError / 408 / network error all map to "fall back".
        return null;
      }
    }

    /** Fall-back polling path (the original v2.0.28.19 behavior). */
    async function tick() {
      try {
        const s = await getModelsStatus();
        if (cancelled) return;
        const next = await apply(s);
        // Terminal: stop polling. Caller renders banner for non-ready
        // terminal states via the ``status`` field.
        if (s.ready || next === "missing" || next === "failed") {
          return;
        }
      } catch {
        /* server may still be starting up — keep polling silently */
      }
      if (!cancelled) timer = setTimeout(tick, POLL_MS);
    }

    // v2.0.30.0 — long-poll first. On mount we ask the backend to
    // block until models are ready (or 60 s elapses, whichever comes
    // first). When the backend reports ready (typical: ~250 ms on a
    // warm cache, ~30 s on cold load), we apply the snapshot and
    // exit. On timeout / abort / network error we fall back to the
    // 2 s polling cadence below.
    waitController = new AbortController();
    longPoll(waitController.signal).then((snapshot) => {
      if (cancelled) return;
      if (snapshot) {
        apply(snapshot).then((next) => {
          if (cancelled) return;
          // Terminal: nothing more to do. Otherwise keep polling.
          if (!snapshot.ready && next !== "missing" && next !== "failed") {
            tick();
          }
        });
        return;
      }
      // Timeout / abort / error — fall back to 2 s polling.
      tick();
    });

    // v2.0.28.20 — background-tab throttling guard. Chrome/Edge throttle
    // ``setTimeout`` in hidden tabs to once per ~60 s, which would
    // stretch our 2 s poll cadence to a minute and leave the spinner
    // stuck for the entire interval even after the user comes back.
    // Force an immediate re-poll on ``visibilitychange`` → visible so
    // the user sees the spinner clear the moment they focus the tab.
    //
    // v2.0.30.0 — also abort the in-flight long-poll when the tab
    // hides, so we don't hold a backend request open while the user
    // is elsewhere. On focus we kick a fresh long-poll (or fall back
    // to a poll if the previous one was aborted). Cheap (one
    // fetch per focus), bounded by ``cancelled``, and refcount-safe
    // via the effect's own cleanup.
    function onVisibility() {
      if (cancelled) return;
      if (document.visibilityState !== "visible") {
        // Cancel any in-flight long-poll; on next focus we'll kick
        // a fresh one. The visibilitychange → visible branch below
        // handles the restart.
        if (waitController) {
          waitController.abort();
          waitController = null;
        }
        return;
      }
      // Cancel the throttled timer (if any) so we don't double-poll,
      // then run an immediate tick.
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
      // If no long-poll is in flight, start one; otherwise just let
      // the existing one resolve. ``waitController === null`` is
      // the canonical "no wait in flight" signal.
      if (!waitController) {
        waitController = new AbortController();
        longPoll(waitController.signal).then((snapshot) => {
          if (cancelled) return;
          if (snapshot) {
            apply(snapshot).then((next) => {
              if (cancelled) return;
              if (!snapshot.ready && next !== "missing" && next !== "failed") {
                tick();
              }
            });
            return;
          }
          tick();
        });
        return;
      }
      tick();
    }
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
      if (waitController) waitController.abort();
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [refreshTick]);

  const refresh = useCallback(() => {
    // Reset to "not ready" so the spinner comes back until the next
    // poll confirms. Without this the user sees a stale "ready" state
    // for up to 2 s after the download completes.
    setReady(false);
    setStatus("pending");
    setRefreshTick((n) => n + 1);
  }, []);

  return {
    ready,
    checked,
    status,
    apiKeyConfigured,
    embedding,
    reranker,
    refresh,
  };
}