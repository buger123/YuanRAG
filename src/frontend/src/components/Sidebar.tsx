import { memo, useEffect, useRef, useState } from "react";
import {
  clearThreadDocuments,
  deleteDocument,
  deleteSession,
  listDocuments,
  listFormats,
  listSessions,
  uploadDocument,
  type Document,
  type SessionSummary,
  type SupportedFormats,
} from "../api/client";
import { Logo } from "./Logo";
import { ConfirmDialog } from "./ConfirmDialog";
import { useLocale } from "../i18n";
import { useToast } from "./Toast";

interface Props {
  threadId: string;
  onSelectThread: (id: string) => void;
  onOpenSettings: () => void;
  /** Handle for both "+ 新对话" and "delete the active session, then
   *  start a fresh empty chat" flows. The parent mints a new UUID,
   *  clears the ``rag.active_thread`` localStorage entry, and
   *  registers the UUID in its ``freshUuidsRef`` so ``loadHistory``
   *  skips the ``/sessions/{tid}/messages`` GET that would 404 anyway —
   *  without this, ChatPane briefly flashes the "正在加载对话历史…"
   *  spinner every time the user opens a new chat. */
  onNewChat: () => void;
  // v2.0.26.2 (PR-5) — mobile drawer state. When ``drawerOpen`` is
  // true the aside is positioned fixed and slides in from the left.
  // ``onCloseDrawer`` fires when the user taps the backdrop or
  // selects a session (so they don't need a second tap to dismiss
  // the drawer after navigating).
  drawerOpen: boolean;
  onCloseDrawer: () => void;
}

/** Per-file ingestion progress. Shown in the documents panel while a
 *  file is being uploaded / parsed / indexed so the UI never feels
 *  stuck. ``stage`` transitions: uploading → parsing → indexing →
 *  done. ``chunkCount`` is updated from each polling tick once the
 *  backend starts producing chunks. */
interface UploadProgress {
  tempId: string;
  filename: string;
  stage: "uploading" | "parsing" | "indexing";
  chunkCount: number;
}

function SidebarInner({
  threadId,
  onSelectThread,
  onOpenSettings,
  onNewChat,
  drawerOpen,
  onCloseDrawer,
}: Props) {
  const { t } = useLocale();
  // v2.0.26.3 (PR-6) — toast dispatcher for destructive-action
  // failure feedback. Replaces the PR-2 inline ``deleteError``
  // banner. The toast lives at the app root via ``ToastProvider``
  // in ``main.tsx`` so we don't need to mount anything here.
  const { showToast } = useToast();
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [dragging, setDragging] = useState(false);
  // In-flight uploads. Each item is shown above the indexed docs in
  // the documents panel; once ingestion completes, the item is removed
  // from this list and the freshly-indexed doc appears via the regular
  // ``documents`` state on the next ``listDocuments`` poll.
  const [uploadProgress, setUploadProgress] = useState<UploadProgress[]>(
    [],
  );
  // v2.0.10.3 — fallback refresh intervals for uploads that exceed the
  // 3-minute ``pollUntilIndexed`` budget. If the polling times out
  // (large PDF + slow Docling + BGE-M3), we keep periodically
  // refreshing the doc list for up to 5 more minutes so the user
  // eventually sees the terminal status without an F5. Active
  // intervals are tracked here so unmount cleans them up; each
  // tick clears its own entry once the doc reaches terminal status.
  const fallbackRefreshRef = useRef<Set<ReturnType<typeof setInterval>>>(
    new Set(),
  );

  // Pending deletion for a document — drives the ConfirmDialog.
  const [pendingDocDelete, setPendingDocDelete] = useState<Document | null>(
    null,
  );
  // Pending deletion for a session — drives the ConfirmDialog.
  const [pendingSessionDelete, setPendingSessionDelete] =
    useState<SessionSummary | null>(null);
  // Upload-error notification — replaces the native ``alert()`` so the
  // copy can be styled + localized and the dialog matches the rest.
  const [uploadError, setUploadError] = useState<string | null>(null);
  // v2.0.25.1 P0-extension — inline error banner for destructive
  // actions (deleteDocument / deleteSession). The previous code
  // swallowed failures via ``console.error``, leaving the user with
  // no feedback when their click did nothing. v2.0.26.3 (PR-6) —
  // REPLACED this inline banner with the toast system. The
  // ``useToast`` hook below is now what surfaces failures; the
  // ``deleteError`` state + ``sidebar-error-banner`` JSX block were
  // deleted (see git diff).
  // (No local state needed — toasts live in the module-level store.)
  // Accepted upload formats. Fetched once from ``GET /formats`` on mount
  // (single source of truth — backend ``src/ingestion/format_detect.py``
  // owns the list). While the request is in flight we fall back to a
  // permissive accept filter so the OS picker isn't blank; once the
  // response arrives we replace it. Hover tooltip rebuilds from the
  // server-provided labels, so a future format added to the backend
  // shows up here with zero frontend work.
  const [formats, setFormats] = useState<SupportedFormats | null>(null);
  useEffect(() => {
    listFormats().then(setFormats).catch(console.error);
  }, []);

  // v2.0.25.1 P0-extension — removed the redundant
// ``useEffect(() => listSessions(), [threadId])`` listener. The
// sessions list is thread-independent (it's the whole sidebar
// catalog), so refetching it on every thread switch was wasted
// bandwidth PLUS a race condition: if the user's last action
// refreshed sessions via the ``rag:session-updated`` event handler
// below AND the threadId listener fires concurrently, the slower
// response overwrites the fresher one. Now only the event handler
// refreshes sessions, and on the initial mount via the ``useEffect
// with [] deps`` that already exists below.

// Strict per-conversation isolation: when the active thread changes,
  // re-fetch ONLY that conversation's documents.
  useEffect(() => {
    listDocuments(threadId)
      .then(setDocuments)
      .catch(console.error);
  }, [threadId]);

  // ChatPane fires ``rag:session-updated`` on every WS ``done``. Only
  // refresh ``sessions`` here — documents are already up-to-date (a
  // ``done`` doesn't touch the doc list). See also Sidebar's commit
  // message history for the full reasoning.
  //
  // v2.0.25.1 P0-extension — also fetch the initial sessions list on
  // mount (``[]`` deps below still doesn't fire until the event lands,
  // so the user would see an empty sidebar until the first ``done``
  // event). The explicit one-shot initial fetch closes that gap.
  useEffect(() => {
    function refresh() {
      listSessions().then(setSessions).catch(console.error);
    }
    refresh();
    window.addEventListener("rag:session-updated", refresh);
    return () => window.removeEventListener("rag:session-updated", refresh);
  }, []);

  // v2.0.10.3 — cleanup any in-flight fallback-refresh intervals on
  // unmount. Without this, navigating away from a thread mid-ingest
  // leaves a setInterval firing forever (and React would warn about
  // state updates on an unmounted component if the interval ticks
  // after the Sidebar is gone).
  useEffect(() => {
    return () => {
      fallbackRefreshRef.current.forEach((id) => clearInterval(id));
      fallbackRefreshRef.current.clear();
    };
  }, []);

  async function handleFile(file: File) {
    // Add an in-flight row so the user sees the file immediately, with
    // a spinner. The row mutates through three stages (uploading →
    // parsing → indexing) and is removed when the doc is queryable.
    const tempId = crypto.randomUUID();
    setUploadProgress((prev) => [
      ...prev,
      {
        tempId,
        filename: file.name,
        stage: "uploading",
        chunkCount: 0,
      },
    ]);
    try {
      const result = await uploadDocument(file, threadId);
      // POST is back — transition to "parsing" while Docling turns the
      // file into text + chunks. The bar becomes indeterminate (the
      // CSS ``upload-progress-bar-fill`` animation handles this) since
      // chunkCount is still 0.
      setUploadProgress((prev) =>
        prev.map((p) =>
          p.tempId === tempId ? { ...p, stage: "parsing" } : p,
        ),
      );
      const reachedTerminal = await pollUntilIndexed(
        threadId,
        result.doc_id,
        (chunkCount) => {
          setUploadProgress((prev) =>
            prev.map((p) =>
              p.tempId === tempId
                ? {
                    ...p,
                    // Stay in "parsing" until at least one chunk exists,
                    // then move to "indexing" so the chunk count can be
                    // shown as a fixed-width bar (CSS swaps to no
                    // animation once the data attr is set).
                    stage: chunkCount > 0 ? "indexing" : "parsing",
                    chunkCount,
                  }
                : p,
            ),
          );
        },
        // v2.0.10.4 — surface every poll's fresh ``listDocuments``
        // response to the sidebar's ``documents`` state. This is the
        // belt-and-suspenders that closes the stale-chip window: even
        // if some future edge case (race / bug / schema drift) makes
        // ``pollUntilIndexed`` exit with a non-terminal doc, the chip
        // will already reflect whatever the backend actually returned
        // on the last tick. The polling's own internal exit is then
        // idempotent — the post-polling snapshot below is just a
        // final reconciliation, not the user's only chance to see
        // the latest status.
        (docs) => setDocuments(docs),
      );
      // Ingestion complete (or timed out). Drop the progress row and
      // pull the canonical doc list (with the now-final chunk_count).
      setUploadProgress((prev) => prev.filter((p) => p.tempId !== tempId));
      listDocuments(threadId).then(setDocuments).catch(console.error);
      listSessions().then(setSessions).catch(console.error);
      window.dispatchEvent(new CustomEvent("rag:session-updated"));
      // v2.0.10.4 — ALWAYS schedule the fallback refresh after upload,
      // regardless of whether polling saw a terminal status. The
      // previous ``if (!reachedTerminal)`` branch left a hole: when
      // the polling exited prematurely via the register-race path
      // (``mine === undefined`` defaulted to ``"indexed"``),
      // ``reachedTerminal`` was ``true``, no follow-up refresh was
      // scheduled, and the doc sat at "向量化中" until F5. The
      // fallback is cheap (one ``listDocuments`` every 5 s, capped
      // at 60 s) so we can afford to schedule it unconditionally
      // — and it self-stops the moment the doc reaches a terminal
      // status, so the post-polling-exit case has near-zero cost.
      scheduleFallbackRefresh(threadId, result.doc_id, 5_000, 60_000);
      // Back-compat: keep the ``!reachedTerminal`` branch for the
      // very-slow-poll case (>180 s) where the budget expired.
      // ``scheduleFallbackRefresh`` already idempotently no-ops if
      // the doc has reached a terminal state, so double-scheduling
      // is harmless; we just give the slow-poll path a longer
      // window (the legacy 15 s / 5 min budget) so an actually-hung
      // backend still has a chance to recover.
      if (!reachedTerminal) {
        scheduleFallbackRefresh(threadId, result.doc_id, 15_000, 300_000);
      }
    } catch (e) {
      setUploadProgress((prev) => prev.filter((p) => p.tempId !== tempId));
      setUploadError((e as Error).message);
    }
  }

  /** Poll the documents list until ``docId`` reaches a terminal status
   *  (``indexed`` / ``empty`` / ``failed``) or the timeout fires.
   *  Returns ``true`` if the terminal status was observed, ``false``
   *  if the budget expired (caller should schedule a fallback
   *  refresh so the user doesn't see a stale chip until F5).
   *  Swallows network errors so a transient blip doesn't kill the
   *  poll. ``onChunkCount`` fires on every tick so the UI can drive
   *  a chunk-aware progress bar; ``onDocuments`` (v2.0.10.4) fires
   *  whenever a fresh ``listDocuments`` response arrives so the
   *  caller can mirror the backend's view in real time — closing
   *  the stale-chip window between upload and the eventual flip.
   *
   *  v2.0.10 — the prior ``if (cc > 0) return`` check raced against
   *  the server-side ``add_chunks`` → ``doc_registry.update("indexed")``
   *  sequence: chunks would land in LanceDB (cc > 0), the polling
   *  exits, the sidebar immediately calls ``listDocuments`` for the
   *  doc list, and the registry was still at "embedding" → chip
   *  stuck at "向量化中(N 块)" indefinitely (no auto-refresh hook).
   *  Now we wait for the terminal status so the doc list we render
   *  matches the chunk store. The server-side atomic fix in
   *  ``add_chunks`` already narrows the window to ~zero, but this
   *  is the defensive belt for any future race.
   *
   *  v2.0.10.3 — budget extended from 60 s → 180 s. The pre-fix
   *  timeout was set when Docling was assumed to be the only slow
   *  stage; in practice a large PDF + BGE-M3 embedding routinely
   *  runs 90-180 s end-to-end (the e2e suite uses a 180 s budget).
   *  Once the timeout was hit the sidebar gave up without
   *  scheduling a follow-up refresh, so the user saw a stuck
   *  "向量化中(N 块)" chip until F5 — exactly the bug this version
   *  closes by returning ``false`` and letting the caller schedule
   *  a fallback refresh.
   *
   *  v2.0.10.4 — the v2.0.10.3 fix was insufficient: the polling
   *  could exit *prematurely* (returning ``reachedTerminal=true``)
   *  on the FIRST tick when ``mine === undefined`` — the race
   *  where the backend's background task hadn't finished its
   *  ``doc_registry.register`` call yet. The pre-fix code defaulted
   *  ``status = mine?.status ?? "indexed"`` — which silently turned
   *  "doc not in list yet" into "indexed", exiting the loop before
   *  the doc ever appeared. The post-polling ``listDocuments`` then
   *  caught the doc mid-``"embedding"`` state, the chip rendered
   *  "向量化中(N 块)", and because ``reachedTerminal===true`` the
   *  fallback refresh was NEVER scheduled. The user saw a stuck
   *  chip until manual F5. The fix is twofold:
   *    (a) Don't assume ``"indexed"`` when ``mine`` is undefined —
   *        just keep polling until the doc appears or the budget
   *        expires. The first ``listDocuments`` that finds the doc
   *        will see its real status and exit normally.
   *    (b) Surface every poll's fresh ``listDocuments`` response to
   *        the caller via ``onDocuments`` so the docs state mirrors
   *        the backend view in real time, not just on the post-poll
   *        snapshot. This means even if some other edge case exits
   *        the polling early, the chip will already reflect whatever
   *        status the backend reported at that tick. */
  async function pollUntilIndexed(
    threadId: string,
    docId: string,
    onChunkCount?: (chunkCount: number) => void,
    onDocuments?: (docs: Document[]) => void,
  ): Promise<boolean> {
    const deadline = Date.now() + 180_000;
    while (Date.now() < deadline) {
      try {
        const docs = await listDocuments(threadId);
        const mine = docs.find((d) => d.doc_id === docId);
        const cc = mine?.chunk_count ?? 0;
        // v2.0.10.4 — DO NOT default ``status`` to ``"indexed"`` when
        // ``mine`` is undefined. The previous default turned the
        // brief register-race window (where the backend hadn't yet
        // surfaced the new doc) into a false-positive terminal exit,
        // stranding the chip at "向量化中" forever. We now leave
        // ``status`` undefined and just continue the loop; the next
        // tick will see the real status once ``register`` lands.
        const status = mine?.status;
        onChunkCount?.(cc);
        // v2.0.10.4 — surface the full docs list to the caller so the
        // sidebar's docs state mirrors the backend view in real time,
        // not just on the post-polling snapshot. This means the chip
        // flips from "解析中" → "向量化中" → "X 块" live, instead of
        // jumping only at the end of the polling loop.
        onDocuments?.(docs);
        // Terminal statuses: ingestion has ended (success or failure).
        // - "indexed": chunks landed in LanceDB
        // - "empty": parser ran but produced no chunks
        // - "failed": parser / persist threw (error_message populated)
        // ``status === undefined`` (doc not yet in list) is NOT a
        // terminal state — keep polling until it appears or the
        // budget expires. This is the v2.0.10.4 belt-and-suspenders
        // for the register-race window.
        if (
          status === "indexed" ||
          status === "empty" ||
          status === "failed"
        ) {
          return true;
        }
      } catch {
        // network blip; keep polling
      }
      await new Promise((r) => setTimeout(r, 500));
    }
    return false;
  }

  /** v2.0.10.3 / v2.0.10.4 — slow-rate periodic doc-list refresh used
   *  as a fallback after ``pollUntilIndexed`` exits. The intent is
   *  to catch the moment the backend eventually flips the doc to
   *  a terminal status so the user sees the correct chip without
   *  a manual F5.
   *
   *  v2.0.10.4 — generalized: ``tickMs`` and ``capMs`` are now
   *  parameters so callers can pick the appropriate cadence. The
   *  fast path (5 s tick / 60 s cap) closes the register-race window
   *  that v2.0.10.3 left open; the slow path (15 s / 5 min) is
   *  retained for the genuine >180 s budget-expiration case where
   *  the backend might actually be slow.
   *
   *  Behavior:
   *    - Refetches every ``tickMs`` (vs 500 ms during the active
   *      poll) so the cost is negligible.
   *    - On every tick, replaces the documents state with the
   *      fresh list so any status flip is reflected immediately.
   *    - Clears the interval the moment the doc reaches a terminal
   *      status, OR after ``capMs``, whichever comes first.
   *    - The interval id is tracked in ``fallbackRefreshRef`` so
   *      unmount can clean up everything (avoids the React
   *      "state update on unmounted component" warning if the
   *      user navigates away during a long ingestion).
   *    - Idempotent on terminal state: if the doc is already
   *      terminal on the first tick, the helper fires one
   *      ``listDocuments`` (to keep the docs state fresh) then
   *      immediately self-stops. Cheap (~50-200 ms per call) and
   *      harmless if scheduled redundantly. */
  function scheduleFallbackRefresh(
    threadId: string,
    docId: string,
    tickMs: number = 15_000,
    capMs: number = 300_000,
  ) {
    // Defensive: don't even start the fallback if the doc is
    // already terminal (e.g., the polling observed it but the
    // caller still scheduled us — paying 50 ms for one
    // ``listDocuments`` and self-stopping is fine, but we can save
    // it by checking the latest known docs state).
    const startedAt = Date.now();
    const intervalId = setInterval(() => {
      const elapsed = Date.now() - startedAt;
      if (elapsed >= capMs) {
        clearInterval(intervalId);
        fallbackRefreshRef.current.delete(intervalId);
        return;
      }
      listDocuments(threadId)
        .then((docs) => {
          setDocuments(docs);
          const mine = docs.find((d) => d.doc_id === docId);
          const status = mine?.status;
          if (
            status === "indexed" ||
            status === "empty" ||
            status === "failed"
          ) {
            // Doc reached a terminal state — stop the fallback.
            clearInterval(intervalId);
            fallbackRefreshRef.current.delete(intervalId);
          }
        })
        .catch(console.error);
    }, tickMs);
    fallbackRefreshRef.current.add(intervalId);
  }

  async function confirmDeleteDoc() {
    if (!pendingDocDelete) return;
    const docId = pendingDocDelete.doc_id;
    setPendingDocDelete(null);
    try {
      await deleteDocument(docId);
    } catch (e) {
      // v2.0.26.3 (PR-6) — toast replaces PR-2's inline banner.
      // The localized message is the same string we used for the
      // banner (kept in the i18n catalog as ``sidebar.deleteFailedBody``
      // for back-compat with any external link / screenshot).
      // eslint-disable-next-line no-console
      console.error("deleteDocument failed", e);
      showToast({
        kind: "error",
        message: t("sidebar.deleteFailedBody", { filename: pendingDocDelete.filename }),
      });
      return;
    }
    listDocuments(threadId).then(setDocuments).catch(console.error);
    listSessions().then(setSessions).catch(console.error);
  }

  /** Files are tightly bound to the conversation that uploaded them —
   *  deleting the conversation ALWAYS clears its indexed chunks too.
   *  The dialog no longer offers "keep documents" because there's no
   *  use case for orphans; documents are never referenced from any
   *  thread other than the one they were uploaded into. */
  async function confirmDeleteSession() {
    if (!pendingSessionDelete) return;
    const id = pendingSessionDelete.thread_id;
    const sessionLabel = pendingSessionDelete.title ?? id;
    setPendingSessionDelete(null);
    try {
      await deleteSession(id);
    } catch (e) {
      // v2.0.26.3 (PR-6) — toast replaces PR-2's inline banner.
      // Same shape as ``confirmDeleteDoc`` above; we do NOT fall
      // through to ``clearThreadDocuments`` because if the session
      // still exists server-side, those documents are still tied
      // to it (backend is source of truth for that relationship).
      // eslint-disable-next-line no-console
      console.error("deleteSession failed", e);
      showToast({
        kind: "error",
        message: t("sidebar.deleteSessionFailedBody", { title: sessionLabel }),
      });
      return;
    }
    try {
      await clearThreadDocuments(id);
    } catch (e) {
      console.warn("clearThreadDocuments failed", e);
    }
    if (id === threadId) {
      // Deleting the active session lands the user on a fresh empty
      // chat. Routed through ``onNewChat`` so the parent can register
      // the freshly-minted UUID and ``loadHistory`` skips the wasted
      // GET — same reason as the "+ 新对话" button.
      onNewChat();
    }
    listSessions().then(setSessions).catch(console.error);
    listDocuments(threadId).then(setDocuments).catch(console.error);
  }

  function newChat() {
    onNewChat();
  }

  function selectExistingSession(id: string) {
    onSelectThread(id);
    window.dispatchEvent(new CustomEvent("rag:session-updated"));
    // v2.0.26.2 (PR-5) — close the mobile drawer after navigating
    // to a session so the chat pane is immediately usable. On
    // desktop this is a no-op (the drawer is always "open" as a
    // fixed sidebar) but the cleanup matters for the mobile flow.
    onCloseDrawer();
  }

  // Builds the upload button's hover tooltip from server-provided
  // ``label_zh`` strings. Falls back to a generic message while the
  // ``/formats`` request is still in flight (or if it fails) so the
  // button still works on a cold start.
  const UPLOAD_HELPER_TITLE = formats
    ? t("sidebar.dropHint", {
        formats: formats.formats.map((f) => f.label_zh).join("、"),
      })
    : t("sidebar.dropHintGeneric");
  const ACCEPT_ATTR = formats ? formats.extensions.join(",") : undefined;

  function pickFile() {
    const input = document.createElement("input");
    input.type = "file";
    if (ACCEPT_ATTR) input.accept = ACCEPT_ATTR;
    input.onchange = () => {
      const file = input.files?.[0];
      if (file) handleFile(file);
    };
    input.click();
  }

  return (
    <>
      {/* v2.0.26.2 (PR-5) — mobile backdrop. Sits behind the
          drawer (z-index 89) and intercepts taps to close it.
          ``pointer-events: none`` while closed so the rest of the
          UI stays interactive. */}
      <div
        className={`sidebar-backdrop ${drawerOpen ? "is-open" : ""}`}
        onClick={onCloseDrawer}
        aria-hidden="true"
      />
      <aside
        id="primary-sidebar"
        className={`sidebar ${drawerOpen ? "is-open" : ""}`}
        aria-label={t("sidebar.toggleAria")}
      >
      {/* v2.0.26.3 (PR-6) — the inline sidebar-error-banner was
          DELETED here. Destructive-action failures now surface as
          toasts at the app root (mounted via ``<ToastViewport>``
          inside ``main.tsx``). The i18n keys
          ``sidebar.deleteFailedBody`` / ``sidebar.deleteSessionFailedBody``
          / ``sidebar.deleteErrorDismiss`` are kept in the catalog
          (toast text + back-compat) but the banner JSX is gone. */}
      {/* Brand row — always visible at the top. Logo + name +
          settings gear. Keeps the product name consistent across the
          tab title, sidebar, and dialogs. */}
      <div className="sidebar-brand">
        <Logo size={28} />
        <div className="sidebar-brand-title">Yuan RAG</div>
        <button
          className="sidebar-icon-btn"
          onClick={onOpenSettings}
          title={t("sidebar.openSettings")}
          aria-label={t("sidebar.openSettings")}
        >
          ⚙
        </button>
      </div>

      {/* Primary actions — always visible. The upload button here is
          the deliberate fix for the "long session list pushes upload
          off-screen" bug: keeping it pinned in the sidebar header
          guarantees one-click access regardless of how many sessions
          the user has accumulated. */}
      <div className="sidebar-actions">
        <button className="btn-primary sidebar-new-chat" onClick={newChat}>
          <span className="sidebar-new-chat-plus">+</span>
          <span>{t("sidebar.newChat")}</span>
        </button>
        <button
          className={`sidebar-upload-btn ${dragging ? "dragging" : ""}`}
          onClick={pickFile}
          title={UPLOAD_HELPER_TITLE}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const file = e.dataTransfer.files[0];
            if (file) handleFile(file);
          }}
        >
          <span className="sidebar-upload-icon">📎</span>
          <span>{dragging ? t("sidebar.dropActive") : t("sidebar.dropIdle")}</span>
        </button>
      </div>

      {/* Sessions list — middle scrollable area. */}
      <div className="sidebar-section sidebar-sessions">
        <div className="section-title">{t("sidebar.historyTitle")}</div>
        {sessions.map((s) => (
          <div
            key={s.thread_id}
            className={`session-item ${s.thread_id === threadId ? "active" : ""}`}
            onClick={() => selectExistingSession(s.thread_id)}
          >
            <div className="title">
              <span className="session-item-title-text">{s.title}</span>
              {s.doc_count ? (
                <span className="session-item-count">({s.doc_count})</span>
              ) : null}
            </div>
            <button
              className="session-item-delete"
              onClick={(e) => {
                e.stopPropagation();
                setPendingSessionDelete(s);
              }}
              title={t("sidebar.delete")}
              aria-label={t("sidebar.delete")}
            >
              ×
            </button>
          </div>
        ))}
        {sessions.length === 0 && (
          <div className="empty-hint">{t("sidebar.emptyHistoryDetailed")}</div>
        )}
      </div>

      {/* Documents — bottom panel, scrolls internally when many files
          are uploaded so it never grows past its cap. Visible at all
          times so the user can see what's indexed for the active
          conversation without scrolling back up. Upload-progress rows
          (spinner + status + bar) appear ABOVE the indexed docs so
          they read as "in-flight work" rather than "documents". */}
      <div className="sidebar-section sidebar-documents">
        <div className="section-title">
          {t("sidebar.threadDocsTitle")}
          {documents.length + uploadProgress.length > 0 && (
            <span className="section-count">
              {" "}
              · {documents.length}
              {uploadProgress.length > 0
                ? t("sidebar.uploadProgressSuffix", {
                    count: uploadProgress.length,
                  })
                : ""}
            </span>
          )}
        </div>

        {/* In-flight uploads first. Each shows a spinner, the current
            stage, and (when chunks start landing) the chunk count. */}
        {uploadProgress.map((p) => {
          const statusText =
            p.stage === "uploading"
              ? t("sidebar.uploadingDoc")
              : p.stage === "parsing"
                ? t("sidebar.parsingDocDetailed")
                : t("sidebar.indexingChunks", { count: p.chunkCount });
          // Bar width is fixed once we know the chunk count (so the
          // user gets a real sense of progress); indeterminate until
          // then. Cap at 99 % so the bar never quite fills — the final
          // drop-to-doc-list signals completion.
          const widthStyle =
            p.stage === "indexing" && p.chunkCount > 0
              ? { animation: "none", width: "99%", marginLeft: "0%" }
              : undefined;
          return (
            <div key={p.tempId} className="upload-progress">
              <div className="upload-progress-row">
                <span className="upload-progress-spinner" aria-hidden="true" />
                <span className="upload-progress-filename" title={p.filename}>
                  {p.filename}
                </span>
              </div>
              <div className="upload-progress-status">{statusText}</div>
              <div className="upload-progress-bar">
                <div
                  className="upload-progress-bar-fill"
                  style={widthStyle}
                />
              </div>
            </div>
          );
        })}

        {documents.map((d) => {
          // Status-aware metadata: indexed → chunk count, the rest →
          // a colored chip so the user can see at a glance which docs
          // failed extraction. The full error message surfaces as a
          // tooltip on hover.
          const status = d.status ?? "indexed";
          let meta: React.ReactNode;
          let itemTitle = d.source_path;
          if (status === "indexed") {
            meta = (
              <span className="doc-item-meta">
                {t("sidebar.chunkCountSuffix", { count: d.chunk_count })}
              </span>
            );
          } else if (status === "empty") {
            itemTitle = d.error_message
              ? `${d.filename}\n\n${d.error_message}`
              : d.filename;
            meta = (
              <span
                className="doc-item-status doc-item-status-empty"
                title={d.error_message || t("sidebar.emptyDocText")}
              >
                {t("sidebar.emptyDocText")}
              </span>
            );
          } else if (status === "failed") {
            itemTitle = d.error_message
              ? `${d.filename}\n\n${d.error_message}`
              : d.filename;
            meta = (
              <span
                className="doc-item-status doc-item-status-failed"
                title={d.error_message || t("sidebar.emptyParse")}
              >
                {t("sidebar.emptyParse")}
              </span>
            );
          } else if (status === "parsing") {
            // v2.0.5 — per-stage chip while the parser is the active
            // worker. Replaces the generic "处理中" with a stage-
            // specific label so the user can see where the 30-90 s
            // ingestion budget is going.
            meta = (
              <span
                className="doc-item-status doc-item-status-pending"
                title={t("sidebar.docTitleParsing")}
              >
                <span className="doc-item-status-spinner" aria-hidden="true" />
                {t("sidebar.parsing")}
              </span>
            );
          } else if (status === "embedding") {
            // v2.0.5 — per-stage chip while BGE-M3 is embedding the
            // parsed chunks. Shows the running chunk count so the
            // user sees progress (the embedder processes all chunks
            // in one batch today, but the field is forward-
            // compatible with chunk-batched progress).
            meta = (
              <span
                className="doc-item-status doc-item-status-pending"
                title={t("sidebar.docTitleEmbedding")}
              >
                <span className="doc-item-status-spinner" aria-hidden="true" />
                {t("sidebar.embedding", { count: d.chunk_count })}
              </span>
            );
          } else {
            // "pending" — visible only during the brief window between
            // upload and the parser starting. Once the parser kicks
            // off the status flips to "parsing" (see above).
            meta = (
              <span
                className="doc-item-status doc-item-status-pending"
                title={t("sidebar.docTitleQueued")}
              >
                <span className="doc-item-status-spinner" aria-hidden="true" />
                {t("sidebar.processing")}
              </span>
            );
          }
          return (
            <div key={d.doc_id} className="doc-item" title={itemTitle}>
              <div className="filename" title={itemTitle}>
                {d.filename}
              </div>
              {meta}
              <button
                className="doc-item-delete"
                onClick={() => setPendingDocDelete(d)}
                title={t("sidebar.removeDocAria")}
                aria-label={t("sidebar.removeDocAria")}
              >
                ×
              </button>
            </div>
          );
        })}
        {documents.length === 0 && uploadProgress.length === 0 && (
          <div className="empty-hint">{t("sidebar.emptyDocsDetailed")}</div>
        )}
      </div>

      {/* ─── Confirmation dialogs ─── */}

      {pendingDocDelete && (
        <ConfirmDialog
          title={t("sidebar.deleteDocTitle")}
          message={
            <>
              {t("sidebar.deleteDocBody1", {
                filename: pendingDocDelete.filename,
              })}
              <br />
              {t("sidebar.deleteDocBody2")}
            </>
          }
          confirmLabel={t("sidebar.deleteDocLabel")}
          cancelLabel={t("confirm.cancel")}
          variant="danger"
          onConfirm={confirmDeleteDoc}
          onCancel={() => setPendingDocDelete(null)}
        />
      )}

      {/* Session delete — files are deeply bound to the conversation,
          so deleting the conversation ALWAYS removes its indexed
          chunks too. Single-confirm dialog, no "keep files" option. */}
      {pendingSessionDelete && (
        <ConfirmDialog
          title={t("sidebar.deleteSessionLabel")}
          message={
            <>
              {t("sidebar.deleteSessionBody1", {
                title: pendingSessionDelete.title,
              })}
              <br />
              {t("sidebar.deleteSessionBody2")}
            </>
          }
          confirmLabel={t("sidebar.deleteSessionAndDocsLabel")}
          cancelLabel={t("confirm.cancel")}
          variant="danger"
          onConfirm={confirmDeleteSession}
          onCancel={() => setPendingSessionDelete(null)}
        />
      )}

      {uploadError && (
        <ConfirmDialog
          title={t("sidebar.uploadErrorTitle")}
          message={t("sidebar.uploadErrorBody", { error: uploadError })}
          confirmLabel={t("confirm.dismiss")}
          cancelLabel={t("confirm.cancel")}
          variant="warning"
          onConfirm={() => setUploadError(null)}
          onCancel={() => setUploadError(null)}
        />
      )}
    </aside>
    </>
  );
}

// Wrap in React.memo. Sidebar doesn't read ``messagesByThread`` but it
// lives in the same tree as ChatPane, and every App re-render (e.g.
// every streamed token from any thread) would otherwise re-run its
// body, re-render every session / document row, and recreate the
// inline-style objects inside ``documents.map``. With App.tsx now
// passing stable ``setThreadId`` and ``useCallback`` wrappers for the
// other callbacks, shallow-equality memo is enough to skip renders
// when none of Sidebar's actual inputs changed.
export const Sidebar = memo(SidebarInner);