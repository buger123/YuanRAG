/**
 * REST API client. The base URL is the same origin the SPA is loaded from.
 */
import type { ModelStatus } from "../i18n/status";
import { apiError } from "../i18n/errors";
import type { Source, ToolCall } from "../chat-utils";

export interface Document {
  doc_id: string;
  filename: string;
  source_type: string;
  source_path: string;
  chunk_count: number;
  ingested_at: string;
  thread_id?: string;
  /**
   * Ingestion status from the backend ``doc_registry``:
   * - ``indexed`` — at least one chunk landed in the vector store.
   * - ``empty`` — parsing finished but no text was extracted (typical
   *   for image-only PDFs where both Docling's OCR and the pypdfium2
   *   fallback returned nothing). The doc still appears here so the
   *   user can see and delete it; the sidebar shows a "未提取到文本"
   *   chip instead of a chunk count.
   * - ``failed`` — ingestion raised an exception (e.g. corrupt PDF).
   * - ``pending`` — between upload and the parser starting.
   * - ``parsing`` (v2.0.5) — ``run_ingestion`` is the active worker.
   *   The sidebar renders "解析中…" while Docling is processing.
   * - ``embedding`` (v2.0.5) — parser finished, BGE-M3 is computing
   *   vectors. The sidebar renders "向量化中 (N 块)" while the
   *   embedder is the active worker.
   *
   * Pre-v2.0.5 the sidebar only distinguished pending / indexed /
   * empty / failed; the parser and embedder both fell under the
   * generic "处理中" chip. The split lets the frontend surface
   * where the 30-90 s ingestion budget is going.
   *
   * Defaults to ``indexed`` for backward compatibility with older
   * backend responses that don't include the field.
   */
  status?:
    | "pending"
    | "parsing"
    | "embedding"
    | "indexed"
    | "empty"
    | "failed";
  error_message?: string;
}

export interface Settings {
  llm_provider: "openai" | "anthropic";
  llm_model: string;
  has_api_key: boolean;
  api_key_source: "env" | "keyring" | "none";
}

export interface SessionSummary {
  thread_id: string;
  title: string;
  updated_at: string;
  message_count: number;
  doc_count?: number;
}

/** A single persisted chat message. Mirrors ``MessageRecord`` in
 * ``src/api/schemas.py``. ``sources`` is populated for assistant turns
 * that emitted citations — the chat pane rehydrates the inline `[n]`
 * chips from this when the user reopens an old conversation.
 * ``reasoning`` is the accumulated LLM thinking text (Anthropic
 * extended-thinking blocks) — also persisted since v0.4, so the 💭
 * drawer survives a page reload / thread switch, not just the live
 * stream. ``undefined`` for older messages and for providers that
 * don't emit thinking blocks.
 *
 * ``source_kinds`` (v1.1.15) carries the aggregated "what kinds of
 * sources shaped this answer" — e.g. ``["local"]`` for a doc-only
 * turn, ``["web"]`` for a web-search-only turn, ``["local","web"]``
 * for the v1.1.13 complementary path, ``[]`` for greetings / direct.
 * ``undefined`` for messages persisted before v1.1.15 — the frontend
 * falls back to no banner rather than guessing from sources.
 *
 * ``tool_calls`` (v2.0) — rehydrated into per-message ToolCall
 * cards. Each record has ``id``, ``name``, ``args``, ``result``,
 * ``ok``, ``step``, ``elapsed_ms``, ``started_at``, ``ended_at``.
 * ``undefined`` for messages persisted before v2.0 (no tool cards
 * on history reload).
 *
 * ``created_at`` (v2.0) — server-side ISO 8601 timestamp on the
 * turn. Both user and assistant messages carry it (the runner stamps
 * it on the HumanMessage in ``_initial_state`` and on the AIMessage
 * in ``react_generate`` / ``react_generate_direct``). ``undefined``
 * for messages persisted before v2.0. */
export interface MessageRecord {
  role: "user" | "assistant" | "system";
  content: string;
  /** v2.0.7 SoT-4: tightened from
   *  ``Array<Record<string, unknown>>`` to ``Array<Source>`` —
   *  the wire payload shape was always ``Source`` (see
   *  ``src.api.schemas.Source``) so the unknown catch-all was a
   *  silent contract violator. Optional because older persisted
   *  rows may pre-date the field. */
  sources?: Array<Source>;
  reasoning?: string;
  source_kinds?: string[];
  /** v2.0.7 SoT-4: tightened from ``Array<Record<string, unknown>>``
   *  to ``Array<ToolCall>``. The wire shape is the same — see
   *  ``src.agent.state.ToolCallRecord``. */
  tool_calls?: Array<ToolCall>;
  created_at?: string;
}

const BASE = ""; // same-origin

async function jsonOrThrow<T>(resp: Response): Promise<T> {
  if (!resp.ok) {
    // QW #8: surface a localized ApiError instead of the raw
    // ``<status> <statusText>: <body>`` string. The error carries
    // the status code + body text for callers that want to
    // introspect (e.g. show retry guidance for 5xx), but the
    // default ``.message`` is already user-facing Chinese.
    const text = await resp.text().catch(() => "");
    throw apiError(resp.status, text);
  }
  return resp.json();
}

/**
 * QW #8: wrap ``fetch`` so network-layer failures (TypeError from
 * "Failed to fetch" / "Load failed") get the same localized
 * treatment as HTTP errors. ``fetch`` throws on DNS failure,
 * offline mode, CORS preflight failure, and the like — none of
 * those reach ``jsonOrThrow`` so we need a separate catcher.
 */
async function fetchOrThrow(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  try {
    // v1.1.4 — was ``fetchOrThrow(input, init)`` which calls itself
    // instead of ``fetch``, so every REST call through this wrapper
    // stack-overflowed on first invocation. The wrapper exists to
    // translate network-layer failures (TypeError from "Failed to
    // fetch" / "Load failed" on DNS / offline / CORS preflight
    // errors) into the same localized ``apiError`` shape as HTTP
    // errors, since those don't reach ``jsonOrThrow``.
    //
    // v2.0.25.1 P0-extension — ``init.signal`` is forwarded so callers
    // can attach an ``AbortController`` and cancel the request when
    // the result is no longer needed (e.g. switching to a different
    // thread mid-load). Without this, stale fetches keep running and
    // their result lands on top of the now-active thread's history.
    // ``fetch`` already throws ``AbortError`` on abort; we re-throw as
    // an ``apiError(0, undefined, err)`` so the caller can check
    // ``err.aborted`` or string-match for "abort" — keeping the
    // ``console.error`` happy rather than logging a noise line per
    // user-initiated abort.
    return await fetch(input, init);
  } catch (e) {
    throw apiError(0, undefined, e);
  }
}

export async function getSettings(): Promise<Settings> {
  return jsonOrThrow(await fetchOrThrow(`${BASE}/settings`));
}

export async function updateSettings(payload: {
  llm_provider?: string;
  llm_model?: string;
}): Promise<Settings> {
  return jsonOrThrow(
    await fetchOrThrow(`${BASE}/settings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  );
}

export async function listDocuments(threadId?: string): Promise<Document[]> {
  const q = threadId ? `?thread_id=${encodeURIComponent(threadId)}` : "";
  return jsonOrThrow(await fetchOrThrow(`${BASE}/documents${q}`));
}

export async function deleteDocument(docId: string): Promise<void> {
  await jsonOrThrow(
    await fetchOrThrow(`${BASE}/documents/${docId}`, { method: "DELETE" }),
  );
}

export async function uploadDocument(
  file: File,
  threadId?: string,
): Promise<{ doc_id: string; filename: string; thread_id?: string }> {
  const form = new FormData();
  form.append("file", file);
  const q = threadId ? `?thread_id=${encodeURIComponent(threadId)}` : "";
  return jsonOrThrow(
    await fetchOrThrow(`${BASE}/documents/upload${q}`, { method: "POST", body: form }),
  );
}

export async function clearThreadDocuments(threadId: string): Promise<void> {
  await jsonOrThrow(
    await fetchOrThrow(`${BASE}/documents/clear-thread/${encodeURIComponent(threadId)}`, {
      method: "POST",
    }),
  );
}

export async function listSessions(): Promise<SessionSummary[]> {
  return jsonOrThrow(await fetchOrThrow(`${BASE}/sessions`));
}

export async function deleteSession(threadId: string): Promise<void> {
  await jsonOrThrow(
    await fetchOrThrow(`${BASE}/sessions/${threadId}`, { method: "DELETE" }),
  );
}

/**
 * Replay the persisted message history for a thread. Called from
 * ChatPane whenever `threadId` changes so clicking an old session in
 * the sidebar actually shows the conversation that was on it, rather
 * than an empty pane that only fills in once the user sends a new
 * message (which was the previous behavior and the source of the
 * "session history is broken" bug).
 */
export async function getSessionMessages(
  threadId: string,
  // v2.0.25.1 P0-extension — callers pass an AbortController's signal
  // so thread switches can cancel the previous fetch. ``fetch``
  // natively respects it (rejects with AbortError on abort). See
  // ``App.tsx:loadHistory`` for the per-thread controller map.
  signal?: AbortSignal,
): Promise<MessageRecord[]> {
  return jsonOrThrow(
    await fetchOrThrow(
      `${BASE}/sessions/${encodeURIComponent(threadId)}/messages`,
      signal ? { signal } : undefined,
    ),
  );
}

export async function getModelsStatus(): Promise<{
  embedding: { name: string; status: ModelStatus };
  reranker: { name: string; status: ModelStatus };
  ready: boolean;
}> {
  return jsonOrThrow(await fetchOrThrow(`${BASE}/models/status`));
}

/**
 * v2.0.30.0 — long-poll for model readiness.
 *
 * Calls ``GET /models/wait?timeout=N`` once and resolves when the
 * backend reports both embedder + reranker ready (200 + ready=true),
 * or rejects on a non-2xx (typically 408 timeout). On timeout the
 * caller is expected to fall back to ``getModelsStatus`` polling.
 *
 * ``AbortSignal`` hooks into React unmount + tab visibility so we
 * don't keep a long-poll in flight once the result is no longer
 * needed. The server-side deadline still applies if the caller
 * forgets to abort — no infinite-hang risk.
 */
export async function waitForModelsReady(
  timeoutSeconds: number = 60,
  signal?: AbortSignal,
): Promise<{
  embedding: { name: string; status: ModelStatus };
  reranker: { name: string; status: ModelStatus };
  ready: boolean;
}> {
  const url = `${BASE}/models/wait?timeout=${encodeURIComponent(String(timeoutSeconds))}`;
  return jsonOrThrow(
    await fetchOrThrow(url, signal ? { signal } : undefined),
  );
}

export async function triggerModelDownload(): Promise<void> {
  await fetchOrThrow(`${BASE}/models/download`, { method: "POST" });
}

/**
 * Single source of truth for accepted upload formats. Sourced from the
 * backend's ``GET /formats`` so the frontend never hard-codes a list that
 * can drift from what the backend actually accepts. Returned by
 * ``src/api/routes/formats.py``.
 */
export interface SupportedFormats {
  extensions: string[];
  labels: Record<string, string>;
  formats: Array<{
    format: string;
    extensions: string[];
    label_zh: string;
  }>;
}

export async function listFormats(): Promise<SupportedFormats> {
  return jsonOrThrow(await fetchOrThrow(`${BASE}/formats`));
}
