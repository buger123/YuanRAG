import { memo, useEffect, useId, useRef, useState } from "react";
import type { ChatMessage } from "../chat-utils";
import { Logo } from "./Logo";
import { Markdown } from "../Markdown";
import { StreamingMarkdown } from "../StreamingMarkdown";
import { useLocale } from "../i18n";
import { ToolCallCard } from "./ToolCallCard";

interface Props {
  threadId: string;
  messages: ChatMessage[];
  isStreaming: boolean;
  historyLoaded: boolean;
  webSearching: boolean;
  /** v2.0 — current ReAct step (1-based). 0 = idle. Rendered
   *  as "Step N" next to the "正在思考" pill while >0. */
  currentStep: number;
  modelsReady: boolean;
  /** Called when the user submits a non-empty input. The parent
   *  (App.tsx) handles optimistic UI, WebSocket dispatch, and
   *  routing the agent's response events back into this thread's
   *  buffer. */
  onSend: (text: string) => void;
  /** QW #4: called when the user clicks the "停止生成" button
   *  mid-stream. The parent (App.tsx) closes the WS so the drain
   *  mechanism in P1-3 keeps the backend run alive for history
   *  while the client immediately stops listening for tokens. */
  onStop: () => void;
  // v2.0.26.2 (PR-5) — mobile drawer state mirrored into the
  // chat pane so the hamburger can read its current value for
  // aria-expanded. ``onOpenDrawer`` is invoked on tap.
  drawerOpen: boolean;
  onOpenDrawer: () => void;
  // v2.0.29.9 (Phase 8) — per-thread verbatim extraction toggle.
  // ``"auto"`` (default) lets the backend run hybrid detect (regex
  // + cheap-LLM); ``"on"`` forces extractive mode (verbatim quote,
  // no rewriting); ``"off"`` forces normal synthesis EVEN if the
  // backend detect fires (per user decision 2026-09-28, user OFF
  // always wins). Forwarded via ``App.sendToThread`` to the WS
  // payload's ``high_precision`` field.
  highPrecision: "auto" | "on" | "off";
  /** Called when the user clicks one of the segmented buttons. */
  onHighPrecisionChange: (value: "auto" | "on" | "off") => void;
}

/** Collapsible drawer that renders an assistant message's accumulated
 *  reasoning text. Per-message collapse state lives in this component
 *  (default expanded) and is intentionally NOT persisted — the global
 *  ``showThinking`` flag in the parent controls visibility across
 *  messages, while this drawer only handles individual fold/unfold. */
function ThinkingDrawer({ text }: { text: string }) {
  // v2.0.6 P2-4 — translate the drawer header label. ``ThinkingDrawer``
  // is rendered outside ``ChatPaneInner`` (it's a per-message
  // component), so it gets its own ``useLocale`` call.
  const { t } = useLocale();
  const [open, setOpen] = useState(true);
  // v2.0.26.2 (PR-5) — QW #13 carryover: ARIA plumbing for the
  // collapsible drawer so screen readers can announce expanded vs
  // collapsed state. ``useId`` produces a per-instance id that
  // stays stable across renders (unlike an index or counter).
  const bodyId = useId();
  return (
    <div className="thinking-drawer">
      <div
        className="thinking-drawer-header"
        role="button"
        tabIndex={0}
        aria-expanded={open}
        aria-controls={bodyId}
        onClick={() => setOpen((v) => !v)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            setOpen((v) => !v);
          }
        }}
        title={open ? t("chat.thinkingDrawerCollapse") : t("chat.thinkingDrawerExpand")}
      >
        <span className="thinking-drawer-chevron">{open ? "▾" : "▸"}</span>
        <span className="thinking-drawer-icon">💭</span>
        <span>{t("chat.toggleThinking")}</span>
      </div>
      {open && (
        <div id={bodyId} className="thinking-drawer-body" role="region">
          {text}
        </div>
      )}
    </div>
  );
}

/** Compact, locale-friendly timestamp for the per-message footer.
 *
 *  Server stamps ISO 8601 UTC (e.g. ``"2026-09-06T14:23:11+00:00"``);
 *  we render a small "MM-DD HH:MM" so the timestamp doesn't
 *  dominate the message bubble. Falls back to the raw input on
 *  parse failure so a malformed timestamp can't break the layout. */
function formatTimestamp(iso: string): string {
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(
    d.getHours(),
  )}:${pad(d.getMinutes())}`;
}

function ChatPaneInner({
  threadId,
  messages,
  isStreaming,
  historyLoaded,
  webSearching,
  currentStep,
  modelsReady,
  onSend,
  onStop,
  drawerOpen,
  onOpenDrawer,
  highPrecision,
  onHighPrecisionChange,
}: Props) {
  const { t } = useLocale();
  const [input, setInput] = useState("");
  // Auto-grow the composer textarea as the user types. Set to ``auto``
  // first to read the natural content height, then clamp to a max
  // (handled by CSS ``max-height`` + the MIN/MAX constants below).
  // Without the ``auto`` reset the textarea would only grow and never
  // shrink when the user deletes lines.
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  // Bumped from 200 → 380 px so multi-paragraph questions grow line by
  // line (most natural) and only fall back to the inner scrollbar for
  // very long pasted content. The hard ceiling is in CSS at the same
  // 380 px; the JS clamp below just keeps height === scrollHeight so
  // the box always fits its content.
  const COMPOSER_MAX_HEIGHT = 380;
  // ``showThinking`` controls whether the per-message reasoning drawer
  // renders. Persisted to localStorage so the preference survives page
  // reloads (default ON; only "false" in storage disables it). Read at
  // mount; written on toggle.
  const [showThinking, setShowThinking] = useState<boolean>(() => {
    return localStorage.getItem("rag.show_thinking") !== "false";
  });
  const messagesRef = useRef<HTMLDivElement>(null);
  // True while the user is parked at (or very near) the bottom of the
  // message list. We only auto-scroll on new content when this is
  // true — otherwise scrolling up to read history would yank the
  // viewport back down on every token that streams in. The flag flips
  // back to true the moment the user scrolls down again.
  const stickToBottomRef = useRef(true);
  // We track the previously-seen threadId so we can detect a thread
  // switch and force-scroll-to-bottom on the new thread (regardless
  // of where the user was scrolled on the old thread).
  const prevThreadRef = useRef(threadId);

  useEffect(() => {
    if (prevThreadRef.current !== threadId) {
      prevThreadRef.current = threadId;
      stickToBottomRef.current = true;
      const el = messagesRef.current;
      if (el) el.scrollTop = el.scrollHeight;
    }
  }, [threadId]);

  useEffect(() => {
    if (!stickToBottomRef.current) return;
    const el = messagesRef.current;
    if (!el) return;
    // Direct scrollTop assignment — smooth-scroll on every streamed
    // token kicks off competing animations and thrashes the compositor.
    el.scrollTop = el.scrollHeight;
  }, [messages]);

  // Auto-grow the composer. Runs whenever the input string changes
  // (typing, paste, clear-after-send). Setting ``height = "auto"``
  // first collapses the box to its intrinsic size so the next line
  // can be detected as "new" rather than fitting inside the existing
  // scrollHeight (which would otherwise plateau).
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    // ``scrollHeight`` already includes padding, so this is the true
    // visible content height — no extra padding math needed.
    const next = Math.min(el.scrollHeight, COMPOSER_MAX_HEIGHT);
    el.style.height = `${next}px`;
  }, [input]);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || isStreaming) return;
    // Refuse to send while history is still being fetched for the
    // current thread — otherwise the optimistic user message in state
    // can race with the history response.
    if (!historyLoaded) return;
    setInput("");
    onSend(text);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    // Enter to send, Shift+Enter to insert newline. Without this guard,
    // the textarea would either insert a newline on plain Enter (forcing
    // the user to manually click Send) or submit on every Shift+Enter
    // (no way to add a line break at all).
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      handleSubmit(e as unknown as React.FormEvent);
    }
  }

  return (
    <section className="chat-pane">
      <header className="chat-header">
        <div className="chat-header-brand">
          <Logo size={24} />
          <span className="chat-header-title">Yuan RAG</span>
          <span className="chat-header-subtitle">{t("chat.headerSubtitle")}</span>
        </div>
        <div className="chat-header-actions">
          {/* v2.0.26.2 (PR-5) — hamburger toggle. CSS hides it on
              desktop and reveals it on mobile via the @media block.
              ``aria-controls`` points at the sidebar ``id`` so screen
              readers can announce the expanded/collapsed state. */}
          <button
            type="button"
            className="btn-hamburger"
            aria-label={t("sidebar.toggleAria")}
            aria-controls="primary-sidebar"
            aria-expanded={drawerOpen}
            onClick={onOpenDrawer}
          >
            ☰
          </button>
          {/* Global show/hide for the LLM thinking drawer. Persists to
              localStorage so the preference survives reloads. The button
              is always visible (even when the model produces no
              reasoning, e.g. OpenAI gpt-4o) so the affordance stays
              consistent — it just toggles between ON and OFF without
              changing anything visible if there's no thinking content. */}
          <button
            className="btn-secondary btn-thinking-toggle"
            onClick={() => {
              setShowThinking((v) => {
                const nv = !v;
                localStorage.setItem("rag.show_thinking", String(nv));
                return nv;
              });
            }}
            title={
              showThinking ? t("chat.showThinkingOff") : t("chat.showThinkingOn")
            }
            aria-pressed={showThinking}
          >
            <span aria-hidden="true">💭</span>
            <span>
              {showThinking
                ? t("chat.showThinkingStateOn")
                : t("chat.showThinkingStateOff")}
            </span>
          </button>
          {/* Inline loading hint while the server's background warmup
              finishes (~10-30 s on first launch, instant thereafter). */}
          {!modelsReady && (
            <span
              className="chat-header-loading"
              title={t("chat.modelsLoadingTitle")}
            >
              <span className="chat-header-spinner" /> {t("chat.modelLoading")}
            </span>
          )}
        </div>
      </header>

      <div
        className="chat-messages"
        ref={messagesRef}
        onScroll={() => {
          const el = messagesRef.current;
          if (!el) return;
          const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
          stickToBottomRef.current = distance < 80;
        }}
      >
        {/* Three states for an empty messages array:
             1. ``historyLoaded === false`` → thread is being fetched
                from the server; show a quiet loading indicator. The
                previous behaviour of rendering "开始一次新的对话"
                here was confusing — the user clicked an existing
                conversation in the sidebar and saw the "new chat"
                empty state during the network round-trip, which made
                it look like the click hadn't done anything yet.
             2. ``historyLoaded === true && messages.length === 0`` →
                truly empty thread (either brand-new or one with no
                saved messages). Show the empty state.
            The branch order matters: the loading state must take
            precedence so we never flash the "new chat" copy during
            a switch. ``!historyLoaded`` is the canonical flag —
            ``App.tsx`` flips it to ``false`` BEFORE the fetch starts
            and back to ``true`` in the ``finally`` block, so it
            tracks the in-flight window exactly. */}
        {messages.length === 0 && !historyLoaded && (
          <div className="chat-loading" aria-live="polite">
            <div className="chat-loading-spinner" aria-hidden="true" />
            <div className="chat-loading-text">{t("chat.loadingHistory")}</div>
          </div>
        )}
        {messages.length === 0 && historyLoaded && (
          <div className="chat-empty">
            <div className="chat-empty-icon" aria-hidden="true">
              <Logo size={56} />
            </div>
            <div className="chat-empty-title">{t("chat.emptyTitle")}</div>
            <div className="chat-empty-hint">
              {t("chat.emptyBody1")}
              <br />
              {t("chat.emptyBody2")}
            </div>
          </div>
        )}
        {messages.map((m, idx) => {
          // P1-1: only the in-flight last assistant message uses
          // StreamingMarkdown (frozen prefix + live tail). All other
          // messages — including earlier turns in the same thread and
          // the final state of the streaming message after it
          // completes — render via the static ``<Markdown>`` once.
          const isLiveAssistantTurn =
            m.role === "assistant" &&
            isStreaming &&
            idx === messages.length - 1;
          return (
            <div key={m.id} className={`message ${m.role}`}>
              {/* Reasoning drawer renders ABOVE the answer text inside
                  the assistant bubble. Gated on the global ``showThinking``
                  flag — when OFF, no drawer renders anywhere. */}
              {m.role === "assistant" && m.reasoning && showThinking && (
                <ThinkingDrawer text={m.reasoning} />
              )}
              {/* v2.0 — tool-call cards render BELOW the reasoning
                  drawer and ABOVE the prose body. The order matters:
                  users see (1) the agent's thinking, then (2) what
                  actions the agent took, then (3) the final answer
                  — a faithful ReAct visualization. Each card
                  collapses by default to keep long tool chains from
                  dominating the message bubble. */}
              {m.role === "assistant" && m.tool_calls && m.tool_calls.length > 0 && (
                <div className="tool-call-list">
                  {m.tool_calls.map((call) => (
                    <ToolCallCard key={call.id} call={call} />
                  ))}
                </div>
              )}
              {/* v2.0.29.9 (Phase 8) — verbatim-extraction indicator.
                  Shows a small 🔒 icon at the top of the bubble when
                  the message was synthesized in extractive mode
                  (verbatim quote, no rewriting). Two trigger paths:
                    1. Any source carries ``verbatim=true`` — the
                       backend's react_generate_extractive stamped
                       it. This is the authoritative signal.
                    2. The thread's ``highPrecision==="on"`` was set
                       by the user — the bubble shows the icon
                       even if the message has zero sources (rare
                       but possible on the refusal-fallback path
                       where docs were empty).
                  Title attribute is i18n-keyed so screen readers
                  and hover tooltips both surface the meaning. */}
              {m.role === "assistant" &&
                (m.sources?.some((s) => s.verbatim) || highPrecision === "on") && (
                  <div className="verbatim-lock-indicator" title={t("bubble.highPrecisionHint")}>
                    <span className="verbatim-lock-icon" aria-hidden="true">🔒</span>
                    <span>{t("bubble.highPrecisionActive")}</span>
                  </div>
                )}
              {m.role === "assistant" ? (
                isLiveAssistantTurn ? (
                  // Split at the last paragraph / sentence boundary so
                  // the prefix is full markdown (parsed once) and the
                  // tail is plain text + citation chips (re-rendered
                  // per token without re-parsing).
                  <StreamingMarkdown content={m.content} sources={m.sources} />
                ) : (
                  // Markdown rendering for assistant answers. The
                  // ``Markdown`` component handles citation chips inline
                  // via a remark plugin, so [n] markers become clickable
                  // chips even inside headings/lists/tables.
                  <Markdown content={m.content} sources={m.sources} />
                )
              ) : (
                <div className="message-text">{m.content}</div>
              )}
              {/* v2.0 — per-message timestamp (server-side ISO 8601).
                  Renders below the message body for both user and
                  assistant roles. UTC value; we don't localize in this
                  revision. Hidden when ``created_at`` is missing
                  (older messages or optimistic inserts the user typed
                  before the agent reply arrived). */}
              {m.created_at && (
                <div className="message-timestamp" title={m.created_at}>
                  {formatTimestamp(m.created_at)}
                </div>
              )}
            {(() => {
              // v1.1.15 — banner text driven by ``source_kinds``
              // (an aggregated list of "what kinds of sources shaped
              // this answer"), not by the brittle per-thread
              // ``webSearched`` flag. The user-reported bug was
              // two-headed:
              //   * every answer showed "本回答参考了联网搜索结果"
              //     regardless of whether web search was actually
              //     used (the old flag fired on ``web_search.
              //     attempted`` which was True even when DDG
              //     returned zero hits).
              //   * the banner vanished on page refresh because the
              //     old flag was live-only.
              //
              // Mapping (matches the user's literal spec for the two
              // single-source cases; both banners render when the
              // v1.1.13 complementary path uses both):
              //   local only       → "本回答参考了文档内容"
              //   web   only       → "🔎 本回答参考了联网搜索结果"
              //   both             → render both
              //   empty / undefined→ no banner
              //
              // Legacy fallback: messages predating v1.1.15 don't
              // have ``source_kinds`` — render the OLD
              // ``webSearched`` boolean instead, so users reloading
              // an old conversation don't lose the banner they saw
              // when the turn originally ran.
              const k = m.source_kinds;
              if (Array.isArray(k) && k.length > 0) {
                const hasLocal = k.includes("local");
                const hasWeb = k.includes("web");
                return (
                  <>
                    {hasLocal && (
                      <div className="message-flag message-flag-doc">
                        {t("chat.bannerAnswerLocal")}
                      </div>
                    )}
                    {hasWeb && (
                      <div className="message-flag message-flag-web">
                        <span aria-hidden="true">🔎</span> {t("chat.bannerAnswerWeb")}
                      </div>
                    )}
                  </>
                );
              }
              if (m.webSearched) {
                return (
                  <div className="message-flag message-flag-web">
                    <span aria-hidden="true">🔎</span> {t("chat.bannerAnswerWeb")}
                  </div>
                );
              }
              return null;
            })()}
            {m.grounding && (
              <div className="message-flag message-flag-grounding">
                {t("chat.groundingPrefix", { name: m.grounding })}
              </div>
            )}
            {m.sources && m.sources.length > 0 && (
              <div className="citation-row">
                {m.sources.map((s) => {
                  const isWeb = s.source_kind === "web" && s.url;
                  if (isWeb) {
                    return (
                      <a
                        key={s.index}
                        href={s.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="citation-chip-link"
                        title={s.text}
                      >
                        [{s.index}] {s.domain || s.url}
                      </a>
                    );
                  }
                  return (
                    <span
                      key={s.index}
                      className="citation-chip"
                      data-citation-index={s.index}
                      onClick={() => {
                        // v2.0.29.4 (Phase 4 PR-1) — ship chunk_id /
                        // doc_id / source_kind alongside ``index`` so
                        // App.tsx's ``rag:citation-click`` listener
                        // can identify the exact source this chip
                        // points at. Mirrors the same change in
                        // ``CitationChip.tsx`` (PR-1 #2). Pre-PR-1
                        // only ``index`` was forwarded; a chip that
                        // disagreed with the LLM's intent (e.g. due
                        // to renumber drift between live and reload)
                        // had no chip-level data to cross-check
                        // against. The new fields are additive —
                        // App.tsx still only reads ``detail.index``
                        // today.
                        window.dispatchEvent(
                          new CustomEvent("rag:citation-click", {
                            detail: {
                              index: s.index,
                              chunk_id: s.chunk_id,
                              doc_id: s.doc_id,
                              source_kind: s.source_kind,
                            },
                          }),
                        );
                      }}
                      // Multi-line tooltip: filename + page on the
                      // first line (already shown in the chip text),
                      // then the snippet preview so hover reveals
                      // the actual cited text without leaving the
                      // answer.
                      title={
                        s.text
                          ? `${s.filename}${
                              s.page
                                ? t("chat.pageFooter", { page: s.page })
                                : s.sheet
                                ? ` · ${s.sheet}`
                                : ""
                            }\n${s.text.slice(0, 160).replace(/\s+/g, " ")}${
                              s.text.length > 160 ? "…" : ""
                            }`
                          : s.filename
                      }
                    >
                      [{s.index}] {s.filename}
                      {s.page ? ` p.${s.page}` : ""}
                      {s.sheet ? ` · ${s.sheet}` : ""}
                    </span>
                  );
                })}
              </div>
            )}
            </div>
          );
        })}
        {webSearching && (
          <div className="thinking thinking-status">
            <span aria-hidden="true">🔎</span> {t("chat.searchingWeb")}
            {currentStep > 0 && (
              // PR-4 (v2.0.26.1): was hardcoded English "Step {n}";
              // the existing ``chat.step`` key already carried the
              // right placeholder shape, so this is just a one-line
              // hook-up rather than a new key.
              <span className="thinking-step">
                {t("chat.step", { n: currentStep })}
              </span>
            )}
          </div>
        )}
        {!webSearching && isStreaming && (
          <div className="thinking thinking-status">
            <span className="thinking-pulse" aria-hidden="true" />
            {t("chat.thinking")}
            {currentStep > 0 && (
              <span className="thinking-step">
                {t("chat.step", { n: currentStep })}
              </span>
            )}
          </div>
        )}
      </div>

      <form className="chat-input" onSubmit={handleSubmit}>
        {/* v2.0.31.1 — per-thread verbatim-extraction toggle. The row
            wraps a small label (so first-time users know what the
            segmented IS), the 3-state segmented control, and a
            dynamic description that updates on selection (so users
            see what the CURRENT mode does without hovering).
            Tooltips on each button are still set for users who hover
            on desktop, but mobile users now get the same information
            inline. User choice is forwarded via the WS ``high_precision``
            field on the next send. */}
        <div className="high-precision-toggle-row">
          <div className="high-precision-toggle-label">
            <span className="verbatim-lock-icon" aria-hidden="true">🔒</span>
            <span>{t("chatInput.highPrecisionLabel")}</span>
          </div>
          <div className="high-precision-toggle segmented" role="radiogroup" aria-label={t("chatInput.highPrecisionAria")}>
            {(["auto", "on", "off"] as const).map((value) => (
              <button
                key={value}
                type="button"
                role="radio"
                aria-checked={highPrecision === value}
                data-active={highPrecision === value ? "true" : undefined}
                className="segmented-option high-precision-toggle-btn"
                onClick={() => onHighPrecisionChange(value)}
                title={t(`chatInput.mode${value === "auto" ? "Auto" : value === "on" ? "On" : "Off"}Hint`)}
              >
                {value === "on" ? <span className="verbatim-lock-icon" aria-hidden="true">🔒</span> : null}
                {t(`chatInput.mode${value === "auto" ? "Auto" : value === "on" ? "On" : "Off"}`)}
              </button>
            ))}
          </div>
          <div
            className="high-precision-toggle-description"
            aria-live="polite"
          >
            {t(
              `chatInput.mode${highPrecision === "auto" ? "Auto" : highPrecision === "on" ? "On" : "Off"}Hint`,
            )}
          </div>
        </div>
        <textarea
          ref={textareaRef}
          className="chat-textarea"
          rows={1}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={t("chat.inputPlaceholder")}
          disabled={!modelsReady || !historyLoaded || isStreaming}
        />
        {/* QW #4: when streaming, the send button becomes a stop
            button. Same DOM position (right of the textarea) so the
            layout doesn't jump; same width via the shared
            ``btn-send`` / ``btn-stop`` base class. ``type="button"``
            is critical here — without it, clicking stop would
            submit the form and try to send the (empty) textarea
            content as a new message. ``aria-label`` makes the
            intent clear to screen readers since the visible text is
            short ("停止") and the icon is decorative. */}
        {isStreaming ? (
          <button
            type="button"
            className="btn-stop"
            onClick={onStop}
            aria-label={t("chat.stopButtonAria")}
            title={t("chat.stopButtonTitle")}
          >
            {t("chat.stop")}
            <span className="btn-stop-icon" aria-hidden="true">■</span>
          </button>
        ) : (
          <button
            type="submit"
            className="btn-send"
            disabled={!modelsReady || !input.trim() || !historyLoaded}
          >
            {t("chat.send")}
            <span className="btn-send-icon" aria-hidden="true">⏎</span>
          </button>
        )}
      </form>
    </section>
  );
}

// Wrap in React.memo so the whole message tree (and its input,
// ThinkingDrawer, every inline-style object, every effect) doesn't
// re-render on every App state change — including every streamed
// token from the parent. With stable props from App.tsx, the
// component only re-renders when its inputs actually change.
export const ChatPane = memo(ChatPaneInner);