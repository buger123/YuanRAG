// v2.0.6 P2-4 — English lookup table. Every key here matches one
// in :mod:`./zh`; ``useLocale`` falls back to the Chinese value when
// a key is missing so a partial translation never produces an empty
// label (and the user can still see the message in a language they
// may not speak, which beats a blank UI).
//
// Translation notes:
//
// - Technical terms (BGE-M3, Reranker, ``.env``, ``rag.active_thread``)
//   stay as-is — those are proper nouns the user already knows.
// - Bracketed placeholders follow ICU-lite semantics: ``{count}``,
//   ``{n}``, ``{name}``. The lookup helper passes plain string
//   substitution so callers don't need a full ICU dependency.
// - Voice is light and conversational, matching the existing Chinese
//   register; we deliberately avoid stiff "Configuration" /
//   "Preferences" wording.

const en = {
  // Sidebar
  "sidebar.newChat": "New chat",
  "sidebar.uploadDoc": "Upload document",
  "sidebar.uploading": "Uploading…",
  "sidebar.processing": "Processing…",
  "sidebar.parsing": "Parsing…",
  "sidebar.embedding": "Embedding ({count} chunks)",
  "sidebar.indexed": "Indexed",
  "sidebar.empty": "No content extracted",
  "sidebar.failed": "Failed",
  "sidebar.deleteFailed": "Delete failed",
  "sidebar.deleteDocTitle": "Delete document",
  "sidebar.deleteDocConfirm":
    "Delete this document? This cannot be undone.",
  "sidebar.delete": "Delete",
  "sidebar.cancel": "Cancel",
  "sidebar.openSettings": "Settings",
  "sidebar.sessionsTitle": "Conversations",
  "sidebar.documentsTitle": "Documents",
  "sidebar.emptySessions": "No conversations yet",
  "sidebar.emptyDocuments": "No documents yet",

  // Chat pane
  "chat.placeholder": "Type a message…",
  "chat.send": "Send",
  "chat.stop": "Stop",
  "chat.thinking": "Thinking…",
  "chat.step": "Step {n}",
  "chat.searchingWeb": "Searching the web…",
  "chat.connecting": "Connecting…",
  "chat.bannerWeb": "Web search",
  "chat.bannerLocal": "Uploaded documents",
  "chat.bannerBoth": "Uploaded documents + Web search",
  "chat.bannerNone": "Local knowledge",
  "chat.error": "Something went wrong. Please try again.",
  "chat.noDocs": "No documents uploaded for this conversation",
  "chat.history": "Conversation history",
  "chat.loadingHistory": "Loading conversation history…",
  "chat.emptyHistory": "No conversation history yet",
  "chat.toggleThinking": "Toggle thinking trace",

  // Settings dialog
  "settings.title": "Settings",
  "settings.close": "Close",
  "settings.save": "Save",
  "settings.saving": "Saving…",
  "settings.provider": "LLM provider",
  "settings.model": "Model",
  "settings.language": "Language",
  "settings.langZh": "中文",
  "settings.langEn": "English",
  "settings.models": "Embedding models",
  "settings.embeddingModel": "BGE-M3 (embeddings)",
  "settings.rerankerModel": "BGE Reranker v2-M3 (reranker)",
  "settings.downloadModels": "Download / refresh models",
  "settings.downloading": "Downloading…",
  "settings.apiKeyEnv": "API key loaded from env ({name})",
  "settings.apiKeyKeyring": "API key loaded from system keyring",
  "settings.apiKeyMissing":
    "No API key configured. Please set {name} in your .env file.",
  "settings.footnote":
    "Your API key lives in your .env file (or container secret); it is never persisted in plaintext by this app.",

  // Connection banner
  "connection.close1001": "Chat connection lost, please refresh the page",
  "connection.close1006":
    "Network connection lost, please check your network and retry",
  "connection.close1008":
    "Authentication failed, please refresh the page or sign in again",
  "connection.close1009": "Message too large, please split it and retry",
  "connection.close1011": "Internal server error, please retry shortly",
  "connection.close1014": "Gateway error, please retry shortly",
  "connection.closeGeneric": "Chat connection lost",
  "connection.dismissAria": "Dismiss",
  // v2.0.25 P0-F2 — reconnect-state + retry affordance for the
  // connection-error banner. The retry button only appears when
  // there's a remembered last-sent message; reconnecting/failed
  // states update the message text in place via the WS hook's
  // onReconnectAttempt/onReconnectExhausted callbacks.
  "connection.retry": "Retry",
  "connection.retryAria": "Retry last message",
  "connection.reconnecting": "Reconnecting (attempt {attempt}/3)…",
  "connection.failed":
    "Could not reconnect. Please check your network and retry.",

  // Confirm dialog
  "confirm.ok": "OK",
  "confirm.cancel": "Cancel",
  "confirm.confirmDefault": "Confirm",
  "confirm.cancelDefault": "Cancel",
  "confirm.dismiss": "Got it",

  // v2.0.7 i18n-1 — high-visibility strings that were still
  // hardcoded. Used by ThinkingDrawer, ToolCallCard, Sidebar
  // upload widget, ChatPane header / input / show-thinking
  // toggle, ModelDownloadProgress first-run modal.

  // Thinking drawer (collapsible reasoning section)
  "chat.thinkingDrawerCollapse": "Collapse thinking trace",
  "chat.thinkingDrawerExpand": "Expand thinking trace",
  "chat.showThinkingOn": "Show thinking trace",
  "chat.showThinkingOff": "Hide thinking trace",
  "chat.showThinkingStateOn": "Thinking trace visible",
  "chat.showThinkingStateOff": "Thinking trace hidden",

  // Chat header + input
  "chat.headerSubtitle": "Smart Q&A over your local documents",
  "chat.inputPlaceholder": "Ask the assistant, or upload a file first…",
  "chat.stopButtonAria": "Stop generation",
  "chat.stopButtonTitle":
    "Stop generation (the run continues in the background — view it in history)",
  "chat.modelsLoadingTitle": "Loading models in the background…",

  // v2.0.29.9 (Phase 8) — verbatim extraction mode bubble indicator.
  "bubble.highPrecisionActive": "Verbatim mode active",
  "bubble.highPrecisionHint":
    "This answer quotes source text verbatim without rewriting or reasoning",

  // v2.0.29.9 (Phase 8) — per-thread verbatim toggle (above the
  // chat textarea). Three buttons; the active one carries ``active``
  // CSS class.
  // v2.0.31.1 — added ``highPrecisionLabel`` so the toggle renders a
  // visible label above the segmented control. The 3 existing hint
  // strings are now ALSO shown inline below the segmented as a
  // dynamic description (changes with selection), not just on hover.
  "chatInput.highPrecisionLabel": "Verbatim mode",
  "chatInput.highPrecisionAria": "Verbatim mode",
  "chatInput.modeAuto": "Auto",
  "chatInput.modeAutoHint":
    "Auto-detect: enable verbatim extraction on keyword hits (law / contract / verbatim / etc.)",
  "chatInput.modeOn": "On",
  "chatInput.modeOnHint":
    "Force on: every question uses verbatim mode (no rewriting)",
  "chatInput.modeOff": "Off",
  "chatInput.modeOffHint":
    "Force off: even auto-detect hits use normal synthesis",

  // Tool call card
  "toolCard.collapseDetails": "Hide details",
  "toolCard.expandDetails": "Show details",

  // Sidebar upload widget
  "sidebar.dropHint": "Supports uploading {formats} and similar files",
  "sidebar.dropHintGeneric": "Supports document uploads",
  "sidebar.dropIdle": "Upload file",
  "sidebar.dropActive": "Drop to upload",
  "sidebar.emptyDocText": "No text extracted",
  "sidebar.emptyParse": "Parse failed",
  "sidebar.docTitleParsing": "Parsing document…",
  "sidebar.docTitleEmbedding": "Generating embeddings…",
  "sidebar.docTitleQueued": "Queued…",

  // Sidebar dialog labels (i18n-1 high-priority subset — the
  // labels on the delete / upload-failure dialogs that the user
  // clicks to confirm).
  "sidebar.deleteDocLabel": "Delete document",
  "sidebar.deleteSessionLabel": "Delete conversation",
  "sidebar.deleteSessionAndDocsLabel": "Delete conversation and documents",
  "sidebar.removeDocAria": "Remove document",

  // First-run model download modal
  "modelDownload.title": "Downloading models…",
  "modelDownload.body":
    "First run needs to download the BGE-M3 embedding model and BGE reranker model (about 900 MB total). This only happens once — subsequent launches load straight from the local cache, no waiting.",
  "modelDownload.footnote":
    "Please keep this page open. The app will continue automatically once the models are ready.",
  "modelDownload.embeddingLabel": "Embedding (BGE-M3)",
  "modelDownload.rerankerLabel": "Reranker (BGE v2-M3)",

  // v2.0.8 i18n-2 — long-tail strings that were still hardcoded in
  // ChatPane (entry state, answer banners, citation footer, stop
  // label), Sidebar (history / thread-doc sections, upload progress
  // chip, delete / upload-error dialog bodies), and App.tsx (wsError
  // fallback, first-run API-key missing modal). All three files now
  // route their user-visible strings through ``t(...)``.

  // ChatPane — entry state, answer banners, citation footer
  "chat.modelLoading": "Loading model",
  "chat.emptyTitle": "Start a new conversation",
  "chat.emptyBody1":
    "Ask the assistant directly, or upload a file on the left and ask about its contents.",
  "chat.emptyBody2":
    "Answers include source citations. Press Shift+Enter to insert a newline.",
  "chat.bannerAnswerLocal": "This answer references uploaded documents",
  "chat.bannerAnswerWeb": "This answer references web search results",
  "chat.groundingPrefix": "Citation check: {name}",
  "chat.pageFooter": " · Page {page}",

  // Sidebar — section titles + thread-doc headers
  "sidebar.historyTitle": "Saved conversations",
  "sidebar.emptyHistoryDetailed": "No saved conversations yet",
  "sidebar.threadDocsTitle": "This conversation's documents",
  "sidebar.emptyDocsDetailed": "No files uploaded for this conversation",

  // Sidebar — upload progress chip states (used by the inline
  // ``<SidebarUploadProgress/>`` indicator above the doc list).
  "sidebar.uploadProgressSuffix": " (+{count} in progress)",
  "sidebar.uploadingDoc": "Uploading…",
  "sidebar.parsingDocDetailed": "Parsing document…",
  "sidebar.indexingChunks": "Indexing ({count} chunks)",
  "sidebar.chunkCountSuffix": "{count} chunks",

  // Sidebar — delete-doc / delete-session dialog bodies. The
  // ``{filename}`` / ``{title}`` placeholders are filled in by the
  // dialog component to highlight the specific target.
  "sidebar.deleteDocBody1": "Delete {filename}?",
  "sidebar.deleteDocBody2":
    "All text chunks from this document will be permanently removed and cannot be recovered.",
  "sidebar.deleteSessionBody1": "Delete conversation {title}?",
  "sidebar.deleteSessionBody2":
    "All documents uploaded in this conversation, including their indices, will be permanently deleted and cannot be recovered.",

  // v2.0.25.1 P0-extension — inline error banners for failed
  // destructive actions. PR-6 replaces these with the proper toast
  // system; for now they live in a sidebar-scoped banner so the
  // user sees what failed.
  "sidebar.deleteFailedBody":
    "Could not delete {filename}. Please check your network and try again.",
  "sidebar.deleteSessionFailedBody":
    "Could not delete conversation {title}. Please check your network and try again.",
  "sidebar.deleteErrorDismiss": "Dismiss",

  // Sidebar — upload-error dialog. ``{error}`` is the raw message
  // returned by the backend; this wrapper just frames it.
  "sidebar.uploadErrorTitle": "Upload failed",
  "sidebar.uploadErrorBody":
    "Upload failed: {error}. Please check the file format and try again.",

  // App.tsx — top-level error fallback + first-run API-key missing
  // modal. The modal is shown when no LLM API key is configured and
  // the user tries to send the first message.
  "app.wsErrorFallback":
    "Chat connection error. Please check your network.",
  "app.apiKeyTitle": "API key required",
  "app.apiKeyBody":
    "MINIMAX_API_KEY is not set. Please add it to your .env file and click the gear icon in the top right to retry.",
  "app.apiKeyOpenSettings": "Open settings",

  // v2.0.26.1 (PR-4) — i18n 3-layer consistency. These keys replace
  // the remaining hardcoded English / Chinese literals that the
  // earlier PRs left in place. They are split into the three layers
  // from the audit:
  //
  // - toolCard.*    — ToolCallCard.tsx (steps / args / result / name)
  // - markdown.*    — CitationChip / Markdown.tsx / StreamingMarkdown.tsx
  //                   (chip aria-label / tooltip / fallback)
  // - settings.*    — SettingsDialog.tsx (provider label, was hardcoded
  //                   English "Anthropic (Claude)")
  // - status.*      — i18n/status.ts mirror (was Chinese-only)
  // - errors.*      — i18n/errors.ts mirror (was Chinese-only) and the
  //                   generic fallback used when status code is unknown

  // Tool call card — name fallback + step / args / result labels.
  // ``{n}`` for the step number (1-based; matches the existing
  // ``chat.step`` placeholder shape).
  "toolCard.unknown": "(unknown)",
  "toolCard.step": "Step {n}",
  "toolCard.args": "args",
  "toolCard.result": "result",

  // Markdown / StreamingMarkdown citation chip.
  // ``{idx}`` = 1-based source index; ``{filename}`` / ``{page}`` /
  // ``{sheet}`` are filled by CitationChip. The aria-label is what
  // screen readers announce; the title is the hover tooltip.
  "markdown.citationAria":
    "Jump to source {idx}: {filename}{pageSuffix}{sheetSuffix}",
  "markdown.citationAriaFallback": "Jump to source {idx}",
  "markdown.citationTitle": "Source {idx}",
  "markdown.noPreview": "(no content preview)",
  "markdown.pageSuffix": " · Page {page}",
  "markdown.sheetSuffix": " · {sheet}",

  // Settings dialog — provider label was hardcoded English. The
  // miniMax "兼容" parenthetical is dropped in English — the model
  // name alone communicates what the dropdown selects.
  "settings.providerAnthropic": "Anthropic (Claude)",

  // Status labels — mirrors src/frontend/src/i18n/status.ts.
  // Model lifecycle states shown next to the model name in
  // SettingsDialog / ModelDownloadProgress.
  "status.ready": "Ready",
  "status.downloading": "Downloading…",
  "status.missing": "Not started",
  "status.pending": "Pending",

  // Error labels — mirrors src/frontend/src/i18n/errors.ts.
  // Used by apiError() in src/frontend/src/api/client.ts.
  "errors.fallback": "Unknown error",
  "errors.network.failedToFetch": "Failed to fetch",
  "errors.network.networkError":
    "NetworkError when attempting to fetch resource",
  "errors.network.loadFailed": "Load failed",
  "errors.network.generic": "Network error",

  // HTTP status → label (mirrors STATUS_LABEL_ZH in errors.ts).
  // Generic enough to cover the full 4xx/5xx range; status codes
  // we don't have a specific copy for fall back to
  // ``errors.fallback``.
  "errors.status.400": "Bad request",
  "errors.status.401": "Unauthorized — please check your API key",
  "errors.status.403": "Permission denied",
  "errors.status.404": "Not found",
  "errors.status.408": "Request timed out",
  "errors.status.409": "Conflict",
  "errors.status.413": "File too large",
  "errors.status.415": "Unsupported file format",
  "errors.status.422": "Invalid request",
  "errors.status.429": "Too many requests — please slow down",
  "errors.status.500": "Internal server error",
  "errors.status.502": "Gateway error",
  "errors.status.503": "Service temporarily unavailable",
  "errors.status.504": "Gateway timeout",

  // v2.0.26.2 (PR-5) — responsive + dark mode foundation.
  // - settings.theme*       — segmented control in SettingsDialog for
  //                           picking light / dark / system
  // - sidebar.toggleAria    — hamburger button on mobile that opens
  //                           the slide-in sidebar drawer
  "settings.theme": "Appearance",
  "settings.themeLight": "Light",
  "settings.themeDark": "Dark",
  "settings.themeSystem": "System",
  "sidebar.toggleAria": "Toggle navigation",

  // v2.0.26.3 (PR-6) — toast notification system.
  // - toast.regionAria    — accessible name for the container that
  //                         holds stacked toasts (so screen readers
  //                         announce the new content as it arrives).
  // - toast.dismissAria   — aria-label on the per-toast × button.
  // - toast.stoppedBody   — success feedback when the user clicks
  //                         "停止生成" (PR-6 wires onStop to a
  //                         success toast so the cancel feels
  //                         explicit, not silent).
  "toast.regionAria": "Notifications",
  "toast.dismissAria": "Dismiss notification",
  "toast.stoppedBody": "Generation stopped — answer saved to history",
  "toast.wsParseError":
    "Chat message could not be displayed (server sent an unexpected format). Please try again.",
} as const;

export type EnKey = keyof typeof en;

export default en;
