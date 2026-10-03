// v2.0.6 P2-4 — simplified-Chinese lookup table. Every key here
// has a matching entry in :mod:`./en`; ``useLocale`` falls back to
// the Chinese value when a key is missing from the active locale so
// a partial translation never produces an empty label.
//
// Scope (this version): only the strings that the user actually sees
// in the sidebar / chat header / settings dialog. Tool result bodies,
// citation chips, tool-call cards, and streamed LLM text are out of
// scope — those would require a more invasive change (passing the
// locale into the agent prompts and re-translating every server-
// emitted message) and are tracked as a follow-up.

const zh = {
  // Sidebar
  "sidebar.newChat": "新建对话",
  "sidebar.uploadDoc": "上传文档",
  "sidebar.uploading": "上传中…",
  "sidebar.processing": "处理中…",
  "sidebar.parsing": "解析中…",
  "sidebar.embedding": "向量化中({count} 块)",
  "sidebar.indexed": "已索引",
  "sidebar.empty": "解析为空",
  "sidebar.failed": "失败",
  "sidebar.deleteFailed": "删除失败",
  "sidebar.deleteDocTitle": "删除文档",
  "sidebar.deleteDocConfirm": "确定删除这份文档吗?此操作不可撤销。",
  "sidebar.delete": "删除",
  "sidebar.cancel": "取消",
  "sidebar.openSettings": "设置",
  "sidebar.sessionsTitle": "对话",
  "sidebar.documentsTitle": "文档",
  "sidebar.emptySessions": "暂无对话",
  "sidebar.emptyDocuments": "暂无文档",

  // Chat pane
  "chat.placeholder": "输入消息…",
  "chat.send": "发送",
  "chat.stop": "停止",
  "chat.thinking": "思考中…",
  "chat.step": "第 {n} 步",
  "chat.searchingWeb": "联网搜索中…",
  "chat.connecting": "正在连接…",
  "chat.bannerWeb": "联网搜索",
  "chat.bannerLocal": "已上传文档",
  "chat.bannerBoth": "已上传文档 + 联网搜索",
  "chat.bannerNone": "本地知识",
  "chat.error": "出错了,请稍后重试。",
  "chat.noDocs": "此对话尚未上传文档",
  "chat.history": "对话历史",
  "chat.loadingHistory": "正在加载对话历史…",
  "chat.emptyHistory": "暂无对话记录",
  "chat.toggleThinking": "切换思考过程",

  // Settings dialog
  "settings.title": "设置",
  "settings.close": "关闭",
  "settings.save": "保存",
  "settings.saving": "保存中…",
  "settings.provider": "语言模型供应商",
  "settings.model": "模型",
  "settings.language": "语言",
  "settings.langZh": "中文",
  "settings.langEn": "English",
  "settings.models": "嵌入模型",
  "settings.embeddingModel": "BGE-M3 (向量化)",
  "settings.rerankerModel": "BGE Reranker v2-M3 (重排序)",
  "settings.downloadModels": "下载 / 刷新模型",
  "settings.downloading": "正在下载…",
  "settings.apiKeyEnv": "API 密钥已从环境变量加载({name})",
  "settings.apiKeyKeyring": "API 密钥已从系统钥匙串加载",
  "settings.apiKeyMissing":
    "尚未配置 API 密钥。请先在 .env 文件中设置 {name}。",
  "settings.footnote":
    "API 密钥存储在你的 .env 文件(或容器密钥)中,不会以明文写入磁盘,也不会被本应用持久化。",

  // Connection banner
  "connection.close1001": "聊天连接已断开,请刷新页面",
  "connection.close1006": "网络连接中断,请检查网络后重试",
  "connection.close1008": "认证失败,请刷新页面或重新登录",
  "connection.close1009": "消息过大,请拆分后重试",
  "connection.close1011": "服务器内部错误,请稍后重试",
  "connection.close1014": "网关错误,请稍后重试",
  "connection.closeGeneric": "聊天连接已断开",
  "connection.dismissAria": "关闭提示",
  // v2.0.25 P0-F2 — reconnect-state + retry affordance for the
  // connection-error banner. The retry button only appears when
  // there's a remembered last-sent message; reconnecting/failed
  // states update the message text in place via the WS hook's
  // onReconnectAttempt/onReconnectExhausted callbacks.
  "connection.retry": "重试",
  "connection.retryAria": "重新发送上一条消息",
  "connection.reconnecting": "正在重连(第 {attempt}/3 次)…",
  "connection.failed": "无法重新连接,请检查网络后重试。",

  // Confirm dialog
  "confirm.ok": "确定",
  "confirm.cancel": "取消",
  "confirm.confirmDefault": "确认",
  "confirm.cancelDefault": "取消",
  "confirm.dismiss": "知道了",

  // v2.0.7 i18n-1 — high-visibility strings that were still
  // hardcoded. Used by ThinkingDrawer, ToolCallCard, Sidebar
  // upload widget, ChatPane header / input / show-thinking
  // toggle, ModelDownloadProgress first-run modal.

  // Thinking drawer (collapsible reasoning section)
  "chat.thinkingDrawerCollapse": "收起思考过程",
  "chat.thinkingDrawerExpand": "展开思考过程",
  "chat.showThinkingOn": "显示模型的思考过程",
  "chat.showThinkingOff": "隐藏模型的思考过程",
  "chat.showThinkingStateOn": "已显示思考过程",
  "chat.showThinkingStateOff": "已隐藏思考过程",

  // Chat header + input
  "chat.headerSubtitle": "基于本地文档的智能问答",
  "chat.inputPlaceholder": "向助手提问,或上传文件后提问…",
  "chat.stopButtonAria": "停止生成",
  "chat.stopButtonTitle": "停止生成(后台继续运行,可在历史中查看)",

  // v2.0.29.9 (Phase 8) — verbatim extraction mode bubble indicator.
  // Shown above the assistant message body when the answer was
  // synthesized via react_generate_extractive (any source has
  // verbatim=true OR the thread toggle is "on"). Lock emoji prefix
  // matches the segmented control's "on" state button.
  "bubble.highPrecisionActive": "高精确模式已启用",
  "bubble.highPrecisionHint": "本回答只引用原文,不进行改写或推理",

  // v2.0.29.9 (Phase 8) — per-thread verbatim toggle (above the
  // chat textarea). Three buttons; the active one carries ``active``
  // CSS class.
  // v2.0.31.1 — added ``highPrecisionLabel`` so the toggle renders a
  // visible label above the segmented control. The 3 existing hint
  // strings are now ALSO shown inline below the segmented as a
  // dynamic description (changes with selection), not just on hover.
  "chatInput.highPrecisionLabel": "高精确模式",
  "chatInput.highPrecisionAria": "高精确模式",
  "chatInput.modeAuto": "自动",
  "chatInput.modeAutoHint": "自动检测:命中关键词(法律 / 合同 / 原话 / verbatim 等)时启用 verbatim 抽取",
  "chatInput.modeOn": "开启",
  "chatInput.modeOnHint": "强制开启:所有问题都用 verbatim 模式(不进行改写)",
  "chatInput.modeOff": "关闭",
  "chatInput.modeOffHint": "强制关闭:即使问题触发自动检测也用普通模式回答",
  "chat.modelsLoadingTitle": "正在后台加载模型…",

  // Tool call card
  "toolCard.collapseDetails": "收起详情",
  "toolCard.expandDetails": "展开详情",

  // Sidebar upload widget
  "sidebar.dropHint": "支持上传 {formats} 等文件",
  "sidebar.dropHintGeneric": "支持上传文档文件",
  "sidebar.dropIdle": "上传文件",
  "sidebar.dropActive": "松开以上传",
  "sidebar.emptyDocText": "未提取到文本",
  "sidebar.emptyParse": "解析失败",
  "sidebar.docTitleParsing": "正在解析文档…",
  "sidebar.docTitleEmbedding": "正在生成向量…",
  "sidebar.docTitleQueued": "正在排队…",

  // Sidebar dialog labels (i18n-1 high-priority subset — the
  // labels on the delete / upload-failure dialogs that the user
  // clicks to confirm).
  "sidebar.deleteDocLabel": "删除文档",
  "sidebar.deleteSessionLabel": "删除对话",
  "sidebar.deleteSessionAndDocsLabel": "删除对话及文档",
  "sidebar.removeDocAria": "移除文档",

  // First-run model download modal
  "modelDownload.title": "正在下载模型…",
  "modelDownload.body":
    "首次运行需要下载 BGE-M3 向量化模型与 BGE 重排序模型(总计约 900 MB)。下载只会发生一次,后续启动会直接从本地缓存加载,无需等待。",
  "modelDownload.footnote":
    "请保持此页面打开。模型就绪后会自动进入对话界面。",
  "modelDownload.embeddingLabel": "向量化 (BGE-M3)",
  "modelDownload.rerankerLabel": "重排序 (BGE v2-M3)",

  // v2.0.8 i18n-2 — long-tail strings that were still hardcoded in
  // ChatPane (entry state, answer banners, citation footer, stop
  // label), Sidebar (history / thread-doc sections, upload progress
  // chip, delete / upload-error dialog bodies), and App.tsx (wsError
  // fallback, first-run API-key missing modal). All three files now
  // route their user-visible strings through ``t(...)``.

  // ChatPane — entry state, answer banners, citation footer
  "chat.modelLoading": "模型加载中",
  "chat.emptyTitle": "开始一次新的对话",
  "chat.emptyBody1":
    "你可以直接向助手提问,也可以在左侧上传文件后,围绕文件内容进行提问。",
  "chat.emptyBody2": "回答会标注引用来源,Shift+Enter 换行。",
  "chat.bannerAnswerLocal": "本回答参考了文档内容",
  "chat.bannerAnswerWeb": "本回答参考了联网搜索结果",
  "chat.groundingPrefix": "引用核查:{name}",
  "chat.pageFooter": " · 第 {page} 页",

  // Sidebar — section titles + thread-doc headers
  "sidebar.historyTitle": "历史对话",
  "sidebar.emptyHistoryDetailed": "还没有保存的对话",
  "sidebar.threadDocsTitle": "本对话文档",
  "sidebar.emptyDocsDetailed": "这个对话还没有上传文件",

  // Sidebar — upload progress chip states (used by the inline
  // ``<SidebarUploadProgress/>`` indicator above the doc list).
  "sidebar.uploadProgressSuffix": " (+{count} 处理中)",
  "sidebar.uploadingDoc": "正在上传…",
  "sidebar.parsingDocDetailed": "正在解析文档…",
  "sidebar.indexingChunks": "正在建立索引 ({count} 块)",
  "sidebar.chunkCountSuffix": "{count} 块",

  // Sidebar — delete-doc / delete-session dialog bodies. The
  // ``{filename}`` / ``{title}`` placeholders are filled in by the
  // dialog component to highlight the specific target.
  "sidebar.deleteDocBody1": "确定要删除 {filename} 吗?",
  "sidebar.deleteDocBody2": "该文档的所有文本块都会被永久移除,无法恢复。",
  "sidebar.deleteSessionBody1": "确定要删除对话 {title} 吗?",
  "sidebar.deleteSessionBody2":
    "该对话上传的所有文档及其索引会一并永久删除,无法恢复。",

  // v2.0.25.1 P0-extension — inline error banners for failed
  // destructive actions. PR-6 replaces these with the proper toast
  // system; for now they live in a sidebar-scoped banner so the
  // user sees what failed.
  "sidebar.deleteFailedBody": "删除 {filename} 失败,请检查网络后重试。",
  "sidebar.deleteSessionFailedBody":
    "删除对话 {title} 失败,请检查网络后重试。",
  "sidebar.deleteErrorDismiss": "关闭提示",

  // Sidebar — upload-error dialog. ``{error}`` is the raw message
  // returned by the backend; this wrapper just frames it.
  "sidebar.uploadErrorTitle": "上传失败",
  "sidebar.uploadErrorBody":
    "上传文件时出错:{error}。请检查文件格式或稍后重试。",

  // App.tsx — top-level error fallback + first-run API-key missing
  // modal. The modal is shown when no LLM API key is configured and
  // the user tries to send the first message.
  "app.wsErrorFallback": "聊天连接出错,请检查网络",
  "app.apiKeyTitle": "需要配置 API 密钥",
  "app.apiKeyBody":
    "尚未配置 MINIMAX_API_KEY。请在 .env 文件中添加后,点击右上角的齿轮图标重新检查。",
  "app.apiKeyOpenSettings": "打开设置",

  // v2.0.26.1 (PR-4) — i18n 3-layer consistency. Mirror of en.ts
  // (same key set; the values are the existing Chinese copy that
  // was hardcoded in the components). Keep in lockstep with en.ts
  // — the ``test_catalog_has_both_locales_for_every_entry`` test
  // on the backend, plus the symmetric TS key types, would catch a
  // drift here at compile/test time.

  // Tool call card
  "toolCard.unknown": "(未知)",
  "toolCard.step": "第 {n} 步",
  "toolCard.args": "参数",
  "toolCard.result": "结果",

  // Markdown / StreamingMarkdown citation chip
  "markdown.citationAria":
    "跳转到来源 {idx}:{filename}{pageSuffix}{sheetSuffix}",
  "markdown.citationAriaFallback": "跳转到来源 {idx}",
  "markdown.citationTitle": "引用 {idx}",
  "markdown.noPreview": "(无内容预览)",
  "markdown.pageSuffix": " · 第 {page} 页",
  "markdown.sheetSuffix": " · {sheet}",

  // Settings dialog — provider label
  "settings.providerAnthropic": "Anthropic (MiniMax 兼容)",

  // Status labels — mirrors src/frontend/src/i18n/status.ts.
  "status.ready": "就绪",
  "status.downloading": "下载中…",
  "status.missing": "未开始",
  "status.pending": "等待中",

  // Error labels — mirrors src/frontend/src/i18n/errors.ts.
  "errors.fallback": "未知错误",
  "errors.network.failedToFetch": "请求失败",
  "errors.network.networkError": "网络请求出错",
  "errors.network.loadFailed": "加载失败",
  "errors.network.generic": "网络错误",

  // HTTP status → label
  "errors.status.400": "请求格式错误",
  "errors.status.401": "未授权,请检查 API 密钥",
  "errors.status.403": "没有访问权限",
  "errors.status.404": "资源不存在",
  "errors.status.408": "请求超时",
  "errors.status.409": "资源冲突",
  "errors.status.413": "文件过大",
  "errors.status.415": "不支持的文件格式",
  "errors.status.422": "请求参数无效",
  "errors.status.429": "请求过于频繁,请稍后重试",
  "errors.status.500": "服务器内部错误",
  "errors.status.502": "网关错误",
  "errors.status.503": "服务暂时不可用",
  "errors.status.504": "网关超时",

  // v2.0.26.2 (PR-5) — responsive + dark mode foundation.
  // - settings.theme*       — 设置对话框里的浅色 / 深色 / 跟随系统三选一
  // - sidebar.toggleAria    — 移动端汉堡按钮的 aria-label
  "settings.theme": "外观",
  "settings.themeLight": "浅色",
  "settings.themeDark": "深色",
  "settings.themeSystem": "跟随系统",
  "sidebar.toggleAria": "切换导航",

  // v2.0.26.3 (PR-6) — toast 通知系统。
  "toast.regionAria": "通知",
  "toast.dismissAria": "关闭通知",
  "toast.stoppedBody": "生成已停止,回复已保存到历史记录",
  "toast.wsParseError": "聊天消息无法显示(服务器返回了意外格式)。请重试。",
} as const;

export type ZhKey = keyof typeof zh;

export default zh;
