"""Locale-aware error-message helper for backend route handlers.

PR-4 (Item 6, v2.0.26.1) — the previous code path raised
``HTTPException(detail="empty upload")`` or sent
``{"type": "error", "message": "Invalid JSON"}`` with hardcoded
English (or Chinese in the ``_EMPTY_MESSAGES`` block). When the
frontend flipped the language to English via SettingsDialog, these
strings stayed in their original language — the user saw English
UI + Chinese error copy, the worst-of-both-worlds visual.

The fix is a stable code catalog: every user-visible error gets a
short identifier (``ws.invalid_json``, ``upload.empty``, etc.) and
the catalog maps each code to its ``zh`` and ``en`` strings. Route
handlers call :func:`error_message` with the parsed locale from the
``Accept-Language`` header; the response carries the localized
message directly.

Design notes
------------

1. **Catalog-driven, not magic-string-driven.** Old code used raw
   English/Chinese literals; two callers could drift the wording
   even though they meant the same thing. Codes make the contract
   explicit.

2. **Default to zh, the project default.** The frontend still
   defaults to Chinese on first load (see
   ``src/frontend/src/i18n/index.tsx``), so the backend mirror
   defaulting to zh means a missing ``Accept-Language`` header
   produces Chinese copy — same as if the user had the frontend
   in Chinese.

3. **Both locales required for every entry.** A missing locale is
   a build-time fail by convention (catalog audit before merge),
   not a runtime fallback. Half-localized UX is worse than
   fully-localized.

4. **Template kwargs use ICU-lite ``{key}`` placeholders.** Same
   shape as the frontend's ``t(key, params)`` so the call sites
   read symmetrically.

5. **Unknown code returns the code itself.** This is intentional:
   if a handler calls ``error_message("made.up.code")`` it's a bug
   worth surfacing in QA, but we don't want to crash the request
   path. The literal code string is unhelpful to users but obvious
   to developers (``grep`` for it in the catalog to find the gap).

6. **Format-failure falls back to the raw template.** Same reason:
   surface the bug rather than mask it with a string that looks
   fine but is wrong.
"""
from __future__ import annotations

# Locale parsing — Accept-Language per RFC 7231 §5.3.5.
# We support only "zh" and "en" for now (the only two locales the
# frontend ships — see ``src/frontend/src/i18n/index.tsx``). Anything
# else (including q=0 explicit exclusions, which the parser treats as
# "not preferred") falls back to zh.
SUPPORTED_LOCALES: tuple[str, ...] = ("zh", "en")
DEFAULT_LOCALE: str = "zh"


def parse_accept_language(header: str | None) -> str:
    """Extract the best-supported locale from an Accept-Language header.

    Returns ``"zh"`` or ``"en"``. Unknown / missing → ``"zh"``.

    The parser handles the common shapes:

        "zh-CN,zh;q=0.9,en;q=0.8"  → "zh"  (zh-CN primary, supported)
        "en-US,en;q=0.9"            → "en"
        "fr-FR,fr;q=0.9"            → "zh"  (fr not supported → default)
        "" / None                    → "zh"

    Quality factors are ignored — only the order matters for our
    two-locale case. If a future PR adds a third locale with
    meaningful q-factor priority, this can be extended to sort
    candidates by q before returning the best.
    """
    if not header:
        return DEFAULT_LOCALE
    # Walk left-to-right (highest priority first per RFC 7231).
    for raw in header.split(","):
        tag = raw.split(";", 1)[0].strip().lower()
        if not tag:
            continue
        primary = tag.split("-", 1)[0]
        if primary in SUPPORTED_LOCALES:
            return primary
    return DEFAULT_LOCALE


# Error code catalog. Each entry MUST have both ``zh`` and ``en`` —
# a missing locale is a CI fail (grep gate in the PR review), not a
# runtime fallback. Templates use ``{key}`` placeholders that get
# filled by ``str.format(**kwargs)`` at call time.
_MESSAGES: dict[str, dict[str, str]] = {
    # ---- WebSocket errors (websocket.py) ----
    "ws.invalid_json": {
        "zh": "无效的 JSON",
        "en": "Invalid JSON",
    },
    "ws.empty_message": {
        "zh": "消息为空",
        "en": "Empty message",
    },
    "ws.rate_limited": {
        "zh": "请求过于频繁,请稍后再试。",
        "en": "Rate limit exceeded. Please slow down.",
    },
    # ---- Upload errors (documents.py) ----
    "upload.empty": {
        "zh": "上传的文件为空",
        "en": "Empty upload",
    },
    "upload.sniff_mismatch": {
        "zh": "文件内容与 {ext} 扩展名不匹配(magic-byte sniff 失败)",
        "en": "File contents do not match the {ext} extension (magic-byte sniff failed)",
    },
    # ---- Session errors (sessions.py) ----
    "session.thread_id_required": {
        "zh": "缺少 thread_id",
        "en": "thread_id required",
    },
    "session.load_history_failed": {
        "zh": "加载历史失败",
        "en": "load history failed",
    },
    "session.clear_failed": {
        "zh": "清理会话失败",
        "en": "clear_session failed",
    },
    # ---- Chat errors (chat.py) ----
    "chat.clear_failed": {
        "zh": "清理失败",
        "en": "clear failed",
    },
    # ---- Document empty-status messages (documents.py _EMPTY_MESSAGES) ----
    "docs.empty_status.xlsx": {
        "zh": (
            "未提取到文本内容。Excel 文件每个 sheet 至少需要一行数据;"
            "若含公式,请在 Excel 中打开后按 Ctrl+S 保存以填充缓存值后再上传。"
        ),
        "en": (
            "No text content extracted. Each Excel sheet must contain "
            "at least one row of data; if the file uses formulas, open "
            "it in Excel and press Ctrl+S to fill the cached values "
            "before uploading."
        ),
    },
    "docs.empty_status.pdf": {
        "zh": (
            "未提取到文本内容。可能原因:扫描件无文字层、加密 PDF、"
            "或全部为图片。请检查文档或尝试重新生成 PDF 后再上传。"
        ),
        "en": (
            "No text content extracted. Possible causes: scanned PDF "
            "without a text layer, encrypted PDF, or all images. "
            "Please check the document or try regenerating the PDF "
            "before uploading."
        ),
    },
    "docs.empty_status.docx": {
        "zh": "未提取到文本内容。Word 文档可能为空、加密或全部为图片。",
        "en": (
            "No text content extracted. The Word document may be "
            "empty, encrypted, or contain only images."
        ),
    },
    "docs.empty_status.pptx": {
        "zh": "未提取到文本内容。PPT 文件可能为空、加密或全部为图片。",
        "en": (
            "No text content extracted. The PowerPoint file may be "
            "empty, encrypted, or contain only images."
        ),
    },
    "docs.empty_status.html": {
        "zh": "未提取到文本内容。HTML 文档可能为空或仅含图片 / 脚本。",
        "en": (
            "No text content extracted. The HTML document may be "
            "empty or contain only images / scripts."
        ),
    },
}


def error_message(code: str, locale: str = DEFAULT_LOCALE, **kwargs) -> str:
    """Look up a localized error message by code.

    Args:
        code: stable identifier from the catalog (e.g. ``"ws.invalid_json"``)
        locale: ``"zh"`` or ``"en"``; unknown → ``"zh"``
        **kwargs: template variables filled into the ``{key}`` placeholders

    Returns:
        The formatted localized message.

    Failure modes (intentional — surface bugs, don't mask them):

    - Unknown code → returns the literal code string. Visible in QA
      logs, ``grep``-able against the catalog to find the gap.
    - Missing locale for a known code → returns the ``DEFAULT_LOCALE``
      copy (zh). Indicates the catalog drifted; fix in code, not at
      runtime.
    - Format error (missing kwarg) → returns the raw template. Same
      reason: bad template is a bug, not a UX choice.
    """
    entry = _MESSAGES.get(code)
    if entry is None:
        return code
    template = entry.get(locale) or entry.get(DEFAULT_LOCALE) or code
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError):
        return template