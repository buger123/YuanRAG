"""Tests for the v2.0.8 P1 i18n 长尾 + tokenizer warmup batch.

Bug history
-----------
v2.0.7 ship 后, user wanted the remaining i18n + perf items that
were deferred to a follow-up batch. The P1 tier bundles:

  perf-7  ``hybrid_chunker._get_chunker`` is lazy-init and costs
          5-15 s on a cold HF cache (AutoTokenizer deserializes a
          2.3 GB HF model on first call + HybridChunker internal
          state). The previous v2.0.7 ship only warmed up the
          BGE-M3 / reranker models in ``app.py`` lifespan — the
          chunker was paid on the first user upload. New
          ``warmup_chunker()`` runs from the lifespan hook so the
          first upload hits a warm tokenizer.

  i18n-2  Three frontend files had ~30 user-visible zh strings
          still hardcoded after v2.0.7 (ChatPane answer banners +
          entry state + citation footer + stop label; Sidebar
          section titles + upload-progress chip + delete / upload-
          error dialog bodies; App.tsx wsError fallback + first-
          run API-key missing modal). Extracted to
          ``i18n/zh.ts`` + ``i18n/en.ts`` via the existing
          ``useLocale()`` Context.

  perf-8  ``bge_m3._infer_lock`` split was REJECTED — FlagEmbedding
          ``BGEM3FlagModel.encode_queries`` / ``encode_corpus`` are
          not thread-safe (concurrent calls produce interleaved/
          wrong embeddings silently). The v2.0.7 perf-5 64-sub-
          batch lock release is the only safe + efficient
          workaround in the current architecture. The single
          ``_infer_lock`` invariant is locked down by a regression
          test so a future attempt to split is forced to think.

These tests pin the behavior at three levels:

  1. **Backend** — direct unit tests against the new helpers and
     module-level invariants (warmup idempotency, lock invariant).
  2. **Frontend** — static-grep tests that confirm the new
     strings exist as ``t(...)`` calls and the dictionary keys
     line up between zh.ts and en.ts.
  3. **Cross-cutting** — the i18n key alignment is byte-identical
     between zh and en, and no zh strings leak into the three
     target files (ChatPane / Sidebar / App.tsx).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ZH_TS = REPO_ROOT / "src" / "frontend" / "src" / "i18n" / "zh.ts"
EN_TS = REPO_ROOT / "src" / "frontend" / "src" / "i18n" / "en.ts"
HYBRID_CHUNKER_PY = REPO_ROOT / "src" / "ingestion" / "chunkers" / "hybrid_chunker.py"
APP_PY = REPO_ROOT / "src" / "app.py"
BGE_M3_PY = REPO_ROOT / "src" / "embeddings" / "bge_m3.py"
CHATPANE_TSX = REPO_ROOT / "src" / "frontend" / "src" / "components" / "ChatPane.tsx"
SIDEBAR_TSX = REPO_ROOT / "src" / "frontend" / "src" / "components" / "Sidebar.tsx"
APP_TSX = REPO_ROOT / "src" / "frontend" / "src" / "App.tsx"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_keys(text: str) -> set[str]:
    """Pull the set of i18n keys (string-literal-keys before colon)."""
    return set(re.findall(r'"([a-zA-Z0-9_.]+)"\s*:', text))


_CJK_RE = re.compile(r"[一-鿿]")


def _non_comment_cjk_lines(text: str) -> list[tuple[int, str]]:
    """Return [(line_no, line_text), ...] of non-comment lines
    that contain CJK characters. Strips out:

    - Single-line ``//`` and ``/*`` comments
    - Continuation lines of multi-line ``/* ... */`` blocks
    - JSX block-comment regions (``{/* ... */}``) that may span
      multiple lines and contain CJK in the prose.

    Anything that survives the filter is a user-visible zh
    string (a regression for v2.0.8 i18n-2)."""
    out: list[tuple[int, str]] = []
    in_block_comment = False
    in_jsx_block_comment = False
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()

        # /* ... */ multi-line block comments (C-style).
        if in_block_comment:
            if "*/" in stripped:
                in_block_comment = False
            continue
        if stripped.startswith("/*"):
            if "*/" not in stripped:
                in_block_comment = True
            continue

        # {/* ... */} JSX block comments — span lines but always
        # appear inside JSX (i.e. inside the return body).
        if in_jsx_block_comment:
            if "*/}" in stripped:
                in_jsx_block_comment = False
            continue
        if "{/*" in stripped:
            if "*/}" not in stripped:
                in_jsx_block_comment = True
            # If the opening and closing are on the same line,
            # fall through (no CJK should be there anyway).
            continue

        # Single-line // comments.
        if stripped.startswith("//") or stripped.startswith("*"):
            continue

        if _CJK_RE.search(line):
            out.append((i, line))
    return out


# ============================================================
# perf-7 — chunker warmup
# ============================================================


class TestChunkerWarmup:
    def test_warmup_chunker_function_exists(self):
        text = _read(HYBRID_CHUNKER_PY)
        assert "def warmup_chunker()" in text

    def test_warmup_calls_get_chunker(self):
        text = _read(HYBRID_CHUNKER_PY)
        # The warmup body must call the existing lazy init so it
        # inherits the double-checked lock.
        assert re.search(
            r"def warmup_chunker\(\).*?_get_chunker\(\)",
            text,
            re.DOTALL,
        )

    def test_warmup_in_all_exports(self):
        text = _read(HYBRID_CHUNKER_PY)
        assert '"warmup_chunker"' in text or "'warmup_chunker'" in text
        # Specifically the __all__ list export.
        assert re.search(
            r"__all__\s*=\s*\[[^\]]*warmup_chunker[^\]]*\]",
            text,
            re.DOTALL,
        )

    def test_warmup_swallows_exceptions(self):
        text = _read(HYBRID_CHUNKER_PY)
        # The warmup must never raise — failures fall back to
        # lazy init on the first user upload. We capture the
        # function body until the NEXT top-level ``def `` or EOF
        # so the ``except`` clause (which lives AFTER ``return True``)
        # is included in the assertion.
        m = re.search(
            r"def warmup_chunker\(\) -> bool:\n(?P<body>.*?)(?=\n(?:def |@|\Z))",
            text,
            re.DOTALL,
        )
        assert m is not None, "warmup_chunker() body not found"
        body = m.group("body")
        assert "except Exception" in body
        assert "return False" in body
        assert "logger.exception" in body

    def test_lifespan_calls_warmup_chunker(self):
        text = _read(APP_PY)
        assert "from src.ingestion.chunkers.hybrid_chunker import warmup_chunker" in text
        assert "warmup_chunker()" in text
        # Order: BGE-M3 warmup must come BEFORE the chunker warmup
        # (BGE-M3 is the slow one, ~10-30 s, and runs in background;
        # chunker warmup is the fast follow-up that completes during
        # the same startup window).
        bge_pos = text.find("start_warmup_if_needed")
        chunker_pos = text.find("warmup_chunker")
        assert bge_pos != -1 and chunker_pos != -1
        assert bge_pos < chunker_pos

    def test_warmup_is_idempotent_under_repeated_call(self):
        """Repeated ``warmup_chunker()`` must not raise or do
        duplicate work — the underlying ``_get_chunker`` uses a
        double-checked lock, so a second call returns the cached
        instance immediately. We verify the contract by reading
        the function body (no I/O under test) and by confirming
        the global ``_chunker`` flag is what guards re-entry."""
        text = _read(HYBRID_CHUNKER_PY)
        # _get_chunker uses a double-checked lock.
        assert "_chunker_lock" in text
        # The warmup is a thin wrapper that calls _get_chunker
        # exactly once per invocation.
        m = re.search(
            r"def warmup_chunker\(\) -> bool:\n(?P<body>.*?)(?=\n(?:def |@|\Z))",
            text,
            re.DOTALL,
        )
        assert m is not None
        body = m.group("body")
        # Exactly one _get_chunker call inside the body.
        assert body.count("_get_chunker()") == 1


# ============================================================
# perf-8 — _infer_lock split REJECTED, invariant locked
# ============================================================


class TestInferLockInvariant:
    def test_single_infer_lock_still_present(self):
        text = _read(BGE_M3_PY)
        # The single _infer_lock must NOT be split into _query_infer_lock
        # + _corpus_infer_lock. Splitting is silently unsafe — FlagEmbedding
        # model is not thread-safe and concurrent encode_* calls produce
        # wrong embeddings.
        assert "_infer_lock = threading.Lock()" in text
        # No split aliases.
        assert "_query_infer_lock" not in text
        assert "_corpus_infer_lock" not in text


# ============================================================
# i18n-2 — zh/en key alignment
# ============================================================


class TestI18nKeyAlignment:
    def test_zh_and_en_key_sets_identical(self):
        zh = _extract_keys(_read(ZH_TS))
        en = _extract_keys(_read(EN_TS))
        only_zh = zh - en
        only_en = en - zh
        assert not only_zh, f"zh-only keys (missing in en): {sorted(only_zh)}"
        assert not only_en, f"en-only keys (missing in zh): {sorted(only_en)}"

    def test_total_key_count_at_least_120(self):
        """v2.0.7 baseline was 98 keys. v2.0.8 adds 27, so total
        is 125. We assert >= 120 to allow for future micro-edits
        without breaking this regression test."""
        zh = _extract_keys(_read(ZH_TS))
        assert len(zh) >= 120, f"only {len(zh)} zh keys, expected >= 120"

    def test_v2_0_8_new_keys_present_in_zh(self):
        keys = _extract_keys(_read(ZH_TS))
        expected = [
            # ChatPane
            "chat.modelLoading",
            "chat.emptyTitle",
            "chat.emptyBody1",
            "chat.emptyBody2",
            "chat.bannerAnswerLocal",
            "chat.bannerAnswerWeb",
            "chat.groundingPrefix",
            "chat.pageFooter",
            # Sidebar — sections + thread-doc headers
            "sidebar.historyTitle",
            "sidebar.emptyHistoryDetailed",
            "sidebar.threadDocsTitle",
            "sidebar.emptyDocsDetailed",
            # Sidebar — upload progress chip
            "sidebar.uploadProgressSuffix",
            "sidebar.uploadingDoc",
            "sidebar.parsingDocDetailed",
            "sidebar.indexingChunks",
            "sidebar.chunkCountSuffix",
            # Sidebar — dialog bodies
            "sidebar.deleteDocBody1",
            "sidebar.deleteDocBody2",
            "sidebar.deleteSessionBody1",
            "sidebar.deleteSessionBody2",
            # Sidebar — upload-error dialog
            "sidebar.uploadErrorTitle",
            "sidebar.uploadErrorBody",
            # App.tsx
            "app.wsErrorFallback",
            "app.apiKeyTitle",
            "app.apiKeyBody",
            "app.apiKeyOpenSettings",
        ]
        for k in expected:
            assert k in keys, f"missing v2.0.8 key: {k}"

    def test_v2_0_8_new_keys_present_in_en(self):
        keys = _extract_keys(_read(EN_TS))
        # Every key added in zh must also be in en (mirror test).
        zh_keys = _extract_keys(_read(ZH_TS))
        v2_0_8_keys = {
            "chat.modelLoading",
            "chat.emptyTitle",
            "chat.emptyBody1",
            "chat.emptyBody2",
            "chat.bannerAnswerLocal",
            "chat.bannerAnswerWeb",
            "chat.groundingPrefix",
            "chat.pageFooter",
            "sidebar.historyTitle",
            "sidebar.emptyHistoryDetailed",
            "sidebar.threadDocsTitle",
            "sidebar.emptyDocsDetailed",
            "sidebar.uploadProgressSuffix",
            "sidebar.uploadingDoc",
            "sidebar.parsingDocDetailed",
            "sidebar.indexingChunks",
            "sidebar.chunkCountSuffix",
            "sidebar.deleteDocBody1",
            "sidebar.deleteDocBody2",
            "sidebar.deleteSessionBody1",
            "sidebar.deleteSessionBody2",
            "sidebar.uploadErrorTitle",
            "sidebar.uploadErrorBody",
            "app.wsErrorFallback",
            "app.apiKeyTitle",
            "app.apiKeyBody",
            "app.apiKeyOpenSettings",
        }
        missing_in_en = v2_0_8_keys - keys
        assert not missing_in_en, f"v2.0.8 keys missing in en: {sorted(missing_in_en)}"
        # And the keysets overall must be identical.
        assert zh_keys == keys

    def test_v2_0_7_keys_still_present_no_regression(self):
        """The v2.0.7 i18n-1 keys must still exist in both
        locales after v2.0.8 — we ADD keys, we don't remove."""
        for path in (ZH_TS, EN_TS):
            keys = _extract_keys(_read(path))
            for k in (
                "chat.thinkingDrawerCollapse",
                "chat.thinkingDrawerExpand",
                "chat.inputPlaceholder",
                "chat.stopButtonAria",
                "chat.headerSubtitle",
                "toolCard.collapseDetails",
                "confirm.confirmDefault",
                "modelDownload.title",
                "modelDownload.body",
                "sidebar.dropHint",
                "sidebar.docTitleParsing",
                "sidebar.docTitleEmbedding",
            ):
                assert k in keys, f"{path.name} lost v2.0.7 key: {k}"


class TestI18nPlaceholderSubstitution:
    """Placeholder-bearing keys must use the same ``{name}``
    shape on both sides so ``useLocale``'s ``replace(/\\{...}/g, ...)``
    picks them up correctly."""

    def test_sidebar_delete_doc_body_uses_filename_placeholder(self):
        for path in (ZH_TS, EN_TS):
            text = _read(path)
            assert '"sidebar.deleteDocBody1":' in text
            m = re.search(
                r'"sidebar\.deleteDocBody1":\s*"([^"]*)"', text
            )
            assert m is not None
            assert "{filename}" in m.group(1), (
                f"{path.name}: deleteDocBody1 missing {{filename}} placeholder"
            )

    def test_sidebar_delete_session_body_uses_title_placeholder(self):
        for path in (ZH_TS, EN_TS):
            text = _read(path)
            assert '"sidebar.deleteSessionBody1":' in text
            m = re.search(
                r'"sidebar\.deleteSessionBody1":\s*"([^"]*)"', text
            )
            assert m is not None
            assert "{title}" in m.group(1), (
                f"{path.name}: deleteSessionBody1 missing {{title}} placeholder"
            )

    def test_chat_grounding_prefix_uses_name_placeholder(self):
        for path in (ZH_TS, EN_TS):
            text = _read(path)
            m = re.search(r'"chat\.groundingPrefix":\s*"([^"]*)"', text)
            assert m is not None
            assert "{name}" in m.group(1)

    def test_chat_page_footer_uses_page_placeholder(self):
        for path in (ZH_TS, EN_TS):
            text = _read(path)
            m = re.search(r'"chat\.pageFooter":\s*"([^"]*)"', text)
            assert m is not None
            assert "{page}" in m.group(1)


# ============================================================
# i18n-2 — representative t(...) call sites
# ============================================================


class TestI18nCallSites:
    def test_chatpane_uses_locale(self):
        text = _read(CHATPANE_TSX)
        for k in (
            "t(\"chat.modelLoading\")",
            "t(\"chat.emptyTitle\")",
            "t(\"chat.bannerAnswerLocal\")",
            "t(\"chat.bannerAnswerWeb\")",
            "t(\"chat.groundingPrefix\",",
            "t(\"chat.pageFooter\",",
            "t(\"chat.thinking\")",
            "t(\"chat.searchingWeb\")",
            "t(\"chat.stop\")",
            "t(\"chat.loadingHistory\")",
        ):
            assert k in text, f"ChatPane missing {k}"

    def test_sidebar_uses_locale(self):
        text = _read(SIDEBAR_TSX)
        for k in (
            "t(\"sidebar.historyTitle\")",
            "t(\"sidebar.emptyHistoryDetailed\")",
            "t(\"sidebar.threadDocsTitle\")",
            "t(\"sidebar.emptyDocsDetailed\")",
            "t(\"sidebar.uploadProgressSuffix\",",
            "t(\"sidebar.uploadingDoc\")",
            "t(\"sidebar.parsingDocDetailed\")",
            "t(\"sidebar.indexingChunks\",",
            "t(\"sidebar.chunkCountSuffix\",",
            "t(\"sidebar.deleteDocBody1\",",
            "t(\"sidebar.deleteDocBody2\")",
            "t(\"sidebar.deleteSessionBody1\",",
            "t(\"sidebar.deleteSessionBody2\")",
            "t(\"sidebar.uploadErrorTitle\")",
            "t(\"sidebar.uploadErrorBody\",",
        ):
            assert k in text, f"Sidebar missing {k}"

    def test_app_uses_locale(self):
        text = _read(APP_TSX)
        for k in (
            "t(\"app.wsErrorFallback\")",
            "t(\"app.apiKeyTitle\")",
            "t(\"app.apiKeyBody\")",
            "t(\"app.apiKeyOpenSettings\")",
        ):
            assert k in text, f"App.tsx missing {k}"


# ============================================================
# Cross-cutting — no zh hardcoded strings leak through
# ============================================================


class TestNoHardcodedChinese:
    """After v2.0.8 i18n-2, the three target files must have
    ZERO non-comment CJK lines. Any residual hardcoded zh is a
    bug — it bypasses the locale switcher and freezes the user
    in zh even after they pick English."""

    @pytest.mark.parametrize(
        "path",
        [CHATPANE_TSX, SIDEBAR_TSX, APP_TSX],
        ids=["chatpane", "sidebar", "app"],
    )
    def test_zero_user_visible_zh_lines(self, path: Path):
        text = _read(path)
        leaks = _non_comment_cjk_lines(text)
        assert leaks == [], (
            f"{path.name} still has hardcoded zh on lines: "
            f"{[(i, l[:80]) for i, l in leaks]}"
        )


# ============================================================
# Cross-cutting — wire-format invariants from v2.0.7 preserved
# ============================================================


class TestWireFormatInvariants:
    """v2.0.7 Pydantic Source + history replay renumber must
    still hold after v2.0.8 (we don't touch those code paths)."""

    def test_source_pydantic_model_still_present(self):
        from src.api.schemas import Source as SourceWire

        s = SourceWire(index=1, source_kind="local")
        d = s.model_dump()
        for k in (
            "index", "chunk_id", "filename", "text",
            "score", "source_kind",
        ):
            assert k in d

    def test_history_replay_still_calls_renumber(self):
        text = _read(REPO_ROOT / "src" / "api" / "routes" / "sessions.py")
        assert "renumber_citations_and_sources" in text


__all__ = [
    "TestChunkerWarmup",
    "TestInferLockInvariant",
    "TestI18nKeyAlignment",
    "TestI18nPlaceholderSubstitution",
    "TestI18nCallSites",
    "TestNoHardcodedChinese",
    "TestWireFormatInvariants",
]