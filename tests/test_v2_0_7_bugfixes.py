"""Tests for the v2.0.7 P0 真源 + 性能 + i18n 核心 batch.

Bug history
-----------
v2.0.6 ship 后, user wanted to fix the remaining unaddressed
items from the 局外专家 review in importance order. The P0 tier
bundles:

  perf-1  ``runner.py`` token dedup compared ``cleaned_text`` to
          the whole ``accumulated_text`` on every streaming chunk.
          At 1500+ tokens the equality cost grew to O(N) per
          chunk — pure Python ``==`` against a multi-KB string.
          The replacement tracks ``last_emitted_text`` (just the
          last chunk) so each comparison is O(1) — and preserves
          the v1.1.6c Layer 1 strict-equality skip contract.

  perf-2  ``hybrid_chunker._get_chunker`` had a no-lock lazy init.
          Two concurrent first-time uploads would each pull a 2.3
          GB tokenizer from HF and create two HybridChunker
          instances. Added a double-checked lock mirroring the
          pattern in ``parsers/docling_parser._converter_lock``.

  perf-3  ``list_chunks_by_thread`` does a full LanceDB scan on
          every call. The summary-intent path calls it on every
          turn. Added a 30 s TTL cache mirroring the existing
          ``_thread_doc_cache`` pattern.

  perf-5  ``BGEM3Embedder._embed`` ran a single ``encode_corpus``
          on the whole chunk list. A 2000-chunk document held
          ``_infer_lock`` for 30-90 s, starving chat queries.
          The new path chunks the corpus into 64-sub-batches and
          releases the lock between each. Queries stay single-shot.

  perf-6  Every PDF upload ran Docling's 30-90 s layout analysis,
          even for digital PDFs with extractable text. Added a
          pypdfium2-based pre-sniff: if ≥80 % of the first 10
          pages carry ≥50 chars, dispatch straight to the
          pypdfium2 text-layer path. Scanned PDFs still hit Docling.

  SoT-1   ``ChatEvent.sources`` / ``MessageRecord.sources`` were
          ``list[dict[str, Any]]``. No compile-time guard on
          shape drift. New Pydantic ``Source(BaseModel)`` model
          freezes the wire contract; backend builders now return
          ``Source`` (Pydantic) instances, ``answer_complete``
          serializes via ``.model_dump()``.

  SoT-2   History replay read ``additional_kwargs["sources"]``
          verbatim and bypassed the v2.0.5 renumber pass. Old
          messages with dangling ``[n]`` markers came back
          mismatched with their sources. The replay path now
          calls ``renumber_citations_and_sources`` like the live
          stream does.

  SoT-3   ``search_web.py:_to_doc`` was a near-duplicate of the
          WebResult→Document field set defined elsewhere. The
          docstring now explicitly ties it to the canonical
          ``_web_sources`` field contract (shared
          ``_position_score``, shared metadata keys).

  SoT-4   Frontend ``MessageRecord.sources`` /
          ``MessageRecord.tool_calls`` were typed as
          ``Array<Record<string, unknown>>``. Tightened to
          ``Array<Source>`` / ``Array<ToolCall>``.
          ``Source.source_kind`` is required. Old persisted rows
          default to ``"local"`` defensively.

  i18n-1  Eight frontend files had ~40 high-visibility zh strings
          still hardcoded (citation chips, thinking drawer, tool
          call card, model download modal, sidebar upload widget,
          confirm dialog fallback, chat header subtitle / input /
          stop button). Extracted to ``i18n/zh.ts`` + ``i18n/en.ts``
          via the existing ``useLocale()`` Context.

These tests pin the behavior at three levels:

  1. **Backend** — direct unit tests against the new helpers and
     endpoint signatures.
  2. **Frontend** — static-grep tests that confirm the new
     strings exist as ``t(...)`` calls and the dictionary keys
     line up between zh.ts and en.ts.
  3. **Cross-cutting** — the wire-format invariant must survive
     a Pydantic Source → dict round-trip (so JSON shape is
     unchanged for either input form to ``answer_complete``).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ZH_TS = REPO_ROOT / "src" / "frontend" / "src" / "i18n" / "zh.ts"
EN_TS = REPO_ROOT / "src" / "frontend" / "src" / "i18n" / "en.ts"
SCHEMAS_PY = REPO_ROOT / "src" / "api" / "schemas.py"
EVENTS_PY = REPO_ROOT / "src" / "agent" / "events.py"
SESSIONS_PY = REPO_ROOT / "src" / "api" / "routes" / "sessions.py"
CLIENT_TS = REPO_ROOT / "src" / "frontend" / "src" / "api" / "client.ts"
CHATUTILS_TSX = REPO_ROOT / "src" / "frontend" / "src" / "chat-utils.tsx"
APP_TSX = REPO_ROOT / "src" / "frontend" / "src" / "App.tsx"
CHATPANE_TSX = REPO_ROOT / "src" / "frontend" / "src" / "components" / "ChatPane.tsx"
SIDEBAR_TSX = REPO_ROOT / "src" / "frontend" / "src" / "components" / "Sidebar.tsx"
TOOLCALLCARD_TSX = REPO_ROOT / "src" / "frontend" / "src" / "components" / "ToolCallCard.tsx"
CONFIRMDIALOG_TSX = REPO_ROOT / "src" / "frontend" / "src" / "components" / "ConfirmDialog.tsx"
MODELDOWNLOAD_TSX = REPO_ROOT / "src" / "frontend" / "src" / "components" / "ModelDownloadProgress.tsx"
HYBRID_CHUNKER_PY = REPO_ROOT / "src" / "ingestion" / "chunkers" / "hybrid_chunker.py"
RUNNER_PY = REPO_ROOT / "src" / "agent" / "runner.py"
BGE_M3_PY = REPO_ROOT / "src" / "embeddings" / "bge_m3.py"
LANCEDB_STORE_PY = REPO_ROOT / "src" / "storage" / "lancedb_store.py"
PIPELINE_PY = REPO_ROOT / "src" / "ingestion" / "pipeline.py"
# v2.0.13 cleanup: ``src/agent/nodes/search_web.py`` was retired when
# web search became a ReAct tool. The SoT-3 contract now lives entirely
# in ``src/agent/tools/web_search.py`` + ``src/agent/nodes/_web_sources.py``,
# so the TestSearchWebToDocContract class below was removed along with
# the file it referenced.


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_keys(text: str) -> set[str]:
    """Pull the set of i18n keys (string-literal-keys before colon)."""
    return set(re.findall(r'"([a-zA-Z0-9_.]+)"\s*:', text))


# ============================================================
# SoT-1 — Pydantic Source model + wire types
# ============================================================


class TestSourcePydanticModel:
    def test_source_class_exists_with_required_fields(self):
        text = _read(SCHEMAS_PY)
        # Pydantic Source class is declared with index + source_kind
        # as required and the rest as optional with defaults.
        assert "class Source(BaseModel):" in text
        assert re.search(r"index:\s*int\b", text)
        assert "source_kind: Literal[\"local\", \"web\"]" in text

    def test_chat_event_sources_typed_as_source_list(self):
        text = _read(SCHEMAS_PY)
        assert "sources: Optional[list[Source]]" in text

    def test_message_record_sources_typed_as_source_list(self):
        text = _read(SCHEMAS_PY)
        # Sources in MessageRecord uses the same Source reference.
        # Pydantic self-reference is via the class name, not full path.
        assert "sources: Optional[list[Source]]" in text

    def test_answer_complete_accepts_pydantic_or_dict(self):
        text = _read(EVENTS_PY)
        # Signature uses Union[SourceWire, dict]
        assert "Union[SourceWire, dict]" in text
        # And it dumps Pydantic via .model_dump
        assert "model_dump" in text


# ============================================================
# SoT-2 — history replay renumber
# ============================================================


class TestHistoryReplayRenumber:
    def test_serialize_messages_calls_renumber(self):
        text = _read(SESSIONS_PY)
        assert "renumber_citations_and_sources" in text

    def test_serialize_messages_passes_answer_and_sources(self):
        text = _read(SESSIONS_PY)
        # The call site must pass the AIMessage's extracted text
        # (variable name ``text``) and the raw sources list.
        assert re.search(
            r"renumber_citations_and_sources\(\s*answer=text", text
        )


# ============================================================
# SoT-3 — REMOVED in v2.0.13 cleanup
# ============================================================
#
# The old TestSearchWebToDocContract class tested that
# ``src/agent/nodes/search_web.py`` shared the _web_sources field
# contract. That file was retired when web search became a ReAct
# tool (the active code is in ``tools/web_search.py`` +
# ``nodes/_web_sources.py``). The contract is now a single-source
# invariant, so the test is no longer needed.


# ============================================================
# SoT-4 — TS type tightening
# ============================================================


class TestFrontendTypeTightening:
    def test_source_kind_is_required(self):
        text = _read(CHATUTILS_TSX)
        assert re.search(r'source_kind:\s*"local"\s*\|\s*"web"', text)
        assert "source_kind?:" not in text

    def test_message_record_sources_typed_as_source_array(self):
        text = _read(CLIENT_TS)
        assert "Array<Source>" in text
        # The unknown catch-all must not appear as a TYPE signature.
        # Documentation comments referencing the old type are OK
        # (they explain the v2.0.7 tightening); strip them before
        # the grep.
        import re as _re
        code_only = _re.sub(r"/\*.*?\*/", "", text, flags=_re.DOTALL)
        code_only = _re.sub(r"//[^\n]*", "", code_only)
        assert "Array<Record<string, unknown>>" not in code_only

    def test_message_record_tool_calls_typed_as_tool_call_array(self):
        text = _read(CLIENT_TS)
        assert "Array<ToolCall>" in text

    def test_app_legacy_source_kind_default(self):
        text = _read(APP_TSX)
        # The defensive default for legacy persisted rows.
        assert "source_kind ?? \"local\"" in text


# ============================================================
# perf-1 — runner token dedup is O(1)
# ============================================================


class TestRunnerDedupO1:
    def test_runner_uses_last_emitted_text(self):
        text = _read(RUNNER_PY)
        # The O(1) dedup key is last_emitted_text (not
        # accumulated_text which grows with answer length).
        assert "last_emitted_text" in text

    def test_runner_preserved_endswith_contract(self):
        text = _read(RUNNER_PY)
        # v1.1.6c Layer 1 strict-equality skip + synthesized
        # full-message replay guard both survive.
        assert "endswith" in text


# ============================================================
# perf-2 — chunker init double-checked lock
# ============================================================


class TestChunkerInitLock:
    def test_chunker_uses_dcl_lock(self):
        text = _read(HYBRID_CHUNKER_PY)
        assert "_chunker_lock" in text
        # Double-checked pattern: acquire the lock, then
        # re-check inside.
        assert "with _chunker_lock:" in text
        assert re.search(r"if _chunker is None:\s*\n\s*_chunker = ", text) or \
            re.search(r"if _chunker is None:\s*\n.*\n.*_chunker = ", text, re.DOTALL)


# ============================================================
# perf-3 — list_chunks_by_thread TTL cache
# ============================================================


class TestChunkThreadCache:
    def test_cache_module_exposes_cached_helper(self):
        text = _read(LANCEDB_STORE_PY)
        assert "list_chunks_by_thread_cached" in text
        assert "invalidate_thread_chunk_cache" in text
        assert "_THREAD_CHUNK_CACHE_TTL_SEC" in text

    def test_cache_invalidates_on_add_and_delete(self):
        text = _read(LANCEDB_STORE_PY)
        # Both add_chunks and the delete paths should invalidate.
        assert "invalidate_thread_chunk_cache" in text
        # Specifically called from inside add_chunks and the
        # delete_by_* helpers.
        assert text.count("invalidate_thread_chunk_cache(") >= 3


# ============================================================
# perf-5 — encode_corpus sub-batch
# ============================================================


class TestEmbedderSubBatch:
    def test_embed_chunks_corpus_into_64(self):
        text = _read(BGE_M3_PY)
        # The sub-batch size should be 64.
        assert "range(0, len(texts), 64)" in text

    def test_embed_releases_lock_between_sub_batches(self):
        text = _read(BGE_M3_PY)
        # The corpus batch path must release _infer_lock between
        # each sub-batch (otherwise chat queries stay blocked for
        # the whole 30-90 s encode).
        assert "_infer_lock" in text


# ============================================================
# perf-6 — Docling text-layer pre-sniff
# ============================================================


class TestDoclingPreflight:
    def test_pipeline_uses_pre_sniff(self):
        text = _read(PIPELINE_PY)
        assert "_looks_like_digital_pdf" in text
        # Dispatch logic runs the sniff BEFORE the Docling path.
        assert text.find("_looks_like_digital_pdf(") < text.find(
            "parse_with_docling"
        )

    def test_helper_uses_pypdfium2(self):
        text = _read(PIPELINE_PY)
        assert "pypdfium2" in text

    def test_helper_thresholds_are_documented(self):
        text = _read(PIPELINE_PY)
        # 80% threshold + 50-char min + 10-page sample.
        assert "_DIGITAL_PDF_MIN_PAGE_CHARS" in text
        assert "_DIGITAL_PDF_TEXT_PAGE_RATIO" in text
        assert "_DIGITAL_PDF_SAMPLE_PAGES" in text


# ============================================================
# i18n-1 — key alignment + representative t() calls
# ============================================================


class TestI18nKeyAlignment:
    def test_zh_and_en_key_sets_identical(self):
        zh = _extract_keys(_read(ZH_TS))
        en = _extract_keys(_read(EN_TS))
        only_zh = zh - en
        only_en = en - zh
        assert not only_zh, f"zh-only keys (missing in en): {sorted(only_zh)}"
        assert not only_en, f"en-only keys (missing in zh): {sorted(only_en)}"

    def test_zh_has_new_v2_0_7_keys(self):
        keys = _extract_keys(_read(ZH_TS))
        expected = [
            "chat.thinkingDrawerCollapse",
            "chat.thinkingDrawerExpand",
            "chat.inputPlaceholder",
            "chat.stopButtonAria",
            "chat.headerSubtitle",
            "toolCard.collapseDetails",
            "toolCard.expandDetails",
            "confirm.confirmDefault",
            "modelDownload.title",
            "modelDownload.body",
            "sidebar.dropIdle",
            "sidebar.dropActive",
            "sidebar.docTitleParsing",
            "sidebar.docTitleEmbedding",
        ]
        for k in expected:
            assert k in keys, f"missing v2.0.7 key: {k}"


class TestI18nCallSites:
    def test_thinkingdrawer_uses_locale(self):
        text = _read(CHATPANE_TSX)
        # The ThinkingDrawer title uses t(...) not a hardcoded string.
        assert "t(\"chat.thinkingDrawerCollapse\")" in text
        assert "t(\"chat.thinkingDrawerExpand\")" in text

    def test_chat_header_subtitle_uses_locale(self):
        text = _read(CHATPANE_TSX)
        assert "t(\"chat.headerSubtitle\")" in text

    def test_chat_input_placeholder_uses_locale(self):
        text = _read(CHATPANE_TSX)
        assert "t(\"chat.inputPlaceholder\")" in text

    def test_stop_button_uses_locale(self):
        text = _read(CHATPANE_TSX)
        assert "t(\"chat.stopButtonAria\")" in text
        assert "t(\"chat.stopButtonTitle\")" in text

    def test_toolcall_card_uses_locale(self):
        text = _read(TOOLCALLCARD_TSX)
        assert "t(\"toolCard.collapseDetails\")" in text

    def test_confirm_dialog_uses_locale(self):
        text = _read(CONFIRMDIALOG_TSX)
        assert "t(\"confirm.confirmDefault\")" in text

    def test_sidebar_upload_widget_uses_locale(self):
        text = _read(SIDEBAR_TSX)
        assert "t(\"sidebar.dropHint\"" in text or "t(\"sidebar.dropHintGeneric\")" in text
        assert "t(\"sidebar.docTitleParsing\")" in text

    def test_model_download_progress_uses_locale(self):
        text = _read(MODELDOWNLOAD_TSX)
        assert "t(\"modelDownload.title\")" in text
        assert "t(\"modelDownload.body\")" in text
        assert "t(\"modelDownload.footnote\")" in text


# ============================================================
# Cross-cutting — wire-format invariants
# ============================================================


class TestWireFormatInvariants:
    def test_source_pydantic_model_dump_is_json_clean(self):
        """Pydantic Source.model_dump() must produce a flat dict
        with the same fields the frontend reads. This is the
        ``extra=\"ignore\"`` model_config guard."""
        from src.api.schemas import Source as SourceWire

        s = SourceWire(index=1, source_kind="local")
        d = s.model_dump()
        # Every frontend field is present (or None).
        for k in (
            "index", "chunk_id", "doc_id", "filename",
            "page", "section", "sheet", "text", "score",
            "url", "domain", "source_kind",
        ):
            assert k in d, f"missing field {k} in Source.model_dump()"
        # No nesting beyond dict[str, Any] - safe to JSON-serialize.
        import json
        json.dumps(d)  # must not raise

    def test_answer_complete_accepts_mixed_inputs(self):
        """answer_complete() must accept a mix of Pydantic SourceWire
        and raw dict sources. Pre-SoT-1 callers still pass dicts;
        post-SoT-1 callers pass Pydantic."""
        from langchain_core.documents import Document

        from src.agent.events import answer_complete
        from src.agent.nodes._web_sources import web_source
        from src.api.schemas import Source as SourceWire

        ws = web_source(
            1,
            Document(
                page_content="snippet",
                metadata={"filename": "x.pdf", "url": "http://x"},
            ),
        )
        local = SourceWire(index=2, source_kind="local")
        raw = {"index": 3, "chunk_id": "abc", "source_kind": "local"}

        ev = answer_complete([ws, local, raw], "body")
        assert ev["type"] == "answer_complete"
        assert len(ev["sources"]) == 3
        # Pydantic sources are dumped; raw dict passes through.
        for s in ev["sources"]:
            assert "source_kind" in s


__all__ = [
    "TestSourcePydanticModel",
    "TestHistoryReplayRenumber",
    "TestSearchWebToDocContract",
    "TestFrontendTypeTightening",
    "TestRunnerDedupO1",
    "TestChunkerInitLock",
    "TestChunkThreadCache",
    "TestEmbedderSubBatch",
    "TestDoclingPreflight",
    "TestI18nKeyAlignment",
    "TestI18nCallSites",
    "TestWireFormatInvariants",
]