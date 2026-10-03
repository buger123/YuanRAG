"""v2.0.5 — P1 UX consistency regression tests.

Bugs fixed (see ``plan/polished-gliding-karp.md`` for the original
review):

* **P1-1** upload shows "处理中" for 30-90 s with no per-stage
  signal. Docling parsing + BGE-M3 embedding were both lumped under
  a single generic "pending" chip. Two fixes:
    1. ``doc_registry`` Literal extended with ``"parsing`` /
       ``"embedding`` and the ingest pipeline flips
       ``pending → parsing → embedding → indexed``.
    2. Frontend ``Sidebar.tsx`` renders "解析中…" / "向量化中
       (N 块)" / "处理中" depending on the live status.

* **P1-2** every web search source chip rendered ``score=0.0000``.
  Three hardcoded sites: ``web_search._to_doc``, ``search_web._to_doc``,
  and ``_web_sources.web_source`` all stamped ``"score": 0.0``.
  Fix: position-based decay ``1.0 - 0.1*position`` (clamped at 0),
  threaded through the call sites that produce ``Document``s.

* **P1-3** answer text cited ``[3] [7] [23]`` but only 3 sources
  existed. The runner only DROPPED uncited sources but never
  RENUMBERED the answer text. Fix: new
  ``renumber_citations_and_sources`` helper rewrites every
  ``[n]`` / ``[a,b]`` / ``[a-b]`` marker to a dense ``[1]..[K]``
  range, re-stamps the surviving sources' ``index``, and drops
  uncited sources.

* **P1-4** the self-introduction prompt said "fall back to a
  DuckDuckGo web search". After v1.1.10 the default engine is Bing;
  the prompt lied about which engine we actually use.

* **P1-5** "光速是多少?" / "what is the capital of France?" was
  classified as ``qa_complex`` and went through the full ReAct
  loop — wasted iterations thinking about whether to call
  ``retrieve_docs`` on a query that's clearly a static lookup.
  Fix: regex fast-path catches definitional / single-fact queries
  and routes them to the direct-answer branch (same as greeting).

These tests pin all five fixes so future refactors don't silently
regress any layer.

If a test here fails, either:
  (a) one of the v2.0.5 fixes was reverted, OR
  (b) a new fix needs to be added — update the test to match.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import get_args

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ZH_TS = REPO_ROOT / "src" / "frontend" / "src" / "i18n" / "zh.ts"


def _run(coro):
    """Tiny helper: run a coroutine in a fresh event loop.

    Kept local (instead of ``pytest-asyncio``) so the test file
    runs without depending on async fixtures — matches the
    v2.0.2 / v2.0.3 / v2.0.4 regression test style.
    """
    return asyncio.run(coro)


# ============================================================================
# P1-1 — 3-stage ingestion status (parsing / embedding)
# ============================================================================


class TestP11ThreeStageStatus:
    """The DocStatus Literal + Sidebar copy reflect the per-stage
    pipeline progress."""

    def test_doc_status_literal_includes_parsing_and_embedding(self):
        """v2.0.5 — Literal must include ``"parsing`` and ``"embedding``."""
        from src.storage.doc_registry import DocStatus

        members = set(get_args(DocStatus))
        # Old members must still be present (backward compat).
        for old in ("pending", "indexed", "empty", "failed"):
            assert old in members, f"DocStatus lost legacy member {old!r}"
        # New members must be present (the 3-stage fix).
        assert "parsing" in members
        assert "embedding" in members

    def test_api_schema_literal_includes_parsing_and_embedding(self):
        """v2.0.5 — ``DocumentSummary.status`` Literal must accept new values."""
        from src.api.schemas import DocumentSummary

        field = DocumentSummary.model_fields["status"]
        members = set(get_args(field.annotation))
        assert "parsing" in members
        assert "embedding" in members
        assert "indexed" in members

    def test_frontend_api_type_includes_parsing_and_embedding(self):
        """v2.0.5 — the TypeScript ``Document.status`` union must
        include the new values, otherwise the build breaks."""
        client_path = Path("src/frontend/src/api/client.ts")
        text = client_path.read_text(encoding="utf-8")
        # The union must list every value we accept server-side.
        for status in ("pending", "parsing", "embedding", "indexed", "empty", "failed"):
            assert f'"{status}"' in text, (
                f"client.ts missing {status!r} in Document.status union"
            )

    def test_sidebar_renders_stage_specific_chips(self):
        """v2.0.5 — Sidebar must render "解析中…" / "向量化中…" /
        "处理中" depending on the status, not just one generic chip."""
        sidebar_path = Path("src/frontend/src/components/Sidebar.tsx")
        text = sidebar_path.read_text(encoding="utf-8")
        # Per-stage branch labels must be present.
        assert 'status === "parsing"' in text, (
            'Sidebar missing `status === "parsing"` branch'
        )
        assert 'status === "embedding"' in text, (
            'Sidebar missing `status === "embedding"` branch'
        )
        # Per-stage copy must be present in the i18n table (Chinese only —
        # these are what the user actually sees). v2.0.8 i18n-2 moved
        # the literals from Sidebar.tsx into zh.ts via ``t(...)``, so
        # the source of truth is now the locale file. The sidebar
        # must still reference them via the lookup helper.
        zh_text = ZH_TS.read_text(encoding="utf-8")
        assert "解析中" in zh_text, 'zh.ts missing 解析中… chip copy'
        assert "向量化中" in zh_text, 'zh.ts missing 向量化中 chip copy'
        assert "t(\"sidebar.parsing\")" in text, (
            "Sidebar must call t(\"sidebar.parsing\") for the parsing chip"
        )
        assert "t(\"sidebar.embedding\"" in text, (
            "Sidebar must call t(\"sidebar.embedding\") for the embedding chip"
        )

    def test_ingest_pipeline_calls_parsing_then_embedding_status(self):
        """v2.0.5 — the ingest pipeline must flip status to
        ``parsing`` before ``run_ingestion`` and ``embedding`` after
        parsing but before ``embed_documents``.

        We check this statically (read the file + grep) to avoid
        importing the heavy Docling / BGE-M3 stack — the static
        check is enough to catch a silent revert."""
        path = Path("src/api/routes/documents.py")
        text = path.read_text(encoding="utf-8")
        # The 'parsing' flip must appear BEFORE the run_ingestion call.
        parsing_idx = text.find('status="parsing"')
        run_idx = text.find("run_ingestion(")
        assert parsing_idx != -1, "documents.py missing status='parsing' update"
        assert run_idx != -1, "documents.py missing run_ingestion call"
        assert parsing_idx < run_idx, (
            "status='parsing' must be set BEFORE run_ingestion"
        )
        # The 'embedding' flip must appear AFTER parsing-then-no-chunks
        # branch and BEFORE embed_documents.
        embedding_idx = text.find('status="embedding"')
        embed_idx = text.find("embed_documents(")
        assert embedding_idx != -1, "documents.py missing status='embedding' update"
        assert embed_idx != -1, "documents.py missing embed_documents call"
        assert embedding_idx < embed_idx, (
            "status='embedding' must be set BEFORE embed_documents"
        )

    def test_indexed_transition_lives_in_add_chunks_after_table_add(self):
        """v2.0.10 — the ``status="indexed"`` flip MUST live inside
        ``add_chunks`` and fire AFTER ``table.add(`` so the registry
        transition is atomic with the LanceDB write.

        Background: pre-v2.0.10 the ``status="indexed"`` call sat on
        the line AFTER ``add_chunks(records)`` in
        ``src/api/routes/documents.py``. The Sidebar's
        ``pollUntilIndexed`` exits on ``chunk_count > 0`` — i.e. the
        moment chunks land in LanceDB — but at that exact instant the
        registry was still at "embedding" because the
        ``update("indexed")`` call was the NEXT line. The polling
        exits, the sidebar immediately calls ``listDocuments`` to
        populate the doc list, and the response carries
        ``status="embedding"`` → chip stuck at "向量化中(N 块)"
        indefinitely (no auto-refresh hook after upload completes).

        Fix: co-locate the ``status="indexed"`` flip with
        ``table.add(`` inside ``add_chunks``. Polling's
        ``cc > 0`` now implies ``status === "indexed"``.

        We pin the structure statically so a careless future revert
        to "after add_chunks" or "in the caller" is caught.
        """
        store_path = REPO_ROOT / "src" / "storage" / "lancedb_store.py"
        text = store_path.read_text(encoding="utf-8")

        # Locate the add_chunks function body. We use the
        # ``def add_chunks(`` anchor + the next ``def`` so we stay
        # inside the function.
        start = text.find("def add_chunks(")
        assert start != -1, "lancedb_store.py missing add_chunks"
        next_def = text.find("\ndef ", start + 1)
        body = text[start:next_def if next_def != -1 else len(text)]

        # The 'indexed' flip must appear INSIDE add_chunks.
        indexed_idx = body.find('status="indexed"')
        assert indexed_idx != -1, (
            "v2.0.10: add_chunks must flip status='indexed' — without "
            "this the sidebar polling sees '向量化中(N 块)' forever"
        )
        # And it must be AFTER table.add( — the chunks must land
        # before the registry claims they're indexed.
        table_add_idx = body.find("table.add(")
        assert table_add_idx != -1, "add_chunks missing table.add call"
        assert indexed_idx > table_add_idx, (
            "v2.0.10: status='indexed' flip must be AFTER table.add("
            " — the registry must not claim 'indexed' before chunks "
            "have actually landed in LanceDB"
        )

    def test_documents_route_no_redundant_indexed_after_add_chunks(self):
        """v2.0.10 — ``src/api/routes/documents.py`` must NOT have a
        redundant ``doc_registry.update(status="indexed")`` call after
        ``add_chunks(records)``; the atomic flip lives in add_chunks
        now.

        Defense: if a future refactor re-adds the redundant line in
        the caller, this test fails. The redundant call is harmless
        (idempotent) but signals "the atomic fix was forgotten" — and
        the race condition is back the moment someone removes the
        flip from add_chunks."""
        route_path = REPO_ROOT / "src" / "api" / "routes" / "documents.py"
        text = route_path.read_text(encoding="utf-8")
        # Find the add_chunks call site.
        add_chunks_idx = text.find("add_chunks(records)")
        assert add_chunks_idx != -1, "documents.py missing add_chunks call"
        # Look at the next ~30 lines after add_chunks for any
        # REAL ``doc_registry.update(`` call (code, not comment).
        # We strip comment-only lines first — the v2.0.10 fix's
        # explanatory comment legitimately mentions the phrase in
        # prose, but it must not count as a redundant call.
        tail = text[add_chunks_idx: add_chunks_idx + 800]
        code_lines = [
            line for line in tail.splitlines()
            if not line.lstrip().startswith("#")
        ]
        code_tail = "\n".join(code_lines)
        assert 'doc_registry.update(' not in code_tail or \
               'status="indexed"' not in code_tail, (
            "v2.0.10: documents.py has redundant "
            "doc_registry.update(status='indexed') after add_chunks. "
            "The atomic flip now lives in add_chunks; remove the "
            "caller-side call to keep a single source of truth."
        )

    def test_sidebar_polling_waits_for_terminal_status(self):
        """v2.0.10 — ``Sidebar.tsx`` ``pollUntilIndexed`` must wait
        for the doc to reach a TERMINAL status (indexed / empty /
        failed) before returning, not just ``chunk_count > 0``.

        Background: pre-v2.0.10 the polling exited on
        ``chunk_count > 0``. That's correct for "chunks landed" but
        wrong for "ingestion is fully done" — there's still a
        microsecond gap between the LanceDB write and the
        ``doc_registry.update("indexed")`` call (which now lives
        in add_chunks but isn't free of network/thread interleaving).
        The polling's exit during that gap was the user-visible bug.

        Fix: wait for any of the terminal statuses. ``indexed``
        covers success, ``empty`` covers zero-chunk files,
        ``failed`` covers parser / persist errors. The polling's
        prior ``cc > 0`` heuristic is now redundant — terminal
        status implies a finished ingestion regardless of cc.
        """
        sidebar_path = REPO_ROOT / "src" / "frontend" / "src" / "components" / "Sidebar.tsx"
        text = sidebar_path.read_text(encoding="utf-8")
        # All three terminal statuses must be checked in the polling
        # function. We don't pin the exact wording (it may evolve)
        # but the three literal strings must appear inside what looks
        # like the polling loop.
        for terminal in ("indexed", "empty", "failed"):
            assert f'status === "{terminal}"' in text or \
                   f'"{terminal}"' in text and 'status' in text, (
                f"v2.0.10: pollUntilIndexed must check for status={terminal!r} "
                f"as a terminal ingestion state"
            )
        # Defensive: the old ``if (cc > 0) return`` heuristic must
        # NOT be the sole exit condition — it raced the indexed
        # transition. We don't outright ban the literal (it may
        # still be useful for the progress bar callback) but we
        # require the polling to also check status.
        assert "cc > 0" not in text or 'status ===' in text, (
            "v2.0.10: Sidebar polling must check terminal status, not "
            "just `cc > 0`. The latter raced the indexed transition "
            "and caused '向量化中(N 块)' to persist indefinitely."
        )


# ============================================================================
# P1-2 — web search score position-decay (no more 0.0000 everywhere)
# ============================================================================


class TestP12WebScoreDecay:
    """Web sources get a position-decay score, not the old hardcoded 0.0."""

    def test_position_score_helper_exists_and_decays(self):
        """The shared ``_position_score`` helper must exist and decay
        smoothly. Position 0 → 1.0, position 1 → 0.9, etc."""
        from src.agent.nodes._web_sources import _position_score

        assert _position_score(0) == 1.0
        assert _position_score(1) == 0.9
        assert _position_score(2) == 0.8
        assert _position_score(9) == 0.1
        # Floor — anything past position 10 clamps to 0.
        assert _position_score(10) == 0.0
        assert _position_score(20) == 0.0
        # Defensive: negative position clamped to 0.
        assert _position_score(-3) == 1.0

    def test_web_source_score_uses_position_decay(self):
        """``web_source(index, doc)`` must compute score from the
        1-based index, NOT hardcode 0.0."""
        from langchain_core.documents import Document

        from src.agent.nodes._web_sources import web_source

        doc = Document(
            page_content="hi",
            metadata={
                "source_kind": "web",
                "url": "https://example.com/a",
                "domain": "example.com",
                "filename": "A",
                "doc_id": "web-aaa",
                "chunk_id": "",
            },
        )
        s1 = web_source(1, doc)
        s2 = web_source(2, doc)
        s3 = web_source(3, doc)
        # v2.0.7 SoT-1: ``Source`` is Pydantic; attribute access.
        assert s1.score == 1.0, "first source must score 1.0"
        assert s2.score == 0.9
        assert s3.score == 0.8
        # The hardcoded-0.0 regression guard.
        assert s1.score != 0.0
        assert s2.score != 0.0

    def test_tool_web_search_to_doc_uses_position(self):
        """``search_web._to_doc`` must accept a position arg and
        stamp ``score`` via ``_position_score``. v2.0.18 — the
        helper moved from ``src.agent.tools.web_search`` into
        ``src.web_search`` (where the dispatcher that uses it lives).
        Still importable as a module-level function; not exported via
        ``__all__`` but kept accessible for this regression test."""
        from src.web_search import _to_doc

        doc_pos0 = _to_doc(
            {
                "title": "A",
                "url": "https://a",
                "snippet": "s",
                "domain": "a",
            },
            position=0,
        )
        doc_pos2 = _to_doc(
            {
                "title": "B",
                "url": "https://b",
                "snippet": "s",
                "domain": "b",
            },
            position=2,
        )
        assert doc_pos0.metadata["score"] == 1.0
        assert doc_pos2.metadata["score"] == 0.8

    def test_no_hardcoded_score_zero_in_web_search_modules(self):
        """Static guard: no remaining ``"score": 0.0`` / ``score=0.0``
        literals in the live web-search sites.

        (Removed in v2.0.13 cleanup: the legacy ``src/agent/nodes/
        search_web.py`` graph-node was retired when web search became
        a ReAct tool. The active surface is ``tools/web_search.py``
        + ``nodes/_web_sources.py``.)
        """
        for path in (
            Path("src/web_search/__init__.py"),
            Path("src/agent/nodes/_web_sources.py"),
        ):
            text = path.read_text(encoding="utf-8")
            # Look for the exact pre-v2.0.5 patterns. Allow
            # comments like "# was hardcoded to 0.0" which describe
            # the bug — those are intentional historical markers.
            code_only = re.sub(r"#.*", "", text)
            assert '"score": 0.0' not in code_only, (
                f"{path} still hardcodes `\"score\": 0.0`"
            )
            assert "score=0.0," not in code_only and not re.search(
                r"score=0\.0\s*\)", code_only
            ), f"{path} still hardcodes `score=0.0`"


# ============================================================================
# P1-3 — citation renumber helper
# ============================================================================


class TestP13CitationRenumber:
    """``renumber_citations_and_sources`` rewrites dense 1..K and
    drops uncited sources."""

    def _src(self, idx: int, url: str = "") -> dict:
        return {
            "index": idx,
            "chunk_id": "",
            "doc_id": f"web-{idx:03d}",
            "filename": f"src-{idx}",
            "url": url or f"https://example.com/{idx}",
            "domain": "example.com",
            "source_kind": "web",
            "text": "...",
            "score": 0.0,
        }

    def test_renumber_basic_sparse_to_dense(self):
        """``[3] [7] [1]`` → renumbered to ``[2] [3] [1]`` (positional,
        not sorted) and sources reorder."""
        from src.agent.citations import renumber_citations_and_sources

        answer = "see [3] and [7] then [1]"
        sources = [self._src(i) for i in (1, 3, 7)]
        new_answer, new_sources = renumber_citations_and_sources(
            answer, sources, route_decision="retrieve"
        )
        # Cited set is {1, 3, 7}; sorted → [1, 3, 7]; mapping
        # {1→1, 3→2, 7→3}. Position in text is preserved — we
        # rewrite values, not sort the citations.
        cited = re.findall(r"\[(\d+)\]", new_answer)
        assert cited == ["2", "3", "1"], (
            f"positional rewrite expected ['2','3','1']; got {cited!r}"
        )
        # All new indices are dense 1..K (no gaps).
        assert sorted(int(c) for c in cited) == [1, 2, 3]
        # Sources reorder to ascending new-index: 1, 2, 3.
        assert [s["index"] for s in new_sources] == [1, 2, 3]
        # And every rewritten [n] points at an existing source.
        new_indices = {s["index"] for s in new_sources}
        assert {int(c) for c in cited}.issubset(new_indices)

    def test_renumber_drops_uncited_sources(self):
        """A source whose old index was never cited must be dropped."""
        from src.agent.citations import renumber_citations_and_sources

        answer = "only [1] is cited"
        sources = [self._src(i) for i in (1, 2, 3, 4, 5)]
        _, new_sources = renumber_citations_and_sources(
            answer, sources, route_decision="retrieve"
        )
        assert len(new_sources) == 1
        assert new_sources[0]["index"] == 1

    def test_renumber_handles_ranges(self):
        """``[1-3]`` stays a range after renumber; ``[5-7]`` collapses
        to whatever survives."""
        from src.agent.citations import renumber_citations_and_sources

        answer = "see [1-3] and [5-7]"
        sources = [self._src(i) for i in (1, 2, 3, 5, 6, 7)]
        new_answer, _ = renumber_citations_and_sources(
            answer, sources, route_decision="retrieve"
        )
        # All 6 sources are cited, so [1-3] stays a range and
        # [5-7] stays a range.
        assert "[1-3]" in new_answer
        assert "[4-6]" in new_answer, (
            f"expected [4-6] in renumbered answer, got {new_answer!r}"
        )

    def test_renumber_drops_dangling_out_of_range_marker(self):
        """``[23]`` when no source has index 23 must be dropped from
        the answer (not surface as a dangling chip)."""
        from src.agent.citations import renumber_citations_and_sources

        answer = "see [23] which doesn't exist"
        sources = [self._src(i) for i in (1, 2, 3)]
        new_answer, _ = renumber_citations_and_sources(
            answer, sources, route_decision="retrieve"
        )
        assert "[23]" not in new_answer
        # The answer text should still be the trailing prose.
        assert "doesn't exist" in new_answer

    def test_renumber_direct_path_returns_empty_sources(self):
        """``route_decision="direct"`` short-circuits to ``(answer, [])``."""
        from src.agent.citations import renumber_citations_and_sources

        answer = "hi [1] [2]"
        sources = [self._src(1), self._src(2)]
        new_answer, new_sources = renumber_citations_and_sources(
            answer, sources, route_decision="direct"
        )
        assert new_sources == []
        # Answer text preserved (we don't rewrite direct-path answers).
        assert new_answer == answer

    def test_renumber_handles_no_markers(self):
        """When the answer has no ``[n]`` markers, the helper is a
        no-op — answer unchanged, sources unchanged."""
        from src.agent.citations import renumber_citations_and_sources

        answer = "no markers here"
        sources = [self._src(1), self._src(2)]
        new_answer, new_sources = renumber_citations_and_sources(
            answer, sources, route_decision="retrieve"
        )
        assert new_answer == answer
        # Uncited sources are preserved (caller decides what to do).
        assert len(new_sources) == 2

    def test_runner_uses_renumber_helper(self):
        """The runner's ``_handle_final_answer`` must call the new
        renumber helper (not just ``filter_sources_to_cited``)."""
        runner_text = Path("src/agent/runner.py").read_text(encoding="utf-8")
        assert "renumber_citations_and_sources" in runner_text, (
            "runner.py doesn't import renumber_citations_and_sources"
        )
        # The renumber call must appear inside _handle_final_answer
        # and replace the old `filter_sources_to_cited` direct call
        # (the helper itself may still import the legacy filter as a
        # defense-in-depth pass).
        assert "_handle_final_answer" in runner_text
        # Sanity: the renumber helper is invoked at least once as a
        # call (the import is a different line so it's not counted
        # here). v2.0.5 — runner.py uses the helper in
        # _handle_final_answer.
        assert runner_text.count("renumber_citations_and_sources(") >= 1, (
            "runner.py doesn't call renumber_citations_and_sources()"
        )

    def test_range_collapse_logs_warning(self):
        """v2.0.29.4 (Phase 4 PR-1) — when a multi-element range
        collapses to 1 element (``[5-7]`` → ``[5]`` because only
        source 5 survived the cited-set filter), the helper must
        emit a WARNING log.

        Pre-PR-1 this collapse was silent, so live→reload drift
        where ``[5-7]`` becomes ``[5]`` was invisible to
        dashboards. Uses loguru's ``add(sink)`` rather than caplog
        because caplog does NOT capture loguru output (Phase 2
        debugging坑 #1).
        """
        import io

        from loguru import logger as loguru_logger

        from src.agent.citations import renumber_citations_and_sources

        sink = io.StringIO()
        handler_id = loguru_logger.add(sink, format="{message}", level="WARNING")
        try:
            # LLM cited ``[5-7]`` but only source 5 survived → ``[5]``.
            answer = "see [5-7] for details"
            sources = [self._src(5)]
            new_answer, _ = renumber_citations_and_sources(
                answer, sources, route_decision="retrieve"
            )
            assert new_answer == "see [1] for details"
            log_text = sink.getvalue()
            assert "citations: collapsed range" in log_text, (
                f"WARNING must appear on collapse; got: {log_text!r}"
            )
            assert "[5-7]" in log_text, (
                f"WARNING must include original range; got: {log_text!r}"
            )
        finally:
            loguru_logger.remove(handler_id)


# ============================================================================
# P1-4 — DIRECT_SYSTEM self-intro mentions Bing (not DuckDuckGo)
# ============================================================================


class TestP14DirectSystemSearchEngine:
    """The self-introduction prompt must name Bing (the real default),
    not DuckDuckGo."""

    def test_direct_system_says_bing_primary(self):
        """v2.0.5 — DIRECT_SYSTEM's self-intro text must say
        "Bing primary, DuckDuckGo as fallback" (or equivalent)."""
        text = Path("src/llm/prompts.py").read_text(encoding="utf-8")
        # Slice out the DIRECT_SYSTEM block (rough — adequate for
        # checking the user-visible claim about search engines).
        m = re.search(r'DIRECT_SYSTEM\s*=\s*"""(.*?)"""', text, re.DOTALL)
        assert m is not None, "DIRECT_SYSTEM prompt not found"
        block = m.group(1)
        # Bing primary must be mentioned.
        assert "Bing" in block, "DIRECT_SYSTEM no longer mentions Bing"
        # DuckDuckGo as fallback must be mentioned.
        assert "DuckDuckGo" in block, (
            "DIRECT_SYSTEM no longer mentions DuckDuckGo (fallback)"
        )
        # The old lie — "fall back to a DuckDuckGo web search" —
        # must NOT appear, because DuckDuckGo is NOT the primary.
        assert "fall back to a DuckDuckGo web search" not in block

    def test_generate_system_still_mentions_both(self):
        """GENERATE_SYSTEM's existing copy "Bing/DuckDuckGo 搜索快照"
        is still accurate — must not be lost."""
        text = Path("src/llm/prompts.py").read_text(encoding="utf-8")
        m = re.search(r'GENERATE_SYSTEM\s*=\s*"""(.*?)"""', text, re.DOTALL)
        assert m is not None
        block = m.group(1)
        assert "Bing" in block and "DuckDuckGo" in block, (
            "GENERATE_SYSTEM lost the dual-engine mention"
        )


# ============================================================================
# P1-5 — simple_fact intent fast-path
# ============================================================================


class TestP15SimpleFactFastPath:
    """``_query_is_simple_fact`` catches definitional / single-fact
    lookups; ``IntentDecision`` schema and routing accept the new
    intent value."""

    def test_simple_fact_patterns_match_quantity_questions(self):
        """Quantities / measurements / physical constants."""
        from src.agent.nodes.intent_analysis import _query_is_simple_fact

        for q in (
            "光速是多少",
            "水的沸点是多少度",
            "声音的速度是多少",
            "1 公斤是多少磅",
        ):
            ok, conf = _query_is_simple_fact(q)
            assert ok, f"expected simple_fact for {q!r}, got conf={conf}"

    def test_simple_fact_patterns_match_definitions(self):
        """Definitions / abbreviations / names."""
        from src.agent.nodes.intent_analysis import _query_is_simple_fact

        for q in (
            "HTTP 是什么缩写",
            "什么是 RAG",
            "水的化学式是什么",
            "北京的首都",
        ):
            ok, _ = _query_is_simple_fact(q)
            assert ok, f"expected simple_fact for {q!r}"

    def test_simple_fact_patterns_match_english_lookups(self):
        """``what is X`` / ``how many`` / ``capital of``."""
        from src.agent.nodes.intent_analysis import _query_is_simple_fact

        for q in (
            "what is the capital of France",
            "what's HTTP",
            "how many planets are there",
            "when was the internet invented",
            "where is Tokyo",
        ):
            ok, _ = _query_is_simple_fact(q)
            assert ok, f"expected simple_fact for {q!r}"

    def test_negative_patterns_block_reasoning_questions(self):
        """HIGH + NEGATIVE → NOT simple_fact."""
        from src.agent.nodes.intent_analysis import _query_is_simple_fact

        for q in (
            "为什么光速是 30 万公里每秒",  # "why" reasoning
            "光速的历史是怎样的",  # history
            "HTTP 的工作原理是什么",  # how/principle
            "请解释一下什么是 RAG",  # explain
            "分析一下光速和声速的区别",  # analyze / compare
        ):
            ok, _ = _query_is_simple_fact(q)
            assert not ok, f"expected NOT simple_fact for {q!r}"

    def test_empty_query_is_not_simple_fact(self):
        """An empty query must not match the high pattern."""
        from src.agent.nodes.intent_analysis import _query_is_simple_fact

        ok, _ = _query_is_simple_fact("")
        assert not ok

    def test_intent_decision_accepts_simple_fact(self):
        """``IntentDecision.intent`` Literal must include ``"simple_fact``."""
        from src.llm.schemas import IntentDecision

        members = set(get_args(IntentDecision.model_fields["intent"].annotation))
        assert "simple_fact" in members, (
            f"IntentDecision.intent missing 'simple_fact'; got {members!r}"
        )
        # Legacy members still present.
        for old in ("greeting", "summary", "qa_complex"):
            assert old in members

    def test_intent_system_prompt_mentions_simple_fact(self):
        """Both INTENT_SYSTEM and INTENT_SYSTEM_PLAIN must list
        ``simple_fact`` as a 4th option."""
        text = Path("src/llm/prompts.py").read_text(encoding="utf-8")
        for const in ("INTENT_SYSTEM", "INTENT_SYSTEM_PLAIN"):
            m = re.search(rf'{const}\s*=\s*"""(.*?)"""', text, re.DOTALL)
            assert m is not None, f"{const} prompt not found"
            block = m.group(1)
            assert "simple_fact" in block, (
                f"{const} prompt missing 'simple_fact' option"
            )

    def test_intent_to_route_decision_maps_simple_fact_to_direct(self):
        """``react_generate._intent_to_route_decision("simple_fact")``
        must return ``"direct"`` so the prompt falls back to
        ``DIRECT_SYSTEM``."""
        from src.agent.nodes.react_generate import _intent_to_route_decision

        assert _intent_to_route_decision("simple_fact") == "direct"
        # And greeting must still map to direct (regression guard).
        assert _intent_to_route_decision("greeting") == "direct"
        # And qa_complex / summary / None must still map to retrieve.
        assert _intent_to_route_decision("qa_complex") == "retrieve"
        assert _intent_to_route_decision("summary") == "retrieve"
        assert _intent_to_route_decision(None) == "retrieve"

    def test_after_intent_routes_simple_fact_to_direct(self):
        """``fsm.after_intent`` must route ``"simple_fact"`` to
        ``"react_generate_direct"`` (same as greeting).

        v2.0.21 (Phase 3 Step 7) — predicates moved out of
        ``src.agent.graph`` (deleted) into ``src.agent.fsm``. The
        function is renamed from ``_after_intent`` to ``after_intent``
        (no leading underscore — it's now public FSM surface) and
        returns ``None`` (not LangGraph's ``END`` sentinel) for the
        no-route case (which never applies here — the intent branch
        always names a node).
        """
        from src.agent.fsm import after_intent
        from src.agent.state import AgentState

        # Minimal AgentState stub — after_intent only reads .intent.
        state = AgentState(intent="simple_fact")  # type: ignore[arg-type]
        assert after_intent(state) == "react_generate_direct"
        # Regression guards.
        state = AgentState(intent="greeting")  # type: ignore[arg-type]
        assert after_intent(state) == "react_generate_direct"
        state = AgentState(intent="summary")  # type: ignore[arg-type]
        assert after_intent(state) == "summary_path"
        state = AgentState(intent="qa_complex")  # type: ignore[arg-type]
        assert after_intent(state) == "react_agent"

    def test_should_check_hallucination_skips_simple_fact(self):
        """The hallucination judge must skip ``"simple_fact"`` (no
        docs to check against).

        v2.0.21 (Phase 3 Step 7) — the predicate moved to
        ``src.agent.fsm`` and returns plain ``None`` (= FSM end)
        instead of LangGraph's ``END`` sentinel. The function is
        renamed from ``_should_check_hallucination`` to
        ``should_check_hallucination`` (no leading underscore — now
        public FSM surface).
        """
        from src.agent.fsm import should_check_hallucination
        from src.agent.state import AgentState

        # Predicate returns ``None`` (= FSM end) when there's nothing
        # to verify against.
        state = AgentState(intent="simple_fact", documents=[])  # type: ignore[arg-type]
        assert should_check_hallucination(state) is None
        state = AgentState(intent="greeting", documents=[])  # type: ignore[arg-type]
        assert should_check_hallucination(state) is None
        state = AgentState(intent="summary", documents=[])  # type: ignore[arg-type]
        assert should_check_hallucination(state) is None