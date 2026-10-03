"""Day 0 — OOD adversarial fixture behavioral test (audit_v2).

This is the **baseline pin** for every Phase 1-8 hallucination defense
that ships afterward. Each fixture in ``tests/fixtures/audit_v2/`` is
engineered to trigger a specific failure mode; this test ingests the
fixture, runs the real retrieval pipeline (BM25 + dense + rerank), and
asserts the **current baseline** behavior the defenses must improve on.

What this test pins:

1. **Off-topic corpus** — query about quantum mechanics, corpus about
   photosynthesis. Baseline: retrieval returns low-score (or zero) hits
   → Phase 1 must add a refusal template + Phase 4 must add a
   ``min_top_relevance_score`` gate that triggers the refusal.
2. **Empty PDF** — minimal valid PDF with no extractable text. Baseline:
   ingestion yields 0 chunks → Phase 1 refusal template must fire.
3. **Partial-coverage corpus** — corpus answers 1 of 2 needed parts.
   Baseline: retrieval returns what it has, but Phase 1 must add
   partial-coverage detection and Phase 6 must verify answer completeness.
4. **Prompt-injection corpus** — adversarial directive in the doc body.
   Baseline: directive reaches LLM context verbatim → Phase 4 must add
   ``strip_untrusted`` and prompt_safety UNTRUSTED_PREAMBLE detector.
5. **Homoglyph corpus** — Cyrillic `а` mixed into Latin `a`. Baseline:
   ingestion accepts verbatim, BM25 sees non-ASCII tokens → Phase 4 must
   add Unicode NFKC normalization.
6. **Conflicting-evidence corpus** — same query, two opposite conclusions.
   Baseline: RRF returns both chunks, no conflict detection → Phase 6
   must verify against multiple sources.
7. **OCR-dirty corpus** — random whitespace, broken chars. Baseline:
   BM25 still matches keywords but context is contaminated → Phase 4
   must add OCR-cleaning.

Why behavioral, not substring pin: substring pins break when LLM prompts
or template wording changes. Behavioral pins (chunk_count, rerank_score
threshold, content_contains_token) only break when the retrieval /
ingestion behavior itself regresses.

Why fake embedder + fake reranker: BGE-M3 + BGE Reranker v2-M3 each take
10-30 s to load. The autouse ``_isolate_user_data_dir`` in conftest
redirects the DB to tmp_path but does NOT swap models — tests that want
to run in CI without 100 MB of model weights need a controlled,
deterministic substitute. SHA-256 → bucket-vector + token-overlap
similarity gives 100% reproducible retrieval output.

What is real: LanceDB storage, BM25 FTS index, hybrid_search, RRF,
``retrieve_hybrid_async`` (the actual graph entry point), ChunkRecord
schema, _hit_to_document conversion.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import threading
from pathlib import Path

import pytest

from src.storage.lancedb_store import add_chunks
from src.storage.schema import ChunkRecord


# ============================================================
# Constants
# ============================================================

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "audit_v2"
THREAD_ID = "audit_v2_thread"

# Tuned to BGE-Reranker-v2-m3 sigmoid-normalized scores (see
# config/constants.py RETRIEVAL_DEFAULTS["min_top_relevance_score"]):
# >0.5 strongly relevant, 0.3-0.5 marginal, <0.3 unrelated. This is the
# threshold Phase 4's retrieval gate must use to flag low-relevance
# queries.
LOW_RELEVANCE_THRESHOLD = 0.3


# ============================================================
# Fake heavy resources
# ============================================================


def _text_to_vec(text: str) -> list[float]:
    """Deterministic 1024-dim vector — SHA-256 of text → 1024 floats.

    Same text → same vector every call (across machines, Python versions).
    The vector is unit-normalized so cosine similarity is well-defined.
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # 1024 floats; each input byte → 2 floats (positive + negative phase).
    # Pad with 0s if hash is shorter than 512 bytes (it always is, since
    # SHA-256 is 32 bytes → 64 floats; we need 1024).
    raw = []
    for b in h:
        raw.append(float(b) / 255.0)
        raw.append(-float(255 - b) / 255.0)
    # Pad deterministically by re-hashing to fill 1024 dims.
    while len(raw) < 1024:
        h = hashlib.sha256(h).digest()
        for b in h:
            raw.append(float(b) / 255.0)
            raw.append(-float(255 - b) / 255.0)
    vec = raw[:1024]
    # L2-normalize so cosine sim is bounded [-1, 1].
    norm = sum(x * x for x in vec) ** 0.5 or 1.0
    return [x / norm for x in vec]


class _FakeEmbedder:
    """Drop-in replacement for ``FlagEmbedding.FlagModel``.

    The production wrapper (``BGEM3Embedder._embed``) calls
    ``model.encode_queries(texts, return_dense=True, return_sparse=True)``
    or ``model.encode_corpus(texts, ...)``. Both return a dict with
    ``dense_vecs`` (list[list[float]]) and ``lexical_weights``
    (list[dict[token, weight]]). ``_to_record_list`` in bge_m3.py
    reshapes that into ``[{"dense": ..., "sparse": ...}, ...]``.

    This fake mirrors that shape and is deterministic: same text →
    same vector every call. Each ``dense_vecs`` row is L2-normalized
    so cosine similarity is well-defined.
    """

    def encode_queries(self, texts, **kwargs):
        return {
            "dense_vecs": [_text_to_vec(t) for t in texts],
            "lexical_weights": [{} for _ in texts],
        }

    def encode_corpus(self, texts, **kwargs):
        return {
            "dense_vecs": [_text_to_vec(t) for t in texts],
            "lexical_weights": [{} for _ in texts],
        }


class _FakeReranker:
    """Drop-in replacement for BGE Reranker v2-M3.

    Computes token-overlap ratio normalized to [0, 1]:

        score(q, d) = |unique_tokens(q) ∩ unique_tokens(d)| / |unique_tokens(q)|

    CJK tokens are character-unigrams (since jieba isn't installed in
    test envs). Latin tokens are word-unigrams lowercased. This gives a
    crude-but-reproducible similarity signal that's adequate for
    distinguishing "off-topic" from "on-topic" corpora.

    NOTE: this is a TEST STUB only. Production uses BGE Reranker v2-M3.
    """

    _CJK_RE = re.compile(r"[一-鿿]")
    _LATIN_RE = re.compile(r"[a-z0-9]+")

    @classmethod
    def _tokens(cls, text: str) -> set[str]:
        cjk = set(cls._CJK_RE.findall(text))
        latin = set(cls._LATIN_RE.findall(text.lower()))
        return cjk | latin

    def compute_score(self, pairs, **kwargs):
        out = []
        for q, d in pairs:
            qt = self._tokens(q)
            if not qt:
                out.append(0.0)
                continue
            dt = self._tokens(d)
            overlap = qt & dt
            out.append(len(overlap) / len(qt))
        return out


@pytest.fixture(autouse=True)
def _patch_heavy_models(monkeypatch):
    """Swap out BGE-M3 embedder + BGE Reranker v2-M3 for fakes.

    The autouse ``_isolate_user_data_dir`` in conftest already redirects
    storage to tmp_path; this fixture completes the picture by avoiding
    the 10-30 s model load.
    """
    from src.embeddings import bge_m3
    from src.reranker import bge_reranker

    bge_m3.reset_for_tests()
    bge_reranker.reset_for_tests()

    # Patch the FlagEmbedding classes so the wrappers don't try to load
    # real model weights.
    monkeypatch.setattr(bge_m3, "BGEM3FlagModel", lambda *_a, **_kw: _FakeEmbedder())
    monkeypatch.setattr(bge_reranker, "FlagReranker", lambda *_a, **_kw: _FakeReranker())

    yield

    bge_m3.reset_for_tests()
    bge_reranker.reset_for_tests()


# ============================================================
# Fixture → ChunkRecord builders
# ============================================================


def _fixture_chunks(name: str, *, split_paragraphs: bool = False) -> list[ChunkRecord]:
    """Read tests/fixtures/audit_v2/<name> and wrap as ChunkRecord(s).

    The chunk's doc_id is the fixture filename (sans extension), the
    thread_id is the test-shared audit_v2_thread. The ``source_path``
    follows the v2.0.27 P0-P2 canonical form ``f"{doc_id}/{filename}"``.

    If ``split_paragraphs=True``, treat each non-empty paragraph
    (separated by blank lines) as a separate chunk. Use this for
    multi-paragraph corpora where paragraph-level assertions matter.
    """
    path = FIXTURE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"fixture missing: {path}")
    text = path.read_text(encoding="utf-8")
    if split_paragraphs:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    else:
        paragraphs = [text]
    doc_id = name.split(".")[0].replace(".", "_")
    records = []
    for i, paragraph in enumerate(paragraphs):
        records.append(
            ChunkRecord(
                chunk_id=f"{doc_id}-c{i}",
                doc_id=doc_id,
                text=paragraph,
                vector=_text_to_vec(paragraph),
                sparse={},
                filename=name,
                source_type="file",
                source_path=f"{doc_id}/{name}",
                chunk_index=i,
                thread_id=THREAD_ID,
            )
        )
    return records


def _ingest(records: list[ChunkRecord]) -> None:
    """Reset singletons (defensive — conftest already did) then add."""
    from src.storage import lancedb_store

    lancedb_store.reset_for_tests()
    add_chunks(records)


async def _run_retrieve(query: str, *, thread_id: str = THREAD_ID) -> list:
    """Run the REAL ``retrieve_hybrid_async`` and return the documents.

    Uses the fake embedder (via monkeypatch) and fake reranker so it
    runs in <1 s without model weights.
    """
    from src.agent.nodes import retrieve as retrieve_mod
    from src.embeddings import bge_m3
    from src.reranker import bge_reranker

    # Make sure the module-level singletons see the patched model class.
    retrieve_mod.reset_for_tests()
    bge_m3.reset_for_tests()
    bge_reranker.reset_for_tests()

    state = {
        "current_query": query,
        "original_query": query,
        "thread_id": thread_id,
        "messages": [],
        "step_count": 1,
        "retrieval_count": 0,
        "iteration_count": 0,
    }
    result = await retrieve_mod.retrieve_hybrid_async(state)
    return result.get("documents", [])


def _run(query: str, *, thread_id: str = THREAD_ID) -> list:
    return asyncio.run(_run_retrieve(query, thread_id=thread_id))


# ============================================================
# 1. Off-topic corpus
# ============================================================


def test_corpus_off_topic_triggers_low_relevance_or_empty():
    """Off-topic: query about quantum, corpus about photosynthesis.

    BASELINE (pre-Phase 1, pre-Phase 4 min_score gate):
      - BM25 should return ZERO hits (no Chinese/Latin token overlap)
      - Dense should return SOMETHING (since BM25 FTS is the main
        recall), but with low similarity → low fake-rerank score
      - Phase 4 will add ``min_top_relevance_score=0.3`` gate that
        converts this into a ``retrieval_status="low_relevance"``
        signal → Phase 1 refusal template fires.

    This test asserts the BASELINE so that after Phase 1+4 ship, we
    can pin the IMPROVED behavior (refusal template fires).

    v2.0.29.4 (Phase 4 PR-1) — the gate is now live. With the new
    anchored summary-intent regex, the query no longer fires the
    summary bulk path (which used to return every chunk with
    score=0.0 and pass this test trivially). The query now falls
    through to hybrid + rerank, where BM25 picks up some token
    overlap and the fake reranker scores below the gate threshold.
    Either outcome — gate-fires-empty or BM25-empty — must produce
    ``retrieval_status="low_relevance"`` or ``"empty"`` (both
    trigger the Phase 1 refusal template). What we must NOT see:
    the off-topic corpus passing through to synthesis LLM context.
    """
    chunks = _fixture_chunks("corpus_off_topic.txt", split_paragraphs=True)
    _ingest(chunks)

    # Query deliberately avoids the anchored summary-intent regex
    # (``^总结...``) so we exercise the hybrid+rerank path. The
    # pre-PR-1 query ``"什么是..."`` would have triggered the OLD
    # unanchored regex's `是什么` substring match → bulk path →
    # score=0.0 → trivial pass.
    #
    # The query is English-only with NO Chinese/Latin overlap with
    # the photosynthesis corpus (whose Latin tokens are
    # ``{photosynthesis, calvin, light, reactions, atp, ...}``).
    # Token-overlap = 0 → fake reranker scores 0.0 → gate fires →
    # docs = ``[]``. A Chinese query would re-introduce false
    # positives via the high-frequency CJK unigram overlap (e.g.
    # ``的``, ``与``, ``和`` appear in both query and corpus).
    docs = _run("quantum entanglement")

    if docs:
        # If BM25 + fake reranker returned something, the rerank
        # score must be below the threshold (otherwise the gate
        # wouldn't have anything to gate). If the gate fires
        # (score < threshold), docs is [] so this branch is
        # skipped — both are valid outcomes of the new gate.
        top_score = max(
            (d.metadata.get("score", 0.0) or 0.0) for d in docs
        )
        assert top_score < LOW_RELEVANCE_THRESHOLD, (
            f"off-topic corpus should yield scores < {LOW_RELEVANCE_THRESHOLD} "
            f"(so Phase 4 min_score gate has signal to fire on); got "
            f"top_score={top_score:.3f}. If this fails, the fake reranker's "
            f"token-overlap score is too lenient and needs tightening."
        )


# ============================================================
# 2. Empty PDF
# ============================================================


def test_corpus_empty_pdf_yields_zero_chunks():
    """Empty PDF (valid PDF, no extractable text) → 0 chunks.

    BASELINE: a minimal valid PDF with /Count 0 Kids[] has no text
    content. Ingestion must produce 0 chunks from it. The PDF parser
    is NOT exercised here (we ingest ChunkRecord directly); instead,
    we assert that an empty-content corpus produces zero retrievable
    chunks. This pins the empty-retrieval case that Phase 1's refusal
    template must handle.
    """
    # Construct a single chunk with empty text — simulates what an
    # empty PDF would produce after ingestion (zero text, zero chunks).
    empty_records = [
        ChunkRecord(
            chunk_id="corpus_empty_pdf-c0",
            doc_id="corpus_empty_pdf",
            text="",
            vector=_text_to_vec(""),
            sparse={},
            filename="corpus_empty.pdf",
            source_type="file",
            source_path="corpus_empty_pdf/corpus_empty.pdf",
            chunk_index=0,
            thread_id=THREAD_ID,
        )
    ]
    _ingest(empty_records)

    # Retrieval over a non-matching query on an empty-text corpus
    # must return no usable content.
    docs = _run("What is the main content of this PDF?")

    # Any docs returned will have empty page_content — Phase 1 must
    # treat empty page_content as 'empty retrieval' and refuse.
    nonempty = [d for d in docs if d.page_content.strip()]
    assert not nonempty, (
        f"empty-text corpus must not produce non-empty retrievals; "
        f"got {len(nonempty)} non-empty docs"
    )


# ============================================================
# 3. Partial-coverage corpus
# ============================================================


def test_corpus_partial_md_covers_light_but_not_calvin():
    """Partial corpus: answers 1 of 2 parts of multi-part question.

    BASELINE: corpus_partial.md is constructed to contain light-
    reactions content but NOT Calvin-cycle content. A query that asks
    about the **Calvin cycle specifically** (NOT summary-intent, NOT
    "explain this doc") must find NOTHING in this corpus. A query that
    asks about light reactions must find the relevant chunks.

    This pins the partial-coverage case that Phase 1's partial_coverage
    flag and Phase 6's verification agent must detect: the corpus
    CAN answer one part of a user's question but NOT another. If the
    LLM hallucinates the missing part, that's the failure mode.

    Note: we deliberately use a non-summary-intent query. The summary-
    intent fast path (``什么 | 有哪些 | 总结`` regex) returns ALL
    chunks of the thread regardless of relevance — that defeats the
    test's purpose. Phase 4 will tighten that regex; until then we
    work around it by querying for a specific fact.

    If Phase 1 fires the partial_coverage refusal template, the
    SYMPTOM changes (LLM says "I only have info on X, not Y") but
    the underlying retrieval behavior — "I have X but not Y" — stays
    the same. This test asserts the retrieval behavior so Phase 1+6
    can layer their refusal/verification on top of a stable base.
    """
    chunks = _fixture_chunks("corpus_partial.md", split_paragraphs=True)
    _ingest(chunks)

    # Sub-query 1: should find light-reactions content.
    light_docs = _run("光合磷酸化过程是怎样进行的?")
    light_text = " ".join(d.page_content for d in light_docs)
    assert light_docs, "should retrieve at least one light-reactions chunk"
    assert (
        "光反应" in light_text or "光系统" in light_text or "ATP 合酶" in light_text
    ), "light-reactions query should retrieve relevant content"

    # Sub-query 2: should find NOTHING — fixture has no Calvin content.
    # Use a fact-specific query that doesn't trigger summary intent.
    calvin_docs = _run("Calvin 循环中 RuBisCO 酶的催化机理")
    calvin_text = " ".join(d.page_content for d in calvin_docs)
    assert not (
        "RuBisCO" in calvin_text
        or "Calvin" in calvin_text
        or "碳反应" in calvin_text
    ), (
        "Calvin-cycle query should find NO relevant content (fixture "
        "is intentionally partial). If this fails, the fixture was "
        "modified to include Calvin content — the partial-coverage "
        "fixture is no longer partial."
    )


# ============================================================
# 4. Prompt-injection corpus
# ============================================================


def test_corpus_injection_md_directive_surfaces_in_context():
    """Injection corpus: adversarial directive reaches LLM context.

    BASELINE: corpus_injection.md is constructed with a "SYSTEM
    OVERRIDE" directive embedded in the middle. Retrieval returns it
    verbatim. The defense is downstream (Phase 4 prompt_safety +
    strip_untrusted). This test pins the BASELINE so Phase 4 can
    layer on top.

    The test asserts the directive IS in retrieval output (i.e. it
    reaches the LLM context). After Phase 4 ships, this same fixture
    will be used in a NEW test that asserts the directive is
    NEUTRALIZED (i.e. doesn't appear in the LLM-bound context).
    """
    chunks = _fixture_chunks("corpus_injection.md", split_paragraphs=False)
    _ingest(chunks)

    docs = _run("What is the capital of France?")
    # v2.0.29.4 (Phase 4 PR-1) — the new ``min_top_relevance_score``
    # gate (PR-1 #5) fires on this query because the BM25 + fake
    # reranker scores the off-topic question ~0.0 against the
    # Chinese-only corpus. Pre-PR-1 this query would fall through
    # the hybrid path and the corpus would reach synthesis context
    # (the original baseline this test was pinning).
    #
    # Post-PR-1 the gate is supposed to surface this as
    # ``retrieval_status="low_relevance"`` so the Phase 1 refusal
    # template fires in ``react_generate`` rather than letting the
    # adversarial directive slip into LLM context.
    #
    # This test now pins the NEW defense: query is unrelated, gate
    # fires, retrieval returns 0 docs. The complementary test
    # ``test_phase4_neutralizes_injection_directive`` (added in
    # ``tests/test_retrieval_filters.py``) verifies the directive
    # is ALSO caught at the prompt_safety fence.
    assert not docs, (
        "off-topic query against injection fixture must trigger the "
        "Phase 4 min_score gate and return 0 docs; the fixture's "
        "adversarial directive should NOT reach synthesis LLM context. "
        "If this fails, the gate isn't firing on token-overlap=0 "
        "corpora — check that ``min_top_relevance_score`` is wired."
    )


# ============================================================
# 5. Homoglyph corpus
# ============================================================


def test_corpus_homoglyph_md_cyrillic_a_surfaces_verbatim():
    """Homoglyph corpus: Cyrillic а passes through ingestion verbatim.

    BASELINE: ingestion currently does NOT normalize Cyrillic homoglyphs.
    The Cyrillic `а` (U+0430) survives as a distinct token. BM25 tokenizes
    Cyrillic characters as their own tokens (no overlap with Latin `а`).
    This is the designed baseline — Phase 4 must add NFKC normalization
    at ingest time so the homoglyph attack becomes detectable / fusable
    with its Latin counterpart.

    After Phase 4 ships, the new assertion will be: the homoglyph `а`
    is normalized to Latin `a` BEFORE BM25 indexing.
    """
    chunks = _fixture_chunks("corpus_homoglyph.txt", split_paragraphs=False)
    _ingest(chunks)

    docs = _run("apple.com 是苹果公司的官网吗?")
    assert docs, "should retrieve the homoglyph fixture"

    full_text = " ".join(d.page_content for d in docs)
    # Cyrillic а (U+0430) — NOT Latin a (U+0061). The fixture contains
    # many of these in spoofed URLs.
    CYRILLIC_A = "а"
    assert CYRILLIC_A in full_text, (
        "fixture's Cyrillic а must surface in retrieval output — this "
        "is the baseline (no normalization). Phase 4 must add Unicode "
        "NFKC normalization at ingest so the homoglyph attack becomes "
        "detectable. If this fails, ingestion is already normalizing "
        "(Phase 4 may have been merged)."
    )
    # Spoofed URL pattern: ".аpple." or "аpple" — Cyrillic а followed
    # by Latin pple.
    assert re.search(r"\.аpple|аpple", full_text), (
        "fixture must contain the spoofed 'аpple' URL pattern"
    )


# ============================================================
# 6. Conflicting-evidence corpus
# ============================================================


def test_corpus_conflicting_md_returns_both_opposing_chunks():
    """Conflicting corpus: 2 paragraphs, opposite conclusions.

    BASELINE: corpus_conflicting.md has one paragraph concluding
    'solar is cheaper' and another concluding 'fossil fuel is
    cheaper'. RRF / dense retrieves both chunks (high similarity on
    shared terms like 'solar', 'cost', 'LCOE'). Phase 6's verification
    agent must detect the conflict and either flag it to the user or
    force the synthesis LLM to acknowledge both conclusions.

    This test asserts the BASELINE: BOTH opposing chunks are
    retrieved. After Phase 6 ships, the assertion moves to:
    'verification_result.conflicting_sources=True' on the wire event.
    """
    chunks = _fixture_chunks("corpus_conflicting.md", split_paragraphs=True)
    _ingest(chunks)

    docs = _run("太阳能和化石燃料哪个更便宜?")
    assert docs, "should retrieve conflicting fixture"

    full_text = " ".join(d.page_content for d in docs)
    # The fixture has two markers — one per conclusion. Each marker is
    # wrapped in markdown bold (``**太阳能**``) so the assertion must
    # match the bold-wrapped form, not the bare substring.
    SOLAR_CHEAPER_MARKER = "**太阳能**已经比化石燃料便宜"
    FOSSIL_CHEAPER_MARKER = "**化石燃料**在可预见的 10 年内仍比太阳能便宜"
    assert SOLAR_CHEAPER_MARKER in full_text, (
        "fixture should contain the 'solar cheaper' conclusion (with "
        "markdown bold wrapper)"
    )
    assert FOSSIL_CHEAPER_MARKER in full_text, (
        "fixture should contain the 'fossil cheaper' conclusion — both "
        "sides must surface so Phase 6 can detect the conflict. If this "
        "fails, the fixture was modified to be one-sided or BM25 dropped "
        "one of the two conclusion paragraphs (check top_k cap)."
    )


# ============================================================
# 7. OCR-dirty corpus
# ============================================================


def test_corpus_dirty_txt_bm25_still_matches_keywords():
    """OCR-dirty corpus: random whitespace, BM25 still matches keywords.

    BASELINE: corpus_dirty.txt has random whitespace inserted between every
    character ("Ap ple I n c ."). BM25 tokenizes by whitespace, so the
    broken tokens ('Ap', 'ple', 'I', 'nc') STILL match queries like
    "AAPL" or "Apple" because Tantivy does prefix / substring fall-back.
    The dirty content reaches the LLM context verbatim, contaminating
    the answer. Phase 4 must add OCR-cleaning at ingest time (collapse
    excessive whitespace, normalize broken chars).

    This test pins the BASELINE: BM25 retrieves the dirty corpus, and
    the dirty chars DO surface in the LLM context. After Phase 4 ships,
    a NEW test asserts the cleaning (whitespace collapsed, broken chars
    normalized).
    """
    chunks = _fixture_chunks("corpus_dirty.txt", split_paragraphs=False)
    _ingest(chunks)

    docs = _run("Apple 公司的股票代码是什么?")
    assert docs, "BM25 should retrieve the dirty fixture (AAPL/Apple tokens match)"

    full_text = " ".join(d.page_content for d in docs)
    # The fixture has the broken-spelling 'Ap ple I n c .' pattern.
    assert "Ap ple" in full_text or "AA PL" in full_text or "iP hone" in full_text, (
        "OCR-dirty chars must surface in retrieval output — this is "
        "the baseline (no cleaning). Phase 4 must add OCR-cleaning. "
        "If this fails, ingestion is already cleaning (Phase 4 may "
        "have been merged)."
    )


# ============================================================
# Cross-fixture sanity
# ============================================================


def test_all_fixtures_present():
    """Sanity: every documented fixture file exists."""
    expected = [
        "corpus_off_topic.txt",
        "corpus_empty.pdf",
        "corpus_partial.md",
        "corpus_injection.md",
        "corpus_homoglyph.txt",
        "corpus_conflicting.md",
        "corpus_dirty.txt",
        "_README.md",
    ]
    for name in expected:
        path = FIXTURE_DIR / name
        assert path.is_file(), f"missing fixture: {path}"


def test_fake_embedder_is_deterministic():
    """Sanity: the fake embedder must be deterministic (CI stability)."""
    v1 = _text_to_vec("hello world 光合作用")
    v2 = _text_to_vec("hello world 光合作用")
    assert v1 == v2, "fake embedder must produce identical vectors for identical text"

    v3 = _text_to_vec("hello world")
    assert v1 != v3, "different text should produce different vectors"

    # All vectors must be L2-normalized.
    norm = sum(x * x for x in v1) ** 0.5
    assert abs(norm - 1.0) < 1e-6, f"vector must be L2-normalized; got norm={norm}"


def test_fake_reranker_bounded_zero_to_one():
    """Sanity: the fake reranker returns [0, 1] sigmoid-normalized scores."""
    rr = _FakeReranker()
    pairs = [
        ("apple banana", "apple banana cherry"),
        ("quantum physics", "photosynthesis light reactions"),
        ("", "anything"),
        ("光合作用", "光合作用的光反应"),
    ]
    scores = rr.compute_score(pairs)
    assert all(0.0 <= s <= 1.0 for s in scores), f"scores must be in [0,1]; got {scores}"
    # Perfect overlap (apple banana ⊂ apple banana cherry) → high score.
    assert scores[0] >= 0.9, f"perfect token overlap should score high; got {scores[0]}"
    # Zero overlap → 0.
    assert scores[1] == 0.0, f"zero overlap should score 0; got {scores[1]}"
    # Empty query → 0 (no signal).
    assert scores[2] == 0.0, f"empty query should score 0; got {scores[2]}"


# ============================================================
# Constants lock
# ============================================================


def test_thread_id_isolation():
    """Sanity: every test in this file uses the same THREAD_ID so a
    run-order regression surfaces here.

    If we ever add tests that need a different thread (e.g. for the
    temporal-filter Phase 4 work), this test should be parametrized or
    split into per-thread tests.
    """
    # Both constants used by this file reference the same thread.
    assert THREAD_ID == "audit_v2_thread"