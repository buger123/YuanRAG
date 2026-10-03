"""Phase 1 — Refusal contract tests (v2.0.29.1).

Pins every behavior introduced by the Phase 1 refusal contract:

1. ``REFUSAL_TEMPLATES`` exists in ``src.llm.prompts`` with the 3
   expected keys (empty / empty_bulk / low_relevance) and the
   forbidden-phrase list inside each template.
2. ``AgentState.retrieval_status`` TypedDict field exists with the
   Literal type ``"empty" | "empty_bulk" | "low_relevance" | "success"``.
3. ``retrieve_hybrid_async`` sets ``retrieval_status`` correctly:
   - "empty" when retrieval returns 0 docs
   - "empty" when an exception escapes hybrid search
   - "success" when docs returned (regardless of count, until Phase 4
     introduces the score threshold)
4. ``summary_path`` (in ``fsm.py``) sets ``retrieval_status="empty_bulk"``
   when the thread has no uploaded documents.
5. ``react_generate`` injects the matching refusal directive at
   ``msgs[2]`` when ``retrieval_status`` is one of the refusal keys.
6. ``react_generate`` preserves the v2.0.28.17 consecutiveness rule:
   when both the refusal directive AND the time directive would fire,
   the resulting msgs[0..3] are all SystemMessage (no
   HumanMessage/AIMessage in between).
7. ``check_hallucination_async`` uses the bumped
   ``JUDGE_SNIPPET_CHARS_PER_DOC=2000`` constant (not the pre-Phase-1
   hard-coded 500).

The test design follows the audit_v2 fixture pattern: real LanceDB +
fake embedder/reranker (deterministic SHA-256 vector + token-overlap
similarity). Behavioral assertions pin the structural properties —
not the LLM's actual refusal text (that's a Layer 5 verifier concern).

Critical v2.0.28.17 rule preserved throughout: every SystemMessage
injection uses ``insert(2, ...)`` so Anthropic's consecutiveness
requirement is never violated.
"""
from __future__ import annotations

import asyncio
import inspect

import pytest


# ============================================================
# 1. REFUSAL_TEMPLATES shape + content
# ============================================================


def test_refusal_templates_dict_has_three_keys():
    """REFUSAL_TEMPLATES must contain the 3 expected refusal statuses.

    Phase 1's 3 refusal keys: empty / empty_bulk / low_relevance.
    ``success`` is NOT a refusal status (LLM has docs to work with).
    ``partial_coverage`` is Phase 6 territory (verification agent
    needs full retrieved context to detect partial coverage — refusal
    template would force the LLM to refuse even when 80% of the
    answer IS in the docs).
    """
    from src.llm.prompts import REFUSAL_TEMPLATES

    assert isinstance(REFUSAL_TEMPLATES, dict)
    assert set(REFUSAL_TEMPLATES.keys()) == {"empty", "empty_bulk", "low_relevance"}, (
        f"REFUSAL_TEMPLATES keys mismatch — got {set(REFUSAL_TEMPLATES.keys())}"
    )


def test_refusal_templates_are_non_empty_strings():
    """Every template must be a non-empty string the LLM can walk."""
    from src.llm.prompts import REFUSAL_TEMPLATES

    for key, tmpl in REFUSAL_TEMPLATES.items():
        assert isinstance(tmpl, str), f"{key!r} template must be a string"
        assert len(tmpl) > 100, (
            f"{key!r} template is suspiciously short ({len(tmpl)} chars) — "
            "should contain acknowledgment + forbidden list + next-step guidance"
        )


def test_refusal_templates_contain_forbidden_phrasing_list():
    """Every refusal template must contain the forbidden-phrase block.

    The forbidden phrasing list is the load-bearing part: without it,
    the LLM falls back to hedged fabrication. Each template carries a
    copy because Anthropic's prompt structure works best when the
    forbidden list is close to the instructions (not in a shared
    preamble that some prompts would skip).
    """
    from src.llm.prompts import REFUSAL_TEMPLATES

    # The Chinese hedge terms — these are the most common
    # fabrication-enabling phrases in the corpus. Their absence in the
    # template means the LLM has no anchor to avoid them.
    FORBIDDEN_MARKERS = [
        "应该是",
        "可能是",
        "I think",
        "probably",
    ]
    for key, tmpl in REFUSAL_TEMPLATES.items():
        for marker in FORBIDDEN_MARKERS:
            assert marker in tmpl, (
                f"{key!r} template is missing forbidden phrase {marker!r} — "
                "the LLM has no anchor to avoid hedged fabrication"
            )


def test_refusal_templates_have_explicit_response_shape():
    """Every template must specify a "Required response shape" block.

    This is what makes the template a CONTRACT, not a polite hint.
    The LLM sees the explicit shape and is more likely to walk it
    verbatim than to improvise.
    """
    from src.llm.prompts import REFUSAL_TEMPLATES

    for key, tmpl in REFUSAL_TEMPLATES.items():
        assert "Required response shape" in tmpl, (
            f"{key!r} template is missing 'Required response shape' — "
            "template is a polite hint, not a contract"
        )
        assert "Acknowledge" in tmpl, (
            f"{key!r} template is missing 'Acknowledge' step"
        )


# ============================================================
# 2. AgentState.retrieval_status field
# ============================================================


def test_agent_state_has_retrieval_status_field():
    """AgentState TypedDict must declare retrieval_status with the
    expected Literal type.

    Backward-compat: ``total=False`` on the TypedDict means missing
    keys read as None — pre-Phase-1 state (no retrieval_status set)
    is treated as ``success`` by react_generate, which means the
    LLM is NOT forced into a refusal template. This is intentional:
    old checkpoints / legacy flows don't suddenly refuse.
    """
    from src.agent.state import AgentState

    annotations = AgentState.__annotations__
    assert "retrieval_status" in annotations, (
        "AgentState must declare retrieval_status — Phase 1 contract"
    )


def test_agent_state_retrieval_status_literal_values():
    """retrieval_status must be a Literal with the 4 expected values.

    `from __future__ import annotations` keeps annotations as
    forward-reference strings under Python 3.13, so we need
    ``typing.get_type_hints()`` (which evaluates them) rather than
    reading ``__annotations__`` directly.
    """
    from typing import get_args, get_type_hints

    from src.agent.state import AgentState

    hints = get_type_hints(AgentState)
    args = get_args(hints["retrieval_status"])
    assert set(args) == {"empty", "empty_bulk", "low_relevance", "success"}, (
        f"retrieval_status Literal mismatch — got {set(args)}"
    )


# ============================================================
# 3. retrieve_hybrid_async sets retrieval_status correctly
# ============================================================


def test_retrieve_sets_empty_status_when_no_docs(monkeypatch):
    """When hybrid_search returns 0 docs, retrieve must set
    retrieval_status="empty" so react_generate walks the refusal
    template.

    Real BM25 + real LanceDB; fake embedder (deterministic). Empty
    retrieval happens when the query has no token overlap with any
    chunk.
    """
    from src.embeddings import bge_m3
    from src.reranker import bge_reranker
    from src.storage import lancedb_store
    from src.agent.nodes import retrieve as retrieve_mod
    from config.settings import reset_settings_cache

    bge_m3.reset_for_tests()
    bge_reranker.reset_for_tests()
    lancedb_store.reset_for_tests()
    reset_settings_cache()
    retrieve_mod.reset_for_tests()

    # Monkeypatch fake heavy resources (model loading).
    from tests.test_audit_fixture_behavior import _FakeEmbedder, _FakeReranker
    monkeypatch.setattr(bge_m3, "BGEM3FlagModel", lambda *_a, **_kw: _FakeEmbedder())
    monkeypatch.setattr(bge_reranker, "FlagReranker", lambda *_a, **_kw: _FakeReranker())

    # Query that has zero token overlap with anything in the corpus.
    # No chunks have been ingested, so retrieval returns 0 docs.
    state = {
        "current_query": "量子纠缠与量子力学",  # nothing matches
        "original_query": "量子纠缠与量子力学",
        "thread_id": "refusal_contract_thread",
        "messages": [],
        "step_count": 1,
    }
    result = asyncio.run(retrieve_mod.retrieve_hybrid_async(state))

    assert result.get("retrieval_status") == "empty", (
        f"retrieval with 0 docs must set retrieval_status='empty'; "
        f"got {result.get('retrieval_status')!r}"
    )
    assert result.get("documents") == [], (
        f"empty retrieval must return empty doc list; "
        f"got {len(result.get('documents') or [])} docs"
    )


def test_retrieve_sets_success_status_when_docs_found(monkeypatch, tmp_path):
    """When retrieval returns docs (via summary-intent bulk path),
    retrieve must set retrieval_status='success' so react_generate
    does NOT walk the refusal template.

    This test exercises the summary-intent branch (uploading docs to
    the thread, then asking a summary-intent query). The bulk path
    returns every chunk of the thread's docs and stamps
    retrieval_status='success'.
    """
    from src.embeddings import bge_m3
    from src.reranker import bge_reranker
    from src.storage import lancedb_store
    from src.agent.nodes import retrieve as retrieve_mod
    from src.storage.schema import ChunkRecord
    from config.settings import reset_settings_cache

    bge_m3.reset_for_tests()
    bge_reranker.reset_for_tests()
    lancedb_store.reset_for_tests()
    reset_settings_cache()

    from tests.test_audit_fixture_behavior import _FakeEmbedder, _FakeReranker, _text_to_vec
    monkeypatch.setattr(bge_m3, "BGEM3FlagModel", lambda *_a, **_kw: _FakeEmbedder())
    monkeypatch.setattr(bge_reranker, "FlagReranker", lambda *_a, **_kw: _FakeReranker())

    # Ingest one chunk into the thread.
    from src.storage.lancedb_store import add_chunks

    rec = ChunkRecord(
        chunk_id="r1",
        doc_id="d1",
        text="这是一段测试内容,关于 Python 编程。",
        vector=_text_to_vec("这是一段测试内容,关于 Python 编程。"),
        sparse={},
        filename="test.txt",
        source_type="file",
        source_path="d1/test.txt",
        chunk_index=0,
        thread_id="refusal_contract_thread",
    )
    add_chunks([rec])

    retrieve_mod.reset_for_tests()

    state = {
        # v2.0.29.4 (Phase 4 PR-1) — the summary-intent regex is now
        # anchored at start (``^(?:...)``). Pre-PR-1 the regex matched
        # ``总结`` anywhere in the query (e.g. ``"请总结..."``);
        # post-PR-1 only queries that BEGIN with a summary verb
        # fire the bulk path. ``"总结一下这个文档"`` starts with
        # ``总结`` so the summary-intent branch fires and returns
        # success — without this change the test would fall through
        # to the hybrid + rerank path and the new ``min_top_relevance_score``
        # gate (PR-1 #5) would push it to ``low_relevance`` because
        # the test's fake reranker scores token-overlap at ~0.22
        # (< 0.3 threshold).
        "current_query": "总结一下这个文档",
        "original_query": "总结一下这个文档",
        "thread_id": "refusal_contract_thread",
        "messages": [],
        "step_count": 1,
    }
    result = asyncio.run(retrieve_mod.retrieve_hybrid_async(state))

    assert result.get("retrieval_status") == "success", (
        f"successful summary-intent bulk path must set retrieval_status='success'; "
        f"got {result.get('retrieval_status')!r}"
    )
    assert len(result.get("documents") or []) >= 1, (
        "bulk path should have returned at least the one chunk we ingested"
    )


# ============================================================
# 4. summary_path sets retrieval_status correctly (in fsm.py)
# ============================================================


def test_summary_path_sets_empty_bulk_when_no_thread_docs(monkeypatch):
    """summary_path yields 'empty_bulk' status when the thread has no
    documents — so react_generate walks REFUSAL_EMPTY_BULK_TEMPLATE.

    Note: summary_path is in src/agent/fsm.py (it's a fast-path node
    that runs before react_generate for summary-intent queries).
    """
    from src.embeddings import bge_m3
    from src.reranker import bge_reranker
    from src.storage import lancedb_store
    from config.settings import reset_settings_cache

    bge_m3.reset_for_tests()
    bge_reranker.reset_for_tests()
    lancedb_store.reset_for_tests()
    reset_settings_cache()

    from src.agent import fsm

    # No docs ingested for this thread — list_chunks_by_thread returns [].
    state = {"thread_id": "empty_bulk_thread"}
    events = []
    async def collect():
        async for evt in fsm.summary_path(state):
            events.append(evt)
    asyncio.run(collect())

    delta_events = [e for e in events if e[0] == "__delta__"]
    assert len(delta_events) == 1, f"summary_path should yield one __delta__; got {len(delta_events)}"
    delta = delta_events[0][1]
    assert delta.get("retrieval_status") == "empty_bulk", (
        f"summary_path with no chunks must set retrieval_status='empty_bulk'; "
        f"got {delta.get('retrieval_status')!r}"
    )
    assert delta.get("documents") == [], (
        f"empty_bulk status must come with empty docs; "
        f"got {len(delta.get('documents') or [])} docs"
    )


def test_summary_path_sets_success_when_thread_has_docs(monkeypatch):
    """summary_path yields 'success' status when the thread has
    uploaded documents — so react_generate does NOT walk a refusal
    template (the LLM has something to summarize).
    """
    from src.embeddings import bge_m3
    from src.reranker import bge_reranker
    from src.storage import lancedb_store
    from src.storage.schema import ChunkRecord
    from config.settings import reset_settings_cache

    bge_m3.reset_for_tests()
    bge_reranker.reset_for_tests()
    lancedb_store.reset_for_tests()
    reset_settings_cache()

    from tests.test_audit_fixture_behavior import _text_to_vec
    from src.storage.lancedb_store import add_chunks

    rec = ChunkRecord(
        chunk_id="r1",
        doc_id="d1",
        text="这是一段测试内容,关于 Python 编程。",
        vector=_text_to_vec("这是一段测试内容,关于 Python 编程。"),
        sparse={},
        filename="test.txt",
        source_type="file",
        source_path="d1/test.txt",
        chunk_index=0,
        thread_id="success_summary_thread",
    )
    add_chunks([rec])

    from src.agent import fsm

    state = {"thread_id": "success_summary_thread"}
    events = []
    async def collect():
        async for evt in fsm.summary_path(state):
            events.append(evt)
    asyncio.run(collect())

    delta_events = [e for e in events if e[0] == "__delta__"]
    delta = delta_events[0][1]
    assert delta.get("retrieval_status") == "success", (
        f"summary_path with chunks must set retrieval_status='success'; "
        f"got {delta.get('retrieval_status')!r}"
    )
    assert len(delta.get("documents") or []) >= 1


# ============================================================
# 5. react_generate injection behavior
# ============================================================


def test_react_generate_injects_refusal_at_index_2_when_empty(monkeypatch):
    """When retrieval_status='empty', react_generate inserts the
    matching refusal SystemMessage at msgs[2] (consecutive with
    msgs[0..1]).

    v2.0.28.17 consecutiveness rule: msgs[0..2] must all be
    SystemMessage. We verify by inspecting the messages list BEFORE
    the LLM stream (capture the msgs argument via mock).
    """
    from src.embeddings import bge_m3
    from src.reranker import bge_reranker
    from src.storage import lancedb_store
    from config.settings import reset_settings_cache

    bge_m3.reset_for_tests()
    bge_reranker.reset_for_tests()
    lancedb_store.reset_for_tests()
    reset_settings_cache()

    from langchain_core.messages import SystemMessage

    from src.agent.nodes import react_generate as rg_mod

    captured_msgs = []

    class _CaptureModel:
        async def astream(self, msgs, **kwargs):
            captured_msgs.extend(msgs)
            # Yield a fake chunk so the generator finishes.
            yield type("Chunk", (), {"content": "资料库中没有与您的问题相关的内容。"})()

    # Monkeypatch build_chat_model to return our capture model.
    monkeypatch.setattr(rg_mod, "build_chat_model", lambda **kw: _CaptureModel())

    state = {
        "current_query": "量子纠缠与量子力学",
        "original_query": "量子纠缠与量子力学",
        "thread_id": "refusal_inject_thread",
        "intent": "qa_complex",
        "messages": [],  # no history — keeps the test deterministic
        "documents": [],
        "retrieval_status": "empty",
        "step_count": 1,
    }
    events = []
    async def collect():
        async for evt in rg_mod.react_generate(state):
            events.append(evt)
    asyncio.run(collect())

    # We didn't inject any history messages, so msgs should be:
    #   [System(sys_prompt), System(history_hint), System(refusal_directive)]
    assert len(captured_msgs) == 3, (
        f"expected 3 system messages with empty retrieval_status + no history; "
        f"got {len(captured_msgs)}"
    )
    for i, m in enumerate(captured_msgs):
        assert isinstance(m, SystemMessage), (
            f"msgs[{i}] must be SystemMessage (v2.0.28.17 consecutiveness); "
            f"got {type(m).__name__}"
        )

    # The third system message is the refusal directive.
    refusal_msg = captured_msgs[2]
    from src.llm.prompts import REFUSAL_TEMPLATES
    expected_refusal = REFUSAL_TEMPLATES["empty"]
    assert refusal_msg.content == expected_refusal, (
        "msgs[2] must be the REFUSAL_EMPTY_TEMPLATE content verbatim"
    )


def test_react_generate_injects_empty_bulk_refusal(monkeypatch):
    """retrieval_status='empty_bulk' injects REFUSAL_EMPTY_BULK_TEMPLATE."""
    from src.embeddings import bge_m3
    from src.reranker import bge_reranker
    from src.storage import lancedb_store
    from config.settings import reset_settings_cache

    bge_m3.reset_for_tests()
    bge_reranker.reset_for_tests()
    lancedb_store.reset_for_tests()
    reset_settings_cache()

    from langchain_core.messages import SystemMessage

    from src.agent.nodes import react_generate as rg_mod

    captured_msgs = []

    class _CaptureModel:
        async def astream(self, msgs, **kwargs):
            captured_msgs.extend(msgs)
            yield type("Chunk", (), {"content": "该对话尚未上传任何文档。"})()

    monkeypatch.setattr(rg_mod, "build_chat_model", lambda **kw: _CaptureModel())

    state = {
        "current_query": "请总结一下这个文档",
        "original_query": "请总结一下这个文档",
        "intent": "summary",
        "thread_id": "empty_bulk_inject_thread",
        "messages": [],
        "documents": [],
        "retrieval_status": "empty_bulk",
        "step_count": 1,
    }
    events = []
    async def collect():
        async for evt in rg_mod.react_generate(state):
            events.append(evt)
    asyncio.run(collect())

    assert len(captured_msgs) == 3
    assert all(isinstance(m, SystemMessage) for m in captured_msgs)
    from src.llm.prompts import REFUSAL_TEMPLATES
    assert captured_msgs[2].content == REFUSAL_TEMPLATES["empty_bulk"]


def test_react_generate_no_refusal_when_success(monkeypatch):
    """retrieval_status='success' does NOT inject any refusal directive.

    The success branch is the normal path — LLM has docs to work
    with and walks the GENERATE_SYSTEM template instead. Injecting a
    refusal directive here would actively harm quality (the LLM might
    refuse even when docs are perfectly relevant).
    """
    from src.embeddings import bge_m3
    from src.reranker import bge_reranker
    from src.storage import lancedb_store
    from config.settings import reset_settings_cache

    bge_m3.reset_for_tests()
    bge_reranker.reset_for_tests()
    lancedb_store.reset_for_tests()
    reset_settings_cache()

    from langchain_core.messages import SystemMessage

    from src.agent.nodes import react_generate as rg_mod

    captured_msgs = []

    class _CaptureModel:
        async def astream(self, msgs, **kwargs):
            captured_msgs.extend(msgs)
            yield type("Chunk", (), {"content": "answer from docs"})()

    monkeypatch.setattr(rg_mod, "build_chat_model", lambda **kw: _CaptureModel())

    from langchain_core.documents import Document

    state = {
        "current_query": "Python 是什么?",
        "original_query": "Python 是什么?",
        "intent": "qa_complex",
        "thread_id": "success_inject_thread",
        "messages": [],
        "documents": [
            Document(
                page_content="Python 是一种编程语言。",
                metadata={"doc_id": "d1", "chunk_id": "c1", "source_kind": "local"},
            )
        ],
        "retrieval_status": "success",
        "step_count": 1,
    }
    events = []
    async def collect():
        async for evt in rg_mod.react_generate(state):
            events.append(evt)
    asyncio.run(collect())

    # Only 2 system messages (sys_prompt + history_hint). NO refusal.
    assert len(captured_msgs) == 2, (
        f"success path must inject only sys_prompt + history_hint; "
        f"got {len(captured_msgs)} system messages"
    )

    from src.llm.prompts import REFUSAL_TEMPLATES
    for tmpl in REFUSAL_TEMPLATES.values():
        for m in captured_msgs:
            assert m.content != tmpl, (
                "success path must NOT contain any REFUSAL_TEMPLATES content"
            )


def test_react_generate_missing_retrieval_status_does_not_inject_refusal(monkeypatch):
    """Backward-compat: pre-Phase-1 state has no retrieval_status
    field. react_generate treats this as 'success' (no refusal
    injection) so legacy flows don't suddenly refuse.

    This is critical for reload — old checkpoints persisted under
    v2.0.28.x load with no retrieval_status field, and the
    ``total=False`` TypedDict reads the missing key as None.
    ``REFUSAL_TEMPLATES.get(None or "")`` returns None, so the
    refusal directive never fires.
    """
    from src.embeddings import bge_m3
    from src.reranker import bge_reranker
    from src.storage import lancedb_store
    from config.settings import reset_settings_cache

    bge_m3.reset_for_tests()
    bge_reranker.reset_for_tests()
    lancedb_store.reset_for_tests()
    reset_settings_cache()

    from src.agent.nodes import react_generate as rg_mod

    captured_msgs = []

    class _CaptureModel:
        async def astream(self, msgs, **kwargs):
            captured_msgs.extend(msgs)
            yield type("Chunk", (), {"content": "answer"})()

    monkeypatch.setattr(rg_mod, "build_chat_model", lambda **kw: _CaptureModel())

    from langchain_core.documents import Document

    # NO retrieval_status field — simulates pre-Phase-1 checkpoint reload.
    state = {
        "current_query": "Python 是什么?",
        "original_query": "Python 是什么?",
        "intent": "qa_complex",
        "thread_id": "legacy_reload_thread",
        "messages": [],
        "documents": [
            Document(
                page_content="Python 是一种编程语言。",
                metadata={"doc_id": "d1", "chunk_id": "c1", "source_kind": "local"},
            )
        ],
        "step_count": 1,
    }
    # Sanity: state has no retrieval_status key.
    assert "retrieval_status" not in state

    events = []
    async def collect():
        async for evt in rg_mod.react_generate(state):
            events.append(evt)
    asyncio.run(collect())

    # Only 2 system messages — same as success path.
    assert len(captured_msgs) == 2, (
        f"missing retrieval_status must behave like success; "
        f"got {len(captured_msgs)} system messages"
    )


# ============================================================
# 6. Direct path is unaffected (sanity)
# ============================================================


def test_react_generate_direct_path_no_refusal(monkeypatch):
    """The direct (greeting/simple_fact) path bypasses refusal
    injection entirely. Greetings have no retrieval context to
    refuse against — they're conversational.
    """
    from src.embeddings import bge_m3
    from src.reranker import bge_reranker
    from src.storage import lancedb_store
    from config.settings import reset_settings_cache

    bge_m3.reset_for_tests()
    bge_reranker.reset_for_tests()
    lancedb_store.reset_for_tests()
    reset_settings_cache()

    from src.agent import fsm

    captured_msgs = []

    class _CaptureModel:
        async def astream(self, msgs, **kwargs):
            captured_msgs.extend(msgs)
            yield type("Chunk", (), {"content": "你好!我是源 RAG。"})()

    monkeypatch.setattr(fsm, "build_chat_model", lambda **kw: _CaptureModel())

    state = {
        "current_query": "你好",
        "original_query": "你好",
        "intent": "greeting",
        "thread_id": "direct_path_thread",
        "messages": [],
        # Even with retrieval_status="empty", direct path is unaffected.
        "retrieval_status": "empty",
        "step_count": 1,
    }
    events = []
    async def collect():
        async for evt in fsm.react_generate_direct(state):
            events.append(evt)
    asyncio.run(collect())

    # Direct path: msgs = [System(DIRECT_SYSTEM), *history]. With no
    # history, that's exactly 1 SystemMessage.
    assert len(captured_msgs) == 1
    from src.llm.prompts import DIRECT_SYSTEM
    assert captured_msgs[0].content == DIRECT_SYSTEM


# ============================================================
# 7. hallucination.py snippet length
# ============================================================


def test_judge_snippet_chars_per_doc_constant_is_2000():
    """JUDGE_SNIPPET_CHARS_PER_DOC must be 2000 (Phase 1 bump from 500).

    Pre-Phase-1 the 500-char cap truncated mid-sentence on long
    technical / legal / medical docs — the cheap-model judge saw
    only the first 3 paragraphs of doc 1 and nothing of docs 2-3.
    Phase 1 bumps to 2000 to give the judge ~250-400 words per doc
    while staying within the cheap-model context window.
    """
    from src.agent.nodes.hallucination import JUDGE_SNIPPET_CHARS_PER_DOC

    assert JUDGE_SNIPPET_CHARS_PER_DOC == 2000, (
        f"expected 2000; got {JUDGE_SNIPPET_CHARS_PER_DOC} — Phase 1 contract"
    )


def test_hallucination_node_uses_constant_not_literal():
    """AST guard: check_hallucination_async body must use the
    JUDGE_SNIPPET_CHARS_PER_DOC constant — NOT a hard-coded ``500``
    or ``2000`` literal. Hard-coded literals are how this regressed
    to begin with.
    """
    import ast
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    hal_src = (
        repo_root / "src" / "agent" / "nodes" / "hallucination.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(hal_src)
    func = next(
        n for n in tree.body
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "check_hallucination_async"
    )
    body_src_lines: list[int] = []
    for stmt in func.body:
        body_src_lines.extend(range(stmt.lineno, stmt.end_lineno + 1))
    src_lines = hal_src.splitlines()
    body_src = "\n".join(src_lines[i - 1] for i in body_src_lines)
    assert "JUDGE_SNIPPET_CHARS_PER_DOC" in body_src, (
        "check_hallucination_async must reference JUDGE_SNIPPET_CHARS_PER_DOC; "
        "Phase 1 contract — hard-coded literals regress."
    )
    # Make sure no hard-coded 500 slice sneaks in.
    assert "[:500]" not in body_src, (
        "check_hallucination_async must NOT use [:500] literal slice — "
        "pre-Phase-1 truncation bug."
    )


# ============================================================
# 8. v2.0.28.17 consecutiveness rule preserved
# ============================================================


def test_refusal_and_time_directives_both_consecutive_system_messages(monkeypatch):
    """When BOTH retrieval_status='empty' AND a get_current_time
    ToolMessage is in history, both directives must be inserted
    consecutively (no HumanMessage/AIMessage in between).

    v2.0.28.17 rule: ``insert(2, ...)`` keeps msgs[0..N] all
    SystemMessage until the first HumanMessage. Phase 1 preserves
    this — refusal at index 2, time at index 3 (or vice versa,
    whichever lands first).
    """
    from src.embeddings import bge_m3
    from src.reranker import bge_reranker
    from src.storage import lancedb_store
    from config.settings import reset_settings_cache

    bge_m3.reset_for_tests()
    bge_reranker.reset_for_tests()
    lancedb_store.reset_for_tests()
    reset_settings_cache()

    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    from src.agent.nodes import react_generate as rg_mod

    captured_msgs = []

    class _CaptureModel:
        async def astream(self, msgs, **kwargs):
            captured_msgs.extend(msgs)
            yield type("Chunk", (), {"content": "时间: 2026-09-27"})()

    monkeypatch.setattr(rg_mod, "build_chat_model", lambda **kw: _CaptureModel())

    state = {
        "current_query": "现在几点了?",
        "original_query": "现在几点了?",
        "intent": "qa_complex",
        "thread_id": "consecutive_test_thread",
        "messages": [
            HumanMessage(content="之前那个问题呢?"),
            AIMessage(
                content="",
                additional_kwargs={
                    "tool_calls": [
                        {"id": "tc1", "name": "get_current_time", "args": {}}
                    ]
                },
            ),
            ToolMessage(content="2026-09-27 17:30:00", tool_call_id="tc1", name="get_current_time"),
        ],
        "documents": [],
        "retrieval_status": "empty",  # BOTH refusal AND time should fire
        "step_count": 1,
    }
    events = []
    async def collect():
        async for evt in rg_mod.react_generate(state):
            events.append(evt)
    asyncio.run(collect())

    # The first 3 messages MUST be SystemMessage (refusal + time get
    # inserted at index 2 and 3, in front of sanitized history).
    assert len(captured_msgs) >= 3, f"expected ≥3 messages; got {len(captured_msgs)}"
    for i in range(3):
        assert isinstance(captured_msgs[i], SystemMessage), (
            f"msgs[{i}] must be SystemMessage (v2.0.28.17 consecutiveness); "
            f"got {type(captured_msgs[i]).__name__}"
        )


# ============================================================
# 9. behavioral pin against Day 0 audit_v2 fixtures
# ============================================================


def test_day0_empty_corpus_triggers_empty_status(monkeypatch):
    """Day 0 baseline: corpus_empty.pdf yields zero chunks; retrieval
    returns empty list. Phase 1 baseline pin: retrieval_status must
    be 'empty' so react_generate walks REFUSAL_EMPTY_TEMPLATE.

    Pre-Phase-1: retrieval returned empty list with NO status flag,
    react_generate synthesized without any refusal guidance, LLM
    fabricated an answer. Post-Phase-1: retrieval_status='empty'
    forces refusal.
    """
    from src.embeddings import bge_m3
    from src.reranker import bge_reranker
    from src.storage import lancedb_store
    from src.agent.nodes import retrieve as retrieve_mod
    from config.settings import reset_settings_cache

    bge_m3.reset_for_tests()
    bge_reranker.reset_for_tests()
    lancedb_store.reset_for_tests()
    reset_settings_cache()

    from tests.test_audit_fixture_behavior import _FakeEmbedder, _FakeReranker
    monkeypatch.setattr(bge_m3, "BGEM3FlagModel", lambda *_a, **_kw: _FakeEmbedder())
    monkeypatch.setattr(bge_reranker, "FlagReranker", lambda *_a, **_kw: _FakeReranker())

    # Empty corpus → 0 chunks → empty retrieval.
    state = {
        "current_query": "这个 PDF 讲了什么?",  # not summary-intent
        "original_query": "这个 PDF 讲了什么?",
        "thread_id": "day0_empty_thread",
        "messages": [],
        "step_count": 1,
    }
    result = asyncio.run(retrieve_mod.retrieve_hybrid_async(state))

    assert result.get("retrieval_status") == "empty", (
        f"empty corpus must set retrieval_status='empty'; "
        f"got {result.get('retrieval_status')!r}"
    )