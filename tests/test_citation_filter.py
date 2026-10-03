"""Tests for the citation filter in ``stream_agent``.

Bug history
-----------
The user reported: "我观察到上传多个文件之后,针对其中一个文件内容提问,
答案内容只有正确文件的引用,但是回答下面的引用部分罗列出了多个文件所有
的chunk,修复一下".

The LLM correctly emits only ``[1]`` for the relevant file, but the
sidebar still rendered every retrieved chunk as a citation chip — the
runner was passing the full retrieved set through to the
``answer_complete`` event.

Fix: in ``stream_agent``'s ``generate_answer`` branch, extract every
``[n]`` marker from the answer text and keep only matching sources. If
no ``[n]`` markers are found (the direct greeting path), keep all
sources — the LLM answered without retrieval rather than the markers
being stripped.

These tests verify:

1. ``_extract_cited_indices`` parses every marker shape the LLM emits.
2. The ``answer_complete`` event carries ONLY the cited sources.
3. Empty citation text keeps all sources (greeting / direct path).
4. Sources whose ``index`` doesn't appear in the cited set are dropped.
5. Sources without an ``index`` field are also dropped (the LLM should
   always have indexed them).

Bug #2 (caught by these tests after fixture rework)
---------------------------------------------------
The runner used ``getattr(s, "index", None)`` to read the citation
index. That works on dataclasses / objects but ALWAYS returns ``None``
on the ``dict``-shaped Source the production pipeline actually emits
(via ``generate._to_sources`` / ``_web_sources.web_source``) — the
filter then drops every source and the user sees ``[1]`` markers in
the answer but no chips in the sidebar. The old test suite didn't
catch this because it used a ``@dataclass _FakeSource`` rather than
the real ``Source`` TypedDict. We now build sources via the actual
``generate._to_sources()`` helper so the fixture shape matches what
the runner actually consumes (dicts with ``.get``, not attribute access).
"""
from __future__ import annotations

from typing import Any

import pytest
from langchain_core.documents import Document


# ============================================================
# _extract_cited_indices — pure-string parser
# ============================================================


def test_extract_single_index():
    from src.agent.runner import _extract_cited_indices

    assert _extract_cited_indices("根据文档[1]的描述,答案为……") == {1}


def test_extract_comma_separated_indices():
    from src.agent.runner import _extract_cited_indices

    assert _extract_cited_indices("见[1, 3]以及[5]") == {1, 3, 5}


def test_extract_range_of_indices():
    from src.agent.runner import _extract_cited_indices

    assert _extract_cited_indices("参考资料[1-3]") == {1, 2, 3}


def test_extract_mixed_range_and_singles():
    from src.agent.runner import _extract_cited_indices

    assert _extract_cited_indices("见[1, 3-5, 7]") == {1, 3, 4, 5, 7}


def test_extract_reversed_range_still_yields_all_indices():
    """``[5-3]`` is malformed but should still yield {3,4,5}, not crash."""
    from src.agent.runner import _extract_cited_indices

    assert _extract_cited_indices("[5-3]") == {3, 4, 5}


def test_extract_empty_string_returns_empty_set():
    from src.agent.runner import _extract_cited_indices

    assert _extract_cited_indices("") == set()


def test_extract_text_without_any_brackets_returns_empty_set():
    from src.agent.runner import _extract_cited_indices

    assert _extract_cited_indices("这是一段没有任何引用的文本。") == set()


def test_extract_ignores_non_numeric_brackets():
    """``[link]`` and ``[foo]`` must not be treated as citations."""
    from src.agent.runner import _extract_cited_indices

    assert _extract_cited_indices("see [link] or [foo]") == set()


def test_extract_ignores_brackets_with_letters_and_digits():
    """``[v1]`` / ``[a1]`` are not citations — drop them."""
    from src.agent.runner import _extract_cited_indices

    assert _extract_cited_indices("[v1] [a1]") == set()


def test_extract_ignores_garbage_in_ranges():
    """``[1-abc]`` is malformed; parser should not raise and should drop
    the entire bracket rather than silently salvage a partial value."""
    from src.agent.runner import _extract_cited_indices

    # Whitelist-only regex rejects the whole group. We never want to
    # silently keep ``1`` out of ``[1-abc]`` — that would let a typo
    # in the LLM's answer accidentally count as a citation.
    assert _extract_cited_indices("[1-abc]") == set()


def test_extract_multiple_groups_in_one_string():
    from src.agent.runner import _extract_cited_indices

    text = "根据[1]和[2, 4]的对比,再加上[6-7]的内容,可以得出结论。"
    assert _extract_cited_indices(text) == {1, 2, 4, 6, 7}


def test_extract_strips_whitespace_in_groups():
    """``[ 1 , 3 ]`` with stray whitespace inside should still parse."""
    from src.agent.runner import _extract_cited_indices

    assert _extract_cited_indices("[ 1 , 3 ]") == {1, 3}


# ============================================================
# stream_agent answer_complete filtering
# ============================================================


def _build_sources(filenames: list[str]) -> list[dict]:
    """Build a list of real ``Source`` dicts via ``generate._to_sources``.

    The runner reads sources with ``s.get("index")`` — attribute access
    silently returns ``None`` on a dict, which is exactly the bug class
    this test suite exists to catch. Building via the real helper
    (instead of a dataclass stand-in) keeps the fixture shape in sync
    with the production wire format.
    """
    from src.agent.legacy_helpers.doc_formatting import _to_sources

    docs = [
        Document(
            page_content=f"content of {fname}",
            metadata={"filename": fname, "chunk_id": f"c{i}", "doc_id": f"d{i}"},
        )
        for i, fname in enumerate(filenames)
    ]
    return _to_sources(docs)


def _patch_runner_run_fsm_with_answer(
    monkeypatch, answer: str, sources: list, route_decision: str | None = None,
):
    """Patch ``runner.run_fsm`` to yield a single ``answer_complete``
    FSMEvent carrying the test's answer + sources.

    v2.0.19 (Phase 3) — the runner consumes FSMEvent dicts from
    ``run_fsm`` (no LangGraph graph / no messages+updates channels).
    ``answer_complete`` is the single FSM event that drives the
    ``answer_complete`` wire event.
    """
    from src.agent import runner

    async def fake_run_fsm(inputs, *, thread_id="", max_steps=None):
        yield {
            "kind": "answer_complete",
            "answer": answer,
            "sources": sources,
            "source_kinds": [],
            "created_at": "2024-01-01T00:00:00+00:00",
            "route_decision": route_decision,
        }

    # ``stream_agent`` does ``from src.embeddings.bge_m3 import
    # models_loaded`` lazily inside the function, so the binding lives
    # on the source module. Patch there.
    monkeypatch.setattr("src.embeddings.bge_m3.models_loaded", lambda: True)
    monkeypatch.setattr("src.reranker.bge_reranker.model_loaded", lambda: True)
    monkeypatch.setattr(runner, "run_fsm", fake_run_fsm)


async def _collect_answer_events(monkeypatch, answer: str, sources: list[dict]) -> list[dict]:
    """Drive ``stream_agent`` with a fake FSM and return ``answer_complete`` events.

    Centralized so each test only has to wire sources + answer — the
    ``models_loaded`` / ``model_loaded`` / ``run_fsm`` patches are
    identical across every test and live here once.
    """
    from src.agent import runner

    _patch_runner_run_fsm_with_answer(monkeypatch, answer, sources)

    events = []
    async for ev in runner.stream_agent("t", "q"):
        events.append(ev)
    return [e for e in events if e["type"] == "answer_complete"]


@pytest.mark.asyncio
async def test_answer_complete_filters_to_cited_only(monkeypatch):
    """The runner must drop sources whose ``index`` is NOT cited."""
    sources = _build_sources(["relevant.txt", "unrelated.txt", "also_unrelated.txt"])
    answer_events = await _collect_answer_events(
        monkeypatch, "答案见文档[1]。", sources
    )
    assert len(answer_events) == 1
    payload = answer_events[0]
    assert payload["answer"] == "答案见文档[1]。"
    # dict access — mirrors the runner's filter path. Previously a
    # dataclass fixture would have hidden the ``getattr(dict,
    # "index")`` bug; this version catches it.
    assert len(payload["sources"]) == 1
    assert payload["sources"][0]["index"] == 1
    assert payload["sources"][0]["filename"] == "relevant.txt"


@pytest.mark.asyncio
async def test_answer_complete_keeps_multiple_cited(monkeypatch):
    """If the LLM cites ``[1, 3]``, both sources ship through.

    v2.0.5 — the runner now renumbers citations to a dense 1..K
    range (closing the "answer says [1]–[23] but only 3 chips"
    bug). ``[1]`` stays ``[1]`` (it's already the smallest),
    and ``[3]`` becomes ``[2]`` (since the only other cited
    source is at new index 1). ``b.txt`` (the uncited source)
    is dropped entirely. The answer text is also rewritten so
    the citation markers point at the surviving chips.
    """
    sources = _build_sources(["a.txt", "b.txt", "c.txt"])
    answer_events = await _collect_answer_events(
        monkeypatch, "结合[1]和[3]的内容……", sources
    )
    assert len(answer_events) == 1
    payload = answer_events[0]
    indices = sorted(s["index"] for s in payload["sources"])
    assert indices == [1, 2]
    # v2.0.5 — answer text rewritten so the markers match the
    # surviving sources (1..2 instead of 1, 3).
    assert "[1]" in payload["answer"]
    assert "[3]" not in payload["answer"]
    assert "[2]" in payload["answer"]
    # Uncited source dropped.
    # v2.0.7 SoT-1: ``Source`` is Pydantic; attribute access.
    filenames = {s["filename"] for s in payload["sources"]}
    assert "b.txt" not in filenames


@pytest.mark.asyncio
async def test_answer_complete_keeps_range_cited(monkeypatch):
    """``[1-3]`` should keep all three sources."""
    sources = _build_sources(["a.txt", "b.txt", "c.txt", "d.txt"])
    answer_events = await _collect_answer_events(
        monkeypatch, "见参考资料[1-3]。", sources
    )
    indices = sorted(s["index"] for s in answer_events[0]["sources"])
    assert indices == [1, 2, 3]


@pytest.mark.asyncio
async def test_answer_complete_no_citations_on_retrieve_keeps_all_sources(monkeypatch):
    """The retrieve path: LLM emits no ``[n]`` markers, so we
    must NOT drop every source. The runner falls back to passing all
    sources through. (Note: this scenario is now rare in production
    because ``generate.py`` sets ``sources=[]`` on the direct path —
    see :mod:`tests.test_citation_direct_path` — but the legacy
    filter behavior is preserved when the caller passes
    ``route_decision=None`` or omits the kwarg.)"""
    sources = _build_sources(["a.txt", "b.txt"])
    answer_events = await _collect_answer_events(
        monkeypatch, "你好,我是 Yuan RAG,很高兴为你服务。", sources
    )
    indices = sorted(s["index"] for s in answer_events[0]["sources"])
    # The test fixture doesn't pass ``route_decision`` through the
    # fake graph (legacy contract). Without it the filter falls back
    # to the legacy "no markers → all sources" branch — confirmed
    # still wired so callers in the wild keep working.
    assert indices == [1, 2]


@pytest.mark.asyncio
async def test_answer_complete_drops_sources_without_index(monkeypatch):
    """A source whose ``index`` is missing is dropped once a cited
    set exists — we can't pair it to any ``[n]``, so showing it would
    be confusing."""
    sources = _build_sources(["a.txt", "no_index.txt"])
    # Wipe the index off the second source to simulate a malformed
    # checkpoint that pre-dates the ``index`` field.
    # v2.0.7 SoT-1: ``Source`` is Pydantic; use ``del`` not ``.pop()``.
    del sources[1].index
    answer_events = await _collect_answer_events(monkeypatch, "见[1]。", sources)
    assert len(answer_events) == 1
    assert len(answer_events[0]["sources"]) == 1
    assert answer_events[0]["sources"][0]["index"] == 1


@pytest.mark.asyncio
async def test_answer_complete_empty_answer_keeps_all_sources(monkeypatch):
    """Edge case: the LLM emits an empty answer. No citations, no
    filter — keep every source so the UI doesn't go blank."""
    sources = _build_sources(["a.txt", "b.txt"])
    answer_events = await _collect_answer_events(monkeypatch, "", sources)
    assert len(answer_events) == 1
    assert len(answer_events[0]["sources"]) == 2


@pytest.mark.asyncio
async def test_answer_complete_done_event_still_emitted(monkeypatch):
    """Filter logic must not break the post-try ``done`` event — the
    cancellation regression suite depends on this."""
    from src.agent import runner

    sources = _build_sources(["a.txt", "b.txt"])
    _patch_runner_run_fsm_with_answer(monkeypatch, "[2] only", sources)

    events = []
    async for ev in runner.stream_agent("t", "q"):
        events.append(ev)

    assert events[-1]["type"] == "done"
