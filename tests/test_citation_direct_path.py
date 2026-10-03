"""Tests for the direct-path citation guard (v1.1.3).

The ``filter_sources_to_cited`` helper used to fall back to "return
all sources" when the LLM emitted no ``[n]`` markers, on the
assumption that "no markers = direct greeting path = the user
might still want to see what was retrieved". That assumption was
wrong on the doc-bearing greeting turn: the LLM router sometimes
misclassifies "你是谁" as "retrieve" (because there's an uploaded
PDF in the thread), 8 chunks get retrieved and attached as
sources, and the user sees 8 spurious ``[n]`` citation chips
pointing at the unrelated PDF on a "who are you?" answer.

The fix is two-layered:

1. ``generate.py`` returns ``sources=[]`` when
   ``route_decision == "direct"``. The conversational answer
   should never carry sources at all.

2. ``filter_sources_to_cited`` accepts an optional ``route_decision``
   kwarg. When provided and not ``"retrieve"``, it returns ``[]``
   unconditionally — a defense-in-depth check at the runner
   boundary so a future generate-node change that forgets to
   null out sources on the direct path still gets caught here.

These tests pin both layers.
"""
from __future__ import annotations

from langchain_core.documents import Document


def _local_doc(*, filename: str, chunk_id: str, doc_id: str) -> Document:
    return Document(
        page_content=f"content of {filename}",
        metadata={"filename": filename, "chunk_id": chunk_id, "doc_id": doc_id},
    )


def _build_sources(filenames: list[str]) -> list[dict]:
    """Same shape ``_to_sources`` produces (in
    ``legacy_helpers/doc_formatting``)."""
    from src.agent.legacy_helpers.doc_formatting import _to_sources

    docs = [_local_doc(filename=f, chunk_id=f"c{i}", doc_id=f"d{i}") for i, f in enumerate(filenames)]
    return _to_sources(docs)


# ============================================================
# filter_sources_to_cited — direct-path guard
# ============================================================


def test_direct_path_returns_empty_regardless_of_markers():
    """On ``route_decision == "direct"`` the runner MUST return [],
    even if the LLM happened to emit ``[n]`` markers (which it
    shouldn't — DIRECT_SYSTEM forbids them — but a future prompt
    regression shouldn't leak sources)."""
    from src.agent.citations import filter_sources_to_cited

    sources = _build_sources(["a.txt", "b.txt"])
    out = filter_sources_to_cited(sources, "见[1]和[2]", route_decision="direct")
    assert out == []


def test_direct_path_returns_empty_with_no_markers():
    """The original bug: greeting answer has no markers, the
    pre-fix filter returned ALL sources. Now returns []."""
    from src.agent.citations import filter_sources_to_cited

    sources = _build_sources(["a.txt", "b.txt", "c.txt"])
    out = filter_sources_to_cited(sources, "你好,我是 Yuan RAG", route_decision="direct")
    assert out == []


def test_retrieve_path_with_markers_filters_to_cited():
    """Sanity: the legacy filter behavior is preserved on the
    retrieve path — markers are honored, only cited sources pass."""
    from src.agent.citations import filter_sources_to_cited

    sources = _build_sources(["a.txt", "b.txt", "c.txt"])
    out = filter_sources_to_cited(sources, "答案见[2]", route_decision="retrieve")
    # v2.0.7 SoT-1: ``Source`` is Pydantic; attribute access.
    indices = [s["index"] for s in out]
    assert indices == [2]


def test_retrieve_path_without_markers_keeps_all_sources():
    """Legacy fallback preserved: on retrieve, no markers means
    surface all sources (the model may have answered from memory
    but the user might still want to see what was retrieved)."""
    from src.agent.citations import filter_sources_to_cited

    sources = _build_sources(["a.txt", "b.txt"])
    out = filter_sources_to_cited(sources, "我不确定具体答案", route_decision="retrieve")
    indices = sorted(s["index"] for s in out)
    assert indices == [1, 2]


def test_none_route_decision_falls_back_to_legacy():
    """A ``None`` route_decision means the caller (typically an
    older test path) didn't bother to look. We must NOT change
    the legacy filter behavior in that case — existing tests
    that call filter_sources_to_cited with two positional args
    keep working."""
    from src.agent.citations import filter_sources_to_cited

    sources = _build_sources(["a.txt", "b.txt"])
    # No route_decision kwarg → legacy behavior: no markers → all sources
    out = filter_sources_to_cited(sources, "你好")
    indices = sorted(s["index"] for s in out)
    assert indices == [1, 2]


# v2.0.16 — ``generate.generate_answer_async`` is dead (ReAct owns the
# synthesis path). The direct-path citation guard is now exercised by
# the live ReAct graph: see ``react_generate_direct`` in
# ``src/agent/graph.py`` which already returns ``sources=[]`` for the
# greeting / ``simple_fact`` fast-path. The retrieve-path equivalent
# is in ``react_generate``. End-to-end coverage moved to the
# runner-level tests (``test_citation_filter.py``).
