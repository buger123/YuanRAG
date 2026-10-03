"""Tests for v2.0.22 (Item 7 Step 12) — docs_for_hallucination helper.

Pre-Step-12, two sites independently decided which field to read
for the hallucination judgment:

* ``src.agent.fsm.should_check_hallucination`` (FSM predicate)
  checked ``state.get("documents") or []`` only.
* ``src.agent.nodes.hallucination.check_hallucination_async``
  (judge node) checked
  ``state.get("graded_documents") or state.get("documents") or []``.

This is the P1-B6 audit finding: predicate ↔ node disagreement on
"what counts as docs". In the current ReAct topology they happen
to agree (``graded_documents`` is always empty — see state.py
docstring "currently unused in ReAct topology"), but a future
node that writes ``graded_documents`` would silently bypass the
predicate's skip-on-no-docs guard while the node would still
find docs and proceed.

Step 12 collapses the divergence into
:func:`src.agent.nodes._docs_for_hallucination.docs_for_hallucination`.
Both the predicate and the node call the helper, so the
"what counts as docs" definition lives in one place. The
fallback chain (``graded_documents → documents → []``) is the
node's pre-Step-12 behavior; the predicate inherits it for free.

These tests pin:

1. Helper resolution order: ``graded_documents`` wins over
   ``documents`` (forward-compat with a future grader step).
2. Empty inputs → ``[]`` (not ``None`` — callers do truthiness
   checks).
3. Returns a **list copy** (mutating the returned list must not
   mutate the state's source field — the node / predicate read
   vs the helper return should be independent references).
4. Predicate ↔ node agreement (the integration check that
   closed the audit gap).
"""
from __future__ import annotations

from langchain_core.documents import Document


def _doc(*, content: str = "x", doc_id: str = "d", chunk_id: str = "c") -> Document:
    return Document(page_content=content, metadata={"doc_id": doc_id, "chunk_id": chunk_id})


# ---------------------------------------------------------------------------
# 1. Helper resolution order
# ---------------------------------------------------------------------------


def test_returns_documents_when_only_documents_set():
    """The common case in v2.0: ``state["documents"]`` is populated
    by ``react_agent`` accumulating tool results, ``graded_documents``
    is empty. Helper returns the documents list.
    """
    from src.agent.nodes._docs_for_hallucination import docs_for_hallucination

    docs = [_doc(content="a"), _doc(content="b")]
    state = {"documents": docs, "graded_documents": []}

    out = docs_for_hallucination(state)
    assert out == docs


def test_graded_documents_wins_when_both_set():
    """Forward-compat: if a future grader step populates
    ``graded_documents``, the judge uses that (more authoritative)
    set instead of the raw ``documents``. The fallback chain
    ``graded_documents → documents → []`` is the pre-Step-12 node
    behavior; the predicate inherits it via this helper.
    """
    from src.agent.nodes._docs_for_hallucination import docs_for_hallucination

    raw = [_doc(content="raw")]
    graded = [_doc(content="graded1"), _doc(content="graded2")]
    state = {"documents": raw, "graded_documents": graded}

    out = docs_for_hallucination(state)
    assert out == graded
    assert out != raw  # explicit: graded wins, not raw


def test_returns_empty_list_when_neither_set():
    """Greeting / summary fast path where retrieval never ran —
    both fields are absent / empty. Helper returns ``[]`` (not
    ``None``) so caller's ``if not docs:`` truthiness check works
    uniformly.
    """
    from src.agent.nodes._docs_for_hallucination import docs_for_hallucination

    assert docs_for_hallucination({}) == []
    assert docs_for_hallucination({"documents": []}) == []
    assert docs_for_hallucination({"graded_documents": []}) == []
    assert docs_for_hallucination(
        {"documents": [], "graded_documents": []}
    ) == []


def test_handles_none_values():
    """Defensive: ``state.get("documents")`` can return ``None`` if
    a previous step explicitly set ``documents=None`` (rare but
    possible — pydantic-validated TypedDict can have ``None`` for
    non-required fields). Helper must not crash on ``None``.
    """
    from src.agent.nodes._docs_for_hallucination import docs_for_hallucination

    assert docs_for_hallucination({"documents": None}) == []
    assert docs_for_hallucination(
        {"documents": None, "graded_documents": None}
    ) == []
    # graded_documents=None falls through to documents branch.
    docs = [_doc(content="x")]
    assert docs_for_hallucination(
        {"documents": docs, "graded_documents": None}
    ) == docs


# ---------------------------------------------------------------------------
# 2. Return value is a list copy (not a view)
# ---------------------------------------------------------------------------


def test_returned_list_is_a_copy_not_a_view():
    """Mutating the returned list must not mutate the state's
    source field — the node reads the state multiple times during
    the judgment loop (once for the skip-check, once for the LLM
    prompt) and would observe stale / mutated data otherwise.
    """
    from src.agent.nodes._docs_for_hallucination import docs_for_hallucination

    docs = [_doc(content="a"), _doc(content="b")]
    state = {"documents": docs}

    out = docs_for_hallucination(state)
    out.append(_doc(content="mutated"))
    out.clear()

    # state["documents"] must remain the original list.
    assert state["documents"] == docs
    assert len(state["documents"]) == 2


def test_returned_list_is_a_copy_for_graded_branch_too():
    """Same copy semantics for the ``graded_documents`` branch."""
    from src.agent.nodes._docs_for_hallucination import docs_for_hallucination

    graded = [_doc(content="g1")]
    state = {"documents": [], "graded_documents": graded}

    out = docs_for_hallucination(state)
    out.clear()
    assert state["graded_documents"] == graded


# ---------------------------------------------------------------------------
# 3. Integration — predicate ↔ node agreement
# ---------------------------------------------------------------------------


def test_predicate_and_node_agree_on_no_docs():
    """The integration check that closed P1-B6: the FSM predicate
    (``should_check_hallucination``) and the judge node
    (``check_hallucination_async``) must return the same docs
    list when given the same state. Pre-Step-12 they didn't (the
    predicate read ``documents`` directly while the node read
    ``graded_documents → documents``). Step 12 collapses them
    through the helper.

    Specifically: given an empty state, the predicate should route
    to FSM_END (skip the judge) AND the node would skip (emit
    ``status="skipped"``). The decision boundary matches.
    """
    from src.agent.fsm import should_check_hallucination
    from src.agent.nodes._docs_for_hallucination import docs_for_hallucination

    state = {}  # no documents, no graded_documents
    # Predicate → None (= FSM end, judge skipped)
    assert should_check_hallucination(state) is None
    # Helper returns the same empty list the node would consume.
    assert docs_for_hallucination(state) == []


def test_predicate_and_node_agree_when_graded_set():
    """Forward-compat integration: if a future grader writes
    ``graded_documents``, the predicate sees the same docs the
    node would. Pre-Step-12 the predicate only looked at
    ``documents`` and would incorrectly skip the judge.
    """
    from src.agent.fsm import should_check_hallucination
    from src.agent.nodes._docs_for_hallucination import docs_for_hallucination

    docs = [_doc(content="only graded")]
    state = {
        "intent": "qa_complex",  # not greeting/summary/simple_fact
        "documents": [],  # raw retrieval empty (the future-grader case)
        "graded_documents": docs,  # but graded has docs
    }
    # Predicate → "check_hallucination" (don't skip — there ARE docs).
    assert should_check_hallucination(state) == "check_hallucination"
    # Helper returns the same docs the node would judge.
    assert docs_for_hallucination(state) == docs


def test_predicate_and_node_agree_on_documents_only():
    """Common ReAct case: only ``documents`` populated. Both sites
    see the same list.
    """
    from src.agent.fsm import should_check_hallucination
    from src.agent.nodes._docs_for_hallucination import docs_for_hallucination

    docs = [_doc(content="d1"), _doc(content="d2")]
    state = {
        "intent": "qa_complex",
        "documents": docs,
        "graded_documents": [],
    }
    assert should_check_hallucination(state) == "check_hallucination"
    assert docs_for_hallucination(state) == docs


def test_predicate_skips_greeting_intent():
    """``should_check_hallucination`` skips the judge when
    ``intent`` is ``greeting`` / ``summary`` / ``simple_fact``,
    regardless of docs. This is intent-based routing, not
    docs-based — the helper doesn't touch this branch. Pin so
    the predicate's intent logic doesn't drift.
    """
    from src.agent.fsm import should_check_hallucination

    # Even with docs present, greeting intent skips.
    state = {
        "intent": "greeting",
        "documents": [_doc()],
        "graded_documents": [_doc()],
    }
    assert should_check_hallucination(state) is None


# ---------------------------------------------------------------------------
# 4. Module surface
# ---------------------------------------------------------------------------


def test_helper_module_exports():
    """``__all__`` exposes the helper so callers can
    ``from src.agent.nodes._docs_for_hallucination import
    docs_for_hallucination`` without reaching into private module
    state.
    """
    from src.agent.nodes import _docs_for_hallucination

    assert "docs_for_hallucination" in _docs_for_hallucination.__all__


# ---------------------------------------------------------------------------
# 5. AST guard — no inline docs fallback chain in fsm.py or hallucination.py
# ---------------------------------------------------------------------------


def test_fsm_should_check_hallucination_uses_helper():
    """AST guard: the predicate's body must call
    ``docs_for_hallucination`` (no inline ``state.get("documents")``
    docs lookup). Catches accidental regression where a future
    contributor edits the predicate to bypass the helper.
    """
    import ast
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    fsm_src = (repo_root / "src" / "agent" / "fsm.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(fsm_src)
    func = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "should_check_hallucination"
    )
    body_src_lines: list[int] = []
    for stmt in func.body:
        body_src_lines.extend(range(stmt.lineno, stmt.end_lineno + 1))
    src_lines = fsm_src.splitlines()
    body_src = "\n".join(src_lines[i - 1] for i in body_src_lines)
    assert "docs_for_hallucination" in body_src, (
        "should_check_hallucination must call docs_for_hallucination "
        "in its body; P1-B6 audit gap closed in Step 12."
    )


def test_hallucination_node_uses_helper():
    """AST guard: the judge node's body must call
    ``docs_for_hallucination`` (no inline fallback chain).
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
    assert "docs_for_hallucination" in body_src, (
        "check_hallucination_async must call docs_for_hallucination "
        "(P1-B6 audit gap closed in Step 12)."
    )
