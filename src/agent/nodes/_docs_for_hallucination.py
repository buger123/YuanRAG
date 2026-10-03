"""Single source for "what docs does the hallucination judge look at?".

v2.0.22 (Item 7 Step 12) — P1-B6 关掉.

Pre-Step-12, two sites independently decided which field to read
for the hallucination judgment:

* :func:`src.agent.fsm.should_check_hallucination` (the FSM
  predicate) checked ``state.get("documents") or []`` only.
* :func:`src.agent.nodes.hallucination.check_hallucination_async`
  (the judge node) checked
  ``state.get("graded_documents") or state.get("documents") or []``.

This divergence is the P1-B6 audit finding: the predicate and the
node disagree on "what counts as docs". In practice they always
agree (the v2.0 ReAct topology never populates ``graded_documents``
— it was a v1.1.x field that the state docstring explicitly marks
``currently unused``). But the asymmetry is a footgun: a future
node that writes ``graded_documents`` (e.g. a re-ranker with a
dedicated grader step) would silently bypass the predicate's
skip-on-no-docs guard, and the judge would either run with
``graded_documents`` (today's node behavior) or be incorrectly
skipped (today's predicate behavior) — depending on which site
ran first.

Step 12 collapses the divergence into a single helper. Both the
predicate and the node call :func:`docs_for_hallucination` so the
"what counts as docs" definition lives in one place. The
fallback chain (``graded_documents → documents → []``) is the
node's pre-Step-12 behavior; the predicate inherits it for free
(no behavior change in the current ReAct topology where
``graded_documents`` is always empty).

Why a function not a constant
----------------------------
A module-level constant like ``_DOCS_FIELDS = ("graded_documents",
"documents")`` would require callers to loop themselves; the
helper returns the resolved list (or ``[]``), saving every
caller the ``or []`` fallback. Centralization is the win, not
micro-optimization.

Why not in ``hallucination.py``
-------------------------------
The helper is shared between an FSM predicate (top-level
``fsm.py``) and a node (``hallucination.py``). Putting it in
the node module would create a fsm → node import direction that
already exists in reverse (``fsm.py`` imports
``check_hallucination_async`` from ``hallucination.py``).
Putting it in ``_docs_for_hallucination.py`` (this file) makes
both directions depend on the helper, not on each other.
"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document


def docs_for_hallucination(state) -> List[Document]:
    """Return the documents the hallucination judge should look at.

    Resolves the fallback chain ``graded_documents → documents → []``:

    * ``graded_documents`` is a v1.1.x field that the current ReAct
      topology never populates (see ``state.py`` docstring
      "currently unused in ReAct topology"). Kept in the fallback
      chain for forward-compat with a future grader step that might
      write it.
    * ``documents`` is the v2.0 primary field (populated by
      ``react_agent`` accumulating tool results).
    * ``[]`` if neither is set (greeting / summary fast path
      where retrieval never ran — there's nothing to verify
      against, and both the predicate and the node agree the
      judge should skip).

    Used by:

    * :func:`src.agent.fsm.should_check_hallucination` — the FSM
      predicate that decides whether to route to the judge node
      at all. If this returns ``[]``, the predicate routes to
      ``FSM_END`` (skip the judge).
    * :func:`src.agent.nodes.hallucination.check_hallucination_async`
      — the judge node itself. If this returns ``[]``, the node
      emits a ``grounding`` event with ``status="skipped"``.

    The two callers must agree on the answer for the FSM ↔ node
    contract to hold (skip-on-no-docs means predicate skips AND
    node would have skipped).
    """
    graded = state.get("graded_documents") or []
    if graded:
        return list(graded)
    return list(state.get("documents") or [])


__all__ = ["docs_for_hallucination"]
