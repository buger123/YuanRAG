"""Agent node subpackage — ReAct-era live nodes.

The v2.0 ReAct rewrite (2026-09) replaced the legacy 5-stage linear
graph (route_query → retrieve → grade → rewrite → search_web →
fetch_web_content → generate_answer → check_hallucination) with a
ReAct loop where the LLM autonomously decides which tools to invoke.
The graph (see ``src/agent/graph.py``) now wires only:

    intent_analysis ──┬── react_generate_direct ──→ check_hallucination
                      ├── summary_path ──→ react_generate ──→ …
                      └── react_agent ⇄ tools (ToolNode)
                          ──→ react_generate ──→ check_hallucination

Submodules (access via ``src.agent.nodes.<submodule>``):

* ``intent_analysis`` — first-pass intent classifier (greeting /
  summary / simple_fact / qa_complex); consumed by ``graph.build_graph``.
* ``react_agent`` — the ReAct loop node (LLM picks tools).
* ``react_generate`` — final-answer synthesis after the ReAct loop.
* ``hallucination`` — guard node after answer synthesis.
* ``retrieve`` — hybrid BGE-M3 + BM25 + rerank helper (called by the
  ``retrieve_docs`` tool, not by the graph).

Re-exports in this ``__init__`` (names that do NOT collide with a
submodule name):

* :func:`retrieve_hybrid_async` — the async retrieval helper.

v2.0.16 — the previous ``__init__`` re-exported
``from .react_agent import react_agent`` and
``from .react_generate import react_generate``. That binding shadows
the submodule name in ``src.agent.nodes.__dict__``, so any test doing
``import src.agent.nodes.react_agent as M`` got the *function* (not
the module), which made ``patch.object(M, "build_chat_model", ...)``
fail with ``AttributeError: function ... has no attribute
'build_chat_model'``. The submodule re-exports are removed here so
``import src.agent.nodes.react_agent`` always resolves to the
submodule. Callers that need the function should use the explicit
submodule path: ``from src.agent.nodes.react_agent import
react_agent``.

v2.0.22 (Item 7 Step 6) — the ``from .intent_analysis import
intent_analysis`` re-export was also removed because it caused the
same shadow: ``import src.agent.nodes.intent_analysis as M`` resolved
to the function, not the module, breaking ``vars(M)`` / AST-style
module scans in ``tests/test_node_protocol.py::test_all_nodes_are_async_gen_in_source``.

The dead 5-stage-graph nodes (``route_query`` / ``generate_answer``
/ ``fetch_web_content``) were removed in v2.0.16. Their private
helpers (``_is_obvious_greeting`` / ``_format_documents`` /
``_derive_source_kinds`` / ``_to_sources`` / ``_fetch_one``) live
in :mod:`src.agent.legacy_helpers` so the dependency on the live
code paths is explicit and one-place-discoverable.
"""
from .retrieve import retrieve_hybrid_async

__all__ = [
    "retrieve_hybrid_async",
]