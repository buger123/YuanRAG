"""Legacy 5-stage graph helpers — the live private utilities the
ReAct rewrite still depends on.

v2.0 ReAct rewrite (2026-09) replaced the linear
``route_query → retrieve → grade → search_web → fetch_web_content →
generate_answer → check_hallucination`` topology with a ReAct loop.
The graph nodes themselves are dead and have been deleted, but the
private helpers inside them are real contract that active code paths
still import:

* ``greeting._is_obvious_greeting`` — used by
  ``nodes.intent_analysis`` as the no-LLM greeting pre-filter.
* ``doc_formatting._format_documents`` / ``_derive_source_kinds`` /
  ``_to_sources`` — used by ``nodes.react_generate``,
  ``nodes.react_generate_direct`` (inlined in ``graph.py``),
  ``runner.py``, and ``graph.py`` to fence untrusted documents,
  aggregate banner kinds, and produce the citation chip list for the
  WS payload.

v2.0.18 — ``page_fetch.py`` was deleted in Phase 2 Step 7. The
URL fetcher + readability helpers now live in ``src/web_search/``
(``fetch.py`` + ``readability.py``); the ``@tool`` web_search wrapper
imports ``_fetch_one`` from there directly.

These helpers were previously scattered across the three DEPRECATED
files (``nodes/route.py`` / ``nodes/generate.py`` /
``nodes/fetch_web_content.py``) as private utility functions. Moving
them to a dedicated ``legacy_helpers`` package makes the dependency
explicit — a future maintainer who wants to delete "the legacy 5-stage
graph" can find the one contract that's still in use in one place.

Why a separate package and not inlined into the active node files
----------------------------------------------------------------
``_format_documents`` / ``_to_sources`` / ``_derive_source_kinds`` are
~60 lines of citation-formatting logic. Inlining them into
``react_generate.py`` would push that file past 800 lines and obscure
the ReAct flow. Keeping them in ``legacy_helpers`` keeps each module
focused and the helpers independently testable.
"""