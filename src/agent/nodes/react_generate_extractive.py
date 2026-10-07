"""v2.0.29.9 (Phase 8) — Verbatim extraction FSM node.

Runs INSTEAD of ``react_agent`` when ``state["high_precision"] == "on"``
(user toggle or hybrid auto-detect fired). The synthesis LLM is
constrained to verbatim quotes from chunks — NO reasoning, NO
rewriting, NO summary, NO inference. Each fact is on its own line,
prefixed with the existing ``[n]`` citation index (no new citation
marker shape per user decision 2026-09-28).

If the retrieved documents don't contain the answer (empty / empty_bulk /
low_relevance), the node falls back to the matching entry in
``src.llm.prompts.REFUSAL_TEMPLATES`` — same refusal contract Phase 1
established. Bumps ``EXTRACTIVE_FALLBACK`` counter (Phase 2 forward-
compat, pre-registered) on every fallback so dashboards can spot the
"user asked verbatim but corpus has nothing" pattern.

Async-generator protocol (per :mod:`src.agent.fsm`): yields zero or
more ``("answer_complete", payload)`` wire events, then ends with the
mandatory ``("__delta__", delta_dict)`` terminator. The FSM
(``fsm._run_node``) enforces exactly-one-``__delta__``.

Wire event reuse: Phase 8 emits ``("answer_complete", payload)`` —
identical shape to ``react_generate``. Each source in ``payload["sources"]``
carries ``verbatim=True`` (additive field, Phase 8 — see
``src.api.schemas.Source.verbatim``). The frontend reads this flag to
render a 🔒 icon on the bubble; old clients that don't know about it
simply render the source normally.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Tuple

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, SystemMessage

# IMPORTANT: import the metrics MODULE (not the Counter symbol). Tests
# elsewhere in this project call ``importlib.reload(src.agent.metrics)``
# to test reload semantics (see test_metrics_aggregation.py::metrics_app
# fixture). A bare ``from src.agent.metrics import EXTRACTIVE_FALLBACK``
# captures a stale Counter instance when the module is reloaded, so the
# bump here would land in the OLD instance while other tests / the
# /debug/metrics endpoint read from the NEW instance. Same pattern as
# verify_answer.py (Phase 6 v2.0.29.7).
from src.agent import metrics as _metrics_mod
from src.agent._time import now_iso as _now_iso
from src.agent.state import AgentState
from src.core.logging import logger
from src.llm.factory import build_chat_model
from src.llm.prompts import EXTRACTIVE_SYSTEM, REFUSAL_TEMPLATES
from src.agent.nodes.retrieve import retrieve_hybrid_async


def _build_sources(docs: list[Document], *, max_count: int = 8) -> list[dict]:
    """Build the wire-format sources list with ``verbatim=True``.

    Mirrors the dict shape emitted by ``react_generate`` (see
    ``runner._emit_fsm_event`` answer_complete branch) but stamps the
    ``verbatim`` flag on every entry. The frontend's
    ``withDefaultSourceKind`` + ``CitationChip`` code path tolerates
    the new field (additive field, ``Source`` Pydantic has
    ``extra="ignore"``).

    Each source is limited to the first 500 chars of
    ``page_content`` for the wire (UI preview; the full chunk lives
    in LanceDB if the user wants to drill in).
    """
    out: list[dict] = []
    for i, d in enumerate(docs[:max_count]):
        meta = d.metadata or {}
        out.append({
            "index": i + 1,
            "chunk_id": str(meta.get("chunk_id") or ""),
            "doc_id": str(meta.get("doc_id") or ""),
            "filename": str(meta.get("filename") or ""),
            "page": meta.get("page"),
            "section": meta.get("section"),
            "sheet": meta.get("sheet"),
            "text": (d.page_content or "")[:500],
            "score": float(meta.get("score", 0.0) or 0.0),
            "url": meta.get("url"),
            "domain": meta.get("domain"),
            "source_kind": "local",
            # v2.0.29.9 (Phase 8) — verbatim marker. Frontend renders
            # 🔒 icon when this is true.
            "verbatim": True,
        })
    return out


async def react_generate_extractive(
    state: AgentState, *, step: int = 0
) -> AsyncIterator[Tuple[str, Any]]:
    """FSM node: verbatim extraction (no reasoning, no rewriting).

    Async-gen protocol — yields ``("answer_complete", payload)`` wire
    events, then ends with the mandatory ``("__delta__", delta_dict)``
    terminator. ``payload`` carries ``answer`` + ``sources`` (each
    with ``verbatim=True``) + ``source_kinds``.

    Branches (mirrors Phase 6 verify_answer.py's layered design):

    1. **Refusal fallback (empty / empty_bulk / low_relevance)** —
       docs are empty or irrelevant → emit
       ``REFUSAL_TEMPLATES[retrieval_status]`` as the answer, empty
       sources, bump ``EXTRACTIVE_FALLBACK{reason=retrieval_status}``.

    2. **LLM exception** — LLM.ainvoke raises (network, rate limit,
       etc.) → bump ``EXTRACTIVE_FALLBACK{reason="llm_error"}``, emit
       ``REFUSAL_TEMPLATES["low_relevance"]`` as the answer.

    3. **Happy path** — LLM returns verbatim answer (with [n]
       citations) → emit answer + sources with ``verbatim=True``.

    ``state["high_precision"]`` is preserved (``"on"``) in the
    ``__delta__`` payload so downstream predicates / persistence
    layers know this turn was verbatim-mode.
    """
    docs = list(state.get("documents") or [])
    retrieval_status = state.get("retrieval_status") or "success"
    query = (
        state.get("original_query")
        or state.get("current_query")
        or ""
    )

    # v2.0.32.6 (Stage 5.8 real bug fix, 2026-10-07) — inline
    # retrieval load-bearing. The FSM (``fsm.after_intent``) routes
    # ``intent=qa_complex + high_precision=on`` DIRECTLY to
    # ``react_generate_extractive``, SKIPPING the
    # ``react_agent → tools → retrieve`` path that normally
    # populates ``state["documents"]``. Pre-fix, every verbatim
    # turn fell into the empty-docs refusal fallback below — the
    # 33.3% eval ceiling's golden-008-verbatim-zh/en root cause.
    # Phase 8 [[v2.0.29.9]] memory line 121 says "Phase 8 runs
    # AFTER retrieval" but the code never wired the inline retrieve
    # call; this 1-node fix mirrors the ``summary_path`` pattern
    # (which does inline retrieval for summary intents). Principle
    # (per [[v2.0.28.18]]): actual fix is often much simpler than
    # planned defense-in-depth — load-bearing 1 node function call.
    #
    # Cost: ~200-500 ms on first verbatim turn per thread (BGE-M3
    # embed + LanceDB hybrid search). Same cost as the normal
    # path's first retrieve call, so no new latency surface.
    #
    # Fall-through on retrieval failure: log + continue with empty
    # docs → refusal fallback below fires with honest "no docs"
    # template. Mirrors summary_path's behavior on storage hiccup.
    if not docs:
        try:
            ret = await retrieve_hybrid_async(state)
            docs = list(ret.get("documents") or [])
            retrieval_status = ret.get("retrieval_status") or retrieval_status
        except Exception as exc:
            logger.warning(
                "react_generate_extractive: inline retrieval failed: "
                "%s: %s — falling through with empty docs (refusal "
                "fallback will fire)",
                type(exc).__name__, exc,
            )
            # Leave docs empty; refusal fallback below handles it.

    # ---- Branch 1: refusal fallback (empty / low_relevance) ----
    if not docs or retrieval_status in ("empty", "empty_bulk", "low_relevance"):
        template_key = retrieval_status if retrieval_status in REFUSAL_TEMPLATES else "empty"
        answer = REFUSAL_TEMPLATES[template_key]
        _metrics_mod.EXTRACTIVE_FALLBACK.inc(reason=template_key)
        logger.info(
            "react_generate_extractive: refusal fallback reason=%s (docs=%d, retrieval_status=%s)",
            template_key, len(docs), retrieval_status,
        )
        yield ("answer_complete", {
            "answer": answer,
            "sources": [],
            "source_kinds": [],
            "route_decision": "extractive_refusal",
        })
        yield ("__delta__", {
            "answer": answer,
            "sources": [],
            "source_kinds": [],
            "high_precision": "on",
            # v2.0.32.6 — propagate inline-retrieval output so
            # ``should_check_hallucination`` predicate sees fresh
            # ``documents`` + ``retrieval_status`` (matches the
            # ``summary_path`` pattern at fsm.py:639-644).
            "documents": docs,
            "retrieval_status": retrieval_status,
        })
        return

    # ---- Branch 2 / 3: LLM call ----
    model = build_chat_model(enable_thinking=False)  # no reasoning in verbatim mode
    docs_blob = "\n\n---\n\n".join(
        f"[{i+1}] doc_id={d.metadata.get('doc_id','')} "
        f"chunk_id={d.metadata.get('chunk_id','')} "
        f"filename={d.metadata.get('filename','')}\n"
        f"{(d.page_content or '')[:2000]}"
        for i, d in enumerate(docs[:8])
    )
    user_msg = (
        f"Question (verbatim extraction required):\n{query}\n\n"
        f"Retrieved documents:\n{docs_blob}"
    )

    answer_text = ""
    try:
        # Phase 8 — disable extended thinking in extractive mode. The
        # synthesis LLM must NOT have a thinking channel to leak
        # meta-commentary into; verbatim output is mechanical, no
        # reasoning needed. ``enable_thinking=False`` is honored by
        # build_chat_model (see src/llm/factory.py).
        resp = await model.ainvoke([
            SystemMessage(content=EXTRACTIVE_SYSTEM),
            user_msg,
        ])
        answer_text = (getattr(resp, "content", "") or "").strip()
        if not answer_text:
            # Empty answer is a soft failure — fall through to refusal.
            _metrics_mod.EXTRACTIVE_FALLBACK.inc(reason="empty_answer")
            answer_text = REFUSAL_TEMPLATES["low_relevance"]
    except Exception as exc:
        # Last-ditch defense: any LLM exception (network, rate limit,
        # structured-output failure, etc.) → emit refusal template.
        # Fail-closed semantics match Phase 6 verify_answer_node.
        logger.exception(
            "react_generate_extractive: LLM call failed: %s: %s",
            type(exc).__name__, exc,
        )
        _metrics_mod.EXTRACTIVE_FALLBACK.inc(reason="llm_error")
        answer_text = REFUSAL_TEMPLATES["low_relevance"]

    sources = _build_sources(docs)
    created_at = _now_iso()
    additional_kwargs: dict = {
        "source_kinds": ["local"],
        "created_at": created_at,
        # v2.0.29.9 (Phase 8) — verbatim marker on AIMessage so
        # history replay rehydrates the 🔒 icon.
        "verbatim": True,
    }
    response = AIMessage(content=answer_text, additional_kwargs=additional_kwargs)

    yield ("answer_complete", {
        "answer": answer_text,
        "sources": sources,
        "source_kinds": ["local"],
        "created_at": created_at,
        "route_decision": "extractive",
    })
    yield ("__delta__", {
        "answer": answer_text,
        "sources": sources,
        "source_kinds": ["local"],
        "messages": [response],
        "created_at": created_at,
        # Preserve the verbatim-mode flag across the FSM so the
        # history / checkpointer sees this turn was extracted.
        "high_precision": "on",
        # v2.0.32.6 — propagate inline-retrieval output so the
        # verification chain (``after_react_generate_extractive``
        # predicate → ``verify_answer`` → ``check_hallucination``)
        # sees the same ``documents`` the extractive LLM saw.
        # Pre-fix this field was missing, so the predicate
        # ``docs_for_hallucination`` walked the empty
        # ``state["documents"]`` fallback chain and skipped the
        # whole verification chain.
        "documents": docs,
        "retrieval_status": retrieval_status,
    })


__all__ = ["react_generate_extractive"]