"""Shared builder for ``source_kind="web"`` Source dicts.

Two nodes render web Documents as Source dicts for the chat UI:

* ``generate.py`` builds a mixed list (local + web) and needs the web
  branch specifically.
* ``search_web.py`` builds a web-only list.

The two implementations previously diverged (slightly different field
coercions); this module is the single source of truth.

v2.0.7 SoT-1: returns Pydantic ``SourceWire`` instances (was raw
``Source`` TypedDicts). The TypedDict is still importable for
``AgentState.sources`` typing; the wire payload is now Pydantic.
"""
from __future__ import annotations

from langchain_core.documents import Document

from src.agent.state import Source
from src.api.schemas import Source as SourceWire


def _is_web_doc(doc: Document) -> bool:
    """True if the document carries ``metadata.source_kind == "web"``.

    v2.0 — added for the ``web_search`` tool's filtering path.
    Documents produced by the tool are stamped with this metadata so
    downstream consumers (citations, react_generate merge) can
    distinguish them from local-document chunks.
    """
    meta = getattr(doc, "metadata", None) or {}
    return isinstance(meta, dict) and meta.get("source_kind") == "web"


def _position_score(position: int) -> float:
    """Position-decay score for a web result.

    Used as the ``score`` field of a web ``Source`` so the frontend
    ``<WebSource>`` chip can render a non-zero number. The web-search
    engines (Bing / DDG) rank by their own relevance signal which
    isn't exposed via the result dict, so we approximate with a
    smooth position-based decay: position 0 → 1.0, position 1 → 0.9,
    ..., position 9 → 0.1, position ≥ 10 → 0.0.

    Decay step is 0.1 so a single result list still spans most of
    [0, 1] — gives the chip card something to show rather than every
    source flattening to "0.0". Cheap (no embeddings, no LLM) and
    monotonic so position order is preserved when the chip is
    sorted by score.
    """
    if position < 0:
        position = 0
    return max(0.0, round(1.0 - 0.1 * position, 4))


def web_source(index: int, doc: Document) -> SourceWire:
    """Build a Source dict for a web Document at position ``index`` (1-based).

    Field contract (matches what the frontend ``<WebSource>`` chip reads):

    * ``index`` — 1-based position, mirrors ``[n]`` markers in the LLM's prose.
    * ``chunk_id`` — always empty (web snippets don't carry one).
    * ``doc_id`` — stringified (web results use ``web-<sha1[:12]>``).
    * ``filename`` — falls back to ``""`` so missing titles render
      as blank instead of the literal string ``"None"``.
    * ``url`` / ``domain`` — verbatim from metadata (may be ``None``).
    * ``text`` — first 300 chars of the snippet, the same cap used by
      the local-source branch so the chip cards are visually balanced.
    * ``score`` — position-decay value via :func:`_position_score`
      (``1.0`` for the first result, ``0.9`` for the second, etc).
      v2.0.5 — was hardcoded to ``0.0`` which made every chip
      render ``"0.0000"`` and obscured which result was the
      engine's top pick.

    v2.0.7 SoT-1: returns ``SourceWire`` (Pydantic) so the WS payload
    is type-checked at serialization. ``Source`` (TypedDict) remains
    the ``AgentState`` annotation; consumers of the wire payload see
    a fully-validated dict after ``.model_dump()``.
    """
    meta = doc.metadata or {}
    return SourceWire(
        index=index,
        chunk_id="",
        doc_id=str(meta.get("doc_id") or ""),
        filename=str(meta.get("filename") or ""),
        url=meta.get("url"),
        domain=meta.get("domain"),
        source_kind="web",
        text=(doc.page_content or "")[:300],
        score=_position_score(index - 1),
    )


def web_sources(docs: list[Document]) -> list[SourceWire]:
    """Convenience: build a list of web Sources for every Document."""
    return [web_source(i, doc) for i, doc in enumerate(docs, start=1)]


__all__ = ["web_source", "web_sources"]
