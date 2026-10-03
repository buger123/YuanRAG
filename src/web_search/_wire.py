"""Web search wire-format serializer — Document list ↔ JSON string.

v2.0.18 — extracted from ``src/agent/tools/web_search.py`` where it
sat alongside the ``@tool`` wrapper. Lives here (rather than in
``tool.py``) so both the current ``@tool`` wrapper and Phase 3's
pure-async state machine can call it without pulling in LangChain's
``tool`` decorator.

Wire shape (the v2.0.2 contract)
--------------------------------
A list of Documents serializes to::

    [{"page_content": str, "metadata": {source_kind, url, domain,
    filename, doc_id, chunk_id, score, ...}}, ...]

Empty list → literal ``"[]"`` (NOT ``"[null]"`` or ``""``). The
``react_generate._extract_docs_from_messages`` consumer does ``json.loads``
on this string; a failed parse falls through to "no usable docs", so the
shape MUST be a valid JSON list.

Failure modes
-------------
* Defensive coercion: a metadata value that ``json.dumps`` rejects
  (a set, datetime, custom object) is converted via ``str()`` rather
  than crashing the tool call. The LLM can handle the resulting
  representation; the tool itself can't afford to raise (Phase 3 will
  feed this through the same exception handler).
"""
from __future__ import annotations

import json
from typing import Iterable


def _docs_to_json(docs: Iterable) -> str:
    """Serialize a list of Documents to a JSON string.

    Empty input → ``"[]"``.

    Defensive against unusual metadata values (sets, datetimes, custom
    objects) — those get coerced via ``str()`` rather than crashing the
    tool call. The list shape is preserved so ``react_generate`` can
    ``json.loads`` it back to reconstruct Document objects.
    """
    out: list[dict] = []
    for d in docs:
        meta = getattr(d, "metadata", None) or {}
        meta_safe: dict = {}
        for k, v in meta.items():
            try:
                json.dumps(v, ensure_ascii=False)
                meta_safe[k] = v
            except Exception:
                meta_safe[k] = str(v)
        out.append(
            {
                "page_content": getattr(d, "page_content", "") or "",
                "metadata": meta_safe,
            }
        )
    return json.dumps(out, ensure_ascii=False)


__all__ = ["_docs_to_json"]