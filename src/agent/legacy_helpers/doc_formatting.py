"""Citation-formatting helpers — single source of truth for
``[n] filename (page N) <text>`` rendering, banner-kind aggregation,
and the wire-format ``Source`` chip list.

Originally lived in ``nodes/generate.py`` (DEPRECATED). Moved here
during the Phase-1 cleanup so the live ``react_generate`` /
``react_generate_direct`` / ``runner`` / ``graph`` imports point at
a contract file rather than a module whose main ``generate_answer``
entrypoint is dead.

Three helpers, all called by ReAct code:

* :func:`_format_documents` — wraps each Document in a
  ``<documents source="…" trust="untrusted">…</documents>`` fence
  with an injection-disarming preamble. Consumed by
  ``react_generate`` when building the LLM prompt for the citation
  path (NOT direct / greeting).

* :func:`_derive_source_kinds` — aggregates the ``["local", "web"]``
  kinds the banner uses. Consumed by ``react_generate``,
  ``react_generate_direct``, ``runner.py`` (as a fallback when the
  state delta didn't carry one), and ``graph.py``.

* :func:`_to_sources` — maps Documents to the Pydantic ``SourceWire``
  instances the frontend consumes as citation chips. Consumed by
  ``react_generate`` when building the ``answer_complete`` event.
"""
from __future__ import annotations

from src.agent.nodes._web_sources import web_source
from src.api.schemas import Source as SourceWire
from src.security.prompt_safety import wrap_documents


def _format_documents(docs) -> str:
    """Format docs as ``[i] header\\nbody`` blocks, then fence them.

    P1-4: every block is stripped of zero-width / RTL-override chars
    (so a doc can't smuggle ``system prompt`` past a string match)
    and the whole bundle is wrapped in a ``<documents>`` fence with a
    trust marker. The fence turns out-of-band instruction-override
    attempts into data the LLM is told to ignore.
    """
    blocks: list[str] = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        kind = meta.get("source_kind") or "local"
        if kind == "web":
            title = meta.get("filename") or meta.get("url") or "web result"
            domain = meta.get("domain") or ""
            url = meta.get("url") or ""
            header = f"[{i}] {title}"
            if domain:
                header += f" ({domain})"
            body = doc.page_content or ""
            url_line = f"\nURL: {url}" if url else ""
            blocks.append(f"{header}\n{body}{url_line}")
            continue
        # local
        label = meta.get("filename") or meta.get("source_path") or "doc"
        page = meta.get("page_number")
        sheet = meta.get("sheet")
        section = meta.get("section")
        loc = []
        if page:
            loc.append(f"page {page}")
        if sheet:
            loc.append(f"sheet {sheet}")
        if section:
            loc.append(section)
        loc_str = f" ({', '.join(loc)})" if loc else ""
        blocks.append(f"[{i}] {label}{loc_str}\n{doc.page_content}")

    if not blocks:
        return wrap_documents([], source="none")
    # Web sources are inherently more adversarial (arbitrary third
    # party) than local uploads (the user's own files). Tag them
    # separately in the fence so the LLM can weight them.
    web_count = sum(
        1
        for d in docs
        if (d.metadata or {}).get("source_kind") == "web"
    )
    source_label = "user-uploaded+web" if web_count else "user-uploaded"
    return wrap_documents(blocks, source=source_label)


def _derive_source_kinds(docs) -> list[str]:
    """Return the deduplicated, sorted kinds of ``docs``.

    Empty list when ``docs`` is empty (direct / greeting path). Each
    doc carries ``metadata["source_kind"]`` ∈ {``"local"``, ``"web"``};
    fixtures / old payloads may omit the field, in which case we
    default to ``"local"`` (the legacy behaviour — everything used to
    be a local doc). Sorted for wire stability so the frontend's
    deep-equality check (``source_kinds.includes("web")`` etc.)
    doesn't churn across renders.
    """
    kinds: set[str] = set()
    for d in docs or []:
        meta = getattr(d, "metadata", None) or {}
        kind = meta.get("source_kind") or "local"
        kinds.add(str(kind))
    return sorted(kinds)


def _to_sources(docs) -> list[SourceWire]:
    """Build the citation chip list for the wire.

    v2.0.7 SoT-1: returns Pydantic ``SourceWire`` instances (was raw
    ``Source`` TypedDicts). The TypedDict still flows through the
    LangGraph ``AgentState.sources`` field (Pydantic's
    ``extra="ignore"`` accepts dict-shaped input from older
    checkpointer rows), but the WS payload is now type-checked at
    serialization time. Web branches go through
    :func:`src.agent.nodes._web_sources.web_source` which is the
    canonical builder; local branches build directly.
    """
    out: list[SourceWire] = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        kind = meta.get("source_kind") or "local"
        if kind == "web":
            out.append(web_source(i, doc))
        else:
            out.append(
                SourceWire(
                    index=i,
                    chunk_id=str(meta.get("chunk_id") or ""),
                    doc_id=str(meta.get("doc_id") or ""),
                    filename=str(meta.get("filename") or ""),
                    page=meta.get("page_number"),
                    section=meta.get("section"),
                    sheet=meta.get("sheet"),
                    source_kind="local",
                    text=doc.page_content[:300],
                    score=float(meta.get("score") or 0.0),
                )
            )
    return out