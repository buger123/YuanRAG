"""v2.0.28.5 Item 11 PR-3 F-R7 — ``_enrich_with_pages`` returns
exceptions independently instead of cancelling siblings.

Pre-PR-3, ``asyncio.gather(*tasks, return_exceptions=False)`` would
cancel all sibling tasks the moment one raised. A single bad URL
(e.g. one whose parser hits an unexpected ``MemoryError``) would
poison the whole enrichment — every other URL's page_content would
be lost. Post-PR-3, ``return_exceptions=True`` + the
``isinstance(body, BaseException)`` filter let each task raise
independently; the surviving fetches complete normally and the
failed doc keeps its snippet.

This file pins the per-task exception isolation contract.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_enrich_handles_per_task_exceptions_isolated():
    """3 URLs, the middle one raises ``MemoryError`` — the other 2
    must complete with their page_content replaced, and the failing
    doc must keep its snippet (not be dropped, not be enriched).

    Pre-PR-3: ``gather(..., return_exceptions=False)`` would cancel
    the 3rd task as soon as the 2nd raised, AND the outer ``except
    Exception`` would log + return docs unchanged — losing the
    successful 1st enrichment too. Post-PR-3: the 1st and 3rd
    succeed, the 2nd keeps its snippet, and the doc list is
    preserved.
    """
    from langchain_core.documents import Document

    import src.web_search as ws

    # Build 3 docs, each with a URL.
    docs = [
        Document(
            page_content="snippet-A",
            metadata={"source_kind": "web", "url": "https://a.example/", "domain": "a.example", "filename": "A", "doc_id": "web-a", "chunk_id": "", "score": 1.0},
        ),
        Document(
            page_content="snippet-B",
            metadata={"source_kind": "web", "url": "https://b.example/", "domain": "b.example", "filename": "B", "doc_id": "web-b", "chunk_id": "", "score": 0.9},
        ),
        Document(
            page_content="snippet-C",
            metadata={"source_kind": "web", "url": "https://c.example/", "domain": "c.example", "filename": "C", "doc_id": "web-c", "chunk_id": "", "score": 0.8},
        ),
    ]

    # Spy _fetch_one: A succeeds, B raises MemoryError, C succeeds.
    async def spy_fetch_one(url: str, max_chars: int):
        if "b.example" in url:
            raise MemoryError("simulated parser blow-up")
        # Pad so body > 200 chars (anti-bot filter).
        return f"body of {url} " + ("x" * 300)

    with patch("src.web_search._fetch_one", side_effect=spy_fetch_one):
        enriched = await ws._enrich_with_pages(docs, max_chars=4000)

    # All 3 docs should survive (none dropped).
    assert len(enriched) == 3, (
        f"expected 3 docs to survive, got {len(enriched)} (pre-PR-3: gather cancel drops siblings)"
    )
    # Doc A and C should have their snippet replaced with the fetched body.
    by_url = {d.metadata["url"]: d for d in enriched}
    assert "body of https://a.example/" in by_url["https://a.example/"].page_content
    assert "body of https://c.example/" in by_url["https://c.example/"].page_content
    # Doc B keeps its snippet (fetch failed).
    assert by_url["https://b.example/"].page_content == "snippet-B"


@pytest.mark.asyncio
async def test_enrich_warning_log_on_per_task_exception():
    """When a per-task exception occurs, an ``op=web_search.enrich
    fetch_exception`` WARNING must be emitted (operator visibility).
    """
    import loguru

    from langchain_core.documents import Document

    import src.web_search as ws

    captured: list[dict] = []

    def _sink(message) -> None:
        record = message.record
        captured.append({"level": record["level"].name, "message": record["message"]})

    sink_id = loguru.logger.add(_sink, level="DEBUG")

    try:
        docs = [
            Document(
                page_content="snippet-X",
                metadata={"source_kind": "web", "url": "https://x.example/", "domain": "x.example", "filename": "X", "doc_id": "web-x", "chunk_id": "", "score": 1.0},
            ),
        ]

        async def spy_fetch_one(url: str, max_chars: int):
            raise ValueError("simulated unexpected parser error")

        with patch("src.web_search._fetch_one", side_effect=spy_fetch_one):
            await ws._enrich_with_pages(docs, max_chars=4000)

        enrich_warnings = [
            c for c in captured
            if c["level"] == "WARNING" and "op=web_search.enrich" in c["message"]
        ]
        assert len(enrich_warnings) >= 1, (
            f"expected ≥1 op=web_search.enrich WARNING, got: "
            f"{[c for c in captured if c['level'] == 'WARNING']}"
        )
        # The WARNING must mention the exception type.
        assert any("ValueError" in w["message"] for w in enrich_warnings), (
            f"WARNING should include exception type, got: {enrich_warnings}"
        )
    finally:
        loguru.logger.remove(sink_id)


@pytest.mark.asyncio
async def test_enrich_no_exceptions_returns_normally():
    """Sanity check: when no task raises, all docs are enriched
    (no regression vs. pre-PR-3 happy path).
    """
    from langchain_core.documents import Document

    import src.web_search as ws

    docs = [
        Document(
            page_content="snippet-Y",
            metadata={"source_kind": "web", "url": "https://y.example/", "domain": "y.example", "filename": "Y", "doc_id": "web-y", "chunk_id": "", "score": 1.0},
        ),
    ]

    async def spy_fetch_one(url: str, max_chars: int):
        return "fetched body " + ("y" * 300)

    with patch("src.web_search._fetch_one", side_effect=spy_fetch_one):
        enriched = await ws._enrich_with_pages(docs, max_chars=4000)

    assert len(enriched) == 1
    assert "fetched body" in enriched[0].page_content
