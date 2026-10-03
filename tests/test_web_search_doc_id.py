"""v2.0.29.6 (Phase 5) — web_search ``_doc_id_for_url`` includes domain.

Pre-Phase-5: ``web-{sha1[:12]}`` of URL only — domain not encoded.
Two URLs that hashed to the same digest on different domains collided
with no host disambiguation.

Phase 5 format: ``web-{domain}-{digest}`` where ``domain`` is the
URL's netloc (with leading ``www.`` stripped, non-``[a-z0-9-]``
chars replaced by ``-``, truncated to 32 chars) and ``digest`` is
the first 8 hex chars of ``sha1(url)``.

Also pins that ``chunk_id`` is populated in ``_to_doc`` (was ``""``
pre-Phase-5), so the Phase 4 PR-1 dedup key
``(doc_id, chunk_id, content_lead)`` keeps distinct web results from
the same URL distinct.
"""
from __future__ import annotations

import inspect

import pytest

from src.web_search import _doc_id_for_url, _to_doc


# ============================================================
# 1. Format shape
# ============================================================


def test_doc_id_includes_domain_segment():
    """``_doc_id_for_url`` returns ``web-{domain}-{digest}``."""
    out = _doc_id_for_url("https://example.com/article")
    parts = out.split("-")
    assert parts[0] == "web"
    assert "example" in out, (
        f"domain segment missing from doc_id {out!r}"
    )
    assert "com" in out
    # digest is 8 hex chars (sha1 truncated)
    digest_part = parts[-1]
    assert len(digest_part) == 8
    assert all(c in "0123456789abcdef" for c in digest_part), (
        f"digest part {digest_part!r} should be lowercase hex"
    )


def test_doc_id_strips_leading_www():
    """``www.example.com`` and ``example.com`` share the same domain segment."""
    a = _doc_id_for_url("https://www.example.com/foo")
    b = _doc_id_for_url("https://example.com/foo")
    # Different URLs → different digests. Same domain → same domain segment.
    assert a.startswith("web-example-com-")
    assert b.startswith("web-example-com-")


def test_doc_id_dedup_per_host():
    """Two URLs on different domains with similar paths should NOT collide."""
    a = _doc_id_for_url("https://example.com/foo")
    b = _doc_id_for_url("https://other.org/foo")
    assert "example-com" in a
    assert "other-org" in b
    assert a != b


def test_doc_id_stable_across_calls():
    """Same URL → same doc_id (deterministic)."""
    url = "https://news.example.com/article/123"
    assert _doc_id_for_url(url) == _doc_id_for_url(url)


def test_doc_id_sanitizes_unsafe_domain_chars():
    """Domain with port or unusual chars gets sanitized."""
    # Port is stripped by urlparse; underscore → dash
    out = _doc_id_for_url("https://my_host.example.org:8080/foo")
    # The colon-port should NOT appear in the domain segment.
    domain_seg = out.split("-", 1)[1].rsplit("-", 1)[0]
    assert ":" not in domain_seg


# ============================================================
# 2. chunk_id is populated (Phase 5 changes _to_doc)
# ============================================================


def test_to_doc_populates_chunk_id():
    """v2.0.29.6 — ``_to_doc`` sets ``metadata.chunk_id`` to a non-empty string."""
    from src.web_search._domain_hints import WebResult

    result: WebResult = {
        "url": "https://example.com/foo",
        "title": "Title",
        "snippet": "snippet",
        "domain": "example.com",
    }
    doc = _to_doc(result, position=0)
    assert doc.metadata.get("chunk_id"), (
        "_to_doc should populate chunk_id (was empty pre-Phase-5)"
    )
    assert doc.metadata["chunk_id"] == doc.metadata["doc_id"], (
        "chunk_id and doc_id should both derive from URL for web docs"
    )


def test_to_doc_keeps_other_metadata_intact():
    """``_to_doc`` still sets source_kind=web, url, domain, filename."""
    from src.web_search._domain_hints import WebResult

    result: WebResult = {
        "url": "https://example.com/foo",
        "title": "Title",
        "snippet": "snippet",
        "domain": "example.com",
    }
    doc = _to_doc(result, position=0)
    assert doc.metadata["source_kind"] == "web"
    assert doc.metadata["url"] == "https://example.com/foo"
    assert doc.metadata["domain"] == "example.com"
    assert doc.metadata["filename"] == "Title"


# ============================================================
# 3. Source inspection: dispatcher call site uses new format
# ============================================================


def test_dispatcher_uses_doc_id_for_url_with_domain():
    """v2.0.29.6 — _to_doc uses _doc_id_for_url for both doc_id and chunk_id."""
    src = inspect.getsource(_to_doc)
    # Both fields are populated by the helper.
    assert src.count("_doc_id_for_url(url)") >= 2, (
        "_to_doc should call _doc_id_for_url for both doc_id and chunk_id"
    )