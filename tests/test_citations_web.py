"""Phase 6 — citation rendering distinguishes local from web sources.

The frontend contract:

* Local source chip is a static span; clicking it highlights the source
  card in the right-hand panel.
* Web source chip is an ``<a target="_blank">`` whose href is the URL;
  the chip's text is ``[n] domain``.

We verify two helper contracts that drive this:

1. ``_format_documents`` (used inside the LLM prompt) renders local vs
   web headers differently so the LLM emits citations matching what the
   frontend expects.
2. ``_to_sources`` (the data shipped to the WS client) tags every
   source with ``source_kind`` and populates ``url`` / ``domain`` for
   web results, leaving them as ``None`` for local results.
"""
from __future__ import annotations

from langchain_core.documents import Document


def _local_doc(
    *,
    filename: str = "notes.txt",
    page: int | None = 1,
    sheet: str | None = None,
    section: str | None = None,
    text: str = "local content",
    score: float = 0.8,
) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source_kind": "local",
            "filename": filename,
            "page_number": page,
            "sheet": sheet,
            "section": section,
            "chunk_id": "c1",
            "doc_id": "d1",
            "score": score,
        },
    )


def _web_doc(
    *,
    title: str = "Example Domain",
    url: str = "https://example.com/article",
    domain: str = "example.com",
    snippet: str = "This is a web snippet.",
) -> Document:
    return Document(
        page_content=f"{title}\n\n{snippet}",
        metadata={
            "source_kind": "web",
            "url": url,
            "domain": domain,
            "filename": title,  # human-readable label
            "doc_id": "web-deadbeef1234",
            "score": 0.0,
        },
    )


# ============================================================
# _format_documents
# ============================================================


def test_format_documents_local_emits_filename_page():
    """Local docs render with ``[i] filename (page N) \\n text``."""
    from src.agent.legacy_helpers.doc_formatting import _format_documents

    out = _format_documents([_local_doc(filename="notes.txt", page=3)])
    assert "[1] notes.txt (page 3)" in out
    assert "local content" in out


def test_format_documents_local_includes_sheet_when_present():
    """Sheets on Excel docs surface as ``[i] file.xlsx (sheet Sheet1)``."""
    from src.agent.legacy_helpers.doc_formatting import _format_documents

    out = _format_documents(
        [_local_doc(filename="data.xlsx", sheet="Sheet1", page=None)]
    )
    assert "[1] data.xlsx (sheet Sheet1)" in out


def test_format_documents_web_emits_title_domain_url():
    """Web docs render with ``[i] title (domain) \\n snippet \\n URL:``."""
    from src.agent.legacy_helpers.doc_formatting import _format_documents

    out = _format_documents(
        [
            _web_doc(
                title="Example Domain",
                url="https://example.com/article",
                domain="example.com",
                snippet="snippet content",
            )
        ]
    )
    assert "[1] Example Domain (example.com)" in out
    assert "snippet content" in out
    assert "URL: https://example.com/article" in out


def test_format_documents_mixed_local_and_web():
    """A mix of local + web docs gets correctly indexed and rendered."""
    from src.agent.legacy_helpers.doc_formatting import _format_documents

    docs = [
        _local_doc(filename="a.txt", page=1),
        _web_doc(title="Web A", url="https://a.test/", domain="a.test"),
        _local_doc(filename="b.txt", page=2),
        _web_doc(title="Web B", url="https://b.test/", domain="b.test"),
    ]
    out = _format_documents(docs)
    # Indices are 1-based and contiguous
    assert "[1] a.txt" in out
    assert "[2] Web A (a.test)" in out
    assert "[3] b.txt" in out
    assert "[4] Web B (b.test)" in out
    # Web URLs follow their respective blocks
    assert "URL: https://a.test/" in out
    assert "URL: https://b.test/" in out


def test_format_documents_empty_returns_placeholder():
    """An empty doc list renders as a clear placeholder so the LLM
    doesn't hallucinate content.

    P1-4: the empty case now returns a self-closing ``<documents/>``
    fence rather than a free-form string — the fence format is the
    same shape the LLM will see for non-empty lists, just with an
    empty body. Asserting on the fence tag is more stable than
    matching the placeholder text.
    """
    from src.agent.legacy_helpers.doc_formatting import _format_documents

    out = _format_documents([])
    assert "<documents" in out
    assert "/>" in out  # self-closing


def test_format_documents_local_without_page_omits_paren():
    """Local docs without a page number don't render a stray ``()``."""
    from src.agent.legacy_helpers.doc_formatting import _format_documents

    out = _format_documents([_local_doc(filename="x.txt", page=None)])
    assert "[1] x.txt" in out
    assert "()" not in out
    assert "(page" not in out


# ============================================================
# _to_sources
# ============================================================


def test_to_sources_local_emits_source_kind_local():
    """Local docs become Sources with ``source_kind='local'`` and
    page / sheet / chunk_id populated; ``url`` / ``domain`` are absent
    so the frontend falls back to the static chip."""
    from src.agent.legacy_helpers.doc_formatting import _to_sources

    sources = _to_sources([_local_doc(filename="a.txt", page=2, sheet="S1")])
    assert len(sources) == 1
    s = sources[0]
    # v2.0.7 SoT-1: ``Source`` is now a Pydantic model, access via attribute.
    assert s.source_kind == "local"
    assert s.filename == "a.txt"
    assert s.page == 2
    assert s.sheet == "S1"
    assert s.chunk_id == "c1"
    assert s.doc_id == "d1"
    # url / domain are not populated for local
    assert s.url in (None, "")
    assert s.domain in (None, "")


def test_to_sources_web_emits_source_kind_web_with_url_and_domain():
    """Web docs become Sources with ``source_kind='web'`` and url /
    domain populated so the frontend renders the chip as an
    ``<a target="_blank">``."""
    from src.agent.legacy_helpers.doc_formatting import _to_sources

    sources = _to_sources(
        [
            _web_doc(
                title="Example Domain",
                url="https://example.com/article",
                domain="example.com",
            )
        ]
    )
    assert len(sources) == 1
    s = sources[0]
    # v2.0.7 SoT-1: ``Source`` is Pydantic; attribute access.
    assert s.source_kind == "web"
    assert s.url == "https://example.com/article"
    assert s.domain == "example.com"
    assert s.filename == "Example Domain"  # human-readable label
    # local-only fields are absent / empty
    assert s.page in (None, 0)
    assert s.chunk_id == ""
    # text is truncated to ~300 chars to keep the panel readable
    assert isinstance(s.text, str)
    assert len(s.text) <= 300


def test_to_sources_indices_are_1_based_and_contiguous():
    """Indices must start at 1 and be contiguous so the [n] markers
    in the LLM answer map 1:1 to the source cards."""
    from src.agent.legacy_helpers.doc_formatting import _to_sources

    sources = _to_sources(
        [
            _local_doc(filename="a.txt"),
            _web_doc(),
            _local_doc(filename="b.txt"),
        ]
    )
    assert [s.index for s in sources] == [1, 2, 3]


def test_to_sources_defaults_kind_to_local_when_missing():
    """If a doc's metadata omits ``source_kind`` (legacy data, manual
    ingestion), we default to ``"local"`` so it still renders — a
    web chip with no URL would be broken."""
    from src.agent.legacy_helpers.doc_formatting import _to_sources

    legacy = Document(
        page_content="legacy",
        metadata={"filename": "old.txt", "doc_id": "old"},
    )
    sources = _to_sources([legacy])
    assert sources[0].source_kind == "local"
    # And url / domain stay empty
    assert sources[0].url in (None, "")


def test_to_sources_truncates_text_to_300_chars():
    """The panel renders source cards inline with the answer; a 10 KB
    snippet would blow up the layout. We hard-cap at 300 chars."""
    from src.agent.legacy_helpers.doc_formatting import _to_sources

    long_text = "x" * 5000
    doc = _web_doc(snippet=long_text)
    sources = _to_sources([doc])
    # v2.0.7 SoT-1: ``Source`` is Pydantic; attribute access.
    assert len(sources[0].text) == 300


# ============================================================
# Mixed-source WS payload contract
# ============================================================


def test_to_sources_mixed_payload_keeps_kind_per_source():
    """The full WS payload for a mixed answer correctly tags each source
    so the per-source rendering decision (link vs static chip) is local
    to each source, not the whole answer."""
    from src.agent.legacy_helpers.doc_formatting import _to_sources

    docs = [
        _local_doc(filename="local.pdf"),
        _web_doc(title="Web", url="https://x.test/", domain="x.test"),
    ]
    sources = _to_sources(docs)

    assert sources[0].source_kind == "local"
    assert sources[0].filename == "local.pdf"
    assert sources[0].url in (None, "")

    # v2.0.7 SoT-1: ``Source`` is Pydantic; attribute access.
    assert sources[1].source_kind == "web"
    assert sources[1].url == "https://x.test/"
    assert sources[1].domain == "x.test"