"""Tests for ``src.ingestion.parsers.pdf_fallback``.

The fallback wraps ``pypdfium2`` to extract the text layer of a PDF
when Docling's OCR pipeline returns zero chunks. It should:

- Return ``[]`` cleanly when pypdfium2 isn't available.
- Yield one record per non-empty page when a text layer exists.
- Yield ``[]`` for genuinely image-only PDFs (no text layer).
- Tolerate corrupt pages without aborting the whole extraction.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.parsers import pdf_fallback


class _FakeTextPage:
    def __init__(self, text: str):
        self._text = text
        self.closed = False

    def get_text_range(self):
        return self._text

    def close(self):
        self.closed = True


class _FakePage:
    def __init__(self, text: str = ""):
        self._tp = _FakeTextPage(text)

    def get_textpage(self):
        return self._tp


class _FakePdf:
    def __init__(self, pages: list[_FakePage]):
        self._pages = pages
        self.closed = False

    def __len__(self):
        return len(self._pages)

    def __getitem__(self, i):
        return self._pages[i]

    def close(self):
        self.closed = True


def test_returns_empty_when_pypdfium2_missing(monkeypatch):
    """If pypdfium2 isn't installed, the fallback is a no-op."""
    monkeypatch.setattr(pdf_fallback, "pdfium", None)
    assert pdf_fallback.extract_pdf_text_layer(b"%PDF-fake") == []


def test_extracts_text_layer_page_by_page():
    """One record per non-empty page, with sequential page numbers."""
    fake = _FakePdf(
        [
            _FakePage("Hello, world."),
            _FakePage(""),  # empty page → skipped
            _FakePage("Second page.\nWith two lines."),
            _FakePage("   \n\t  "),  # whitespace-only → skipped
        ]
    )
    with patch.object(pdf_fallback, "pdfium") as pdfium_mock:
        pdfium_mock.PdfDocument.return_value = fake
        result = pdf_fallback.extract_pdf_text_layer(b"%PDF-fake")
    assert len(result) == 2
    assert result[0]["text"] == "Hello, world."
    assert result[0]["meta"]["page_number"] == 1
    assert result[0]["meta"]["chunk_type"] == "text"
    assert result[1]["text"] == "Second page.\nWith two lines."
    assert result[1]["meta"]["page_number"] == 3
    # All textpages should be closed to avoid native-handle leaks.
    assert all(p._tp.closed for p in fake._pages)
    assert fake.closed


def test_returns_empty_for_image_only_pdf():
    """Every page returns empty text → no records (genuine image-only)."""
    fake = _FakePdf([_FakePage(""), _FakePage(""), _FakePage("")])
    with patch.object(pdf_fallback, "pdfium") as pdfium_mock:
        pdfium_mock.PdfDocument.return_value = fake
        result = pdf_fallback.extract_pdf_text_layer(b"%PDF-fake")
    assert result == []


def test_handles_open_failure():
    """A non-PDF or corrupt payload shouldn't crash the caller."""
    with patch.object(pdf_fallback, "pdfium") as pdfium_mock:
        pdfium_mock.PdfDocument.side_effect = RuntimeError("not a PDF")
        result = pdf_fallback.extract_pdf_text_layer(b"garbage")
    assert result == []


def test_continues_on_per_page_error():
    """One broken page shouldn't abort the whole extraction."""
    bad_page = MagicMock()
    bad_page.get_textpage.side_effect = RuntimeError("broken")
    good_page = _FakePage("This page is fine.")

    fake = _FakePdf([bad_page, good_page])
    with patch.object(pdf_fallback, "pdfium") as pdfium_mock:
        pdfium_mock.PdfDocument.return_value = fake
        result = pdf_fallback.extract_pdf_text_layer(b"%PDF-fake")
    assert len(result) == 1
    assert result[0]["text"] == "This page is fine."
    assert result[0]["meta"]["page_number"] == 2


def test_run_ingestion_triggers_fallback_when_docling_returns_empty(monkeypatch):
    """End-to-end: Docling returns no chunks, fallback returns 2 pages.
    The pipeline must return the fallback records (not empty)."""
    from src.ingestion import pipeline
    from src.ingestion.chunkers import hybrid_chunker
    from src.ingestion.format_detect import Format
    from src.ingestion.parsers import docling_parser
    from src.ingestion.parsers import pdf_fallback as pdf_fallback_mod

    monkeypatch.setattr(
        pipeline, "detect", lambda source, content: Format.PDF
    )
    monkeypatch.setattr(
        docling_parser,
        "parse_with_docling",
        lambda target: object(),  # any non-None sentinel
    )
    # ``chunk_docling_doc`` and ``extract_pdf_text_layer`` are imported
    # inside ``run_ingestion`` via deferred imports, so we patch them
    # at their source modules (the local-name rebinding inside
    # ``run_ingestion`` is invisible to ``monkeypatch.setattr``).
    monkeypatch.setattr(
        hybrid_chunker,
        "chunk_docling_doc",
        lambda doc: iter([]),  # 0 chunks — the broken case
    )
    fallback_records = [
        {"text": "page 1 text", "meta": {"page_number": 1}},
        {"text": "page 2 text", "meta": {"page_number": 2}},
    ]
    monkeypatch.setattr(
        pdf_fallback_mod,
        "extract_pdf_text_layer",
        lambda content: fallback_records,
    )
    result = pipeline.run_ingestion("ticket.pdf", b"%PDF-fake")
    assert result == fallback_records


def test_run_ingestion_skips_fallback_when_docling_returns_chunks(monkeypatch):
    """If Docling produced chunks, we should never consult the fallback —
    its result is ignored even if it would return data."""
    from src.ingestion import pipeline
    from src.ingestion.chunkers import hybrid_chunker
    from src.ingestion.format_detect import Format
    from src.ingestion.parsers import docling_parser
    from src.ingestion.parsers import pdf_fallback as pdf_fallback_mod

    monkeypatch.setattr(
        pipeline, "detect", lambda source, content: Format.PDF
    )
    monkeypatch.setattr(
        docling_parser, "parse_with_docling", lambda target: object()
    )
    docling_chunks = [
        {"text": "from docling", "meta": {"page_number": 1}},
    ]
    monkeypatch.setattr(
        hybrid_chunker,
        "chunk_docling_doc",
        lambda doc: iter(docling_chunks),
    )

    called = {"count": 0}

    def fake_fallback(content):
        called["count"] += 1
        return [{"text": "from fallback", "meta": {"page_number": 1}}]

    monkeypatch.setattr(
        pdf_fallback_mod,
        "extract_pdf_text_layer",
        fake_fallback,
    )
    result = pipeline.run_ingestion("docling-worked.pdf", b"%PDF-fake")
    assert result == docling_chunks
    assert called["count"] == 0  # fallback never called


def test_run_ingestion_skips_fallback_for_non_pdf(monkeypatch):
    """DOCX / PPTX / HTML don't go through the PDF fallback even when
    Docling returns zero chunks — only PDFs have a text-layer
    alternative."""
    from src.ingestion import pipeline
    from src.ingestion.chunkers import hybrid_chunker
    from src.ingestion.format_detect import Format
    from src.ingestion.parsers import docling_parser
    from src.ingestion.parsers import pdf_fallback as pdf_fallback_mod

    monkeypatch.setattr(
        pipeline, "detect", lambda source, content: Format.DOCX
    )
    monkeypatch.setattr(
        docling_parser, "parse_with_docling", lambda target: object()
    )
    monkeypatch.setattr(
        hybrid_chunker, "chunk_docling_doc", lambda doc: iter([])
    )

    called = {"count": 0}

    def fake_fallback(content):
        called["count"] += 1
        return [{"text": "should not happen", "meta": {}}]

    monkeypatch.setattr(
        pdf_fallback_mod,
        "extract_pdf_text_layer",
        fake_fallback,
    )
    result = pipeline.run_ingestion("report.docx", b"PK-fake")
    assert result == []
    assert called["count"] == 0