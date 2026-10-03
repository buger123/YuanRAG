"""Tests for ``src.ingestion.parsers.pdf_ocr`` and the 3-tier PDF fallback.

The OCR module bypasses Docling's layout analysis and runs RapidOCR
directly on each rendered page. Tests focus on the behavior that
matters for correctness:

- Dependency-guard: returns ``[]`` cleanly when pypdfium2 / numpy /
  RapidOCR are unavailable.
- Per-page independence: one broken page doesn't abort the rest.
- Reading order: text is sorted top-to-bottom, left-to-right, not in
  recognition order.
- Pipeline integration: when Docling + pypdfium2 text layer both
  return nothing, the OCR fallback fires; when Docling already
  succeeded, OCR is skipped.
- Scale clamping: rendering scales pass through without crashing.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.parsers import pdf_ocr


class _FakeRapidOCROutput:
    """v1.1.12: ``boxes`` must be a real ``np.ndarray`` with shape
    ``(N, 4, 2)`` (or ``None``) — using a plain Python list would hide
    the ``ValueError: truth value ambiguous`` bug the test is meant
    to guard against. ``txts`` stays a list of str per RapidOCR's
    documented output shape."""

    def __init__(self, boxes, txts):
        self.boxes = boxes
        self.txts = txts


def _box_array(np, coords):
    """Build a real ``np.ndarray`` of shape ``(N, 4, 2)`` from a
    list-of-quad-coords, matching RapidOCR's actual output shape.
    Each entry of ``coords`` is a 4-point quad ``[[x,y], [x,y], [x,y], [x,y]]``.
    """
    return np.array(coords, dtype=np.float32)


class _FakeEngine:
    """Stand-in for ``RapidOCR`` that records calls and returns scripted
    results. We control the per-page output via a queue so the test can
    simulate detection (with text) / blank pages / errors."""

    def __init__(self, scripted_results: list):
        self.scripted = list(scripted_results)
        self.calls: list = []
        self.raise_next: BaseException | None = None

    def __call__(self, img):
        self.calls.append(img)
        if self.raise_next is not None:
            exc = self.raise_next
            self.raise_next = None
            raise exc
        if not self.scripted:
            return _FakeRapidOCROutput(None, None)
        result = self.scripted.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class _FakePage:
    def __init__(self, render_array):
        self._arr = render_array
        self.closed = False

    def render(self, scale=1, **kw):
        return self._arr

    def close(self):
        self.closed = True


class _FakeBitmap:
    def __init__(self, arr):
        self._arr = arr

    def to_numpy(self):
        return self._arr


@pytest.fixture
def fake_numpy():
    """Lazy import guard — numpy is available in the test env but the
    module under test imports it lazily, so we patch the module-level
    binding for tests that don't actually render real PDFs."""
    pytest.importorskip("numpy")
    import numpy as np

    return np


def test_returns_empty_when_pypdfium2_missing(monkeypatch):
    monkeypatch.setattr(pdf_ocr, "pdfium", None)
    assert pdf_ocr.extract_pdf_via_ocr(b"%PDF-fake") == []


def test_returns_empty_when_numpy_missing(monkeypatch):
    monkeypatch.setattr(pdf_ocr, "pdfium", object())  # pretend present
    monkeypatch.setattr(pdf_ocr, "np", None)
    assert pdf_ocr.extract_pdf_via_ocr(b"%PDF-fake") == []


def test_returns_empty_when_rapidocr_missing(monkeypatch):
    monkeypatch.setattr(pdf_ocr, "pdfium", object())
    monkeypatch.setattr(pdf_ocr, "np", object())
    monkeypatch.setattr(pdf_ocr, "RapidOCR", None)
    monkeypatch.setattr(pdf_ocr, "_engine", None)
    monkeypatch.setattr(pdf_ocr, "_engine_lock_imports_failed", True)
    assert pdf_ocr.extract_pdf_via_ocr(b"%PDF-fake") == []


def test_engine_init_failure_marks_engine_disabled(monkeypatch):
    """If RapidOCR() raises during init, subsequent calls return []
    without retrying (the failure is sticky)."""

    def boom():
        raise RuntimeError("model not found")

    monkeypatch.setattr(pdf_ocr, "RapidOCR", boom)
    monkeypatch.setattr(pdf_ocr, "_engine", None)
    monkeypatch.setattr(pdf_ocr, "_engine_lock_imports_failed", False)
    # Need pdfium + np set to non-None so the early-return doesn't fire
    monkeypatch.setattr(pdf_ocr, "pdfium", object())
    monkeypatch.setattr(pdf_ocr, "np", object())
    assert pdf_ocr.extract_pdf_via_ocr(b"%PDF-fake") == []
    assert pdf_ocr._engine_lock_imports_failed is True


def test_engine_singleton_reused_across_calls(monkeypatch, fake_numpy):
    """Two calls should reuse the same RapidOCR instance — model load
    is ~3-5 s, so per-call instantiation would dominate latency."""
    instances: list = []
    real_engine_cls_calls = {"n": 0}

    class CountingRapidOCR:
        def __init__(self):
            real_engine_cls_calls["n"] += 1
            instances.append(self)

        def __call__(self, img):
            return _FakeRapidOCROutput(None, None)

    monkeypatch.setattr(pdf_ocr, "RapidOCR", CountingRapidOCR)
    monkeypatch.setattr(pdf_ocr, "_engine", None)
    monkeypatch.setattr(pdf_ocr, "_engine_lock_imports_failed", False)

    arr = fake_numpy.full((10, 10, 3), 255, dtype=fake_numpy.uint8)
    fake_pdf = MagicMock()
    fake_pdf.__len__ = lambda self: 2
    fake_pdf.__getitem__ = lambda self, i: _FakePage(_FakeBitmap(arr))
    fake_pdfium = MagicMock()
    fake_pdfium.PdfDocument = lambda _: fake_pdf
    monkeypatch.setattr(pdf_ocr, "pdfium", fake_pdfium)
    monkeypatch.setattr(pdf_ocr, "np", fake_numpy)

    pdf_ocr.extract_pdf_via_ocr(b"%PDF-fake")
    pdf_ocr.extract_pdf_via_ocr(b"%PDF-fake")
    assert real_engine_cls_calls["n"] == 1


def test_renders_page_drops_alpha_channel(fake_numpy):
    """PdfBitmap returns RGBA; OCR only needs RGB."""
    rgba = fake_numpy.zeros((5, 5, 4), dtype=fake_numpy.uint8)
    rgba[..., 3] = 200  # alpha varies
    page = _FakePage(_FakeBitmap(rgba))
    out = pdf_ocr._render_page_to_rgb_array(page, scale=1.0)
    assert out.shape == (5, 5, 3)
    # Alpha values from input must NOT appear in the output channel
    # count — they were dropped, not assigned to a colour channel.
    assert out.shape[2] == 3


def test_ocr_page_sorts_top_to_bottom(fake_numpy):
    """Lines on the same page come back in reading order, not in
    recognition order."""
    boxes = _box_array(
        fake_numpy,
        [
            [[0, 100], [10, 100], [10, 110], [0, 110]],  # y=100 — bottom
            [[0, 0], [10, 0], [10, 10], [0, 10]],  # y=0 — top
            [[0, 50], [10, 50], [10, 60], [0, 60]],  # y=50 — middle
        ],
    )
    txts = ["BOTTOM", "TOP", "MIDDLE"]
    engine = _FakeEngine([_FakeRapidOCROutput(boxes, txts)])
    text = pdf_ocr._ocr_page(fake_numpy.zeros((10, 10, 3), dtype=fake_numpy.uint8), engine)
    assert text == "TOP\nMIDDLE\nBOTTOM"


def test_ocr_page_returns_empty_when_no_detections(fake_numpy):
    engine = _FakeEngine([_FakeRapidOCROutput(None, None)])
    text = pdf_ocr._ocr_page(fake_numpy.zeros((10, 10, 3), dtype=fake_numpy.uint8), engine)
    assert text == ""


def test_ocr_page_with_ndarray_boxes_does_not_raise(fake_numpy):
    """v1.1.12 regression: a successful OCR result has ``boxes`` as
    a real ``np.ndarray`` with shape ``(N, 4, 2)``. The pre-fix
    ``if not boxes`` check raised ``ValueError: truth value ambiguous``
    for any non-empty array — this test would have failed (or
    silently dropped the page) before the fix."""
    boxes = _box_array(
        fake_numpy,
        [
            [[0, 0], [10, 0], [10, 10], [0, 10]],
            [[0, 20], [10, 20], [10, 30], [0, 30]],
        ],
    )
    txts = ["line A", "line B"]
    engine = _FakeEngine([_FakeRapidOCROutput(boxes, txts)])
    text = pdf_ocr._ocr_page(
        fake_numpy.zeros((10, 10, 3), dtype=fake_numpy.uint8), engine
    )
    # Reading order: y=0 first, y=20 second.
    assert text == "line A\nline B"


def test_ocr_page_with_zero_length_ndarray_boxes(fake_numpy):
    """Edge: ``boxes`` is a real ndarray but shape ``(0, 4, 2)`` (the
    OCR engine reported zero detected regions but didn't return
    ``None``). Should return empty string, not raise."""
    boxes = fake_numpy.empty((0, 4, 2), dtype=fake_numpy.float32)
    engine = _FakeEngine([_FakeRapidOCROutput(boxes, [])])
    text = pdf_ocr._ocr_page(
        fake_numpy.zeros((10, 10, 3), dtype=fake_numpy.uint8), engine
    )
    assert text == ""


def test_extract_pdf_via_ocr_happy_path(monkeypatch, fake_numpy):
    """Three pages: 1st has text, 2nd is blank, 3rd has text. Result
    should have two records, in page order, with sequential page_numbers."""
    boxes1 = _box_array(
        fake_numpy, [[[0, 0], [10, 0], [10, 10], [0, 10]]]
    )
    txts1 = ["Hello"]
    boxes3 = _box_array(
        fake_numpy, [[[0, 0], [10, 0], [10, 10], [0, 10]]]
    )
    txts3 = ["World"]
    engine = _FakeEngine(
        [
            _FakeRapidOCROutput(boxes1, txts1),  # page 1
            _FakeRapidOCROutput(None, None),  # page 2 — blank, skipped
            _FakeRapidOCROutput(boxes3, txts3),  # page 3
        ]
    )
    monkeypatch.setattr(pdf_ocr, "_get_engine", lambda: engine)
    monkeypatch.setattr(pdf_ocr, "RapidOCR", object)  # present

    arr = fake_numpy.full((10, 10, 3), 255, dtype=fake_numpy.uint8)
    page_objs = [
        _FakePage(_FakeBitmap(arr)),
        _FakePage(_FakeBitmap(arr)),
        _FakePage(_FakeBitmap(arr)),
    ]
    fake_pdf = MagicMock()
    fake_pdf.__len__ = lambda self: 3
    fake_pdf.__getitem__ = lambda self, i: page_objs[i]
    fake_pdfium = MagicMock()
    fake_pdfium.PdfDocument = lambda _: fake_pdf
    monkeypatch.setattr(pdf_ocr, "pdfium", fake_pdfium)
    monkeypatch.setattr(pdf_ocr, "np", fake_numpy)

    result = pdf_ocr.extract_pdf_via_ocr(b"%PDF-fake")
    assert len(result) == 2
    assert result[0]["text"] == "Hello"
    assert result[0]["meta"]["page_number"] == 1
    assert result[0]["meta"]["chunk_type"] == "text"
    assert result[1]["text"] == "World"
    assert result[1]["meta"]["page_number"] == 3
    # All pages should have been closed to release native handles.
    assert all(p.closed for p in page_objs)


def test_extract_pdf_via_ocr_continues_on_per_page_error(
    monkeypatch, fake_numpy
):
    """One page raising during render/OCR must not abort subsequent pages."""
    boxes = _box_array(
        fake_numpy, [[[0, 0], [10, 0], [10, 10], [0, 10]]]
    )
    engine = _FakeEngine(
        [
            RuntimeError("page render failed"),
            _FakeRapidOCROutput(boxes, ["survived"]),
        ]
    )
    monkeypatch.setattr(pdf_ocr, "_get_engine", lambda: engine)
    monkeypatch.setattr(pdf_ocr, "RapidOCR", object)

    arr = fake_numpy.full((10, 10, 3), 255, dtype=fake_numpy.uint8)
    page_objs = [_FakePage(_FakeBitmap(arr)), _FakePage(_FakeBitmap(arr))]
    fake_pdf = MagicMock()
    fake_pdf.__len__ = lambda self: 2
    fake_pdf.__getitem__ = lambda self, i: page_objs[i]
    fake_pdfium = MagicMock()
    fake_pdfium.PdfDocument = lambda _: fake_pdf
    monkeypatch.setattr(pdf_ocr, "pdfium", fake_pdfium)
    monkeypatch.setattr(pdf_ocr, "np", fake_numpy)

    result = pdf_ocr.extract_pdf_via_ocr(b"%PDF-fake")
    assert len(result) == 1
    assert result[0]["text"] == "survived"
    assert result[0]["meta"]["page_number"] == 2


def test_extract_pdf_via_ocr_caps_max_pages(monkeypatch, fake_numpy):
    """A 5-page PDF with ``max_pages=2`` should only OCR the first 2."""
    boxes = _box_array(
        fake_numpy, [[[0, 0], [10, 0], [10, 10], [0, 10]]]
    )
    # Only need 2 results queued; the 3rd should never be reached.
    engine = _FakeEngine(
        [
            _FakeRapidOCROutput(boxes, ["p1"]),
            _FakeRapidOCROutput(boxes, ["p2"]),
        ]
    )
    monkeypatch.setattr(pdf_ocr, "_get_engine", lambda: engine)
    monkeypatch.setattr(pdf_ocr, "RapidOCR", object)

    arr = fake_numpy.full((10, 10, 3), 255, dtype=fake_numpy.uint8)
    fake_pdf = MagicMock()
    fake_pdf.__len__ = lambda self: 5
    fake_pdf.__getitem__ = lambda self, i: _FakePage(_FakeBitmap(arr))
    fake_pdfium = MagicMock()
    fake_pdfium.PdfDocument = lambda _: fake_pdf
    monkeypatch.setattr(pdf_ocr, "pdfium", fake_pdfium)
    monkeypatch.setattr(pdf_ocr, "np", fake_numpy)

    result = pdf_ocr.extract_pdf_via_ocr(b"%PDF-fake", max_pages=2)
    assert len(result) == 2
    assert len(engine.calls) == 2  # never called for pages 3-5


def test_extract_pdf_via_ocr_returns_empty_when_pdf_unreadable(monkeypatch):
    monkeypatch.setattr(pdf_ocr, "_get_engine", lambda: MagicMock())
    monkeypatch.setattr(pdf_ocr, "RapidOCR", object)

    class BoomPdf:
        def __init__(self, *a, **k):
            raise RuntimeError("not a real PDF")

        def __len__(self):
            return 0

        def __getitem__(self, i):
            raise IndexError

    monkeypatch.setattr(pdf_ocr, "pdfium", MagicMock(PdfDocument=BoomPdf))
    monkeypatch.setattr(pdf_ocr, "np", object())

    assert pdf_ocr.extract_pdf_via_ocr(b"garbage") == []


def test_pipeline_runs_three_tiers_in_order(monkeypatch):
    """End-to-end: Docling 0 chunks → pypdfium2 text layer 0 → OCR
    succeeds. The pipeline should return the OCR records."""
    from src.ingestion import pipeline
    from src.ingestion.chunkers import hybrid_chunker
    from src.ingestion.format_detect import Format
    from src.ingestion.parsers import docling_parser
    from src.ingestion.parsers import pdf_fallback
    from src.ingestion.parsers import pdf_ocr

    monkeypatch.setattr(pipeline, "detect", lambda s, c: Format.PDF)
    monkeypatch.setattr(
        docling_parser, "parse_with_docling", lambda target: object()
    )
    monkeypatch.setattr(
        hybrid_chunker, "chunk_docling_doc", lambda doc: iter([])
    )
    monkeypatch.setattr(
        pdf_fallback, "extract_pdf_text_layer", lambda content: []
    )
    ocr_records = [
        {"text": "scanned page 1", "meta": {"page_number": 1}},
        {"text": "scanned page 2", "meta": {"page_number": 2}},
    ]
    monkeypatch.setattr(
        pdf_ocr, "extract_pdf_via_ocr", lambda content, **kw: ocr_records
    )
    result = pipeline.run_ingestion("homework.pdf", b"%PDF-fake")
    assert result == ocr_records


def test_pipeline_skips_ocr_when_docling_succeeds(monkeypatch):
    """If Docling returned chunks, OCR fallback must NOT be invoked
    (it would waste seconds re-rendering the PDF)."""
    from src.ingestion import pipeline
    from src.ingestion.chunkers import hybrid_chunker
    from src.ingestion.format_detect import Format
    from src.ingestion.parsers import docling_parser
    from src.ingestion.parsers import pdf_fallback
    from src.ingestion.parsers import pdf_ocr

    monkeypatch.setattr(pipeline, "detect", lambda s, c: Format.PDF)
    monkeypatch.setattr(
        docling_parser, "parse_with_docling", lambda target: object()
    )
    docling_chunks = [
        {"text": "from docling", "meta": {"page_number": 1}},
    ]
    monkeypatch.setattr(
        hybrid_chunker, "chunk_docling_doc", lambda doc: iter(docling_chunks)
    )

    calls = {"text_layer": 0, "ocr": 0}
    monkeypatch.setattr(
        pdf_fallback,
        "extract_pdf_text_layer",
        lambda c: (calls.__setitem__("text_layer", calls["text_layer"] + 1) or []),
    )
    monkeypatch.setattr(
        pdf_ocr,
        "extract_pdf_via_ocr",
        lambda c, **kw: (calls.__setitem__("ocr", calls["ocr"] + 1) or []),
    )
    result = pipeline.run_ingestion("digital.pdf", b"%PDF-fake")
    assert result == docling_chunks
    assert calls["text_layer"] == 0
    assert calls["ocr"] == 0


def test_pipeline_skips_ocr_when_text_layer_succeeds(monkeypatch):
    """If pypdfium2 text layer recovered content, OCR must NOT be
    invoked."""
    from src.ingestion import pipeline
    from src.ingestion.chunkers import hybrid_chunker
    from src.ingestion.format_detect import Format
    from src.ingestion.parsers import docling_parser
    from src.ingestion.parsers import pdf_fallback
    from src.ingestion.parsers import pdf_ocr

    monkeypatch.setattr(pipeline, "detect", lambda s, c: Format.PDF)
    monkeypatch.setattr(
        docling_parser, "parse_with_docling", lambda target: object()
    )
    monkeypatch.setattr(
        hybrid_chunker, "chunk_docling_doc", lambda doc: iter([])
    )
    text_layer_records = [
        {"text": "from text layer", "meta": {"page_number": 1}},
    ]
    monkeypatch.setattr(
        pdf_fallback,
        "extract_pdf_text_layer",
        lambda c: text_layer_records,
    )
    calls = {"ocr": 0}
    monkeypatch.setattr(
        pdf_ocr,
        "extract_pdf_via_ocr",
        lambda c, **kw: (calls.__setitem__("ocr", calls["ocr"] + 1) or []),
    )
    result = pipeline.run_ingestion("mixed.pdf", b"%PDF-fake")
    assert result == text_layer_records
    assert calls["ocr"] == 0


def test_pipeline_skips_ocr_for_non_pdf(monkeypatch):
    """For DOCX/PPTX/HTML, neither text-layer fallback nor OCR
    fallback should be attempted — only Docling is wired up."""
    from src.ingestion import pipeline
    from src.ingestion.chunkers import hybrid_chunker
    from src.ingestion.format_detect import Format
    from src.ingestion.parsers import docling_parser
    from src.ingestion.parsers import pdf_fallback
    from src.ingestion.parsers import pdf_ocr

    monkeypatch.setattr(pipeline, "detect", lambda s, c: Format.DOCX)
    monkeypatch.setattr(
        docling_parser, "parse_with_docling", lambda target: object()
    )
    monkeypatch.setattr(
        hybrid_chunker, "chunk_docling_doc", lambda doc: iter([])
    )

    calls = {"text_layer": 0, "ocr": 0}
    monkeypatch.setattr(
        pdf_fallback,
        "extract_pdf_text_layer",
        lambda c: (calls.__setitem__("text_layer", calls["text_layer"] + 1) or []),
    )
    monkeypatch.setattr(
        pdf_ocr,
        "extract_pdf_via_ocr",
        lambda c, **kw: (calls.__setitem__("ocr", calls["ocr"] + 1) or []),
    )
    result = pipeline.run_ingestion("report.docx", b"PK-fake")
    assert result == []
    assert calls["text_layer"] == 0
    assert calls["ocr"] == 0