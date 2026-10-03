"""Direct page-render + RapidOCR fallback for image-only PDFs.

When Docling returns zero chunks AND the underlying PDF has no text layer
(everything is a scanned page rendered as an image), Docling's pipeline
becomes the bottleneck: layout analysis runs first, takes ~30 seconds
per page even for a single sheet of homework, and when it returns no
regions the OCR step is skipped entirely. The user just sees "0 chunks"
after 2+ minutes of waiting.

This module bypasses Docling's layout analysis and runs the OCR engine
directly on each rendered page:

    page N: render with pypdfium2 → numpy RGB array
           → RapidOCR engine → list of recognized lines
           → join lines into one chunk with page_number=N

The same RapidOCR models that Docling already pulled in via its
dependencies (``PP-OCRv6_det_small``, ``ch_ppocr_mobile_v2.0_cls_mobile``,
``PP-OCRv6_rec_small``) are used here, so no extra downloads. The
tradeoff vs Docling's full pipeline: we lose layout-aware chunking
(tables, sections) but gain a working extraction for the common case of
scanned homework / invoice / ticket PDFs.

Performance:
- ``scale=2`` (144 DPI) is the sweet spot — fast enough that 10 pages
  parse in ~10-15 s, sharp enough that RapidOCR reads ≥ 12pt text
  accurately. Lower scales hurt recognition; higher scales linearly
  increase render time without measurable accuracy gains.
- ``RapidOCR`` is a module-level singleton so the ~3-5 s model load
  happens once per process, not per page.

Output schema matches ``chunk_docling_doc`` so downstream code
(embedder, ``add_chunks``) needs no special-casing.
"""
from __future__ import annotations

from typing import Optional

from src.core.logging import logger

try:
    import pypdfium2 as pdfium  # type: ignore
except Exception:  # pragma: no cover
    pdfium = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    from rapidocr import RapidOCR  # type: ignore
except Exception:  # pragma: no cover
    RapidOCR = None  # type: ignore


# Module-level singleton. ``RapidOCR()`` loads three ONNX models
# (~3-5 s on first call). Reusing the instance across calls avoids
# paying that cost per page / per upload.
_engine: Optional["RapidOCR"] = None
_engine_lock_imports_failed = False


def _get_engine():
    global _engine, _engine_lock_imports_failed
    if _engine is not None:
        return _engine
    if RapidOCR is None or _engine_lock_imports_failed:
        return None
    try:
        _engine = RapidOCR()
    except Exception as e:
        logger.warning(f"pdf_ocr: failed to init RapidOCR engine ({e})")
        _engine_lock_imports_failed = True
        return None
    return _engine


def _render_page_to_rgb_array(page, scale: float) -> "np.ndarray":
    """Render a pypdfium2 ``PdfPage`` to an HxWx3 uint8 RGB array.

    pypdfium2 returns an RGBA bitmap by default; we drop the alpha
    channel because RapidOCR expects RGB. ``np.ascontiguousarray``
    ensures the array is C-contiguous (RapidOCR's preprocessor
    internally calls ``np.ascontiguousarray`` anyway, but doing it
    here saves a copy and avoids surprising shape mutations on
    contiguous arrays returned by some bitmap_maker paths).
    """
    bitmap = page.render(scale=scale)
    arr = bitmap.to_numpy()  # RGBA, H×W×4
    if arr.shape[2] == 4:
        # Drop the alpha channel — OCR engines don't use it and the
        # extra channel doubles the memory of every input frame.
        arr = arr[:, :, :3]
    return np.ascontiguousarray(arr)


def _ocr_page(img_array: "np.ndarray", engine: "RapidOCR") -> str:
    """Run RapidOCR on a single page image; return the joined text.

    ``RapidOCR.__call__`` returns a ``RapidOCROutput`` with three
    parallel lists (``boxes``, ``txts``, ``scores``) of equal length,
    one entry per detected text region. We sort top-to-bottom / left-
    to-right using the bounding-box coordinates before joining so the
    output reads in document order rather than recognition order
    (RapidOCR doesn't preserve reading order out of the box).
    """
    result = engine(img_array)
    boxes = getattr(result, "boxes", None)
    txts = getattr(result, "txts", None)
    # v1.1.12 fix: ``boxes`` is a ``np.ndarray`` of shape ``(N, 4, 2)``
    # when detection succeeded — the previous ``not boxes`` check
    # raised ``ValueError: The truth value of an array with more than
    # one element is ambiguous`` and the per-page exception handler
    # swallowed it, so the OCR tier returned ``[]`` for every page
    # that had text (i.e. the only pages where it was useful). Use
    # explicit None / length checks instead.
    if boxes is None or txts is None or len(txts) == 0:
        return ""
    if len(boxes) != len(txts):
        # Defensive: should never happen, but a mismatch would mean
        # we're silently dropping regions.
        logger.warning(
            f"pdf_ocr: boxes/txts length mismatch "
            f"({len(boxes)} vs {len(txts)}); joining raw order"
        )
        return "\n".join(str(t) for t in txts if t)
    # Sort by top-left corner so the text reads in document order.
    # ``boxes`` are 4-point quads (Nx4x2 arrays in pixel coords).
    try:
        order = sorted(
            range(len(boxes)),
            key=lambda i: (
                float(boxes[i][0][1]),  # y of top-left
                float(boxes[i][0][0]),  # x of top-left
            ),
        )
    except Exception:
        order = list(range(len(txts)))
    lines = [str(txts[i]) for i in order if txts[i]]
    return "\n".join(lines)


def extract_pdf_via_ocr(
    content_bytes: bytes, scale: float = 2.0, max_pages: int = 200
) -> list[dict]:
    """Render each page and OCR it; return one chunk per non-empty page.

    Each chunk::

        {"text": str,
         "meta": {"page_number": int, "chunk_type": "text"}}

    Returns ``[]`` if any required dependency is missing, if the PDF
    can't be opened, or if every page returned empty OCR output.
    Each page is rendered and OCR'd independently so a single broken
    page doesn't abort the rest.

    ``scale`` defaults to ``2.0`` (144 DPI) — fast enough that a typical
    10-page homework PDF parses in ~10-15 s while still being sharp
    enough for RapidOCR to read ≥ 12pt text. Callers that need higher
    quality (small fonts, dense tables) can pass ``scale=3.0``.

    ``max_pages`` caps the work at 200 pages to prevent accidental
    DoS on a 1000-page thesis PDF. Files exceeding the cap still
    process the first 200 pages and we log a warning so the user
    sees the truncation in logs (not silently missing the rest).
    """
    if pdfium is None or np is None or RapidOCR is None:
        return []
    engine = _get_engine()
    if engine is None:
        return []
    pdf = None
    try:
        try:
            pdf = pdfium.PdfDocument(content_bytes)
        except Exception as e:
            logger.warning(f"pdf_ocr: failed to open PDF ({e})")
            return []
        n = len(pdf)
        if n > max_pages:
            logger.warning(
                f"pdf_ocr: PDF has {n} pages; truncating OCR to "
                f"{max_pages} pages. Split the document or raise "
                f"max_pages if you need more."
            )
            n = max_pages
        out: list[dict] = []
        for i in range(n):
            try:
                page = pdf[i]
                try:
                    img = _render_page_to_rgb_array(page, scale)
                finally:
                    try:
                        page.close()
                    except Exception:
                        pass
                text = _ocr_page(img, engine).strip()
                if text:
                    out.append(
                        {
                            "text": text,
                            "meta": {
                                "page_number": i + 1,
                                "chunk_type": "text",
                            },
                        }
                    )
            except Exception as e:
                logger.warning(
                    f"pdf_ocr: page {i} failed ({type(e).__name__}: {e})"
                )
                continue
        return out
    finally:
        if pdf is not None:
            try:
                pdf.close()
            except Exception:
                pass


__all__ = ["extract_pdf_via_ocr"]