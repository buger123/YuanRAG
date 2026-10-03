"""PDF text-layer fallback for when Docling's OCR pipeline returns nothing.

Some PDFs (often computer-generated documents like exam tickets / invoices
/ PDF/A exports) have a thin text layer that Docling's OCR pipeline can
miss because it first rasterizes the page and runs OCR on the image
instead of extracting the underlying text. ``pypdfium2`` reads the text
layer directly — much faster than OCR and reliable whenever a layer
exists.

This module is invoked from ``src.ingestion.pipeline`` ONLY when Docling
returned zero chunks and the input format is PDF. In that case it's a
strictly cheaper attempt: text-layer extraction is ~milliseconds per page
vs seconds-per-page for OCR.

Returns ``[]`` if ``pypdfium2`` isn't installed, if the PDF has no text
layer (genuinely image-only), or if any error occurs. Callers treat an
empty list the same as any other parser returning nothing.
"""
from __future__ import annotations

from src.core.logging import logger

# ``pypdfium2`` is already pulled in as a transitive dependency of
# ``docling`` (which uses it for image rendering), so it's available
# without an explicit ``pip install``. The ``try/except`` guard makes
# the import optional so the project still imports cleanly in
# environments where docling isn't installed (e.g. some unit-test
# contexts).
try:
    import pypdfium2 as pdfium  # type: ignore
except Exception:  # pragma: no cover
    pdfium = None  # type: ignore


def extract_pdf_text_layer(content_bytes: bytes) -> list[dict]:
    """Return one record per non-empty page of text-layer content.

    Each record has the same shape as ``chunk_docling_doc`` output so the
    downstream embedder / storage layer doesn't need a special case::

        {"text": str,
         "meta": {"page_number": int, "chunk_type": "text"}}

    Pages with empty text layers are skipped (genuine image-only pages,
    which the embedding pipeline can't help with anyway). The whole
    operation is wrapped in try/except so a corrupt page doesn't abort
    the upload.
    """
    if pdfium is None:
        return []
    out: list[dict] = []
    pdf = None
    try:
        try:
            pdf = pdfium.PdfDocument(content_bytes)
        except Exception as e:
            logger.warning(f"pdf_fallback: failed to open PDF ({e})")
            return []
        n = len(pdf)
        for i in range(n):
            try:
                page = pdf[i]
                # ``get_textpage`` is the documented API for text-layer
                # access. The returned page object must be closed
                # explicitly to release native handles (memory leaks
                # otherwise, especially with hundreds of pages).
                tp = page.get_textpage()
                try:
                    text = tp.get_text_range() or ""
                finally:
                    try:
                        tp.close()
                    except Exception:
                        pass
                text = text.strip()
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
                # One bad page shouldn't tank the whole fallback; log
                # and continue.
                logger.warning(f"pdf_fallback: failed to read page {i} ({e})")
                continue
    finally:
        if pdf is not None:
            try:
                pdf.close()
            except Exception:
                pass
    return out


__all__ = ["extract_pdf_text_layer"]