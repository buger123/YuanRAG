"""End-to-end ingestion pipeline.

Dispatch:
    file → format_detect → parser → chunker → list[ChunkRecord dict]

Each emitted dict has shape::

    {
        "text": str,
        "meta": {
            "chunk_type": str,
            "page_number": int | None,
            "section": str | None,
            "sheet": str | None,
            "headers": list[str] | None,
            "row_range": [int, int] | None,
        }
    }

Embedding and LanceDB indexing happen downstream (see ``storage.lancedb_store``).
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional

from src.core.logging import logger
from src.ingestion.cleaning import clean_chunks
from src.ingestion.format_detect import Format, detect
from src.ingestion.parsers.text_parser import parse_text
from src.ingestion.parsers.xlsx_parser import parse_xlsx


def run_ingestion(
    source: str,
    content_bytes: Optional[bytes] = None,
) -> list[dict]:
    """Detect, parse, and chunk an input. Returns ``[chunk_dict, ...]``.

    PDF dispatch uses a three-tier strategy:

    1. **Docling** — layout-aware parsing for digital PDFs (preserves
       tables, sections, reading order). Fast when the PDF has a text
       layer; slow and unreliable when the PDF is image-only.
    2. **pypdfium2 text-layer extraction** — instant catch for PDFs
       where Docling failed but a text layer actually exists (e.g.
       computer-generated exam tickets, invoice PDFs).
    3. **pypdfium2 page render + RapidOCR** — bypasses Docling's slow
       layout analysis and runs OCR directly on each rendered page.
       This is the path that finally works for scanned homework PDFs:
       each page is rendered to a 144 DPI RGB image, fed to the same
       RapidOCR engine Docling uses internally, and the recognized
       lines are joined into one chunk per page.

    v2.0.7 perf-6: a quick pypdfium2 text-layer pre-sniff runs BEFORE
    Docling for PDF inputs. If the majority of the first ~10 pages
    already carry extractable text we skip Docling entirely (its
    layout analysis takes 30-90 s per file even for digital PDFs)
    and go straight to the text-layer extraction. Docling remains
    the dispatcher for anything ambiguous or image-only.
    """
    fmt = detect(source, content_bytes)

    # v2.0.7 perf-6 — digital PDF fast path. The vast majority of
    # text-layer PDFs are produced by office software and don't need
    # Docling's layout analysis. Detect-and-skip saves the 30-90 s
    # Docling warm+analyze on every digital PDF. Tier-2 only ever
    # ran AFTER Docling returned 0 chunks, so the optimization
    # below makes it the primary path instead.
    if (
        fmt == Format.PDF
        and content_bytes is not None
        and _looks_like_digital_pdf(content_bytes)
    ):
        from src.ingestion.parsers.pdf_fallback import (
            extract_pdf_text_layer,
        )

        chunks = extract_pdf_text_layer(content_bytes)
        if chunks:
            logger.info(
                f"Digital PDF detected for {source}; "
                f"skipped Docling, recovered {len(chunks)} page(s) "
                f"via pypdfium2 text layer (v2.0.7 perf-6 fast path)."
            )
            return clean_chunks(chunks)
        # Text-layer sniff said "yes" but extraction came back empty
        # (rare — a PDF can have glyphs without Unicode mapping). Fall
        # through to Docling.

    # HTML fast path: small, plain webpages skip Docling entirely. Docling's
    # layout pipeline is overkill for ``<p>``-only pages — it costs ~10 s to
    # warm the ML models for content that BeautifulSoup can strip in <100 ms.
    # Anything with images / tables / SVG / oversized content still goes
    # through Docling below.
    if fmt == Format.HTML and content_bytes is not None:
        from src.ingestion.parsers.html_parser import (
            is_plain_html,
            parse_html_plain,
        )

        if is_plain_html(content_bytes):
            text = parse_html_plain(content_bytes)
            from src.ingestion.chunkers.hybrid_chunker import chunk_plain_text

            return clean_chunks(chunk_plain_text(text))

    if fmt == Format.PDF or fmt == Format.DOCX or fmt == Format.PPTX or fmt == Format.HTML:
        from docling_core.types.io import DocumentStream
        from src.ingestion.parsers.docling_parser import parse_with_docling
        from src.ingestion.chunkers.hybrid_chunker import chunk_docling_doc

        if content_bytes is None:
            target = source  # str path on disk
        else:
            target = DocumentStream(name=source, stream=BytesIO(content_bytes))
        doc = parse_with_docling(target)
        chunks = list(chunk_docling_doc(doc))
        if chunks:
            return clean_chunks(chunks)

        # Tier 2: PDF-only — pypdfium2 text-layer extraction. Instant
        # for PDFs with embedded text, returns ``[]`` for image-only.
        if fmt == Format.PDF and content_bytes is not None:
            from src.ingestion.parsers.pdf_fallback import (
                extract_pdf_text_layer,
            )

            fallback = extract_pdf_text_layer(content_bytes)
            if fallback:
                logger.info(
                    f"Docling returned 0 chunks for {source}; "
                    f"pypdfium2 text-layer fallback recovered "
                    f"{len(fallback)} page(s)."
                )
                return clean_chunks(fallback)

            # Tier 3: image-only PDF — render each page and OCR it
            # directly. Bypasses Docling's slow layout analysis
            # (which is what makes the standard pipeline hang for
            # scanned homework PDFs). Uses the same RapidOCR models
            # that Docling already pulled in via its dependencies.
            from src.ingestion.parsers.pdf_ocr import (
                extract_pdf_via_ocr,
            )

            ocr_chunks = extract_pdf_via_ocr(content_bytes)
            if ocr_chunks:
                logger.info(
                    f"Docling returned 0 chunks for {source}; "
                    f"pypdfium2 text-layer found nothing; "
                    f"direct page-render OCR recovered "
                    f"{len(ocr_chunks)} page(s)."
                )
                return clean_chunks(ocr_chunks)
            logger.warning(
                f"Docling, pypdfium2 text layer, and direct OCR all "
                f"returned 0 chunks for {source}. The PDF appears to "
                f"be image-only with no recognizable text."
            )
        else:
            # Non-PDF (DOCX/PPTX/HTML) — only Docling is wired up.
            # These formats typically carry text content; if Docling
            # fails the result really is empty.
            logger.warning(
                f"Docling returned 0 chunks for {source}; "
                f"no fallback available for format {fmt.value}."
            )
        return clean_chunks(chunks)

    if fmt == Format.XLSX:
        from src.ingestion.chunkers.excel_chunker import chunk_xlsx_sheets

        sheets = list(parse_xlsx(content_bytes if content_bytes is not None else source))
        chunks = list(chunk_xlsx_sheets(sheets))
        # v1.1.12: log a structured warning when the parser yielded
        # no chunks (most common cause of the user-reported "未提取
        # 到文本" complaint on .xlsx — typically a header-only sheet
        # or one whose data cells are all formulas with no cached
        # value). ``xlsx_parser.parse_xlsx`` already warns per empty
        # sheet; this gives the per-file rollup so an operator can
        # see "xlsx X.xlsx: 1 sheet / 0 chunks" at a glance.
        if not chunks:
            logger.warning(
                f"xlsx '{source}': {len(sheets)} sheet(s) parsed, "
                f"0 chunks produced"
            )
        return clean_chunks(chunks)

    if fmt in (Format.MD, Format.TXT):
        text = parse_text(content_bytes or Path(source).read_bytes())
        from src.ingestion.chunkers.hybrid_chunker import chunk_plain_text

        chunks = list(chunk_plain_text(text))
        return clean_chunks(chunks)

    raise ValueError(f"Unsupported format for: {source}")


# v2.0.7 perf-6 — sniff threshold for "this PDF has a usable text
# layer; skip Docling". Tunable constants sit here so unit tests can
# exercise edge cases (a 1-page borderline PDF, an all-image PDF
# that happens to render glyphs, etc.) without monkey-patching the
# import site.
_DIGITAL_PDF_MIN_PAGE_CHARS = 50
_DIGITAL_PDF_TEXT_PAGE_RATIO = 0.8
_DIGITAL_PDF_SAMPLE_PAGES = 10


def _looks_like_digital_pdf(content_bytes: bytes) -> bool:
    """Return True if the PDF appears to carry an extractable text
    layer on the majority of its sampled pages.

    Sniffs the first ``_DIGITAL_PDF_SAMPLE_PAGES`` pages via
    pypdfium2's ``get_textpage`` API; if >=80% of the sampled pages
    yield >=50 non-whitespace characters the PDF is considered
    "digital" and the dispatcher can safely skip Docling. The
    sampler caps at 10 pages because the smell test is just
    "is there a text layer at all" — a deeper read would defeat the
    speedup.

    pypdfium2 is imported lazily inside the function so the pipeline
    stays import-cheap when the caller is on a non-PDF code path.
    """
    try:
        import pypdfium2 as pdfium  # type: ignore
    except Exception:
        # If pypdfium2 isn't installed, behave conservatively and
        # let the caller fall through to Docling (which has its own
        # pypdfium2 dependency).
        return False

    try:
        doc = pdfium.PdfDocument(content_bytes)
    except Exception as exc:
        # Corrupt / non-PDF bytes — let Docling surface the error
        # rather than masking it with a silent skip.
        logger.debug(f"_looks_like_digital_pdf: PdfDocument open failed: {exc}")
        return False

    n_pages = len(doc)
    sample_n = min(n_pages, _DIGITAL_PDF_SAMPLE_PAGES)
    if sample_n == 0:
        return False

    text_pages = 0
    for i in range(sample_n):
        try:
            page = doc[i]
            text_obj = page.get_textpage()
            try:
                text = text_obj.get_text_range().strip()
            finally:
                text_obj.close()
        except Exception as exc:
            logger.debug(
                f"_looks_like_digital_pdf: text extraction failed on "
                f"page {i}: {exc}"
            )
            continue
        if len(text) >= _DIGITAL_PDF_MIN_PAGE_CHARS:
            text_pages += 1

    return text_pages >= int(sample_n * _DIGITAL_PDF_TEXT_PAGE_RATIO)


__all__ = ["run_ingestion", "_looks_like_digital_pdf"]
