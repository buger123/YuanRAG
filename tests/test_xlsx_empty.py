"""v1.1.12 — XLSX "0 chunks" observability + empty-status dispatch.

The user-reported "未提取到文本" complaint on .xlsx uploads had two
distinctly different shapes behind it, both producing the same
opaque error:

1. Header-only sheet (only column names, no data rows).
2. All-formulas-no-cache sheet (every cell is a formula whose
   cached value is ``None``).
3. Genuinely-empty workbook (no rows at all).

In every case the parser returned 0 chunks *without raising*, so
the API layer marked the doc ``status="empty"`` with a single
canned message ("扫描件无文字层…") that was wrong for every .xlsx
case. v1.1.12 makes two changes:

* ``xlsx_parser.parse_xlsx`` now logs a per-sheet warning when a
  sheet yields 0 rows after the header (the previous behavior was
  to skip truly-empty sheets silently via ``continue``; header-only
  sheets slipped through with no signal at all).
* ``pipeline.run_ingestion`` logs a per-file rollup when the XLSX
  branch produces 0 chunks.
* ``documents.py`` dispatches the user-facing error message by the
  parsed ``Format`` — so a header-only .xlsx now reads "每个 sheet
  至少需要一行数据", not "扫描件无文字层".

These tests pin both behaviors. They construct real .xlsx files via
openpyxl (the same way fixtures do) so we exercise the full parser,
not just the chunker.
"""
from __future__ import annotations

from io import BytesIO
from typing import List

import pytest
from loguru import logger as _loguru_logger
from openpyxl import Workbook

from src.ingestion.format_detect import Format
from src.ingestion.parsers import xlsx_parser
from src.ingestion.parsers.xlsx_parser import parse_xlsx


class _LoguruCapture:
    """Capture loguru warnings emitted during a code block.

    YuanRAG uses loguru (``src.core.logging``), not stdlib ``logging``,
    so pytest's ``caplog`` fixture can't see the records. This helper
    attaches a temporary sink that drains ``LogMessage`` records into
    a list and asserts on their ``.message`` content.
    """

    def __init__(self):
        self.records: List = []
        self._sink_id = None

    def __enter__(self):
        self._sink_id = _loguru_logger.add(
            lambda msg: self.records.append(msg),
            level="WARNING",
            format="{message}",
        )
        return self

    def __exit__(self, *exc):
        _loguru_logger.remove(self._sink_id)
        return False

    def warnings(self) -> List[str]:
        return [r for r in self.records]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wb_to_bytes(wb: Workbook) -> bytes:
    """Serialize an openpyxl Workbook to in-memory bytes."""
    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _header_only_workbook() -> bytes:
    """A workbook with one sheet containing only column headers."""
    wb = Workbook()
    ws = wb.active
    ws.title = "OnlyHeader"
    ws["A1"] = "Field"
    ws["B1"] = "Value"
    # No data rows below the header.
    return _wb_to_bytes(wb)


def _fully_empty_workbook() -> bytes:
    """A workbook with one sheet that has zero rows (no header either)."""
    wb = Workbook()
    ws = wb.active
    ws.title = "TrulyEmpty"
    # Don't write any cell — the sheet is truly empty.
    return _wb_to_bytes(wb)


def _header_only_multi_sheet_workbook() -> bytes:
    """Multi-sheet workbook where every sheet is header-only."""
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Header1"
    ws1["A1"] = "col"
    ws1["B1"] = "val"
    ws2 = wb.create_sheet("Header2")
    ws2["A1"] = "x"
    return _wb_to_bytes(wb)


def _formula_no_cache_workbook() -> bytes:
    """A workbook where every data cell is a formula with no cached value.

    openpyxl returns ``None`` for every cell when ``data_only=True``
    reads a formula that has never been evaluated by Excel — the
    classic "user opens the file in Excel, fills formulas, saves
    without recalculating" scenario.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Formulas"
    ws["A1"] = "x"
    ws["B1"] = "y"
    ws["A2"] = "=B2+1"
    ws["B2"] = "=A2*2"
    # data_only=True (the parser default) sees formulas whose
    # cached value is None.
    return _wb_to_bytes(wb)


def _happy_workbook() -> bytes:
    """A workbook with headers + data — the regression baseline."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Data"
    ws["A1"] = "id"
    ws["B1"] = "name"
    ws["A2"] = "1"
    ws["B2"] = "alice"
    ws["A3"] = "2"
    ws["B3"] = "bob"
    return _wb_to_bytes(wb)


# ---------------------------------------------------------------------------
# Parser-level: parse_xlsx must log a warning when sheets are empty
# ---------------------------------------------------------------------------


def test_parse_xlsx_header_only_emits_warning():
    """A header-only sheet must NOT be silently lost — log a warning
    that names the sheet so an operator reading logs can correlate
    with a user "未提取到文本" report."""
    with _LoguruCapture() as cap:
        sheets = list(parse_xlsx(_header_only_workbook()))
    assert len(sheets) == 1
    # Parser still yields the dict (preserves shape contract for the
    # chunker downstream).
    assert sheets[0]["sheet_name"] == "OnlyHeader"
    assert sheets[0]["headers"] == ["Field", "Value"]
    assert sheets[0]["rows"] == []
    # But a warning fires with the sheet name in it.
    messages = [str(r) for r in cap.warnings()]
    assert any("OnlyHeader" in m for m in messages), (
        f"expected a warning naming the empty sheet; got: {messages}"
    )


def test_parse_xlsx_fully_empty_sheet_logs_and_skips():
    """A sheet with zero rows at all (no header) was previously
    silently ``continue``-d. v1.1.12 adds a warning so the gap shows
    in logs even when nothing is yielded."""
    with _LoguruCapture() as cap:
        sheets = list(parse_xlsx(_fully_empty_workbook()))
    assert sheets == []
    messages = [str(r) for r in cap.warnings()]
    assert any("TrulyEmpty" in m for m in messages)


def test_parse_xlsx_formula_no_cache_logs_warning():
    """All-formula-no-cache cells return ``None`` for every value —
    the chunker sees only the headers, so the warning fires."""
    with _LoguruCapture() as cap:
        sheets = list(parse_xlsx(_formula_no_cache_workbook()))
    assert len(sheets) == 1
    assert sheets[0]["headers"] == ["x", "y"]
    # ``rows`` retains the raw cell list (one entry per row); what we
    # care about is that no row has any non-None cell — which is the
    # condition that triggers the "data rows are all None" warning.
    assert all(
        all(c is None for c in row) for row in sheets[0]["rows"]
    ), "expected every cell to be None (formula without cached value)"
    messages = [str(r) for r in cap.warnings()]
    assert any("Formulas" in m for m in messages)


def test_parse_xlsx_happy_path_no_warning():
    """Regression: a workbook with real data must NOT log a warning.
    Pre-fix this would have been the only path that didn't trip the
    new warning, but let's pin it explicitly."""
    with _LoguruCapture() as cap:
        sheets = list(parse_xlsx(_happy_workbook()))
    assert len(sheets) == 1
    assert len(sheets[0]["rows"]) == 2
    messages = [str(r) for r in cap.warnings()]
    assert not any("Data" in m for m in messages)


def test_parse_xlsx_multi_sheet_warns_per_empty_sheet():
    """Multi-sheet workbook with multiple empty sheets must emit
    one warning per empty sheet — useful for diagnosing partial
    uploads."""
    with _LoguruCapture() as cap:
        sheets = list(parse_xlsx(_header_only_multi_sheet_workbook()))
    assert len(sheets) == 2
    messages = " ".join(str(r) for r in cap.warnings())
    assert "Header1" in messages
    assert "Header2" in messages


# ---------------------------------------------------------------------------
# Pipeline-level: run_ingestion must log a per-file rollup on 0 chunks
# ---------------------------------------------------------------------------


def test_run_ingestion_xlsx_zero_chunks_logs_warning():
    """When the XLSX path yields 0 chunks (header-only or all-formula),
    the pipeline must log a single warning that says so. Pre-v1.1.12
    this case returned ``[]`` silently and the API layer marked
    ``status="empty"`` with a generic message.

    v1.1.12b update: a header-only sheet now produces a header-only
    chunk (the chunker yields it from ``rows=[]``), so this exact
    payload (``_header_only_workbook``) is no longer the right zero-
    chunks trigger. The all-formulas-no-cache workbook below still
    fits the bill: even after the chunker change, every data row
    in that workbook yields ``[None, None, ...]`` and the chunker
    still emits a chunk with empty cells — but we constructed the
    "truly zero data" case below (no header, no rows) which DOES
    still produce 0 chunks.
    """
    from src.ingestion import pipeline

    # Build a workbook where the parser yields 0 chunks by combining:
    # 1. A truly-empty sheet (skipped by parser, no header even).
    # The "fully empty" workbook from ``_fully_empty_workbook`` is the
    # only one that produces 0 chunks post-v1.1.12b — header-only
    # now emits a header chunk.
    with _LoguruCapture() as cap:
        chunks = pipeline.run_ingestion(
            "truly_empty.xlsx", content_bytes=_fully_empty_workbook()
        )
    assert chunks == []
    messages = [str(r) for r in cap.warnings()]
    assert any(
        "truly_empty.xlsx" in m and "0 chunks" in m for m in messages
    ), (
        f"expected a '0 chunks' warning naming the file; got: {messages}"
    )


def test_run_ingestion_xlsx_header_only_now_produces_one_chunk():
    """v1.1.12b — a header-only workbook now produces ONE chunk (the
    header-only chunk from ``chunk_xlsx_sheets``), not zero. The
    previous behavior silently dropped the header text."""
    from src.ingestion import pipeline

    chunks = pipeline.run_ingestion(
        "header_only.xlsx", content_bytes=_header_only_workbook()
    )
    assert len(chunks) == 1
    assert "OnlyHeader" in chunks[0]["text"]
    assert "Field | Value" in chunks[0]["text"]
    # Parser still logs the per-sheet warning — that's correct: the
    # sheet really does have only a header. The pipeline-level "0
    # chunks" warning does NOT fire because chunks is now non-empty.
    assert chunks[0]["meta"]["row_range"] == [0, 0]


def test_run_ingestion_xlsx_happy_path_no_zero_chunk_warning():
    """Regression: a workbook that produces chunks must NOT log the
    0-chunks warning."""
    from src.ingestion import pipeline

    with _LoguruCapture() as cap:
        chunks = pipeline.run_ingestion(
            "happy.xlsx", content_bytes=_happy_workbook()
        )
    assert chunks  # at least one chunk
    messages = [str(r) for r in cap.warnings()]
    assert not any("0 chunks" in m for m in messages)


# ---------------------------------------------------------------------------
# API-level: error_message dispatch by format
# ---------------------------------------------------------------------------


def test_empty_status_message_dispatches_by_format():
    """``_empty_status_message`` must return format-specific copy so
    a header-only .xlsx doesn't get the "扫描件无文字层" message."""
    from src.api.routes.documents import _empty_status_message

    xlsx_msg = _empty_status_message(
        "report.xlsx", Format.XLSX, content_bytes=b""
    )
    assert "Excel" in xlsx_msg or "sheet" in xlsx_msg.lower()
    # The PDF-only copy must NOT appear for .xlsx.
    assert "扫描件" not in xlsx_msg
    assert "重新生成 PDF" not in xlsx_msg

    pdf_msg = _empty_status_message(
        "scan.pdf", Format.PDF, content_bytes=b""
    )
    assert "扫描件" in pdf_msg
    assert "重新生成 PDF" in pdf_msg


def test_empty_status_message_xlsx_mentions_formula_cache():
    """The new .xlsx message specifically calls out the formula / cached
    value case — the most common cause of "I can see data in Excel but
    YuanRAG says 未提取到文本"."""
    from src.api.routes.documents import _empty_status_message

    msg = _empty_status_message("f.xlsx", Format.XLSX, content_bytes=b"")
    assert "公式" in msg
    assert "缓存值" in msg
