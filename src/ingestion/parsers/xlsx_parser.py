"""Excel parser using openpyxl in read-only mode.

Yields a dict per sheet so the chunker can group rows + headers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

try:
    from openpyxl import load_workbook  # type: ignore
except Exception:  # pragma: no cover
    load_workbook = None  # type: ignore

from src.core.logging import logger


def parse_xlsx(path_or_bytes) -> Iterator[dict]:
    """Yield ``{sheet_name, headers, rows}`` for every sheet in an .xlsx.

    ``path_or_bytes`` may be a ``Path``, ``str``, or an in-memory bytes
    object (in which case it is wrapped via ``BytesIO``).

    v1.1.12: emits ``logger.warning`` when a sheet yields zero rows
    after the header (header-only or genuinely-empty sheets). The
    parser still yields the dict so the chunker can decide what to
    do — the warning gives operators a paper trail when a user
    reports "未提取到文本" on a workbook that has data they can
    obviously see in Excel.
    """
    if load_workbook is None:
        raise RuntimeError("openpyxl not installed. Run: pip install openpyxl")

    from io import BytesIO

    if isinstance(path_or_bytes, (str, Path)):
        wb = load_workbook(filename=str(path_or_bytes), read_only=True, data_only=True)
    else:
        wb = load_workbook(filename=BytesIO(path_or_bytes), read_only=True, data_only=True)

    try:
        for sheet in wb.worksheets:
            # v1.1.12b — In ``read_only=True`` mode openpyxl trusts the
            # worksheet XML's ``<dimension ref="..."/>`` element as the
            # authoritative extent and ``iter_rows`` stops there. Some
            # writers (WPS Office in particular, very common in China;
            # also Apache POI / Google Sheets exports / ERP report
            # generators) emit a stale or placeholder dimension
            # (``A1:A1``, ``A1:C1``) on sheets that actually have
            # hundreds of rows, causing the chunker to see ``rows=[]``
            # and the upload to land ``status="empty"``. Calling
            # ``reset_dimensions()`` discards the cached dimension and
            # falls back to scanning actual cells, which is what we
            # want. Memory profile is unchanged — ``read_only=True``
            # still streams row-by-row.
            sheet.reset_dimensions()
            rows_iter = sheet.iter_rows(values_only=True)
            try:
                first = next(rows_iter)
            except StopIteration:
                # Genuinely empty sheet (no header, no rows). Skip it
                # entirely — no point emitting an empty chunk.
                logger.warning(
                    f"xlsx_parser: sheet '{sheet.title}' is empty (0 rows)"
                )
                continue
            headers = [str(c) if c is not None else "" for c in first]
            rows = []
            for row in rows_iter:
                rows.append([c for c in row])
            # v1.1.12 — also warn on "all-None" rows (formulas with
            # no cached value under data_only=True). These are the
            # common cause of user reports "I can see data in Excel
            # but YuanRAG says 未提取到文本" — Excel's displayed
            # computed value never makes it into the file's
            # cached-value table when the file is saved without
            # recalculating.
            non_empty_rows = [r for r in rows if any(c is not None for c in r)]
            if not non_empty_rows:
                logger.warning(
                    f"xlsx_parser: sheet '{sheet.title}' has only a header "
                    f"({len(headers)} columns) and no data rows "
                    f"(rows may be formulas without cached values)"
                )
            yield {
                "sheet_name": sheet.title,
                "headers": headers,
                "rows": rows,
            }
    finally:
        wb.close()


__all__ = ["parse_xlsx"]
