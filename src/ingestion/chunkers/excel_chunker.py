"""Excel row-grouped chunker with header context injection.

Each chunk looks like:

    Sheet: <sheet_name>
    Columns: <col1> | <col2> | <col3>
    Rows <a>-<b>:
    <row a> |
    <row b> |
    ...
"""
from __future__ import annotations

from typing import Iterable, Iterator

from config.constants import CHUNKING


def chunk_xlsx_sheets(sheets: Iterable[dict]) -> Iterator[dict]:
    """Chunk Excel sheets emitted by ``xlsx_parser.parse_xlsx``."""
    rows_per_chunk = CHUNKING["excel_rows_per_chunk"]
    for sheet in sheets:
        sheet_name = sheet["sheet_name"]
        headers = sheet["headers"]
        rows = sheet["rows"]
        header_line = " | ".join(headers)
        # v1.1.12b — A sheet whose first row IS the only data row (no
        # separate header row) used to yield zero chunks because the
        # parser consumes row 1 as headers and ``rows`` then has length
        # 0. Header text is real content, though — a single-row sheet
        # ("this is a one-page form" / "summary totals" / "key-value
        # pairs") still describes the data — so emit a header-only
        # chunk when ``rows`` is empty rather than silently dropping it.
        if not rows:
            yield {
                "text": (
                    f"Sheet: {sheet_name}\n"
                    f"Columns: {header_line}"
                ),
                "meta": {
                    "chunk_type": "row_group",
                    "sheet": sheet_name,
                    "headers": headers,
                    "row_range": [0, 0],
                },
            }
            continue
        for i in range(0, len(rows), rows_per_chunk):
            group = rows[i : i + rows_per_chunk]
            body = "\n".join(_row_to_str(r) for r in group)
            text = (
                f"Sheet: {sheet_name}\n"
                f"Columns: {header_line}\n"
                f"Rows {i + 1}-{i + len(group)}:\n{body}"
            )
            yield {
                "text": text,
                "meta": {
                    "chunk_type": "row_group",
                    "sheet": sheet_name,
                    "headers": headers,
                    "row_range": [i + 1, i + len(group)],
                },
            }


def _row_to_str(row) -> str:
    return " | ".join("" if c is None else str(c) for c in row)


__all__ = ["chunk_xlsx_sheets"]
