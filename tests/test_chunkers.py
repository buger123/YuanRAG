"""Phase 2 — chunker unit tests (no Docling required)."""
from __future__ import annotations

from src.ingestion.chunkers.excel_chunker import chunk_xlsx_sheets


def _make_sheet(name: str, headers: list[str], rows: list[list]) -> dict:
    return {"sheet_name": name, "headers": headers, "rows": rows}


def test_excel_chunker_single_chunk():
    sheets = [
        _make_sheet(
            "Users",
            ["id", "name"],
            [[1, "Alice"], [2, "Bob"], [3, "Carol"]],
        )
    ]
    chunks = list(chunk_xlsx_sheets(sheets))
    assert len(chunks) == 1
    assert "Sheet: Users" in chunks[0]["text"]
    assert "Columns: id | name" in chunks[0]["text"]
    assert "Alice" in chunks[0]["text"]
    assert chunks[0]["meta"]["chunk_type"] == "row_group"
    assert chunks[0]["meta"]["sheet"] == "Users"
    assert chunks[0]["meta"]["row_range"] == [1, 3]


def test_excel_chunker_multiple_chunks():
    sheets = [
        _make_sheet(
            "Data",
            ["a", "b"],
            [[i, i * 2] for i in range(10)],
        )
    ]
    # rows_per_chunk=4 → 3 chunks (4+4+2)
    chunks = list(chunk_xlsx_sheets(sheets))
    assert len(chunks) == 3
    assert "Rows 1-4" in chunks[0]["text"]
    assert "Rows 5-8" in chunks[1]["text"]
    assert "Rows 9-10" in chunks[2]["text"]
    assert chunks[0]["meta"]["row_range"] == [1, 4]
    assert chunks[2]["meta"]["row_range"] == [9, 10]


def test_excel_chunker_empty_sheet_emits_header_chunk():
    """v1.1.12b — a sheet whose parser produced only a header row
    (e.g. single-row sheet where row 1 is consumed as the header)
    used to silently yield zero chunks. The header text is still real
    content (column names carry semantic meaning), so the chunker
    now emits a header-only chunk with ``row_range=[0, 0]``. The
    previous behavior made sheets like a single-row summary or a
    1-page form effectively invisible to retrieval."""
    sheets = [_make_sheet("Empty", ["a", "b"], [])]
    chunks = list(chunk_xlsx_sheets(sheets))
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Sheet: Empty\nColumns: a | b"
    assert chunks[0]["meta"]["row_range"] == [0, 0]
    assert chunks[0]["meta"]["headers"] == ["a", "b"]
    assert chunks[0]["meta"]["sheet"] == "Empty"


def test_excel_chunker_handles_none_cells():
    sheets = [
        _make_sheet(
            "Mixed",
            ["a", "b"],
            [[1, None], [None, "x"], [None, None]],
        )
    ]
    chunks = list(chunk_xlsx_sheets(sheets))
    assert len(chunks) == 1
    # None cells become empty strings
    assert "1 | " in chunks[0]["text"]
    assert " | x" in chunks[0]["text"]


def test_excel_chunker_multiple_sheets():
    sheets = [
        _make_sheet("S1", ["x"], [[1], [2], [3]]),
        _make_sheet("S2", ["y"], [[10], [20], [30], [40]]),
    ]
    chunks = list(chunk_xlsx_sheets(sheets))
    # S1 → 1 chunk (3 rows ≤ 4); S2 → 1 chunk
    assert len(chunks) == 2
    assert "Sheet: S1" in chunks[0]["text"]
    assert "Sheet: S2" in chunks[1]["text"]


def test_text_parser_handles_utf8():
    from src.ingestion.parsers.text_parser import parse_text

    text = "Hello, 世界! 🌍"
    assert parse_text(text.encode("utf-8")) == text


def test_text_parser_falls_back_on_invalid_utf8():
    from src.ingestion.parsers.text_parser import parse_text

    bad = b"\xff\xfe invalid"
    # Should not raise; falls back to latin-1
    result = parse_text(bad)
    assert result
