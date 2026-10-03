"""Detect input format (extension + magic bytes).

This module is the **single source of truth** for which file types the
system supports. The same list is consumed by:

- ``src/api/routes/formats.py`` -> ``GET /formats`` for the frontend.
- ``src/ingestion/pipeline.py`` to route to the right parser.
- Tests in ``tests/test_format_detect.py`` to assert what is supported.

Adding a new file type means adding ONE row to ``SUPPORTED_FORMATS``
plus an extension-to-Format entry in ``_EXT_MAP`` — the rest of the
system picks it up automatically.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional


class Format(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    HTML = "html"
    MD = "md"
    TXT = "txt"
    UNKNOWN = "unknown"


_EXT_MAP: dict[str, Format] = {
    ".pdf": Format.PDF,
    ".docx": Format.DOCX,
    ".pptx": Format.PPTX,
    ".xlsx": Format.XLSX,  # v1.1.12 — only .xlsx is supported; legacy .xls (BIFF), .xlsm (macro) and .csv are deliberately omitted (no parser wired up)
    ".html": Format.HTML,
    ".htm": Format.HTML,
    ".md": Format.MD,
    ".markdown": Format.MD,
    ".txt": Format.TXT,
}


# ---- Public canonical list ----------------------------------------------
# Each row pairs a ``Format`` with all extensions the system accepts for it
# plus a human-readable label for tooltips / UI copy. Keep in sync with
# ``_EXT_MAP`` above — if you add a new extension there, add it here too.

SUPPORTED_FORMATS: list[dict] = [
    {"format": Format.PDF,  "extensions": [".pdf"], "label_zh": "PDF"},
    {"format": Format.DOCX, "extensions": [".docx"], "label_zh": "Word (.docx)"},
    {"format": Format.PPTX, "extensions": [".pptx"], "label_zh": "PPT (.pptx)"},
    {"format": Format.XLSX, "extensions": [".xlsx"], "label_zh": "Excel (.xlsx)"},
    {"format": Format.HTML, "extensions": [".html", ".htm"], "label_zh": "HTML"},
    {"format": Format.MD,   "extensions": [".md", ".markdown"], "label_zh": "Markdown"},
    {"format": Format.TXT,  "extensions": [".txt"], "label_zh": "纯文本 (.txt)"},
]


def all_extensions() -> list[str]:
    """Flatten list of every accepted extension, preserving ``SUPPORTED_FORMATS`` order."""
    out: list[str] = []
    for row in SUPPORTED_FORMATS:
        out.extend(row["extensions"])
    return out


def extension_to_label() -> dict[str, str]:
    """Map each accepted extension to its Chinese display label.

    Used by ``src/api/routes/formats.py`` to render the same labels in the
    hover tooltip displayed on the upload button.
    """
    out: dict[str, str] = {}
    for row in SUPPORTED_FORMATS:
        for ext in row["extensions"]:
            out[ext] = row["label_zh"]
    return out


def detect(
    source: str,
    content_bytes: Optional[bytes] = None,
) -> Format:
    """Detect format from a file path, optionally sniffing magic bytes."""
    ext = Path(source).suffix.lower()
    if ext in _EXT_MAP:
        return _EXT_MAP[ext]

    if content_bytes:
        head = content_bytes[:8]
        if head.startswith(b"%PDF"):
            return Format.PDF
        if head.startswith(b"PK"):
            # ZIP-based Office formats; let Docling decide between DOCX/XLSX/PPTX
            return Format.DOCX

    return Format.UNKNOWN


__all__ = [
    "Format",
    "SUPPORTED_FORMATS",
    "all_extensions",
    "detect",
    "extension_to_label",
]
