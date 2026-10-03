"""Public endpoint exposing the list of accepted file formats.

Single source of truth is ``src.ingestion.format_detect.SUPPORTED_FORMATS``.
The frontend's upload button reads this to derive its ``accept`` filter
and the Chinese tooltip shown on hover. Returning a JSON array means
adding a row to ``SUPPORTED_FORMATS`` in one place updates both the
backend routing AND the frontend affordances — no hand-syncing lists.
"""
from __future__ import annotations

from fastapi import APIRouter

from src.ingestion.format_detect import (
    SUPPORTED_FORMATS,
    all_extensions,
    extension_to_label,
)


router = APIRouter(tags=["formats"])


@router.get("/formats")
async def list_formats() -> dict:
    """Accepted upload formats.

    Schema:

    - ``extensions``: flat list of every file extension (with leading dot).
      Suitable for an ``<input accept="...">`` filter.
    - ``labels``: ``{ ".pdf": "PDF", ... }`` map for tooltips and copy.
    - ``formats``: structured rows, present so the frontend can render
      extended copy if desired without a second round trip.
    """
    return {
        "extensions": all_extensions(),
        "labels": extension_to_label(),
        "formats": [
            {
                "format": row["format"].value,
                "extensions": row["extensions"],
                "label_zh": row["label_zh"],
            }
            for row in SUPPORTED_FORMATS
        ],
    }


__all__ = ["router"]
