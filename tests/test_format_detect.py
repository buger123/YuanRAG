"""Verify the single source of truth for supported formats.

Drives detection off ``SUPPORTED_FORMATS`` rather than re-listing
extensions — so if the canonical list ever changes here, both the
detection table and the test follow in lock-step.
"""
from __future__ import annotations

from src.ingestion.format_detect import (
    SUPPORTED_FORMATS,
    all_extensions,
    detect,
    extension_to_label,
    Format,
)


def _ext_map() -> dict[str, Format]:
    """Flat ``ext → Format`` lookup built from SUPPORTED_FORMATS."""
    out: dict[str, Format] = {}
    for row in SUPPORTED_FORMATS:
        for ext in row["extensions"]:
            out[ext] = row["format"]
    return out


def test_canonical_extensions_present():
    """Every supported format maps to at least one extension."""
    assert SUPPORTED_FORMATS, "SUPPORTED_FORMATS must not be empty"
    for row in SUPPORTED_FORMATS:
        assert row["extensions"], f"format {row['format']} has no extensions"
        assert row["label_zh"], f"format {row['format']} has no label"


def test_all_extensions_helper_covers_every_row():
    flat = set(all_extensions())
    for row in SUPPORTED_FORMATS:
        for ext in row["extensions"]:
            assert ext in flat


def test_extension_to_label_is_total():
    """The labels map must cover every accepted extension."""
    labels = extension_to_label()
    for ext in all_extensions():
        assert ext in labels


def test_extension_detection_from_canonical_list():
    """Every ``ext → format`` mapping from SUPPORTED_FORMATS routes correctly."""
    for ext, fmt in _ext_map().items():
        assert detect(f"name{ext}") == fmt, f"{ext} -> {fmt}"


def test_extension_is_case_insensitive():
    assert detect("X.PDF") == Format.PDF
    assert detect("Foo.HTM") == Format.HTML


def test_magic_bytes_pdf():
    assert detect("blob.bin", b"%PDF-1.7\n") == Format.PDF


def test_magic_bytes_zip_family():
    # ZIP-based Office; ambiguous, defaults to DOCX
    assert detect("blob.bin", b"PK\x03\x04...") == Format.DOCX


def test_unknown_fallback():
    assert detect("mystery.xyz", b"random") == Format.UNKNOWN


def test_empty_bytes_unknown():
    assert detect("anything", b"") == Format.UNKNOWN
