"""Docling-based parser for PDF / DOCX / PPTX / HTML.

Docling (https://docling.ai) returns a ``DoclingDocument`` that preserves
layout, tables, and reading order. We hand the parsed items to
``HybridChunker`` (in chunkers/hybrid_chunker.py).
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterable, Optional, Union

from src.core.logging import logger

# Docling imports are deferred so that unit tests can mock them and so that
# unrelated modules don't pay the import cost.
try:
    from docling.document_converter import DocumentConverter  # type: ignore
except Exception:  # pragma: no cover - import guard
    DocumentConverter = None  # type: ignore


ConverterInput = Union[str, Path, "DocumentStream"]


# Module-level singleton: instantiating ``DocumentConverter()`` warms up
# the ML model pipelines (layout, table-structure, OCR backends) which
# takes 5-15 s on a cold cache. Reusing one instance across every upload
# saves that startup cost on the second-and-onwards documents in a batch.
# Threadsafe via a lock — first-call setup is serialized, subsequent calls
# hit the fast path with no contention.
_converter: Optional["DocumentConverter"] = None
_converter_lock = threading.Lock()


def _get_converter() -> "DocumentConverter":
    """Lazy singleton accessor. First call constructs the converter; every
    subsequent call returns the cached instance."""
    global _converter
    if _converter is not None:
        return _converter
    with _converter_lock:
        if _converter is None:
            _converter = DocumentConverter()
    return _converter


def reset_for_tests() -> None:
    """Drop the cached converter. Tests only."""
    global _converter
    with _converter_lock:
        _converter = None


def parse_with_docling(source: ConverterInput) -> "object":
    """Parse a PDF/DOCX/PPTX/HTML file using Docling.

    Returns the Docling ``DoclingDocument`` for downstream chunking.
    """
    if DocumentConverter is None:
        raise RuntimeError(
            "docling is not installed. Run: pip install docling docling-core docling-parse"
        )

    converter = _get_converter()
    logger.info(f"Docling parsing: {source}")
    result = converter.convert(source)
    if getattr(result, "status", None) and str(result.status).endswith("FAILURE"):
        raise RuntimeError(f"Docling parse failed for {source}: {result.status}")
    return result.document


def iter_docling_items(doc: "object") -> Iterable["object"]:
    """Yield the text-bearing items from a Docling document.

    This wraps the various ways Docling exposes text so chunkers can treat
    them uniformly.
    """
    texts = getattr(doc, "texts", None) or []
    tables = getattr(doc, "tables", None) or []
    for t in texts:
        yield t
    for t in tables:
        yield t


__all__ = ["parse_with_docling", "iter_docling_items", "reset_for_tests"]
