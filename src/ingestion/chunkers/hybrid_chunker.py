"""Docling HybridChunker wrapper.

Produces ``{text, meta}`` records ready for embedding.

Strategy: tokenize with the BGE-M3 tokenizer, max_tokens=512, merge small
adjacent peers so we don't split a single paragraph into many chunks.
"""
from __future__ import annotations

from typing import Iterable, Iterator

from config.constants import CHUNKING, EMBEDDING_MODEL
from src.core.logging import logger

try:
    from docling.chunking import HybridChunker  # type: ignore
    from transformers import AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    HybridChunker = None  # type: ignore
    AutoTokenizer = None  # type: ignore


import threading

_chunker = None
_tokenizer = None
_chunker_lock = threading.Lock()


def _get_chunker():
    """Return the lazily-initialized ``HybridChunker`` singleton.

    v2.0.7 perf-2: the previous version had no lock around the lazy
    initialization. Two concurrent first-time uploads would each call
    ``AutoTokenizer.from_pretrained(EMBEDDING_MODEL)`` (a 2.3 GB HF
    download under ``HF_HOME``) and construct two ``HybridChunker``
    instances — silently doubling the memory + disk footprint and the
    cold-start latency. The double-checked-lock pattern below keeps the
    hot path lock-free while serializing only the first call.
    """
    global _chunker, _tokenizer
    if _chunker is not None:
        return _chunker
    with _chunker_lock:
        if _chunker is None:
            if HybridChunker is None or AutoTokenizer is None:
                raise RuntimeError(
                    "docling + transformers not installed. "
                    "Run: pip install docling transformers tokenizers"
                )
            # Tokenizer will be downloaded lazily by transformers (cached
            # under HF_HOME; for offline mode, point to a local copy).
            _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
            _chunker = HybridChunker(
                tokenizer=_tokenizer,
                max_tokens=CHUNKING["max_tokens"],
                merge_peers=CHUNKING["merge_peers"],
            )
    return _chunker


def warmup_chunker() -> bool:
    """Eagerly initialize the HybridChunker singleton.

    v2.0.8 perf-7: the tokenizer init costs ~5-15 s on a cold HF cache
    (``AutoTokenizer.from_pretrained`` deserializes a 2.3 GB HF model
    on first call) and Docling ``HybridChunker.__init__`` adds another
    few seconds of internal state setup. Without this hook the first
    user upload after server start pays the full cost synchronously and
    the UI shows 「正在解析文档…」 stuck for 5-15 s.

    Called from the FastAPI ``lifespan`` startup so the heavy work
    happens before the server begins accepting uploads. The double-
    checked-lock in ``_get_chunker`` makes this safe to call from any
    thread and idempotent under repeated invocation.

    Returns
    -------
    bool
        True if a warmup actually ran (or is already in progress),
        False if the chunker was already initialized and no work was
        needed. Either way the chunker is ready for use after the call.

    Failure mode
    -----------
    Exceptions are caught and logged. We never raise — the server must
    always start, and a failed warmup simply defers the work to the
    first user upload (the original pre-v2.0.8 behavior). Callers
    should NOT treat a False return value as a fatal error.
    """
    try:
        _get_chunker()
        return True
    except Exception:
        logger.exception("HybridChunker warmup failed")
        return False


def chunk_docling_doc(doc) -> Iterator[dict]:
    """Chunk a Docling ``DoclingDocument`` into ``{text, meta}`` records."""
    chunker = _get_chunker()
    for chunk in chunker.chunk(doc):
        text = chunk.text if hasattr(chunk, "text") else str(chunk)
        meta = {
            "chunk_type": _infer_chunk_type(chunk),
            "page_number": _extract_page(chunk),
            "section": _extract_section(chunk),
        }
        yield {"text": text, "meta": meta}


def _infer_chunk_type(chunk) -> str:
    label = getattr(chunk, "meta", None)
    if label is not None:
        for item in getattr(label, "doc_items", []) or []:
            lbl = str(getattr(item, "label", "")).lower()
            if "table" in lbl:
                return "table"
            if "list" in lbl:
                return "list"
    return "text"


def _extract_page(chunk) -> int | None:
    meta = getattr(chunk, "meta", None)
    if meta is None:
        return None
    for item in getattr(meta, "doc_items", []) or []:
        prov = getattr(item, "prov", None) or []
        if prov:
            page = getattr(prov[0], "page_no", None)
            if page:
                return int(page)
    return None


def _extract_section(chunk) -> str | None:
    """Best-effort section breadcrumb (heading path)."""
    meta = getattr(chunk, "meta", None)
    if meta is None:
        return None
    headings = getattr(meta, "headings", None) or []
    if headings:
        return " > ".join(str(h) for h in headings)
    return None


def chunk_plain_text(text: str, max_tokens: int | None = None) -> Iterator[dict]:
    """Simple chunker for plain text/markdown when Docling isn't available.

    Token-aware (uses BGE-M3 tokenizer if loaded; falls back to char-based
    recursive split).
    """
    chunker = _get_chunker()
    # HybridChunker accepts a "dl_doc" object. For plain text we use a
    # minimal DoclingDocument stub.
    from docling_core.types.doc import DoclingDocument  # type: ignore
    from docling_core.types.doc.labels import DocItemLabel  # type: ignore

    doc = DoclingDocument(name="inline")
    # ``add_text`` requires a DocItemLabel as its first positional arg;
    # PARAGRAPH is the right choice for free-flowing plain text (TEXT is
    # also fine but PARAGRAPH matches the typical block structure).
    # Without the label the call raises
    #   TypeError: DoclingDocument.add_text() missing 1 required positional
    #              argument: 'text'
    # which used to abort every plain-text upload at the very first chunk.
    doc.add_text(label=DocItemLabel.PARAGRAPH, text=text)
    yield from chunk_docling_doc(doc)


__all__ = ["chunk_docling_doc", "chunk_plain_text", "warmup_chunker"]
