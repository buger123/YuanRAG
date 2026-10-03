"""Plain-text / Markdown parser.

Reads bytes and returns the full content. Chunking is delegated to the
generic HybridChunker downstream.
"""
from __future__ import annotations

import codecs


# v1.1.12b — encoding chain mirrors ``documents._looks_like_text`` so a
# file that passes the validator decodes the same way the parser decodes
# it. Pre-fix the parser only knew UTF-8 → latin-1, which silently
# mojibake'd a GBK/Big5/CP1252 file into unsearchable garbage. Order
# matches the sniff (UTF-8 first; UTF-16 BOMs already short-circuited
# upstream). Each step uses an incremental decoder with ``final=False``
# so a trailing partial byte does not blow up a file that the sniff
# accepted.
_PARSE_ENCODINGS = ("utf-8", "gb18030", "big5", "cp1252")


def parse_text(content_bytes: bytes, encoding: str = "utf-8") -> str:
    """Decode text bytes with the given encoding.

    Tries ``encoding`` first, then falls back through the same chain the
    sniff uses (``utf-8 → gb18030 → big5 → cp1252``), and finally
    ``latin-1`` which never raises for any byte sequence — the
    absolute last resort so the parser can never throw on a payload
    the validator accepted.
    """
    try:
        return content_bytes.decode(encoding)
    except UnicodeDecodeError:
        pass
    for enc in _PARSE_ENCODINGS:
        if enc == encoding:
            continue
        dec = codecs.getincrementaldecoder(enc)("strict")
        try:
            return dec.decode(content_bytes, final=True)
        except UnicodeDecodeError:
            continue
    return content_bytes.decode("latin-1")


__all__ = ["parse_text"]
