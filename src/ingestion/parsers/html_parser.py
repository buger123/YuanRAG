"""HTML parser with a Docling fast path.

Docling's layout pipeline is overkill for plain HTML — a 2 KB marketing
page triggers the same ML layout/OCR model warmup as a 50-page PDF, taking
~10 s end-to-end. For HTML that looks like a normal webpage (no embedded
images, no SVG, no tables), we strip tags with BeautifulSoup and feed the
plain text to the existing chunker in <100 ms.

Anything complex still falls through to Docling via
``src.ingestion.pipeline``.
"""
from __future__ import annotations

from bs4 import BeautifulSoup


# Tags that signal "this needs Docling's layout-aware extraction":
# * <img> / <svg> — visual content the user expects OCR'd or described
# * <table> — table structure preservation matters for RAG context
# * <picture> / <canvas> — modern equivalents of <img>
_HEAVY_TAGS = ("img", "svg", "table", "picture", "canvas")


def is_plain_html(content_bytes: bytes, max_bytes: int = 16 * 1024) -> bool:
    """Decide whether an HTML document is "plain enough" to bypass Docling.

    Plain-HTML criteria:

    * Small enough that we'd rather re-parse than boot Docling (``max_bytes``,
      default 16 KiB — covers the vast majority of web pages the user might
      paste in).
    * No tags requiring layout/OCR/table extraction.

    Returns False for ``None``, oversized payloads, or pages carrying
    visual / tabular content — those keep the Docling path.
    """
    if not content_bytes or len(content_bytes) > max_bytes:
        return False
    # Cheap precheck: bail out the moment we see a heavy tag opening. We
    # intentionally scan the raw bytes (not a parsed tree) because the
    # bottleneck we want to skip is BeautifulSoup itself — a 16 KiB page
    # parses in ~3 ms, so this branch is the cheap one.
    lowered = content_bytes.lower()
    for tag in _HEAVY_TAGS:
        if f"<{tag}".encode() in lowered or f"<{tag} ".encode() in lowered:
            return False
    return True


def parse_html_plain(content_bytes: bytes, encoding: str = "utf-8") -> str:
    """Strip HTML tags and return readable text.

    Uses ``html.parser`` (the stdlib backend) — fastest, no lxml /
    html5lib overhead. Output joins block-level elements with newlines
    so the downstream chunker gets sensible paragraph boundaries
    instead of one long line.
    """
    try:
        text = content_bytes.decode(encoding)
    except UnicodeDecodeError:
        text = content_bytes.decode("latin-1")
    soup = BeautifulSoup(text, "html.parser")
    # Drop script/style wholesale (their text content is never useful).
    for tag in soup(["script", "style"]):
        tag.decompose()
    # get_text with a space separator preserves inter-tag whitespace.
    # strip=True drops leading/trailing whitespace from each chunk.
    return soup.get_text(" ", strip=True)


__all__ = ["is_plain_html", "parse_html_plain"]