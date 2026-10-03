"""Filename sanitization for upload destinations.

Why this exists
---------------
``file.filename`` from FastAPI / Starlette is a client-controlled string.
The previous code wrote ``uploads_dir() / doc_id / file.filename`` directly,
which gave a malicious client four easy escape paths:

* Traversal segments: ``../../etc/passwd`` → walks out of ``doc_id``.
* Absolute paths: ``/etc/passwd`` → resolves outside the upload root.
* Drive letters on Windows: ``C:\\Windows\\system.ini``.
* NUL bytes / control characters: ``foo\\x00.txt`` truncates to ``foo``
  on some filesystems but blows up others.

Defense contract
----------------
:func:`safe_filename` REJECTS (raises ``ValueError``) any input that
isn't a clean ``stem.ext`` shape — including inputs that *contained*
path separators, traversal segments, or other junk even if the leaf
would be safe. The caller should map the exception to a 400 so the
client sees a clear error rather than silently ending up with
``upload.bin``. For belt-and-suspenders safety against symlink-into-
root attacks that bypass the character filter, also call
:func:`assert_within_root` on the joined upload path.

Allowed character set
---------------------
Display filenames are written to disk verbatim (joined onto a per-
``doc_id`` directory), then stored in the doc registry for UI display
— they **never** flow into SQL/LanceDB filters or shell commands.
That means the character filter can be permissive about Unicode
content (CJK, emoji, parentheses, accented Latin, etc.) while still
hard-blocking the shapes that enable path traversal or break OS
filesystems:

* Path separators (``/`` / ``\\``) and NUL bytes — caught up front.
* Unicode category ``C*`` — control (``Cc``), format (``Cf``),
  surrogate (``Cs``), private-use (``Co``), unassigned (``Cn``) —
  filtered per-character via :func:`unicodedata.category`. Blocks
  zero-width joiners, RTL/LTR overrides, BOM, and other smuggled
  chars that ``str.startswith`` / ``endswith`` won't catch.
* Leading ``.`` (hidden file, breaks some UIs) and trailing ``.``
  (Windows silently strips trailing dots — ``CON.pdf.`` becomes
  ``CON.pdf`` on a Win desktop sync).
* Reserved Windows device names (``CON``, ``PRN``, ``AUX``, ``NUL``,
  ``COM1-9``, ``LPT1-9``) even when followed by a legitimate-looking
  extension.

Anything else — CJK ideographs, emoji, Latin-1 accents, parentheses,
brackets, regular spaces, hyphens, dots in the middle of the name —
passes through unchanged. The previous ASCII-only whitelist was
overly defensive: a Chinese CET-6 admit card named
``2025年上半年英语六级笔试准考证(150823200403172817).pdf`` is
a perfectly legitimate upload that the old rule rejected outright.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

# Reserved Windows device names (case-insensitive). Even on a non-Windows
# server, a malicious client could craft a PDF named ``CON.pdf`` that
# would misbehave if the file ever ended up on a Windows desktop via
# sync — refuse them up front.
_WIN_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _is_unsafe_char(ch: str) -> bool:
    """Per-character filter: reject path separators, NUL, and Unicode
    category ``C*`` (control / format / surrogate / private-use /
    unassigned).

    Everything else — letters (``L*``, including CJK ideographs),
    digits (``N*``), combining marks (``M*``), punctuation (``P*``,
    including ``()[]``), symbols (``S*``, including emoji), and the
    regular space separator (``Zs``) — is allowed through.
    """
    # Path separators are ASCII bytes that don't appear in any
    # ``unicodedata.category`` output. Reject them up front so the
    # rule is obvious to readers and so backslash on Windows or
    # forward-slash on POSIX is never normalized away by some
    # upstream library.
    if ch in ("/", "\\", "\x00"):
        return True
    # ``unicodedata.category`` returns a 2-letter code: first letter
    # is the major class. ``C`` covers control chars, format chars
    # (zero-width joiners, RTL/LTR overrides, BOM), surrogates,
    # private-use, and unassigned code points — all of which can
    # smuggle payload past naive string compares.
    return unicodedata.category(ch).startswith("C")


def safe_filename(name: str | None) -> str:
    """Sanitize a client-supplied filename.

    Returns the validated name (the same string when it was already
    clean). Raises ``ValueError`` on any input that isn't a clean
    ``stem.ext`` — including inputs that contain path separators or
    traversal segments, even if the leaf alone would pass the
    filter.

    Why reject instead of silently stripping
    ---------------------------------------
    Silently stripping ``../../etc/passwd`` → ``passwd`` would still
    leave a valid filename the caller might use to overwrite a
    legitimate ``passwd`` file. Rejecting with a clear 400 lets the
    client know their input was bad AND keeps the audit log honest —
    no silent rewrites of hostile filenames.

    Allowed: any Unicode ``stem.ext`` made of printable letters,
    digits, marks, punctuation, symbols, or single spaces — so CJK
    filenames like ``2025年上半年英语六级笔试准考证.pdf`` and
    emoji-heavy names like ``📚paper.pdf`` pass through.
    """
    if name is None:
        raise ValueError("filename is required")
    if not name or name in {".", ".."}:
        raise ValueError(f"unsafe filename: {name!r}")
    # Length 1-255 in CODE POINTS (not bytes). Most modern
    # filesystems enforce a byte cap (ext4 = 255 B, NTFS = 255
    # chars, APFS = 255 UTF-16 code units), but Python ``len()``
    # counts code points which is what users perceive as "length"
    # in their UI. A 100-char Chinese filename is ~300 bytes in
    # UTF-8, well under all caps; a 255-char Chinese filename is
    # ~765 B and might trip a bytes-based filesystem limit on
    # older Windows shares. Trade-off chosen: match user-visible
    # length, document the byte-vs-char caveat in the error.
    if not (1 <= len(name) <= 255):
        raise ValueError(
            f"unsafe filename {name!r}: length must be 1-255 characters"
        )
    # Reject path separators / NUL bytes up front so the per-character
    # loop below can stay focused on Unicode category filtering.
    if "/" in name or "\\" in name or "\x00" in name:
        raise ValueError(
            f"unsafe filename {name!r}: path separators / NUL bytes "
            f"are not allowed"
        )
    # Per-character filter: any control / format / surrogate /
    # private-use / unassigned code point is rejected.
    for ch in name:
        if _is_unsafe_char(ch):
            raise ValueError(
                f"unsafe filename {name!r}: contains control or "
                f"format characters (codepoint U+{ord(ch):04X})"
            )
    # Windows trims trailing dots and spaces on disk — a file
    # uploaded as ``report.pdf.`` would land as ``report.pdf`` on
    # a Win desktop sync, hiding the difference between the
    # original name and the on-disk name. Reject up front.
    if name.endswith(".") or name.endswith(" "):
        raise ValueError(
            f"unsafe filename {name!r}: trailing dots / spaces are "
            f"not allowed (Windows strips them on disk)"
        )
    # Leading dot = hidden file on POSIX, confuses some UIs that
    # filter for them.
    if name.startswith("."):
        raise ValueError(
            f"unsafe filename {name!r}: leading '.' is not allowed "
            f"(hidden file)"
        )
    # Reserved Windows device names — split off the extension and
    # compare case-insensitively so ``CON.pdf`` / ``Con`` both fail.
    # ``str.split('.', 1)`` is safe on Unicode — splitting ``报告.pdf``
    # gives ``['报告', 'pdf']`` and we check the stem against the
    # ASCII reserved set, so a CJK stem can never accidentally
    # collide with ``CON`` / ``PRN`` etc.
    stem = name.split(".", 1)[0].upper()
    if stem in _WIN_RESERVED:
        raise ValueError(f"unsafe filename: {name!r} (reserved device name)")
    # A stem of ``.`` (e.g. ``.pdf`` → stem ``''``) is technically
    # allowed by the regex but invisible in many UIs. ``.split('.', 1)``
    # on ``.pdf`` returns ``['', 'pdf']`` → stem is empty.
    if not stem or stem == ".":
        raise ValueError(f"unsafe filename: {name!r}")
    return name


def assert_within_root(path: Path, root: Path) -> Path:
    """Verify ``path`` resolves UNDER ``root``.

    Catches symlink-into-root bypasses that the character filter can't
    see: a ``doc_id`` directory could be a symlink that points outside
    the upload root. ``resolve()`` follows every symlink and returns
    the canonical absolute path; we then check that ``root.resolve()``
    is a strict prefix of ``path.resolve()``.

    Raises ``ValueError`` if the path escapes the root.
    """
    real_root = root.resolve()
    real_path = path.resolve()
    try:
        real_path.relative_to(real_root)
    except ValueError as exc:
        raise ValueError(
            f"path {path!r} resolves outside of root {root!r}"
        ) from exc
    return real_path


__all__ = ["safe_filename", "assert_within_root"]
