"""Tests for ``src.utils.safe_filename`` + ``assert_within_root``.

Two related defenses:

1. ``safe_filename`` — filters directory segments, control chars, and
   OS-incompatible shapes. Used before joining a user-supplied
   filename onto an upload root. Allows Unicode content (CJK, emoji,
   parentheses, spaces) because display filenames never flow into SQL
   filters or shell commands — they live in a per-doc dir on disk and
   in the doc registry as UI metadata.

2. ``assert_within_root`` — checks that a (possibly symlinked) path
   still resolves under its intended root. Defense in depth: catches
   ``doc_id`` symlinks that bypass the character filter.

These tests intentionally cover the surprising attack shapes (empty
stem, trailing dot, reserved device names, NUL bytes, control chars)
AND the surprising acceptance shapes (CJK, emoji, parens, spaces)
rather than just the happy path, so future regressions land here
first regardless of which direction the policy moves.
"""
from __future__ import annotations

from pathlib import Path

import pytest


# ============================================================
# safe_filename — happy path
# ============================================================


def test_safe_filename_accepts_plain_name():
    from src.utils.safe_filename import safe_filename

    assert safe_filename("report.pdf") == "report.pdf"


def test_safe_filename_accepts_dots_and_dashes():
    from src.utils.safe_filename import safe_filename

    assert safe_filename("draft-v2.0.final.md") == "draft-v2.0.final.md"


def test_safe_filename_accepts_long_name_up_to_255():
    from src.utils.safe_filename import safe_filename

    name = "a" * 200 + ".txt"
    assert safe_filename(name) == name


def test_safe_filename_accepts_single_internal_space():
    """Spaces mid-name are allowed — they're a normal printable char
    in every modern OS. (The previous policy rejected them; that was
    overly defensive for filenames that never enter a shell command.)"""
    from src.utils.safe_filename import safe_filename

    assert safe_filename("my file.txt") == "my file.txt"


def test_safe_filename_accepts_parentheses():
    """``()`` are Unicode punctuation (``Pe``), not control chars, and
    are legal in filenames on every supported OS. The original CET-6
    ticket was blocked because ``safe_filename`` was ASCII-only."""
    from src.utils.safe_filename import safe_filename

    assert (
        safe_filename("2025年上半年英语六级笔试准考证(150823200403172817).pdf")
        == "2025年上半年英语六级笔试准考证(150823200403172817).pdf"
    )


def test_safe_filename_accepts_cjk():
    """Chinese / Japanese / Korean ideographs are category ``Lo``
    (letter, other) — perfectly safe filename content. The previous
    ASCII-only whitelist rejected these, which broke real user uploads
    like CET-6 admit cards and Chinese-language research papers."""
    from src.utils.safe_filename import safe_filename

    assert safe_filename("报告.pdf") == "报告.pdf"
    assert safe_filename("日本語のメモ.txt") == "日本語のメモ.txt"
    assert safe_filename("한글-파일명.docx") == "한글-파일명.docx"


def test_safe_filename_accepts_emoji():
    """Emoji are category ``So`` (symbol, other). Modern OSes all
    support them in filenames; treating them as unsafe would block
    casual user uploads (``📚paper.pdf``) without any security
    benefit, since the downstream storage is per-doc-dir only."""
    from src.utils.safe_filename import safe_filename

    assert safe_filename("📚paper.pdf") == "📚paper.pdf"


def test_safe_filename_accepts_accented_latin():
    """Latin-1 supplement and combining marks must pass — they're
    common in non-English user names and product labels."""
    from src.utils.safe_filename import safe_filename

    assert safe_filename("café-résumé.pdf") == "café-résumé.pdf"


def test_safe_filename_strips_leading_directory_segments():
    """Path separators are rejected outright (not silently stripped).

    Silently rewriting ``subdir/file.txt`` → ``file.txt`` would let an
    attacker overwrite a legitimate file under ``doc_id``. We surface
    the attempt as a 400 instead, so the audit log catches it.
    """
    from src.utils.safe_filename import safe_filename

    with pytest.raises(ValueError):
        safe_filename("subdir/file.txt")


# ============================================================
# safe_filename — rejections (must raise ValueError)
# ============================================================


def test_safe_filename_rejects_parent_dir_traversal():
    from src.utils.safe_filename import safe_filename

    with pytest.raises(ValueError):
        safe_filename("../etc/passwd")


def test_safe_filename_rejects_absolute_path():
    from src.utils.safe_filename import safe_filename

    with pytest.raises(ValueError):
        safe_filename("/etc/passwd")


def test_safe_filename_rejects_windows_drive_letter():
    from src.utils.safe_filename import safe_filename

    with pytest.raises(ValueError):
        safe_filename("C:\\Windows\\system.ini")


def test_safe_filename_rejects_nul_byte():
    from src.utils.safe_filename import safe_filename

    with pytest.raises(ValueError):
        safe_filename("foo\x00.txt")


def test_safe_filename_rejects_path_separator_in_name():
    """Even a single slash mid-name is rejected (path-separator check)."""
    from src.utils.safe_filename import safe_filename

    with pytest.raises(ValueError):
        safe_filename("foo/bar")


def test_safe_filename_rejects_control_chars():
    """ASCII control chars (U+0001..U+001F) are rejected via the
    Unicode category ``Cc`` filter."""
    from src.utils.safe_filename import safe_filename

    for ch in ("\x01", "\x07", "\x1b", "\x1f"):
        with pytest.raises(ValueError):
            safe_filename(f"foo{ch}.txt")


def test_safe_filename_rejects_zero_width_chars():
    """Zero-width joiners / non-joiners / BOM / RTL overrides are
    category ``Cf`` (format) — historically abused to bypass string
    comparisons by inserting visually-invisible chars. Reject
    explicitly even though the per-char filter already catches them,
    because this is the most subtle of the categories."""
    from src.utils.safe_filename import safe_filename

    for ch in ("\u200b", "\u200c", "\u200d", "\ufeff", "\u202e"):
        with pytest.raises(ValueError):
            safe_filename(f"foo{ch}bar.pdf")


def test_safe_filename_rejects_empty_string():
    from src.utils.safe_filename import safe_filename

    with pytest.raises(ValueError):
        safe_filename("")


def test_safe_filename_rejects_none():
    from src.utils.safe_filename import safe_filename

    with pytest.raises(ValueError):
        safe_filename(None)


def test_safe_filename_rejects_dotdot_alone():
    from src.utils.safe_filename import safe_filename

    with pytest.raises(ValueError):
        safe_filename("..")


def test_safe_filename_rejects_dot_alone():
    from src.utils.safe_filename import safe_filename

    with pytest.raises(ValueError):
        safe_filename(".")


def test_safe_filename_rejects_dotfile():
    """Hidden files (leading dot) confuse some UIs and aren't useful
    in a chat-upload context."""
    from src.utils.safe_filename import safe_filename

    with pytest.raises(ValueError):
        safe_filename(".hidden")


def test_safe_filename_rejects_trailing_dot():
    """Windows strips trailing dots on disk (``report.pdf.`` becomes
    ``report.pdf``), which would silently rename the upload when
    synced to a Win client. Reject to keep the on-disk name
    byte-equal to the user-supplied name."""
    from src.utils.safe_filename import safe_filename

    with pytest.raises(ValueError):
        safe_filename("report.pdf.")


def test_safe_filename_rejects_trailing_space():
    """Same Windows-on-disk stripping concern as trailing dots —
    ``report.pdf`` (with literal trailing space) lands as
    ``report.pdf`` on POSIX but as ``report`` on a Win desktop sync.
    Note the trailing space is in the source, not in the test
    string above; visualizer UIs sometimes strip it on display."""
    from src.utils.safe_filename import safe_filename

    with pytest.raises(ValueError):
        safe_filename("report.pdf ")  # noqa: W291 — intentional trailing space


def test_safe_filename_rejects_reserved_windows_device():
    from src.utils.safe_filename import safe_filename

    for reserved in ("CON", "PRN", "AUX", "NUL", "COM1", "LPT1"):
        with pytest.raises(ValueError):
            safe_filename(f"{reserved}.pdf")


def test_safe_filename_rejects_reserved_case_insensitive():
    from src.utils.safe_filename import safe_filename

    with pytest.raises(ValueError):
        safe_filename("con.pdf")


def test_safe_filename_rejects_reserved_without_extension():
    """``CON`` alone (no extension) is also reserved on Windows."""
    from src.utils.safe_filename import safe_filename

    with pytest.raises(ValueError):
        safe_filename("CON")


def test_safe_filename_rejects_overly_long_name():
    from src.utils.safe_filename import safe_filename

    long = "a" * 256 + ".txt"
    with pytest.raises(ValueError):
        safe_filename(long)


def test_safe_filename_rejects_extreme_cjk_length():
    """A 256-character CJK filename is ~765 bytes in UTF-8 — over the
    255-char cap and would trip filesystem limits on legacy Windows
    shares. Length check fires before per-char filter, so the error
    points at the actual issue."""
    from src.utils.safe_filename import safe_filename

    long = "文" * 256 + ".pdf"
    with pytest.raises(ValueError, match="length"):
        safe_filename(long)


def test_safe_filename_rejects_empty_stem():
    """``.pdf`` → stem ``''`` after split. Hidden in some UIs."""
    from src.utils.safe_filename import safe_filename

    with pytest.raises(ValueError):
        safe_filename(".pdf")


# ============================================================
# assert_within_root — happy path
# ============================================================


def test_assert_within_root_accepts_path_inside_root(tmp_path: Path):
    from src.utils.safe_filename import assert_within_root

    target = tmp_path / "doc1" / "report.pdf"
    target.parent.mkdir()
    target.write_bytes(b"x")
    assert assert_within_root(target, tmp_path) == target.resolve()


def test_assert_within_root_accepts_root_itself(tmp_path: Path):
    from src.utils.safe_filename import assert_within_root

    assert assert_within_root(tmp_path, tmp_path) == tmp_path.resolve()


def test_assert_within_root_accepts_cjk_filename_inside_root(tmp_path: Path):
    """A CJK filename written to disk must round-trip through
    ``assert_within_root`` without the resolver tripping on the
    multibyte name — i.e. the upload pipeline (which writes
    ``uploads/doc_id/<CJK_name>`` and then validates the join)
    works end-to-end on non-ASCII filenames."""
    from src.utils.safe_filename import assert_within_root

    target = tmp_path / "doc1" / "2025年上半年英语六级笔试准考证(150823200403172817).pdf"
    target.parent.mkdir()
    target.write_bytes(b"%PDF-1.4\n")
    assert assert_within_root(target, tmp_path) == target.resolve()


# ============================================================
# assert_within_root — rejection (symlink bypass)
# ============================================================


def test_assert_within_root_rejects_path_outside_root(tmp_path: Path):
    """A path that points to a sibling directory must be rejected."""
    from src.utils.safe_filename import assert_within_root

    other = tmp_path.parent / "outside.txt"
    other.write_text("x")
    with pytest.raises(ValueError):
        assert_within_root(other, tmp_path)
    other.unlink()


def test_assert_within_root_rejects_symlink_escape(tmp_path: Path):
    """The classic bypass: ``doc_id`` is a symlink to ``/etc`` — even
    if the filename is clean, the resolved path leaves the upload root."""
    from src.utils.safe_filename import assert_within_root

    real_target = tmp_path.parent / f"symlink_target_{tmp_path.name}"
    real_target.mkdir()
    symlink_dir = tmp_path / "doc1"
    symlink_dir.symlink_to(real_target)

    with pytest.raises(ValueError):
        assert_within_root(symlink_dir / "passwd", tmp_path)

    # Cleanup
    symlink_dir.unlink()
    real_target.rmdir()
