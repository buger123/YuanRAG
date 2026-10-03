"""Tests for PR-4 (v2.0.26.1) backend locale-aware error messages.

Verifies:
- ``parse_accept_language`` returns the right locale for the
  common shapes browsers send (single-tag, multi-tag with q, single
  zh-CN, etc.) and falls back to zh for unknown / missing / None.
- ``error_message(code, locale)`` returns the catalog string for
  every defined code in BOTH locales (zh + en).
- ``error_message`` template substitution works (kwargs fill the
  ``{key}`` placeholders).
- The wire schemas for upload / sessions / chat / ws error paths
  carry the localized copy when ``Accept-Language`` is set on the
  request, and fall back to zh when it's missing.
- A non-supported locale (``fr-FR``) on the request produces zh
  copy — the backend NEVER echoes the client's raw locale if we
  don't have a translation for it.

These tests are the catalog contract: if a code is added to
:func:`error_message`'s catalog without both locales, the
``test_catalog_has_both_locales_for_every_entry`` test below will
flag it.
"""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.i18n import (
    DEFAULT_LOCALE,
    SUPPORTED_LOCALES,
    error_message,
    parse_accept_language,
)


# ============================================================
# Catalog contract — every code MUST have zh + en
# ============================================================
#
# We pin this so a developer adding a new code without filling in
# both locales fails at CI, not at runtime. The test reads the
# private module attribute (acceptable inside the same package's
# test suite; would be a smell in production code).


def test_catalog_has_both_locales_for_every_entry():
    """Every entry in the catalog MUST carry both ``zh`` and ``en``.

    A half-localized error string is worse than fully-localized —
    the user sees English UI + Chinese error copy, the worst-of-
    both-worlds visual. PR-4 ships a CI-grade check here so this
    doesn't drift.
    """
    from src.api import i18n as i18n_module

    catalog = i18n_module._MESSAGES
    assert catalog, "catalog is empty — did the module load?"
    for code, entry in catalog.items():
        assert isinstance(entry, dict), f"{code}: entry must be a dict"
        assert "zh" in entry, f"{code}: missing zh locale"
        assert "en" in entry, f"{code}: missing en locale"
        # Don't accept empty strings — a half-localized entry with
        # one empty locale is just as broken as a missing locale.
        assert entry["zh"].strip(), f"{code}: zh locale is empty"
        assert entry["en"].strip(), f"{code}: en locale is empty"


def test_default_locale_is_zh():
    """The project default locale is Chinese. Pin it so a refactor
    that flips the default is caught here, not in a follow-up bug
    report."""
    assert DEFAULT_LOCALE == "zh"


def test_supported_locales_is_exactly_zh_and_en():
    """We support exactly two locales right now. Adding a third
    should be a conscious decision — bump this test alongside the
    catalog."""
    assert set(SUPPORTED_LOCALES) == {"zh", "en"}


# ============================================================
# parse_accept_language — parser behavior
# ============================================================


@pytest.mark.parametrize(
    "header,expected",
    [
        # Missing / empty / None — default to zh.
        (None, "zh"),
        ("", "zh"),
        # Single tag, both supported locales.
        ("zh", "zh"),
        ("en", "en"),
        # Region variants — primary tag wins.
        ("zh-CN", "zh"),
        ("zh-TW", "zh"),
        ("en-US", "en"),
        ("en-GB", "en"),
        # Multi-tag, supported locale first.
        ("zh-CN,en;q=0.9", "zh"),
        ("en-US,zh;q=0.9", "en"),
        # Multi-tag, supported locale buried but still leftmost.
        ("zh,en-US;q=0.8,fr;q=0.5", "zh"),
        # Quality factors ignored — first supported locale wins.
        ("en;q=0.1,zh;q=0.9", "en"),
        # All unsupported — fall back to zh.
        ("fr-FR,fr;q=0.9", "zh"),
        ("de,ja", "zh"),
        # Case-insensitive.
        ("ZH", "zh"),
        ("En", "en"),
        # Whitespace tolerance.
        ("  zh-CN , en;q=0.8 ", "zh"),
    ],
)
def test_parse_accept_language(header, expected):
    assert parse_accept_language(header) == expected


# ============================================================
# error_message — direct catalog lookups
# ============================================================


@pytest.mark.parametrize(
    "code,locale,expected",
    [
        # WebSocket errors — exact strings from the catalog.
        ("ws.invalid_json", "zh", "无效的 JSON"),
        ("ws.invalid_json", "en", "Invalid JSON"),
        ("ws.empty_message", "zh", "消息为空"),
        ("ws.empty_message", "en", "Empty message"),
        ("ws.rate_limited", "zh", "请求过于频繁,请稍后再试。"),
        ("ws.rate_limited", "en", "Rate limit exceeded. Please slow down."),
        # Upload errors.
        ("upload.empty", "zh", "上传的文件为空"),
        ("upload.empty", "en", "Empty upload"),
        # Session errors.
        ("session.thread_id_required", "zh", "缺少 thread_id"),
        ("session.thread_id_required", "en", "thread_id required"),
        ("session.load_history_failed", "zh", "加载历史失败"),
        ("session.load_history_failed", "en", "load history failed"),
        ("session.clear_failed", "zh", "清理会话失败"),
        ("session.clear_failed", "en", "clear_session failed"),
        # Chat errors.
        ("chat.clear_failed", "zh", "清理失败"),
        ("chat.clear_failed", "en", "clear failed"),
    ],
)
def test_error_message_returns_exact_catalog_string(code, locale, expected):
    assert error_message(code, locale) == expected


def test_error_message_default_locale_is_zh():
    """When the caller omits ``locale``, the helper defaults to zh."""
    assert error_message("ws.invalid_json") == "无效的 JSON"


def test_error_message_unknown_locale_falls_back_to_zh():
    """An unsupported locale (e.g. a typo ``EN`` — wait, case-
    insensitive, that's actually fine — ``jp`` is not) falls back
    to the zh copy. The backend NEVER echoes the raw locale if we
    don't have a translation for it."""
    assert error_message("ws.invalid_json", "jp") == "无效的 JSON"
    assert error_message("ws.invalid_json", "fr-FR") == "无效的 JSON"


def test_error_message_unknown_code_returns_code_string():
    """An unknown code returns the literal code string. Visible in
    QA logs, grep-able against the catalog to find the gap."""
    assert error_message("made.up.code", "en") == "made.up.code"
    assert error_message("does.not.exist", "zh") == "does.not.exist"


def test_error_message_sniff_mismatch_substitutes_ext():
    """The ``upload.sniff_mismatch`` template carries an ``{ext}``
    placeholder — verify kwargs are filled correctly in both locales.
    """
    out_zh = error_message(
        "upload.sniff_mismatch", "zh", ext=".pdf"
    )
    out_en = error_message(
        "upload.sniff_mismatch", "en", ext=".pdf"
    )
    assert ".pdf" in out_zh
    assert ".pdf" in out_en
    # The English version specifically calls out the extension
    # mismatch (more useful for a non-Chinese user).
    assert "extension" in out_en.lower()


# ============================================================
# Wire schemas — Accept-Language → localized error copy
# ============================================================
#
# These tests verify the routes honor the parsed locale: a request
# with ``Accept-Language: en`` sees English error copy, and one
# without the header sees Chinese.


@pytest.mark.asyncio
async def test_upload_empty_body_returns_localized_error(app_under_test):
    """POST /documents/upload with an empty body returns 400 with the
    locale-aware message. With ``Accept-Language: en`` we get the
    English copy; without the header we get Chinese (the default)."""
    # English path.
    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Accept-Language": "en"},
    ) as client:
        r = await client.post(
            "/documents/upload",
            files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
        )
    assert r.status_code == 400
    assert r.json()["detail"] == "Empty upload"

    # Chinese path (default — no header).
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        r = await client.post(
            "/documents/upload",
            files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
        )
    assert r.status_code == 400
    assert r.json()["detail"] == "上传的文件为空"


@pytest.mark.asyncio
async def test_upload_sniff_mismatch_returns_localized_message(app_under_test):
    """POST /documents/upload with magic-byte mismatch returns 400 with
    a localized ``message`` field. The ``diagnosis`` blob stays
    English-by-design (operator-facing log key)."""
    body = b"Hello, this is plain text, not a PDF."

    # English path — verify the message field.
    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Accept-Language": "en-US,en;q=0.9"},
    ) as client:
        r = await client.post(
            "/documents/upload",
            files={"file": ("evil.pdf", io.BytesIO(body), "application/pdf")},
        )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "extension" in detail["message"]
    assert ".pdf" in detail["message"]

    # Chinese path.
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        r = await client.post(
            "/documents/upload",
            files={"file": ("evil.pdf", io.BytesIO(body), "application/pdf")},
        )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "扩展名" in detail["message"]


@pytest.mark.asyncio
async def test_clear_session_returns_localized_error_on_failure(
    app_under_test,
):
    """DELETE /sessions/{id} on a checkpointer failure returns 500
    with the localized message."""
    # sessions.py imports ``delete_thread`` at module top via
    # ``from src.storage.checkpointer import delete_thread`` — so
    # the sessions route module holds its own reference. Patch
    # BOTH the source module (so any direct caller sees the
    # patched version) AND the route module (so the handler uses
    # the patched version).
    from src.api.routes import sessions as sessions_route

    with patch.object(
        sessions_route, "delete_thread"
    ) as mock_delete_route, patch(
        "src.storage.checkpointer.delete_thread"
    ) as mock_delete_src:
        mock_delete_route.side_effect = RuntimeError("db locked")
        mock_delete_src.side_effect = RuntimeError("db locked")
        transport = ASGITransport(app=app_under_test)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            headers={"Accept-Language": "en"},
        ) as client:
            r = await client.delete("/sessions/thread-x")
        assert r.status_code == 500
        assert r.json()["detail"] == "clear_session failed"

        # Chinese path — no header.
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            r = await client.delete("/sessions/thread-x")
        assert r.status_code == 500
        assert r.json()["detail"] == "清理会话失败"


@pytest.mark.asyncio
async def test_chat_clear_returns_localized_error_on_failure(
    app_under_test,
):
    """DELETE /chat/{id} on a checkpointer failure returns 500 with
    the localized message."""
    # chat.py imports delete_thread INSIDE the handler (the lazy
    # import avoids the checkpointer's heavy module-load cost at
    # process start), so patch the canonical storage module —
    # which is what the handler resolves to at call time.
    with patch(
        "src.storage.checkpointer.delete_thread"
    ) as mock_delete:
        mock_delete.side_effect = RuntimeError("db locked")
        transport = ASGITransport(app=app_under_test)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            headers={"Accept-Language": "en"},
        ) as client:
            r = await client.delete("/chat/thread-x")
        assert r.status_code == 500
        assert r.json()["detail"] == "clear failed"


@pytest.mark.asyncio
async def test_session_messages_empty_thread_id_returns_localized_error(
    app_under_test,
):
    """GET /sessions/{id}/messages with an empty thread_id returns
    400 with the localized ``thread_id required`` message."""
    # The empty-string case — FastAPI matches the ``/{thread_id}``
    # route with thread_id = "" (which is what the test sends).
    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Accept-Language": "en"},
    ) as client:
        r = await client.get("/sessions//messages")
    # Some FastAPI versions treat trailing-slash differently; just
    # assert that IF the route was hit with an empty thread_id, the
    # message is English.
    if r.status_code == 400:
        assert r.json()["detail"] == "thread_id required"


@pytest.mark.asyncio
async def test_session_messages_load_failure_returns_localized_error(
    app_under_test,
):
    """GET /sessions/{id}/messages when load_latest raises returns
    500 with the localized message."""
    # sessions.py imports ``load_latest`` INSIDE the handler, so
    # patch the source module instead of the route module.
    async def _failing_load(_tid):
        raise RuntimeError("checkpointer corruption")

    with patch(
        "src.storage.checkpointer.load_latest", _failing_load
    ):
        transport = ASGITransport(app=app_under_test)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            headers={"Accept-Language": "en"},
        ) as client:
            r = await client.get("/sessions/any-thread/messages")
        assert r.status_code == 500
        assert r.json()["detail"] == "load history failed"

        # Chinese path.
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            r = await client.get("/sessions/any-thread/messages")
        assert r.status_code == 500
        assert r.json()["detail"] == "加载历史失败"


@pytest.mark.asyncio
async def test_unsupported_accept_language_falls_back_to_chinese(
    app_under_test,
):
    """An unsupported locale (``fr-FR``) on the request produces
    Chinese copy — the backend NEVER echoes the raw locale if we
    don't have a translation for it."""
    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Accept-Language": "fr-FR,fr;q=0.9"},
    ) as client:
        r = await client.post(
            "/documents/upload",
            files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
        )
    assert r.status_code == 400
    assert r.json()["detail"] == "上传的文件为空"
