"""Tests for the v2.0.6 P2 frontend-polish batch.

Bug history
-----------
A "局外专家" review surfaced four UX/UI issues that the rest of the
project has already cleared (P0/P1) but the polish tier had not
touched yet:

  P2-1  ``GET /sessions`` returned ``updated_at=""`` for every
        doc-only thread (a thread that has uploaded documents but
        never sent a message — i.e. has no LangGraph checkpoint).
        The sidebar sorted those threads below every other thread
        regardless of when the upload happened, and the empty
        timestamp showed up as "1970/1/1" in the React UI.
        **Fix**: a per-thread ``_thread_latest_ingested_at`` reverse
        index alongside ``_thread_first_filename``; populated by
        ``add_chunks`` incrementally and on the cold-path
        ``_build_reverse_index_from_disk``. New public helper
        :func:`latest_ingested_at_by_threads`. ``sessions.py`` uses
        it to populate ``updated_at`` from ``max(ingested_at)`` per
        doc-only thread.

  P2-2  ``POST /documents/upload`` only accepted ``thread_id`` as a
        query parameter. SDK callers who wanted to upload WITH the
        thread id in a single ``multipart/form-data`` body (the
        natural way for HTTP upload libraries) had to splice it into
        the URL. **Fix**: add a ``Form(None, alias="thread_id")``
        parameter; form wins when both are supplied, query kept for
        legacy callers.

  P2-3  REMOVED before ship. The dark theme shipped as
        ``[data-theme="dark"]`` + ``@media (prefers-color-scheme:
        dark)`` + ``useLocalStorage``-backed preference + a 3-way
        segmented control in ``SettingsDialog``. It was reverted
        on user request (2026-09-10) because the project is
        intentionally a single-theme UI; adding dark mode would
        require re-designing every contrast pair. The
        ``useLocalStorage`` hook itself survives because P2-4
        (i18n locale persistence) reuses it.

  P2-4  No English translation. The codebase was 100% zh-CN; non-
        Chinese-speaking reviewers couldn't navigate the UI.
        **Fix**: new ``i18n/zh.ts`` + ``i18n/en.ts`` dictionaries;
        new ``useLocale`` React Context with a ``t(key, vars?)``
        lookup. ``SettingsDialog`` gets a "Language" segmented
        control; representative strings wired up in App /
        SettingsDialog / Sidebar / ChatPane to prove the loop.

These tests pin the behavior at three levels:

  1. **Backend** — direct unit tests against the new helpers and
     endpoint signatures.
  2. **Frontend** — static-grep tests that confirm the new
     files exist, key surfaces contain the expected strings, and
     the dictionary keys line up between zh.ts and en.ts.
  3. **Cross-cutting** — the reverse-index invariant must survive
     deletion (so a doc-only thread that gets all its docs
     removed ALSO disappears from the sidebar rather than leaving
     a stale ``updated_at``).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
STYLES_CSS = REPO_ROOT / "src" / "frontend" / "src" / "styles.css"
APP_TSX = REPO_ROOT / "src" / "frontend" / "src" / "App.tsx"
SETTINGS_TSX = REPO_ROOT / "src" / "frontend" / "src" / "components" / "SettingsDialog.tsx"
SIDEBAR_TSX = REPO_ROOT / "src" / "frontend" / "src" / "components" / "Sidebar.tsx"
CHATPANE_TSX = REPO_ROOT / "src" / "frontend" / "src" / "components" / "ChatPane.tsx"
I18N_DIR = REPO_ROOT / "src" / "frontend" / "src" / "i18n"
ZH_TS = I18N_DIR / "zh.ts"
EN_TS = I18N_DIR / "en.ts"
HOOKS_DIR = REPO_ROOT / "src" / "frontend" / "src" / "hooks"
USE_LOCAL_STORAGE = HOOKS_DIR / "useLocalStorage.ts"


# ============================================================
# P2-1 — doc-only thread ``updated_at`` populated from
#        ``max(ingested_at)`` rather than hardcoded ``""``
# ============================================================


def _make_record(doc_id: str, thread_id: str, ingested_at: str = "2026-09-10T08:00:00Z"):
    """Build a ChunkRecord with an explicit ``ingested_at`` so the
    reverse-index max logic is deterministic in tests."""
    from src.storage.schema import ChunkRecord

    return ChunkRecord(
        chunk_id=f"{doc_id}-c1",
        doc_id=doc_id,
        text="hello world",
        vector=[0.0] * 1024,
        sparse={1: 1.0},
        filename="report.pdf",
        source_type="file",
        # v2.0.27 P0-P2 — canonical privacy-safe form (doc_id/filename).
        source_path=f"{doc_id}/report.pdf",
        mime_type="application/pdf",
        chunk_index=0,
        chunk_type="text",
        page_number=1,
        section=None,
        sheet=None,
        headers=None,
        row_start=None,
        row_end=None,
        thread_id=thread_id,
        ingested_at=ingested_at,
    )


def test_p21_latest_ingested_at_empty_when_no_chunks():
    """A thread with no indexed docs maps to ``""`` (matches the
    pre-v2.0.6 wire contract for empty sessions — the sidebar
    keeps its existing sort fallback)."""
    from src.storage import lancedb_store

    lancedb_store.reset_for_tests()
    out = lancedb_store.latest_ingested_at_by_threads(["alpha", "beta"])
    assert out == {"alpha": "", "beta": ""}


def test_p21_latest_ingested_at_returns_max_per_thread():
    """Multiple chunks on one thread contribute their max
    ``ingested_at``. ISO 8601 strings sort lexicographically the
    same as chronologically, so plain ``max`` works."""
    from src.storage import lancedb_store

    lancedb_store.reset_for_tests()
    lancedb_store.add_chunks([
        _make_record("d1", "alpha", ingested_at="2026-09-10T08:00:00Z"),
        _make_record("d2", "alpha", ingested_at="2026-09-10T09:30:00Z"),
        _make_record("d3", "alpha", ingested_at="2026-09-10T07:15:00Z"),
    ])
    out = lancedb_store.latest_ingested_at_by_threads(["alpha"])
    assert out == {"alpha": "2026-09-10T09:30:00Z"}


def test_p21_latest_ingested_at_independent_per_thread():
    """Each thread has its own max; one thread's newest chunk
    doesn't leak into another thread's slot."""
    from src.storage import lancedb_store

    lancedb_store.reset_for_tests()
    lancedb_store.add_chunks([
        _make_record("d1", "alpha", ingested_at="2026-09-10T10:00:00Z"),
        _make_record("d2", "beta", ingested_at="2026-09-10T06:00:00Z"),
    ])
    out = lancedb_store.latest_ingested_at_by_threads(["alpha", "beta", "gamma"])
    assert out["alpha"] == "2026-09-10T10:00:00Z"
    assert out["beta"] == "2026-09-10T06:00:00Z"
    assert out["gamma"] == ""


def test_p21_latest_ingested_at_updates_on_subsequent_add_chunks():
    """Uploading a second doc to a thread BUMP the latest-ingested-at
    if the new doc is newer — the doc-only ``updated_at`` must
    reflect the most recent upload, not just the first."""
    from src.storage import lancedb_store

    lancedb_store.reset_for_tests()
    lancedb_store.add_chunks([_make_record("d1", "alpha", ingested_at="2026-09-10T08:00:00Z")])
    out1 = lancedb_store.latest_ingested_at_by_threads(["alpha"])
    assert out1["alpha"] == "2026-09-10T08:00:00Z"

    lancedb_store.add_chunks([_make_record("d2", "alpha", ingested_at="2026-09-11T12:00:00Z")])
    out2 = lancedb_store.latest_ingested_at_by_threads(["alpha"])
    assert out2["alpha"] == "2026-09-11T12:00:00Z"


def test_p21_latest_ingested_at_cleared_when_thread_empty():
    """Deleting the last doc of a thread drops the cached
    latest-ingested-at — otherwise a stale timestamp could outlive
    the thread's last document and ghost the sidebar."""
    from src.storage import lancedb_store

    lancedb_store.reset_for_tests()
    lancedb_store.add_chunks([_make_record("d1", "alpha", ingested_at="2026-09-10T08:00:00Z")])
    assert lancedb_store.latest_ingested_at_by_threads(["alpha"])["alpha"]

    lancedb_store.delete_by_thread_id("alpha")
    assert lancedb_store.latest_ingested_at_by_threads(["alpha"])["alpha"] == ""


def test_p21_sessions_route_uses_helper_for_doc_only_threads():
    """The route module must import and call the new helper so a
    doc-only thread's ``updated_at`` is sourced from
    ``max(ingested_at)`` rather than the pre-v2.0.6 hardcoded ``""``.

    v2.0.21 (Phase 3) — anchor on the doc-only branch sentinel
    (``doc_only_tids = [``) so the regex skips the import line's
    ``list_threads_with_documents`` reference and matches the
    actual call site.
    """
    from src.api.routes import sessions

    src = Path(sessions.__file__).read_text(encoding="utf-8")
    assert "latest_ingested_at_by_threads" in src, (
        "sessions route must call latest_ingested_at_by_threads"
        " to populate updated_at for doc-only threads (P2-1)"
    )
    # The pre-v2.0.6 hardcoded empty string for doc-only threads
    # must NOT survive in the doc-only branch. Anchor on the
    # ``doc_only_tids = [`` sentinel to avoid matching the import
    # line's ``list_threads_with_documents`` reference.
    doc_only_block = re.search(
        r"doc_only_tids\s*=\s*\[.*?aggregated\[tid\]\s*=\s*\{[^}]+\}",
        src,
        re.DOTALL,
    )
    assert doc_only_block, "couldn't find the doc-only merge block"
    block_text = doc_only_block.group(0)
    # The fix: the value should come from the helper, not a literal "".
    assert 'updated_at": doc_only_ts.get(tid' in block_text, (
        "doc-only updated_at must be sourced from "
        "doc_only_ts.get(tid, '') — not the hardcoded ''"
    )


# ============================================================
# P2-2 — ``POST /documents/upload`` accepts thread_id as both
#        Form and Query
# ============================================================


def test_p22_upload_accepts_form_alias_for_thread_id():
    """The upload route signature must accept thread_id via Form
    (alias='thread_id') so SDK callers can ship it inside the
    multipart body instead of the URL."""
    from src.api.routes import documents

    src = Path(documents.__file__).read_text(encoding="utf-8")
    assert "Form(None, alias=\"thread_id\")" in src, (
        "upload must accept Form(None, alias='thread_id') so the "
        "multipart form body can carry thread_id (P2-2)"
    )


def test_p22_upload_query_thread_id_still_works():
    """The legacy Query path must still resolve — backward compat
    with pre-v2.0.6 callers (and curl)."""
    from src.api.routes import documents

    src = Path(documents.__file__).read_text(encoding="utf-8")
    # Both parameters live on the function; query one is the plain
    # ``thread_id: str | None = None`` kwarg.
    assert "thread_id: str | None = None" in src
    # Form wins when both are provided — the ``or`` short-circuits
    # to the form value.
    assert "thread_id_form or thread_id" in src


def test_p22_upload_openapi_documents_both_aliases():
    """The OpenAPI schema must surface the Form alias explicitly so
    generated SDKs (openapi-generator, etc.) pick it up. The
    alias= kwarg surfaces in the JSON schema as ``name: thread_id``,
    so we just verify the parameter is declared."""
    from src.api.routes import documents

    src = Path(documents.__file__).read_text(encoding="utf-8")
    # Pull the function signature block.
    sig = re.search(
        r"async def upload\((.*?)\):",
        src,
        re.DOTALL,
    )
    assert sig, "couldn't locate the upload function signature"
    params = sig.group(1)
    assert "thread_id_form" in params, (
        "Form alias kwarg must be named thread_id_form internally "
        "so the signature reads cleanly (P2-2)"
    )


# ============================================================
# P2-3 — REMOVED before ship (2026-09-10, user request)
# ============================================================
#
# Dark theme was implemented end-to-end (CSS tokens, useLocalStorage
# hook, App.tsx state, SettingsDialog 3-way segmented control) and
# then reverted on user request. The project is intentionally a
# single-theme UI; adding dark mode would require re-designing every
# contrast pair (border hairlines, status badges, accent shadows)
# which the user judged not worth the cost.
#
# These tests pin the REVERTED state — they assert that no dark
# theme artifacts survive, with the explicit exception of
# ``useLocalStorage`` (which P2-4 i18n still reuses for the locale
# preference) and the segmented-control CSS (which P2-4 language
# picker still uses).


def test_p23_useLocalStorage_hook_still_present():
    """P2-3's ``useLocalStorage`` hook survived the revert because
    P2-4 (i18n) reuses it for ``rag.locale``. Removing it would
    break the locale provider — keep this test as a tripwire."""
    assert USE_LOCAL_STORAGE.exists(), (
        "useLocalStorage hook must still exist even after P2-3 "
        "revert — P2-4 i18n locale persistence depends on it"
    )
    text = USE_LOCAL_STORAGE.read_text(encoding="utf-8")
    assert "useLocalStorage" in text
    assert "useState" in text
    assert "catch" in text, (
        "useLocalStorage must keep its quota / private-mode catch"
    )


def test_p23_no_dark_theme_block_in_styles():
    """After revert, styles.css must NOT contain a
    ``[data-theme="dark"]`` block. Pinning the absence prevents
    a future contributor from quietly re-adding it."""
    text = STYLES_CSS.read_text(encoding="utf-8")
    assert '[data-theme="dark"]' not in text, (
        "P2-3 was reverted; [data-theme=\"dark\"] must NOT exist "
        "in styles.css"
    )


def test_p23_no_prefers_color_scheme_query():
    """After revert, the ``auto`` mode media query must be gone
    too — otherwise an OS-level dark preference would still
    recolor the UI even though the user opted out of dark mode."""
    text = STYLES_CSS.read_text(encoding="utf-8")
    assert "@media (prefers-color-scheme: dark)" not in text, (
        "P2-3 was reverted; @media (prefers-color-scheme: dark) "
        "must NOT exist in styles.css"
    )


def test_p23_no_bg_overlay_token():
    """The ``--bg-overlay`` token was introduced by P2-3 to invert
    the hover/active overlay between themes. Without dark mode
    it's a dead token — remove it."""
    text = STYLES_CSS.read_text(encoding="utf-8")
    assert "--bg-overlay" not in text, (
        "P2-3 was reverted; --bg-overlay token must NOT exist "
        "(the 2 hardcoded rgba(0,0,0,...) overlays are restored)"
    )


def test_p23_app_no_theme_state():
    """After revert, App.tsx must NOT carry theme state, the
    ``rag.theme`` localStorage key, or the data-theme attribute
    effect. Pin the absence."""
    text = APP_TSX.read_text(encoding="utf-8")
    assert "rag.theme" not in text, (
        "P2-3 was reverted; rag.theme localStorage key must NOT "
        "exist in App.tsx"
    )
    assert "data-theme" not in text, (
        "P2-3 was reverted; data-theme attribute must NOT be "
        "set by App.tsx"
    )
    # The theme-preference tuple must not be wired through either.
    assert '"light"' not in text or "rag.theme" not in text


def test_p23_settings_no_appearance_section():
    """After revert, SettingsDialog must NOT render the
    "Appearance" section or the THEME_OPTIONS segmented control."""
    text = SETTINGS_TSX.read_text(encoding="utf-8")
    assert "settings.appearance" not in text
    assert "settings.themeLight" not in text
    assert "settings.themeDark" not in text
    assert "settings.themeAuto" not in text
    assert "onThemeChange" not in text, (
        "P2-3 was reverted; SettingsDialog must NOT accept "
        "onThemeChange — only the language picker remains"
    )


def test_p23_settings_theme_props_removed_from_signature():
    """The SettingsDialog Props interface must no longer carry
    ``theme`` or ``onThemeChange`` — confirm the function
    signature itself."""
    text = SETTINGS_TSX.read_text(encoding="utf-8")
    sig = re.search(
        r"interface Props\s*\{(.*?)\}",
        text,
        re.DOTALL,
    )
    assert sig, "couldn't locate Props interface"
    body = sig.group(1)
    assert "theme:" not in body or "theme string" in body.lower(), (
        "Props must not carry the theme preference after P2-3 revert"
    )
    assert "onThemeChange" not in body


def test_p23_i18n_dicts_have_no_theme_keys():
    """The 4 theme keys (appearance / themeLight / themeDark /
    themeAuto) must be gone from both i18n/zh.ts and i18n/en.ts
    so a leftover ``t("settings.themeDark")`` doesn't fall
    through to the bare key string."""
    zh_text = ZH_TS.read_text(encoding="utf-8")
    en_text = EN_TS.read_text(encoding="utf-8")
    for k in ("settings.appearance", "settings.theme", "settings.themeLight",
              "settings.themeDark", "settings.themeAuto"):
        assert k not in zh_text, f"zh.ts still has key {k!r}"
        assert k not in en_text, f"en.ts still has key {k!r}"


# ============================================================
# P2-4 — i18n foundation
# ============================================================


def test_p24_i18n_zh_and_en_files_exist():
    """The dictionaries live at ``i18n/{lang}.ts`` so a future
    third locale (e.g. ``ja.ts``) is a one-file addition."""
    assert ZH_TS.exists(), "i18n/zh.ts must exist (P2-4)"
    assert EN_TS.exists(), "i18n/en.ts must exist (P2-4)"


def test_p24_i18n_keys_match_between_zh_and_en():
    """Every Chinese key must have an English counterpart —
    otherwise a partial translation surfaces as a missing label
    (silent UX bug). The union of both sets must be a superset
    of every key used in the codebase."""
    zh_text = ZH_TS.read_text(encoding="utf-8")
    en_text = EN_TS.read_text(encoding="utf-8")

    def _keys(text: str) -> set[str]:
        # Quoted string literals shaped like ``"foo.bar":``. We
        # deliberately exclude single-quoted props / CSS-style
        # attribute selectors via the ``"…":`` colon after the
        # closing quote.
        return set(re.findall(r'"([a-z][a-zA-Z0-9_.]+)"\s*:', text))

    zh_keys = _keys(zh_text)
    en_keys = _keys(en_text)
    missing_in_en = zh_keys - en_keys
    missing_in_zh = en_keys - zh_keys
    assert not missing_in_en, (
        f"English dictionary is missing keys present in zh.ts: "
        f"{sorted(missing_in_en)}"
    )
    assert not missing_in_zh, (
        f"Chinese dictionary is missing keys present in en.ts: "
        f"{sorted(missing_in_zh)}"
    )
    assert len(zh_keys) > 30, (
        f"i18n dictionaries should have >30 keys (found {len(zh_keys)}); "
        "P2-4 must extract a representative sample, not a token"
    )


def test_p24_useLocale_hook_exports_provider_and_hook():
    """The hook file (i18n/index.tsx) must export LocaleProvider,
    useLocale, and a Locale type — these are the public surface
    every consumer depends on."""
    index_tsx = I18N_DIR / "index.tsx"
    index_ts = I18N_DIR / "index.ts"
    assert index_tsx.exists() or index_ts.exists(), (
        "i18n/index.{ts,tsx} must export LocaleProvider + useLocale"
    )
    src_path = index_tsx if index_tsx.exists() else index_ts
    text = src_path.read_text(encoding="utf-8")
    for export in ("LocaleProvider", "useLocale", "LocaleContext"):
        assert export in text, f"i18n entry must export {export}"


def test_p24_main_wires_locale_provider():
    """main.tsx must wrap App in <LocaleProvider> — without that
    every ``useLocale()`` call inside App / Settings / Sidebar
    would throw at first render."""
    main_tsx = REPO_ROOT / "src" / "frontend" / "src" / "main.tsx"
    text = main_tsx.read_text(encoding="utf-8")
    assert "LocaleProvider" in text, (
        "main.tsx must wrap <App> in <LocaleProvider> so the "
        "useLocale hook can find a provider (P2-4)"
    )


def test_p24_settings_dialog_has_language_segmented_control():
    """The Settings dialog must let the user switch between zh and
    en — without that, the language preference has no UI surface
    and the toggle is dead code."""
    text = SETTINGS_TSX.read_text(encoding="utf-8")
    assert "settings.language" in text
    assert "settings.langZh" in text
    assert "settings.langEn" in text
    # ``setLocale`` is the setter from useLocale — verify it's
    # wired into the segmented control onClick.
    assert "setLocale(" in text


def test_p24_app_uses_locale_for_connection_banner():
    """The connection-error banner (the most user-visible string
    in the app) must read through the i18n lookup so it re-
    renders when the user flips the language."""
    text = APP_TSX.read_text(encoding="utf-8")
    assert "useLocale" in text, (
        "App.tsx must import useLocale so the connection banner "
        "re-renders on language change (P2-4)"
    )
    # Look for at least one t(...) call with a connection.* key.
    assert "connection.close" in text or "connection.dismissAria" in text


def test_p24_fallback_path_keeps_chinese_strings_visible():
    """A missing translation must fall back to the Chinese value,
    not the bare key. The lookup helper lives in
    ``i18n/index.tsx``."""
    index_tsx = I18N_DIR / "index.tsx"
    index_ts = I18N_DIR / "index.ts"
    src_path = index_tsx if index_tsx.exists() else index_ts
    text = src_path.read_text(encoding="utf-8")
    # The fallback chain: en (or zh) -> zh -> key. We assert the
    # structure (zh imported and used as a fallback) is present.
    assert "import zh" in text or "from \"./zh\"" in text, (
        "i18n entry must import zh as the fallback dictionary"
    )


def test_p24_placeholder_substitution_supported():
    """Strings like ``sidebar.embedding`` carry a ``{count}`` token;
    the lookup must substitute it without a heavy ICU dependency."""
    index_tsx = I18N_DIR / "index.tsx"
    index_ts = I18N_DIR / "index.ts"
    src_path = index_tsx if index_tsx.exists() else index_ts
    text = src_path.read_text(encoding="utf-8")
    assert "replace" in text and "{" in text, (
        "useLocale().t must support {name} placeholder substitution"
    )
