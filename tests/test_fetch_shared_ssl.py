"""v2.0.28.5 Item 11 PR-3 P1-R3 — shared SSL context.

Pre-PR-3, ``_make_ssl_context()`` was rebuilt on every
``_fetch_url_sync`` call (~0.5-2 ms each: ``ssl.create_default_context``
+ 2 attribute sets). A typical 5-URL page enrichment paid that
cost 5×. Post-PR-3, the context is cached at module scope and
returned by identity on subsequent calls.

This file pins the cache behavior + back-compat (callers still get
a usable ``ssl.SSLContext``).
"""
from __future__ import annotations

import ssl

from src.web_search import fetch as fetch_mod


def test_make_ssl_context_returns_cached_identity():
    """Two calls to ``_make_ssl_context()`` must return the SAME
    instance (identity, not equality).

    We use ``is`` to verify the module-level cache is wired up —
    not a fresh rebuild. Pre-PR-3, every call created a new object
    (different ``id()``); post-PR-3, the cache returns the same one.
    """
    ctx_a = fetch_mod._make_ssl_context()
    ctx_b = fetch_mod._make_ssl_context()
    assert ctx_a is ctx_b, "expected cached identity, got fresh object"


def test_make_ssl_context_settings_preserved():
    """The shared context must still set ``check_hostname=False`` +
    ``verify_mode=CERT_NONE`` (the original permissive config).

    Caching the context must not lose these settings — that's the
    whole reason we use ``_make_ssl_context`` instead of stdlib's
    default-verify ``ssl.create_default_context``.
    """
    ctx = fetch_mod._make_ssl_context()
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.check_hostname is False
    assert ctx.verify_mode == ssl.CERT_NONE


def test_shared_ssl_context_module_attribute():
    """The module-level ``_SHARED_SSL_CONTEXT`` attribute must
    exist (cache target) and be the same instance as what
    ``_make_ssl_context()`` returns.

    Direct attribute check guards against accidental renames or
    accidental fresh-build in the function body.
    """
    cached = getattr(fetch_mod, "_SHARED_SSL_CONTEXT", None)
    assert cached is not None, "_SHARED_SSL_CONTEXT not initialized"
    assert cached is fetch_mod._make_ssl_context()


def test_bing_uses_shared_ssl_context():
    """``bing.py`` imports ``_make_ssl_context`` and uses it for the
    urllib SSL context. After v2.0.28.5, Bing's ``_fetch_one`` also
    gets the shared instance.

    We don't actually call ``_fetch_one`` (would hit the network);
    we just verify the import resolves to the same function and
    that it returns the shared context.
    """
    from src.web_search import bing

    ctx_via_bing = bing._make_ssl_context()
    ctx_via_fetch = fetch_mod._make_ssl_context()
    assert ctx_via_bing is ctx_via_fetch, (
        "Bing should use the shared SSL context, not its own copy"
    )
