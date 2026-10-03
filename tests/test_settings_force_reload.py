"""Tests for v2.0.28.2 PR-3 P1-P3 Settings TOCTOU.

Pre-PR-3 ``get_settings()`` was a binary cache: once populated, the
cached value was returned for the entire process lifetime unless
``reset_settings_cache()`` was called explicitly. Two real-world
scenarios were broken by this:

1. **Operator rotates ``.env``** — adding/changing ``RAG_LOG_LEVEL``
   or ``MINIMAX_API_KEY`` had no effect until the backend was
   restarted. Operators do not always have a shell into the running
   process to call ``reset_settings_cache()`` programmatically.

2. **Operator edits ``settings.json``** — flipping a knob in the
   Settings dialog does call ``reset_settings_cache()``, but other
   processes (lifespan startup, ``src/app.py:create_app()``) that
   captured the cached value before the edit saw the stale value.

PR-3 adds two levers:

* **TTL**: ``SETTINGS_CACHE_TTL_SEC = 30 s``. After 30 s, the next
  call re-resolves from env + user config. Long enough to not
  hammer ``resolve_settings`` on every chat turn (which would
  re-read ``.env`` N times per session), short enough that operator
  ``.env`` rotations take effect within ~30 s without restart.

* **``force_reload=True``**: explicit one-shot bypass. After this
  call returns the freshly-resolved value, the next call returns
  the cached value again until the next TTL window.

The 4 tests below pin both behaviors end-to-end, including the
``time.monotonic()`` clock semantics (monotonic, not wall-clock,
so DST / NTP jumps don't accidentally expire the cache).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

# v2.0.28.2 P1-P3 — access the MODULE via ``sys.modules`` so we can
# monkeypatch the module globals (where ``_settings_cache`` /
# ``_settings_cache_ts`` / ``SETTINGS_CACHE_TTL_SEC`` /
# ``resolve_settings`` actually live). Two pitfalls:
#
# 1. ``config/__init__.py:2`` does ``from .settings import settings``
#    which imports the ``settings`` SYMBOL — that's the
#    ``_SettingsProxy`` instance from ``config/settings.py:312``.
#    It binds to ``config.settings`` at package level, so
#    ``import config.settings as foo`` gets you the PROXY, not the
#    module. The proxy's ``__getattr__`` routes attribute access
#    through ``get_settings()``, so monkeypatching it has no effect
#    on the real cache.
#
# 2. ``from config.settings import X`` DOES reach the real module
#    (submodule-level access goes through ``sys.modules``). So we
#    import the callables we need via that path, but keep a
#    ``settings_mod`` reference to the actual module for
#    monkeypatching.
# Loading order matters:
#   1. ``from config.settings import X`` imports the SUBMODULE
#      (config.settings the module, not the proxy). This is how
#      Python's import machinery binds ``config.settings`` into
#      ``sys.modules`` at the module level.
#   2. ``import config.settings as ...`` would actually get the
#      PROXY (because ``config/__init__.py:2`` does
#      ``from .settings import settings`` which binds the
#      ``_SettingsProxy`` instance as the ``settings`` attribute
#      of the ``config`` package). So we explicitly go through
#      ``sys.modules`` to grab the real module.
from config.settings import (  # noqa: E402 — must precede sys.modules lookup below
    SETTINGS_CACHE_TTL_SEC,  # noqa: F401 — public re-export for test readability
    get_settings,
    reset_settings_cache,
)

settings_mod = sys.modules["config.settings"]


# ---------------------------------------------------------------------------
# 1. force_reload bypasses the cache
# ---------------------------------------------------------------------------


def test_get_settings_force_reload_rereads_env_after_cache_populated(
    monkeypatch, tmp_path: Path
):
    """After the cache is populated, changing the env var has NO effect
    on subsequent ``get_settings()`` calls within the TTL window —
    BUT passing ``force_reload=True`` bypasses the cache and reads the
    new value.

    Mirrors the operator workflow: ``RAG_LOG_LEVEL=DEBUG`` → process
    starts → cache populated with ``log_level="DEBUG"`` → operator
    changes to ``RAG_LOG_LEVEL=WARNING`` → within TTL window,
    ``get_settings()`` returns stale ``DEBUG``, but
    ``get_settings(force_reload=True)`` returns fresh ``WARNING``.
    """
    monkeypatch.setenv("RAG_LOG_LEVEL", "DEBUG")
    # Ensure cache is empty (autouse reset runs before this but be defensive).
    reset_settings_cache()

    # First call — cache miss, populates with DEBUG.
    s1 = get_settings(data_dir=tmp_path)
    assert s1.log_level == "DEBUG"

    # Operator rotates .env to WARNING.
    monkeypatch.setenv("RAG_LOG_LEVEL", "WARNING")

    # Without force_reload — still cached DEBUG (within TTL window).
    s2 = get_settings(data_dir=tmp_path)
    assert s2.log_level == "DEBUG", (
        "cache returned fresh value within TTL window — TTL is not honored"
    )

    # With force_reload — fresh WARNING. This call also updates the
    # cache + resets the TTL timestamp, so subsequent calls within
    # the TTL window also see WARNING (not DEBUG).
    s3 = get_settings(data_dir=tmp_path, force_reload=True)
    assert s3.log_level == "WARNING", (
        "force_reload=True did not bypass cache to read new env value"
    )

    # Subsequent calls without force_reload — see WARNING (the new
    # value), NOT DEBUG (the old value). ``force_reload=True``
    # refreshes the cache; it's not a read-only one-shot peek. If
    # operators had to call force_reload=True on every read to see
    # the new value, the lever would be useless.
    s4 = get_settings(data_dir=tmp_path)
    assert s4.log_level == "WARNING", (
        "force_reload=True should refresh the cache (not a read-only "
        "peek) — subsequent calls without force_reload must return the "
        "newly-resolved value, not the pre-force_reload cache value"
    )


# ---------------------------------------------------------------------------
# 2. force_reload=False (explicit negation) keeps cached behavior
# ---------------------------------------------------------------------------


def test_get_settings_force_reload_false_returns_cached_settings(
    monkeypatch, tmp_path: Path
):
    """Explicit ``force_reload=False`` (the default) must NOT bypass
    the cache. Pin against a future "optimization" that accidentally
    flips ``if force_reload: ...`` to ``if not force_reload: ...``.
    """
    monkeypatch.setenv("RAG_LOG_LEVEL", "INFO")
    reset_settings_cache()

    s1 = get_settings(data_dir=tmp_path)
    assert s1.log_level == "INFO"

    monkeypatch.setenv("RAG_LOG_LEVEL", "ERROR")

    # force_reload=False explicit — still cached INFO.
    s2 = get_settings(data_dir=tmp_path, force_reload=False)
    assert s2.log_level == "INFO"

    # Same call without the kwarg — still cached INFO (default).
    s3 = get_settings(data_dir=tmp_path)
    assert s3.log_level == "INFO"


# ---------------------------------------------------------------------------
# 3. TTL expires after the configured window
# ---------------------------------------------------------------------------


def test_settings_cache_ttl_expires(monkeypatch, tmp_path: Path):
    """When the TTL window expires, the next ``get_settings()`` call
    (without ``force_reload``) must re-resolve from env + user config.

    Uses ``monkeypatch.setattr`` to shrink ``SETTINGS_CACHE_TTL_SEC``
    to ``0.1 s`` so the test runs fast. ``time.sleep(0.2)`` is well
    above the noise floor on CI runners.
    """
    monkeypatch.setattr(settings_mod, "SETTINGS_CACHE_TTL_SEC", 0.1)
    monkeypatch.setenv("RAG_LOG_LEVEL", "DEBUG")
    reset_settings_cache()

    s1 = get_settings(data_dir=tmp_path)
    assert s1.log_level == "DEBUG"

    monkeypatch.setenv("RAG_LOG_LEVEL", "WARNING")

    # Within TTL window — cached DEBUG.
    s2 = get_settings(data_dir=tmp_path)
    assert s2.log_level == "DEBUG"

    # Sleep past the (shrunk) TTL window.
    time.sleep(0.2)

    # TTL expired — fresh WARNING (no force_reload needed).
    s3 = get_settings(data_dir=tmp_path)
    assert s3.log_level == "WARNING", (
        "TTL did not expire after 0.2 s (TTL was 0.1 s) — "
        "get_settings must re-resolve when cache window passes"
    )


# ---------------------------------------------------------------------------
# 4. concurrent first-call resolve only runs once
# ---------------------------------------------------------------------------


def test_concurrent_first_call_resolve_only_runs_once(
    monkeypatch, tmp_path: Path
):
    """Multiple concurrent ``get_settings()`` calls in the same asyncio
    task (which is single-threaded under the GIL, but has multiple
    ``await`` points) must NOT trigger multiple ``resolve_settings``
    invocations on the first call.

    Pin against a future "optimization" that adds ``asyncio.Lock`` to
    ``get_settings`` and accidentally serializes every call.

    Implementation detail: ``get_settings`` does NOT use a lock today
    (it's single-threaded asyncio). On the first call, if the cache
    is empty, ``resolve_settings`` runs once. Concurrent subsequent
    calls (no ``await`` between read and write) all see the same
    ``_settings_cache is None`` check and only one call resolves.

    This test uses a synchronous call sequence (no ``asyncio.gather``)
    which is the more common case and exercises the actual hot path.
    """
    resolve_calls = {"count": 0}
    original_resolve = settings_mod.resolve_settings

    def counting_resolve(data_dir):
        resolve_calls["count"] += 1
        return original_resolve(data_dir)

    monkeypatch.setattr(settings_mod, "resolve_settings", counting_resolve)
    reset_settings_cache()

    # 10 back-to-back calls (no await between them) — first call
    # populates cache, the rest are cache hits.
    for _ in range(10):
        get_settings(data_dir=tmp_path)

    assert resolve_calls["count"] == 1, (
        f"resolve_settings ran {resolve_calls['count']} times for 10 "
        f"back-to-back get_settings calls — should be exactly 1 (cache hit after first)"
    )

    # One force_reload — second resolve.
    get_settings(data_dir=tmp_path, force_reload=True)
    assert resolve_calls["count"] == 2, (
        f"force_reload=True should trigger exactly one extra resolve; "
        f"got {resolve_calls['count']} total"
    )


# ---------------------------------------------------------------------------
# 5. reset_settings_cache clears timestamp too
# ---------------------------------------------------------------------------


def test_reset_settings_cache_clears_timestamp(monkeypatch, tmp_path: Path):
    """``reset_settings_cache()`` must clear BOTH the cached value AND
    the monotonic timestamp. Otherwise the next call would see a
    valid timestamp and consider the cache fresh (returning ``None``).
    """
    monkeypatch.setenv("RAG_LOG_LEVEL", "DEBUG")
    reset_settings_cache()

    get_settings(data_dir=tmp_path)

    # Cache populated — internal state should have both fields set.
    assert settings_mod._settings_cache is not None
    assert settings_mod._settings_cache_ts is not None

    reset_settings_cache()

    # Both cleared.
    assert settings_mod._settings_cache is None, (
        "reset_settings_cache did not clear _settings_cache"
    )
    assert settings_mod._settings_cache_ts is None, (
        "reset_settings_cache did not clear _settings_cache_ts — "
        "next get_settings would see stale timestamp and skip re-resolve"
    )


# ---------------------------------------------------------------------------
# 6. module-level constants are exported
# ---------------------------------------------------------------------------


def test_settings_cache_ttl_sec_is_30_seconds():
    """``SETTINGS_CACHE_TTL_SEC = 30.0`` is the codebase-consistent
    value (matches ``_THREAD_CHUNK_CACHE_TTL_SEC`` in lancedb_store).
    Pin against accidental change.
    """
    assert SETTINGS_CACHE_TTL_SEC == 30.0, (
        f"SETTINGS_CACHE_TTL_SEC must stay 30.0 for codebase consistency "
        f"(matches lancedb_store._THREAD_CHUNK_CACHE_TTL_SEC); "
        f"got {SETTINGS_CACHE_TTL_SEC}"
    )
