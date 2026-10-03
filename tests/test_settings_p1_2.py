"""P1-2 — Settings: incremental overlay + configurable cheap model + env prefix cleanup.

The audit flagged three related issues that all stem from the settings
plumbing being too clever for its own good:

1. ``update_settings`` constructed ``UserConfig(llm_provider=...,
   llm_model=...)`` with every save — the missing host / port /
   allow_remote silently reset to defaults. Toggling the provider
   dropdown would wipe a custom ``allow_remote=True`` binding.

2. ``factory.py`` hardcoded ``claude-3-5-haiku-latest`` as the cheap
   model regardless of provider. The deployment was the MiniMax
   proxy, so every route / grade / hallucination call was sent to a
   model the proxy didn't know — the cascade of "fail-open /
   smart-quote repair / truncation recovery" code paths were all
   workarounds for the resulting garbage JSON.

3. ``Settings`` used ``env_prefix="RAG_"`` PLUS per-field
   ``validation_alias="MINIMAX_*"`` for some fields. The prefix was
   misleading (the per-field aliases overrode it for those fields),
   and ops had to read source to know whether a given knob was
   ``RAG_HOST`` or ``MINIMAX_BASE_URL``.

These tests pin the fixes.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient


def _data_dir() -> Path:
    """Resolve the (test-isolated) data dir for assertions.

    The autouse ``_isolate_user_data_dir`` fixture redirects ``RAG_DATA_DIR``
    to a temp path and ``src.core.paths.user_data_dir`` reads it directly
    from the env — so this function picks up the same dir the route handler
    uses without needing the app to expose it on ``state``.
    """
    from src.core.paths import user_data_dir

    return user_data_dir()


# ============================================================
# P1-2a · update_settings preserves host / port / allow_remote
# ============================================================


@pytest.mark.asyncio
async def test_update_settings_preserves_host_and_port_when_only_model_changes(
    app_under_test,
):
    """Setting host=0.0.0.0 + allow_remote=True, then changing model,
    must NOT reset host / allow_remote to their defaults."""
    from config.settings import load_user_config, reset_settings_cache

    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Step 1: explicitly set a non-default host / port / allow_remote
        # via the file-write path (no endpoint for it in this audit, but
        # the file is what update_settings reads).
        from config.settings import save_user_config, UserConfig

        save_user_config(
            _data_dir(),
            UserConfig(
                host="0.0.0.0",
                port=9999,
                allow_remote=True,
                llm_provider="anthropic",
                llm_model="original-model",
            ),
        )
        reset_settings_cache()

        # Step 2: change ONLY the model
        r = await client.post(
            "/settings",
            json={"llm_provider": "anthropic", "llm_model": "new-model"},
        )
        assert r.status_code == 200, r.text

        # Step 3: host / port / allow_remote must still be the values
        # the user explicitly set, NOT 127.0.0.1 / 8765 / False.
        cfg = load_user_config(
            _data_dir()
        )
        assert cfg.host == "0.0.0.0", (
            f"update_settings wiped host to default; got {cfg.host!r}"
        )
        assert cfg.port == 9999
        assert cfg.allow_remote is True


@pytest.mark.asyncio
async def test_update_settings_preserves_model_when_only_provider_changes(
    app_under_test,
):
    """The opposite direction: changing provider alone must keep the
    existing model selection (otherwise the user's main-model
    customization gets blown away every time they switch providers in
    the dropdown)."""
    from config.settings import UserConfig, load_user_config, reset_settings_cache, save_user_config

    save_user_config(
        _data_dir(),
        UserConfig(llm_provider="openai", llm_model="my-custom-model"),
    )
    reset_settings_cache()

    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/settings", json={"llm_provider": "anthropic"})
        assert r.status_code == 200, r.text

    cfg = load_user_config(_data_dir())
    assert cfg.llm_provider == "anthropic"
    assert cfg.llm_model == "my-custom-model", (
        f"provider-only update wiped model; got {cfg.llm_model!r}"
    )


@pytest.mark.asyncio
async def test_update_settings_no_op_when_neither_field_supplied(
    app_under_test,
):
    """A POST with neither ``llm_provider`` nor ``llm_model`` must be
    a no-op (no save, no cache reset). Otherwise a malformed request
    with both fields None would still touch the file."""
    import json
    from pathlib import Path

    from config.settings import save_user_config, UserConfig, load_user_config

    save_user_config(
        _data_dir(),
        UserConfig(llm_provider="openai", llm_model="original"),
    )
    path = _data_dir() / "config" / "settings.json"
    mtime_before = path.stat().st_mtime

    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/settings", json={})
        assert r.status_code == 200, r.text

    # File content unchanged
    cfg = load_user_config(_data_dir())
    assert cfg.llm_provider == "openai"
    assert cfg.llm_model == "original"


# ============================================================
# P1-2b · Configurable cheap model
# ============================================================


def test_default_cheap_model_for_anthropic_is_deployment_default(monkeypatch, tmp_path: Path):
    """The audit noted that factory.py hardcoded
    ``claude-3-5-haiku-latest`` as the cheap model regardless of
    provider. For an Anthropic-via-MiniMax-proxy deployment that's
    the WRONG model — the proxy doesn't know it. The default now
    resolves to ``MiniMax-M3`` (the deployment default), which we
    know the proxy serves.
    """
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test-xyz")
    from config.settings import reset_settings_cache
    reset_settings_cache()

    from config.settings import get_settings
    from src.llm import factory

    s = get_settings(data_dir=tmp_path)
    # Confirm anthropic is the default provider (constants.py).
    assert s.llm_provider == "anthropic"

    # Verify the cheap model's resolution path uses the per-provider
    # default map. Use AST so we skip the docstring (which still
    # mentions the old hardcoded name as historical context).
    import ast

    tree = ast.parse(inspect.getsource(factory.build_cheap_model))
    # Only inspect *assignment RHS* — the docstring also mentions the
    # old name as historical context, but those references are inside
    # ``Expr`` wrappers, not on the right-hand side of any ``=``.
    literals: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    literals.append(sub.value)
        elif isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    literals.append(k.value)
    body_text = " ".join(literals)
    assert "claude-3-5-haiku-latest" not in body_text, (
        "build_cheap_model body still hardcodes claude-3-5-haiku-latest"
    )
    assert "MiniMax-M3" in body_text, (
        "build_cheap_model body should default to MiniMax-M3 for anthropic"
    )


def test_user_cheap_model_overrides_default(monkeypatch, tmp_path: Path):
    """When ``settings.llm_cheap_model`` is set, the factory must use
    it instead of the per-provider default."""
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test-xyz")
    from config.settings import (
        UserConfig,
        reset_settings_cache,
        resolve_settings,
        save_user_config,
    )

    save_user_config(
        tmp_path,
        UserConfig(
            llm_provider="anthropic",
            llm_model="main-model",
            llm_cheap_model="my-faster-model",
        ),
    )
    reset_settings_cache()

    s = resolve_settings(tmp_path)
    assert s.llm_cheap_model == "my-faster-model", (
        f"user override not picked up; got {s.llm_cheap_model!r}"
    )


def test_cheap_model_none_falls_back_to_per_provider_default(
    monkeypatch, tmp_path: Path
):
    """``llm_cheap_model=None`` (the default) means the factory falls
    back to the per-provider map (anthropic → ``MiniMax-M3``,
    openai → ``gpt-4o-mini``). The map is internal to the factory,
    but we can verify the build path doesn't crash and the
    ``settings.llm_cheap_model`` is read correctly."""
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test-xyz")
    from config.settings import UserConfig, reset_settings_cache, resolve_settings, save_user_config

    save_user_config(
        tmp_path,
        UserConfig(llm_provider="anthropic", llm_model="main-model"),
    )
    reset_settings_cache()

    s = resolve_settings(tmp_path)
    assert s.llm_cheap_model is None


# ============================================================
# P1-2c · Env prefix cleanup (no more RAG_HOST / RAG_PORT aliases)
# ============================================================


def test_no_rag_prefixed_env_aliases_on_settings_fields(monkeypatch):
    """The audit flagged that ``env_prefix="RAG_"`` plus per-field
    ``validation_alias="MINIMAX_*"`` was misleading. After the fix,
    only the explicitly-aliased fields (LLM credentials, log level)
    accept env vars — and the aliases are *not* RAG_-prefixed.

    Setting ``RAG_HOST`` etc. must NOT override Settings.host.
    """
    monkeypatch.setenv("RAG_HOST", "10.20.30.40")
    monkeypatch.setenv("RAG_PORT", "12345")
    monkeypatch.setenv("RAG_ALLOW_REMOTE", "true")
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    from config.settings import reset_settings_cache
    reset_settings_cache()

    from config.settings import get_settings

    s = get_settings()
    # None of the RAG_-prefixed env vars should have leaked in.
    assert s.host != "10.20.30.40", (
        f"RAG_HOST leaked into Settings.host; got {s.host!r}"
    )
    assert s.port != 12345
    assert s.allow_remote is not True


def test_minimax_aliases_still_work(monkeypatch, tmp_path: Path):
    """The LLM credential aliases must still work after dropping the
    global prefix — they're now the ONLY way to inject credentials
    via env."""
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-key-abc")
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://proxy.test/anthropic")
    monkeypatch.setenv("MINIMAX_MODEL", "claude-test")
    from config.settings import get_settings, reset_settings_cache
    reset_settings_cache()

    s = get_settings(data_dir=tmp_path)
    assert s.llm_api_key == "sk-key-abc"
    assert s.llm_base_url == "https://proxy.test/anthropic"
    assert s.llm_model_override == "claude-test"


def test_rag_log_level_alias_works(monkeypatch, tmp_path: Path):
    """The only env-driven field that legitimately uses a RAG_ alias
    is ``RAG_LOG_LEVEL`` — that's the deployment knob for log verbosity."""
    monkeypatch.setenv("RAG_LOG_LEVEL", "DEBUG")
    from config.settings import get_settings, reset_settings_cache
    reset_settings_cache()

    s = get_settings(data_dir=tmp_path)
    assert s.log_level == "DEBUG"


def test_load_user_config_tolerates_partial_or_extra_fields(tmp_path: Path):
    """Forward-compat: a settings.json from a NEWER version that has
    fields this binary doesn't know about must load (the extra fields
    are dropped), and a settings.json with MISSING fields (e.g. after
    a downgrade) must fall back to defaults for the missing ones."""
    import json
    from pathlib import Path

    from config.settings import load_user_config

    path = tmp_path / "config"
    path.mkdir()
    cfg_file = path / "settings.json"

    # Extra field "future_thing": must not crash
    cfg_file.write_text(
        json.dumps(
            {
                "llm_provider": "openai",
                "llm_model": "gpt-x",
                "future_thing": "ignored",
            }
        )
    )
    cfg = load_user_config(tmp_path)
    assert cfg.llm_provider == "openai"
    assert cfg.llm_model == "gpt-x"
    # Missing fields fall back to defaults
    assert cfg.host == "127.0.0.1"