"""Pydantic Settings for runtime configuration.

Reads from environment variables (.env file supported) AND a per-user
settings.json under the user data directory. The JSON file overrides env.

Env-var naming convention
-------------------------
Each Settings field uses an explicit ``validation_alias`` instead of a
global ``env_prefix``, so the names of the env vars that control each
piece of config are right there in the field declaration. We ended up
here after the audit flagged:

* An earlier version used ``env_prefix="RAG_"`` for *every* field AND
  separate ``validation_alias="MINIMAX_*"`` for the LLM credentials.
  That made the prefix convention inconsistent — ``RAG_HOST`` and
  ``RAG_PORT`` existed alongside ``MINIMAX_API_KEY`` and
  ``MINIMAX_BASE_URL``, and ops had to read source to know which one a
  given field used. Worse, the per-field alias silently overrode the
  prefix for those three fields, so the prefix declaration was
  misleading.

* ``RAG_DATA_DIR`` is also read directly by ``src.core.paths`` (which
  predates the Settings system and resolves the user-data root on its
  own). Reading the same var from two places — once through pydantic's
  prefix scan and once through ``os.environ.get`` — works but means
  the source of truth for that one var isn't pinned.

The fix: drop the global ``env_prefix`` and use explicit per-field
aliases. The fields that should be env-driven (log level, LLM
credentials, base URL, model override) get explicit
``validation_alias`` declarations. The fields that are user-only
(host / port / allow_remote / llm_provider / llm_model /
llm_cheap_model) are stored in ``settings.json`` under the user data
dir and have NO env alias — flipping them via .env is not supported,
which matches the intent (server bind / model choice are per-deployment,
per-user).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .constants import (
    CONFIG_FILE,
    DEFAULT_HOST,
    DEFAULT_LLM_MODELS,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_PORT,
    KEYRING_SERVICE,
)


class Settings(BaseSettings):
    """Runtime configuration.

    Resolution order: defaults → env vars (per-field alias) → user
    config.json overlay (applied by :func:`resolve_settings`).

    LLM credentials (``llm_api_key``, ``llm_base_url``,
    ``llm_model_override``) are env-driven (no JSON persistence) so
    deployments can inject them via ``.env`` / container secrets / CI
    variables — no keyring write UI required. If ``llm_api_key`` is
    unset, :func:`src.llm.factory._resolve_api_key` falls back to the
    keyring for the legacy path.
    """

    # ``env_prefix`` deliberately empty — every env-driven field has
    # an explicit ``validation_alias`` so the prefix convention is
    # documented per-field rather than implicit. See the module
    # docstring for why.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Server (user-only, no env alias) ----
    # The host/port/allow_remote triple is per-deployment, not per-user,
    # but in practice ops doesn't flip it via env — the deployment
    # hard-codes it in the systemd unit / container command. Users
    # can change it via the Settings dialog, which persists to
    # settings.json. We deliberately don't accept a RAG_* env alias
    # for these to avoid the "double source of truth" trap.
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    allow_remote: bool = False  # If True and host != 127.0.0.1, accept external connections

    # ---- LLM (user-selected) ----
    llm_provider: Literal["openai", "anthropic"] = DEFAULT_LLM_PROVIDER
    llm_model: str = DEFAULT_LLM_MODELS[DEFAULT_LLM_PROVIDER]

    # Small / fast model used by route / rewrite / grade /
    # hallucination nodes (called via ``build_cheap_model``). Previously
    # hardcoded to ``claude-3-5-haiku-latest`` in factory.py, which
    # silently ran on a model the deployment didn't actually
    # configure — the "fail-open / smart-quote repair / truncation
    # recovery" code paths were all workarounds for that mismatch.
    # Now this is per-user-settable so a deployment pointing at the
    # MiniMax proxy can name the cheap model the proxy actually
    # supports (often the same as the main model). ``None`` ⇒ fall
    # back to the per-provider default.
    llm_cheap_model: Optional[str] = None

    # ---- LLM (env-driven, takes precedence) ----
    # ``validation_alias`` declarations document which env var name
    # each credential lives under. Previously this used the global
    # ``env_prefix="RAG_"`` plus per-field overrides, which made it
    # impossible to know from the field name which prefix to expect.
    llm_api_key: Optional[str] = Field(default=None, validation_alias="MINIMAX_API_KEY")
    llm_base_url: Optional[str] = Field(default=None, validation_alias="MINIMAX_BASE_URL")
    llm_model_override: Optional[str] = Field(default=None, validation_alias="MINIMAX_MODEL")

    # ---- Logging (env-driven) ----
    # ``RAG_LOG_LEVEL`` is the deployment knob; the user-data-dir
    # doesn't override it (logs aren't a per-user concept).
    log_level: str = Field(default="INFO", validation_alias="RAG_LOG_LEVEL")

    # ---- Keyring (fallback only — used when MINIMAX_API_KEY is unset) ----
    keyring_service: str = KEYRING_SERVICE


# ---- User config (JSON) ---------------------------------------------------

_USER_CFG_FILENAME = "settings.json"


class UserConfig(BaseModel):
    """Persisted user-level config. Lives in the user data dir.

    P1-2-fix note: the previous implementation constructed ``UserConfig``
    with ONLY the fields the request supplied (``UserConfig(
    llm_provider=..., llm_model=...)``), which silently reset
    host/port/allow_remote to defaults every time the user toggled the
    provider dropdown. The new endpoint code loads the existing
    config first and overlays only the supplied fields, so changing
    one knob doesn't wipe the others.
    """

    llm_provider: Literal["openai", "anthropic"] = DEFAULT_LLM_PROVIDER
    llm_model: str = DEFAULT_LLM_MODELS[DEFAULT_LLM_PROVIDER]
    llm_cheap_model: Optional[str] = None
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    allow_remote: bool = False


def user_config_path(data_dir: Path) -> Path:
    return data_dir / CONFIG_FILE.split("/", 1)[0] / _USER_CFG_FILENAME


def load_user_config(data_dir: Path) -> UserConfig:
    """Load user config from JSON; create default if missing.

    Tolerates partial / extra fields by passing through ``**`` —
    forward-compat: a NEW field saved by a newer version won't crash
    an older binary that's behind on a deploy.
    """
    path = user_config_path(data_dir)
    if not path.exists():
        return UserConfig()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return UserConfig()
        return UserConfig(**raw)
    except Exception:
        # Corrupted JSON; fall back to defaults
        return UserConfig()


def save_user_config(data_dir: Path, cfg: UserConfig) -> Path:
    """Persist user config; ensure parent dir exists."""
    path = user_config_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cfg.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def resolve_settings(data_dir: Path) -> Settings:
    """Build final Settings: defaults + env (per-field aliases) + user config overlay.

    User config is the source of truth for the fields it owns
    (host/port/allow_remote/llm_provider/llm_model/llm_cheap_model);
    env vars are the source of truth for the fields we don't persist
    (LLM credentials, log level, keyring service). The two never
    overlap — each field belongs to exactly one channel.
    """
    base = Settings()
    user_cfg = load_user_config(data_dir)
    return Settings(
        host=user_cfg.host,
        port=user_cfg.port,
        allow_remote=user_cfg.allow_remote,
        llm_provider=user_cfg.llm_provider,
        llm_model=user_cfg.llm_model,
        llm_cheap_model=user_cfg.llm_cheap_model,
        # Env-driven fields flow through untouched
        llm_api_key=base.llm_api_key,
        llm_base_url=base.llm_base_url,
        llm_model_override=base.llm_model_override,
        log_level=base.log_level,
        keyring_service=base.keyring_service,
    )


# ---- Singleton-style accessor (resolved later when data_dir is known) ----

# v2.0.28.2 P1-P3 — TTL cache with explicit ``force_reload`` escape
# hatch. Pre-PR-3 the cache was a binary "once populated, never refresh
# until reset_settings_cache() is called" — operator rotating ``.env``
# or editing ``settings.json`` saw the stale value for the entire
# process lifetime. Two new levers:
#
#   1. TTL: ``SETTINGS_CACHE_TTL_SEC = 30.0`` — every 30 s the next
#      call re-resolves from env + user config. 30 s matches
#      ``src.storage.lancedb_store._THREAD_CHUNK_CACHE_TTL_SEC`` for
#      codebase consistency. Long enough to not hammer ``resolve_settings``
#      on every chat turn, short enough that an operator rotating
#      ``.env`` sees the new value within ~30 s without restart.
#
#   2. ``force_reload=True`` — explicit cache refresh. The call
#      re-resolves from env + user config AND updates the cache +
#      resets the TTL timestamp. Subsequent calls (within the new
#      TTL window) also see the freshly-resolved value, not the
#      pre-force_reload cache. This is the operator escape hatch
#      for "I just rotated .env and need it NOW" without bouncing
#      the whole backend — operator calls ``get_settings(force_reload=True)``
#      once, and every subsequent read in the next 30 s sees the new
#      value too. (Originally conceived as a "read-only peek", but
#      the operator workflow needs the refreshed cache to stick —
#      otherwise they'd have to call force_reload=True on every
#      read, which is no better than not caching at all.)
#
# Multi-worker uvicorn (future): each worker has its own cache +
# timestamp; they're not coordinated. Single-worker is the current
# deployment model so this is acceptable. Multi-worker would need
# either a file watcher on .env / settings.json or a Redis-backed
# pub/sub — out of scope for PR-3.
SETTINGS_CACHE_TTL_SEC: float = 30.0

_settings_cache: Optional[Settings] = None
# v2.0.28.2 P1-P3 — monotonic timestamp of when the cache was last
# populated. ``None`` means "never populated" (next call resolves
# fresh). ``time.monotonic()`` (not wall-clock) so DST / NTP jumps
# don't accidentally expire or extend the cache window.
_settings_cache_ts: Optional[float] = None


def get_settings(
    data_dir: Optional[Path] = None,
    *,
    force_reload: bool = False,
) -> Settings:
    """Resolve and cache settings.

    Pass ``data_dir`` on first call. Subsequent calls return the
    cached value until either:

    * the TTL window (``SETTINGS_CACHE_TTL_SEC = 30 s``) expires
      — the next call re-resolves from env + user config.
    * ``force_reload=True`` is passed — the call re-resolves and
      updates the cache (subsequent reads within the TTL window
      also see the freshly-resolved value).

    Pass ``force_reload=True`` to bypass the TTL AND refresh the
    cache. Operator use case: rotate ``.env`` → call
    ``get_settings(force_reload=True)`` once → every subsequent
    read in the next 30 s sees the new value.

    Thread-safety: this function is NOT thread-safe. Python's GIL
    makes the simple ``_settings_cache`` read-then-assign safe for
    asyncio code (no preemption between the read and the assign
    unless we ``await``), but multiple OS threads calling
    ``get_settings`` concurrently could race. In practice YuanRAG
    uses single-worker uvicorn + asyncio throughout — no OS-thread
    access to ``get_settings`` exists. If multi-thread access is
    added in the future, wrap the body in a ``threading.Lock``.
    """
    global _settings_cache, _settings_cache_ts
    now = time.monotonic()
    cache_stale = (
        _settings_cache is None
        or _settings_cache_ts is None
        or (now - _settings_cache_ts) >= SETTINGS_CACHE_TTL_SEC
    )
    if force_reload or cache_stale:
        if data_dir is None:
            from src.core.paths import user_data_dir

            data_dir = user_data_dir()
        _settings_cache = resolve_settings(data_dir)
        _settings_cache_ts = now
    return _settings_cache


def reset_settings_cache() -> None:
    """Clear the cache AND timestamp.

    Used by settings updates (e.g. when the user toggles a knob in
    the Settings dialog) AND by test isolation (autouse fixture in
    ``tests/conftest.py``). After this call the next ``get_settings``
    call is a cache miss and re-resolves from env + user config.
    """
    global _settings_cache, _settings_cache_ts
    _settings_cache = None
    _settings_cache_ts = None


# Expose `settings` as a lazy proxy. Importing modules should call
# `get_settings(data_dir)` from lifespan — this proxy keeps module imports
# side-effect free.
class _SettingsProxy:
    def __getattr__(self, name: str):
        return getattr(get_settings(), name)

    def __repr__(self) -> str:
        return repr(get_settings())


settings = _SettingsProxy()