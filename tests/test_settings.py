"""Phase 1 — settings + paths + keyring store."""
from __future__ import annotations

from pathlib import Path

from config.constants import AGENT_BUDGETS, DEFAULT_HOST, DEFAULT_PORT, RRF_K
from config.settings import (
    UserConfig,
    get_settings,
    load_user_config,
    resolve_settings,
    save_user_config,
)
from src.core.paths import (
    APP_DIR_NAME,
    history_db_path,
    lancedb_path,
    logs_dir,
    models_dir,
    uploads_dir,
    user_data_dir,
)
from src.security.keyring_store import KeyringStore


def test_constants_sane():
    assert RRF_K == 60
    # v2.0 ReAct rewrite: the legacy per-node budgets
    # (max_iterations / max_retrieval_rounds / max_rewrites) are
    # gone — the graph now has a single recursion budget keyed on
    # ``max_steps``. Pin the surface so a future refactor can't
    # silently drop or rename it without a test failing.
    assert isinstance(AGENT_BUDGETS.get("max_steps"), int)
    assert AGENT_BUDGETS["max_steps"] >= 1
    assert DEFAULT_HOST
    assert DEFAULT_PORT > 0


def test_user_data_dir_uses_env(tmp_path: Path):
    """RAG_DATA_DIR override creates the directory."""
    p = user_data_dir()
    assert p == tmp_path
    assert p.exists()
    assert p.is_dir()


def test_subdirs_created(tmp_path: Path):
    lancedb_path()
    history_db_path()
    models_dir()
    uploads_dir()
    logs_dir()
    # Subdirectories sit directly under user_data_dir() (no `data/`
    # wrapper layer) — see config/constants.py. Note: history.db is a
    # file, not a directory — it's only created lazily on first write.
    for sub in ("lancedb", "models", "uploads", "logs"):
        assert (tmp_path / sub).exists(), f"Missing: {sub}"
    # history.db's parent directory must exist after ensure_all_dirs().
    assert (tmp_path / "history.db").parent.exists()


def test_default_settings(tmp_path: Path):
    s = get_settings()
    assert s.host == DEFAULT_HOST
    assert s.port == DEFAULT_PORT
    assert s.llm_provider in ("openai", "anthropic")
    # Env-driven fields may be None unless the test env set MINIMAX_* vars
    assert s.llm_api_key is None or isinstance(s.llm_api_key, str)
    assert s.llm_base_url is None or s.llm_base_url.startswith("http")
    assert s.llm_model_override is None or isinstance(s.llm_model_override, str)


def test_env_aliases_picked_up(monkeypatch, tmp_path: Path):
    """MINIMAX_API_KEY / MINIMAX_BASE_URL / MINIMAX_MODEL flow through
    ``validation_alias`` into the Settings object even though the model
    uses the RAG_ prefix."""
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test-xyz")
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://example.test/anthropic")
    monkeypatch.setenv("MINIMAX_MODEL", "test-model-7")
    # Clear the singleton so the new env is read
    from config.settings import reset_settings_cache
    reset_settings_cache()
    s = get_settings(data_dir=tmp_path)
    assert s.llm_api_key == "sk-test-xyz"
    assert s.llm_base_url == "https://example.test/anthropic"
    assert s.llm_model_override == "test-model-7"
    reset_settings_cache()


def test_base_url_maps_to_correct_provider_kwarg(monkeypatch, tmp_path: Path):
    """Regression: langchain clients use different param names for the
    custom endpoint — ``openai_api_base`` for ChatOpenAI, ``anthropic_api_url``
    for ChatAnthropic. The factory must translate ``MINIMAX_BASE_URL`` into
    the right kwarg, otherwise the custom endpoint is silently dropped."""
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test-xyz")
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://example.test/anthropic")
    from config.settings import reset_settings_cache
    reset_settings_cache()

    from src.llm import factory

    monkeypatch.setattr(factory, "_resolve_api_key", lambda provider: "sk-test-xyz")

    # Anthropic path
    anthropic_kwargs = factory._base_url_kwargs("anthropic")
    assert anthropic_kwargs == {"anthropic_api_url": "https://example.test/anthropic"}

    # OpenAI path
    openai_kwargs = factory._base_url_kwargs("openai")
    assert openai_kwargs == {"openai_api_base": "https://example.test/anthropic"}

    # When no base URL is set, no kwargs should be returned
    monkeypatch.setenv("MINIMAX_BASE_URL", "")
    reset_settings_cache()
    assert factory._base_url_kwargs("anthropic") == {}
    assert factory._base_url_kwargs("openai") == {}
    reset_settings_cache()


def test_user_config_roundtrip(tmp_path: Path):
    cfg = UserConfig(llm_provider="anthropic", llm_model="claude-3-5-haiku-latest")
    save_user_config(tmp_path, cfg)
    loaded = load_user_config(tmp_path)
    assert loaded.llm_provider == "anthropic"
    assert loaded.llm_model == "claude-3-5-haiku-latest"


def test_resolve_settings_overrides(tmp_path: Path):
    cfg = UserConfig(host="0.0.0.0", port=9999, allow_remote=True)
    save_user_config(tmp_path, cfg)
    s = resolve_settings(tmp_path)
    assert s.host == "0.0.0.0"
    assert s.port == 9999
    assert s.allow_remote is True


def test_keyring_store_roundtrip():
    """Keyring round-trip. May use a different backend in CI, but should not raise."""
    store = KeyringStore(service="rag_assistant_test")
    test_account = "test_account_phase1"
    test_value = "secret-xyz-123"
    try:
        store.set(test_account, test_value)
        got = store.get(test_account)
        assert got == test_value
        assert store.has(test_account)
    finally:
        store.delete(test_account)
