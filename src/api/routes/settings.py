"""Settings endpoints — provider, model selection.

API keys are NOT managed here. They come from environment variables
(``MINIMAX_API_KEY`` / ``MINIMAX_BASE_URL`` / ``MINIMAX_MODEL``) so the
deployment can rotate credentials without touching the running app.
"""
from __future__ import annotations

from fastapi import APIRouter

from config.settings import (
    get_settings,
    load_user_config,
    reset_settings_cache,
    save_user_config,
)
from src.api.schemas import SettingsRequest, SettingsResponse
from src.core.logging import logger
from src.core.paths import user_data_dir
from src.llm.factory import get_api_key


router = APIRouter(prefix="/settings", tags=["settings"])


def _key_source() -> tuple[bool, str]:
    """Whether a usable API key is configured, and where it came from."""
    s = get_settings()
    if s.llm_api_key:
        return True, "env"
    # Fall back to keyring for the legacy path
    if get_api_key(s.llm_provider, s.keyring_service):
        return True, "keyring"
    return False, "none"


@router.get("", response_model=SettingsResponse)
async def get_settings_endpoint() -> SettingsResponse:
    s = get_settings()
    has_key, source = _key_source()
    # Show the effective model: user choice > env override > default
    from config.constants import DEFAULT_LLM_MODELS, DEFAULT_LLM_PROVIDER
    effective_model = s.llm_model or s.llm_model_override or DEFAULT_LLM_MODELS[s.llm_provider]
    return SettingsResponse(
        llm_provider=s.llm_provider,
        llm_model=effective_model,
        has_api_key=has_key,
        api_key_source=source,
    )


@router.post("", response_model=SettingsResponse)
async def update_settings(req: SettingsRequest) -> SettingsResponse:
    """Apply a settings update IN PLACE — only the fields the request
    supplies are touched.

    P1-2-fix: the previous implementation constructed a fresh
    ``UserConfig(llm_provider=..., llm_model=...)`` with every save,
    which silently reset ``host`` / ``port`` / ``allow_remote`` to
    defaults every time the user toggled the provider dropdown. We
    now load the existing config and overlay only the supplied fields
    so unrelated knobs survive.
    """
    data_dir = user_data_dir()
    # Load existing config first — this is the incremental-overlay fix.
    # ``None`` is treated as "leave this field alone" so the request
    # shape stays optional even though we always write the whole object.
    existing = load_user_config(data_dir)

    # Track which fields actually changed so the log line is honest.
    changed: list[str] = []

    if req.llm_provider is not None and req.llm_provider != existing.llm_provider:
        existing.llm_provider = req.llm_provider
        changed.append("provider")
    if req.llm_model is not None and req.llm_model != existing.llm_model:
        existing.llm_model = req.llm_model
        changed.append("model")

    if changed:
        save_user_config(data_dir, existing)
        reset_settings_cache()
        logger.info(
            f"Settings updated: {', '.join(changed)} "
            f"(host={existing.host}, port={existing.port}, "
            f"allow_remote={existing.allow_remote})"
        )

    return await get_settings_endpoint()
