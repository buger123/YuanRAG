"""OS-native secret storage via the `keyring` package.

Uses Windows Credential Manager on Windows, Keychain on macOS,
Secret Service on Linux. Service name is fixed (config.KEYRING_SERVICE).

Account names:
- ``"openai_api_key"``  → OpenAI key
- ``"anthropic_api_key"`` → Anthropic key
"""
from __future__ import annotations

from typing import Optional

import keyring
from keyring.errors import KeyringError, PasswordDeleteError

from src.core.logging import logger


class KeyringStore:
    """Thin wrapper around the `keyring` package."""

    def __init__(self, service: str):
        self.service = service

    def get(self, account: str) -> Optional[str]:
        try:
            value = keyring.get_password(self.service, account)
            return value if value else None
        except KeyringError as exc:
            logger.warning(f"Keyring get failed for {account}: {exc}")
            return None

    def set(self, account: str, value: str) -> bool:
        try:
            keyring.set_password(self.service, account, value)
            return True
        except KeyringError as exc:
            logger.error(f"Keyring set failed for {account}: {exc}")
            return False

    def delete(self, account: str) -> bool:
        try:
            keyring.delete_password(self.service, account)
            return True
        except PasswordDeleteError:
            return False
        except KeyringError as exc:
            logger.warning(f"Keyring delete failed for {account}: {exc}")
            return False

    def has(self, account: str) -> bool:
        return self.get(account) is not None


# ---- Module-level singletons ----

_openai: Optional[KeyringStore] = None
_anthropic: Optional[KeyringStore] = None


def openai_store(service: str) -> KeyringStore:
    global _openai
    if _openai is None or _openai.service != service:
        _openai = KeyringStore(service)
    return _openai


def anthropic_store(service: str) -> KeyringStore:
    global _anthropic
    if _anthropic is None or _anthropic.service != service:
        _anthropic = KeyringStore(service)
    return _anthropic


def reset_for_tests() -> None:
    """Drop the cached module-level store singletons. Tests only.

    The two singletons (``_openai``, ``_anthropic``) hold a ``KeyringStore``
    bound to a ``service`` string — the store creation is cheap (just
    stores the service name in ``self.service``), but the cache means a
    test that swaps ``config.KEYRING_SERVICE`` to a per-test value would
    otherwise keep getting the OLD store. Tests must call this in
    ``_reset_all_singletons`` (in conftest.py) to ensure isolation.

    Called by ``tests/conftest.py:_reset_all_singletons`` between tests.
    """
    global _openai, _anthropic
    _openai = None
    _anthropic = None
