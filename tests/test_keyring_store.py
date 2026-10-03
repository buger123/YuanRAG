"""v2.0.28.6 Item 11 PR-4 — ``keyring_store.reset_for_tests`` + CI-safe behavior.

Pre-PR-4, the two module-level singletons in
``src/security/keyring_store.py`` (``_openai`` / ``_anthropic``)
were never reset between tests. A test that swapped
``config.KEYRING_SERVICE`` to a per-test value would keep getting
the previous test's store instance — a classic context-interference
bug. On CI runners without a keyring backend, ``keyring.get_password``
raises ``KeyringError`` which the store caught and returned None
from, but the SWALLOW path was untested.

Post-PR-4:
  - ``reset_for_tests()`` drops both module globals so the next
    ``openai_store(...)`` / ``anthropic_store(...)`` call rebuilds
    the singleton against the current ``KEYRING_SERVICE``.
  - Tests pin the swallow-KeyringError contract so a regression
    that let the exception propagate wouldn't break CI.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _reset_keyring_singletons():
    """Clean module-level singletons before AND after each test."""
    from src.security import keyring_store

    keyring_store.reset_for_tests()
    yield
    keyring_store.reset_for_tests()


def test_keyring_store_set_get_delete_happy_path():
    """Round-trip a value through the store when the keyring backend
    works. We mock ``keyring.set_password`` / ``get_password`` /
    ``delete_password`` so the test doesn't depend on a real OS
    credential store.
    """
    from src.security import keyring_store

    service = "test-svc-happy"
    store = keyring_store.openai_store(service)

    with patch("src.security.keyring_store.keyring") as mock_kr:
        mock_kr.set_password.return_value = None
        mock_kr.get_password.return_value = "sk-test-12345"
        mock_kr.delete_password.return_value = None

        # set returns True on no exception.
        assert store.set("openai_api_key", "sk-test-12345") is True
        mock_kr.set_password.assert_called_with(
            service, "openai_api_key", "sk-test-12345"
        )

        # get returns the value.
        assert store.get("openai_api_key") == "sk-test-12345"
        mock_kr.get_password.assert_called_with(service, "openai_api_key")

        # delete returns True.
        assert store.delete("openai_api_key") is True
        mock_kr.delete_password.assert_called_with(service, "openai_api_key")


def test_keyring_store_swallows_keyring_error_on_set():
    """``store.set`` must return False (not propagate) when the
    keyring backend raises ``KeyringError``.

    On CI runners without an interactive keyring backend,
    ``keyring.set_password`` raises ``KeyringError`` — the store
    must catch it and return False so the caller's UI can show
    "couldn't save" rather than crashing.
    """
    from keyring.errors import KeyringError

    from src.security import keyring_store

    store = keyring_store.anthropic_store("test-svc-swallows")

    with patch(
        "src.security.keyring_store.keyring.set_password",
        side_effect=KeyringError("no backend"),
    ):
        assert store.set("anthropic_api_key", "sk-test") is False


def test_keyring_store_swallows_keyring_error_on_get():
    """``store.get`` must return None (not propagate) when the
    keyring backend raises ``KeyringError``.
    """
    from keyring.errors import KeyringError

    from src.security import keyring_store

    store = keyring_store.anthropic_store("test-svc-get-swallows")

    with patch(
        "src.security.keyring_store.keyring.get_password",
        side_effect=KeyringError("no backend"),
    ):
        assert store.get("anthropic_api_key") is None


def test_keyring_store_reset_for_tests_clears_globals():
    """After ``reset_for_tests()``, the module-level singletons are None.

    This pins the test-isolation contract — a subsequent
    ``openai_store("different-service")`` must build a new store,
    not return the cached one with the OLD service.
    """
    from src.security import keyring_store

    # Force creation of both singletons.
    s1 = keyring_store.openai_store("svc-A")
    s2 = keyring_store.anthropic_store("svc-A")
    assert keyring_store._openai is not None
    assert keyring_store._anthropic is not None

    keyring_store.reset_for_tests()

    assert keyring_store._openai is None
    assert keyring_store._anthropic is None

    # After reset, a new call rebuilds the singleton.
    s1_new = keyring_store.openai_store("svc-A")
    assert s1_new is not s1  # different instance, fresh service bind