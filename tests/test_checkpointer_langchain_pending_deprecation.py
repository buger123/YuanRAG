"""v2.0.28.11 (polish) — pin suppression of LangChain's
``LangChainPendingDeprecationWarning`` about ``loads(allowed_objects=...)``.

Background
----------
``langchain_core.load.loads`` ships with an ``allowed_objects`` parameter
whose DEFAULT will change in a future release from "permit everything"
to "must be explicit". Until then every call without an explicit value
emits:

    LangChainPendingDeprecationWarning: The default value of
    `allowed_objects` will change in a future version. Pass an explicit
    list of allowed classes (or 'messages' for untrusted input that
    contains only chat messages) to suppress this warning.

v2.0.28.11 originally pinned ``allowed_objects="messages"`` in
``src/storage/checkpointer.py`` because the only chat messages
round-tripped via ``loads()`` in our checkpointer; everything else
(``Source`` Pydantic, custom dicts) was revived separately (see v2.0.23
hotfix for the Source revival path).

v2.0.28.18 P1-P5 — broaden the pin to ``"core"``. Pre-fix, "messages"
rejected the legacy Document envelope
``["langchain","schema","document","Document"]`` (Document is a
langchain_core class but NOT in the messages namespace) → the corrupt
thread was invisible in the sidebar. Verified in REPL that
``allowed_objects="core"`` accepts BOTH legacy + modern IDs for
Document AND all message classes. "core" is the narrowest scope that
covers every langchain_core class we use (messages + documents)
without enabling partner integrations. We pin the contract here to
permit either value, so a future tightening can't silently regress to
"permit everything".

This test pins the contract so a refactor that drops the kwarg and
silently regresses to "permit everything" fails CI immediately.

Note: there is a separate ``LangChainBetaWarning`` (a ``DeprecationWarning``
subclass) about ``loads()`` itself being in beta. That one is NOT
addressable from our side (the function is beta-marked by LangChain)
and is intentionally not pinned here.
"""
from __future__ import annotations

import warnings

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning


class TestCheckpointerPendingDeprecationSuppressed:
    """The checkpointer's ``_deserialize_state`` must call ``loads(...)``
    with an explicit ``allowed_objects`` kwarg to suppress the pending-
    deprecation warning."""

    def test_loads_called_with_explicit_allowed_objects(self):
        """Source pin: ``src/storage/checkpointer.py`` must pass
        ``allowed_objects="<some-explicit-string>"`` to ``loads(...)``.
        Without this kwarg the future LangChain release will tighten the
        default and break our serialization path. The explicit string
        must be one of the narrow langchain_core scopes — currently
        ``"messages"`` (v2.0.28.11, chat-only) or ``"core"``
        (v2.0.28.18, chat + documents). Any other string (e.g. a typo)
        would silently widen the allowlist and fail the contract.
        """
        from pathlib import Path
        path = Path("D:/prep for work/Projects/YuanRAG/src/storage/checkpointer.py")
        text = path.read_text(encoding="utf-8")
        # Locate the ACTUAL loads() call (not docstring references).
        # The call lives inside _deserialize_state as
        # ``return loads(revived_json, allowed_objects="...")``.
        # Docstring references use ``loads()`` (with ``)``) — find the
        # call that uses `(` without an immediately preceding `` ` ``.
        import re
        # Match the call site: ``loads(`` preceded by whitespace and an
        # identifier character (so we skip ```` `` loads()```` docstring).
        matches = list(re.finditer(r"\bloads\(", text))
        assert matches, "checkpointer.py must contain a loads(...) call"
        # The call site is the LAST match (docstring references come first).
        last_match = matches[-1]
        body = text[last_match.start():last_match.start() + 200]
        assert "allowed_objects" in body, (
            "v2.0.28.11 polish -- the indented loads(...) call in "
            "_deserialize_state must pass allowed_objects=<explicit> to "
            "suppress LangChainPendingDeprecationWarning. Without this "
            "kwarg the future LangChain release will tighten the default "
            "and break our serialization path. Body inspected: "
            + repr(body)
        )
        # Pin the value to one of the two acceptable narrow scopes.
        # v2.0.28.18 expanded from "messages" to also accept "core" so
        # Document can round-trip (Document is a langchain_core class
        # but is NOT in the "messages" namespace allowlist).
        acceptable = ('"messages"', "'messages'", '"core"', "'core'")
        assert any(v in body for v in acceptable), (
            "v2.0.28.18 P1-P5 -- allowed_objects must be the narrow "
            "string 'messages' (v2.0.28.11, chat-only) or 'core' "
            "(v2.0.28.18, chat + documents). Other strings would either "
            "break Document deserialization or silently widen the "
            "allowlist. Body inspected: " + repr(body)
        )

    def test_deserialize_state_does_not_emit_pending_deprecation(self, tmp_path, monkeypatch):
        """End-to-end: drive ``_deserialize_state`` through a real
        ``loads()`` round-trip with a ``HumanMessage`` payload and
        assert no ``LangChainPendingDeprecationWarning`` fires.

        Uses ``warnings.catch_warnings`` + ``simplefilter("error", ...)``
        so any future regression raises immediately."""
        from langchain_core.messages import HumanMessage
        from langchain_core.load import dumps

        from src.storage.checkpointer import _deserialize_state

        # Serialize a real HumanMessage — exercises the same code path
        # the checkpointer hits during load_latest().
        serialized = dumps(HumanMessage(content="hi"))

        with warnings.catch_warnings():
            warnings.simplefilter("error", LangChainPendingDeprecationWarning)
            # If the suppression regresses, the filter promotes the
            # warning to an error and _deserialize_state propagates it.
            result = _deserialize_state(serialized, thread_id="t-1")

        # Sanity: round-trip preserved the message. The exact return
        # shape depends on whether the input looked like a single
        # message or a state envelope. We accept any HumanMessage with
        # the expected content anywhere in the result tree.
        from langchain_core.messages import HumanMessage as HM

        def _find_human(node):
            if isinstance(node, HM):
                return node
            if isinstance(node, list):
                for item in node:
                    found = _find_human(item)
                    if found is not None:
                        return found
            if isinstance(node, dict):
                for value in node.values():
                    found = _find_human(value)
                    if found is not None:
                        return found
            return None

        found = _find_human(result)
        assert found is not None and found.content == "hi", (
            f"round-trip should preserve HumanMessage content; got result="
            f"{result!r}"
        )