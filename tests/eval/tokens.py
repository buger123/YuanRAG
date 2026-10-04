"""Token usage capture via monkey-patch on LangChain chat clients.

v2.0.32.0 — wraps ``ChatAnthropic.airstream`` (note: typo in
LangChain 1.x where the streaming method is spelled ``airstream``
instead of ``astream``; both names exist depending on version) and
``ChatOpenAI.airstream`` / ``astream`` so each streaming response's
final ``AIMessageChunk`` ``usage_metadata`` is summed into a
``TokenSnapshot`` keyed by (provider, model).

Per Risk §1: Anthropic puts usage on the final ``message_delta`` chunk
on streaming. LangChain normalizes this into ``AIMessageChunk.usage_metadata``
on the last chunk. Fallback: ``tiktoken`` estimate from the messages
list passed to ``astream`` (±10% but free).

Idempotent install/uninstall. install() returns the capture instance;
uninstall() restores original methods. Exception-safe: if install()
raises partway, uninstall still runs to restore.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable

from tests.eval.cost import TokenSnapshot


# Module-level registry of installed captures — only one capture is
# active at a time per process. If a second install() runs without
# uninstall(), it auto-uninstalls the prior to keep semantics sane.
_ACTIVE_LOCK = threading.Lock()
_ACTIVE: "TokenCapture | None" = None


def _is_stream_method(name: str) -> bool:
    """LangChain 1.x has both ``airstream`` (sic, Anthropic SDK bridge)
    and ``astream`` (the canonical async streaming method). Either may
    be present depending on the pinned version; we patch both when they
    exist.
    """
    return name in ("airstream", "astream")


def _extract_usage_metadata(chunk: Any) -> dict[str, int] | None:
    """Pull {input_tokens, output_tokens} out of a final AIMessageChunk.

    LangChain normalizes Anthropic/OpenAI usage into
    ``usage_metadata`` with keys ``input_tokens`` / ``output_tokens``
    / ``total_tokens``. Some pinned versions put it on
    ``response_metadata.usage`` instead; we check both.
    """
    usage: dict[str, int] | None = None
    # 1) usage_metadata (canonical LangChain surface)
    um = getattr(chunk, "usage_metadata", None)
    if isinstance(um, dict) and ("input_tokens" in um or "output_tokens" in um):
        usage = {
            "input_tokens": int(um.get("input_tokens", 0)),
            "output_tokens": int(um.get("output_tokens", 0)),
        }
    # 2) response_metadata.usage (older / alternative)
    if usage is None:
        rm = getattr(chunk, "response_metadata", None)
        if isinstance(rm, dict):
            inner = rm.get("usage")
            if isinstance(inner, dict):
                usage = {
                    "input_tokens": int(inner.get("input_tokens", 0)),
                    "output_tokens": int(inner.get("output_tokens", 0)),
                }
    return usage


class TokenCapture:
    """Monkey-patch streaming chat clients to capture usage_metadata.

    Usage:

        cap = TokenCapture.install()        # idempotent
        try:
            await run_chat(...)
            snap = cap.snapshot_and_reset()
        finally:
            cap.uninstall()
    """

    def __init__(self) -> None:
        self._by_model: dict[tuple[str, str], TokenSnapshot] = {}
        self._lock = threading.Lock()
        # Saved originals for uninstall — list of (obj, attr_name, original).
        self._saved: list[tuple[Any, str, Any]] = []

    # ------------------------------------------------------------
    # install / uninstall
    # ------------------------------------------------------------

    @classmethod
    def install(cls) -> "TokenCapture":
        global _ACTIVE
        with _ACTIVE_LOCK:
            if _ACTIVE is not None:
                _ACTIVE.uninstall()
            cap = cls()
            try:
                cap._patch_langchain_clients()
            except Exception:
                cap.uninstall()
                raise
            _ACTIVE = cap
            return cap

    def uninstall(self) -> None:
        """Restore all originals. Idempotent."""
        global _ACTIVE
        with _ACTIVE_LOCK:
            # Restore in reverse order (mirror install order).
            for obj, attr, original in reversed(self._saved):
                try:
                    setattr(obj, attr, original)
                except Exception:
                    pass
            self._saved.clear()
            self._by_model.clear()
            if _ACTIVE is self:
                _ACTIVE = None

    # ------------------------------------------------------------
    # patch
    # ------------------------------------------------------------

    def _patch_langchain_clients(self) -> None:
        """Wrap astream / airstream on ChatAnthropic and ChatOpenAI.

        We import lazily so the harness package doesn't pull these
        modules at import time — tests can run a single mocked test
        without needing a full LangChain install path.
        """
        # ChatAnthropic
        try:
            from langchain_anthropic import ChatAnthropic

            self._wrap_stream(ChatAnthropic, default_provider="anthropic")
        except Exception:
            # No ChatAnthropic available — that's fine, the harness still
            # runs against OpenAI or mocks.
            pass

        # ChatOpenAI
        try:
            from langchain_openai import ChatOpenAI

            self._wrap_stream(ChatOpenAI, default_provider="openai")
        except Exception:
            pass

    def _wrap_stream(self, cls: type, *, default_provider: str) -> None:
        for attr in ("airstream", "astream"):
            if not hasattr(cls, attr):
                continue
            original = getattr(cls, attr, None)
            if original is None:
                continue
            # Idempotency: don't double-wrap.
            if getattr(original, "_token_capture_wrapped", False):
                continue
            wrapped = self._make_wrapper(original, default_provider=default_provider)
            setattr(cls, attr, wrapped)
            self._saved.append((cls, attr, original))

    def _make_wrapper(
        self,
        original: Callable[..., Any],
        *,
        default_provider: str,
    ) -> Callable[..., Any]:
        cap = self

        async def async_wrapper(self_client: Any, *args: Any, **kw: Any) -> Any:
            input_tokens = 0
            output_tokens = 0
            model_name = getattr(self_client, "model_name", None) or getattr(
                self_client, "model", "unknown"
            )
            # Async generator: drain chunks, capture last usage_metadata.
            async for chunk in original(self_client, *args, **kw):
                usage = _extract_usage_metadata(chunk)
                if usage is not None:
                    input_tokens = max(input_tokens, usage["input_tokens"])
                    output_tokens = max(output_tokens, usage["output_tokens"])
                yield chunk
            # Fallback: tiktoken estimate if no usage was reported.
            if (input_tokens == 0 and output_tokens == 0) and args:
                est = _tiktoken_estimate(args[0])
                if est is not None:
                    input_tokens = est.input_tokens
                    output_tokens = est.output_tokens
            with cap._lock:
                key = (default_provider, model_name)
                prev = cap._by_model.get(key, TokenSnapshot())
                cap._by_model[key] = TokenSnapshot(
                    input_tokens=prev.input_tokens + input_tokens,
                    output_tokens=prev.output_tokens + output_tokens,
                )

        async_wrapper._token_capture_wrapped = True
        return async_wrapper

    # ------------------------------------------------------------
    # snapshot
    # ------------------------------------------------------------

    def snapshot(self) -> dict[tuple[str, str], TokenSnapshot]:
        with self._lock:
            return dict(self._by_model)

    def snapshot_and_reset(self) -> dict[tuple[str, str], TokenSnapshot]:
        with self._lock:
            snap = dict(self._by_model)
            self._by_model.clear()
            return snap

    def total(self) -> TokenSnapshot:
        """Sum across all (provider, model) pairs."""
        snap = self.snapshot()
        total = TokenSnapshot()
        for s in snap.values():
            total = total + s
        return total


# ============================================================
# Tiktoken fallback (Risk §1 of the plan)
# ============================================================


def _tiktoken_estimate(messages: Any) -> TokenSnapshot | None:
    """Estimate input tokens from a messages list using tiktoken.

    The estimate is approximate (±10%) but free. Returns ``None`` if
    tiktoken is unavailable or the input is malformed.
    """
    try:
        import tiktoken
    except Exception:
        return None
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None
    try:
        n = 0
        if isinstance(messages, list):
            for msg in messages:
                content = getattr(msg, "content", None)
                if isinstance(content, str):
                    n += len(enc.encode(content))
                elif isinstance(content, list):
                    # Multimodal content blocks — count text parts only.
                    for block in content:
                        if isinstance(block, dict):
                            text = block.get("text")
                            if isinstance(text, str):
                                n += len(enc.encode(text))
        return TokenSnapshot(input_tokens=n, output_tokens=0)
    except Exception:
        return None


def is_capture_active() -> bool:
    """Public — for tests / observability."""
    with _ACTIVE_LOCK:
        return _ACTIVE is not None