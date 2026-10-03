"""Centralized error-recovery helpers for FSM nodes.

v2.0.22 (Item 7 Step 11) — P2-B4 关掉.

Pre-Step-11, each except site in the FSM followed its own
recovery convention:

* ``fsm.react_generate_direct`` and
  ``nodes.react_generate.react_generate`` had **byte-identical**
  apology-text + token-yield blocks (4 lines × 2 sites).
* 8 other sites used 5+ different ``logger.exception`` formats
  (``"react_generate LLM call failed"`` /
  ``"summary_path: list_chunks_by_thread failed"`` /
  ``"Retrieval failed"`` / etc.), making
  ``grep "failed:" logs/`` return noisy / non-uniform results.
* ``react_agent.react_agent`` had a bare ``except Exception`` that
  swallowed ``list_documents_cached`` failures **without even
  logging** — the most silent failure mode in the codebase.

Step 11 collapses the two patterns that actually generalize:

* :func:`llm_call_failure` — used by both LLM-stream answer
  generators. Returns ``(answer_text, ("token", payload))`` so
  the caller does ``answer_text, token_event = llm_call_failure(...,
  ); yield token_event`` and then proceeds to its normal
  ``__delta__`` path with the apology text as the answer.
* :func:`log_node_failure` — uniform
  ``logger.exception(f"{node_name} failed: {exc}")`` format for
  every defensive site that previously used a bespoke log
  message.

What this module does NOT centralize
-----------------------------------

The plan originally proposed a generic
``_recover_from_error(node_name, exc, state, fallback)`` helper.
On inspection the actual recovery semantics diverge per node
(LLM fail emits a token + apology delta; tool fail emits a
``tool_call_end(ok=False)`` + tool delta; storage fail emits
nothing + empty delta; cache fail emits nothing + empty list).
A closure-based "fallback" callable would just be a logger
wrapper — the real decision stays inline. So Step 11 stops at
the two helpers below: log format (uniform) + apology text
(byte-identical). The node-specific recovery paths stay in their
respective nodes.
"""
from __future__ import annotations

from typing import Any, Tuple

from src.core.logging import logger


# v2.0.22 (Item 7 Step 11) — single source for the LLM-failure
# apology. Pre-Step-11 this string was duplicated byte-for-byte
# in ``fsm.react_generate_direct`` and
# ``nodes.react_generate.react_generate``. Centralizing it here
# means a future i18n pass (e.g. English fallback for non-CN
# users) only needs to touch one line, and any new
# answer-generation node (a future streaming tool, a re-ranker
# post-processor) gets the same fallback text for free.
_LLM_APOLOGY_PREFIX = "抱歉,生成回答时出现了问题"
_LLM_APOLOGY_DETAIL_DEFAULT = "底层语言模型服务暂时不可用"
_LLM_APOLOGY_TAIL = "请稍后重试。"


def llm_call_failure(
    node_name: str,
    exc: BaseException,
    *,
    detail: str = _LLM_APOLOGY_DETAIL_DEFAULT,
) -> Tuple[str, Tuple[str, Any]]:
    """Recover from an LLM-stream failure during answer generation.

    Used by ``fsm.react_generate_direct`` (greeting / simple_fact
    fast path) and ``nodes.react_generate.react_generate`` (full
    qa_complex / summary path). Both had byte-identical recovery
    blocks pre-Step-11: log the exception, build a Chinese apology
    string, yield it as a single token so the user still sees the
    answer stream live, then proceed to the normal
    ``__delta__`` / ``answer_complete`` path with the apology
    text as the answer.

    Returns
    -------
    (answer_text, token_event)
        ``answer_text`` is the apology string the caller assigns
        to its local ``answer_text`` variable so the downstream
        delta carries it. ``token_event`` is a
        ``("token", {"text": answer_text})`` tuple the caller
        ``yield``s (mirrors the live-stream contract so the
        frontend still gets a streaming event on the rare
        failure path).

    Parameters
    ----------
    node_name
        Name of the failing node — used in the log message.
        Convention is the registered ``NODES`` dict key
        (``"react_generate"`` / ``"react_generate_direct"``).
    exc
        The exception caught from ``model.astream(...)``.
    detail
        Optional override for the middle clause (between
        ``"出现了问题("`` and ``")"``). Default
        ``"底层语言模型服务暂时不可用"`` matches the pre-Step-11
        message byte-for-byte. Override to e.g.
        ``"输出超过 token 上限"`` for callers that want a
        different failure reason.

    Why a function not a constant
    -----------------------------
    The pre-Step-11 string was templated on ``{exc!s}`` (Python's
    exception repr). A simple ``_LLM_APOLOGY_FORMAT = "..."``
    constant would force callers to interpolate the exception
    themselves and risk divergence. This helper takes ``exc``,
    formats the message, and returns the rendered string so every
    caller gets the exact same output.
    """
    answer_text = (
        f"{_LLM_APOLOGY_PREFIX}({detail})。"
        f"\n\n技术细节:{exc!s}\n\n{_LLM_APOLOGY_TAIL}"
    )
    logger.exception(f"{node_name} failed: {exc}")
    return answer_text, ("token", {"text": answer_text})


def log_node_failure(node_name: str, exc: BaseException) -> None:
    """Uniform exception log for FSM-node recovery sites.

    Pre-Step-11 the format varied per site:

    * ``logger.exception(f"react_generate LLM call failed: {exc}")``
    * ``logger.exception(f"summary_path: list_chunks_by_thread failed: {exc}")``
    * ``logger.exception(f"Retrieval failed: {exc}")``
    * ``logger.exception(f"BM25 search failed: {exc}")``
    * ``logger.exception(f"Dense search failed: {exc}")``
    * ``logger.exception(f"react_generate_direct failed: {exc}")``
    * ``logger.exception(f"FSM: failed to persist end-of-turn snapshot (thread_id={tid!r})")``
    * ``react_agent.react_agent`` — bare ``except Exception`` with
      **no log at all** (silent failure).

    Post-Step-11 the format is ``{node_name} failed: {exc}`` so
    ``grep "failed:" logs/`` uniformly identifies any FSM-node
    exception. The trailing context (e.g. ``thread_id``) that
    some pre-Step-11 sites added is dropped — exception traceback
    already includes the call site, and a follow-up grep on
    ``node_name`` can re-disambiguate.

    The recovery decision (what events to emit, what delta to
    merge) stays in each node — only the LOGGING was duplicated.
    """
    logger.exception(f"{node_name} failed: {exc}")


__all__ = ["llm_call_failure", "log_node_failure"]
