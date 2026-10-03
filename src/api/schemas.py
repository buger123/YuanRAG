"""Pydantic request/response models for the HTTP/WebSocket API."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---- Chat ----

class Source(BaseModel):
    """Single citation chip rendered in the answer footer.

    v2.0.7 SoT-1: previously every ``sources`` field across the
    wire (``ChatEvent``, ``MessageRecord``) was typed as
    ``Optional[list[dict[str, Any]]]``. The TypedDict
    ``src.agent.state.Source`` is the canonical Python-side shape,
    but neither side enforced it — adding a field meant a silent
    drift. This Pydantic model freezes the contract: any
    ``answer_complete`` payload that doesn't satisfy these fields
    raises at serialization time instead of leaking an
    under-specified dict to the frontend.

    ``source_kind`` is required (was optional in the TypedDict) —
    the backend always emits ``"local"`` or ``"web"`` and the
    frontend always reads it (see
    ``src/frontend/src/chat-utils.tsx``).

    ``model_config = ConfigDict(extra="ignore")`` keeps wire
    compatibility: a backend that adds a new field won't crash an
    older frontend that doesn't know about it.
    """

    index: int
    chunk_id: str = ""
    doc_id: str = ""
    filename: str = ""
    page: Optional[int] = None
    section: Optional[str] = None
    sheet: Optional[str] = None
    text: str = ""
    score: float = 0.0
    url: Optional[str] = None
    domain: Optional[str] = None
    source_kind: Literal["local", "web"] = "local"
    # v2.0.29.9 (Phase 8) — verbatim extraction mode marker.
    # When True, the source was used in ``react_generate_extractive``
    # mode (verbatim quote, no rewriting). Frontend reads this to
    # render a 🔒 icon on the bubble; old clients that don't know
    # about the field render the source normally (additive field,
    # ``extra="ignore"`` keeps wire compat).
    verbatim: bool = False

    model_config = ConfigDict(extra="ignore")


class ChatRequest(BaseModel):
    thread_id: str
    message: str
    stream: bool = True
    # v2.0.29.9 (Phase 8) — verbatim extraction mode tristate.
    # Forwarded from the frontend's per-thread segmented control.
    # Default "auto" → run hybrid detect (regex + cheap-LLM fallback);
    # "on" → force extractive mode (verbatim quote, no rewriting);
    # "off" → force normal mode (skip extractive EVEN if regex/cheap-LLM
    # detects the query as verbatim — per user decision 2026-09-28,
    # user "off" always overrides auto-detect).
    #
    # Pre-Phase-8 callers that don't send this field default to "auto"
    # (no behavioral change for legacy clients).
    high_precision: Literal["auto", "on", "off"] = "auto"


class ChatEvent(BaseModel):
    type: Literal[
        "token",
        "reasoning",
        "answer_complete",
        "grounding",
        "error",
        "done",
        "web_search",
        # v2.0 — ReAct events. See ``src/agent/events.py`` for
        # the canonical builder signatures.
        "intent",
        "step_start",
        "step_end",
        "tool_call_start",
        "tool_call_end",
    ]
    content: Optional[str] = None
    doc_count: Optional[int] = None
    sources: Optional[list[Source]] = None
    answer: Optional[str] = None
    status: Optional[str] = None
    message: Optional[str] = None
    attempted: Optional[bool] = None  # used by "web_search" events
    # v1.1.14 — banner rendering. Drives the frontend's choice of
    # "本回答参考了联网搜索结果" / "本回答参考了文档内容" / both /
    # neither (direct greeting). See ``events.answer_complete`` for
    # the canonical contract. ``None`` on events that don't carry
    # banner info (e.g. ``web_search``) — the frontend should
    # treat that as "no banner update".
    source_kinds: Optional[list[str]] = None
    # v2.0 — server-side ISO 8601 timestamp on the assistant
    # turn. Carried by ``answer_complete`` events; ignored on all
    # others. The frontend renders it as a small footer below the
    # message body; history replay rehydrates it from the
    # persisted AIMessage's ``additional_kwargs["created_at"]``.
    created_at: Optional[str] = None
    # v2.0 — ReAct payload. See ``src/agent/events.py`` for
    # the per-event semantics. ``intent`` / ``corrected_query``
    # ride on ``type="intent"``; ``step`` rides on
    # ``step_start`` / ``step_end`` / ``tool_call_*``;
    # ``name`` / ``args`` / ``tool_call_id`` / ``result_summary`` /
    # ``elapsed_ms`` / ``ok`` ride on ``tool_call_start`` /
    # ``tool_call_end``.
    intent: Optional[str] = None
    corrected_query: Optional[str] = None
    step: Optional[int] = None
    name: Optional[str] = None
    args: Optional[dict[str, Any]] = None
    tool_call_id: Optional[str] = None
    result_summary: Optional[str] = None
    elapsed_ms: Optional[int] = None
    ok: Optional[bool] = None


# ---- Sessions ----

class SessionSummary(BaseModel):
    thread_id: str
    title: str
    updated_at: str
    message_count: int
    # Number of documents uploaded into this conversation. Surfaced in the
    # sidebar so the user can see at a glance which threads have files.
    doc_count: int = 0


class MessageRecord(BaseModel):
    """A single persisted chat message replayed from the checkpointer.

    Returned by ``GET /sessions/{thread_id}/messages`` so the frontend can
    rehydrate the chat pane when the user reopens an old session.

    ``sources`` is populated for assistant messages that emitted citations
    when they were originally generated. ``reasoning`` is populated for
    assistant messages whose underlying LLM produced extended-thinking
    blocks (Anthropic's ``thinking`` / ``redacted_thinking``). Both are
    stored as part of the AIMessage's ``additional_kwargs`` rather than
    the LangGraph ``messages`` channel — that channel is a list of
    LangChain ``BaseMessage`` objects which only carry ``content`` /
    tool calls / metadata, not arbitrary user fields. Stashing them on
    ``additional_kwargs`` lets them survive the SqliteSaver round-trip.
    Old conversations generated before either field was added come back
    with ``None`` and the frontend degrades gracefully (no citations, no
    thinking drawer).
    """

    role: Literal["user", "assistant", "system"]
    content: str
    sources: Optional[list[Source]] = None
    reasoning: Optional[str] = None
    # v1.1.14 — banner rendering on history replay. Forwarded from
    # ``additional_kwargs["source_kinds"]`` on the AIMessage so the
    # banner survives a page refresh (the live ``web_search`` event
    # isn't replayed; without this field the banner would always go
    # blank after the user reopens a session). See
    # ``events.answer_complete`` and ``generate._derive_source_kinds``
    # for the canonical contract.
    source_kinds: Optional[list[str]] = None
    # v2.0 — per-tool-call log persisted on the AIMessage. The
    # runner emits ``tool_call_start`` / ``tool_call_end`` events
    # during the live stream; history replay rehydrates the cards
    # from this list. ``None`` for messages persisted before
    # v2.0 or for turns that didn't call any tool (greeting /
    # direct).
    tool_calls: Optional[list[dict[str, Any]]] = None
    # v2.0 — server-side ISO 8601 timestamp. Forwarded from
    # ``additional_kwargs["created_at"]`` on both HumanMessage and
    # AIMessage. Drives the per-message footer in ChatPane.
    # ``None`` for messages persisted before v2.0.
    created_at: Optional[str] = None


# ---- Documents ----

class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    status: str = "pending"
    thread_id: Optional[str] = None


class DocumentSummary(BaseModel):
    doc_id: str
    filename: str
    source_type: str
    source_path: str
    chunk_count: int
    ingested_at: str
    thread_id: Optional[str] = None
    # Ingestion status from ``doc_registry``. ``indexed`` is the
    # well-behaved happy path (chunks landed in LanceDB). ``empty``
    # means parsing finished but no text was extracted — typical for
    # image-only PDFs where Docling's OCR + pypdfium2 fallback both
    # returned nothing. ``failed`` means ingestion raised. ``pending``
    # is the brief window between upload and the parser starting;
    # ``parsing`` / ``embedding`` (v2.0.5) are the per-stage signals
    # the sidebar uses to render "解析中…" / "向量化中…" while
    # Docling / BGE-M3 are the active worker.
    # Default ``indexed`` keeps backward compatibility with callers
    # that pre-date the registry (or with test fixtures that inject
    # docs straight into the chunk store without registering).
    status: Literal[
        "pending", "parsing", "embedding", "indexed", "empty", "failed"
    ] = "indexed"
    error_message: Optional[str] = None


# ---- Settings ----

class SettingsRequest(BaseModel):
    """Provider + model only. API keys are configured via env vars (.env)."""

    llm_provider: Optional[Literal["openai", "anthropic"]] = None
    llm_model: Optional[str] = None


class SettingsResponse(BaseModel):
    llm_provider: str
    llm_model: str
    # Echo whether the deployment injected credentials via env. The UI uses
    # this to decide whether to show a "missing key" hint.
    has_api_key: bool
    api_key_source: Literal["env", "keyring", "none"]


# ---- Models ----

class ModelInfo(BaseModel):
    name: str
    status: Literal["ready", "downloading", "missing"]
    progress: float = 0.0  # 0.0 - 1.0


class ModelsStatus(BaseModel):
    embedding: ModelInfo
    reranker: ModelInfo
    ready: bool
