"""HTTP chat endpoint (REST mirror of the WebSocket, for non-streaming clients)."""
from __future__ import annotations

from contextlib import aclosing

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse

from src.agent.runner import stream_agent
from src.api.schemas import ChatRequest
from src.api.i18n import error_message, parse_accept_language


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("")
async def chat(req: ChatRequest):
    """Server-Sent Events stream of agent events."""

    async def event_gen():
        # ``aclosing`` ensures the inner async generator is explicitly
        # closed if the client disconnects mid-stream (StreamingResponse
        # calls aclose on cancel). Without this, LangGraph's internal
        # background tasks from astream() can leak past the response.
        # v2.0.29.9 (Phase 8) — forward per-thread verbatim toggle
        # (default "auto"). Pre-Phase-8 clients don't send the field;
        # Pydantic default preserves hybrid-detect behavior.
        async with aclosing(
            stream_agent(req.thread_id, req.message, high_precision=req.high_precision)
        ) as stream:
            async for event in stream:
                # event is a dict; emit as SSE
                import json

                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.delete("/{thread_id}")
async def clear(
    thread_id: str,
    # PR-4 (v2.0.26.1): locale for the error path so the user
    # sees the same locale across all session-clear endpoints.
    accept_language: str | None = Header(None, alias="accept-language"),
) -> dict:
    from src.storage.checkpointer import delete_thread

    locale = parse_accept_language(accept_language)
    try:
        await delete_thread(thread_id)
    except Exception as exc:
        # Surface the failure to the caller rather than returning
        # {"ok": True} for a no-op delete — callers previously trusted the
        # response shape and reported "history still there" bugs that were
        # actually silent failures. PR-4: localized copy.
        raise HTTPException(
            status_code=500,
            detail=error_message("chat.clear_failed", locale),
        ) from exc
    return {"ok": True, "thread_id": thread_id}
