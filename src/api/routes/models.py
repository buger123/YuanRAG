"""Model readiness endpoint.

Reports whether the embedding and reranker models are loaded.

Important: this endpoint must be CHEAP. The frontend polls it every few
seconds, and earlier versions of this handler triggered a fresh ONNX model
load on every poll, hammering the CPU and filling the logs with
"Loading BGE-M3 models" lines forever. We now check ``models_loaded()`` /
``model_loaded()`` which short-circuit to ``True`` once the model has been
instantiated, and only force a load if explicitly requested via POST.

Download safety: ``POST /models/download`` runs the warmup on a background
THREAD (not on the asyncio event loop). The previous version used
``BackgroundTasks.add_task(_warmup)`` which executes the synchronous
warmup body on the event loop thread, freezing the entire server (HTTP,
WebSocket, lifespan) for ~30 s while BGE-M3 loads and another ~30 s while
the reranker loads. Symptom from that bug: after clicking Download, the
WebSocket went silent, the page appeared frozen, and the user thought the
app had crashed. Running the warmup in ``asyncio.to_thread(...)`` keeps
the loop free for polling / WS frames while the load proceeds.
"""
from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import ModelInfo, ModelsStatus
from src.core.logging import logger
from src.embeddings.bge_m3 import (
    load_error as bge_load_error,
    model_loading as bge_loading,
    models_loaded as bge_loaded,
)
from src.reranker.bge_reranker import (
    load_error as reranker_load_error,
    model_loaded as reranker_loaded,
    model_loading as reranker_loading,
)


router = APIRouter(prefix="/models", tags=["models"])


# Tracked background warmup tasks. The /download route creates a task per
# call; the lifespan cancels them on shutdown. Without tracking, an
# in-flight load at shutdown would either (a) be aborted by event-loop
# close with a noisy traceback, or (b) keep a worker thread blocked until
# it finishes naturally — both observable in production as lingering
# process / noisy shutdown logs.
_warmup_tasks: set[asyncio.Task] = set()


def cancel_warmup_tasks() -> int:
    """Cancel any in-flight model warmup tasks. Called from lifespan
    teardown. Returns the number of tasks cancelled.

    Safe to call when no tasks are pending.
    """
    cancelled = 0
    for t in list(_warmup_tasks):
        if not t.done():
            t.cancel()
            cancelled += 1
    return cancelled


def _status(name: str, *, loaded: bool, loading: bool) -> ModelInfo:
    """Build a ModelInfo reflecting the three possible states.

    ``loading`` takes precedence over ``missing`` so the frontend can
    distinguish "click has been registered, download is in flight" from
    "no one has clicked yet".
    """
    if loaded:
        return ModelInfo(name=name, status="ready", progress=1.0)
    if loading:
        return ModelInfo(name=name, status="downloading", progress=0.0)
    return ModelInfo(name=name, status="missing", progress=0.0)


@router.get("/status", response_model=ModelsStatus)
async def status() -> ModelsStatus:
    emb = _status("BAAI/bge-m3", loaded=bge_loaded(), loading=bge_loading())
    rr = _status(
        "BAAI/bge-reranker-v2-m3",
        loaded=reranker_loaded(),
        loading=reranker_loading(),
    )
    return ModelsStatus(
        embedding=emb,
        reranker=rr,
        ready=(emb.status == "ready" and rr.status == "ready"),
    )


@router.get("/wait", response_model=ModelsStatus)
async def wait_ready(
    timeout: float = Query(60.0, ge=1.0, le=300.0),
    poll_interval: float = Query(0.5, ge=0.1, le=5.0),
) -> ModelsStatus:
    """v2.0.30.0 — long-poll until models ready OR timeout.

    Frontend calls this once on mount instead of polling ``/status``
    every 2 s. Behavior:

      - 200 + ``ModelsStatus(ready=True)`` when both models load
      - 200 + ``ModelsStatus`` reflecting the current state on error
        or "not loading and not loaded" (e.g. ``status="missing"`` if
        no one clicked Download) — front-end handles these as terminal
      - 408 Request Timeout when the deadline elapses without a
        state transition; caller falls back to ``/status`` polling

    The poll cadence is ``poll_interval`` seconds (default 500 ms —
    within the 500 ms target latency minus the network roundtrip).
    Backend state is just module-level flags; no asyncio.Event or
    broadcast channel is needed. If concurrent wait traffic grows
    beyond a handful of in-flight requests, switch to an Event-based
    notification (one Event per module, set inside the warmup
    success/failure branches) — left as a future optimization.
    """
    deadline = time.monotonic() + timeout
    while True:
        emb_loaded = bge_loaded()
        rr_loaded = reranker_loaded()
        if emb_loaded and rr_loaded:
            return ModelsStatus(
                embedding=ModelInfo(name="BAAI/bge-m3", status="ready", progress=1.0),
                reranker=ModelInfo(name="BAAI/bge-reranker-v2-m3", status="ready", progress=1.0),
                ready=True,
            )
        # Error, or "not loading and not loaded" — surface immediately.
        # We delegate to /status() for the canonical shape so the
        # frontend can render the same banner it would for a polling
        # read (downloading → missing / failed).
        if bge_load_error() or reranker_load_error() or \
           (not bge_loading() and not reranker_loading() and not (emb_loaded or rr_loaded)):
            return await status()
        if time.monotonic() >= deadline:
            raise HTTPException(
                status_code=408,
                detail="model load wait timed out",
            )
        await asyncio.sleep(poll_interval)


@router.post("/download")
async def trigger_download() -> ModelsStatus:
    """Kick off model warmup in the background and return current status.

    The warmup itself is dispatched onto a worker thread via
    ``asyncio.create_task(asyncio.to_thread(_warmup))`` so the asyncio
    loop stays responsive AND the HTTP response returns immediately. The
    earlier implementation used ``BackgroundTasks.add_task(_warmup)``,
    which Starlette awaits after sending the response body but BEFORE
    closing the connection — so the client still saw the full ~30 s
    load time as response latency, and any in-flight WebSocket frames
    couldn't be processed while the BackgroundTask ran (the loop was
    otherwise free, but BackgroundTasks blocks the response close).
    Using ``create_task`` makes the warmup truly fire-and-forget: the
    response returns in milliseconds and the load runs on a worker
    thread in parallel.

    The ``_loading`` flag is set inside each module's ``get_model``
    while it's running, so the /status endpoint reflects live state.
    """
    start_warmup_if_needed()
    return await status()


def start_warmup_if_needed() -> bool:
    """Idempotent background warmup. Returns True if a new task was spawned.

    Used both by the lifespan (so cold-start users don't have to click
    anything) and by ``POST /models/download`` (so the existing manual
    path still works). A no-op if both models are already loaded.

    The actual work runs on a worker thread via ``asyncio.to_thread``;
    failures are logged but don't crash the server.
    """
    from src.embeddings.bge_m3 import BGEM3Embedder
    from src.reranker.bge_reranker import BGEReranker

    # If both are already loaded, nothing to do — return False to signal
    # "no work needed" (caller can short-circuit any UI feedback).
    if bge_loaded() and reranker_loaded():
        return False

    def _warmup() -> None:
        # Sequence matters: load the embedder first (cheaper + usually faster
        # on disk) so the reranker load doesn't block chat for retriever
        # queries that don't need it. Either call sets its own _loading flag
        # while it's running, so the /status endpoint reflects live state.
        if not bge_loaded():
            BGEM3Embedder().embed_query("warmup")
        if not reranker_loaded():
            BGEReranker().score("warmup", ["warmup"])

    # Fire-and-forget on the event loop, with the heavy work delegated to a
    # worker thread. Failures are logged but don't crash the loop.
    task = asyncio.create_task(asyncio.to_thread(_warmup))
    _warmup_tasks.add(task)
    # Self-removing done callback so the set doesn't grow forever across
    # many warmup calls. Without this, every successful warmup would
    # leave a finished task reference in the set, slowly leaking memory.
    task.add_done_callback(_warmup_tasks.discard)

    def _log_failure(t: asyncio.Task) -> None:
        try:
            t.result()
        except asyncio.CancelledError:
            # Shutdown initiated — don't log as an error, that's noisy.
            logger.info("Model warmup cancelled (shutdown)")
            return
        except Exception as exc:
            # Visible in the log so the user can see the failure when they
            # look at startup logs. Frontend will keep showing "downloading"
            # until the user retries; we never claim ready on partial state.
            logger.exception(f"Model warmup failed: {exc}")

    task.add_done_callback(_log_failure)
    return True