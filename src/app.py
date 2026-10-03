"""FastAPI app factory — wires all routes, WebSocket, static frontend.

Called by uvicorn (``uvicorn src.app:app``) and by src/main.py.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

# Import the bootstrap module FIRST so .env is loaded before any other
# module reads env vars (paths.user_data_dir(), settings, history, ...).
# Without this, `uvicorn src.app:app` (started directly, bypassing
# src/main.py) would silently write to ~/.rag_assistant even when
# RAG_DATA_DIR is set in .env.
from src.core import bootstrap  # noqa: F401 — side effect: load_env()

from config.settings import get_settings
from src.api.routes import chat as chat_routes
from src.api.routes import debug_metrics as debug_metrics_routes
from src.api.routes import documents as documents_routes
from src.api.routes import formats as formats_routes
from src.api.routes import models as models_routes
from src.api.routes import sessions as sessions_routes
from src.api.routes import settings as settings_routes
from src.api.websocket import router as ws_router
from src.api.middleware import TraceIdMiddleware  # v2.0.28.1 P1-P2
from src.core.logging import logger, setup_logging
from src.core.paths import ensure_all_dirs
from src.storage.checkpointer import (
    startup as startup_checkpointer,
    shutdown as shutdown_checkpointer,
)


class _CacheControlledStaticFiles(StaticFiles):
    """StaticFiles subclass that sets Cache-Control headers based on path.

    See the mount-site comment in :func:`create_app` for the rationale.

    Overrides :meth:`StaticFiles.get_response` (Starlette >=0.27) so we
    can attach headers to whatever Response class Starlette picks
    (FileResponse / Response). The previous implementation returned a
    new Response and dropped the body — DO NOT do that.
    """

    async def get_response(self, path: str, scope) -> Response:  # type: ignore[override]
        response = await super().get_response(path, scope)
        # Starlette passes the path that was matched by the mount.
        # For ``GET /``, the path is empty ``""`` (then ``html=True``
        # resolves it to ``index.html``); for ``GET /index.html`` it's
        # ``"index.html"``; for ``GET /assets/index-<hash>.js`` it's
        # ``"assets/index-<hash>.js"``. Handle all three.
        basename = path.rsplit("/", 1)[-1]
        if path == "" or basename == "index.html":
            response.headers["Cache-Control"] = "no-cache, must-revalidate"
        elif path.startswith("assets/") and (
            basename.endswith(".js") or basename.endswith(".css")
        ):
            # Vite content hash means URL is unique per content.
            response.headers["Cache-Control"] = (
                "public, max-age=31536000, immutable"
            )
        else:
            response.headers["Cache-Control"] = "no-cache"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: warm up storage + ensure dirs."""
    ensure_all_dirs()
    settings = get_settings()
    setup_logging(level=settings.log_level)
    logger.info("RAG Assistant starting up")

    # Initialize the LanceDB table (creates if missing) so the first request
    # doesn't pay the cost.
    try:
        from src.storage.lancedb_store import get_table

        get_table()
    except Exception:
        logger.exception("LanceDB warmup failed")

    # Open the checkpointer eagerly so the first /ws/chat doesn't pay
    # the setup cost (and so any setup error surfaces at startup, not mid-chat).
    # v2.0.21 (Phase 3) — own checkpointer replaces LangGraph
    # ``AsyncSqliteSaver``. ``startup`` also runs the one-shot
    # LangGraph → own-schema migration (no-op after first run).
    try:
        await startup_checkpointer()
    except Exception:
        logger.exception("Checkpointer warmup failed")

    # Kick off the BGE-M3 / reranker warmup in the background so the
    # frontend doesn't have to show a "Download models" dialog on every
    # cold start. Weights are already cached on disk (HF_HOME), so this
    # is just an in-RAM instantiation (PyTorch + fp16 + transformer
    # init, ~10-30 s). The load runs on a worker thread, so the event
    # loop stays responsive for HTTP / WS while it completes.
    # ``start_warmup_if_needed`` is a no-op if both are already loaded.
    try:
        from src.api.routes.models import start_warmup_if_needed

        if start_warmup_if_needed():
            logger.info("Started background model warmup")
    except Exception:
        logger.exception("Failed to schedule background model warmup")

    # v2.0.8 perf-7: warm up the HybridChunker tokenizer eagerly so the
    # first user upload doesn't pay the 5-15 s ``AutoTokenizer`` +
    # ``HybridChunker.__init__`` cost on the upload hot path. The BGE-M3
    # model warmup above is the slow one (~10-30 s); this is the cheap
    # follow-up and runs synchronously here because the server hasn't
    # started serving requests yet. ``warmup_chunker`` swallows errors
    # and returns False — a failure here only means the first upload
    # will pay the cold-start cost (the pre-v2.0.8 behavior), never a
    # server-startup failure.
    try:
        from src.ingestion.chunkers.hybrid_chunker import warmup_chunker

        if warmup_chunker():
            logger.info("HybridChunker warmed up")
    except Exception:
        logger.exception("Failed to warm up HybridChunker")

    yield

    # Cancel any in-flight model warmup tasks BEFORE we tear down the
    # checkpointer / DB. Otherwise a warmup that completes during shutdown
    # would race with shutdown_checkpointer and could leave the load half
    # done, holding a worker thread until the process is killed.
    try:
        from src.api.routes.models import cancel_warmup_tasks

        n = cancel_warmup_tasks()
        if n:
            logger.info(f"Cancelled {n} in-flight model warmup task(s)")
    except Exception:
        logger.exception("Failed to cancel warmup tasks")

    # v2.0.27.1 P1-P4 — same shutdown guarantee for fire-and-forget
    # ingest tasks. If a 50 MB PDF is mid-ingest when shutdown begins,
    # we cancel the worker thread so we don't leave a half-done writer
    # racing with the doc_registry flusher below. The user's doc
    # registry entry stays in ``pending`` state (visible in the
    # sidebar) but no chunks land in LanceDB and no ``indexed`` flip
    # happens — the user can retry by re-uploading.
    try:
        from src.api.routes.documents import cancel_ingest_tasks

        n = cancel_ingest_tasks()
        if n:
            logger.info(f"Cancelled {n} in-flight ingest task(s)")
    except Exception:
        logger.exception("Failed to cancel ingest tasks")

    # v2.0.28.4 P1-4 — close every cached LLM client so the underlying
    # httpx connection pools are released before the process exits.
    # Without this the OS would hold the sockets until GC caught up
    # (often fine, but during fast restart loops the FD count creeps
    # up and eventually the next start fails with "Too many open files").
    # ``close_all_llm_clients`` is best-effort (swallows close errors
    # per-client) and idempotent — safe to call even if the cache is
    # already empty.
    try:
        from src.llm import factory
        from src.llm.factory import close_all_llm_clients

        n = len(factory._LLM_CACHE)
        close_all_llm_clients()
        logger.info(f"Closed {n} cached LLM client(s) on lifespan shutdown")
    except Exception:
        logger.exception("Failed to close cached LLM clients")

    # v2.0.28.6 P1-R4 — release the BGE-M3 + BGE-reranker singletons
    # BEFORE the checkpointer shuts down. On long-running GPU boxes
    # PyTorch's allocator holds VRAM until process exit; ``release()``
    # calls ``torch.cuda.empty_cache()`` to return that memory to the
    # OS proactively. Idempotent (safe to call when the model isn't
    # loaded) and best-effort (swallows CUDA errors). Wired here,
    # not in main.py, because lifespan is the canonical shutdown
    # channel for FastAPI apps — main.py just calls uvicorn.
    try:
        from src.embeddings.bge_m3 import release as release_bge_m3
        from src.reranker.bge_reranker import release as release_bge_reranker

        release_bge_m3()
        release_bge_reranker()
        logger.info("Released BGE-M3 + BGE-reranker singletons on lifespan shutdown")
    except Exception:
        logger.exception("Failed to release model singletons on shutdown")

    await shutdown_checkpointer()
    # Stop the doc_registry background flusher and write any pending
    # state — otherwise the last ingestion of the session could be lost.
    try:
        from src.storage import doc_registry

        doc_registry.shutdown()
    except Exception:
        logger.exception("Failed to shut down doc_registry flusher")
    logger.info("RAG Assistant shutting down")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="RAG Assistant",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS: only allow localhost origins (browser dev server on 5173).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            f"http://{settings.host}:{settings.port}",
            "http://localhost:8765",
            "http://127.0.0.1:8765",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # v2.0.28.1 P1-P2 — TraceIdMiddleware runs OUTERMOST so it stamps
    # trace_id into contextvars BEFORE CORS preflight handlers (which
    # never reach our handlers) can log anything, AND wraps the final
    # response so X-Request-ID round-trips back to the client. Pure
    # ASGI (not BaseHTTPMiddleware) so WS handshakes also get
    # trace_id + X-Request-ID on accept.
    # Starlette middleware order is REVERSE — last add_middleware
    # call runs first. So add TraceId AFTER CORS in source order.
    app.add_middleware(TraceIdMiddleware)

    # REST + WS routes
    app.include_router(chat_routes.router)
    app.include_router(sessions_routes.router)
    app.include_router(documents_routes.router)
    # v2.0.29.2 (Phase 2) — debug metrics endpoint. Gated by
    # ``YuanRAG_DEBUG_METRICS_ENABLED``; default off so an
    # unscanned attacker doesn't see the metric taxonomy. Returns
    # 404 when off (same pattern as the i18n catalog debug route).
    app.include_router(debug_metrics_routes.router)
    app.include_router(formats_routes.router)
    app.include_router(settings_routes.router)
    app.include_router(models_routes.router)
    app.include_router(ws_router)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    # Static frontend (built Vite output). Mount at "/" last so API routes win.
    frontend_dist = Path(__file__).parent / "frontend" / "dist"
    if frontend_dist.exists():
        # v2.0.28.20.1 — Cache-Control headers for the dist mount. Without
        # them, browsers heuristically cache ``index.html`` based on
        # ``Last-Modified`` and may serve a STALE index.html that
        # references a deleted bundle hash → the user sees a broken page
        # even after rebuild. The fix:
        #   * ``index.html`` — ``no-cache, must-revalidate``: browser
        #     must revalidate via ETag on every request. If ETag
        #     changes (e.g. new bundle hash), browser re-fetches; if
        #     unchanged, browser serves cached copy (saves bandwidth).
        #   * ``/assets/index-*.{js,css}`` — ``public, max-age=31536000,
        #     immutable``: Vite content hashes ensure the URL is unique
        #     per content, so safe to cache for a year. Browser never
        #     revalidates these — they change only when hash changes.
        #   * everything else (``favicon.svg``, etc.) — ``no-cache``:
        #     same revalidation strategy as index.html.
        app.mount(
            "/",
            _CacheControlledStaticFiles(directory=str(frontend_dist), html=True),
            name="ui",
        )
    else:
        logger.warning(
            f"Frontend dist not found at {frontend_dist}. "
            "Run `bash scripts/build_frontend.sh` (or `npm run build` in src/frontend)."
        )

    return app


app = create_app()
