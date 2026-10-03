"""Project-wide constants. Tweak these in one place."""

# ---- Agent budget caps ----
# v2.0 — ReAct rewrite. The legacy caps were per-stage budget
# (how many retrieval rounds before falling back to web search; how
# many query rewrites before giving up). ReAct is a single loop and
# the LLM drives the budget via the tools it chooses to call.
#
# v2.0.11 — bumped ``max_steps`` from 10 to 25 (matching LangGraph's
# default). The 10-step cap was a guess from the legacy graph (which
# had separate per-stage caps that summed to fewer nodes per turn).
# The v2.0 ReAct topology has more nodes per turn:
#   1 (intent) + 2*N (react_agent + tools loop iterations) +
#   1 (react_generate) + 1 (check_hallucination)
# A query that needs 3 retrieve + 1 web + 1 final = 8 steps just for
# the tool rounds + 2 entry/exit = 10 steps. With LangGraph's
# conditional edge overhead we land at exactly the cap and a
# legitimate multi-tool query hits GraphRecursionError. 25 gives
# margin for 4-5 retrieval rounds + 2 web searches + final answer,
# which empirically covers ~99% of real queries. Override via env
# RAG_REACT_MAX_STEPS for tests or operators who want a different cap.
#
# ``max_steps`` caps the LangGraph recursion depth. Each step is one
# LLM call (react_agent OR react_generate) plus any tool calls in
# between.
#
# The ``max_iterations`` / ``max_retrieval_rounds`` / ``max_rewrites``
# keys are intentionally REMOVED — no node reads them anymore and
# keeping stale keys invites confusion.
AGENT_BUDGETS = {
    "max_steps": 25,
}


def _resolve_max_steps() -> int:
    """Resolve ReAct max steps from env (RAG_REACT_MAX_STEPS), else default."""
    import os

    raw = os.environ.get("RAG_REACT_MAX_STEPS", "").strip()
    if raw:
        try:
            v = int(raw)
            if 1 <= v <= 100:
                return v
        except ValueError:
            pass
    return AGENT_BUDGETS["max_steps"]

# ---- Retrieval ----
RRF_K = 60  # Reciprocal Rank Fusion constant (industry default)

# v2.0.29.8 (Phase 7) — per-doc chunk cap. When a single document
# contributes more than ``MAX_CHUNKS_PER_DOC`` chunks to a retrieval
# result, apply sliding window (head + middle samples + tail) to
# keep the LLM context window bounded. Triggers on both the
# hybrid-search path (per-doc candidate count from the post-rerank
# top-K) and the summary-intent bulk path (per-doc count from
# ``list_chunks_by_thread``).
#
# 50 is calibrated against BGE-M3 512-token chunks: a 50-chunk doc
# ≈ 25k tokens ≈ 1/4 of a typical 100k-token context window,
# leaving 75k tokens for the answer + system prompt + retrieved
# docs from other sources. Override via env
# ``RAG_MAX_CHUNKS_PER_DOC`` for tests or operators who want a
# different cap.
MAX_CHUNKS_PER_DOC = 50


def _resolve_max_chunks_per_doc() -> int:
    """Resolve MAX_CHUNKS_PER_DOC from env (RAG_MAX_CHUNKS_PER_DOC), else default."""
    import os

    raw = os.environ.get("RAG_MAX_CHUNKS_PER_DOC", "").strip()
    if raw:
        try:
            v = int(raw)
            if 1 <= v <= 10_000:
                return v
        except ValueError:
            pass
    return MAX_CHUNKS_PER_DOC

RETRIEVAL_DEFAULTS = {
    "top_k_pre_rerank": 20,
    "top_k_post_rerank": 5,
    "min_relevance_grade": 1,  # 0=irrelevant, 1=relevant (LLM scale)
    # v1.1.7 (Bug 2) — relevance floor for the LLM grader's "kept this
    # chunk" verdict. The cheap grader is lenient: for an unrelated
    # query (e.g. "告诉我双鱼座男生的特点" against a 形势与政策 doc) it
    # sometimes marks 1-2 generic chunks as "relevant", which then
    # gates the graph straight to ``generate_answer`` and the user
    # gets a confused answer based on irrelevant context. If the TOP
    # graded chunk's rerank score is below this floor, fall through to
    # the rewrite / ``search_web`` path instead. Tuned against BGE
    # reranker-v2-m3 sigmoid-normalized scores (0-1 range, higher =
    # more relevant): >0.5 strongly relevant, 0.3-0.5 marginal,
    # <0.3 unrelated. 0.3 is the working threshold per a sweep of the
    # malformed-vs-correct splits in production session logs (Sept
    # 2026).
    "min_top_relevance_score": 0.3,
}

# ---- Embedding model ----
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
EMBEDDING_MAX_TOKENS = 512

# ---- Reranker ----
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_MAX_TOKENS = 512

# ---- Chunking ----
CHUNKING = {
    "max_tokens": 512,
    "merge_peers": True,
    "excel_rows_per_chunk": 4,
}

# ---- Server ----
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

# ---- Upload ----
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

# ---- Web search (DuckDuckGo via ddgs) ----
# DuckDuckGo's HTML endpoints actively throttle — community reports indicate
# ~30 req/min per IP triggers a 202. We cap ourselves well below that with a
# per-call semaphore + a global wall-clock gate. Exponential backoff retries
# up to ``max_attempts``; if every attempt fails, search_web returns an empty
# list and the graph falls through to ``generate_answer`` with empty docs
# (the existing GENERATE_SYSTEM prompt produces a clean "no information
# found" answer).
WEB_SEARCH = {
    "max_results": 5,                   # web results fed to LLM per query
    "max_concurrency": 2,                # AsyncDDGS(max_clients=...) cap
    "min_interval_seconds": 1.5,         # global wall-clock gate between DDG calls
    "max_attempts": 3,                   # retries on RatelimitException or empty result
    "backoff_seconds": (1.5, 3.0, 6.0),  # delays before attempts 2 and 3
    "region": "wt-wt",                   # DuckDuckGo region code
    "safesearch": "moderate",
    # v1.1.10 — engine chain. The dispatcher in
    # ``src/web_search/__init__.py`` tries each engine in order and
    # returns the first non-empty result. ``["bing"]`` (default) is
    # China-friendly (cn.bing.com resolves in ~0.3s behind the GFW).
    # Override via env var ``RAG_WEB_SEARCH_ENGINES=bing,ddg`` for
    # non-China users who want DuckDuckGo's full backend fan-out.
    # Recognised engine names: "bing", "ddg".
    #
    # v2.0.4 — default chain now includes DDG as the fallback so a
    # user whose primary engine returns empty / anti-bot doesn't see
    # an unsourced fabricated answer. The dispatcher falls through
    # to the next engine on empty results, NOT on partial results
    # (so a single Bing hit doesn't get replaced by a worse DDG hit).
    "engines": ["bing", "ddg"],
    "per_engine_timeout_seconds": 8,
    # v2.0.4 — Bing locale params. ``mkt`` / ``setlang`` / ``cc`` make
    # Bing return language-matched results instead of falling back to
    # its en-US default. ``cn_bing_mkt`` overrides for the CN edge
    # node (which ignores ``cc``). ``count`` would map to Bing's
    # ``count=N`` URL param, but Bing HTML pages cap at 10 regardless
    # so we don't expose it. Detection happens at scrape-time in
    # ``src/web_search/bing.py``.
    "bing_mkt_zh": "zh-CN",
    "bing_mkt_en": "en-US",
    "bing_cc_cn": "CN",
}

# ---- Storage paths (relative to user data dir) ----
LANCEDB_DIR = "lancedb"
HISTORY_DB = "history.db"
MODELS_DIR = "models"
UPLOADS_DIR = "uploads"
LOGS_DIR = "logs"
CONFIG_FILE = "config/settings.json"

# ---- LLM model defaults ----
# Provider is Anthropic by default — the deployment injects a custom
# ``MINIMAX_BASE_URL`` that points at an Anthropic-compatible proxy, and
# ``MINIMAX_MODEL`` sets the model name. Users can switch to OpenAI in
# Settings if they want real OpenAI.
DEFAULT_LLM_PROVIDER = "anthropic"
DEFAULT_LLM_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "MiniMax-M3",
}

# ---- Keyring service name (Windows Credential Manager prefix) ----
KEYRING_SERVICE = "rag_assistant"
