"""Process-wide bootstrap: must run before anything reads env vars.

Loaded eagerly from both ``src/main.py`` (the canonical entry point
``python -m src.main``) and ``src/app.py`` (the FastAPI factory used
by ``uvicorn src.app:app`` and by integration tests). Without this,
``RAG_DATA_DIR`` set in ``.env`` would only be honored when the app
was started via ``main.py`` — running ``uvicorn`` directly (the way
IDEs, tests, and ``start.bat`` alternatives do) would silently skip
the .env load and write to ``~/.rag_assistant`` instead of the
project root.

The contract:

1. Reads ``<project_root>/.env`` if it exists.
2. ``override=False`` — process-level env always wins (so users can
   set ``RAG_DATA_DIR`` ad-hoc without editing the file).
3. ``DOTENV_LOADED`` is set as a sentinel so callers can detect a
   double-load (idempotent).

Any module that wants to be sure env has been loaded should import
this package — or call :func:`load_env` directly.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is a hard requirement
    def load_dotenv(*args, **kwargs):  # type: ignore[no-redef]
        return False


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"

# HuggingFace endpoint default. ``huggingface.co`` is frequently unreachable
# from CN networks (``[WinError 10060]`` connect timeout when the BGE-M3
# loader hits ``hf_api().list_repo_tree``). The community mirror
# ``hf-mirror.com`` proxies all Hub endpoints (repo metadata + blob CDN) and
# is the de-facto default for offline / restricted deployments.
#
# Override behavior mirrors ``HF_HOME``: process env wins, then .env, then
# this default. To force the official endpoint, set ``HF_ENDPOINT=`` (empty)
# in either place — we treat the empty string as "user opted out" rather
# than silently falling back to the mirror.
_DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"

_DOTENV_LOADED = False


def load_env() -> bool:
    """Load ``.env`` from the project root. Idempotent.

    Returns ``True`` if the file was found and loaded, ``False``
    otherwise. Subsequent calls are no-ops.

    Side effects:

    - If ``HF_HOME`` is unset, point it at ``<project_root>/data/hf_cache``.
      HuggingFace Hub (and the FlagEmbedding wrapper that depends on it)
      read ``HF_HOME`` at import time to decide where to read/write model
      blobs. Keeping the cache inside the project (instead of the Windows
      default ``%USERPROFILE%/.cache/huggingface``) keeps multi-GB model
      files off the boot drive and makes the install portable — copy the
      project, you copy the cache.

    - If ``HF_ENDPOINT`` is unset, point it at the ``hf-mirror.com``
      community mirror so BGE-M3 / reranker / Docling loads work from
      networks where ``huggingface.co`` is unreachable. See
      :data:`_DEFAULT_HF_ENDPOINT` for rationale.

    If a user has set either variable explicitly (either in .env or in the
    process env), we never override it. Same convention as ``RAG_DATA_DIR``:
    process env wins, .env is the fallback, hard-coded default is the
    last-resort floor.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return _ENV_PATH.exists()
    if _ENV_PATH.exists():
        load_dotenv(dotenv_path=_ENV_PATH, override=False)
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = str(_PROJECT_ROOT / "data" / "hf_cache")
    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = _DEFAULT_HF_ENDPOINT
    _DOTENV_LOADED = True
    return _ENV_PATH.exists()


def project_root() -> Path:
    """Return the project root directory (containing ``src/``)."""
    return _PROJECT_ROOT


# Load immediately on import — by the time any other code in the app
# references env vars, this module's side effects have already run.
load_env()


__all__ = ["load_env", "project_root"]