#!/usr/bin/env python
"""Pre-download BGE-M3 + BGE Reranker v2-M3 into FastEmbed's cache.

Run this manually before first launch, or let the lifespan auto-call it.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.logging import setup_logging
from src.core.logging import logger
from src.embeddings.bge_m3 import BGEM3Embedder
from src.reranker.bge_reranker import BGEReranker


def main() -> int:
    setup_logging()
    logger.info("Downloading BGE-M3 (dense + sparse)...")
    emb = BGEM3Embedder()
    # Trigger lazy load
    emb._embed(["warmup"])
    logger.info("BGE-M3 ready.")

    logger.info("Downloading BGE Reranker v2-M3...")
    rer = BGEReranker()
    rer.score("warmup", ["warmup doc"])
    logger.info("BGE Reranker ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
