"""Phase 3 — retrieval tests (RRF + LanceDB roundtrip without ML models)."""
from __future__ import annotations

import numpy as np
import pyarrow as pa

from src.embeddings.bge_m3 import BGEM3Embedder
from src.retrieval.hybrid_search import reciprocal_rank_fusion
from src.storage.lancedb_store import add_chunks, delete_by_doc_id, get_table, list_documents
from src.storage.schema import ChunkRecord


def test_rrf_basic():
    """RRF should sum reciprocal ranks across lists."""
    a = [{"chunk_id": "x"}, {"chunk_id": "y"}, {"chunk_id": "z"}]
    b = [{"chunk_id": "y"}, {"chunk_id": "x"}, {"chunk_id": "w"}]
    fused = reciprocal_rank_fusion([a, b])
    # x: 1/(k+1) + 1/(k+2) = > y: 1/(k+2) + 1/(k+1) — tie
    # w: only appears once → score = 1/(k+3)
    # z: only appears once → score = 1/(k+3)
    # x and y tied; both above w and z
    assert fused[0]["chunk_id"] in {"x", "y"}
    assert fused[1]["chunk_id"] in {"x", "y"}
    assert fused[0]["chunk_id"] != fused[1]["chunk_id"]
    assert {f["chunk_id"] for f in fused[-2:]} == {"w", "z"}


def test_rrf_single_list():
    a = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    fused = reciprocal_rank_fusion([a])
    assert [f["chunk_id"] for f in fused] == ["a", "b"]


def test_rrf_handles_missing_in_some_lists():
    a = [{"chunk_id": "a"}]
    b = [{"chunk_id": "b"}, {"chunk_id": "a"}]
    fused = reciprocal_rank_fusion([a, b])
    # a appears in both → highest fused
    assert fused[0]["chunk_id"] == "a"
    assert fused[1]["chunk_id"] == "b"


def test_chunkrecord_roundtrip():
    """ChunkRecord → LanceDB row → ChunkRecord works."""
    rec = ChunkRecord(
        chunk_id="c1",
        doc_id="d1",
        text="hello world",
        vector=[0.0] * 1024,
        sparse={1: 0.5, 2: 0.3},
        filename="a.txt",
        source_type="file",
        # v2.0.27 P0-P2 — canonical privacy-safe form.
        source_path="d1/a.txt",
        chunk_index=0,
    )
    row = rec.to_lancedb_row()
    assert row["chunk_id"] == "c1"
    assert row["vector"] == [0.0] * 1024
    assert "1" in row["sparse"]  # JSON-encoded keys are strings


def test_chunkrecord_thread_id_roundtrip():
    """Per-conversation thread_id survives to_lancedb_row."""
    rec_with = ChunkRecord(
        chunk_id="c1",
        doc_id="d1",
        text="x",
        vector=[0.0] * 1024,
        filename="a.txt",
        source_type="file",
        # v2.0.27 P0-P2 — canonical privacy-safe form.
        source_path="d1/a.txt",
        thread_id="alpha-uuid",
    )
    assert rec_with.to_lancedb_row()["thread_id"] == "alpha-uuid"

    # Default is empty (legacy / globally-visible sentinel).
    rec_default = ChunkRecord(
        chunk_id="c2",
        doc_id="d2",
        text="x",
        vector=[0.0] * 1024,
        filename="b.txt",
        source_type="file",
        # v2.0.27 P0-P2 — canonical privacy-safe form.
        source_path="d2/b.txt",
    )
    assert rec_default.to_lancedb_row()["thread_id"] == ""


def test_lancedb_add_and_list(tmp_path, monkeypatch):
    """Smoke test: add records, list documents, delete by doc_id."""
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    from config.settings import reset_settings_cache
    from src.storage import lancedb_store as store_mod

    reset_settings_cache()
    store_mod._db = None  # reset singleton

    records = [
        ChunkRecord(
            chunk_id=f"c{i}",
            doc_id="d1" if i < 2 else "d2",
            text=f"doc {i}",
            vector=list(np.random.RandomState(42 + i).rand(1024).astype("float32")),
            filename=f"file{i}.txt",
            source_type="file",
            # v2.0.27 P0-P2 — canonical privacy-safe form
            # (doc_id/filename).
            source_path=(
                f"{'d1' if i < 2 else 'd2'}/file{i}.txt"
            ),
            chunk_index=i,
        )
        for i in range(4)
    ]
    add_chunks(records)

    docs = list_documents()
    assert len(docs) == 2
    by_id = {d["doc_id"]: d for d in docs}
    assert by_id["d1"]["chunk_count"] == 2
    assert by_id["d2"]["chunk_count"] == 2

    delete_by_doc_id("d1")
    docs = list_documents()
    assert len(docs) == 1
    assert docs[0]["doc_id"] == "d2"


def test_hit_to_document_propagates_chunk_index():
    """v2.0.29.8 (Phase 7) — ``_hit_to_document`` must propagate
    ``chunk_index`` from the hit, matching the bulk-path sibling
    ``_bulk_chunk_to_document`` (line 134). Without this,
    ``_filter_long_docs`` cannot decide head/middle/tail in the
    hybrid-search path (every hit looks like position 0).

    Pins the contract that the 1-LOC consistency fix is in place
    and the sliding-window helper sees meaningful positions.
    """
    from src.agent.nodes.retrieve import _hit_to_document

    # Position 0 (head)
    hit0 = {"text": "first", "chunk_id": "c0", "doc_id": "d1",
            "chunk_index": 0, "rerank_score": 0.95}
    doc0 = _hit_to_document(hit0)
    assert doc0.metadata["chunk_index"] == 0
    assert doc0.metadata["chunk_id"] == "c0"

    # Position 47 (middle)
    hit_mid = {"text": "middle", "chunk_id": "c47", "doc_id": "d1",
               "chunk_index": 47, "rerank_score": 0.7}
    doc_mid = _hit_to_document(hit_mid)
    assert doc_mid.metadata["chunk_index"] == 47

    # Position 79 (tail) — the most important one for sliding window
    hit_tail = {"text": "last", "chunk_id": "c79", "doc_id": "d1",
                "chunk_index": 79, "rerank_score": 0.5}
    doc_tail = _hit_to_document(hit_tail)
    assert doc_tail.metadata["chunk_index"] == 79

    # Missing chunk_index (legacy / OCR noise) — must propagate as None
    # so ``_filter_long_docs`` can sort them to the end of the window.
    hit_no_idx = {"text": "noindex", "chunk_id": "cx", "doc_id": "d1"}
    doc_no_idx = _hit_to_document(hit_no_idx)
    assert doc_no_idx.metadata["chunk_index"] is None
