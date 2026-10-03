"""LanceDB table schema (PyArrow) + ChunkRecord Pydantic model."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pyarrow as pa
from pydantic import BaseModel, Field

from config.constants import EMBEDDING_DIM


# PyArrow schema used to create the LanceDB table.
# `vector` holds the BGE-M3 dense embedding; sparse weights are stored in
# `sparse` as a JSON string for portability.
LANCEDB_SCHEMA = pa.schema(
    [
        pa.field("chunk_id", pa.string(), nullable=False),
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM), nullable=False),
        pa.field("sparse", pa.string(), nullable=True),  # JSON: {token_id: weight}
        pa.field("filename", pa.string()),
        pa.field("source_type", pa.string()),  # "file" | "url"
        pa.field("source_path", pa.string()),
        pa.field("mime_type", pa.string()),
        pa.field("chunk_index", pa.int32()),
        pa.field("chunk_type", pa.string()),
        pa.field("page_number", pa.int32()),
        pa.field("section", pa.string()),
        pa.field("sheet", pa.string()),
        pa.field("headers", pa.list_(pa.string())),
        pa.field("row_start", pa.int32()),
        pa.field("row_end", pa.int32()),
        pa.field("ingested_at", pa.string()),
        # Per-conversation scoping. Empty string ("") means "globally visible"
        # — used by legacy rows ingested before this column existed and by
        # tests that don't care about thread scoping. New uploads always
        # stamp a non-empty thread_id validated via ``_safe_identifier``.
        pa.field("thread_id", pa.string()),
        # v2.0.29.4 (Phase 4 PR-2) — temporal expiry. NULL means "no
        # expiry"; ISO 8601 string means "filter out rows where
        # expires_at <= now_iso" at retrieval time. Wire via
        # ``_temporal_filter`` in ``hybrid_search.py``. Old rows
        # pre-PR-2 get NULL via ``_maybe_add_expires_at_column``
        # migration so they remain retrievable.
        pa.field("expires_at", pa.string()),
    ]
)


TABLE_NAME = "chunks"


class ChunkRecord(BaseModel):
    """A single chunk to be indexed in LanceDB."""

    chunk_id: str
    doc_id: str
    text: str
    vector: list[float] = Field(..., min_length=EMBEDDING_DIM, max_length=EMBEDDING_DIM)
    sparse: Optional[dict[int, float]] = None
    filename: str
    source_type: str  # "file" | "url"
    # v2.0.27 P0-P2 — PRIVACY INVARIANT: ``source_path`` MUST be the
    # canonical ``f"{doc_id}/{filename}"`` form produced by
    # ``src.storage.doc_registry._serialize_source_path``. Callers MUST NOT
    # pass an absolute server-side filesystem path here — the chunk table
    # is queryable through ``GET /documents`` and is a long-lived artifact
    # (rows survive even after the raw file is deleted), so persisting
    # absolute paths would leak the deployment's storage layout, the OS
    # user, and the volume of activity to any thread-scoped API caller.
    # If you genuinely need to recover the absolute path on the host
    # (e.g. for an operator debugging an ``empty`` / ``failed`` doc), run
    # ``uploads_dir() / doc_id / filename`` from a server-side script —
    # do not add an API endpoint that exposes it.
    source_path: str
    mime_type: str = ""
    chunk_index: int = 0
    chunk_type: str = "text"
    page_number: Optional[int] = None
    section: Optional[str] = None
    sheet: Optional[str] = None
    headers: Optional[list[str]] = None
    row_start: Optional[int] = None
    row_end: Optional[int] = None
    ingested_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    # Per-conversation scoping (added in the multi-conversation extension).
    # Empty default keeps existing test fixtures and pre-existing indexed data
    # globally visible until they are re-ingested. Production uploads
    # stamp a non-empty validated thread_id.
    thread_id: str = ""
    # v2.0.29.4 (Phase 4 PR-2) — temporal expiry (ISO 8601 string or
    # None for "never expires"). Production upload never sets this
    # explicitly (default None) so existing flows are untouched;
    # operators / future ingestion hooks can set an expiry timestamp
    # for docs that have a known staleness horizon (e.g. legal
    # regulations that change annually). Retrieval applies the
    # ``_temporal_filter`` only when a non-None ``temporal_cutoff_iso``
    # is supplied — so default behavior is unchanged for now.
    expires_at: Optional[str] = None

    def to_lancedb_row(self) -> dict:
        """Serialize to a dict matching LANCEDB_SCHEMA."""
        import json

        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "vector": self.vector,
            "sparse": json.dumps({str(k): v for k, v in (self.sparse or {}).items()}),
            "filename": self.filename,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "mime_type": self.mime_type,
            "chunk_index": self.chunk_index,
            "chunk_type": self.chunk_type,
            "page_number": self.page_number,
            "section": self.section,
            "sheet": self.sheet,
            "headers": self.headers or [],
            "row_start": self.row_start,
            "row_end": self.row_end,
            "ingested_at": self.ingested_at,
            "thread_id": self.thread_id,
            # v2.0.29.4 (Phase 4 PR-2) — serialize expires_at. None
            # becomes Python None which LanceDB stores as NULL (the
            # default for the column when absent on the wire is also
            # NULL). ISO strings round-trip as-is.
            "expires_at": self.expires_at,
        }


__all__ = ["LANCEDB_SCHEMA", "TABLE_NAME", "ChunkRecord"]
