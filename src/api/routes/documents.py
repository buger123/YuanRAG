"""Document upload + listing + deletion endpoints."""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import (
    APIRouter,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)

from config.constants import MAX_UPLOAD_BYTES
from src.api.schemas import (
    DocumentSummary,
    UploadResponse,
)
from src.api.i18n import error_message, parse_accept_language
from src.core.logging import logger
from src.core.paths import uploads_dir
from src.ingestion.pipeline import run_ingestion
from src.storage import doc_registry
from src.storage.lancedb_store import (
    _safe_identifier,
    add_chunks,
    delete_by_doc_id,
    delete_by_thread_id,
    list_documents,
)
from src.storage.schema import ChunkRecord
from src.utils.ids import new_id
from src.utils.safe_filename import assert_within_root, safe_filename


router = APIRouter(prefix="/documents", tags=["documents"])


# v2.0.27.1 P1-P4 — fire-and-forget ingest tasks (mirrors the
# ``_warmup_tasks`` pattern from ``src/api/routes/models.py:43``).
# ``BackgroundTasks.add_task`` ran AFTER the response body was sent but
# BEFORE the response connection was closed, so the client still saw
# the ingest latency (parsing + chunking + embedding + LanceDB write
# can take seconds for big files) as response latency, and any
# in-flight WebSocket frames couldn't progress while the BackgroundTask
# was holding the response close path. Replacing with
# ``asyncio.create_task(asyncio.to_thread(...))`` makes the dispatch
# truly fire-and-forget: response returns in milliseconds, ingest runs
# on a worker thread, lifespan shutdown cancels tracked tasks so we
# don't leak a half-done writer. The set self-cleans on completion via
# ``add_done_callback``.
_tracked_ingest_tasks: set[asyncio.Task] = set()


def cancel_ingest_tasks() -> int:
    """Cancel any in-flight ingest tasks. Called from lifespan
    teardown. Returns the number of tasks cancelled.

    Safe to call when no tasks are pending. A cancelled ingest has
    already written its ``pending`` registry entry (so the user still
    sees the doc in the sidebar with the right ``doc_id``), but no
    chunks land in LanceDB and no ``indexed`` flip happens. On the
    next request, the user can re-upload to retry.
    """
    cancelled = 0
    for t in list(_tracked_ingest_tasks):
        if not t.done():
            t.cancel()
            cancelled += 1
    return cancelled


def _log_ingest_failure(t: asyncio.Task) -> None:
    """Done-callback for ingest tasks. Swallows ``CancelledError`` as
    info (lifespan teardown initiated) and logs any other exception
    via ``logger.exception`` so it's visible in the operator log.

    The actual ``status="failed"`` flip is performed inside
    ``_ingest_file`` itself via try/except, so we only need to handle
    the case where the dispatch layer (this wrapping) failed — which
    shouldn't happen in practice, but a silent swallowed exception
    here would mask any future bug. Mirror of
    ``models._log_failure``.
    """
    try:
        t.result()
    except asyncio.CancelledError:
        logger.info("Document ingest cancelled (shutdown)")
        return
    except Exception:
        logger.exception("Document ingest task raised unexpected exception")


# ============================================================
# Upload size / content-type safety
# ============================================================
#
# The previous implementation read the entire uploaded body into memory
# with ``await file.read()`` before checking size. A 500 MB payload
# would therefore pin 500 MB of RAM just to reject the upload — easy
# OOM, easy DoS. We now stream + cap in 1 MiB chunks so the working set
# stays at ≤1 MiB regardless of what the client sends, and the size
# check fires as soon as the cap is exceeded (no point draining the
# rest of the body just to throw it away).
#
# Magic-byte sniffing defends against extension spoofing: a client that
# declares ``evil.pdf`` but sends an .exe or a 4 GB zip gets rejected
# before we touch the chunk store. Without it, the user could trigger
# the wrong ingestion parser (and waste an embedding pass) on a
# malformed payload.
_CHUNK_SIZE = 1 * 1024 * 1024  # 1 MiB

# Sniff the first 64 bytes against known format signatures. Anything
# that doesn't match is rejected — a generic ``application/octet-stream``
# payload is exactly what an exploit would look like. 64 is enough to
# catch UTF-16-LE BOMs (which interleave NUL bytes every other
# character), to spot NULs hiding past byte 16, and to recognize
# GBK two-byte lead bytes — all without pulling the entire body into
# memory.
_SNIFF_BYTES = 64


def _sniff_format(head: bytes, declared_ext: str) -> bool:
    """Lightweight magic-byte check: does ``head`` look like a real file
    of the type the extension claims?

    Return ``True`` if it matches (or we can't verify the type — plain
    text / markdown), ``False`` to reject. ``declared_ext`` includes the
    leading dot, e.g. ``.pdf``. Unknown extensions also return ``True``
    so the ingestion pipeline can fall back to its own format detector.

    We intentionally accept "starts with" prefixes that may not cover
    the whole token: HTML pages often begin ``<!DOCTYPE html PUBLIC
    ...`` or ``<html lang="en">`` etc., so matching just the leading
    tag (not the literal word ``html``) keeps the sniff permissive
    without accepting arbitrary XML / plain text.
    """
    ext = declared_ext.lower()
    # PDF: ``%PDF-``
    if ext == ".pdf":
        return head.startswith(b"%PDF-")
    # DOCX/XLSX/PPTX are all ZIP containers: ``PK\x03\x04``
    if ext in {".docx", ".xlsx", ".pptx"}:
        return head.startswith(b"PK\x03\x04")
    # HTML: starts with optional BOM + ``<!DOCTYPE`` or ``<html``,
    # possibly with whitespace between. Match the leading tag, not
    # the literal word — a page that starts ``<!DOCTYPE html PUBLIC``
    # should be accepted even though the first 16 bytes are
    # ``<!DOCTYPE html PUBLI``.
    if ext in {".html", ".htm"}:
        s = head.lstrip(b"\xef\xbb\xbf").lstrip().lower()
        return s.startswith((b"<!doctype", b"<html"))
    # Plain text / markdown: text-only by definition, no signature to
    # verify. v1.1.12: accept UTF-8 / UTF-16-LE-BOM / UTF-16-BE-BOM /
    # GBK so a Chinese-Windows-default-saved file doesn't get bounced
    # by a strict UTF-8 decode (the parser already falls back to
    # latin-1 so sniff and parse must agree). NUL / control-byte
    # black-list rejects binary uploads pretending to be text.
    if ext in {".txt", ".md", ".markdown"}:
        return _looks_like_text(head)
    # Unknown extension — let the ingestion pipeline handle it.
    return True


# Bytes that disqualify a file from being treated as plain text.
# ``\t`` (9), ``\n`` (10), ``\f`` (12), ``\r`` (13) are the only
# control bytes we permit (real text files contain them; raw binary
# would not).
_TEXT_ALLOWED_CONTROL_BYTES = frozenset({9, 10, 12, 13})


def _looks_like_text(head: bytes) -> bool:
    """Heuristic: does ``head`` look like text (not binary)?

    Accepts (in priority order):
      * UTF-16-LE BOM (``\\xff\\xfe``) or UTF-16-BE BOM (``\\xfe\\xff``)
        — common for Windows Notepad "Unicode" .txt saves.
      * UTF-8 — default everywhere.
      * GB18030 — superset of GBK; the de-facto Chinese Windows
        记事本 / Notepad++ default. Without this fallback a perfectly
        valid Chinese .md / .txt gets rejected by a strict UTF-8
        decode.
      * Big5 — Traditional Chinese (Taiwan / HK Notepad).
      * CP1252 — Western European / English Windows ANSI default;
        catches Word smart-quote artifacts that survive a copy-paste
        into a .md.

    Rejects anything containing ``NUL`` (``\\x00``) or any control byte
    outside ``{tab, LF, FF, CR}``. This guards against attackers
    uploading arbitrary binary blobs (executables, packed data) under a
    ``.md`` / ``.txt`` extension to bypass the magic-byte sniff and
    waste an ingestion pass.

    **Truncation-tolerant:** the head may be cut mid-multibyte-character
    (we only look at the first ``_SNIFF_BYTES = 64`` bytes). A naive
    ``head.decode("utf-8")`` would raise on the dangling lead byte and
    incorrectly reject a perfectly valid Chinese .md. We use an
    incremental decoder with ``final=False`` so a partial trailing
    character does not invalidate the whole decode — only genuinely
    invalid middle bytes will. About half of all Chinese .md uploads
    used to land on a mid-character cut and get rejected; this fix
    closes that gap.

    The function intentionally mirrors ``text_parser.parse_text``'s
    encoding tolerance so the validator never accepts something the
    parser can't decode (or, conversely, rejects something the parser
    would happily read). The NUL guard is a hard line even the
    latin-1 fallback wouldn't cross — text files virtually never
    contain NUL.
    """
    import codecs

    if not head:
        # An empty body is ambiguous; let the parser reject it.
        return True
    # UTF-16 BOM must be detected BEFORE the control-byte black-list
    # because UTF-16-LE/BE interleave ``\x00`` between every ASCII
    # character (``# Title\n`` → ``\xff\xfe# \x00T\x00i\x00t\x00…``).
    # Without this short-circuit, valid UTF-16-LE .md / .txt saves
    # (Windows Notepad "Unicode" default) would all be rejected.
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        return True
    # NUL / control-byte black-list (only inspect the first ~64 bytes —
    # matches ``_SNIFF_BYTES`` so we cover the full sniff window). Must
    # come BEFORE the encoding chain: a binary that happens to
    # decode cleanly under CP1252 (e.g. a short JPEG header) should
    # still be rejected on the basis of its control bytes.
    for b in head[:64]:
        if b == 0 or (b < 32 and b not in _TEXT_ALLOWED_CONTROL_BYTES):
            return False
    # Incremental decode chain. ``final=False`` is load-bearing: it
    # tells the decoder "the input may be incomplete, don't raise on a
    # partial trailing character." A naive ``head.decode("utf-8")``
    # would raise on a 3-byte UTF-8 sequence cut after the first byte,
    # even when the rest of the head is valid UTF-8 — that's the
    # "every Chinese .md of ≥64 bytes has a 50% rejection rate" bug.
    for enc in ("utf-8", "gb18030", "big5", "cp1252"):
        dec = codecs.getincrementaldecoder(enc)("strict")
        try:
            dec.decode(head, final=False)
            return True
        except UnicodeDecodeError:
            continue
    return False


# v1.1.12 — format-specific "未提取到文本" copy. The previous single
# message ("扫描件无文字层…") made sense for PDFs but was wrong for
# every other format (header-only .xlsx got told to "重新生成 PDF").
# We dispatch by ``Format`` instead. Keep the messages short and
# actionable — they show in the sidebar tooltip.
#
# PR-4 (v2.0.26.1): the messages now live in the i18n catalog under
# ``docs.empty_status.*`` and are looked up via :func:`error_message`
# instead of being hardcoded here. The format keys match the
# ``Format`` enum's ``.value`` (xlsx / pdf / docx / pptx / html).
_EMPTY_FORMAT_CODES: dict[str, str] = {
    "xlsx": "docs.empty_status.xlsx",
    "pdf": "docs.empty_status.pdf",
    "docx": "docs.empty_status.docx",
    "pptx": "docs.empty_status.pptx",
    "html": "docs.empty_status.html",
}


def _empty_status_message(filename: str, fmt, content_bytes, locale: str = "zh") -> str:
    """Pick the right "未提取到文本" copy for ``fmt``. Falls back to
    the PDF message when the format is unknown — the most likely
    cause of an unknown-but-empty status is the parser crashed and
    produced no chunks.

    PR-4: ``locale`` parameter threads the user's preferred language
    (parsed from ``Accept-Language`` by the caller). Ingestion runs
    in a background thread that doesn't have request context, so the
    caller must pass the locale in.
    """
    fmt_value = fmt.value if hasattr(fmt, "value") else str(fmt)
    code = _EMPTY_FORMAT_CODES.get(fmt_value, "docs.empty_status.pdf")
    return error_message(code, locale)


def _empty_status_diagnosis(fmt) -> dict:
    """v1.1.12 — structured diagnosis companion to the human empty
    message. Pin the schema here so clients can stably read
    ``hint_zh`` without parsing the free-text copy.

    PR-4: ``hint_zh`` keeps the ``_zh`` suffix for backward
    compatibility with clients that haven't picked up the new
    ``hint`` field (which carries the localized copy). The PDF code
    is used as the fallback so the field is always non-empty.
    """
    from src.ingestion.format_detect import Format as _Fmt

    fmt_value = fmt.value if hasattr(fmt, "value") else str(fmt)
    code = _EMPTY_FORMAT_CODES.get(fmt_value, "docs.empty_status.pdf")
    hint_zh = error_message(code, "zh")
    return {
        "format": fmt_value,
        "hint_zh": hint_zh,
    }


def _sniff_failure_diagnosis(declared_ext: str, head: bytes) -> dict:
    """v1.1.12 — diagnosis payload for the 400 sniff-rejection path.
    Names the extension + encoding attempts so a client / log reader
    can tell at a glance *why* the body was rejected. Encoding list
    mirrors ``_looks_like_text``'s actual accept set; keep in sync."""
    out = {
        "declared_ext": declared_ext,
        "sniff_layer": "encoding" if declared_ext in {
            ".txt", ".md", ".markdown"
        } else declared_ext.lstrip("."),
        "tried_encodings": [],
        "reason": "",
    }
    if declared_ext in {".txt", ".md", ".markdown"}:
        out["tried_encodings"] = [
            "utf-16-le-bom",
            "utf-16-be-bom",
            "utf-8",
            "gb18030",
            "big5",
            "cp1252",
        ]
        # Identify the most likely cause for the failure reason string.
        if any(b == 0 or (b < 32 and b not in (9, 10, 12, 13)) for b in head[:64]):
            out["reason"] = "head_bytes contain NUL or non-tab/LF/FF/CR control characters"
        elif head.startswith((b"\xff\xfe", b"\xfe\xff")):
            out["reason"] = "unexpected — UTF-16 BOM should have been accepted"
        else:
            out["reason"] = "head_bytes not decodable in any supported text encoding"
    elif declared_ext in {".html", ".htm"}:
        out["reason"] = "head_bytes do not start with <!DOCTYPE or <html"
    elif declared_ext in {".docx", ".xlsx", ".pptx"}:
        out["reason"] = "head_bytes do not start with ZIP magic PK\\x03\\x04"
    elif declared_ext == ".pdf":
        out["reason"] = "head_bytes do not start with %PDF-"
    else:
        out["reason"] = "extension not recognized"
    return out


def _validate_thread_id(thread_id: str | None) -> str:
    """Validate a thread_id query param. Empty/None returns empty string
    (legacy / globally-visible sentinel); anything else must pass the same
    banned-char filter used in LanceDB filter expressions.
    """
    if not thread_id:
        return ""
    try:
        return _safe_identifier(thread_id, kind="thread_id")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _read_capped(file: UploadFile, max_bytes: int) -> bytes:
    """Stream the upload body in fixed-size chunks, abort if it exceeds
    ``max_bytes``.

    Returns the full body once collected (≤ ``max_bytes``). Raises
    :class:`HTTPException` with status 413 the moment the cap is
    exceeded — we don't drain the rest of the body, the client only
    needs to know it was rejected.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                413,
                f"File too large (>{max_bytes // (1024 * 1024)} MB)",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _ingest_file(
    doc_id: str,
    filename: str,
    content_bytes: bytes,
    thread_id: str,
    locale: str = "zh",
) -> int:
    """Run ingestion in a background thread; return chunk count.

    Updates the doc registry (see ``src.storage.doc_registry``) at every
    state transition so the sidebar can show pending / indexed / empty /
    failed status for every uploaded file — even ones that produce zero
    chunks (which would otherwise silently vanish from the chunk store).

    PR-4 (v2.0.26.1): ``locale`` is threaded in from the upload
    request (parsed from ``Accept-Language``) so the empty-status
    message stored in the registry matches the user's UI language.
    Defaults to zh for callers that don't pass it (background sweeps,
    tests).
    """
    from src.embeddings.bge_m3 import BGEM3Embedder

    # QW #12: write the raw bytes ONLY so the parser can re-read them
    # via a file path (Docling takes a path, not bytes). Once the
    # chunks are embedded + indexed in LanceDB, the raw bytes are no
    # longer needed for retrieval — and persisting them on disk was
    # the privacy expectation gap (a user's PDF would linger until
    # ``delete_by_doc_id`` was called manually). We delete on
    # success below; on failure we KEEP the bytes for debugging
    # (status="failed" docs are inspectable via the uploads dir).
    #
    # The filename is already sanitized at the endpoint, but we still
    # wrap the join in ``assert_within_root`` so a doc_id directory
    # that's been replaced by a symlink can't escape the upload root.
    target = uploads_dir() / doc_id
    target.mkdir(parents=True, exist_ok=True)
    target_path = target / filename
    target_path.write_bytes(content_bytes)
    try:
        assert_within_root(target_path, uploads_dir())
    except ValueError as exc:
        # If the path resolved outside the upload root, refuse to
        # ingest — the file we just wrote would be invisible to the
        # next list call, but the cleanup contract still requires us to
        # surface it rather than silently orphan it.
        logger.error(f"Upload path escaped root ({exc}); deleting orphan file")
        try:
            target_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="Invalid upload destination") from exc

    # Pre-register the doc so the sidebar shows it immediately with
    # status="pending". If ingestion later lands no chunks, this same
    # row gets flipped to "empty" / "failed" and the user can still
    # see and delete it.
    doc_registry.register(
        doc_id=doc_id,
        filename=filename,
        source_path=str(target_path),
        thread_id=thread_id,
    )

    try:
        # Detect the format here too so the empty-status path
        # (which fires below when ``chunks`` is empty) can look up
        # format-specific hints without re-running detection.
        from src.ingestion.format_detect import detect
        fmt = detect(filename, content_bytes)
        # v2.0.5 — flip to "parsing" BEFORE the parser runs so the
        # sidebar's poll loop picks up a stage-specific chip instead
        # of the generic "处理中" pending one. Docling can take
        # 30-90 s on a large PDF; the per-stage signal is what
        # closes the "stuck for 90 s" UX complaint.
        doc_registry.update(doc_id, status="parsing")
        chunks = run_ingestion(filename, content_bytes)
    except Exception as e:
        # Surface the exception to logs (full traceback via
        # ``logger.exception``) and to the registry so the sidebar can
        # show "解析失败" with the reason. We intentionally do NOT
        # re-raise: the upload endpoint already returned a 200 with the
        # doc_id, so there's no caller waiting for an exception — but
        # leaving the doc in "pending" forever would be a worse bug.
        logger.exception(f"Ingestion failed for {filename}")
        doc_registry.update(
            doc_id,
            status="failed",
            error_message=f"解析失败:{type(e).__name__}: {e}",
        )
        return 0

    texts = [c["text"] for c in chunks]
    if not texts:
        # Parsed but produced no text (e.g. image-only PDF where both
        # Docling's OCR and the pypdfium2 fallback returned nothing).
        # Mark "empty" so the sidebar can show "未提取到文本" with a
        # delete button instead of silently dropping the doc. KEEP the
        # raw bytes here too — the user will likely re-upload after
        # fixing the source file, and ``add_chunks`` later won't need
        # them.
        #
        # v1.1.12: dispatch the error message AND attach a structured
        # ``diagnosis`` blob by the parsed format so a header-only
        # .xlsx doesn't get the "扫描件无文字层" copy that was
        # originally written for PDF.
        empty_msg = _empty_status_message(filename, fmt, content_bytes, locale)
        diagnosis = _empty_status_diagnosis(fmt)
        doc_registry.update(
            doc_id,
            status="empty",
            error_message=empty_msg,
            diagnosis=diagnosis,
        )
        return 0

    # v2.0.5 — flip to "embedding" between parser and embedder so the
    # sidebar renders "向量化中…" while BGE-M3 is the active worker.
    # Emitting this between the two CPU-bound stages lets the chip
    # tell the user where the time is going instead of one giant
    # "处理中".
    doc_registry.update(
        doc_id,
        status="embedding",
        chunk_count=len(texts),
    )

    try:
        # Embed
        embedder = BGEM3Embedder()
        embeddings = embedder.embed_documents(texts)

        # Build ChunkRecords
        records: list[ChunkRecord] = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            meta = chunk.get("meta") or {}
            # v2.0.27 P0-P2: never persist the absolute server-side path
            # in a chunk row. ``_serialize_source_path`` produces
            # ``f"{doc_id}/{filename}"`` — sufficient context for an
            # operator to locate the orphan raw file under
            # ``uploads_dir()/<doc_id>/<filename>`` for the empty/failed
            # debugging path without leaking the deployment's storage
            # layout, OS user, or filesystem footprint.
            from src.storage.doc_registry import _serialize_source_path

            chunk_source_path = _serialize_source_path(
                str(target_path), doc_id, filename
            )
            records.append(
                ChunkRecord(
                    chunk_id=new_id(),
                    doc_id=doc_id,
                    text=chunk["text"],
                    vector=emb["dense"],
                    sparse=emb.get("sparse") or {},
                    filename=filename,
                    source_type="file",
                    source_path=chunk_source_path,
                    mime_type="",
                    chunk_index=i,
                    chunk_type=meta.get("chunk_type", "text"),
                    page_number=meta.get("page_number"),
                    section=meta.get("section"),
                    sheet=meta.get("sheet"),
                    headers=meta.get("headers"),
                    row_start=(meta.get("row_range") or [None, None])[0],
                    row_end=(meta.get("row_range") or [None, None])[1],
                    thread_id=thread_id,
                )
            )

        add_chunks(records)
        # v2.0.10 — ``doc_registry.update(status="indexed")`` moved into
        # ``add_chunks`` so the registry flip is atomic with the LanceDB
        # write. Without this, the Sidebar's ``pollUntilIndexed`` exited on
        # ``cc > 0`` while the registry was still at "embedding", and the
        # doc list showed "向量化中(N 块)" indefinitely. See
        # ``src.storage.lancedb_store.add_chunks`` for the full fix.
        # QW #12: success path — delete the raw bytes. The chunks are
        # persisted in LanceDB; the raw file is no longer needed for
        # retrieval. ``missing_ok=True`` so an already-deleted file (e.g.
        # a previous crash recovery sweep) doesn't surface as an error.
        # We log the path first so a forensic trace exists if a future
        # user complains "the file is gone but the doc still answers".
        try:
            target_path.unlink(missing_ok=True)
            logger.debug(f"QW #12: deleted raw bytes at {target_path} after successful ingest")
        except Exception as exc:
            # Don't fail the request just because cleanup failed — the
            # chunks are already indexed. Log and move on; a periodic
            # sweep job can pick up orphaned uploads dir files later.
            logger.warning(f"QW #12: failed to delete {target_path}: {exc}")
        return len(records)
    except Exception as exc:
        # v2.0.10 — without this handler, ANY exception from
        # ``embed_documents`` (model not loaded, OOM, CUDA error) /
        # ChunkRecord construction (bad sparse-vec) / ``add_chunks``
        # (LanceDB write error) leaves the doc permanently at
        # ``status="embedding"`` with the stale ``chunk_count=len(texts)``
        # already published. The Sidebar then renders "向量化中(N 块)"
        # forever even though chunks never landed. The race fix in
        # ``add_chunks`` covers the indexed flip; this handler covers
        # the failure path: flip the doc to "failed" so the sidebar
        # chip switches to "解析失败" with the reason. Re-raise so the
        # background task still logs the full traceback.
        logger.exception(f"Ingestion persist failed for {filename}")
        doc_registry.update(
            doc_id,
            status="failed",
            error_message=f"向量化/入库失败:{type(exc).__name__}: {exc}",
        )
        raise


async def _ingest_file_async(
    doc_id: str, filename: str, content: bytes, thread_id: str, locale: str = "zh"
) -> int:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _ingest_file, doc_id, filename, content, thread_id, locale
    )


@router.post("/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    # v2.0.6 P2-2 — accept thread_id as EITHER a query parameter
    # (legacy / simple curl) OR a multipart form field (the natural
    # way to upload-with-scope when the file itself is the multipart
    # body). Form wins when both are provided so SDK callers can
    # override the query if they want to.
    thread_id_form: str | None = Form(None, alias="thread_id"),
    thread_id: str | None = None,
    # PR-4 (v2.0.26.1): threads ``Accept-Language`` into the upload
    # error paths so a user with English UI sees English error copy
    # (was hardcoded English/Chinese regardless of locale). Default
    # ``Header(None)`` keeps curl / SDK callers working.
    accept_language: str | None = Header(None, alias="accept-language"),
) -> UploadResponse:
    # PR-4: parse locale once at request entry; reused for every
    # error site below and forwarded to the background ingest task
    # so its "未提取到文本" message can be localized too.
    locale = parse_accept_language(accept_language)
    safe_tid = _validate_thread_id(thread_id_form or thread_id)
    # Sanitize the client-supplied filename BEFORE we touch the
    # filesystem. Without this, ``../`` segments / absolute paths /
    # NUL bytes / Windows device names would either escape the upload
    # root or crash the filesystem driver. ``safe_filename`` raises
    # ``ValueError`` on any of those — mapped to 400 so the client sees
    # a clear error rather than silently ending up with ``upload.bin``.
    try:
        fname = safe_filename(file.filename) if file.filename else "upload.bin"
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Stream + size cap. Previously ``await file.read()`` pulled the
    # entire body into RAM just to check the cap, which is a textbook
    # OOM / DoS vector for large hostile uploads. ``_read_capped``
    # aborts the moment the cap is exceeded.
    content = await _read_capped(file, MAX_UPLOAD_BYTES)
    if not content:
        # PR-4: was hardcoded English "empty upload". Now localized.
        raise HTTPException(
            status_code=400,
            detail=error_message("upload.empty", locale),
        )

    # Magic-byte sniff. A client that declares ``evil.pdf`` but sends
    # an .exe body (or vice versa) would otherwise be routed to the
    # wrong parser and waste an embedding pass. ``_sniff_format``
    # returns False if the head bytes don't match the declared
    # extension's signature — reject early.
    declared_ext = Path(fname).suffix.lower()
    if declared_ext and not _sniff_format(content[:_SNIFF_BYTES], declared_ext):
        # v1.1.12 — wrap the error so the response carries a
        # structured ``diagnosis`` object alongside the human
        # ``detail`` string. Starlette's HTTPException passes dict
        # ``detail`` through verbatim; FastAPI's default exception
        # handler renders it as JSON. Clients that pre-v1.1.12 only
        # read ``detail`` keep working; clients that want to display
        # the canonical reason or hint can read ``diagnosis``.
        #
        # PR-4: the message field is now localized; ``diagnosis``
        # stays English-by-design (operator-facing log key, not user
        # copy).
        raise HTTPException(
            status_code=400,
            detail={
                "message": error_message(
                    "upload.sniff_mismatch", locale, ext=declared_ext
                ),
                "diagnosis": _sniff_failure_diagnosis(
                    declared_ext, content[:_SNIFF_BYTES]
                ),
            },
        )

    doc_id = new_id()
    # v2.0.27.1 P1-P4 — fire-and-forget ingest task (was
    # ``BackgroundTasks.add_task``). The previous pattern blocked the
    # response close until ingest completed; for a 50 MB PDF that's
    # several seconds of client-perceived latency on what should be an
    # "ack + return doc_id" endpoint. ``create_task`` returns
    # immediately and lets the heavy work (parsing + chunking +
    # embedding + LanceDB write) run on a worker thread in parallel.
    # The task is tracked in ``_tracked_ingest_tasks`` so the lifespan
    # can cancel in-flight work on shutdown without leaving a half-
    # done writer holding a worker thread. Self-removing done
    # callback keeps the set bounded across many uploads.
    # ``_log_ingest_failure`` mirrors ``models._log_failure`` —
    # swallow CancelledError as info, log any other exception via
    # ``logger.exception`` so the operator sees unexpected failures.
    # PR-4: forward the locale so the empty-status message written to
    # the registry matches the user's UI language.
    task = asyncio.create_task(
        _ingest_file_async(doc_id, fname, content, safe_tid, locale)
    )
    _tracked_ingest_tasks.add(task)
    task.add_done_callback(_tracked_ingest_tasks.discard)
    task.add_done_callback(_log_ingest_failure)
    return UploadResponse(
        doc_id=doc_id, filename=fname, chunk_count=-1, thread_id=safe_tid or None
    )


@router.get("", response_model=list[DocumentSummary])
async def docs(thread_id: str | None = None) -> list[DocumentSummary]:
    safe_tid = _validate_thread_id(thread_id) if thread_id else None
    docs = list_documents(thread_id=safe_tid if thread_id else None)
    return [DocumentSummary(**d) for d in docs]


@router.delete("/{doc_id}")
async def remove(doc_id: str) -> dict:
    delete_by_doc_id(doc_id)
    # Also drop the registry entry. Without this, deleting an "empty"
    # doc (which has no chunks) would silently leave a stale row in
    # ``_registry.json`` that the next ``list_documents`` would resurrect.
    doc_registry.remove(doc_id)
    return {"ok": True, "doc_id": doc_id}


@router.post("/clear-thread/{thread_id}")
async def clear_thread(thread_id: str) -> dict:
    """Delete every chunk tagged with ``thread_id``. Used by the sidebar's
    'delete conversation + docs' action so removing a thread also cleans
    up its uploaded files (separate from the chat-history clear endpoint
    which only removes checkpoint rows).
    """
    safe_tid = _validate_thread_id(thread_id)
    if not safe_tid:
        raise HTTPException(status_code=400, detail="thread_id required")
    deleted = delete_by_thread_id(safe_tid)
    # Drop registry entries too so we don't surface "empty" docs that
    # belonged to a deleted conversation.
    registry_dropped = doc_registry.remove_thread(safe_tid)
    return {
        "ok": True,
        "thread_id": safe_tid,
        "rows_deleted": deleted,
        "registry_entries_dropped": registry_dropped,
    }