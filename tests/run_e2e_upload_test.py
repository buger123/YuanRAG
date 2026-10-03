"""End-to-end upload + RAG answer test.

For each fixture (.pdf .docx .pptx .xlsx .html .md .txt):

1. Spin up a fresh thread_id so the LLM can't read other formats' content.
2. POST /documents/upload?thread_id=... and poll until ``status == "indexed"``
   with chunk_count > 0.
3. Open WS to /ws/chat?thread_id=..., ask a question whose only correct
   answer requires quoting the sentinel token baked into the file.
4. Verify the model's reply contains that token.

Also reports per-step latency so the user can see how much each format
cost during ingestion and question.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import websockets

BASE = "http://127.0.0.1:8765"
WS_BASE = "ws://127.0.0.1:8765"
FIXTURES = Path(__file__).parent / "fixtures"

# (filename, mime, question the model can ONLY answer from the file)
CASES = [
    (
        "alpha.txt",
        "text/plain",
        "The company notes (alpha.txt) were uploaded. What is the magic "
        "sentinel token recorded in that file? Please quote it exactly.",
    ),
    (
        "bravo.md",
        "text/markdown",
        "I just uploaded the Bravo briefing markdown. Quote its sentinel "
        "token verbatim.",
    ),
    (
        "charlie.html",
        "text/html",
        "I uploaded a Charlie memo in HTML. What sentinel token does it "
        "contain? Quote it exactly.",
    ),
    (
        "delta.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "I uploaded a Delta Dossier .docx. What sentinel token is in that "
        "file? Quote it exactly.",
    ),
    (
        "echo.pptx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "Echo Overview.pptx was just uploaded. Quote its sentinel token.",
    ),
    (
        "foxtrot.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "Foxtrot.xlsx was just uploaded. What sentinel token is in it? "
        "Quote it exactly.",
    ),
    (
        "golf.pdf",
        "application/pdf",
        "A Golf Field Report.pdf was just uploaded. Quote its sentinel "
        "token verbatim.",
    ),
]

SENTINEL_BY_FILE = {
    # Binary Office formats wrap text in XML, so we hard-code. Plain text
    # formats are auto-detected via the regex below so re-builds stay in sync.
    "alpha.txt": "A1PHA-CODE-9988-A1PHA",
    "bravo.md":  "BR4VO-CODE-3322-BR4VO",
    "charlie.html": "CH4RL-CODE-7766-CH4RL",
    "delta.docx":   "DE1TA-CODE-4455-DE1TA",
    "echo.pptx":    "ECH0-CODE-1199-ECH0",
    "foxtrot.xlsx": "FOXTR-CODE-2244-FOXTR",
    "golf.pdf":     "G01F-CODE-3030-G01F",
}

SENTINEL_RX = re.compile(r"[A-Z0-9]{4,5}-CODE-\d{4}-[A-Z0-9]{4,5}")


@dataclass
class Outcome:
    filename: str
    upload_ok: bool
    indexed: bool
    chunks: int
    ingest_seconds: float
    answer: str
    answer_seconds: float
    contains_token: bool
    error: str | None = None


def sentinel_token_for(filename: str) -> str:
    """Pull the sentinel token straight out of the on-disk fixture.

    Binary Office formats (DOCX/PPTX/XLSX) wrap text in XML, so we look
    up by filename. Plain-text formats are checked dynamically so a
    rebuild via ``build_fixtures.py`` stays in sync without editing this
    driver.
    """
    if filename in SENTINEL_BY_FILE:
        return SENTINEL_BY_FILE[filename]
    raw = (FIXTURES / filename).read_bytes()
    text = raw.decode("utf-8", errors="replace")
    m = SENTINEL_RX.search(text)
    if not m:
        raise RuntimeError(f"No sentinel in {filename}")
    return m.group(0)


def upload_file(filename: str, mime: str, data: bytes, thread_id: str) -> dict:
    files = {"file": (filename, data, mime)}
    r = requests.post(
        f"{BASE}/documents/upload",
        params={"thread_id": thread_id},
        files=files,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def list_documents(thread_id: str) -> list[dict]:
    r = requests.get(
        f"{BASE}/documents",
        params={"thread_id": thread_id},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def wait_for_indexed(thread_id: str, doc_id: str, timeout: float = 180) -> tuple[int, float]:
    deadline = time.time() + timeout
    start = time.time()
    while time.time() < deadline:
        docs = list_documents(thread_id)
        for d in docs:
            if d.get("doc_id") == doc_id:
                if d.get("status") == "indexed" and d.get("chunk_count", 0) > 0:
                    return d["chunk_count"], time.time() - start
        time.sleep(0.5)
    return -1, time.time() - start


async def ask_via_ws(thread_id: str, question: str, timeout: float = 180) -> tuple[str, float]:
    url = f"{WS_BASE}/ws/chat?thread_id={thread_id}"
    text_parts: list[str] = []
    start = time.time()
    async with websockets.connect(url, open_timeout=10) as ws:
        await ws.send(json.dumps({"message": question}))
        while True:
            remaining = timeout - (time.time() - start)
            if remaining <= 0:
                break
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            if not raw:
                break
            evt = json.loads(raw)
            t = evt.get("type")
            if t == "token":
                text_parts.append(evt.get("content", ""))
            elif t == "done":
                break
            elif t == "error":
                return f"[WS error] {evt.get('message')}", time.time() - start
        return "".join(text_parts), time.time() - start


def run_case(filename: str, mime: str, question: str) -> Outcome:
    thread_id = uuid.uuid4().hex
    file_bytes = (FIXTURES / filename).read_bytes()
    sentinel = sentinel_token_for(filename)
    try:
        # 1. Upload
        up = upload_file(filename, mime, file_bytes, thread_id)
        doc_id = up["doc_id"]
        # 2. Wait for indexing
        chunks, ingest_seconds = wait_for_indexed(thread_id, doc_id)
        if chunks < 1:
            return Outcome(
                filename=filename, upload_ok=True, indexed=False, chunks=chunks,
                ingest_seconds=ingest_seconds,
                answer="", answer_seconds=0,
                contains_token=False,
                error=f"never reached status=indexed (chunks={chunks})",
            )
        # 3. Ask a question
        answer, answer_seconds = asyncio.run(ask_via_ws(thread_id, question))
        return Outcome(
            filename=filename, upload_ok=True, indexed=True, chunks=chunks,
            ingest_seconds=ingest_seconds,
            answer=answer.strip(), answer_seconds=answer_seconds,
            contains_token=sentinel in answer,
        )
    except Exception as e:
        return Outcome(
            filename=filename, upload_ok=False, indexed=False, chunks=0,
            ingest_seconds=0.0, answer="", answer_seconds=0.0,
            contains_token=False, error=repr(e),
        )


def main() -> int:
    print(f"Running {len(CASES)} upload + answer tests against {BASE}\n")
    results: list[Outcome] = []
    for fname, mime, q in CASES:
        sentinel = sentinel_token_for(fname)
        print(f"=== {fname}  (looking for token: {sentinel}) ===")
        out = run_case(fname, mime, q)
        results.append(out)
        if out.error:
            print(f"   ERROR: {out.error}")
        else:
            print(
                f"   upload={'OK' if out.upload_ok else 'FAIL'}  "
                f"indexed={out.indexed}  chunks={out.chunks}  "
                f"ingest={out.ingest_seconds:.1f}s  "
                f"answer={out.answer_seconds:.1f}s  "
                f"token_present={out.contains_token}"
            )
            snippet = (out.answer[:240] + "…") if len(out.answer) > 240 else out.answer
            for line in snippet.splitlines():
                print(f"   | {line}")
        print()

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    hdr = ("format", "up", "idx", "chunks", "ingest_s", "answer_s", "token_in_answer")
    print(f"{hdr[0]:<10} {hdr[1]:<3} {hdr[2]:<3} {hdr[3]:<7} {hdr[4]:<9} {hdr[5]:<9} {hdr[6]}")
    for r in results:
        tok = "YES" if r.contains_token else ("NO" if r.answer else "—")
        print(
            f"{r.filename:<10} "
            f"{'Y' if r.upload_ok else 'N':<3} "
            f"{'Y' if r.indexed else 'N':<3} "
            f"{r.chunks:<7d} "
            f"{r.ingest_seconds:<9.2f} "
            f"{r.answer_seconds:<9.2f} "
            f"{tok}"
            + (f"  err: {r.error[:60]}" if r.error else "")
        )

    passed = sum(1 for r in results if r.indexed and r.contains_token)
    print(f"\nPassed: {passed}/{len(results)}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
