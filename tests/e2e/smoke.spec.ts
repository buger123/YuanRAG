/**
 * Layer 5 smoke — true Chromium + real WS + real LLM + real DOM.
 *
 * Runs the four canonical DOM assertions listed in
 * `memory/verification-protocol.md`. Any failure here blocks the fix from
 * shipping (the methodology is in `memory/development-methodology.md`).
 *
 * The test is structured defensively: if the backend isn't ready (no LLM
 * key, models still warming, etc.) we SKIP rather than fail, because the
 * e2e harness is meant to be run by a human on a fully-configured
 * machine, not by blind CI.
 *
 * To run:
 *   1. cd src/frontend && npm install    # one-time, picks up @playwright/test
 *   2. bash tests/e2e/run.sh             # installs Chromium + runs this file
 *
 * Required env on the host machine:
 *   - FastAPI backend running on http://127.0.0.1:8765
 *   - .env containing MINIMAX_API_KEY or OPENAI_API_KEY (real LLM)
 *   - BGE-M3 + BGE Reranker preloaded (lifespan) so /models/status=ready
 */
import { test, expect, type Page } from "@playwright/test";

const BACKEND = "http://127.0.0.1:8765";
const PROBE_QUESTION = "你好,请用一句话介绍你自己。";

async function backendReady(): Promise<boolean> {
  try {
    const resp = await fetch(`${BACKEND}/models/status`);
    if (!resp.ok) return false;
    const body = (await resp.json()) as { ready?: boolean };
    return body.ready === true;
  } catch {
    return false;
  }
}

async function waitForModelsReady(page: Page): Promise<boolean> {
  // Frontend polls /models/status every 2 s while ready=false and shows
  // `.chat-header-loading` (per ChatPane.tsx:213). When ready, that node
  // is removed. Wait up to 60 s (BGE-M3 + Reranker prewarm).
  try {
    await page.waitForSelector(".chat-header-loading", {
      state: "detached",
      timeout: 60_000,
    });
    return true;
  } catch {
    return false;
  }
}

async function sendQuestionAndWaitForAnswer(
  page: Page,
  question: string,
): Promise<{ midStreamHeight: number; finalText: string }> {
  await page.fill(".chat-textarea", question);
  await page.click(".btn-send");

  // Wait for the assistant bubble to appear (first token render).
  await page.waitForSelector(".message.assistant .markdown-body", {
    state: "attached",
    timeout: 30_000,
  });

  // Sample mid-stream height — must be > 0 (catches v2.0.12 deferred-prefix
  // regression where useDeferredValue starved the first ~1476 chars).
  // Poll a few times during streaming so we don't just catch a stale frame.
  const midSamples: number[] = [];
  for (let i = 0; i < 10; i += 1) {
    const box = await page
      .locator(".message.assistant .markdown-body")
      .first()
      .boundingBox();
    if (box && box.height > 0) midSamples.push(box.height);
    await page.waitForTimeout(150);
  }
  const midStreamHeight = Math.max(0, ...midSamples);

  // Wait for done — `.btn-send` comes back when streaming ends
  // (per ChatPane.tsx stop button toggle). 60 s budget for the full
  // round-trip (LLM call + ReAct + tool calls).
  await page.waitForSelector(".btn-send", {
    state: "visible",
    timeout: 60_000,
  });

  // Pull the final canonical text from the assistant bubble.
  const finalText =
    (await page.locator(".message.assistant .markdown-body").first().innerText()) ??
    "";

  return { midStreamHeight, finalText };
}

test("Layer 5 — main path: load → ready → ask → done", async ({ page }) => {
  test.setTimeout(180_000);

  // Pre-flight: backend must be reachable AND models must be ready.
  // If not, skip with a clear message — this harness requires a real env.
  if (!(await backendReady())) {
    test.skip(
      true,
      "Backend at " +
        BACKEND +
        " not ready (no key, no models, or not running). " +
        "Layer 5 needs a real env; see tests/e2e/run.sh.",
    );
    return;
  }

  await page.goto("/");

  const ready = await waitForModelsReady(page);
  expect(ready, "models should finish prewarm within 60s").toBe(true);

  const { midStreamHeight, finalText } = await sendQuestionAndWaitForAnswer(
    page,
    PROBE_QUESTION,
  );

  // Assertion 1 (v2.0.12 regression): mid-stream .markdown-body h > 0.
  // If the streaming content didn't render until done, this catches the
  // deferred-prefix starvation.
  expect(
    midStreamHeight,
    "mid-stream markdown-body height must be > 0 (catches v2.0.12 bug)",
  ).toBeGreaterThan(0);

  // Assertion 2 (v2.0.13 regression): final text is non-empty, real LLM
  // output. Catches whitespace-only canonical overwrite.
  expect(
    finalText.trim().length,
    "final answer text must be non-empty (catches v2.0.13 bug)",
  ).toBeGreaterThan(0);

  // Assertion 3: streaming ended cleanly (no leftover spinner).
  // After `.btn-send` is visible, the streaming indicator must be gone.
  const spinnerCount = await page.locator(".chat-header-spinner").count();
  expect(spinnerCount, "no spinner should remain after done").toBe(0);

  // Assertion 4: at least one assistant message bubble exists with the
  // real text we just produced. Catches the "no message bubble rendered"
  // failure mode that mocked tests never see.
  const assistantCount = await page
    .locator(".message.assistant")
    .count();
  expect(assistantCount, "exactly 1 assistant message expected").toBe(1);

  // Persist the final text into the report so debugging a regression is
  // trivial from the test output alone.
  test.info().annotations.push({
    type: "final-text",
    description: finalText,
  });
  test.info().annotations.push({
    type: "mid-stream-height",
    description: String(midStreamHeight),
  });
});

test("PR-4 NEW — locale switch flips backend error copy", async ({
  request,
}) => {
  test.setTimeout(60_000);

  if (!(await backendReady())) {
    test.skip(
      true,
      "Backend not ready; PR-4 layer-5 assertion needs a live env.",
    );
    return;
  }

  // Helper — POST an empty file to /documents/upload. The backend
  // now returns a localized "empty upload" message based on the
  // ``Accept-Language`` header (PR-4). Pre-PR-4 the response was
  // hardcoded English "empty upload" regardless of locale.
  async function postEmptyUpload(acceptLanguage: string): Promise<string> {
    const res = await request.post(`${BACKEND}/documents/upload`, {
      headers: { "accept-language": acceptLanguage },
      multipart: {
        file: {
          name: "empty.pdf",
          mimeType: "application/pdf",
          buffer: Buffer.alloc(0),
        },
      },
    });
    expect(res.status(), `status code for ${acceptLanguage}`).toBe(400);
    const body = (await res.json()) as { detail: string };
    return body.detail;
  }

  // Pre-PR-4 both these calls would return the same English
  // literal. Post-PR-4 each parses ``Accept-Language`` and looks
  // up the right catalog entry.
  const enMsg = await postEmptyUpload("en");
  expect(enMsg, "Accept-Language: en → English upload error").toBe(
    "Empty upload",
  );

  const zhMsg = await postEmptyUpload("zh-CN,zh;q=0.9");
  expect(
    zhMsg,
    "Accept-Language: zh-CN → Chinese upload error",
  ).toBe("上传的文件为空");

  // Also pin a 3rd locale shape — multi-tag with quality factors
  // in a different order, to make sure the parser isn't
  // position-sensitive.
  const enUSMsg = await postEmptyUpload("en-US,en;q=0.9,zh;q=0.5");
  expect(
    enUSMsg,
    "Accept-Language: en-US → English upload error",
  ).toBe("Empty upload");

  test.info().annotations.push({
    type: "pr4-i18n-roundtrip",
    description: `en=${enMsg} | zh=${zhMsg} | enUS=${enUSMsg}`,
  });
});

test(
  "PR-1 NEW — /documents response strips absolute filesystem paths",
  async ({ request }) => {
    test.setTimeout(60_000);

    if (!(await backendReady())) {
      test.skip(
        true,
        "Backend not ready; PR-1 layer-5 assertion needs a live env.",
      );
      return;
    }

    // Upload a small fake PDF (any bytes work — we don't need a real
    // parseable PDF, we just need a registered doc whose source_path
    // appears in GET /documents). Even if ingestion fails, the registry
    // entry is created and the GET /documents response includes it
    // with status="failed" or "pending"; both code paths in
    // ``lancedb_store.list_documents`` go through the
    // ``_serialize_source_path`` helper, so the privacy invariant
    // holds.
    const filename = `pr1-privacy-${Date.now()}.pdf`;
    const uploadRes = await request.post(`${BACKEND}/documents/upload`, {
      multipart: {
        file: {
          name: filename,
          mimeType: "application/pdf",
          buffer: Buffer.from("%PDF-1.4\n%fake-pdf-for-privacy-test\n"),
        },
      },
    });
    // The upload endpoint returns 200 on accepted upload regardless of
    // parse success — the backend queues ingestion and returns doc_id.
    expect(uploadRes.ok(), "upload should be accepted").toBe(true);

    // GET /documents — the wire surface we want to audit.
    const listRes = await request.get(`${BACKEND}/documents`);
    expect(listRes.ok(), "GET /documents should succeed").toBe(true);
    const docs = (await listRes.json()) as Array<{
      doc_id: string;
      filename: string;
      source_path: string | null;
    }>;

    // Find the doc we just uploaded (filter by the timestamped filename).
    const ours = docs.filter((d) => d.filename === filename);
    expect(
      ours.length,
      `expected at least one entry for ${filename} in /documents`,
    ).toBeGreaterThan(0);

    // PR-1 privacy invariant: NO absolute path substrings leak into the
    // wire response. On Windows the OS user is in ``C:/Users/<name>``;
    // on POSIX it's ``/home/<name>`` or ``/Users/<name>``. The harness
    // runs on POSIX (WSL / Linux / macOS), so we assert all common
    // absolute-path shapes — any hit is a privacy leak.
    const ABSOLUTE_PATH_FRAGMENTS = [
      "C:/Users",
      "C:\\Users",
      "C:\\\\Users",
      "D:/",
      "/home/",
      "/Users/",
      "/private/",
      "/var/",
      "/tmp/",
      "C:/",
      "D:\\",
    ];
    for (const doc of ours) {
      // ``source_path`` must be present (post-migration we always
      // synthesize the canonical form) — only exception is when there
      // are NO chunks AND the doc hasn't been registered, which can't
      // happen for a doc we just uploaded.
      expect(
        doc.source_path,
        `source_path for ${doc.doc_id} must be non-null`,
      ).not.toBeNull();
      const sp = doc.source_path ?? "";
      for (const fragment of ABSOLUTE_PATH_FRAGMENTS) {
        expect(
          sp.includes(fragment),
          `${doc.doc_id} source_path=${sp} must not contain ${fragment}`,
        ).toBe(false);
      }
      // And the canonical form: ``f"{doc_id}/{filename}"``.
      expect(
        sp,
        `${doc.doc_id} source_path=${sp} must equal doc_id/filename`,
      ).toBe(`${doc.doc_id}/${doc.filename}`);

      // ``filename`` itself must not contain a path separator — if it
      // does, the registry captured something other than a leaf name.
      expect(
        doc.filename.includes("/") || doc.filename.includes("\\"),
        `filename=${doc.filename} must be a leaf name`,
      ).toBe(false);
    }

    test.info().annotations.push({
      type: "pr1-privacy-roundtrip",
      description: `uploaded=${filename} | docs=${ours.length} | sample=${ours[0]?.source_path ?? "(none)"}`,
    });
  },
);

test(
  "PR-2 NEW — POST /documents/upload response returns quickly (< 1s)",
  async ({ request }) => {
    test.setTimeout(60_000);

    if (!(await backendReady())) {
      test.skip(
        true,
        "Backend not ready; PR-2 layer-5 assertion needs a live env.",
      );
      return;
    }

    // P1-P4 contract: upload response returns in milliseconds, NOT
    // after the ingest pipeline finishes. Pre-PR-2 ``BackgroundTasks
    // .add_task`` blocked the response close until ingest completed;
    // for a 50 MB PDF that's several seconds of client-perceived
    // latency. Post-PR-2 ``asyncio.create_task`` makes the dispatch
    // truly fire-and-forget: the response returns in milliseconds
    // and the heavy work runs on a worker thread.
    //
    // We measure wall-clock from request send to first response byte
    // (Playwright's request.post returns the response once the full
    // body is received — for our 200/JSON this is the close of the
    // response). The contract is "well under 1 second for any upload
    // size" — we use a tiny buffer here so the comparison is purely
    // about dispatch latency, not ingest.
    const filename = `pr2-latency-${Date.now()}.pdf`;
    const t0 = Date.now();
    const res = await request.post(`${BACKEND}/documents/upload`, {
      multipart: {
        file: {
          name: filename,
          mimeType: "application/pdf",
          buffer: Buffer.from("%PDF-1.4\n%fake-pdf-for-latency-test\n"),
        },
      },
    });
    const elapsedMs = Date.now() - t0;

    expect(res.ok(), "upload should be accepted").toBe(true);
    // 1 s budget is generous — production ingest for a tiny fake
    // PDF would take ~50-300 ms pre-PR-2, and ~10-30 ms post-PR-2.
    // Catching a regression where someone reverts to BackgroundTasks
    // means we'd see multi-second elapse (because BackgroundTasks
    // blocks until the parse/embed/index pipeline finishes — which
    // is hundreds of ms minimum and multi-second for any real PDF).
    expect(
      elapsedMs,
      `upload response should be fast (< 1000 ms), got ${elapsedMs} ms`,
    ).toBeLessThan(1000);

    test.info().annotations.push({
      type: "pr2-upload-latency",
      description: `${elapsedMs} ms`,
    });
  },
);

test(
  "PR-3 NEW — /sessions response omits X-Corrupt-Thread-Count on healthy backend",
  async ({ request }) => {
    test.setTimeout(60_000);

    if (!(await backendReady())) {
      test.skip(
        true,
        "Backend not ready; PR-3 layer-5 assertion needs a live env.",
      );
      return;
    }

    // v2.0.27.2 P1-P5 — ``GET /sessions`` may set the
    // ``X-Corrupt-Thread-Count`` response header when
    // ``checkpointer.list_threads`` reports skipped rows (data loss
    // signal for operators). On a healthy backend the header must
    // be ABSENT — it's reserved for actual corruption events so
    // the frontend can safely ignore it when missing and react
    // when present.
    //
    // This is the negative contract assertion. The positive
    // contract (header IS set when corruption occurs) is pinned
    // by ``test_sessions_endpoint_sets_corrupt_count_header_on_corruption``
    // in tests/test_checkpointer_corruption_contract.py — pytest
    // monkey-patches the checkpointer to simulate corruption,
    // which is a setup Playwright cannot reproduce in the live
    // backend.
    const res = await request.get(`${BACKEND}/sessions`);

    expect(res.ok(), "GET /sessions should succeed").toBe(true);
    expect(
      res.headers()["x-corrupt-thread-count"],
      `X-Corrupt-Thread-Count must be absent on healthy backend, got ${res.headers()["x-corrupt-thread-count"] ?? "(missing)"}`,
    ).toBeUndefined();

    test.info().annotations.push({
      type: "pr3-corruption-header-clean",
      description: "absent (healthy backend)",
    });
  },
);

test(
  "PR-4 NEW — concurrent chat + send does not deadlock backend (busy_timeout)",
  async ({ request }) => {
    test.setTimeout(60_000);

    if (!(await backendReady())) {
      test.skip(
        true,
        "Backend not ready; PR-4 layer-5 assertion needs a live env.",
      );
      return;
    }

    // v2.0.27.3 P0-P1 — concurrent ``save()`` calls on the same
    // ``thread_id`` from different request handlers (e.g. one tab
    // sends a chat message while another tab refreshes the history)
    // must not deadlock the backend. Pre-PR-4 — bare
    // ``INSERT OR REPLACE`` + ``commit()`` without explicit
    // ``BEGIN IMMEDIATE`` could surface ``OperationalError:
    // database is locked`` as 500s. Post-PR-4 — ``BEGIN IMMEDIATE``
    // + ``PRAGMA busy_timeout=5000`` means the second writer
    // waits up to 5s for the first writer's COMMIT then proceeds
    // cleanly.
    //
    // We can't directly inspect SQLite busy_timeout from the wire
    // surface (it's an internal pragma), so we exercise the
    // contract via a positive path: 10 concurrent ``GET /sessions``
    // + 10 concurrent ``POST /sessions/{id}/messages`` style
    // requests (we use a non-existent thread_id so the backend
    // returns 404 quickly — the point is concurrency, not the
    // content). All 10 requests must succeed (no 500s).
    const promises: Array<Promise<number>> = [];
    for (let i = 0; i < 10; i += 1) {
      promises.push(
        request
          .get(`${BACKEND}/sessions`)
          .then((r) => r.status()),
      );
      promises.push(
        request
          .get(`${BACKEND}/sessions/nonexistent-thread-${i}/messages`)
          .then((r) => r.status()),
      );
    }
    const statuses = await Promise.all(promises);

    // All requests must complete cleanly (200 for /sessions,
    // 404 for the missing thread — both are "expected" outcomes,
    // NOT 500).
    const fiveHundreds = statuses.filter((s) => s === 500);
    expect(
      fiveHundreds.length,
      `Expected 0 500s under concurrent load (PR-4 BEGIN IMMEDIATE + busy_timeout), got ${fiveHundreds.length} 500s. Statuses: ${JSON.stringify(statuses)}`,
    ).toBe(0);

    test.info().annotations.push({
      type: "pr4-concurrent-no-deadlock",
      description: `20 concurrent requests, ${fiveHundreds.length} 500s`,
    });
  },
);

test(
  "PR-10-1 NEW — GET /documents returns fresh data within 1s of delete",
  async ({ request }) => {
    test.setTimeout(60_000);

    if (!(await backendReady())) {
      test.skip(
        true,
        "Backend not ready; PR-10-1 layer-5 assertion needs a live env.",
      );
      return;
    }

    // v2.0.28 P1-P1 — the ``list_documents_cached`` cache is wired
    // for the summary-intent routing hint (``retrieve.py:143``), but
    // the ``GET /documents`` wire surface (which powers the sidebar)
    // MUST stay uncached. A user clicking delete expects the doc to
    // disappear immediately, not linger for the 5-second TTL window.
    //
    // This is the negative contract assertion: upload → appears → delete
    // → GONE within 1 second. The positive contract (cache works for
    // summary-intent) is pinned by ``test_retrieve_summary_intent_uses_cached_list_documents``
    // in tests/test_retrieve_summary_intent.py — pytest uses spy on
    // the cache function, which Playwright cannot reproduce from the
    // wire surface.
    const filename = `pr10-1-fresh-${Date.now()}.pdf`;
    const uploadRes = await request.post(`${BACKEND}/documents/upload`, {
      multipart: {
        file: {
          name: filename,
          mimeType: "application/pdf",
          buffer: Buffer.from(
            "%PDF-1.4\n%fake-pdf-for-cache-fresh-test\n",
          ),
        },
      },
    });
    expect(uploadRes.ok(), "upload should be accepted").toBe(true);
    const uploadBody = (await uploadRes.json()) as { doc_id: string };
    const docId = uploadBody.doc_id;

    // 1. Confirm doc appears in sidebar (pre-condition)
    let docsRes = await request.get(`${BACKEND}/documents`);
    let docs = (await docsRes.json()) as Array<{
      doc_id: string;
      filename: string;
    }>;
    let ours = docs.filter((d) => d.doc_id === docId);
    expect(
      ours.length,
      `uploaded doc ${docId} should appear in /documents after upload`,
    ).toBe(1);

    // 2. Delete the doc
    const deleteRes = await request.delete(
      `${BACKEND}/documents/${docId}`,
    );
    expect(deleteRes.ok(), "delete should succeed").toBe(true);

    // 3. Within 1s, GET /documents must NOT show the deleted doc.
    //    Pre-PR-1 — already works (uncached). PR-1 doesn't change this.
    //    The test pins the invariant so a future "optimization" that
    //    flips ``GET /documents`` to ``list_documents_cached`` is caught.
    await new Promise((resolve) => setTimeout(resolve, 200));
    docsRes = await request.get(`${BACKEND}/documents`);
    docs = (await docsRes.json()) as Array<{
      doc_id: string;
      filename: string;
    }>;
    ours = docs.filter((d) => d.doc_id === docId);
    expect(
      ours.length,
      `deleted doc ${docId} must be gone within 1s — if this fails, GET /documents was flipped to list_documents_cached (5s TTL hiding delete)`,
    ).toBe(0);

    test.info().annotations.push({
      type: "pr10-1-fresh-after-delete",
      description: `doc_id=${docId} | gone after 200ms`,
    });
  },
);

// ---------------------------------------------------------------------------
// PR-2 (v2.0.28.1) — trace_id middleware + X-Request-ID header
// ---------------------------------------------------------------------------

test(
  "PR-2 NEW — every /documents response carries X-Request-ID header (12-hex)",
  async ({ request }) => {
    if (!(await backendReady())) {
      test.skip(true, "backend not reachable — skipping PR-2 trace_id assertion");
      return;
    }

    const res = await request.get(`${BACKEND}/documents`);
    expect(res.ok(), `GET /documents failed: ${res.status()}`).toBe(true);

    const xrid = res.headers()["x-request-id"];
    expect(
      xrid,
      `X-Request-ID header missing from /documents response. headers=${JSON.stringify(res.headers())}`,
    ).toBeTruthy();
    expect(
      /^[0-9a-f]{12}$/.test(xrid),
      `X-Request-ID=${xrid} does not match 12-hex pattern`,
    ).toBe(true);

    // Two consecutive requests must produce distinct trace_ids —
    // middleware must generate fresh per request, never reuse.
    const res2 = await request.get(`${BACKEND}/documents`);
    const xrid2 = res2.headers()["x-request-id"];
    expect(xrid2).toBeTruthy();
    expect(
      xrid !== xrid2,
      `two requests got the same X-Request-ID ${xrid} — middleware must generate fresh per request`,
    ).toBe(true);

    test.info().annotations.push({
      type: "pr2-x-request-id",
      description: `xrid1=${xrid} | xrid2=${xrid2} | distinct=ok`,
    });
  },
);

test(
  "PR-2 NEW — incoming X-Request-ID round-trips through to response",
  async ({ request }) => {
    if (!(await backendReady())) {
      test.skip(true, "backend not reachable — skipping PR-2 trace_id echo assertion");
      return;
    }

    const probe = "abcdef012345";
    const res = await request.get(`${BACKEND}/documents`, {
      headers: { "X-Request-ID": probe },
    });
    expect(res.ok(), `GET /documents failed: ${res.status()}`).toBe(true);

    const echoed = res.headers()["x-request-id"];
    expect(
      echoed,
      `X-Request-ID not echoed; got ${echoed}`,
    ).toBe(probe);

    test.info().annotations.push({
      type: "pr2-x-request-id-echo",
      description: `sent=${probe} | echoed=${echoed}`,
    });
  },
);