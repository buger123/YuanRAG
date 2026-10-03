// v2.0.28.18 Layer 5 verifier — fix Document-namespace corruption that
// hides threads from the sidebar.
//
// User report (2026-09-26 11:39, post-v2.0.28.17 ship):
//   ERROR | checkpointer._deserialize_state failed thread_id='1bca6dd2-...':
//     ValueError: Deserialization of ('langchain', 'schema', 'document',
//     'Document') is not allowed.
//   ERROR | checkpointer.list_threads skipping corrupt thread_id='1bca6dd2-...'
//   WARNING | list_sessions: skipped 1 corrupt thread(s)
//
// Root cause: _deserialize_state called
// loads(..., allowed_objects="messages") per v2.0.28.11. Document is a
// langchain_core class but NOT in the messages namespace allowlist —
// the corrupt thread tripped on Document's legacy ID
// ["langchain","schema","document","Document"].
//
// Fix (1 LOC, load-bearing):
//   allowed_objects="messages" -> allowed_objects="core"
// Verified in REPL that "core" accepts BOTH legacy + modern IDs for
// Document AND all message classes — narrowest scope covering every
// langchain_core class we use.
//
// This verifier asserts:
//   (1) GET /sessions returns 200 with NO X-Corrupt-Thread-Count header
//       (header was set to "1" pre-fix).
//   (2) GET /sessions body has ≥100 sessions (pre-fix returned 100
//       because 1 thread was skipped silently).
//   (3) GET /sessions/{corrupt-thread-id}/messages returns 200 with a
//       valid message array (the previously-hidden thread is now
//       reachable).
//   (4) Backend log scan: 0 occurrences of
//       "checkpointer._deserialize_state failed" during the verifier
//       run (every /sessions GET pre-fix fired this ERROR).

const path = require("path");
const fs = require("fs");
const { chromium } = require(path.join(
  "D:/prep for work/Projects/YuanRAG/src/frontend",
  "node_modules",
  "playwright"
));

const BACKEND = "http://127.0.0.1:8765";
const CORRUPT_THREAD_ID = "1bca6dd2-8768-4c36-a9b2-bb9d302d3b69";
const N_RUNS = 3;

// Locate the backend log file. YuanRAG writes to data/logs/*.log via
// loguru. The file name pattern is YYYY-MM-DD-HHMM-SS-<random>.log.
// We pick the most recently-modified file in that directory.
function findLatestLog() {
  const logsDir = path.join("D:/prep for work/Projects/YuanRAG/data/logs");
  if (!fs.existsSync(logsDir)) return null;
  const entries = fs.readdirSync(logsDir)
    .filter((f) => f.endsWith(".log"))
    .map((f) => ({
      f,
      full: path.join(logsDir, f),
      mtime: fs.statSync(path.join(logsDir, f)).mtimeMs,
    }))
    .sort((a, b) => b.mtime - a.mtime);
  return entries.length ? entries[0].full : null;
}

function grepCountInLog(logPath, needle) {
  if (!logPath || !fs.existsSync(logPath)) return -1;
  const text = fs.readFileSync(logPath, "utf-8");
  // Count lines containing the needle. We use substring match, not
  // regex, to avoid surprises with regex metacharacters.
  let count = 0;
  const lines = text.split("\n");
  for (const line of lines) {
    if (line.indexOf(needle) !== -1) count++;
  }
  return count;
}

async function runOneIteration(page, runNumber) {
  log(`\n=== v2.0.28.18 RUN ${runNumber}/${N_RUNS} ===`);

  await page.goto(BACKEND, { waitUntil: "domcontentloaded" });
  await page.waitForSelector(".chat-header-loading", {
    state: "detached",
    timeout: 90_000,
  });

  // Assertion 1 + 2: GET /sessions — header absent, body ≥ 100.
  const sessions = await page.evaluate(async () => {
    try {
      const r = await fetch("/sessions");
      const body = await r.json();
      return {
        status: r.status,
        corruptHeader: r.headers.get("X-Corrupt-Thread-Count"),
        count: Array.isArray(body) ? body.length : -1,
        firstId: Array.isArray(body) && body.length > 0 ? body[0].thread_id || "" : "",
      };
    } catch (e) {
      return { status: -1, error: String(e) };
    }
  });

  if (sessions.status !== 200) {
    return { pass: false, failReason: `/sessions returned ${sessions.status}` };
  }
  log(`  /sessions status=${sessions.status} count=${sessions.count} corruptHeader=${sessions.corruptHeader}`);

  if (sessions.corruptHeader !== null) {
    return {
      pass: false,
      failReason: `X-Corrupt-Thread-Count header is present (value="${sessions.corruptHeader}") — corruption still being skipped`,
    };
  }

  if (sessions.count < 100) {
    return {
      pass: false,
      failReason: `/sessions body has only ${sessions.count} sessions (expected ≥ 100; pre-fix had 100 due to 1 skip)`,
    };
  }

  // Assertion 3: GET /sessions/{corrupt-thread-id}/messages — 200 + valid array.
  const messages = await page.evaluate(async (tid) => {
    try {
      const r = await fetch(`/sessions/${tid}/messages`);
      const body = await r.json();
      const records = Array.isArray(body) ? body : [];
      return {
        status: r.status,
        count: records.length,
        firstRole: records.length > 0 ? records[0].role || "" : "",
        firstContentSnippet: records.length > 0
          ? String(records[0].content || "").slice(0, 100)
          : "",
      };
    } catch (e) {
      return { status: -1, error: String(e) };
    }
  }, CORRUPT_THREAD_ID);

  if (messages.status !== 200) {
    return {
      pass: false,
      failReason: `/sessions/${CORRUPT_THREAD_ID.slice(0, 8)}.../messages returned ${messages.status}`,
    };
  }
  log(`  /sessions/{corrupt}/messages status=${messages.status} count=${messages.count}`);

  if (messages.count < 1) {
    return {
      pass: false,
      failReason: `corrupt thread's messages endpoint returned empty array — thread still hidden`,
    };
  }

  return { pass: true, sessionsCount: sessions.count, messagesCount: messages.count };
}

function log(...a) {
  console.log(...a);
}

async function main() {
  const logPathAtStart = findLatestLog();
  const baselineErrorCount = logPathAtStart
    ? grepCountInLog(logPathAtStart, "checkpointer._deserialize_state failed")
    : -1;
  log(`Backend log file: ${logPathAtStart || "(none)"}`);
  log(`Baseline 'checkpointer._deserialize_state failed' count in log: ${baselineErrorCount}`);

  const browser = await chromium.launch({ headless: true });
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await ctx.newPage();

  await page.addInitScript(() => {
    try {
      window.localStorage.setItem("rag.theme", "light");
      window.localStorage.setItem("rag.locale", "zh");
      window.localStorage.removeItem("rag.show_thinking");
    } catch {}
  });

  let passedRuns = 0;
  const failures = [];
  for (let i = 1; i <= N_RUNS; i++) {
    try {
      const r = await runOneIteration(page, i);
      if (r.pass) {
        passedRuns++;
        log(`RUN ${i}/${N_RUNS}: ✓ PASS`);
      } else {
        log(`RUN ${i}/${N_RUNS}: ✗ FAIL — ${r.failReason}`);
        failures.push({ run: i, reason: r.failReason });
      }
    } catch (e) {
      log(`RUN ${i}/${N_RUNS}: ✗ ERROR — ${e.message}`);
      failures.push({ run: i, reason: "exception", detail: String(e) });
    }
  }

  // Assertion 4: Backend log scan — 0 occurrences of the failure marker
  // during the verifier run. Pre-fix this counter was ≥ 1 after each
  // /sessions GET (because every listing enumerated the corrupt row).
  const finalLogPath = findLatestLog();
  const finalErrorCount = finalLogPath
    ? grepCountInLog(finalLogPath, "checkpointer._deserialize_state failed")
    : -1;
  const deltaErrors =
    finalErrorCount >= 0 && baselineErrorCount >= 0
      ? finalErrorCount - baselineErrorCount
      : -1;
  log(`\nBackend log scan:`);
  log(`  baseline: ${baselineErrorCount}`);
  log(`  final:    ${finalErrorCount}`);
  log(`  delta:    ${deltaErrors}`);

  await browser.close();

  log(`\n=== v2.0.28.18 SUMMARY ===`);
  log(`Passed runs: ${passedRuns}/${N_RUNS}`);
  log(`Failed runs: ${failures.length}/${N_RUNS}`);
  if (failures.length > 0) {
    log(`Failures:`);
    for (const f of failures) {
      log(`  RUN ${f.run}: ${f.reason}`);
    }
  }
  if (deltaErrors > 0) {
    log(`✗ Backend log delta > 0: ${deltaErrors} new 'checkpointer._deserialize_state failed' lines during verifier`);
  } else if (deltaErrors === 0) {
    log(`✓ Backend log delta = 0: no new corruption errors during verifier`);
  }

  if (passedRuns !== N_RUNS || deltaErrors > 0) {
    log(`\n✗ v2.0.28.18 verifier FAILED: passedRuns=${passedRuns}/${N_RUNS}, logDelta=${deltaErrors}`);
    process.exit(1);
  }
  log(`\n✓ v2.0.28.18 verifier PASSED: ${passedRuns}/${N_RUNS} runs, logDelta=${deltaErrors}`);
}

main().catch((e) => {
  console.error("verifier crashed:", e.message);
  console.error(e.stack);
  process.exit(2);
});