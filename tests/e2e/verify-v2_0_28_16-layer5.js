// v2.0.28.16 Layer 5 verifier — synthesis LLM tool-result utilization.
//
// User report (2026-09-23 23:44): asking "现在是几点" in a multi-turn
// thread (after warmup turns ending with a chat wrap-up) returned a
// chat reply ("早点休息吧,晚安") instead of the time data the
// get_current_time TM had returned. 5-run stochastic probe confirmed
// 2/5 bug rate.
//
// v2.0.28.16 fix: belt-and-suspenders —
//   (a) preventive post-script SystemMessage appended AFTER the
//       sanitized history, right before the LLM responds.
//   (b) defensive fingerprint-match override: if the synthesis answer
//       lacks the time marker extracted from the get_current_time TM,
//       override with a constructed fallback answer.
//
// This verifier reproduces the user's pattern 5 times and asserts
// the synthesis answer ALWAYS includes the time data on every run.
// Pre-fix this assertion failed ~40% of the time.

const path = require("path");
const { chromium } = require(path.join(
  "D:/prep for work/Projects/YuanRAG/src/frontend",
  "node_modules",
  "playwright"
));

const BACKEND = "http://127.0.0.1:8765";
const N_RUNS = 5;

function log(...a) {
  console.log(...a);
}

function nowIso() {
  // Local-time ISO for fingerprint matching against the get_current_time
  // tool output. The tool returns ``2026-09-23T23:44:26+08:00``, and our
  // fingerprint regex picks up ``YYYY-MM-DD`` first.
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return (
    d.getFullYear() +
    "-" +
    pad(d.getMonth() + 1) +
    "-" +
    pad(d.getDate())
  );
}

async function freshThread(page) {
  // Clear the active thread so the next send creates a fresh thread.
  // Without this, all 5 runs would share one thread and TURN 1 from
  // RUN 1 would leak into RUN 2's history (defeating the test). The
  // frontend key is ``rag.active_thread`` (confirmed via probe).
  await page.evaluate(() => {
    try {
      window.localStorage.removeItem("rag.active_thread");
    } catch {}
  });
  await page.reload({ waitUntil: "domcontentloaded" });
  await page.waitForSelector(".chat-header-loading", {
    state: "detached",
    timeout: 90_000,
  });
}

async function sendAndWait(page, query) {
  await page.fill(".chat-textarea", query);
  await page.click(".btn-send");
  // Poll until the assistant bubble stops growing (stream complete).
  let lastText = "";
  for (let i = 0; i < 60; i++) {
    await page.waitForTimeout(2000);
    const t = await page.evaluate(() => {
      const asst = document.querySelectorAll(".message.assistant");
      const last = asst[asst.length - 1];
      return last ? last.textContent : "";
    });
    if (t === lastText && i > 3) break;
    lastText = t;
  }
}

async function runOneIteration(page, runNumber) {
  log(`\n=== v2.0.28.16 RUN ${runNumber}/${N_RUNS} ===`);
  await freshThread(page);

  // Pattern matches user-reported scenario — multi-turn thread ending
  // with a chat wrap-up, then a factual "现在几点" question.
  await sendAndWait(page, "你是谁");
  await sendAndWait(page, "我是谁");
  await sendAndWait(page, "现在是几点");

  // Assertion 1 — live: TURN 3 synthesis answer includes a time pattern.
  const liveAnswer = await page.evaluate(() => {
    const asst = Array.from(document.querySelectorAll(".message.assistant"));
    const last = asst[asst.length - 1];
    if (!last) return null;
    const body = last.querySelector(".markdown-body");
    return {
      text: body ? body.textContent.trim() : last.textContent.trim(),
    };
  });

  if (!liveAnswer) {
    log(`  ✗ TURN 3 no assistant bubble found`);
    return { pass: false, failReason: "no assistant bubble" };
  }
  // Time pattern — accept either YYYY-MM-DD, HH:MM[:SS], or M月D日.
  const timeRe = /\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}(?::\d{2})?|\d{1,2}月\d{1,2}日/;
  const hasTime = timeRe.test(liveAnswer.text);
  log(
    `  Live answer (TURN 3): ${liveAnswer.text.slice(0, 120)}${liveAnswer.text.length > 120 ? "…" : ""}`
  );
  log(`  Has time pattern? ${hasTime ? "✓" : "✗ BUG REPRODUCED"}`);
  if (!hasTime) {
    return { pass: false, failReason: "live answer lacks time pattern", liveAnswer };
  }

  // Assertion 2 — live: tool card shows get_current_time ✓ completed
  // with elapsed_ms populated.
  const toolCardOk = await page.evaluate(() => {
    const cards = Array.from(document.querySelectorAll(".tool-call-card"));
    if (!cards.length) return false;
    // Find the LAST get_current_time card (TURN 3's call).
    let lastGct = null;
    for (const c of cards) {
      const text = c.textContent || "";
      if (text.includes("get_current_time")) lastGct = c;
    }
    if (!lastGct) return false;
    // Status: card itself has class "ok" (not a child element). v2.0.28.13
    // contract — see ToolCallCard.running = call.ok === undefined.
    const hasCheck = lastGct.classList.contains("ok");
    const hasElapsed = /\d+\s*ms/.test(lastGct.textContent || "");
    return hasCheck && hasElapsed;
  });
  log(`  Tool card (get_current_time) ✓ completed + elapsed_ms? ${toolCardOk ? "✓" : "✗"}`);
  if (!toolCardOk) {
    return { pass: false, failReason: "tool card incomplete" };
  }

  // Assertion 3 — wire: GET /sessions/{tid}/messages returns 5
  // assistant records (TURN 1 synthesis + TURN 2 synthesis + TURN 3
  // planning-step + TURN 3 synthesis = 4 — actually the per-turn shape
  // per v2.0.28.10/14 contract is [HM, AIM_planning, TM, AIM_synthesis].
  // TURN 1 and TURN 2 are text-only so they're each 1 AIMessage.
  // TURN 3 has planning + synthesis. So total = 4 AIMessages).
  const threadId = await page.evaluate(() => {
    try {
      return window.localStorage.getItem("rag.active_thread");
    } catch {
      return null;
    }
  });
  if (!threadId) {
    log(`  ✗ no thread_id in localStorage`);
    return { pass: false, failReason: "no thread_id" };
  }
  const wireRes = await page.evaluate(async (tid) => {
    try {
      const r = await fetch(`/sessions/${tid}/messages`);
      if (!r.ok) return { ok: false, status: r.status };
      const data = await r.json();
      return { ok: true, data };
    } catch (e) {
      return { ok: false, error: String(e) };
    }
  }, threadId);
  if (!wireRes.ok) {
    log(`  ✗ GET /sessions/${threadId}/messages failed:`, wireRes);
    return { pass: false, failReason: "wire fetch failed" };
  }
  // Wire shape is a plain JSON array of records (NOT an object with
  // ``records``/``messages`` field — confirmed via probe
  // tests/e2e/debug-wire-shape.js: `/sessions/{tid}/messages` returns
  // `[{"role":"user",...}, {"role":"assistant",...}]` directly).
  const records = Array.isArray(wireRes.data) ? wireRes.data : [];
  const asstRecords = records.filter((r) => r.role === "assistant");
  log(`  Wire assistant records: ${asstRecords.length} (expected ≥4)`);
  if (asstRecords.length < 4) {
    return { pass: false, failReason: `only ${asstRecords.length} asst records, expected ≥4`, records: records.length };
  }

  // Assertion 4 — wire: the LAST assistant record (TURN 3 synthesis)
  // content includes the time pattern (NOT a chat reply). This is the
  // load-bearing assertion: the DB row carries the correct answer.
  const lastAsst = asstRecords[asstRecords.length - 1];
  const lastAsstContent =
    typeof lastAsst.content === "string"
      ? lastAsst.content
      : JSON.stringify(lastAsst.content || "");
  const lastAsstHasTime = timeRe.test(lastAsstContent);
  log(`  Wire TURN 3 synthesis: ${lastAsstContent.slice(0, 120)}${lastAsstContent.length > 120 ? "…" : ""}`);
  log(`  Wire TURN 3 has time pattern? ${lastAsstHasTime ? "✓" : "✗ BUG REPRODUCED on wire"}`);
  if (!lastAsstHasTime) {
    return { pass: false, failReason: "wire synthesis lacks time pattern", wireContent: lastAsstContent };
  }

  return { pass: true };
}

async function main() {
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

  await page.goto(BACKEND, { waitUntil: "domcontentloaded" });
  await page.waitForSelector(".chat-header-loading", {
    state: "detached",
    timeout: 90_000,
  });

  // Sanity — confirm backend is reachable + uvicorn is on the new code.
  // The header loading-detach wait already verifies the WS handshake
  // completed; if backend is stale, the assistant replies won't
  // include time data and run results will FAIL.

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
        failures.push({ run: i, reason: r.failReason, detail: r });
      }
    } catch (e) {
      log(`RUN ${i}/${N_RUNS}: ✗ ERROR — ${e.message}`);
      failures.push({ run: i, reason: "exception", detail: String(e) });
    }
  }

  log(`\n=== v2.0.28.16 SUMMARY ===`);
  log(`Passed: ${passedRuns}/${N_RUNS}`);
  log(`Failed: ${failures.length}/${N_RUNS}`);
  if (failures.length > 0) {
    log(`Failures:`);
    for (const f of failures) {
      log(`  RUN ${f.run}: ${f.reason}`);
    }
  }
  log(`\nFingerprint ISO date for cross-check: ${nowIso()}`);

  await browser.close();

  // Exit non-zero if not 5/5 — the bug we set out to close was 40% rate.
  if (passedRuns !== N_RUNS) {
    log(`\n✗ v2.0.28.16 verifier FAILED: ${passedRuns}/${N_RUNS} runs`);
    process.exit(1);
  }
  log(`\n✓ v2.0.28.16 verifier PASSED: ${passedRuns}/${N_RUNS} runs`);
}

main().catch((e) => {
  console.error("verifier crashed:", e.message);
  console.error(e.stack);
  process.exit(2);
});