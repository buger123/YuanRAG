"""Pass 3: Claude (operator-invoked) offline verdict generator.

v2.0.32.0 — does NOT call any LLM itself. Builds a structured prompt
that the operator can paste into a Claude Code session to produce a
high-level verdict report.

The verdict JSON: ``tests/eval/reports/<timestamp>.claude.md``. Operator
runs::

    > 读 tests/eval/reports/<timestamp>.jsonl + .rollup.json,
      对每个 failing case 解释失败模式,归类到 6 根因组,
      输出到 tests/eval/reports/<timestamp>.claude.md

The helper here emits a structured "by running X" template so the
operator's prompt is concrete (not vague), and Claude's response is
shaped consistently with the rest of the eval artifacts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VERDICT_PROMPT = """\
You are reviewing a fixed-test-set evaluation of a RAG system. Each
failing case is a regression signal — your job is to explain WHY it
failed and which anti-hallucination root cause group (A-F) it maps
to, then propose a fix or a Phase-number for follow-up.

The 6 anti-hallucination root cause groups (from README.md):

  A — Silent Failure: LLM 拿到坏上下文也不报;同类 bug 一根接一根
  B — No Refusal Contract: "我也不知道" 变成自信回答;"引用过" ≠ "数字对"
  C — Loose Retrieval Gate: 过期片段当现行;长文档只看到一半
  D — Citation Integrity: 引文对但内容是 LLM 编的
  E — Observability Blindspot: 翻车没 log,无法定位
  F — Verbatim Distortion: 法律/医疗/财务场景 LLM 自主改写措辞

For each failing case, output:

  ## {case_id} ({language}/{difficulty})

  **Failure mode:** <one sentence — what went wrong>

  **Root cause group:** <A/B/C/D/E/F — pick the most specific one>

  **Proposed fix:**
  - <line 1: code-level or prompt-level fix>
  - <line 2: verification — what test would catch the regression>

  **Severity:** P0 (blocks ship) / P1 (degrades quality) / P2 (polish)

After the per-case section, output a 3-bullet "Highest-leverage fix"
summary that names the single change most likely to reduce the failure
rate by the largest margin.

The verdict goes into ``<output_path>``. The rollup JSON + JSONL are at
the paths listed below; read them before writing your verdict.
"""


@dataclass(frozen=True)
class ClaudeVerdictPrompt:
    """Pre-shaped prompt for the operator to paste into Claude Code."""

    verdict_prompt: str
    output_path: Path
    jsonl_path: Path
    rollup_path: Path

    def to_text(self) -> str:
        return (
            f"{self.verdict_prompt}\n\n"
            f"---\n\n"
            f"**JSONL (per-case events):** `{self.jsonl_path}`\n\n"
            f"**Rollup (per-suite metrics):** `{self.rollup_path}`\n\n"
            f"**Write your verdict to:** `{self.output_path}`\n"
        )


def build_claude_verdict_prompt(
    *,
    reports_dir: Path,
    timestamp: str,
) -> ClaudeVerdictPrompt:
    """Build a Claude-verdict prompt anchored on a specific report run.

    The output path mirrors the JSONL + rollup file naming convention
    from ``report.py``. Operator drops the rendered text into a Claude
    Code session; Claude reads the artifacts and produces the verdict.
    """
    jsonl_path = reports_dir / f"{timestamp}.jsonl"
    rollup_path = reports_dir / f"{timestamp}.rollup.json"
    output_path = reports_dir / f"{timestamp}.claude.md"
    return ClaudeVerdictPrompt(
        verdict_prompt=VERDICT_PROMPT,
        output_path=output_path,
        jsonl_path=jsonl_path,
        rollup_path=rollup_path,
    )


def load_rollup(path: Path) -> dict[str, Any] | None:
    """Public — for the Claude session to read the rollup first."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None