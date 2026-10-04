# Eval v1 Fixtures — Authoring Guide

YuanRAG v2.0.32.0 — 固定测试集 (golden + adversarial),用于固定回归 + 量化
LLM 幻觉率 / 工具调度 / 引用完整性 / 延迟 / 成本。

## 目录结构

```
tests/fixtures/eval_v1/
  _README.md                       # 本文件
  cases/
    golden.yaml                    # 10 golden cases (happy-path)
    adversarial.yaml               # 22 adversarial cases (Phase 8 6 根因组)
  corpora/
    1.txt                          # 腾讯新闻 (txt)
    2025年上...准考证.pdf          # CET-6 准考证 (pdf)
    Python.md                      # Python 学习路径 (md)
    dy作文.docx                    # 高考作文「通话膨胀」 (docx)
    index.html                     # 网页 (html)
    个人月度收支预算管理表.xlsx     # 预算表 (xlsx)
    如戏摄影剧作工作室.pptx        # 摄影剧本杀 PPT (pptx)
```

## Case YAML schema

```yaml
---
id: <kebab-case-id>
language: zh | en                  # 严格 zh/en (i18n 对称)
difficulty: easy | medium | hard
thread_id: <stable-id>             # runner 加 "eval-<uuid8>" 前缀防碰撞
root_cause_group: null | A | B | C | D | E | F
                                    # null=golden, A-F=对抗 6 根因组
fixture_docs:                      # 相对 corpora/ 的文件名,自动上传
  - 1.txt
tags: [simple_fact, qa_complex, ...]

turns:
  - user: "..."            # 必填
    high_precision: auto | on | off   # 默认 "auto"
    expected_answer_contains: [substring, ...]
    expected_answer_lacks:    [substring, ...]
    forbidden_phrases:        [substring, ...]
    expected_citations:       [chunk_id_substring, ...]
    expected_grounding: grounded | ungrounded | skipped
    expected_route_decision: direct | retrieve | extractive
    expected_tool_call_count: 0..N
    expected_tool_sequence: [intent_analysis, react_agent, ...]
                                    # canonical FSM node names
    expected_refusal:
      template: refusal_empty | refusal_empty_bulk | refusal_low_relevance
      contains_any: [substring, ...]
      contains_all: [substring, ...]
      min_count: 0..N

scoring_weights:
  programmatic_pass: 0.5
  cheap_llm_pass: 0.5
  threshold_pass: 0.7
```

## Suite 加载

- `tests/eval/cases.py::load_suite("golden")` → 10 cases
- `tests/eval/cases.py::load_suite("adversarial")` → 22 cases

## 跑测试

```bash
# CLI (against real backend on :8765)
python -m tests.eval.run --suite golden
python -m tests.eval.run --suite adversarial

# Mocked unit tests (CI default)
pytest tests/test_eval_runner.py -v
```

## i18n 对称

当前 Phase 1 ship 仅 zh(per 2026-10-04 user 决定)。所有 case `language: zh`。
Phase 2 扩 en 时按 schema mirror 即可,新增 case 不改 fixture 文件。

## 调试

- 报告位置:`tests/eval/reports/<timestamp>.{jsonl,rollup.json,md}`
- console 输出含 success_rate / hallucination_rate / worst-3 failing cases
- 失败 case JSONL 含 events 完整 log 便于 trace

## 添加新 case

1. 选 corpus 文件(已有 7 格式,新格式需先 ship 进 src/api/ingestion)
2. 写 YAML 进 `cases/<suite>.yaml`
3. 在 case 上加注释说明目的
4. 不引入对生产代码的修改 — 只新增从 fixtures/