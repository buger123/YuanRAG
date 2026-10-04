"""Judge passes for the eval harness.

3 passes (per plan §D, user-approved 2026-10-03):

  1. programmatic — zero LLM, structural checks (programmatic.py)
  2. cheap_llm    — build_cheap_model accuracy/completeness/relevance (cheap_llm.py)
  3. claudemd     — operator-invoked offline verdict (claudemd.py)

Public API:
    from tests.eval.judge.programmatic import run_programmatic
    from tests.eval.judge.cheap_llm import run_cheap_llm_judge
    from tests.eval.judge.claudemd import build_claude_verdict_prompt
"""