"""Phase 9 — YuanRAG 模型微调 (deferred MLOps effort, 非 code change).

This subpackage is a **placeholder** for future fine-tuning work.

Per [[yuanrag-hallucination-optimization]], Phase 9 is **OUT of scope** for
the current hallucination-optimization project (Day 0 + Phase 1-8 SHIPPED
2026-09-27/28). It is intentionally left empty so future sessions can find
the directory and know what to add.

## What Phase 9 needs (per project plan)

1. **Training data** ≥ 1000 high-quality samples:
   `(query, retrieved_docs, expected_answer_with_citations, refusal_indicators)`
   - Sources:
     - Phase 6 (verify_answer) failure cases — `state["verification_mismatches"]`
       collected from production turn logs
     - Phase 7 (sliding window) cases where `CHUNK_REJECTED_OVERSIZE` exceeded
       threshold (signal that LLM was missing middle-section info)
     - Phase 8 (verbatim extraction) cases where user manually toggled OFF,
       suggesting auto-detect false positives
     - Day 0 OOD fixture set (`tests/fixtures/audit_v2/`) expanded with
       real failure cases from production

2. **Evaluation scripts**:
   - `grounding_accuracy.py` — fraction of answer claims supported by retrieved docs
   - `citation_precision.py` — fraction of `[n]` citations that point to actually-cited chunks
   - `refusal_when_empty_precision.py` — fraction of empty-doc answers that use REFUSAL_TEMPLATES correctly

3. **Infrastructure** (one of):
   - OpenAI fine-tuning API (simpler, paid)
   - LoRA on self-hosted base model (cheaper, more control)
   - DPO/RLHF on top of current model (most expensive, best alignment)

4. **Pre-flight gate** — DO NOT START Phase 9 until:
   - At least 30 days of Phase 1-8 telemetry shows measurable improvement
   - The 6 root-cause groups from [[post-phase3-audit-findings]] are still
     100% closed (no regressions)
   - User explicitly approves fine-tuning budget + infrastructure

## Why deferred

Phase 1-8 ship = "fix structural defenses so the model has the best chance
to be correct by default". Phase 9 = "fine-tune the model itself to be
better at the remaining edge cases". The first 8 Phases are **deterministic
+ regression-guarded** (pytest + Layer 5). Fine-tuning is **probabilistic +
dataset-dependent** — wrong training data makes the model worse, not better.

User explicitly marked Phase 9 as OUT of scope when立项 the hallucination
project (2026-09-27). The full scope decision is in
[[yuanrag-hallucination-optimization]] § OUT (不做 / deferred).

## How to apply

- Mention "fine-tuning" / "模型微调" / "MLOps" → this file
- Want to start Phase 9 → check prerequisites above, talk to user, then
  build evaluation scripts FIRST (before any model training)
- See existing counter telemetry (`EXTRACTIVE_FALLBACK`,
  `VERIFICATION_REGENERATED`, `CHUNK_REJECTED_OVERSIZE`) for what
  failure modes are still happening after Phase 1-8 ship

This file intentionally contains no Python code. It is documentation
about future work, not shippable code.
"""