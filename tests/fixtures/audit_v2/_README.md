# audit_v2 — Day 0 OOD Adversarial Fixtures

7 fixtures, each engineered to trigger a SPECIFIC hallucination failure mode
in the current retrieval → synthesis pipeline. Subsequent Phase 1-8 work will
add defenses that handle these modes; until then, the test
`tests/test_audit_fixture_behavior.py` documents the **baseline** behavior
the defenses must improve on.

## Index

| Fixture | File type | Failure mode (root cause group) | Targeted phase |
|---------|-----------|---------------------------------|----------------|
| `corpus_off_topic.txt` | .txt | **C. retrieval gate loose** — BM25 zero hits OR dense returns low-score hits → LLM has nothing to anchor on but synthesizes | Phase 1 (refusal contract) + Phase 4 (min_score gate) |
| `corpus_empty.pdf` | .pdf | **C. retrieval gate loose** — ingest yields 0 chunks → retrieval returns empty list → LLM synthesizes | Phase 1 (refusal template) |
| `corpus_partial.md` | .md | **B. no refusal contract** — retrieval returns 1 of 2 needed parts; LLM doesn't notice partial coverage and answers the missing part | Phase 1 (partial_coverage refusal) + Phase 6 (verification agent) |
| `corpus_injection.md` | .md | **D. citation integrity / untrusted preamble** — adversarial directive embedded in doc body; reaches LLM context verbatim | Phase 4 (`strip_untrusted` + prompt_safety UNTRUSTED_PREAMBLE detector) |
| `corpus_homoglyph.txt` | .txt | **D. dirty data** — Cyrillic `а` mixed into Latin `a`; URL spoofing / tokenization confusion | Phase 4 (homoglyph normalization at ingest) |
| `corpus_conflicting.md` | .md | **B. no refusal contract** — same query yields two opposite conclusions; RRF can't disambiguate; LLM picks one without flagging conflict | Phase 6 (verification agent detects conflicting evidence) |
| `corpus_dirty.txt` | .txt | **D. dirty data** — PDF-OCR noise (random whitespace, broken chars); BM25 still finds keywords but downstream context is contaminated | Phase 4 (OCR-cleaning in ingestion) |

## Design principle

These fixtures are **raw content** (what a real document looks like), not
synthetic chunk rows. They are intentionally short so the behavioral test
can ingest each one in <100ms. Each fixture:

- **Has a clear failure mode** that a downstream Phase 1-8 defense targets
- **Is deterministically reproducible** (no time-of-day / LLM-stochastic content)
- **Has unique, identifiable terms** so substring-pinning downstream tests
  can verify the fixture content actually reached the LLM context

## What this directory is NOT for

- **NOT a regression corpus** — those live in `tests/fixtures/` (alpha/bravo/
  charlie/delta/echo/foxtrot/golf). Those test happy-path retrieval.
- **NOT a benchmark** — these fixtures are not meant for measuring LLM
  accuracy. They are trigger devices.
- **NOT Phase 1-8 specific tests** — those tests live in
  `tests/test_refusal_contract.py` / `test_verification.py` / etc. and use
  these fixtures as input.

## Adding new fixtures

When a new hallucination failure mode is discovered in production logs,
add a fixture here. Naming convention: `corpus_<failure_mode>.<ext>`.
The behavioral test in `test_audit_fixture_behavior.py` should grow a
matching `test_corpus_<failure_mode>_<assertion>`.

Each fixture must come with:
1. A row in the table above (failure mode + targeted phase)
2. A behavioral assertion in `test_audit_fixture_behavior.py`