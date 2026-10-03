"""v2.0.29.7 (Phase 6) — Rule engine for fact-consistency checking.

The second-pass verifier (see :mod:`src.agent.verification.verifier`)
needs two layers of defense against LLM hallucination:

1. **Rule engine (this file)** — pure regex extraction of structured
   facts from the assistant's answer (dates, currency, percentages,
   article numbers) followed by a deterministic "is this value
   present in the retrieved docs?" check. ~ms cost, no LLM call.
   Catches the most common fabrication class: LLM invents a date /
   amount / clause number that's not in the corpus.

2. **Verifier LLM (verifier.py)** — second-pass cheap-model call
   that catches semantic-level hallucination the rule engine can't
   see (concept drift, paraphrase fabrication, entity confusion).

Per [[yuanrag-hallucination-optimization]] Phase 6: this is the load-
bearing floor of the post-generation safety net. Phase 1's refusal
contract catches "I don't know" cases; this catches "I made it up"
cases where the LLM pretends to know.

Design notes
------------
- Regex only, no NER / spaCy / GLiNER. Phase 9 can swap to a real
  NER model when training data is available; for now regex is
  deterministic, debuggable, and 100% reproducible across runs.
- Regex is compiled once at module import time (single source of
  truth — see :data:`_DATE_PATTERNS` etc.). Tests can import the
  compiled patterns to verify shape without re-parsing the source.
- Normalization collapses variant surface forms ("2026年9月28日" /
  "2026-09-28" / "2026/9/28" → "2026-09-28") so the cross-check
  compares canonical values, not arbitrary surface renderings.
- Conservative semantics: ``check_consistency`` only flags
  answer fields that are MISSING from the doc text. Same value in
  different contexts (e.g. different "§3" sections) are treated as
  consistent. The verifier LLM is the second-pass safety net for
  false positives from this assumption.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional


# Field-type closed vocabulary. Kept narrow on purpose: every regex
# pattern below corresponds to one of these labels. Adding a new
# field type requires updating the regex set + the
# ``_normalize_<type>`` family. Test pinning at the Literal level
# keeps the contract honest.
FieldType = Literal["date", "currency", "percentage", "article_number"]


@dataclass(frozen=True)
class ExtractedField:
    """A concrete fact-shaped token pulled out of ``text``.

    ``field_type``        — which extractor caught this token.
    ``raw_text``          — the original surface form ("¥1,000.50" /
                            "第 12 条" / "2026-09-28"). Useful for
                            human-readable debug output.
    ``normalized``        — canonical form ("1000.50" / "12" /
                            "2026-09-28") for cross-document
                            equality checks.
    ``location``          — char offset in the source text. Used to
                            preserve source order in the output list.
    """

    field_type: FieldType
    raw_text: str
    normalized: str
    location: int


@dataclass(frozen=True)
class Mismatch:
    """An answer field that doesn't appear in the retrieved docs.

    ``field``              — the extracted token from the answer.
    ``expected``           — canonical value found in doc_text, or
                            None if no match anywhere. None is the
                            common case (the answer invented a
                            fact the corpus never had).
    ``actual``             — the answer's normalized value (same as
                            ``field.normalized`` but kept on the
                            mismatch for the verifier LLM payload).
    ``severity``           — "high" for all rule-engine mismatches;
                            the verifier LLM may downgrade to "low"
                            if it judges the mismatch is benign
                            (e.g. a paraphrase that the corpus
                            technically doesn't contain but is
                            semantically equivalent).
    """

    field: ExtractedField
    expected: Optional[str]
    actual: str
    severity: Literal["low", "medium", "high"] = "high"


# ---------------------------------------------------------------------------
# Regex patterns (compiled once, exported for tests)
# ---------------------------------------------------------------------------


# Date formats ordered from most specific (ISO) to most permissive
# (CN-partial = "9月28日"). The ``_normalize_date`` function keys
# off the tuple position so this order matters.
_DATE_PATTERNS = [
    (re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"), "ISO"),
    (re.compile(r"\b\d{4}/\d{1,2}/\d{1,2}\b"), "SLASH"),
    (re.compile(r"\d{4}年\s*\d{1,2}月\s*\d{1,2}日"), "CN"),
    # CN-partial uses a negative lookbehind for "年" so it does NOT
    # double-match when the CN form already covers the substring
    # (e.g. "2026年9月28日" → CN matches, CN-partial does NOT match
    # the inner "9月28日" — that would be a false duplicate).
    (re.compile(r"(?<!年)(\d{1,2}月\s*\d{1,2}日)(?!前)"), "CN-partial"),
]


# Currency: SYMBOL first ("$100"), CN-YUAN second ("100元"), then
# USD / CNY name forms. Symbol-first ordering matters because
# "100" appears in many contexts and we want the most informative
# pattern to claim it first.
_CURRENCY_PATTERNS = [
    (re.compile(r"[¥$€£]\s*[\d,]+(?:\.\d+)?"), "SYMBOL"),
    (re.compile(r"\d[\d,]*(?:\.\d+)?\s*元\b"), "CN-YUAN"),
    (re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:美元|USD)\b"), "USD"),
    (re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:人民币|RMB|元)\b"), "CNY"),
]


# Percentage: pure digit + % sign. Trivial but a separate extractor
# because "12.5" alone (without %) is a number, not a percentage,
# and the rule engine shouldn't conflate them.
_PERCENTAGE_PATTERNS = [
    re.compile(r"\d+(?:\.\d+)?\s*%"),
]


# Article numbers — covers CN legal style (第 12 条), English
# ("Article 12", "Section 12.3"). The CN form's CN_NUM is captured
# and normalized to ASCII digits in :func:`_cn_article_num_to_ascii`.
_ARTICLE_PATTERNS = [
    re.compile(r"第\s*([一二三四五六七八九十百零\d]+)\s*条"),
    re.compile(r"Article\s+(\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"Section\s+(\d+(?:\.\d+)?)", re.IGNORECASE),
]


_CN_NUM_MAP = {
    "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}


def _cn_article_num_to_ascii(raw: str) -> str:
    """Best-effort CN article number → ASCII digit form.

    Handles single-character CN numbers (一二三…) and the common
    two-character composites (十二 = 12, 二十 = 20). Larger numbers
    (一百二十三 etc.) are NOT supported because Chinese legal
    articles rarely exceed 99; if a future corpus uses them, this
    needs a proper parser. Returns the original ``raw`` if it
    contains any digit (mixed form like "12条" already has the
    digit; no translation needed) or any character we can't
    parse (defensive — better to leave the raw form than to
    silently mangle).
    """
    if any(c.isdigit() for c in raw):
        # Mixed CN+ASCII ("十二3条") — keep the ASCII digits and
        # leave the CN chars as-is; downstream search treats this
        # as a fallback.
        return raw
    if len(raw) == 1:
        v = _CN_NUM_MAP.get(raw)
        return str(v) if v is not None else raw
    # Two-char composite: "十X" (10+X), "X十" (X*10), "X十Y" (X*10+Y).
    if len(raw) == 2 and raw[0] == "十":
        v = _CN_NUM_MAP.get(raw[1])
        return f"1{v}" if v is not None else raw
    if len(raw) == 2 and raw[1] == "十":
        v = _CN_NUM_MAP.get(raw[0])
        return f"{v}0" if v is not None else raw
    if len(raw) == 3 and raw[1] == "十":
        v0 = _CN_NUM_MAP.get(raw[0])
        v2 = _CN_NUM_MAP.get(raw[2])
        if v0 is not None and v2 is not None:
            return f"{v0}{v2}"
    return raw  # fallback — return raw form for downstream fallback search


def _normalize_date(raw: str, kind: str) -> str:
    """Normalize a date ``raw`` (per the matching regex kind) to a
    canonical ``YYYY-MM-DD`` string when possible; otherwise return
    the raw form (so :func:`check_consistency` falls back to the
    raw-text search path).

    CN-partial dates ("9月28日") have no year — we leave them as-is
    so the verifier LLM can decide whether a year was implied and
    search accordingly. This is conservative: if the answer says
    "9月28日" but the corpus says "2026年9月28日", the rule engine
    won't match, but the LLM verifier may catch it as a partial-
    match. That's intentional — we don't want to fabricate years.
    """
    if kind in ("ISO", "SLASH"):
        digits = re.sub(r"\D", "-", raw).strip("-")
        # Pad single-digit month/day to two digits for stable compare.
        parts = digits.split("-")
        if len(parts) == 3:
            try:
                return f"{int(parts[0]):04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"
            except ValueError:
                return raw
        return raw
    if kind == "CN":
        m = re.match(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日", raw)
        if m:
            try:
                return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
            except ValueError:
                return raw
    # CN-partial / unknown: leave raw so fallback search can run.
    return raw


def _normalize_currency(raw: str, kind: str) -> str:
    """Normalize a currency amount to its canonical digit form.

    Strips the symbol / unit and keeps only digits + the optional
    decimal portion. "¥1,000.50" → "1000.50"; "1,234元" → "1234".
    The leading comma-strip is important because a doc that renders
    the same amount as "$1000" should still match the answer's
    "$1,000" — both normalize to "1000".
    """
    return re.sub(r"[^\d.]", "", raw)


def _normalize_percentage(raw: str) -> str:
    """Normalize a percentage: strip whitespace, keep digit + sign.

    "12.5%" → "12.5%" (already canonical; we strip internal spaces
    so "12.5 %" / "12.5%" compare equal).
    """
    return raw.replace(" ", "")


def _normalize_article(raw: str, match_group: str) -> str:
    """Normalize an article-number capture group.

    CN articles go through ``_cn_article_num_to_ascii`` to convert
    Chinese numerals to digits; English forms already have digits
    and pass through unchanged. The leading CN form "第 X 条" may
    use either CN numerals or ASCII; we normalize the captured
    group so "第 12 条" / "第十二条" compare equal.
    """
    return _cn_article_num_to_ascii(match_group)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_fields(text: str) -> list[ExtractedField]:
    """Pull every concrete fact-shaped token out of ``text``.

    Order = source order (by ``location``). Deduplicates on
    ``(field_type, normalized)`` so the same fact mentioned twice
    in the answer ("$100... in fact, $100...") appears once. Pure
    regex, no LLM, no fuzzy match. Idempotent.

    Returns a list (possibly empty) of :class:`ExtractedField`.
    Empty input → empty output.
    """
    if not text:
        return []
    fields: list[ExtractedField] = []

    for pat, kind in _DATE_PATTERNS:
        for m in pat.finditer(text):
            fields.append(
                ExtractedField(
                    field_type="date",
                    raw_text=m.group(0),
                    normalized=_normalize_date(m.group(0), kind),
                    location=m.start(),
                )
            )

    for pat, kind in _CURRENCY_PATTERNS:
        for m in pat.finditer(text):
            fields.append(
                ExtractedField(
                    field_type="currency",
                    raw_text=m.group(0),
                    normalized=_normalize_currency(m.group(0), kind),
                    location=m.start(),
                )
            )

    for pat in _PERCENTAGE_PATTERNS:
        for m in pat.finditer(text):
            fields.append(
                ExtractedField(
                    field_type="percentage",
                    raw_text=m.group(0),
                    normalized=_normalize_percentage(m.group(0)),
                    location=m.start(),
                )
            )

    for pat in _ARTICLE_PATTERNS:
        for m in pat.finditer(text):
            fields.append(
                ExtractedField(
                    field_type="article_number",
                    raw_text=m.group(0),
                    normalized=_normalize_article(m.group(0), m.group(1)),
                    location=m.start(),
                )
            )

    fields.sort(key=lambda f: (f.location, f.field_type))
    seen: set[tuple[str, str]] = set()
    out: list[ExtractedField] = []
    for f in fields:
        key = (f.field_type, f.normalized)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def check_consistency(
    answer_fields: list[ExtractedField],
    doc_text: str,
) -> list[Mismatch]:
    """For each answer field, check whether an equivalent value
    appears in ``doc_text``. Returns one :class:`Mismatch` per
    answer field that does NOT match anything in the docs.

    Conservative semantics: presence of any equivalent value
    (canonical or raw form) counts as consistent. False positives
    are acceptable because the verifier LLM is the second-pass
    safety net; false negatives (missing a real fabrication) are
    not.

    The doc_text search uses both the normalized form AND the raw
    text — covers "the answer says 2026-09-28 and the doc says
    2026-09-28" (canonical match) as well as "the answer says
    2026年9月28日 and the doc says 2026年9月28日" (raw-text match).
    Empty answer_fields or doc_text is a no-op (returns []).
    """
    if not answer_fields or not doc_text:
        return []
    out: list[Mismatch] = []
    for f in answer_fields:
        candidates: list[str] = []
        if f.normalized:
            candidates.append(f.normalized)
        if f.raw_text and f.raw_text not in candidates:
            candidates.append(f.raw_text)
        found = any(c and c in doc_text for c in candidates)
        if not found:
            out.append(
                Mismatch(
                    field=f,
                    expected=None,
                    actual=f.normalized,
                    severity="high",
                )
            )
    return out


__all__ = [
    "FieldType",
    "ExtractedField",
    "Mismatch",
    "extract_fields",
    "check_consistency",
    # Exported for tests that want to verify the compiled regex set
    # without re-parsing the source.
    "_DATE_PATTERNS",
    "_CURRENCY_PATTERNS",
    "_PERCENTAGE_PATTERNS",
    "_ARTICLE_PATTERNS",
]