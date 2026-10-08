"""Case dataclass + YAML loader.

v2.0.32.0 — per-case schema lives in tests/fixtures/eval_v1/cases/*.yaml.
Each case is a single conversation (single-turn or multi-turn). Loader is
strict on the field set; downstream code is permissive (extra fields are
ignored).

Authoring guide (see also tests/fixtures/eval_v1/_README.md):

  id            unique kebab-case id; used in JSONL/CSV/CLI filter
  language      "zh" | "en" — strict (per src/api/i18n.py SUPPORTED_LOCALES)
  difficulty    "easy" | "medium" | "hard" — see plan §B rubric
  root_cause_group
                null for golden; "A"-"F" for adversarial (per README.md 6
                anti-hallucination root cause groups)
  thread_id     stable id; runner prefixes with "eval-<uuid8>" to avoid collision
  fixture_docs  list of relative paths under corpora/; auto-uploaded per case
  tags          freeform list, used by --suite filter
  expected_tool_sequence
                per turn; list of canonical FSM node names:
                  intent_analysis, react_agent, react_generate,
                  react_generate_direct, react_generate_extractive,
                  verify_answer, check_hallucination,
                  summary_path, summary_intent
                Use "+" suffix for one-or-more of the previous node (loop
                collapse). Step-level events (step_start/step_end) are ignored.
  turns         ordered list of TurnAnnotation. Last turn's answer is what
           ``answer_complete`` is scored against.
  scoring_weights
                composite weighting for programmatic / cheap_llm pass.

Per-turn fields (TurnAnnotation):
  user                  str — sent as POST /chat message field
  high_precision        "auto" | "on" | "off" — forwarded to ChatRequest
  expected_answer_contains
                         list of substrings; case-insensitive OR-joined.
                         Pass = at least one is present in answer body.
  expected_answer_lacks list of substrings; pass = none is present.
  forbidden_phrases     list of substrings; pass = none is present.
                         Stronger than expected_answer_lacks because it's
                         the documented anti-pattern vocabulary.
  expected_citations    list of chunk_id substrings (substring match so
                         doc-scoped IDs survive UUID prefixes); pass =
                         every expected ID is in sources[].chunk_id.
  expected_grounding    "grounded" | "ungrounded" | "skipped"
  expected_route_decision
                         "direct" | "retrieve" | "extractive" — inferred
                         from wire events (sources presence + verbatim flag)
  expected_tool_call_count
                         int — upper bound on tool invocations
  expected_refusal      adversarial-only; dict with keys template
                         (REFUSAL_TEMPLATES key), contains_any /
                         contains_all / min_count.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import yaml

# i18n SUPPORTED_LOCALES mirror — strict validator (per [[i18n symmetric]]).
SUPPORTED_LANGUAGES: tuple[str, ...] = ("zh", "en")
DIFFICULTY_LEVELS: tuple[str, ...] = ("easy", "medium", "hard")
ROOT_CAUSE_GROUPS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
GROUDING_VALUES: tuple[str, ...] = ("grounded", "ungrounded", "skipped")
HIGH_PRECISION_VALUES: tuple[str, ...] = ("auto", "on", "off")
# v2.0.32.8 — "any" added for summary-eligible doc-questions where the
# cheap-model intent classifier is non-deterministic across qa_complex and
# summary paths but BOTH produce correct answers. The judge accepts
# direct|retrieve|extractive|generate when "any" is set.
ROUTE_DECISIONS: tuple[str, ...] = ("direct", "retrieve", "extractive", "any")

# Canonical FSM node names — keep in sync with src/agent/fsm.py:run_fsm
# docstring + the runner streaming map. The harness is read-only on this
# set; adding a new node requires adding it here too (programmatic check
# will skip unknown names with a warning).
CANONICAL_FSM_NODES: frozenset[str] = frozenset(
    {
        "intent_analysis",
        "react_agent",
        "react_generate",
        "react_generate_direct",
        "react_generate_extractive",
        "verify_answer",
        "check_hallucination",
        "summary_path",
        "summary_intent",
    }
)


# ================================================================
# Dataclasses
# ============================================================


@dataclass(frozen=True)
class ScoringWeights:
    """Composite weighting for the per-case score.

    Both programmatic_pass and cheap_llm_pass are 0-1 (clamped).
    threshold_pass is the pass/fail cut-off for the composite.
    """

    programmatic_pass: float = 0.5
    cheap_llm_pass: float = 0.5
    threshold_pass: float = 0.7

    def clamped(self) -> "ScoringWeights":
        return ScoringWeights(
            programmatic_pass=_clamp01(self.programmatic_pass),
            cheap_llm_pass=_clamp01(self.cheap_llm_pass),
            threshold_pass=_clamp01(self.threshold_pass),
        )


@dataclass(frozen=True)
class ExpectedRefusal:
    """Adversarial-only — what refusal template should fire.

    REFUSAL_TEMPLATES keys: refusal_empty, refusal_empty_bulk,
    refusal_low_relevance. ``contains_any`` / ``contains_all`` test the
    answer body; ``min_count`` checks vocab-cardinality. All optional —
    pass condition is ``template matches AND contains_all ⊆ answer``.
    """

    template: Literal["refusal_empty", "refusal_empty_bulk", "refusal_low_relevance"]
    contains_any: tuple[str, ...] = ()
    contains_all: tuple[str, ...] = ()
    min_count: int = 0


@dataclass(frozen=True)
class TurnAnnotation:
    """Per-turn expectations. Last turn's answer is the scored one."""

    user: str
    high_precision: Literal["auto", "on", "off"] = "auto"
    expected_answer_contains: tuple[str, ...] = ()
    expected_answer_lacks: tuple[str, ...] = ()
    forbidden_phrases: tuple[str, ...] = ()
    expected_citations: tuple[str, ...] = ()
    expected_grounding: str = "skipped"
    expected_route_decision: str = "direct"
    expected_tool_call_count: int = 0
    expected_tool_sequence: tuple[str, ...] = ()
    expected_refusal: Optional[ExpectedRefusal] = None


@dataclass(frozen=True)
class Case:
    """One conversation. Single-turn or multi-turn."""

    id: str
    language: Literal["zh", "en"]
    difficulty: Literal["easy", "medium", "hard"]
    thread_id: str
    turns: tuple[TurnAnnotation, ...]
    fixture_docs: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    root_cause_group: Optional[str] = None
    scoring_weights: ScoringWeights = field(default_factory=ScoringWeights)
    # Suite tag — set by load_suite() so downstream consumers know if this
    # is a golden or adversarial case. Not user-authored.
    suite: str = "unknown"


# ================================================================
# Loader
# ============================================================


class CaseLoadError(ValueError):
    """Raised on any malformed case YAML. Caller should fail the suite."""


# v2.0.32.0 — strict safe_load. Per Risk §8 of the plan, NEVER switch to
# ``yaml.load()`` — a malicious case fixture could construct arbitrary
# Python objects. We allow only standard YAML scalars / sequences / maps.
def _yaml_load_safe(text: str, *, source: str) -> Any:
    return yaml.safe_load(text)


def _validate_locale(value: Any, *, field: str, source: str) -> str:
    if value not in SUPPORTED_LANGUAGES:
        raise CaseLoadError(
            f"{source}: {field!r} must be one of {SUPPORTED_LANGUAGES}, "
            f"got {value!r}"
        )
    return value


def _validate_difficulty(value: Any, *, field: str, source: str) -> str:
    if value not in DIFFICULTY_LEVELS:
        raise CaseLoadError(
            f"{source}: {field!r} must be one of {DIFFICULTY_LEVELS}, "
            f"got {value!r}"
        )
    return value


def _validate_root_cause_group(value: Any, *, source: str) -> Optional[str]:
    if value is None:
        return None
    if value not in ROOT_CAUSE_GROUPS:
        raise CaseLoadError(
            f"{source}: 'root_cause_group' must be null or one of "
            f"{ROOT_CAUSE_GROUPS}, got {value!r}"
        )
    return value


def _validate_grounding(value: Any, *, source: str) -> str:
    if value not in GROUDING_VALUES:
        raise CaseLoadError(
            f"{source}: 'expected_grounding' must be one of "
            f"{GROUDING_VALUES}, got {value!r}"
        )
    return value


def _validate_high_precision(value: Any, *, source: str) -> str:
    if value not in HIGH_PRECISION_VALUES:
        raise CaseLoadError(
            f"{source}: 'high_precision' must be one of "
            f"{HIGH_PRECISION_VALUES}, got {value!r}"
        )
    return value


def _validate_route_decision(value: Any, *, source: str) -> str:
    if value not in ROUTE_DECISIONS:
        raise CaseLoadError(
            f"{source}: 'expected_route_decision' must be one of "
            f"{ROUTE_DECISIONS}, got {value!r}"
        )
    return value


def _str_list(value: Any, *, field: str, source: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise CaseLoadError(
            f"{source}: {field!r} must be a list of strings, got {value!r}"
        )
    return tuple(value)


def _str_tuple_list(value: Any, *, field: str, source: str) -> tuple[str, ...]:
    """Turn-by-turn expected_tool_sequence — a per-turn list of node names.

    YAML author writes either:
        expected_tool_sequence:
          turn_1: [intent_analysis, react_generate_direct, ...]
          turn_2: [...]
    OR a flat list per turn at the same nesting level as other turn keys
    (legacy/simple). For now we accept the dict-of-turn form via the Case
    parser; the per-turn field is parsed from the turn sub-dict directly.
    """
    # Not used at the top level — left here for symmetry / future use.
    return _str_list(value, field=field, source=source)


def _parse_refusal(value: Any, *, source: str) -> Optional[ExpectedRefusal]:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise CaseLoadError(
            f"{source}: 'expected_refusal' must be a dict, got {type(value).__name__}"
        )
    template = value.get("template")
    if template not in ("refusal_empty", "refusal_empty_bulk", "refusal_low_relevance"):
        raise CaseLoadError(
            f"{source}: 'expected_refusal.template' must be one of "
            f"{{'refusal_empty','refusal_empty_bulk','refusal_low_relevance'}}, "
            f"got {template!r}"
        )
    return ExpectedRefusal(
        template=template,
        contains_any=tuple(value.get("contains_any") or ()),
        contains_all=tuple(value.get("contains_all") or ()),
        min_count=int(value.get("min_count", 0)),
    )


def _parse_turn(turn: Any, *, turn_idx: int, source: str) -> TurnAnnotation:
    if not isinstance(turn, dict):
        raise CaseLoadError(
            f"{source}: turn[{turn_idx}] must be a mapping, got "
            f"{type(turn).__name__}"
        )
    user = turn.get("user")
    if not isinstance(user, str) or not user.strip():
        raise CaseLoadError(
            f"{source}: turn[{turn_idx}].user must be a non-empty string"
        )
    return TurnAnnotation(
        user=user,
        high_precision=_validate_high_precision(
            turn.get("high_precision", "auto"), source=f"{source}:turn[{turn_idx}]"
        ),
        expected_answer_contains=_str_list(
            turn.get("expected_answer_contains"),
            field="expected_answer_contains",
            source=f"{source}:turn[{turn_idx}]",
        ),
        expected_answer_lacks=_str_list(
            turn.get("expected_answer_lacks"),
            field="expected_answer_lacks",
            source=f"{source}:turn[{turn_idx}]",
        ),
        forbidden_phrases=_str_list(
            turn.get("forbidden_phrases"),
            field="forbidden_phrases",
            source=f"{source}:turn[{turn_idx}]",
        ),
        expected_citations=_str_list(
            turn.get("expected_citations"),
            field="expected_citations",
            source=f"{source}:turn[{turn_idx}]",
        ),
        expected_grounding=_validate_grounding(
            turn.get("expected_grounding", "skipped"),
            source=f"{source}:turn[{turn_idx}]",
        ),
        expected_route_decision=_validate_route_decision(
            turn.get("expected_route_decision", "direct"),
            source=f"{source}:turn[{turn_idx}]",
        ),
        expected_tool_call_count=int(turn.get("expected_tool_call_count", 0)),
        expected_tool_sequence=tuple(turn.get("expected_tool_sequence") or ()),
        expected_refusal=_parse_refusal(
            turn.get("expected_refusal"), source=f"{source}:turn[{turn_idx}]"
        ),
    )


def _parse_scoring_weights(value: Any, *, source: str) -> ScoringWeights:
    if value is None:
        return ScoringWeights()
    if not isinstance(value, dict):
        raise CaseLoadError(
            f"{source}: 'scoring_weights' must be a dict, got "
            f"{type(value).__name__}"
        )
    return ScoringWeights(
        programmatic_pass=float(value.get("programmatic_pass", 0.5)),
        cheap_llm_pass=float(value.get("cheap_llm_pass", 0.5)),
        threshold_pass=float(value.get("threshold_pass", 0.7)),
    ).clamped()


def _parse_case(raw: Any, *, source: str) -> Case:
    if not isinstance(raw, dict):
        raise CaseLoadError(
            f"{source}: case must be a mapping, got {type(raw).__name__}"
        )
    case_id = raw.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise CaseLoadError(f"{source}: 'id' must be a non-empty string")
    language = _validate_locale(
        raw.get("language", "en"), field="language", source=source
    )
    difficulty = _validate_difficulty(
        raw.get("difficulty", "easy"), field="difficulty", source=source
    )
    thread_id = raw.get("thread_id") or case_id
    if not isinstance(thread_id, str):
        raise CaseLoadError(f"{source}: 'thread_id' must be a string")
    root_cause_group = _validate_root_cause_group(raw.get("root_cause_group"), source=source)
    fixture_docs = _str_list(
        raw.get("fixture_docs"), field="fixture_docs", source=source
    )
    tags = _str_list(raw.get("tags"), field="tags", source=source)
    turns_raw = raw.get("turns")
    if not isinstance(turns_raw, list) or not turns_raw:
        raise CaseLoadError(
            f"{source}: 'turns' must be a non-empty list of turn dicts"
        )
    turns = tuple(
        _parse_turn(t, turn_idx=i, source=f"{source}[{case_id}]") for i, t in enumerate(turns_raw)
    )
    # Cross-validation: golden cases must have null root_cause_group;
    # adversarial cases must have one. This catches a common authoring
    # mistake where someone labels the difficulty "hard" but forgets to
    # tag the root cause group, or vice versa.
    if any(t.expected_refusal is not None for t in turns) and root_cause_group is None:
        warnings.warn(
            f"{source}: case '{case_id}' has expected_refusal on a turn but "
            f"root_cause_group is null — adversarial case should declare "
            f"root_cause_group A-F",
            stacklevel=2,
        )
    return Case(
        id=case_id,
        language=language,
        difficulty=difficulty,
        thread_id=thread_id,
        turns=turns,
        fixture_docs=fixture_docs,
        tags=tags,
        root_cause_group=root_cause_group,
        scoring_weights=_parse_scoring_weights(raw.get("scoring_weights"), source=source),
    )


def load_case_file(path: Path, *, suite: str) -> list[Case]:
    """Load all cases in a YAML file. Empty file or empty list → []."""
    if not path.exists():
        raise CaseLoadError(f"case file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    try:
        raw = _yaml_load_safe(text, source=str(path))
    except yaml.YAMLError as exc:
        # safe_load rejected the payload (e.g. python/object/apply tag,
        # python/name tag, or malformed syntax). Surface as CaseLoadError
        # so the harness's top-level error handler can deal with one type.
        raise CaseLoadError(f"{path}: YAML rejected by safe_load: {exc}") from exc
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise CaseLoadError(
            f"{path}: top-level must be a list of cases (use --- separators); "
            f"got {type(raw).__name__}"
        )
    cases: list[Case] = []
    for i, item in enumerate(raw):
        case = _parse_case(item, source=f"{path}:case[{i}]")
        # Stamp the suite tag (immutable dataclass trick — assign via
        # object.__setattr__ because the field is part of an immutable
        # default). We bypass immutability for the internal-only `suite`
        # tag because every case knows which suite it came from at load
        # time, and reconstructing dataclasses for a read-only field is
        # overkill.
        object.__setattr__(case, "suite", suite)
        cases.append(case)
    return cases


def load_suite(suite: str, *, fixtures_root: Optional[Path] = None) -> list[Case]:
    """Load all cases in ``tests/fixtures/eval_v1/cases/<suite>.yaml``.

    ``suite`` is one of "golden" or "adversarial". Returns an empty list
    if the file does not exist (suite absent in this build).
    """
    if suite not in ("golden", "adversarial"):
        raise CaseLoadError(f"unknown suite {suite!r}; must be golden or adversarial")
    root = fixtures_root or Path(__file__).resolve().parents[1] / "fixtures" / "eval_v1"
    case_file = root / "cases" / f"{suite}.yaml"
    return load_case_file(case_file, suite=suite)


# ================================================================
# Helpers
# ============================================================


def _clamp01(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return float(value)


def fixture_path(name: str, *, fixtures_root: Optional[Path] = None) -> Path:
    """Resolve a fixture_docs entry to an absolute file path."""
    root = fixtures_root or Path(__file__).resolve().parents[1] / "fixtures" / "eval_v1"
    path = (root / "corpora" / name).resolve()
    # Defensive: prevent path traversal via `..` in fixture names.
    try:
        path.relative_to((root / "corpora").resolve())
    except ValueError:
        raise CaseLoadError(f"fixture path {name!r} escapes corpora/ root")
    return path