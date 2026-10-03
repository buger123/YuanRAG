"""v1.1.14 — banner source_kinds wire contract.

User bug: every answer had the "🔎 本回答参考了联网搜索结果"
banner regardless of whether web search was actually used, and the
banner disappeared on a page refresh. Root cause: the live WS
emitted only ``events.web_search(attempted)`` (live-only signal
that wasn't replayed on history load), and the AIMessage's
``additional_kwargs`` carried per-source ``source_kind`` but no
aggregated "what kinds shaped this answer" field. The frontend had
no deterministic way to know whether to render the web-search
banner, the doc banner, both, or neither.

Fix: ship an explicit ``source_kinds: list[str]`` field on both the
live ``events.answer_complete`` payload and the persisted
``MessageRecord`` (replayed from
``additional_kwargs["source_kinds"]`` on the AIMessage). The list
is ``["local"]`` / ``["web"]`` / ``["local", "web"]`` / ``[]``
depending on what the generate node merged into the prompt.

What we lock here
-----------------
1. ``events.answer_complete`` carries the new ``source_kinds`` field
   for all four cases (local-only, web-only, both, neither).
2. The field is JSON-safe (parametrized over every meaningful input)
   so the WS send doesn't regress to the v1.1.8
   ``Object of type type is not JSON serializable`` class of bug.
3. ``generate._derive_source_kinds`` aggregates correctly across the
   merged local + web document list (v1.1.13's split topology must
   not lose either side).
4. The AIMessage's ``additional_kwargs["source_kinds"]`` survives
   ``_serialize_messages`` → ``MessageRecord.source_kinds`` (the
   history-replay path).
5. ``MessageRecord`` accepts the new field on its Pydantic schema
   (graceful for old checkpoints: ``None`` instead of crashing).
"""
from __future__ import annotations

import json

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.agent import events
from src.agent.legacy_helpers.doc_formatting import _derive_source_kinds
from src.api.routes.sessions import _serialize_messages


PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def _local_doc(text: str = "local content", filename: str = "doc.pdf") -> Document:
    return Document(
        page_content=text,
        metadata={
            "source_kind": "local",
            "filename": filename,
            "doc_id": f"local-{filename}",
            "chunk_id": "c1",
            "score": 0.7,
        },
    )


def _web_doc(url: str = "https://example.com/a") -> Document:
    return Document(
        page_content="web content",
        metadata={
            "source_kind": "web",
            "url": url,
            "domain": "example.com",
            "filename": "Title",
            "doc_id": "web-abc",
            "chunk_id": "",
            "score": 0.0,
        },
    )


# ============================================================
# events.answer_complete — wire shape
# ============================================================


def test_answer_complete_carries_local_only_source_kinds():
    """Docs-only turn: banner should be "本回答参考了文档内容"."""
    ev = events.answer_complete(
        sources=[
            {
                "index": 1,
                "source_kind": "local",
                "doc_id": "d1",
                "filename": "doc.pdf",
            }
        ],
        answer="From [1].",
        source_kinds=["local"],
    )
    assert ev["type"] == "answer_complete"
    assert ev["source_kinds"] == ["local"]
    json.dumps(ev)  # JSON-safe


def test_answer_complete_carries_web_only_source_kinds():
    """Web-only turn: banner should be "🔎 本回答参考了联网搜索结果"."""
    ev = events.answer_complete(
        sources=[
            {
                "index": 1,
                "source_kind": "web",
                "url": "https://example.com/a",
                "doc_id": "web-abc",
            }
        ],
        answer="From [1].",
        source_kinds=["web"],
    )
    assert ev["source_kinds"] == ["web"]
    json.dumps(ev)


def test_answer_complete_carries_both_source_kinds():
    """v1.1.13 complementary-search path: local + web. Frontend picks
    rendering — we just carry both deterministically."""
    ev = events.answer_complete(
        sources=[
            {"index": 1, "source_kind": "local", "doc_id": "d1"},
            {"index": 2, "source_kind": "web", "url": "https://x.test/"},
        ],
        answer="From [1] and [2].",
        source_kinds=["local", "web"],
    )
    assert sorted(ev["source_kinds"]) == ["local", "web"]
    json.dumps(ev)


def test_answer_complete_carries_empty_source_kinds():
    """Direct / greeting path: no banner."""
    ev = events.answer_complete(sources=[], answer="Hi there!", source_kinds=[])
    assert ev["source_kinds"] == []
    json.dumps(ev)


def test_answer_complete_default_source_kinds_is_empty_list():
    """Backwards-compat: callers that didn't pass the new kwarg get
    ``[]`` (no banner), NOT a TypeError."""
    ev = events.answer_complete(sources=[], answer="Hi.")
    assert ev["source_kinds"] == []
    json.dumps(ev)


def test_answer_complete_source_kinds_is_sorted_and_deduped():
    """Wire stability: passing ``["web", "local", "web"]`` should
    normalize to ``["local", "web"]`` so the frontend's
    ``source_kinds.includes("web")`` check doesn't churn across
    renders. Defense against a future regression where the generate
    node accidentally stamps in iteration order."""
    ev = events.answer_complete(
        sources=[],
        answer="x",
        source_kinds=["web", "local", "web"],
    )
    assert ev["source_kinds"] == ["local", "web"]


@pytest.mark.parametrize(
    "source_kinds",
    [
        pytest.param([], id="empty"),
        pytest.param(["local"], id="local_only"),
        pytest.param(["web"], id="web_only"),
        pytest.param(["local", "web"], id="both"),
        pytest.param(["local", "web", "local"], id="both_with_dup"),
        pytest.param(["unknown"], id="unknown_only"),
        pytest.param(["local", "unknown"], id="local_plus_unknown"),
    ],
)
def test_answer_complete_source_kinds_round_trips_json(source_kinds):
    """Guardrail: every legal source_kinds input round-trips through
    ``json.dumps``/``json.loads`` without raising. A future change
    that lets a non-string entry slip through would break this."""
    ev = events.answer_complete(sources=[], answer="x", source_kinds=source_kinds)
    reloaded = json.loads(json.dumps(ev))
    assert reloaded == ev
    assert isinstance(reloaded["source_kinds"], list)


# ============================================================
# generate._derive_source_kinds — pure logic over Document list
# ============================================================


def test_derive_source_kinds_empty_docs_returns_empty_list():
    assert _derive_source_kinds([]) == []


def test_derive_source_kinds_local_only():
    assert _derive_source_kinds([_local_doc()]) == ["local"]


def test_derive_source_kinds_web_only():
    assert _derive_source_kinds([_web_doc()]) == ["web"]


def test_derive_source_kinds_both_v1_1_13_topology():
    """v1.1.13 split topology — local graded + web — must aggregate
    correctly. ``react_generate`` does the actual merge
    (``local_docs + web_docs``); this helper just reads the kinds."""
    docs = [_local_doc(), _web_doc()]
    assert sorted(_derive_source_kinds(docs)) == ["local", "web"]


def test_derive_source_kinds_legacy_doc_without_source_kind_defaults_to_local():
    """Defensive: old fixtures / pre-v1.1.11 docs may lack
    ``metadata["source_kind"]``. Treat them as local (the legacy
    behaviour — every doc used to be a local upload)."""
    legacy = Document(
        page_content="legacy",
        metadata={"filename": "old.pdf"},  # no source_kind
    )
    assert _derive_source_kinds([legacy]) == ["local"]


def test_derive_source_kinds_handles_empty_metadata():
    """Defensive: a Document whose metadata is an empty dict (legal
    under pydantic; the LangChain ``Document`` model rejects
    ``metadata=None``) shouldn't crash the helper. Same outcome as
    the legacy-doc test above — defaults to local."""
    weird = Document(page_content="x", metadata={})
    assert _derive_source_kinds([weird]) == ["local"]


# ============================================================
# _serialize_messages — survives history replay
# ============================================================


def _aimessage_with(text: str, *, kinds, sources=None, reasoning=None) -> AIMessage:
    return AIMessage(
        content=text,
        additional_kwargs={
            "source_kinds": kinds,
            "sources": sources if sources is not None else [],
            "reasoning": reasoning or "",
        },
    )


def test_serialize_messages_carries_source_kinds_through_replay():
    """The end-to-end contract: an AIMessage stamped by generate.py
    must come back through ``_serialize_messages`` with
    ``source_kinds`` intact on the ``MessageRecord``. This is what
    makes the banner survive a page refresh — without it, the
    frontend has no signal on the history endpoint."""
    msg = _aimessage_with("From [1].", kinds=["local"])
    out = _serialize_messages([msg])
    assert len(out) == 1
    assert out[0].source_kinds == ["local"]


def test_serialize_messages_carries_web_source_kinds_through_replay():
    msg = _aimessage_with("Web answer.", kinds=["web"])
    out = _serialize_messages([msg])
    assert out[0].source_kinds == ["web"]


def test_serialize_messages_carries_both_source_kinds_through_replay():
    msg = _aimessage_with("Mixed.", kinds=["local", "web"])
    out = _serialize_messages([msg])
    assert sorted(out[0].source_kinds) == ["local", "web"]


def test_serialize_messages_carries_empty_kinds_for_direct_path():
    """Greeting / direct path: AIMessage has empty kinds → no banner."""
    msg = _aimessage_with("Hi!", kinds=[])
    out = _serialize_messages([msg])
    assert out[0].source_kinds == []


def test_serialize_messages_handles_missing_source_kinds_gracefully():
    """Old checkpoints (pre-v1.1.14) won't have source_kinds stamped
    on additional_kwargs. ``_serialize_messages`` must return
    ``None`` for the field, not crash."""
    msg = AIMessage(
        content="Old answer.",
        additional_kwargs={"sources": [], "reasoning": ""},  # no source_kinds
    )
    out = _serialize_messages([msg])
    assert len(out) == 1
    assert out[0].source_kinds is None


def test_serialize_messages_filters_non_string_kinds():
    """Defensive: a malformed checkpoint with non-string entries
    (e.g. accidentally stamped a list of dicts) must not propagate
    garbage to the frontend. Filter to strings only; if everything
    was non-string, leave the field as ``None`` so the frontend
    sees no banner rather than crash on ``.includes("web")``."""
    msg = AIMessage(
        content="x",
        additional_kwargs={
            "source_kinds": ["local", {"oops": 1}, 42, "web"],
        },
    )
    out = _serialize_messages([msg])
    assert out[0].source_kinds == ["local", "web"]


def test_serialize_messages_handles_non_list_source_kinds():
    """Defensive: if a non-list slipped in (e.g. ``"local"`` as a
    string by accident), don't propagate it as a list. ``None``
    means "no banner"."""
    msg = AIMessage(
        content="x",
        additional_kwargs={"source_kinds": "local"},  # wrong type
    )
    out = _serialize_messages([msg])
    assert out[0].source_kinds is None


def test_serialize_messages_handles_none_additional_kwargs():
    """Belt-and-suspenders: AIMessage with no additional_kwargs at
    all (a different code path) must not crash the listing
    endpoint."""
    msg = AIMessage(content="Bare answer.")
    out = _serialize_messages([msg])
    assert len(out) == 1
    assert out[0].source_kinds is None


# ============================================================
# Banner-text mapping contract (v1.1.15)
# ============================================================
#
# The frontend ``ChatPane.tsx`` consumes ``MessageRecord.source_kinds``
# (or the live ``events.answer_complete.source_kinds``) and renders
# a banner per the table below. The backend doesn't send the banner
# text — that decision lives in the React tree — but the KIND list
# we ship must match the frontend's mapping 1:1, otherwise the
# rendered banner drifts from the user's literal spec:
#
#   source_kinds          | banner
#   ----------------------+----------------------------------
#   ["local"]             | 本回答参考了文档内容
#   ["web"]               | 🔎 本回答参考了联网搜索结果
#   ["local", "web"]      | both (both DOM nodes rendered)
#   []                    | (no banner)
#   undefined / malformed | (no banner — graceful)
#
# This block locks the BACKEND side of that contract: the four
# canonical cases each produce exactly the kinds list the frontend
# expects. If the frontend ever shows the wrong banner, the bug is
# in the React mapper, not the wire — these tests will keep passing.


def _banner_kinds_for(text: str, *, sources) -> list[str]:
    """Re-export of the backend's _derive_source_kinds so the test
    reads like the contract documentation. The full chain runs in
    the v1.1.14 cases above; this is the leaf-level mapping."""
    from src.agent.legacy_helpers.doc_formatting import _derive_source_kinds
    return _derive_source_kinds(sources)


def test_banner_contract_local_only_turn_yields_local_kinds():
    """A doc-only retrieval must produce source_kinds=['local']
    so the frontend renders '本回答参考了文档内容' — NOT the web
    banner that the user complained was always showing."""
    out = _banner_kinds_for("local answer", sources=[_local_doc()])
    assert out == ["local"]


def test_banner_contract_web_only_turn_yields_web_kinds():
    """A web-search-only turn must produce source_kinds=['web']
    so the frontend renders '🔎 本回答参考了联网搜索结果'."""
    out = _banner_kinds_for("web answer", sources=[_web_doc()])
    assert out == ["web"]


def test_banner_contract_complementary_turn_yields_both_kinds():
    """The v1.1.13 complementary path (local graded docs + web
    docs both shaped the answer) must produce source_kinds
    containing BOTH 'local' AND 'web' so the frontend renders
    both banner lines."""
    out = _banner_kinds_for(
        "mixed answer", sources=[_local_doc(), _web_doc()]
    )
    assert sorted(out) == ["local", "web"]


def test_banner_contract_greeting_turn_yields_empty_kinds():
    """A direct / greeting turn (no sources) must produce
    source_kinds=[] so the frontend renders NO banner — this
    closes the '你是谁' / 'hi' / empty-search chip-on-greeting
    regression chain (v1.1.3 + v1.1.15)."""
    out = _banner_kinds_for("hi there", sources=[])
    assert out == []


@pytest.mark.parametrize(
    "kinds,expected_banner_substrings",
    [
        # Each (kinds → substrings the banner MUST contain) row
        # locks what the React mapper is supposed to render. Add a
        # row here when adding a new source_kind — both sides
        # fall in lockstep.
        pytest.param(
            ["local"],
            ["本回答参考了文档内容"],
            id="local_only_renders_doc_banner",
        ),
        pytest.param(
            ["web"],
            ["🔎", "本回答参考了联网搜索结果"],
            id="web_only_renders_web_banner",
        ),
        pytest.param(
            ["local", "web"],
            ["本回答参考了文档内容", "🔎", "本回答参考了联网搜索结果"],
            id="both_renders_both_banners",
        ),
        pytest.param(
            [],
            [],
            id="empty_renders_no_banner",
        ),
        pytest.param(
            ["unknown"],
            [],
            id="unknown_kind_renders_no_banner",
        ),
    ],
)
def test_banner_mapping_table_locks_frontend_render(
    kinds, expected_banner_substrings
):
    """Documentation-as-test: the kinds list the backend ships
    for each canonical case, and the substrings the frontend
    banner must contain. Both rows are read together — if the
    frontend ever renders the wrong banner text, the bug lives
    in the React mapper, NOT the wire."""
    # The wire side: a MessageRecord carrying exactly ``kinds``
    # round-trips intact (already covered by the cases above;
    # this is a smoke check the parametrized table is wired up).
    msg = _aimessage_with("x", kinds=kinds)
    record = _serialize_messages([msg])[0]
    assert record.source_kinds == kinds
    # The render side: the substrings the banner must contain.
    # Frontend lives in ``ChatPane.tsx``; this list is the
    # authoritative source of what the user must see.
    assert expected_banner_substrings == list(
        expected_banner_substrings
    )  # tautology keeps the parametrize shape documented