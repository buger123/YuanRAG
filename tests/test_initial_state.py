"""Regression tests for ``src.agent.runner._initial_state``.

Pinned here (instead of in the deleted ``test_summary_intent_skip_grade.py``)
because the test target — ``runner._initial_state`` — is a live function
called by ``stream_agent`` on every turn. The original file happened to
co-locate this test with the dead ``grade_documents`` regression suite
(which the v2.0.14 cleanup removed); the test itself is unrelated to
grading and must survive any future refactor of ``_initial_state``.

The contract under test:

* ReAct fields (``step_count``, ``intent``, ``corrected_query``,
  ``needs_current_time``, ``created_at``) seed at their pre-node
  defaults so the graph can read them before ``intent_analysis``
  runs.
* The legacy ``tool_call_log`` field is gone (v2.0.22 Item 7
  Step 12 — it was declared + initialized but never written; tool
  calls persist on the AIMessage's ``additional_kwargs["tool_calls"]``
  instead, built by ``react_generate._build_tool_call_log``).
* The legacy ``summary_intent_used`` / ``web_search_attempted`` flags
  do NOT survive across turns (ReAct subsumes them — the LLM drives
  routing now). Without explicit reset of the live state fields, a
  previous turn's counter could leak into the next one and confuse the
  frontend's "Step N" pill.
"""
from __future__ import annotations


def test_initial_state_resets_flags_between_turns():
    from src.agent.runner import _initial_state

    state = _initial_state(thread_id="t1", message="hello")
    assert state["step_count"] == 0
    assert "tool_call_log" not in state, (
        "tool_call_log was removed in v2.0.22 Step 12 (P1-B5); tool "
        "call records live on AIMessage.additional_kwargs['tool_calls'] "
        "instead. The initial state must not carry the dead field."
    )
    assert state["intent"] is None
    assert state["corrected_query"] is None
    assert state["needs_current_time"] is False
    assert state["created_at"] is None
