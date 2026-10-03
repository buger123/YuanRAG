"""Shared prompt hint for history framing.

Item 7 Step 2 (v2.0.22 FSM cleanup) — centralizes the ``_HISTORY_FRAME_HINT``
constant. Pre-Step-2 this 4-line Chinese string was copy-pasted into
both ``react_agent.py`` (planning step) and ``react_generate.py``
(synthesis step). Drift was guaranteed the moment someone tweaked the
wording on one side.

The hint itself explains the LangChain message taxonomy to the LLM so
that:

* ``[HumanMessage]`` is read as the user's real prior question;
* ``[AIMessage]`` is read as the assistant's prior reply;
* tool-call / tool-result frames are NOT treated as user input.

Both prompt steps need this framing because the LLM is given the
filtered message history and might otherwise conflate tool plumbing
with the user's actual ask.
"""
from __future__ import annotations

# v2.0.28.12 — clarify the AIMessage taxonomy. The pre-v2.0.28.12
# hint lumped every AIMessage together as "你之前的回答" (your prior
# reply), which is only true for AIMessages whose content is plain
# text and that have no ``tool_calls``. For AIMessages that carry
# ``tool_calls`` (the ReAct "I'll call get_current_time" planning
# step), the hint mislabelled them as prior replies — the synthesis
# LLM would then conflate them with real prior answers and either
# (a) refuse to write a fresh answer ("上一轮的回复简要总结: ...")
# or (b) hallucinate a denial ("I don't have a time tool"). With the
# get_current_time tool result preserved by ``_STRING_RETURNING_TOOLS``
# (whitelist get_current_time etc.), the synthesis prompt also
# contains a ToolMessage the synthesis LLM must use to answer the
# user's question; the updated hint makes that flow explicit.
#
# v2.0.28.12 (post-Layer-5 refine) — keep the AIMessage taxonomy
# clarification, but soften the synthesis-LLM directive. The first
# cut of this hint added a "不要重新生成问候语、不要拒绝回答、不要说
# '我没有这个工具' 或 '请重新发送'" prohibition list, which made
# the synthesis LLM over-think and emit only thinking blocks (no
# visible text) — the user then saw "(no answer text generated)"
# in the live UI. The post-Layer-5 refine drops the prohibition
# list and keeps the positive directive ("use the TM data directly
# to answer the latest HM") instead. GENERATE_SYSTEM + the positive
# TM directive are enough to anchor the LLM on "answer the user's
# question with the tool result"; listing every wrong behavior to
# avoid is anti-helpful on a doc-grounded prompt.
#
# Why a single hint covers both react_agent AND react_generate:
# Both phases benefit from the same role mapping — the planning
# step (react_agent) needs to distinguish its own tool-call
# AIMessages from the user's HMs (to avoid "the user asked me to
# search" confusion), and the synthesis step (react_generate)
# needs to distinguish its own draft / planning-step AIMessages
# from prior turns' final replies (to avoid "I'll summarize my
# previous reply" confusion). Splitting into two hints would
# duplicate the wording and re-introduce v2.0.22's drift risk.
_HISTORY_FRAME_HINT: str = (
    "对话历史说明:[HumanMessage] 是用户的真实提问(无论措辞、无论是否简短)。"
    "[AIMessage] 分两种:有可见文字内容且「没有 tool_calls」的,"
    "是你之前的最终回答;带有「tool_calls」的(无论是否同时有触发文字),"
    "是你为获取数据而发出的「工具调用计划」步骤,后面会紧跟对应的"
    "[ToolMessage](数据),不是你的最终回答。"
    "[ToolMessage](如 get_current_time 的返回结果)是「该轮的工具调用数据」,"
    "直接引用其中的内容回答用户「最近的 HumanMessage」——你的最终答案"
    "必须作为可见的正文文本(text 块)输出,不能只在 thinking 块里写。"
)


__all__ = ["_HISTORY_FRAME_HINT"]
