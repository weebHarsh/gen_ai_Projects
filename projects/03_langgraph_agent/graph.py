"""A tool-calling agent built explicitly as a LangGraph StateGraph.

    START -> agent --(tool calls?)--> tools -> agent -> ... -> END

Building it by hand (rather than create_agent) shows the three ideas interviews probe:
  1. State: a typed dict that flows through nodes (here: the message list).
  2. Edges: conditional routing on the model's output (tool call or final answer).
  3. Checkpointer: persists state per thread_id -> memory, resumability, human-in-the-loop.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.messages import AIMessage, SystemMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode, tools_condition  # noqa: E402
from langgraph.types import Command  # noqa: E402

from common.llm import get_llm  # noqa: E402
from common.observability import trace_config  # noqa: E402
from tools import TOOLS  # noqa: E402

SYSTEM = SystemMessage(
    "You are Acme Corp's internal HR/IT helpdesk agent. Use search_policies for policy facts "
    "and cite sources like [leave_policy.md#3]. Use leave_balance for any leave arithmetic. "
    "Only create IT tickets when the user asks or the issue can't be self-served. Be concise."
)


def _bind_tools(llm):
    """bind_tools, also through a fallback chain (get_llm_chain returns RunnableWithFallbacks)."""
    if hasattr(llm, "runnable") and hasattr(llm, "fallbacks"):
        return _bind_tools(llm.runnable).with_fallbacks([_bind_tools(f) for f in llm.fallbacks])
    return llm.bind_tools(TOOLS)


DEADLINE_MESSAGE = "I ran out of time for this request, so I stopped before asking the model again. Please try a shorter request."


def trim_history(messages: list, keep: int = 8) -> list:
    """The last ~`keep` messages, widened backwards to start at a human message. A tool result is therefore never
    sent without the tool call that produced it (providers reject an orphan tool message)."""
    if keep <= 0 or len(messages) <= keep:
        return list(messages)
    start = len(messages) - keep
    while start > 0 and getattr(messages[start], "type", "") != "human":
        start -= 1
    return list(messages[start:])


def build_graph(checkpointer=None, llm=None, max_history: int | None = None, deadline: float | None = None):
    """`llm` lets a caller choose the model per request (the hosted demo routes by difficulty);
    default behaviour is unchanged: get_llm(). `max_history` sends only the last N messages to the model
    (state keeps everything); `deadline` is a time.monotonic() value after which the model is skipped."""
    llm = _bind_tools(llm or get_llm())

    def agent(state: MessagesState) -> dict:
        if deadline is not None and time.monotonic() >= deadline:
            return {"messages": [AIMessage(DEADLINE_MESSAGE)]}
        msgs = trim_history(state["messages"], max_history) if max_history else state["messages"]
        return {"messages": [llm.invoke([SYSTEM, *msgs])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def run_turn(graph, text: str, thread_id: str, auto_approve: bool | None = None) -> str:
    config = {**trace_config("03-agent", session_id=thread_id),
              "configurable": {"thread_id": thread_id}}
    payload: object = {"messages": [{"role": "user", "content": text}]}
    while True:
        result = graph.invoke(payload, config)
        interrupts = result.get("__interrupt__")
        if not interrupts:
            return result["messages"][-1].content
        request = interrupts[0].value
        print(f"\n[approval needed] {request['action']} {request['args']}")
        if auto_approve is None:
            approved = input(request["question"] + " ").strip().lower().startswith("y")
        else:
            approved = auto_approve
        payload = Command(resume={"approved": approved})  # resume the paused graph


if __name__ == "__main__":
    # Prints a Mermaid diagram of the graph - paste into https://mermaid.live
    print(build_graph().get_graph().draw_mermaid())
