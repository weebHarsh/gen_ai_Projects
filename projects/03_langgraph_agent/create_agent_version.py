"""The same agent in ~15 lines with LangChain 1.x `create_agent` + middleware.

    python projects/03_langgraph_agent/create_agent_version.py "Raise a P2 ticket: VPN down for 6 people"

create_agent builds the same agent->tools loop on the LangGraph runtime. Human approval comes
from HumanInTheLoopMiddleware instead of interrupt() inside the tool, so the ticket tool here
is a plain function. Use create_agent for standard loops; drop to StateGraph (graph.py) when
you need custom nodes, branches, parallel steps or several agents.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain.agents import create_agent  # noqa: E402
from langchain.agents.middleware import HumanInTheLoopMiddleware, ModelCallLimitMiddleware  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from common.llm import get_llm  # noqa: E402
from common.observability import flush, trace_config  # noqa: E402
from graph import SYSTEM  # noqa: E402
from tools import leave_balance, search_policies  # noqa: E402


@tool
def create_it_ticket(summary: str, priority: str = "P3") -> str:
    """Create an IT ServiceDesk ticket."""
    return f"Created ticket INC-20001 ({priority}): {summary}"


def build():
    return create_agent(
        get_llm(),
        tools=[search_policies, leave_balance, create_it_ticket],
        system_prompt=SYSTEM.content,
        middleware=[
            HumanInTheLoopMiddleware(interrupt_on={"create_it_ticket": True}),
            ModelCallLimitMiddleware(run_limit=8),  # guardrail against runaway loops
        ],
        checkpointer=InMemorySaver(),
    )


def main(question: str, approve: bool = True) -> str:
    agent = build()
    config = {**trace_config("03-create-agent"), "configurable": {"thread_id": "demo"}}
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config)
    while result.get("__interrupt__"):
        actions = result["__interrupt__"][0].value["action_requests"]
        print("[approval needed]", [(a["name"], a["args"]) for a in actions])
        decision = {"type": "approve"} if approve else {"type": "reject", "message": "Not approved"}
        result = agent.invoke(Command(resume={"decisions": [decision] * len(actions)}), config)
    flush()
    return result["messages"][-1].content


if __name__ == "__main__":
    print(main(" ".join(sys.argv[1:]) or "What is the hotel cap per night in metro cities?"))
