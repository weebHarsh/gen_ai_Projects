"""LangGraph supervisor pattern: a supervisor LLM routes work between specialist nodes.

    python projects/04_multi_agent/langgraph_supervisor.py

                ┌──────────── supervisor ────────────┐
                ▼                │                    ▼
          researcher ────────────┘               writer ──► supervisor ──► FINISH

Runs in the main venv (no extra installs). Compared with CrewAI/AutoGen you write more code,
but routing, state, limits and persistence are explicit and testable.
"""
from __future__ import annotations

import operator
import sys
from pathlib import Path
from typing import Annotated, Literal, TypedDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "projects" / "02_rag_basics"))

from langgraph.graph import END, START, StateGraph  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from common.llm import get_llm, structured  # noqa: E402
from common.observability import flush, trace_config  # noqa: E402
from rag_pipeline import format_context, get_retriever  # noqa: E402

MAX_STEPS = 6


class TeamState(TypedDict):
    task: str
    research: str
    draft: str
    log: Annotated[list[str], operator.add]  # reducer: each node appends instead of overwriting
    steps: int
    next: str


class Route(BaseModel):
    next: Literal["researcher", "writer", "FINISH"]
    reason: str


def supervisor(state: TeamState) -> dict:
    if state["steps"] >= MAX_STEPS:  # hard guardrail regardless of what the LLM says
        return {"log": ["supervisor: step limit reached"], "next": "FINISH"}
    decision = structured(get_llm(), Route).invoke(
        f"Task: {state['task']}\nHave research: {bool(state['research'])}\n"
        f"Have draft: {bool(state['draft'])}\n"
        "Pick the next worker: researcher (gathers policy facts), writer (drafts from research), "
        "or FINISH when a draft based on research exists."
    )
    return {"log": [f"supervisor -> {decision.next}: {decision.reason}"],
            "steps": state["steps"] + 1, "next": decision.next}


def route(state: TeamState) -> str:
    return END if state["next"] == "FINISH" else state["next"]


def researcher(state: TeamState) -> dict:
    docs = get_retriever().invoke("remote work hybrid office days new joiner onboarding")
    return {"research": format_context(docs), "log": [f"researcher: {len(docs)} chunks"]}


def writer(state: TeamState) -> dict:
    draft = get_llm().invoke(
        "Write a 5-question FAQ (max 150 words) for Acme new joiners about remote/hybrid work. "
        f"Use only these facts and cite [file#chunk]:\n{state['research']}"
    ).content
    return {"draft": draft, "log": ["writer: draft ready"]}


def build():
    g = StateGraph(TeamState)
    g.add_node("supervisor", supervisor)
    g.add_node("researcher", researcher)
    g.add_node("writer", writer)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route, ["researcher", "writer", END])
    g.add_edge("researcher", "supervisor")
    g.add_edge("writer", "supervisor")
    return g.compile()


def main() -> dict:
    result = build().invoke(
        {"task": "Remote-work FAQ for new joiners", "research": "", "draft": "", "log": [], "steps": 0, "next": ""},
        config=trace_config("04-supervisor"),
    )
    print("\n".join(result["log"]), "\n\n", result["draft"] or "(no draft)")
    flush()
    return result


if __name__ == "__main__":
    main()
