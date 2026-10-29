"""Tools for the IT/HR helpdesk agent.

- search_policies     -> RAG over the Acme KB (read-only, safe)
- leave_balance       -> deterministic calculation (LLMs are bad at arithmetic; give them tools)
- create_it_ticket    -> calls the ServiceDesk REST API. It's a write action, so it pauses for
                         human approval with LangGraph's interrupt() first.
"""
from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path
from typing import Literal

import httpx
from langchain_core.tools import tool
from langgraph.types import interrupt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "projects" / "02_rag_basics"))

from rag_pipeline import format_context, get_retriever  # noqa: E402

_ticket_ids = itertools.count(10001)


@tool
def search_policies(query: str) -> str:
    """Search Acme Corp HR, IT, expense, security and AI policies. Use for any policy question.
    Returns policy excerpts with [file#chunk] citation ids."""
    return format_context(get_retriever().invoke(query))


@tool
def leave_balance(months_worked_this_year: int, days_taken: int, carried_forward: int = 0) -> str:
    """Calculate remaining annual leave. Leave accrues at 2 days per completed month."""
    accrued = min(months_worked_this_year, 12) * 2
    remaining = accrued + carried_forward - days_taken
    return f"Accrued {accrued} + carried {carried_forward} - taken {days_taken} = {remaining} days remaining"


@tool
def create_it_ticket(summary: str, priority: Literal["P1", "P2", "P3", "P4"] = "P3") -> str:
    """Create an IT ServiceDesk ticket. Only use when the user asks for a ticket or the
    issue can't be solved from the IT support guide."""
    decision = interrupt({
        "action": "create_it_ticket",
        "args": {"summary": summary, "priority": priority},
        "question": "Approve creating this ticket? (y/n)",
    })
    if not decision.get("approved"):
        return "Ticket NOT created: the human reviewer rejected it."

    api = os.getenv("TICKET_API_URL")  # e.g. the mock endpoint served by project 08
    if api:
        key = os.getenv("TICKET_API_KEY") or os.getenv("APP_API_KEY", "")
        resp = httpx.post(api, json={"summary": summary, "priority": priority},
                          headers={"X-API-Key": key}, timeout=10)
        resp.raise_for_status()
        return f"Created ticket {resp.json()['id']} ({priority})"
    return f"Created ticket INC-{next(_ticket_ids)} ({priority}) [mock - set TICKET_API_URL for a real API]"


TOOLS = [search_policies, leave_balance, create_it_ticket]
