"""08 - Production-style REST API around the RAG pipeline and the LangGraph agent.

    python projects/08_deploy_digitalocean/app.py      # uvicorn on :8080, docs at /docs

Endpoints
    GET  /health                       liveness for DO App Platform
    POST /ask        {question}        RAG answer + sources
    POST /chat       {message, thread_id}      agent turn; may return status=pending_approval
    POST /approve    {thread_id, approved}     resume an agent paused for human approval
    POST /servicedesk/tickets          mock enterprise ticket API the agent calls (TICKET_API_URL)

Auth: send X-API-Key matching APP_API_KEY (skipped when APP_API_KEY is unset, for local dev).
RPA bots (UiPath "HTTP Request", Power Automate "HTTP", Automation Anywhere "REST Web Service")
call these endpoints like any other REST API - see README.
"""
from __future__ import annotations

import itertools
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "projects" / "02_rag_basics"))
sys.path.insert(0, str(ROOT / "projects" / "03_langgraph_agent"))

from fastapi import Depends, FastAPI, Header, HTTPException  # noqa: E402
from langgraph.types import Command  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from common.observability import flush, trace_config  # noqa: E402

app = FastAPI(title="Acme GenAI Helpdesk API", version="1.0")
_graph = None
_ticket_ids = itertools.count(50001)


def graph():
    global _graph
    if _graph is None:  # build lazily so /health is instant on cold start
        from graph import build_graph

        _graph = build_graph()  # InMemorySaver: use PostgresSaver when running >1 instance
    return _graph


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("APP_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid API key")


class AskIn(BaseModel):
    question: str


class ChatIn(BaseModel):
    message: str
    thread_id: str | None = None


class ApproveIn(BaseModel):
    thread_id: str
    approved: bool


class TicketIn(BaseModel):
    summary: str
    priority: str = "P3"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", dependencies=[Depends(require_key)])
def ask(body: AskIn) -> dict:
    from rag_pipeline import answer

    result = answer(body.question, config=trace_config("api-ask", tags=["api"]))
    flush()
    return {"answer": result["answer"], "sources": result["sources"]}


def _agent_step(payload, thread_id: str) -> dict:
    config = {**trace_config("api-chat", session_id=thread_id, tags=["api"]),
              "configurable": {"thread_id": thread_id}}
    result = graph().invoke(payload, config)
    flush()
    if result.get("__interrupt__"):
        return {"status": "pending_approval", "thread_id": thread_id,
                "approval_request": result["__interrupt__"][0].value}
    return {"status": "done", "thread_id": thread_id, "reply": result["messages"][-1].content}


@app.post("/chat", dependencies=[Depends(require_key)])
def chat(body: ChatIn) -> dict:
    thread_id = body.thread_id or f"api-{uuid.uuid4().hex[:8]}"
    return _agent_step({"messages": [{"role": "user", "content": body.message}]}, thread_id)


@app.post("/approve", dependencies=[Depends(require_key)])
def approve(body: ApproveIn) -> dict:
    return _agent_step(Command(resume={"approved": body.approved}), body.thread_id)


@app.post("/servicedesk/tickets", dependencies=[Depends(require_key)])
def create_ticket(body: TicketIn) -> dict:
    """Stand-in for ServiceNow/Jira. In real life the agent's tool calls *their* API."""
    return {"id": f"INC-{next(_ticket_ids)}", "summary": body.summary, "priority": body.priority}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
