"""Offline smoke tests: every project runs end-to-end with LLM_PROVIDER=fake (no keys, no network).

    LLM_PROVIDER=fake pytest -q tests/test_smoke.py

They check wiring (imports, graphs, tools, API routes), not answer quality. Quality is
project 06's job.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

os.environ["LLM_PROVIDER"] = "fake"
os.environ["EMBEDDINGS_PROVIDER"] = "fake"
for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "TICKET_API_URL", "APP_API_KEY"):
    os.environ.pop(key, None)  # keep tests hermetic even if .env sets them

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "projects"


def load(path: Path, name: str):
    """Import a project file by path (project folders start with digits, so not packages)."""
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # pydantic/TypedDict resolve postponed annotations via sys.modules
    spec.loader.exec_module(module)
    return module


def test_01_prompt_engineering(capsys):
    mod = load(P / "01_prompt_engineering" / "main.py", "p01")
    mod.main(["all"])
    out = capsys.readouterr().out
    assert "=== structured ===" in out and "priority=" in out


def test_02_rag_chunks_and_answer():
    rag = load(P / "02_rag_basics" / "rag_pipeline.py", "rag_pipeline")
    chunks = rag.split_documents(rag.load_documents())
    assert len(chunks) > 10
    assert all("\n" in c.page_content and len(c.page_content) > 40 for c in chunks)  # contextual header
    for store in ("chroma", "faiss"):
        result = rag.answer("How many sick days?", store=store)
        assert result["answer"] and result["contexts"] and result["sources"]


def test_03_agent_tool_loop_and_memory():
    graph_mod = load(P / "03_langgraph_agent" / "graph.py", "graph")
    graph = graph_mod.build_graph()
    reply = graph_mod.run_turn(graph, "How many sick days do I get?", "t1", auto_approve=True)
    assert reply
    state = graph.get_state({"configurable": {"thread_id": "t1"}}).values
    assert any(getattr(m, "tool_calls", None) for m in state["messages"])  # a tool was called


def test_03_ticket_tool_interrupts_for_approval():
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode
    from langgraph.types import Command

    tools = load(P / "03_langgraph_agent" / "tools.py", "tools")
    g = StateGraph(MessagesState)
    g.add_node("tools", ToolNode([tools.create_it_ticket]))
    g.add_edge(START, "tools")
    g.add_edge("tools", END)
    app = g.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "ticket"}}
    call = AIMessage(content="", tool_calls=[{"name": "create_it_ticket", "id": "c1",
                                              "args": {"summary": "VPN down", "priority": "P2"}}])
    paused = app.invoke({"messages": [call]}, cfg)
    assert paused["__interrupt__"][0].value["action"] == "create_it_ticket"
    rejected = app.invoke(Command(resume={"approved": False}), cfg)
    assert "NOT created" in rejected["messages"][-1].content


def test_03_create_agent_version():
    mod = load(P / "03_langgraph_agent" / "create_agent_version.py", "p03b")
    assert mod.main("What is the hotel cap?")


def test_04_supervisor_terminates():
    mod = load(P / "04_multi_agent" / "langgraph_supervisor.py", "p04")
    result = mod.main()
    assert result["steps"] <= mod.MAX_STEPS
    assert any("researcher" in line for line in result["log"])


def test_06_evals_offline():
    rag_eval = load(P / "06_agent_evals" / "eval_rag.py", "eval_rag")
    assert rag_eval.main(["--no-judge"]) == 0
    agent_eval = load(P / "06_agent_evals" / "eval_agent.py", "eval_agent")
    assert agent_eval.main(["--no-judge"]) == 0


def test_07_managed_agent_config(capsys):
    mod = load(P / "07_claude_managed_agents" / "setup_agent.py", "p07")
    mod.main(["--print"])
    out = capsys.readouterr().out
    assert "agent_toolset_20260401" in out and "search_acme_policies" in out


def test_08_api_routes():
    from fastapi.testclient import TestClient

    mod = load(P / "08_deploy_digitalocean" / "app.py", "p08")
    client = TestClient(mod.app)
    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/ask", json={"question": "hotel cap?"}).json()["sources"]
    assert client.post("/chat", json={"message": "hi"}).json()["status"] in {"done", "pending_approval"}
    assert client.post("/servicedesk/tickets", json={"summary": "x"}).json()["id"].startswith("INC-")


@pytest.mark.parametrize("script", sorted(str(p.relative_to(ROOT)) for p in P.rglob("*.py")))
def test_every_file_compiles(script):
    compile((ROOT / script).read_text(), script, "exec")
