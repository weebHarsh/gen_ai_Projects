"""Offline tests for the hosted demo API (showcase/server.py). No keys, no network.

    LLM_PROVIDER=fake RETRIEVER=bm25 pytest -q tests/test_showcase.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ["LLM_PROVIDER"] = "fake"
os.environ["EMBEDDINGS_PROVIDER"] = "fake"
os.environ["RETRIEVER"] = "bm25"
for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "TICKET_API_URL", "APP_API_KEY",
            "MANAGED_AGENT_ID", "MANAGED_ENVIRONMENT_ID"):
    os.environ.pop(key, None)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from showcase import server  # noqa: E402
from showcase.ratelimit import RateLimiter  # noqa: E402

SECRET = "sk-secret-123-abcdefghij"


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    server.limiter.reset()
    server._EVAL_CACHE.update(at=0.0, mode=None, payload=None)
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    for k in ("ANTHROPIC_API_KEY", "MANAGED_AGENT_ID", "MANAGED_ENVIRONMENT_ID"):
        monkeypatch.delenv(k, raising=False)
    yield


@pytest.fixture
def client():
    return TestClient(server.app)


def break_every_provider(monkeypatch):
    from common.llm import FakeChatModel

    def boom(self, *a, **k):
        raise RuntimeError("provider down")

    monkeypatch.setattr(FakeChatModel, "_generate", boom)


def script_ticket_model(monkeypatch):
    """The fake model calls the first bound tool; make it ask for a ticket instead, then answer."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from common.llm import FakeChatModel

    def gen(self, messages, stop=None, run_manager=None, **kw):
        if any(isinstance(m, ToolMessage) for m in messages):
            msg = AIMessage(content=f"Done: {messages[-1].content}")
        else:
            human = next(m for m in reversed(messages) if isinstance(m, HumanMessage))
            if "ticket" in human.content:
                msg = AIMessage(content="", tool_calls=[{"name": "create_it_ticket", "id": "c1",
                                                         "args": {"summary": "VPN outage", "priority": "P2"}}])
            else:
                msg = AIMessage(content="plain reply")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    monkeypatch.setattr(FakeChatModel, "_generate", gen)


# --------------------------------------------------------------------------- basics
def test_health_and_index(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["retriever"] == "bm25" and body["mode"] == "llm"
    assert client.get("/").status_code == 200


def test_unknown_route_uses_error_shape(client):
    r = client.get("/api/nope")
    assert r.status_code == 404 and r.json()["error"] == "not_found"


def test_providers_never_leak_secrets(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("GROQ_API_KEY", SECRET + "-groq")
    seen = []
    r = client.get("/api/providers")
    seen.append(r.text)
    body = r.json()
    assert "anthropic" in body["available"] and "groq" in body["available"]
    assert any(p["name"] == "anthropic" and p["configured"] for p in body["providers"])
    seen.append(client.get("/api/health").text)
    seen.append(client.post("/api/ask", json={"question": "What is the hotel cap?", "provider": "auto"}).text)
    seen.append(client.get("/api/managed/status").text)
    assert all("sk-secret-123" not in t for t in seen)


# --------------------------------------------------------------------------- ask
def test_ask_llm_mode_has_route_trace_and_contexts(client):
    r = client.post("/api/ask", json={"question": "How many sick days do I get?"})
    assert r.status_code == 200
    b = r.json()
    assert b["mode"] == "llm" and "notice" not in b
    assert b["answer"].startswith("FAKE ANSWER")
    assert b["sources"] and all(s.endswith(".md") for s in b["sources"])
    assert b["contexts"][0].keys() >= {"source", "chunk_id", "text"}
    assert set(b["route"]) == {"tier", "provider", "model", "reason", "est_cost_usd", "pinned"}
    assert b["route"]["pinned"] is False and b["route"]["tier"] in {"quick", "default", "complex"}
    kinds = {s["kind"] for s in b["trace"]["spans"]}
    assert {"route", "retriever", "llm"} <= kinds
    assert b["trace"]["trace_id"] and b["trace"]["total_ms"] >= 0 and b["latency_ms"] >= 0
    ids = {s["id"] for s in b["trace"]["spans"]}
    assert all(s["parent_id"] in ids | {None} for s in b["trace"]["spans"])


def test_ask_routing_differs_by_difficulty(client):
    easy = client.post("/api/ask", json={"question": "What is the hotel cap?"}).json()["route"]
    hard = client.post("/api/ask", json={
        "question": "Compare the leave and expense policies and explain any conflicts between them"}).json()["route"]
    assert easy["tier"] == "quick" and hard["tier"] == "complex"
    assert hard["reason"] != easy["reason"]


def test_ask_pinned_provider(client):
    b = client.post("/api/ask", json={"question": "hello there", "provider": "fake"}).json()
    assert b["route"]["pinned"] is True and b["route"]["provider"] == "fake"


def test_ask_degrades_when_every_provider_fails(client, monkeypatch):
    break_every_provider(monkeypatch)
    r = client.post("/api/ask", json={"question": "How many sick days do I get?"})
    assert r.status_code == 200
    b = r.json()
    assert b["mode"] == "retrieval-only" and b["notice"]
    assert "leave_policy.md#" in b["answer"] and b["contexts"]
    fb = [s for s in b["trace"]["spans"] if s["kind"] == "fallback"]
    assert fb and fb[0]["status"] == "error"
    assert any(s["kind"] == "llm" and s["status"] == "error" for s in b["trace"]["spans"])


def test_ask_degrades_with_no_provider_configured(client, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")   # real mode, but no keys present
    h = client.get("/api/health").json()
    assert h["mode"] == "retrieval-only" and h["providers"] == []
    b = client.post("/api/ask", json={"question": "What is the password policy?"}).json()
    assert b["mode"] == "retrieval-only" and b["notice"] and b["route"]["provider"] == ""
    c = client.post("/api/chat", json={"message": "hi"}).json()
    assert c["status"] == "done" and c["mode"] == "retrieval-only" and "needs a language model" in c["reply"]


def test_trace_records_token_counts(client, monkeypatch):
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from common.llm import FakeChatModel

    def gen(self, messages, stop=None, run_manager=None, **kw):
        m = AIMessage(content="ok", usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18})
        return ChatResult(generations=[ChatGeneration(message=m)])

    monkeypatch.setattr(FakeChatModel, "_generate", gen)
    spans = client.post("/api/ask", json={"question": "hotel cap?"}).json()["trace"]["spans"]
    llm = next(s for s in spans if s["kind"] == "llm")
    assert (llm["tokens_in"], llm["tokens_out"]) == (11, 7)


# --------------------------------------------------------------------------- guardrails
def test_input_validation(client):
    assert client.post("/api/ask", json={"question": "x" * 501}).status_code == 400
    r = client.post("/api/ask", json={"question": "   "})
    assert r.status_code == 400 and r.json()["error"] == "empty_input"
    r = client.post("/api/ask", json={"question": "hi", "provider": "evil-provider"})
    assert r.status_code == 400 and r.json()["error"] == "unknown_provider"
    r = client.post("/api/ask", json={"question": "hi", "provider": "groq"})   # known but no key
    assert r.status_code == 400 and r.json()["error"] == "provider_not_configured"
    assert client.post("/api/ask", json={}).status_code == 400
    assert client.post("/api/chat", json={"message": "x" * 501}).status_code == 400
    assert client.post("/api/chat", json={"message": "hi", "thread_id": "../../etc"}).status_code == 400


def test_control_characters_are_stripped(client):
    b = client.post("/api/ask", json={"question": "hotel\x00 cap\x07?"}).json()
    assert "\x00" not in json.dumps(b) and "\\u0000" not in json.dumps(b)


def test_rate_limit_burst_returns_429_with_retry_after(client):
    codes = [client.post("/api/ask", json={"question": "hotel cap?"}).status_code for _ in range(10)]
    assert codes[:8] == [200] * 8 and codes[8] == 429
    r = client.post("/api/ask", json={"question": "hotel cap?"})
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1 and r.json()["error"] == "rate_limited"


def test_rate_limit_is_per_ip_via_forwarded_for(client):
    for _ in range(8):
        client.post("/api/ask", json={"question": "hotel cap?"}, headers={"X-Forwarded-For": "1.1.1.1, 10.0.0.1"})
    assert client.post("/api/ask", json={"question": "x y"}, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 429
    assert client.post("/api/ask", json={"question": "x y"}, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 200


def test_overall_limit_and_refill():
    rl = RateLimiter()
    for _ in range(30):
        rl.check("ip", ("overall",), now=0.0)
    with pytest.raises(Exception) as e:
        rl.check("ip", ("overall",), now=0.0)
    assert e.value.retry_after >= 1
    rl.check("ip", ("overall",), now=3.0)   # 30/min refills one token every 2 s


# --------------------------------------------------------------------------- compare
def test_compare(client, monkeypatch):
    r = client.post("/api/compare", json={"question": "What is the hotel cap?", "providers": ["fake"]})
    assert r.status_code == 200
    res = r.json()["results"]
    assert res[0]["provider"] == "fake" and res[0]["answer"].startswith("FAKE ANSWER") and "error" not in res[0]
    r = client.post("/api/compare", json={"question": "q?", "providers": ["fake", "groq"]})
    groq = r.json()["results"][1]
    assert groq["provider"] == "groq" and "not configured" in groq["error"]
    assert client.post("/api/compare", json={"question": "q?", "providers": []}).status_code == 400
    assert client.post("/api/compare", json={"question": "q?", "providers": ["fake"] * 2 + ["a", "b", "c"]}).status_code == 400
    assert client.post("/api/compare", json={"question": "q?", "providers": ["nope"]}).status_code == 400


# --------------------------------------------------------------------------- agent
def test_chat_done_with_steps_and_trace(client):
    b = client.post("/api/chat", json={"message": "How many sick days do I get?"}).json()
    assert b["status"] == "done" and b["thread_id"] and b["mode"] == "llm"
    assert b["steps"][0]["name"] == "search_policies"
    assert b["route"]["tier"] in {"default", "complex"}                         # never below default for tool calling
    quick = client.post("/api/chat", json={"message": "hi"}).json()["route"]
    assert quick["tier"] == "default" and "tool calling" in quick["reason"]
    assert any(s["kind"] == "tool" for s in b["trace"]["spans"])


def test_chat_ticket_needs_approval_then_approve(client, monkeypatch):
    script_ticket_model(monkeypatch)
    b = client.post("/api/chat", json={"message": "please raise a P2 ticket for VPN outage"}).json()
    assert b["status"] == "pending_approval"
    assert b["approval_request"]["action"] == "create_it_ticket"
    assert b["approval_request"]["args"]["priority"] == "P2"
    assert b["steps"] == [{"type": "tool", "name": "create_it_ticket",
                           "args": {"summary": "VPN outage", "priority": "P2"}}]
    tid = b["thread_id"]
    done = client.post("/api/approve", json={"thread_id": tid, "approved": True}).json()
    assert done["status"] == "done" and "Created ticket INC-" in done["reply"]
    assert done["route"] and done["trace"]["spans"]
    again = client.post("/api/approve", json={"thread_id": tid, "approved": True})
    assert again.status_code == 400 and again.json()["error"] == "nothing_pending"


def test_reject_flow(client, monkeypatch):
    script_ticket_model(monkeypatch)
    tid = client.post("/api/chat", json={"message": "raise a ticket"}).json()["thread_id"]
    done = client.post("/api/approve", json={"thread_id": tid, "approved": False}).json()
    assert done["status"] == "done" and "NOT created" in done["reply"]


def test_approve_unknown_thread_is_404(client):
    r = client.post("/api/approve", json={"thread_id": "nope-123", "approved": True})
    assert r.status_code == 404 and r.json()["error"] == "unknown_thread"


def test_chat_survives_provider_failure(client, monkeypatch):
    break_every_provider(monkeypatch)
    b = client.post("/api/chat", json={"message": "How many sick days?"}).json()
    assert b["status"] == "done" and b["mode"] == "retrieval-only" and b["notice"]


def test_graph_default_behaviour_unchanged():
    sys.path.insert(0, str(ROOT / "projects" / "03_langgraph_agent"))
    from graph import build_graph

    assert build_graph().get_graph()   # llm=None -> get_llm() (fake here)


# --------------------------------------------------------------------------- evals
def test_evals(client):
    b = client.get("/api/evals").json()
    assert set(b["metrics"]) == {"source_hit_rate", "refusal_rate_out_of_scope", "citation_rate"}
    assert b["gate"] == {"name": "source_hit_rate", "threshold": 0.8, "passed": b["metrics"]["source_hit_rate"] >= 0.8}
    assert len(b["rows"]) == 10 and b["rows"][0].keys() >= {"question", "expected_source", "sources", "hit"}
    assert b["metrics"]["source_hit_rate"] >= 0.8 and b["gate"]["passed"]
    assert b["mode"] == "llm" and b["latency_ms"] >= 0
    assert client.get("/api/evals").json()["cached"] is True


def test_evals_retrieval_only(client, monkeypatch):
    break_every_provider(monkeypatch)
    b = client.get("/api/evals").json()
    assert b["mode"] == "retrieval-only" and b["metrics"]["refusal_rate_out_of_scope"] is None
    assert b["gate"]["passed"] and b["metrics"]["citation_rate"] == 1.0


# --------------------------------------------------------------------------- managed agents
def test_managed_not_configured(client):
    s = client.get("/api/managed/status").json()
    assert s["configured"] is False and s["hint"] and 2 <= len(s["tasks"]) <= 3
    assert s["tasks"][0].keys() >= {"id", "title", "description"}
    for r in (client.post("/api/managed/start", json={"task_id": s["tasks"][0]["id"]}),
              client.get("/api/managed/events?session_id=sesn_1"),
              client.get("/api/managed/file?session_id=sesn_1&file_id=file_1")):
        assert r.status_code == 503 and r.json()["error"] == "not_configured"


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _managed_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("MANAGED_AGENT_ID", "agent_1")
    monkeypatch.setenv("MANAGED_AGENT_VERSION", "3")
    monkeypatch.setenv("MANAGED_ENVIRONMENT_ID", "env_1")


class FakeSessions:
    """Stands in for client.beta: records sends, serves a scripted event log."""

    def __init__(self):
        self.sent, self.created, self.events_log = [], None, []
        self.beta = _Obj(sessions=_Obj(create=self.create, retrieve=self.retrieve,
                                        events=_Obj(list=self.list, send=self.send)),
                         files=_Obj(list=self.files_list, download=self.download))

    def create(self, **kw):
        self.created = kw
        return _Obj(id="sesn_1")

    def retrieve(self, session_id, **kw):
        return _Obj(metadata={"app": server.APP_TAG})

    def list(self, session_id, **kw):
        return iter(list(self.events_log))

    def send(self, session_id, events, **kw):
        self.sent.extend(events)
        for e in events:
            if e["type"] == "user.custom_tool_result":
                self.events_log.append(_Obj(id=f"r_{e['custom_tool_use_id']}", type="user.custom_tool_result",
                                            custom_tool_use_id=e["custom_tool_use_id"], processed_at=None))

    def files_list(self, **kw):
        return _Obj(data=[_Obj(id="file_1", filename="faq.md", size_bytes=12)])

    def download(self, file_id, **kw):
        return _Obj(read=lambda: b"# hello")


def test_managed_start_poll_dedupes_tool_results_and_downloads(client, monkeypatch):
    _managed_env(monkeypatch)
    fake = FakeSessions()
    monkeypatch.setattr(server, "_client", lambda: fake)
    server._ANSWERED_TOOL_USES.clear()
    server._VERIFIED_SESSIONS.clear()

    started = client.post("/api/managed/start", json={"task_id": "handbook"})
    assert started.status_code == 200 and started.json()["session_id"] == "sesn_1"
    assert fake.created["budget"] == {"type": "limit", "max_list_cost": {"amount": "100", "currency": "USD"}}
    assert fake.created["agent"] == {"type": "agent", "id": "agent_1", "version": "3"}
    assert fake.sent[0]["type"] == "user.message"
    assert client.post("/api/managed/start", json={"task_id": "rm -rf"}).status_code == 400

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    fake.events_log = [
        _Obj(id="e1", type="session.status_running", processed_at=now),
        _Obj(id="e2", type="agent.custom_tool_use", name="search_acme_policies",
             input={"query": "hotel cap per night"}, processed_at=now),
        _Obj(id="e3", type="session.status_idle", processed_at=now, stop_reason=_Obj(type="requires_action")),
    ]
    fake.sent.clear()
    first = client.get("/api/managed/events?session_id=sesn_1").json()
    assert [e["type"] for e in first["events"]] == ["session.status_running", "agent.custom_tool_use", "session.status_idle"]
    assert first["done"] is False and first["files"] == []
    results = [e for e in fake.sent if e["type"] == "user.custom_tool_result"]
    assert len(results) == 1 and results[0]["custom_tool_use_id"] == "e2"
    assert "expense_policy.md" in results[0]["content"][0]["text"]

    for _ in range(3):   # repeated polls (also racing the log) must never answer twice
        client.get("/api/managed/events?session_id=sesn_1")
    server._ANSWERED_TOOL_USES.clear()   # even if memory is lost, the log shows it was answered
    client.get("/api/managed/events?session_id=sesn_1")
    assert len([e for e in fake.sent if e["type"] == "user.custom_tool_result"]) == 1

    fake.events_log += [
        _Obj(id="e4", type="agent.message", content=[_Obj(type="text", text="All done")], processed_at=now),
        _Obj(id="e5", type="session.status_idle", processed_at=now, stop_reason=_Obj(type="end_turn")),
    ]
    last = client.get("/api/managed/events?session_id=sesn_1&after=e3").json()
    ids = [e["id"] for e in last["events"]]
    assert "e4" in ids and "e1" not in ids and last["done"] is True and last["cursor"] == "e5"
    assert last["files"] == [{"id": "file_1", "name": "faq.md", "size": 12}]

    f = client.get("/api/managed/file?session_id=sesn_1&file_id=file_1")
    assert f.status_code == 200 and f.content == b"# hello"
    assert f.headers["content-disposition"] == 'attachment; filename="faq.md"'
    assert client.get("/api/managed/file?session_id=sesn_1&file_id=other").status_code == 404
    assert SECRET not in f.text and SECRET not in json.dumps(last)


def test_managed_refuses_foreign_sessions(client, monkeypatch):
    _managed_env(monkeypatch)
    fake = FakeSessions()
    fake.retrieve = lambda session_id, **kw: _Obj(metadata={})
    fake.beta.sessions.retrieve = fake.retrieve
    monkeypatch.setattr(server, "_client", lambda: fake)
    server._VERIFIED_SESSIONS.clear()
    assert client.get("/api/managed/events?session_id=sesn_other").status_code == 404


# --------------------------------------------------------------------------- bundle weight
def test_heavy_imports_not_loaded():
    """The BM25 path must work with the vector stack unavailable (serverless bundle / cold start)."""
    code = """
import os, sys
os.environ.update(LLM_PROVIDER="fake", RETRIEVER="bm25", EMBEDDINGS_PROVIDER="fake")
for name in ("chromadb", "fastembed", "faiss", "onnxruntime", "langchain_chroma", "langchain_community"):
    sys.modules[name] = None   # any import of these raises ImportError
from fastapi.testclient import TestClient
from showcase.server import app
c = TestClient(app)
assert c.get("/api/health").status_code == 200
r = c.post("/api/ask", json={"question": "What is the hotel cap?"})
assert r.status_code == 200 and r.json()["sources"], r.text
r = c.post("/api/chat", json={"message": "How many sick days?"})
assert r.status_code == 200 and r.json()["status"] == "done", r.text
assert c.get("/api/evals").status_code == 200
print("OK")
"""
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "OK" in out.stdout, out.stderr[-2000:]
