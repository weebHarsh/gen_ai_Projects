"""Offline tests for the hosted demo API (showcase/server.py). No keys, no network.

    LLM_PROVIDER=fake RETRIEVER=bm25 pytest -q tests/test_showcase.py
"""
from __future__ import annotations

import base64
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
CODE = "open-sesame-123"
UNLOCK = {"x-demo-unlock": CODE}


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    server.limiter.reset()
    server.LLM_BUDGET.reset()
    server.MANAGED_BUDGET_COUNTER.reset()
    server._EVAL_CACHE.update(at=0.0, mode=None, payload=None)
    server._THREADS.clear()
    server._THREAD_LOCKS.clear()
    server._CHECKPOINTER = None
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("DEMO_UNLOCK", CODE)
    for k in ("ANTHROPIC_API_KEY", "MANAGED_AGENT_ID", "MANAGED_ENVIRONMENT_ID", "VERCEL", "VERCEL_PROJECT_ID",
              "VERCEL_OIDC_TOKEN", "AI_GATEWAY_API_KEY", "DEMO_LLM_ENABLED", "DEMO_LLM_CALLS_PER_HOUR",
              "MANAGED_STARTS_PER_HOUR", "MAX_TIER", "EVAL_USE_LLM"):
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
    hard = client.post("/api/ask", headers=UNLOCK, json={
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


def test_rate_limit_is_per_ip_via_vercel_header(client, monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    for _ in range(8):
        client.post("/api/ask", json={"question": "hotel cap?"}, headers={"x-vercel-forwarded-for": "1.1.1.1, 10.0.0.1"})
    assert client.post("/api/ask", json={"question": "x y"},
                       headers={"x-vercel-forwarded-for": "1.1.1.1"}).status_code == 429
    assert client.post("/api/ask", json={"question": "x y"},
                       headers={"x-vercel-forwarded-for": "2.2.2.2"}).status_code == 200
    # x-real-ip is the fallback; a client X-Forwarded-For is ignored even on Vercel
    assert client.post("/api/ask", json={"question": "x y"},
                       headers={"x-real-ip": "3.3.3.3", "x-forwarded-for": "9.9.9.9"}).status_code == 200
    from showcase.ratelimit import client_ip
    from starlette.requests import Request
    req = Request({"type": "http", "headers": [(b"x-forwarded-for", b"7.7.7.7"), (b"x-real-ip", b"3.3.3.3")],
                   "client": ("10.1.1.1", 1)})
    assert client_ip(req) == "3.3.3.3"


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
    client.headers.update(UNLOCK)
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
def test_evals(client, monkeypatch):
    monkeypatch.setenv("EVAL_USE_LLM", "1")   # opt in to the model-backed path
    b = client.get("/api/evals").json()
    assert set(b["metrics"]) == {"source_hit_rate", "refusal_rate_out_of_scope", "citation_rate", "top1_rate", "mrr"}
    assert b["gate"] == {"name": "mrr", "threshold": 0.8, "passed": b["metrics"]["mrr"] >= 0.8}
    assert len(b["rows"]) == 10 and b["rows"][0].keys() >= {"question", "expected_source", "sources", "hit", "rank"}
    # Rank-based metrics are the honest ones: hit-in-top-4 is trivially high on an 8-document corpus.
    assert 0.8 <= b["metrics"]["mrr"] <= 1.0 and b["gate"]["passed"]
    assert b["metrics"]["top1_rate"] <= b["metrics"]["source_hit_rate"]
    answerable = [r for r in b["rows"] if r["expected_source"]]
    assert all(isinstance(r["rank"], int) and r["rank"] >= 1 for r in answerable)
    assert [r["rank"] for r in b["rows"] if not r["expected_source"]] == [None]
    assert b["mode"] == "llm" and b["latency_ms"] >= 0
    assert client.get("/api/evals").json()["cached"] is True


def test_evals_retrieval_only_by_default(client, monkeypatch):
    calls = []
    monkeypatch.setattr(server, "build_model", lambda *a, **k: calls.append(1))   # must never be reached
    b = client.get("/api/evals").json()
    assert b["mode"] == "retrieval-only" and not calls
    assert b["metrics"]["refusal_rate_out_of_scope"] is None and b["metrics"]["citation_rate"] is None
    assert b["gate"]["passed"] and 0.8 <= b["metrics"]["mrr"] <= 1.0
    assert all(r["hit"] is None for r in b["rows"] if not r["expected_source"])


def test_evals_llm_failure_falls_back_to_retrieval_only(client, monkeypatch):
    monkeypatch.setenv("EVAL_USE_LLM", "1")
    break_every_provider(monkeypatch)
    b = client.get("/api/evals").json()
    assert b["mode"] == "retrieval-only" and b["metrics"]["citation_rate"] is None


def test_evals_computed_once_under_concurrency(client, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    n = []
    real = server._compute_evals
    monkeypatch.setattr(server, "_compute_evals", lambda use_llm: (n.append(1), real(use_llm))[1])
    with ThreadPoolExecutor(4) as pool:
        out = list(pool.map(lambda _: client.get("/api/evals").json(), range(4)))
    assert len(n) == 1 and sum(bool(o.get("cached")) for o in out) == 3


# --------------------------------------------------------------------------- managed agents
def test_managed_not_configured(client):
    s = client.get("/api/managed/status").json()
    assert s["configured"] is False and s["hint"] and 2 <= len(s["tasks"]) <= 3
    assert s["tasks"][0].keys() >= {"id", "title", "description"}
    for r in (client.post("/api/managed/start", json={"task_id": s["tasks"][0]["id"]}, headers=UNLOCK),
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

    client.headers.update(UNLOCK)
    started = client.post("/api/managed/start", json={"task_id": "handbook"})
    assert started.status_code == 200 and started.json()["session_id"] == "sesn_1"
    assert fake.created["budget"] == {"type": "limit", "max_list_cost": {"amount": "25", "currency": "USD"}}
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


# =========================================================================== v2: access control, cost caps, resilience
def fake_chain(monkeypatch):
    """Keep the real routing/ordering but never build a real provider client (no network in tests)."""
    from common.llm import FakeChatModel

    monkeypatch.setattr(server, "get_llm_chain", lambda **kw: FakeChatModel())
    monkeypatch.setattr(server, "get_llm", lambda **kw: FakeChatModel())


def make_jwt(**claims) -> str:
    def b64(obj) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")

    return f"{b64({'alg': 'RS256'})}.{b64(claims)}.sig"


HARD_Q = "Compare the leave and expense policies and explain any conflicts between them"


# ----- unlock
def test_health_access_block_and_unlock(client):
    a = client.get("/api/health").json()["access"]
    assert a == {"unlock_configured": True, "unlocked": False, "max_tier_public": "default"}
    assert client.get("/api/health", headers=UNLOCK).json()["access"]["unlocked"] is True
    assert client.get("/api/health", headers={"x-demo-unlock": "wrong"}).json()["access"]["unlocked"] is False
    r = client.post("/api/unlock", json={"code": CODE})
    assert r.status_code == 200 and r.json() == {"unlocked": True}
    r = client.post("/api/unlock", json={"code": "nope"})
    assert r.status_code == 403 and r.json()["error"] == "locked"
    assert CODE not in client.get("/api/health", headers=UNLOCK).text
    assert client.post("/api/unlock", json={"code": "é" * 50}).status_code == 403   # non-ASCII must not 500


def test_unlock_fails_closed_when_not_configured(client, monkeypatch):
    monkeypatch.delenv("DEMO_UNLOCK")
    h = client.get("/api/health", headers=UNLOCK).json()["access"]
    assert h["unlock_configured"] is False and h["unlocked"] is False
    assert client.post("/api/unlock", json={"code": ""}).status_code == 403
    assert client.post("/api/unlock", json={"code": CODE}).status_code == 403
    for hdr in ({}, {"x-demo-unlock": ""}, UNLOCK):
        r = client.post("/api/compare", json={"question": "q?", "providers": ["fake"]}, headers=hdr)
        assert r.status_code == 403 and r.json()["error"] == "locked"


def test_unlock_counts_against_heavy_bucket(client):
    codes = [client.post("/api/unlock", json={"code": f"guess{i}"}).status_code for i in range(10)]
    assert codes[:8] == [403] * 8 and codes[8] == 429


def test_compare_and_managed_start_locked_for_public(client, monkeypatch):
    for hdr in ({}, {"x-demo-unlock": "wrong"}):
        r = client.post("/api/compare", json={"question": "q?", "providers": ["fake"]}, headers=hdr)
        assert r.status_code == 403 and r.json()["error"] == "locked" and r.json()["message"]
    _managed_env(monkeypatch)
    fake = FakeSessions()
    monkeypatch.setattr(server, "_client", lambda: fake)
    r = client.post("/api/managed/start", json={"task_id": "handbook"})
    assert r.status_code == 403 and r.json()["error"] == "locked" and fake.created is None
    assert client.post("/api/managed/start", json={"task_id": "handbook"}, headers=UNLOCK).status_code == 200


# ----- tier clamp and provider order (B2)
def test_public_tier_is_clamped_unlocked_gets_opus(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    fake_chain(monkeypatch)
    pub = client.post("/api/ask", json={"question": HARD_Q}).json()["route"]
    assert pub["tier"] == "default" and pub["model"] == "claude-sonnet-5-5" and "capped" in pub["reason"]
    pres = client.post("/api/ask", json={"question": HARD_Q}, headers=UNLOCK).json()["route"]
    assert pres["tier"] == "complex" and pres["model"] == "claude-opus-5-5"
    chat = client.post("/api/chat", json={"message": HARD_Q}).json()["route"]
    assert chat["tier"] == "default"
    monkeypatch.setenv("MAX_TIER", "quick")
    assert client.post("/api/ask", json={"question": HARD_Q}).json()["route"]["tier"] == "quick"
    monkeypatch.setenv("MAX_TIER", "complex")
    assert client.post("/api/ask", json={"question": HARD_Q}).json()["route"]["tier"] == "complex"
    monkeypatch.setenv("MAX_TIER", "bogus")   # invalid value falls back to the safe default
    assert client.post("/api/ask", json={"question": HARD_Q}).json()["route"]["tier"] == "default"


def test_public_cannot_pin_a_directly_billed_provider(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-123456789")
    fake_chain(monkeypatch)
    r = client.post("/api/ask", json={"question": "How many sick days?", "provider": "anthropic"})
    assert r.status_code == 403 and r.json()["error"] == "locked"
    assert SECRET not in r.text
    # free providers can still be pinned by anyone; the presenter can pin paid ones
    assert client.post("/api/ask", json={"question": "How many sick days?", "provider": "groq"}).status_code == 200
    ok = client.post("/api/ask", json={"question": "How many sick days?", "provider": "anthropic"}, headers=UNLOCK)
    assert ok.status_code == 200 and ok.json()["route"]["pinned"] is True


def test_provider_order_public_vs_unlocked(monkeypatch):
    from common import providers as reg

    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "gw-key-123456")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-123456789")
    assert reg.order_for(True) == ["anthropic", "gateway", "groq"] == reg.available()
    assert reg.order_for(False) == ["groq", "gateway", "anthropic"]
    d, _ = server.decide("hello there friend", "auto", unlocked=False)
    assert d.provider == "groq"
    d, _ = server.decide("hello there friend", "auto", unlocked=True)
    assert d.provider == "anthropic"


def test_llm_limits_and_chain_length(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "AI_GATEWAY_API_KEY"):
        monkeypatch.setenv(k, "key-123456789")
    assert server.LLM_KWARGS["max_tokens"] <= 800 and server.LLM_KWARGS["timeout"] == 10
    assert server.LLM_KWARGS["max_retries"] == 0 and server.RECURSION_LIMIT == 8
    d, _ = server.decide("hello there friend", "auto")
    chain = server.build_model(d, False, False)
    assert len(chain.fallbacks) == 1   # at most 2 providers
    for m in (chain.runnable, *chain.fallbacks):
        assert m.max_retries == 0 and m.max_tokens <= 800


# ----- H1: the OIDC header never touches process state
def test_oidc_header_never_changes_providers_or_environ(client, monkeypatch):
    before_env = dict(os.environ)
    before = client.get("/api/health").json()
    for vercel in (None, "1"):
        if vercel:
            monkeypatch.setenv("VERCEL", "1")
            monkeypatch.setenv("VERCEL_PROJECT_ID", "prj_123")
            before_env = dict(os.environ)
        for tok in ("attacker-junk", make_jwt(iss="https://evil.example/x", project_id="prj_123")):
            r = client.get("/api/health", headers={"x-vercel-oidc-token": tok})
            assert r.json()["providers"] == before["providers"]
            assert client.get("/api/health").json()["providers"] == before["providers"]
            assert dict(os.environ) == before_env and "VERCEL_OIDC_TOKEN" not in os.environ
            b = client.post("/api/ask", json={"question": "hello there", "provider": "gateway"},
                            headers={"x-vercel-oidc-token": tok})
            assert b.status_code == 400 and b.json()["error"] == "provider_not_configured"


def test_oidc_forged_or_mismatched_tokens_are_ignored(monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setenv("VERCEL_PROJECT_ID", "prj_123")
    good = dict(iss="https://oidc.vercel.com/team", project_id="prj_123")
    assert server.verified_oidc(make_jwt(**good)) is not None
    for bad in (make_jwt(**{**good, "project_id": "prj_other"}),
                make_jwt(**{**good, "iss": "https://oidc.vercel.com.evil.test/x"}),
                make_jwt(**{**good, "iss": "https://evil.example/"}),
                make_jwt(**good, exp=1),
                make_jwt(iss=good["iss"]),
                "not.a.jwt", "junk", "", None):
        assert server.verified_oidc(bad) is None
    monkeypatch.delenv("VERCEL_PROJECT_ID")   # no configured project: nothing can match
    assert server.verified_oidc(make_jwt(**good)) is None
    monkeypatch.setenv("VERCEL_PROJECT_ID", "prj_123")
    monkeypatch.delenv("VERCEL")              # not on Vercel: ignored even when well-formed
    assert server.verified_oidc(make_jwt(**good)) is None


def test_verified_oidc_is_request_scoped_only(client, monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setenv("VERCEL_PROJECT_ID", "prj_123")
    tok = make_jwt(iss="https://oidc.vercel.com/team", project_id="prj_123")
    seen = []
    from common import providers as reg

    monkeypatch.setattr(server, "health", server.health)   # keep route; sample the registry inside a request
    real = server.active_providers
    monkeypatch.setattr(server, "active_providers", lambda: (seen.append(reg.gateway_token()), real())[1])
    r = client.get("/api/health", headers={"x-vercel-oidc-token": tok})
    assert "gateway" in r.json()["providers"] and seen[0] == tok
    assert "gateway" not in client.get("/api/health").json()["providers"]
    assert "VERCEL_OIDC_TOKEN" not in os.environ
    assert reg.gateway_token() is None


# ----- M1: client IP
def test_xff_spoof_does_not_bypass_rate_limit_off_vercel(client):
    codes = [client.post("/api/ask", json={"question": "hotel cap?"},
                         headers={"X-Forwarded-For": f"10.0.0.{i}", "x-vercel-forwarded-for": f"8.8.8.{i}",
                                  "x-real-ip": f"7.7.7.{i}"}).status_code for i in range(12)]
    assert codes[:8] == [200] * 8 and set(codes[8:]) == {429}


def test_rate_limit_buckets_are_bounded(monkeypatch):
    from showcase import ratelimit

    monkeypatch.setattr(ratelimit, "MAX_KEYS", 50)
    rl = RateLimiter()
    for i in range(500):
        rl.check(f"ip{i}", ("overall", "heavy"), now=float(i))
    assert len(rl._buckets) <= 50


# ----- H3 / H4: threads
def test_thread_ids_are_server_issued_hex_and_validated(client):
    import re

    tid = client.post("/api/chat", json={"message": "hi"}).json()["thread_id"]
    assert re.fullmatch(r"[0-9a-f]{32}", tid)
    for bad in ("a", "t-abc123", "../../etc", "A" * 32, "g" * 32, "0" * 31, "0" * 33):
        r = client.post("/api/chat", json={"message": "hi", "thread_id": bad})
        assert r.status_code == 400 and r.json()["error"] == "bad_request", bad
    server.limiter.reset()
    fresh = "ab" * 16   # well-formed but unknown (e.g. after an instance restart): starts a fresh thread
    r = client.post("/api/chat", json={"message": "hi", "thread_id": fresh})
    assert r.status_code == 200 and r.json()["thread_id"] == fresh
    for bad in ("nope", "g" * 32):
        r = client.post("/api/approve", json={"thread_id": bad, "approved": True})
        assert r.status_code == 404 and r.json()["error"] == "unknown_thread"
    assert client.post("/api/approve", json={"thread_id": "cd" * 16, "approved": True}).status_code == 404


def test_chat_while_approval_pending_is_409_and_thread_survives(client, monkeypatch):
    script_ticket_model(monkeypatch)
    b = client.post("/api/chat", json={"message": "please raise a ticket"}).json()
    assert b["status"] == "pending_approval"
    tid = b["thread_id"]
    r = client.post("/api/chat", json={"message": "something else", "thread_id": tid})
    assert r.status_code == 409 and r.json()["error"] == "approval_pending"
    done = client.post("/api/approve", json={"thread_id": tid, "approved": True}).json()
    assert done["status"] == "done" and "Created ticket INC-" in done["reply"]
    assert client.post("/api/chat", json={"message": "thanks", "thread_id": tid}).status_code == 200


def test_thread_turn_cap(client):
    tid = None
    for i in range(10):
        server.limiter.reset()
        body = {"message": f"hello {i}"} | ({"thread_id": tid} if tid else {})
        r = client.post("/api/chat", json=body)
        assert r.status_code == 200, i
        tid = r.json()["thread_id"]
    server.limiter.reset()
    r = client.post("/api/chat", json={"message": "one more", "thread_id": tid})
    assert r.status_code == 409 and r.json()["error"] == "thread_full"
    # a new thread is fine
    assert client.post("/api/chat", json={"message": "new"}).status_code == 200


def test_checkpoint_deleted_after_recursion_error(client, monkeypatch):
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from common.llm import FakeChatModel

    def loop(self, messages, stop=None, run_manager=None, **kw):   # always asks for another tool call
        msg = AIMessage(content="", tool_calls=[{"name": "search_policies", "id": f"c{len(messages)}",
                                                 "args": {"query": "x"}}])
        return ChatResult(generations=[ChatGeneration(message=msg)])

    monkeypatch.setattr(FakeChatModel, "_generate", loop)
    b = client.post("/api/chat", json={"message": "loop forever"}).json()
    assert b["status"] == "done" and "too many steps" in b["reply"]
    key = {"configurable": {"thread_id": b["thread_id"]}}
    assert server.checkpointer().get_tuple(key) is None
    assert b["thread_id"] not in server._THREADS


def test_checkpoint_deleted_after_agent_exception_and_route_failed(client, monkeypatch):
    break_every_provider(monkeypatch)
    b = client.post("/api/chat", json={"message": "How many sick days?"}).json()
    assert b["mode"] == "retrieval-only" and b["route"]["failed"] is True
    assert server.checkpointer().get_tuple({"configurable": {"thread_id": b["thread_id"]}}) is None
    assert SECRET not in json.dumps(b)


# ----- B2: history trimming
def test_trim_history_never_orphans_tool_messages():
    sys.path.insert(0, str(ROOT / "projects" / "03_langgraph_agent"))
    from graph import trim_history
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    msgs, n = [], 0
    for turn in range(12):
        msgs.append(HumanMessage(f"q{turn}"))
        for _ in range(turn % 3):     # 0-2 tool round trips, sometimes with two parallel calls
            calls = [{"name": "t", "args": {}, "id": f"c{n}a"}, {"name": "t", "args": {}, "id": f"c{n}b"}]
            msgs.append(AIMessage("", tool_calls=calls))
            msgs += [ToolMessage("r", tool_call_id=c["id"]) for c in calls]
            n += 1
        msgs.append(AIMessage(f"a{turn}"))
        for upto in range(1, len(msgs) + 1):
            window = trim_history(msgs[:upto], 8)
            assert window[0].type == "human" or len(window) == upto
            issued: set[str] = set()
            for m in window:
                if m.type == "ai":
                    issued |= {c["id"] for c in (m.tool_calls or [])}
                if m.type == "tool":
                    assert m.tool_call_id in issued, "orphan tool message"
            assert window[-1] is msgs[upto - 1] and len(window) <= max(8, upto)
    assert len(trim_history(msgs, 8)) < len(msgs)


def test_agent_only_sees_trimmed_history(client, monkeypatch):
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from common.llm import FakeChatModel

    seen = []

    def gen(self, messages, stop=None, run_manager=None, **kw):
        seen.append([m.type for m in messages])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    monkeypatch.setattr(FakeChatModel, "_generate", gen)
    tid = None
    for i in range(8):
        server.limiter.reset()
        r = client.post("/api/chat", json={"message": f"m{i}"} | ({"thread_id": tid} if tid else {})).json()
        tid = r["thread_id"]
    assert len(seen[-1]) <= 1 + 9 and seen[-1][0] == "system" and seen[-1][1] == "human"
    assert len(seen[-1]) < 1 + 2 * 8 - 1   # the full history would be 15 messages


# ----- B2 / H2: global kill switch, hourly cap, deadline
def test_kill_switch_degrades_everything_to_retrieval_only(client, monkeypatch):
    monkeypatch.setenv("DEMO_LLM_ENABLED", "0")
    assert client.get("/api/health").json()["mode"] == "retrieval-only"
    b = client.post("/api/ask", json={"question": "How many sick days do I get?"}, headers=UNLOCK).json()
    assert b["mode"] == "retrieval-only" and b["notice"] and "leave_policy.md#" in b["answer"]
    c = client.post("/api/chat", json={"message": "hi"}).json()
    assert c["mode"] == "retrieval-only" and c["status"] == "done"
    monkeypatch.setenv("DEMO_LLM_ENABLED", "1")
    assert client.get("/api/health").json()["mode"] == "llm"


def test_hourly_cap_degrades_public_but_not_unlocked(client, monkeypatch):
    monkeypatch.setenv("DEMO_LLM_CALLS_PER_HOUR", "3")
    modes = [client.post("/api/ask", json={"question": "hotel cap?"}).json() for _ in range(2)]
    assert [m["mode"] for m in modes] == ["llm", "llm"]
    c = client.post("/api/chat", json={"message": "hi"}).json()          # third counted call
    assert c["mode"] == "llm"
    capped = client.post("/api/ask", json={"question": "hotel cap?"}).json()
    assert capped["mode"] == "retrieval-only" and "hourly demo budget" in capped["notice"].lower()
    assert capped["contexts"] and "route" in capped
    chat = client.post("/api/chat", json={"message": "hi"}).json()
    assert chat["mode"] == "retrieval-only" and "hourly demo budget" in chat["notice"].lower()
    assert client.post("/api/ask", json={"question": "hotel cap?"}, headers=UNLOCK).json()["mode"] == "llm"


def test_hourly_cap_window_slides():
    c = server.HourlyCounter()
    assert c.take(1) and not c.take(1)
    c._stamps[0] -= 3601
    assert c.take(1)
    c.refund()
    assert c.take(1)


def test_request_deadline_skips_the_model(client, monkeypatch):
    monkeypatch.setattr(server, "REQUEST_DEADLINE_S", 0)
    called = []
    monkeypatch.setattr(server, "build_model", lambda *a, **k: called.append(1))
    a = client.post("/api/ask", json={"question": "hotel cap?"}).json()
    assert a["mode"] == "retrieval-only" and "ran out of time" in a["notice"]
    c = client.post("/api/chat", json={"message": "hi"}).json()
    assert c["status"] == "done" and c["mode"] == "retrieval-only" and "ran out of time" in c["reply"]
    assert c["steps"] == []


def test_agent_loop_checks_deadline_between_model_calls(monkeypatch):
    import time

    sys.path.insert(0, str(ROOT / "projects" / "03_langgraph_agent"))
    from graph import DEADLINE_MESSAGE, build_graph
    from common.llm import FakeChatModel

    class Boom(FakeChatModel):
        def _generate(self, *a, **k):
            raise AssertionError("model must not be called past the deadline")

        def bind_tools(self, tools, **kw):
            return Boom()

    g = build_graph(llm=Boom(), deadline=time.monotonic() - 1)
    out = g.invoke({"messages": [{"role": "user", "content": "hi"}]}, {"configurable": {"thread_id": "t"}})
    assert out["messages"][-1].content == DEADLINE_MESSAGE


# ----- B1: managed start cap
def test_managed_start_hourly_cap_and_refund(client, monkeypatch):
    _managed_env(monkeypatch)
    monkeypatch.setenv("MANAGED_STARTS_PER_HOUR", "2")
    fake = FakeSessions()
    monkeypatch.setattr(server, "_client", lambda: fake)
    codes = []
    for _ in range(4):
        server.limiter.reset()
        codes.append(client.post("/api/managed/start", json={"task_id": "expenses"}, headers=UNLOCK))
    assert [r.status_code for r in codes] == [200, 200, 429, 429]
    assert codes[2].json()["error"] == "managed_cap" and "Retry-After" in codes[2].headers
    # a failed start does not use up the allowance
    server.MANAGED_BUDGET_COUNTER.reset()
    real_create = fake.create

    def boom(**kw):
        raise RuntimeError("upstream down")

    fake.beta.sessions.create = boom
    for _ in range(3):
        server.limiter.reset()
        assert client.post("/api/managed/start", json={"task_id": "expenses"}, headers=UNLOCK).status_code == 502
    fake.beta.sessions.create = real_create
    server.limiter.reset()
    assert client.post("/api/managed/start", json={"task_id": "expenses"}, headers=UNLOCK).status_code == 200


# ----- M3: file size
def test_managed_file_too_large_is_413(client, monkeypatch):
    _managed_env(monkeypatch)
    fake = FakeSessions()
    monkeypatch.setattr(server, "_client", lambda: fake)
    server._VERIFIED_SESSIONS.clear()
    url = "/api/managed/file?session_id=sesn_1&file_id=file_1"
    assert client.get(url).status_code == 200
    fake.files_list = lambda **kw: _Obj(data=[_Obj(id="file_1", filename="big.bin", size_bytes=4_000_001)])
    fake.beta.files.list = fake.files_list
    r = client.get(url)
    assert r.status_code == 413 and r.json()["error"] == "too_large" and "console" in r.json()["message"]
    # metadata lies about the size: the downloaded bytes are checked too
    fake.files_list = lambda **kw: _Obj(data=[_Obj(id="file_1", filename="big.bin", size_bytes=10)])
    fake.beta.files.list = fake.files_list
    fake.download = lambda file_id, **kw: _Obj(read=lambda: b"x" * 4_000_001)
    fake.beta.files.download = fake.download
    assert client.get(url).status_code == 413


# ----- L5 / L7 / L8
def test_bidi_and_zero_width_characters_are_stripped():
    raw = "ho‮tel​ c⁦a⁩p‏?‪"
    assert server.clean_text(raw, "q") == "hotel cap?"
    with pytest.raises(server.ApiError):
        server.clean_text("‮​" * 50, "q")


def test_route_failed_flag(client, monkeypatch):
    ok = client.post("/api/ask", json={"question": "hotel cap?"}).json()
    assert "failed" not in ok["route"]
    break_every_provider(monkeypatch)
    b = client.post("/api/ask", json={"question": "hotel cap?"}).json()
    assert b["mode"] == "retrieval-only" and b["route"]["failed"] is True


def test_error_strings_never_contain_provider_messages(client, monkeypatch):
    leak = "sk-ant-LEAKED0123456789abcdef and the prompt text"
    monkeypatch.setenv("GROQ_API_KEY", "gsk-groq-key-123456")

    class HttpErr(Exception):
        status_code = 429

    def fake_get_llm(provider=None, **kw):
        if provider == "groq":
            raise HttpErr(leak)
        raise RuntimeError(leak)

    monkeypatch.setattr(server, "get_llm", fake_get_llm)
    r = client.post("/api/compare", headers=UNLOCK,
                    json={"question": "hotel cap?", "providers": ["groq", "fake"]})
    res = {x["provider"]: x for x in r.json()["results"]}
    assert res["groq"]["error"] == "HttpErr (HTTP 429)" and res["fake"]["error"] == "RuntimeError"
    assert "LEAKED" not in r.text and "prompt text" not in r.text


def test_ask_trace_and_notice_do_not_echo_provider_messages(client, monkeypatch):
    from common.llm import FakeChatModel

    def boom(self, *a, **k):
        raise RuntimeError(f"401 invalid x-api-key sk-ant-LEAKED0123456789abcdef {SECRET}")

    monkeypatch.setattr(FakeChatModel, "_generate", boom)
    text = client.post("/api/ask", json={"question": "hotel cap?"}).text
    chat = client.post("/api/chat", json={"message": "hello"}).text
    for t in (text, chat):
        assert "LEAKED" not in t and "invalid x-api-key" not in t and SECRET not in t
    spans = json.loads(text)["trace"]["spans"]
    assert any(s["kind"] == "llm" and s["detail"].endswith("RuntimeError") for s in spans)


def test_docs_and_openapi_disabled(client):
    assert client.get("/docs").status_code == 404 and client.get("/openapi.json").status_code == 404


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
