"""Hosted demo API (FastAPI, deployed on Vercel). Contract: docs/SHOWCASE_API.md.

    LLM_PROVIDER=fake RETRIEVER=bm25 uvicorn showcase.server:app --port 8099

Design rules: never 500 because a model is missing or down (degrade to retrieval-only), never leak a
secret, never trust the client (length caps, allow-lists, per-IP rate limits), always return a trace.
"""
from __future__ import annotations

import os

os.environ.setdefault("RETRIEVER", "bm25")  # must be set before rag_pipeline is imported

import importlib.util  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from collections import OrderedDict  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "projects" / "02_rag_basics", ROOT / "projects" / "03_langgraph_agent"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fastapi import Depends, FastAPI, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402

from common import providers as reg  # noqa: E402
from common.llm import get_llm, get_llm_chain  # noqa: E402
from common.llm import provider as default_provider  # noqa: E402
from common.observability import flush, trace_config  # noqa: E402
from common.router import Decision, route  # noqa: E402
from showcase.ratelimit import RateLimited, RateLimiter, client_ip  # noqa: E402
from showcase.tracing import Tracer, scrub  # noqa: E402

log = logging.getLogger("showcase")
STATIC = Path(__file__).parent / "static"
MAX_INPUT = 500
RECURSION_LIMIT = 12
LLM_KWARGS = {"max_tokens": 1000, "timeout": 25, "max_retries": 1}   # demo-sized, fail fast so the chain can fall through
TIER_ORDER = {"quick": 0, "default": 1, "complex": 2}
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,100}$")
NO_MODEL_NOTICE = ("No model provider is configured on this demo (or every provider failed), so this is the "
                   "retrieval-only fallback: the most relevant policy passages, with citations, and no generated text.")

app = FastAPI(title="GenAI showcase", version="1.0", docs_url=None, redoc_url=None, openapi_url=None)
limiter = RateLimiter()


# --------------------------------------------------------------------------- errors, guardrails
class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict | None = None) -> None:
        super().__init__(message)
        self.status, self.code, self.message, self.headers = status, code, message, headers or {}


def _err(status: int, code: str, message: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": code, "message": message}, status_code=status, headers=headers)


@app.exception_handler(ApiError)
async def _api_error(_: Request, e: ApiError):
    return _err(e.status, e.code, e.message, e.headers)


@app.exception_handler(RateLimited)
async def _rate_limited(_: Request, e: RateLimited):
    return _err(429, "rate_limited", f"Too many requests. Try again in {e.retry_after}s.",
                {"Retry-After": str(e.retry_after)})


@app.exception_handler(RequestValidationError)
async def _validation(_: Request, e: RequestValidationError):
    return _err(400, "bad_request", "Invalid request body or parameters.")


@app.exception_handler(StarletteHTTPException)
async def _http(_: Request, e: StarletteHTTPException):
    code = {404: "not_found", 405: "method_not_allowed"}.get(e.status_code, "http_error")
    return _err(e.status_code, code, "Not found." if e.status_code == 404 else "Request not allowed.")


@app.exception_handler(Exception)
async def _unhandled(_: Request, e: Exception):
    log.exception("unhandled error")
    return _err(500, "internal_error", "Something went wrong on the server.")


def rl_light(request: Request) -> None:
    limiter.check(client_ip(request), ("overall",))


def rl_heavy(request: Request) -> None:
    limiter.check(client_ip(request), ("overall", "heavy"))


def rl_poll(request: Request) -> None:
    limiter.check(client_ip(request), ("poll",))


@app.middleware("http")
async def _headers(request: Request, call_next):
    token = request.headers.get("x-vercel-oidc-token")
    if token:  # lets the `gateway` provider authenticate with the deployment's own identity
        os.environ["VERCEL_OIDC_TOKEN"] = token
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response


def clean_text(raw: Any, field: str) -> str:
    if not isinstance(raw, str):
        raise ApiError(400, "bad_request", f"`{field}` must be a string.")
    if len(raw) > MAX_INPUT:
        raise ApiError(400, "input_too_long", f"`{field}` is limited to {MAX_INPUT} characters.")
    text = CONTROL.sub("", raw).strip()
    if not text:
        raise ApiError(400, "empty_input", f"`{field}` must not be empty.")
    return text


def clean_id(raw: str, field: str) -> str:
    if not isinstance(raw, str) or not ID_RE.match(raw):
        raise ApiError(400, "bad_request", f"`{field}` is not a valid id.")
    return raw


# --------------------------------------------------------------------------- providers / routing
def fake_mode() -> bool:
    return default_provider() == "fake"


def active_providers() -> list[str]:
    names = reg.available()
    return names or (["fake"] if fake_mode() else [])


def llm_available() -> bool:
    return bool(active_providers())


def valid_provider(name: str) -> str:
    """Allow-list check for a caller-pinned provider. Returns the name or raises 400."""
    name = (name or "").strip().lower()
    if name == "fake" and fake_mode():
        return name
    if name not in reg.PROVIDERS:
        raise ApiError(400, "unknown_provider", f"Unknown provider {scrub(name, 40)!r}.")
    return name


def _is_free(provider: str) -> bool:
    return provider == "fake" or (provider in reg.PROVIDERS and reg.PROVIDERS[provider].free)


def route_dict(d: Decision, pinned: bool) -> dict:
    cost = 0.0 if _is_free(d.provider) else d.est_cost_usd()
    return {"tier": d.tier, "provider": d.provider, "model": d.model, "reason": d.reason,
            "est_cost_usd": round(cost, 6), "pinned": pinned}


def decide(question: str, provider: str, min_tier: str = "quick") -> tuple[Decision, bool]:
    """provider 'auto' -> the router picks the tier; a pinned provider skips routing. Never raises on no providers."""
    if provider != "auto":
        name = valid_provider(provider)
        if name != "fake" and not reg.is_configured(name):
            raise ApiError(400, "provider_not_configured", f"Provider {name!r} has no API key on this server.")
        model = "fake-chat" if name == "fake" else reg.PROVIDERS[name].model_for("default")
        return Decision(tier="default", reason="provider pinned by caller (routing skipped)", provider=name, model=model), True
    d = route(question)
    if TIER_ORDER[d.tier] < TIER_ORDER[min_tier]:
        d.reason += f" (raised to {min_tier}: tool calling must be reliable)"
        d.tier = min_tier
        if d.provider:
            d.model = reg.PROVIDERS[d.provider].model_for(min_tier)
    if not d.provider and fake_mode():
        d.provider, d.model = "fake", "fake-chat"
    return d, False


def build_model(d: Decision, pinned: bool):
    """The pinned provider alone, or the fallback chain starting from the best configured one."""
    if pinned:
        return get_llm(provider=d.provider, tier=d.tier, **LLM_KWARGS)
    return get_llm_chain(tier=d.tier, **LLM_KWARGS)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content
                       if not isinstance(b, dict) or b.get("type") == "text")
    return str(content)


def _cfg(tracer: Tracer, name: str, **extra: Any) -> dict:
    cfg = trace_config(name, tags=["showcase"], **extra)
    cfg["callbacks"] = [*cfg["callbacks"], tracer]
    return cfg


def _rag():
    import rag_pipeline

    return rag_pipeline


def _contexts(docs) -> list[dict]:
    return [{"source": d.metadata["source"], "chunk_id": d.metadata.get("chunk_id"), "text": d.page_content} for d in docs]


def extractive_answer(docs) -> str:
    top = docs[:2]
    body = "\n\n".join(f"{d.page_content.strip()} [{d.metadata['source']}#{d.metadata.get('chunk_id')}]" for d in top)
    return "Most relevant passages from the policies:\n\n" + body


# --------------------------------------------------------------------------- basic routes
@app.get("/")
def index():
    page = STATIC / "index.html"
    if page.exists():
        return FileResponse(page, media_type="text/html")
    return HTMLResponse("<!doctype html><title>GenAI showcase</title><h1>GenAI showcase API</h1>"
                        "<p>UI not built yet. See <code>/api/health</code>.</p>")


@app.get("/api/health")
def health():
    return {"status": "ok", "retriever": os.getenv("RETRIEVER", "bm25"),
            "mode": "llm" if llm_available() else "retrieval-only", "providers": active_providers()}


@app.get("/api/providers", dependencies=[Depends(rl_light)])
def providers():
    return {"providers": reg.describe(), "available": active_providers()}


# --------------------------------------------------------------------------- /api/ask
class AskIn(BaseModel):
    question: str
    provider: str = "auto"


def run_rag(question: str, d: Decision, pinned: bool, tracer: Tracer) -> dict:
    """Retrieve, then generate with the chosen model; degrade to extractive answers on any model failure."""
    rag = _rag()
    docs = rag.get_retriever().invoke(question, config={"callbacks": [tracer]})
    out: dict[str, Any] = {"contexts": _contexts(docs), "sources": sorted({x.metadata["source"] for x in docs})}
    if llm_available() and (d.provider or fake_mode()):
        try:
            from langchain_core.output_parsers import StrOutputParser

            chain = rag.RAG_PROMPT | build_model(d, pinned) | StrOutputParser()
            out["answer"] = chain.invoke({"context": rag.format_context(docs), "question": question},
                                         config=_cfg(tracer, "showcase-ask"))
            out["mode"] = "llm"
            return out
        except Exception as e:  # noqa: BLE001  quota, outage, bad key... never a 500
            log.warning("model call failed: %s", type(e).__name__)
            out["notice"] = NO_MODEL_NOTICE + f" (last error: {type(e).__name__})"
    else:
        out["notice"] = NO_MODEL_NOTICE
    out["answer"], out["mode"] = extractive_answer(docs), "retrieval-only"
    return out


@app.post("/api/ask", dependencies=[Depends(rl_heavy)])
def ask(body: AskIn):
    t0 = time.perf_counter()
    question = clean_text(body.question, "question")
    tracer = Tracer()
    with tracer.span("route", "route") as sp:
        d, pinned = decide(question, body.provider)
        sp["detail"] = f"tier={d.tier} provider={d.provider or 'none'} model={d.model or 'none'}"
    result = run_rag(question, d, pinned, tracer)
    flush()
    result["route"] = route_dict(d, pinned)
    result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    result["trace"] = tracer.snapshot()
    return result


# --------------------------------------------------------------------------- /api/compare
class CompareIn(BaseModel):
    question: str
    providers: list[str]


@app.post("/api/compare", dependencies=[Depends(rl_heavy)])
def compare(body: CompareIn):
    question = clean_text(body.question, "question")
    if not 1 <= len(body.providers) <= 4:
        raise ApiError(400, "bad_request", "`providers` needs between 1 and 4 names.")
    names = list(dict.fromkeys(valid_provider(p) for p in body.providers))
    rag = _rag()
    docs = rag.get_retriever().invoke(question)
    prompt_in = {"context": rag.format_context(docs), "question": question}

    def one(name: str) -> dict:
        model_id = "fake-chat" if name == "fake" else reg.PROVIDERS[name].model_for("default")
        res: dict[str, Any] = {"provider": name, "model": model_id}
        t = time.perf_counter()
        try:
            if name != "fake" and not reg.is_configured(name):
                raise RuntimeError("not configured on this server (no API key)")
            msg = (rag.RAG_PROMPT | get_llm(provider=name, tier="default", **LLM_KWARGS)).invoke(prompt_in)
            res["answer"] = _text(msg.content)
            usage = getattr(msg, "usage_metadata", None) or {}
            if usage.get("output_tokens") is not None:
                res["tokens_out"] = usage["output_tokens"]
        except Exception as e:  # noqa: BLE001
            res["answer"] = ""
            res["error"] = f"{type(e).__name__}: {scrub(e, 160)}"
        res["latency_ms"] = int((time.perf_counter() - t) * 1000)
        return res

    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        results = list(pool.map(one, names))
    return {"results": results}


# --------------------------------------------------------------------------- /api/chat + /api/approve
_CHECKPOINTER = None
_THREADS: "OrderedDict[str, bool]" = OrderedDict()   # thread_id -> waiting for approval?
_THREADS_LOCK = threading.Lock()
MAX_THREADS = 200
_THREAD_LOCKS: dict[str, threading.Lock] = {}


def checkpointer():
    global _CHECKPOINTER
    if _CHECKPOINTER is None:
        from langgraph.checkpoint.memory import InMemorySaver

        _CHECKPOINTER = InMemorySaver()
    return _CHECKPOINTER


def _remember(thread_id: str, pending: bool) -> None:
    with _THREADS_LOCK:
        _THREADS[thread_id] = pending
        _THREADS.move_to_end(thread_id)
        while len(_THREADS) > MAX_THREADS:
            old, _ = _THREADS.popitem(last=False)
            _THREAD_LOCKS.pop(old, None)
            try:
                checkpointer().delete_thread(old)
            except Exception:  # noqa: BLE001
                pass


def _thread_lock(thread_id: str) -> threading.Lock:
    with _THREADS_LOCK:
        return _THREAD_LOCKS.setdefault(thread_id, threading.Lock())


class ChatIn(BaseModel):
    message: str
    thread_id: str | None = None


class ApproveIn(BaseModel):
    thread_id: str
    approved: bool


def _steps(messages: list) -> list[dict]:
    """Tool calls made during the latest turn (everything after the last human message)."""
    last_human = max((i for i, m in enumerate(messages) if getattr(m, "type", "") == "human"), default=-1)
    return [{"type": "tool", "name": c["name"], "args": c["args"]}
            for m in messages[last_human + 1:] for c in (getattr(m, "tool_calls", None) or [])]


def agent_turn(payload: Any, thread_id: str, d: Decision, tracer: Tracer, label: str) -> dict:
    """Run one graph invocation. Returns the shared chat/approve response (without route/trace)."""
    from langgraph.errors import GraphRecursionError

    try:
        from graph import build_graph

        graph = build_graph(checkpointer=checkpointer(), llm=build_model(d, False))
        config = {**_cfg(tracer, label, session_id=thread_id),
                  "configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
        with _thread_lock(thread_id):
            result = graph.invoke(payload, config)
    except GraphRecursionError:
        _remember(thread_id, False)
        return {"status": "done", "thread_id": thread_id, "steps": [], "mode": "llm",
                "reply": "I stopped because this took too many steps. Try a simpler request."}
    except Exception as e:  # noqa: BLE001
        log.warning("agent turn failed: %s", type(e).__name__)
        return degraded_chat(thread_id, f" (last error: {type(e).__name__})")
    interrupts = result.get("__interrupt__")
    if interrupts:
        _remember(thread_id, True)
        return {"status": "pending_approval", "thread_id": thread_id, "mode": "llm",
                "approval_request": dict(interrupts[0].value), "steps": _steps(result["messages"])}
    _remember(thread_id, False)
    return {"status": "done", "thread_id": thread_id, "mode": "llm",
            "reply": _text(result["messages"][-1].content), "steps": _steps(result["messages"])}


def degraded_chat(thread_id: str, extra: str = "") -> dict:
    return {"status": "done", "thread_id": thread_id, "steps": [], "mode": "retrieval-only",
            "reply": "The agent needs a language model to decide which tools to call, and none is available right now. "
                     "The Ask section still works in retrieval-only mode.",
            "notice": NO_MODEL_NOTICE + extra}


def _finish_chat(resp: dict, d: Decision, tracer: Tracer) -> dict:
    flush()
    resp["route"] = route_dict(d, False)
    resp["trace"] = tracer.snapshot()
    return resp


@app.post("/api/chat", dependencies=[Depends(rl_heavy)])
def chat(body: ChatIn):
    message = clean_text(body.message, "message")
    thread_id = clean_id(body.thread_id, "thread_id") if body.thread_id else f"t-{uuid.uuid4().hex[:10]}"
    tracer = Tracer()
    with tracer.span("route", "route") as sp:
        d, _ = decide(message, "auto", min_tier="default")
        sp["detail"] = f"tier={d.tier} provider={d.provider or 'none'} model={d.model or 'none'}"
    if not llm_available():
        _remember(thread_id, False)
        return _finish_chat(degraded_chat(thread_id), d, tracer)
    resp = agent_turn({"messages": [{"role": "user", "content": message}]}, thread_id, d, tracer, "showcase-chat")
    return _finish_chat(resp, d, tracer)


@app.post("/api/approve", dependencies=[Depends(rl_heavy)])
def approve(body: ApproveIn):
    from langgraph.types import Command

    thread_id = clean_id(body.thread_id, "thread_id")
    known = checkpointer().get_tuple({"configurable": {"thread_id": thread_id}}) is not None
    if not known:
        raise ApiError(404, "unknown_thread", "No such conversation on this server instance. Start a new chat.")
    if not _THREADS.get(thread_id):
        raise ApiError(400, "nothing_pending", "This conversation has no action waiting for approval.")
    tracer = Tracer()
    with tracer.span("route", "route") as sp:
        d, _ = decide("resume approved action", "auto", min_tier="default")
        d.reason = "resuming a paused agent turn at the default tier"
        sp["detail"] = f"tier={d.tier} provider={d.provider or 'none'} model={d.model or 'none'}"
    if not llm_available():
        return _finish_chat(degraded_chat(thread_id), d, tracer)
    resp = agent_turn(Command(resume={"approved": body.approved}), thread_id, d, tracer, "showcase-approve")
    return _finish_chat(resp, d, tracer)


# --------------------------------------------------------------------------- /api/evals
_EVAL_CACHE: dict[str, Any] = {"at": 0.0, "mode": None, "payload": None}
EVAL_TTL = 600
GATE = 0.8


def _load_eval_module():
    if "showcase_eval_rag" in sys.modules:
        return sys.modules["showcase_eval_rag"]
    path = ROOT / "projects" / "06_agent_evals" / "eval_rag.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("showcase_eval_rag", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["showcase_eval_rag"] = mod
    spec.loader.exec_module(mod)
    return mod


@app.get("/api/evals", dependencies=[Depends(rl_light)])
def evals():
    """Golden-set RAG eval, deterministic metrics only. Cached 10 min so a public URL cannot burn model budget."""
    t0 = time.perf_counter()
    mode_now = "llm" if llm_available() else "retrieval-only"
    if _EVAL_CACHE["payload"] and _EVAL_CACHE["mode"] == mode_now and time.time() - _EVAL_CACHE["at"] < EVAL_TTL:
        return {**_EVAL_CACHE["payload"], "cached": True, "latency_ms": int((time.perf_counter() - t0) * 1000)}
    ev = _load_eval_module()
    rag = _rag()
    cases = ev.DATASET["rag"]
    d, _ = decide("short lookup", "auto") if llm_available() else (None, False)
    if d is not None:
        d.tier = "quick"
        d.model = reg.PROVIDERS[d.provider].model_for("quick") if d.provider in reg.PROVIDERS else d.model
    state = {"llm_ok": d is not None}

    def one(case: dict) -> dict:
        docs = rag.get_retriever().invoke(case["question"])
        sources = sorted({x.metadata["source"] for x in docs})
        answer = None
        if state["llm_ok"]:
            try:
                from langchain_core.output_parsers import StrOutputParser

                chain = rag.RAG_PROMPT | build_model(d, False) | StrOutputParser()
                answer = chain.invoke({"context": rag.format_context(docs), "question": case["question"]})
            except Exception as e:  # noqa: BLE001
                log.warning("eval model call failed: %s", type(e).__name__)
                state["llm_ok"] = False
        return {**case, "sources": sources, "docs": docs, "answer": answer}

    with ThreadPoolExecutor(max_workers=5) as pool:
        raw = list(pool.map(one, cases))
    mode = "llm" if state["llm_ok"] and all(r["answer"] is not None for r in raw) else "retrieval-only"
    for r in raw:
        if r["answer"] is None or mode == "retrieval-only":
            r["answer"] = extractive_answer(r["docs"]) if r["source"] else "I don't know based on the policies."
    metrics = ev.deterministic_metrics([{**r, "contexts": [x.page_content for x in r["docs"]]} for r in raw])
    if mode == "retrieval-only":
        metrics["refusal_rate_out_of_scope"] = None   # needs a model: the extractive fallback cannot refuse
    rows = [{"question": r["question"], "expected_source": r["source"], "sources": r["sources"],
             "hit": (r["source"] in r["sources"]) if r["source"]
             else (ev.REFUSAL in r["answer"].lower() if mode == "llm" else None)} for r in raw]
    payload = {
        "metrics": metrics,
        "gate": {"name": "source_hit_rate", "threshold": GATE, "passed": metrics["source_hit_rate"] >= GATE},
        "rows": rows, "mode": mode,
    }
    _EVAL_CACHE.update(at=time.time(), mode=mode_now, payload=payload)
    return {**payload, "latency_ms": int((time.perf_counter() - t0) * 1000)}


# --------------------------------------------------------------------------- Managed Agents
APP_TAG = "gen-ai-showcase"
CONSOLE = "https://platform.claude.com/workspaces/default/sessions/{}"
MANAGED_BETA = ["managed-agents-2026-04-01"]
MANAGED_BUDGET = {"type": "limit", "max_list_cost": {"amount": "100", "currency": "USD"}}   # amount is in cents
SEARCH_TOOL = "search_acme_policies"
OUT = "/mnt/session/outputs"

MANAGED_TASKS = {
    "handbook": {
        "title": "New-joiner handbook page",
        "description": "Researches leave, hybrid work, expenses and IT security, then writes an 8-question FAQ and a key-numbers CSV.",
        "prompt": ("Create a new-joiner handbook page. Research leave, hybrid work, expenses and IT security with "
                   f"{SEARCH_TOOL}, then write {OUT}/new_joiner_faq.md with 8 Q&As (each with a citation) and "
                   f"{OUT}/key_numbers.csv with columns topic,rule,value,source. Keep it concise."),
    },
    "expenses": {
        "title": "Expense policy one-pager",
        "description": "Summarises the expense policy into a short markdown page with limits and deadlines.",
        "prompt": (f"Using {SEARCH_TOOL}, summarise the expense policy (caps, deadlines, approvals) into "
                   f"{OUT}/expenses.md. One page, every figure cited like [expense_policy.md#3]."),
    },
    "security": {
        "title": "IT security checklist",
        "description": "Builds a checklist of password, device and incident rules a new starter must follow, with sources.",
        "prompt": (f"Using {SEARCH_TOOL}, write {OUT}/security_checklist.md: a checklist of the IT security rules a new "
                   f"starter must follow, each with a citation, plus {OUT}/security_rules.csv with columns rule,value,source."),
    },
}

_VERIFIED_SESSIONS: set[str] = set()
_ANSWERED_TOOL_USES: set[str] = set()
_ANSWER_LOCK = threading.Lock()


def managed_configured() -> bool:
    return all(os.getenv(k) for k in ("ANTHROPIC_API_KEY", "MANAGED_AGENT_ID", "MANAGED_ENVIRONMENT_ID"))


def _require_managed() -> None:
    if not managed_configured():
        raise ApiError(503, "not_configured", "Background agent is not configured on this deployment.")


def _client():
    import anthropic

    return anthropic.Anthropic(timeout=20, max_retries=1)


def _upstream(e: Exception) -> ApiError:
    import anthropic

    if isinstance(e, anthropic.NotFoundError):
        return ApiError(404, "unknown_session", "No such session.")
    log.warning("managed agents call failed: %s", type(e).__name__)
    return ApiError(502, "upstream_error", f"The Managed Agents API call failed ({type(e).__name__}).")


def _owned_session(client, session_id: str) -> None:
    """Only sessions this app created (tagged in metadata) are reachable, so the API cannot be used to peek at others."""
    if session_id in _VERIFIED_SESSIONS:
        return
    s = client.beta.sessions.retrieve(session_id)
    if (getattr(s, "metadata", None) or {}).get("app") != APP_TAG:
        raise ApiError(404, "unknown_session", "No such session.")
    _VERIFIED_SESSIONS.add(session_id)


@app.get("/api/managed/status", dependencies=[Depends(rl_light)])
def managed_status():
    out: dict[str, Any] = {"configured": managed_configured(),
                           "tasks": [{"id": k, "title": v["title"], "description": v["description"]}
                                     for k, v in MANAGED_TASKS.items()]}
    if not out["configured"]:
        out["hint"] = ("Set ANTHROPIC_API_KEY, MANAGED_AGENT_ID, MANAGED_AGENT_VERSION and MANAGED_ENVIRONMENT_ID "
                       "(create them with projects/07_claude_managed_agents/setup_agent.py).")
    return out


class StartIn(BaseModel):
    task_id: str


@app.post("/api/managed/start", dependencies=[Depends(rl_heavy)])
def managed_start(body: StartIn):
    _require_managed()
    task = MANAGED_TASKS.get(body.task_id)
    if task is None:
        raise ApiError(400, "unknown_task", "Unknown task_id.")
    agent: dict[str, Any] = {"type": "agent", "id": os.environ["MANAGED_AGENT_ID"]}
    if os.getenv("MANAGED_AGENT_VERSION"):
        agent["version"] = os.environ["MANAGED_AGENT_VERSION"]
    try:
        client = _client()
        session = client.beta.sessions.create(
            agent=agent, environment_id=os.environ["MANAGED_ENVIRONMENT_ID"], title=f"Showcase: {task['title']}",
            budget=MANAGED_BUDGET, metadata={"app": APP_TAG, "task": body.task_id})
        client.beta.sessions.events.send(
            session_id=session.id, events=[{"type": "user.message", "content": [{"type": "text", "text": task["prompt"]}]}])
    except ApiError:
        raise
    except Exception as e:  # noqa: BLE001
        raise _upstream(e) from e
    _VERIFIED_SESSIONS.add(session.id)
    return {"session_id": session.id, "console_url": CONSOLE.format(session.id)}


def _search_policies(query: str) -> str:
    rag = _rag()
    return rag.format_context(rag.get_retriever().invoke(CONTROL.sub("", query)[:MAX_INPUT] or "policy"))


def _answer_custom_tools(client, session_id: str, events: list) -> int:
    """Answer each agent.custom_tool_use exactly once. The event log itself is the source of truth (a
    user.custom_tool_result already present means answered); the lock + set cover the window before the
    send shows up in the log."""
    answered = {e.custom_tool_use_id for e in events if e.type == "user.custom_tool_result"}
    sent = 0
    for e in events:
        if e.type != "agent.custom_tool_use" or e.id in answered:
            continue
        with _ANSWER_LOCK:
            if e.id in _ANSWERED_TOOL_USES:
                continue
            _ANSWERED_TOOL_USES.add(e.id)
        try:
            if e.name == SEARCH_TOOL:
                payload = {"content": [{"type": "text", "text": _search_policies(str((e.input or {}).get("query", "")))}]}
            else:
                payload = {"content": [{"type": "text", "text": f"Unknown tool {e.name}"}], "is_error": True}
            client.beta.sessions.events.send(session_id=session_id, events=[
                {"type": "user.custom_tool_result", "custom_tool_use_id": e.id, **payload}])
            sent += 1
        except Exception as ex:  # noqa: BLE001
            with _ANSWER_LOCK:
                _ANSWERED_TOOL_USES.discard(e.id)   # allow a retry on the next poll
            log.warning("custom tool result failed: %s", type(ex).__name__)
    if len(_ANSWERED_TOOL_USES) > 5000:
        _ANSWERED_TOOL_USES.clear()
    return sent


def _feed_item(e: Any) -> dict | None:
    at = getattr(e, "processed_at", None)
    item: dict[str, Any] = {"id": e.id, "type": e.type, "at": at.isoformat() if at else None}
    if e.type == "agent.message":
        item["text"] = "".join(getattr(b, "text", "") for b in e.content)[:4000]
    elif e.type == "agent.tool_use":
        item["tool"] = e.name
        item["text"] = scrub(str((e.input or {}).get("command") or (e.input or {}).get("path") or ""), 200)
    elif e.type == "agent.custom_tool_use":
        item["tool"] = e.name
        item["text"] = scrub(str((e.input or {}).get("query", "")), 200)
    elif e.type == "session.status_idle":
        item["text"] = f"idle ({e.stop_reason.type})"
    elif e.type in ("session.status_running", "session.status_terminated"):
        item["text"] = e.type.removeprefix("session.status_")
    elif e.type == "session.error":
        item["text"] = "session error"
    else:
        return None
    return item


def _is_done(events: list) -> bool:
    for e in reversed(events):
        if e.type == "session.status_terminated":
            return True
        if e.type == "session.status_idle":
            return e.stop_reason.type != "requires_action"
        if e.type == "session.status_running":
            return False
    return False


def _list_events(client, session_id: str) -> list:
    out: list = []
    for e in client.beta.sessions.events.list(session_id, order="asc", limit=200):
        out.append(e)
        if len(out) >= 1000:
            break
    return out


def _list_files(client, session_id: str) -> list:
    return list(client.beta.files.list(scope_id=session_id, betas=MANAGED_BETA).data)


@app.get("/api/managed/events", dependencies=[Depends(rl_poll)])
def managed_events(session_id: str, after: str | None = None):
    _require_managed()
    clean_id(session_id, "session_id")
    if after:
        clean_id(after, "after")
    try:
        client = _client()
        _owned_session(client, session_id)
        events = _list_events(client, session_id)
        _answer_custom_tools(client, session_id, events)
        done = _is_done(events)
        files = [{"id": f.id, "name": f.filename, "size": f.size_bytes} for f in _list_files(client, session_id)] if done else []
    except ApiError:
        raise
    except Exception as e:  # noqa: BLE001
        raise _upstream(e) from e
    ids = [e.id for e in events]
    fresh = events[ids.index(after) + 1:] if after in ids else events
    feed = [i for i in map(_feed_item, fresh) if i]
    return {"events": feed, "cursor": ids[-1] if ids else (after or ""), "done": done, "files": files}


@app.get("/api/managed/file", dependencies=[Depends(rl_light)])
def managed_file(session_id: str, file_id: str):
    _require_managed()
    clean_id(session_id, "session_id")
    clean_id(file_id, "file_id")
    try:
        client = _client()
        _owned_session(client, session_id)
        meta = next((f for f in _list_files(client, session_id) if f.id == file_id), None)
        if meta is None:
            raise ApiError(404, "unknown_file", "No such file in this session.")
        data = client.beta.files.download(file_id, betas=MANAGED_BETA).read()
    except ApiError:
        raise
    except Exception as e:  # noqa: BLE001
        raise _upstream(e) from e
    name = re.sub(r"[^A-Za-z0-9._-]", "_", meta.filename or "output.txt")[:100]
    return Response(data, media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


app.mount("/static", StaticFiles(directory=STATIC, check_dir=False), name="static")
