"""Hosted demo API (FastAPI, deployed on Vercel). Contract: docs/SHOWCASE_API.md.

    LLM_PROVIDER=fake RETRIEVER=bm25 uvicorn showcase.server:app --port 8099

Design rules: never 500 because a model is missing or down (degrade to retrieval-only), never leak a
secret, never trust the client (length caps, allow-lists, per-IP rate limits), always return a trace.
"""
from __future__ import annotations

import os

os.environ.setdefault("RETRIEVER", "bm25")  # must be set before rag_pipeline is imported

import base64  # noqa: E402
import contextvars  # noqa: E402
import hmac  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from collections import OrderedDict, deque  # noqa: E402
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
from showcase.tracing import Tracer, safe_error, scrub  # noqa: E402

log = logging.getLogger("showcase")
STATIC = Path(__file__).parent / "static"
MAX_INPUT = 500
RECURSION_LIMIT = 8
MAX_HISTORY = 8                  # messages sent to the model per agent call
MAX_TURNS = 10                   # user turns per agent thread
MAX_CHAIN = 2                    # providers in a fallback chain: 2 x 10 s timeout stays under the 45 s deadline
REQUEST_DEADLINE_S = 45
MAX_FILE_BYTES = 4_000_000
LLM_KWARGS = {"max_tokens": 800, "timeout": 10, "max_retries": 0}   # demo-sized, fail fast so the chain can fall through
TIER_ORDER = {"quick": 0, "default": 1, "complex": 2}
# control chars, plus bidi overrides / zero-width characters (U+200B-200F, U+202A-202E, U+2066-2069)
CONTROL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2066-\u2069]")
ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,100}$")
THREAD_RE = re.compile(r"^[0-9a-f]{32}$")
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
    # The OIDC header is only ever held in a per-request ContextVar, and only after its claims check out.
    # It is never written to os.environ (one request would then authenticate every other request).
    reset = reg.request_oidc.set(verified_oidc(request.headers.get("x-vercel-oidc-token")))
    try:
        response = await call_next(request)
    finally:
        reg.request_oidc.reset(reset)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response


def verified_oidc(token: str | None) -> str | None:
    """The token if it looks like this project's Vercel OIDC token, else None. Only honoured on Vercel.

    Claims are checked (iss, project_id, exp) but the signature is not: the AI Gateway verifies that when the token
    is used, and a forged token only ever authenticates the forger's own request."""
    if not token or os.getenv("VERCEL") != "1" or len(token) > 8192:
        return None
    project = os.getenv("VERCEL_PROJECT_ID", "")
    try:
        parts = token.split(".")
        if len(parts) != 3 or not project:
            return None
        claims = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
        if not isinstance(claims, dict) or not str(claims.get("iss", "")).startswith("https://oidc.vercel.com/"):
            return None
        if claims.get("project_id") != project:
            return None
        exp = claims.get("exp")
        if exp is not None and (not isinstance(exp, (int, float)) or exp < time.time()):
            return None
    except Exception:  # noqa: BLE001
        return None
    return token


# --------------------------------------------------------------------------- access control and global budget
def is_unlocked(request: Request) -> bool:
    """Presenter access: x-demo-unlock must equal env DEMO_UNLOCK. Unset DEMO_UNLOCK means nobody is unlocked."""
    code = os.getenv("DEMO_UNLOCK", "")
    given = request.headers.get("x-demo-unlock", "")
    return bool(code) and bool(given) and hmac.compare_digest(given.encode(), code.encode())


def require_unlocked(request: Request) -> bool:
    if not is_unlocked(request):
        raise ApiError(403, "locked", "This feature is locked on the public demo. Enter the presenter code to use it.")
    return True


def max_tier_public() -> str:
    t = os.getenv("MAX_TIER", "default").strip().lower()
    return t if t in TIER_ORDER else "default"


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


class HourlyCounter:
    """Sliding one-hour window of events, per instance."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stamps: deque[float] = deque()

    def reset(self) -> None:
        with self._lock:
            self._stamps.clear()

    def _prune(self, now: float) -> None:
        while self._stamps and now - self._stamps[0] > 3600:
            self._stamps.popleft()

    def take(self, limit: int) -> bool:
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            if len(self._stamps) >= limit:
                return False
            self._stamps.append(now)
            return True

    def refund(self) -> None:
        with self._lock:
            if self._stamps:
                self._stamps.pop()


LLM_BUDGET = HourlyCounter()       # public LLM-backed requests
MANAGED_BUDGET_COUNTER = HourlyCounter()   # managed-agent sessions started


def llm_gate(unlocked: bool) -> tuple[bool, str]:
    """May this request call a model? Returns (ok, notice-if-not). Consumes one unit of the hourly budget."""
    if not llm_available():
        return False, NO_MODEL_NOTICE
    if unlocked:
        return True, ""
    if not LLM_BUDGET.take(_int_env("DEMO_LLM_CALLS_PER_HOUR", 300)):
        return False, ("The hourly demo budget reached its limit, so this is the retrieval-only fallback. "
                       "Model-backed answers return within the hour.")
    return True, ""


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
    """False when the operator kill switch (DEMO_LLM_ENABLED=0) is on or no provider is configured."""
    if os.getenv("DEMO_LLM_ENABLED", "1").strip().lower() in ("0", "false", "off", "no"):
        return False
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


def route_dict(d: Decision, pinned: bool, failed: bool = False) -> dict:
    cost = 0.0 if _is_free(d.provider) else d.est_cost_usd()
    out = {"tier": d.tier, "provider": d.provider, "model": d.model, "reason": d.reason,
           "est_cost_usd": round(cost, 6), "pinned": pinned}
    if failed:
        out["failed"] = True
    return out


def decide(question: str, provider: str, min_tier: str = "quick", unlocked: bool = False) -> tuple[Decision, bool]:
    """provider 'auto' -> the router picks the tier; a pinned provider skips routing. Never raises on no providers.
    Public requests are clamped to MAX_TIER and use the public provider order (free first, Claude last)."""
    if provider != "auto":
        name = valid_provider(provider)
        # Public visitors may pin free providers or the capped gateway, never a directly billed paid key.
        if not unlocked and name not in ("fake", "gateway") and not reg.PROVIDERS[name].free:
            raise ApiError(403, "locked", f"Pinning {name!r} is a presenter feature. Use Auto-route or a free provider.")
        if name != "fake" and not reg.is_configured(name):
            raise ApiError(400, "provider_not_configured", f"Provider {name!r} has no API key on this server.")
        model = "fake-chat" if name == "fake" else reg.PROVIDERS[name].model_for("default")
        return Decision(tier="default", reason="provider pinned by caller (routing skipped)", provider=name, model=model), True
    d = route(question)
    names = reg.order_for(unlocked)
    d.provider = names[0] if names else ""
    if TIER_ORDER[d.tier] < TIER_ORDER[min_tier]:
        d.reason += f" (raised to {min_tier}: tool calling must be reliable)"
        d.tier = min_tier
    cap = max_tier_public()
    if not unlocked and TIER_ORDER[d.tier] > TIER_ORDER[cap]:
        d.reason += f" (capped at {cap} on the public demo)"
        d.tier = cap
    d.model = reg.PROVIDERS[d.provider].model_for(d.tier) if d.provider else ""
    if not d.provider and fake_mode():
        d.provider, d.model = "fake", "fake-chat"
    return d, False


def build_model(d: Decision, pinned: bool, unlocked: bool = False):
    """The pinned provider alone, or a short fallback chain (public order unless unlocked)."""
    if pinned:
        return get_llm(provider=d.provider, tier=d.tier, **LLM_KWARGS)
    return get_llm_chain(tier=d.tier, names=reg.order_for(unlocked), max_providers=MAX_CHAIN, **LLM_KWARGS)


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
def health(request: Request):
    return {"status": "ok", "retriever": os.getenv("RETRIEVER", "bm25"),
            "mode": "llm" if llm_available() else "retrieval-only", "providers": active_providers(),
            "access": {"unlock_configured": bool(os.getenv("DEMO_UNLOCK")), "unlocked": is_unlocked(request),
                       "max_tier_public": max_tier_public()}}


class UnlockIn(BaseModel):
    code: str


@app.post("/api/unlock", dependencies=[Depends(rl_heavy)])
def unlock(body: UnlockIn):
    code = os.getenv("DEMO_UNLOCK", "")
    if not code or len(body.code) > 200 or not hmac.compare_digest(body.code.encode(), code.encode()):
        raise ApiError(403, "locked", "Wrong presenter code.")
    return {"unlocked": True}


@app.get("/api/providers", dependencies=[Depends(rl_light)])
def providers():
    return {"providers": reg.describe(), "available": active_providers()}


# --------------------------------------------------------------------------- /api/ask
class AskIn(BaseModel):
    question: str
    provider: str = "auto"


def run_rag(question: str, d: Decision, pinned: bool, tracer: Tracer, unlocked: bool = False,
            allow_llm: bool = True, notice: str = NO_MODEL_NOTICE, deadline: float | None = None) -> dict:
    """Retrieve, then generate with the chosen model; degrade to extractive answers on any model failure."""
    rag = _rag()
    docs = rag.get_retriever().invoke(question, config={"callbacks": [tracer]})
    out: dict[str, Any] = {"contexts": _contexts(docs), "sources": sorted({x.metadata["source"] for x in docs})}
    if deadline is not None and time.monotonic() >= deadline:
        allow_llm, notice = False, "This request ran out of time before a model could be called, so this is the retrieval-only fallback."
    if allow_llm and (d.provider or fake_mode()):
        try:
            from langchain_core.output_parsers import StrOutputParser

            chain = rag.RAG_PROMPT | build_model(d, pinned, unlocked) | StrOutputParser()
            out["answer"] = chain.invoke({"context": rag.format_context(docs), "question": question},
                                         config=_cfg(tracer, "showcase-ask"))
            out["mode"] = "llm"
            return out
        except Exception as e:  # noqa: BLE001  quota, outage, bad key... never a 500
            log.warning("model call failed: %s", type(e).__name__)
            out["notice"] = NO_MODEL_NOTICE + f" (last error: {safe_error(e)})"
            out["failed"] = True
    else:
        out["notice"] = notice
    out["answer"], out["mode"] = extractive_answer(docs), "retrieval-only"
    return out


@app.post("/api/ask", dependencies=[Depends(rl_heavy)])
def ask(body: AskIn, request: Request):
    t0 = time.perf_counter()
    deadline = time.monotonic() + REQUEST_DEADLINE_S
    unlocked = is_unlocked(request)
    question = clean_text(body.question, "question")
    tracer = Tracer()
    with tracer.span("route", "route") as sp:
        d, pinned = decide(question, body.provider, unlocked=unlocked)
        sp["detail"] = f"tier={d.tier} provider={d.provider or 'none'} model={d.model or 'none'}"
    ok, why = llm_gate(unlocked)
    result = run_rag(question, d, pinned, tracer, unlocked, ok, why, deadline)
    flush()
    result["route"] = route_dict(d, pinned, result.pop("failed", False))
    result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    result["trace"] = tracer.snapshot()
    return result


# --------------------------------------------------------------------------- /api/compare
class CompareIn(BaseModel):
    question: str
    providers: list[str]


@app.post("/api/compare", dependencies=[Depends(rl_heavy)])
def compare(body: CompareIn, request: Request):
    require_unlocked(request)
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
                res["answer"], res["error"] = "", "not configured on this server (no API key)"
                res["latency_ms"] = int((time.perf_counter() - t) * 1000)
                return res
            msg = (rag.RAG_PROMPT | get_llm(provider=name, tier="default", **LLM_KWARGS)).invoke(prompt_in)
            res["answer"] = _text(msg.content)
            usage = getattr(msg, "usage_metadata", None) or {}
            if usage.get("output_tokens") is not None:
                res["tokens_out"] = usage["output_tokens"]
        except Exception as e:  # noqa: BLE001
            res["answer"] = ""
            res["error"] = safe_error(e)   # type + HTTP status only: provider messages can echo secrets
        res["latency_ms"] = int((time.perf_counter() - t) * 1000)
        return res

    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        results = list(pool.map(lambda n: contextvars.copy_context().run(one, n), names))
    return {"results": results}


# --------------------------------------------------------------------------- /api/chat + /api/approve
_CHECKPOINTER = None
_THREADS: "OrderedDict[str, dict]" = OrderedDict()   # thread_id -> {"pending": waiting for approval?, "turns": int}
_THREADS_LOCK = threading.Lock()
MAX_THREADS = 200
_THREAD_LOCKS: dict[str, threading.Lock] = {}


def checkpointer():
    global _CHECKPOINTER
    if _CHECKPOINTER is None:
        from langgraph.checkpoint.memory import InMemorySaver

        _CHECKPOINTER = InMemorySaver()
    return _CHECKPOINTER


def _drop_checkpoint(thread_id: str) -> None:
    try:
        checkpointer().delete_thread(thread_id)
    except Exception:  # noqa: BLE001
        pass


def _remember(thread_id: str, pending: bool, turn: bool = False) -> None:
    with _THREADS_LOCK:
        st = _THREADS.setdefault(thread_id, {"pending": False, "turns": 0})
        st["pending"] = pending
        st["turns"] += 1 if turn else 0
        _THREADS.move_to_end(thread_id)
        while len(_THREADS) > MAX_THREADS:
            old, _ = _THREADS.popitem(last=False)
            _THREAD_LOCKS.pop(old, None)
            _drop_checkpoint(old)


def _forget(thread_id: str) -> None:
    """Delete a thread's checkpoint (after an error it could be left with a dangling tool call)."""
    with _THREADS_LOCK:
        _THREADS.pop(thread_id, None)
    _drop_checkpoint(thread_id)


def _thread_state(thread_id: str) -> dict:
    with _THREADS_LOCK:
        return dict(_THREADS.get(thread_id) or {"pending": False, "turns": 0})


def _thread_lock(thread_id: str) -> threading.Lock:
    with _THREADS_LOCK:
        if len(_THREAD_LOCKS) > MAX_THREADS * 4:   # locks of forgotten threads: keep the dict bounded
            for k in [k for k, lk in _THREAD_LOCKS.items() if k not in _THREADS and not lk.locked()]:
                del _THREAD_LOCKS[k]
        return _THREAD_LOCKS.setdefault(thread_id, threading.Lock())


def _check_chat_thread(thread_id: str) -> None:
    st = _thread_state(thread_id)
    if st["pending"]:
        raise ApiError(409, "approval_pending", "This conversation is waiting for you to approve or reject an action.")
    if st["turns"] >= MAX_TURNS:
        raise ApiError(409, "thread_full", f"This conversation reached its {MAX_TURNS}-message limit. Start a new chat.")


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


def agent_turn(payload: Any, thread_id: str, d: Decision, tracer: Tracer, label: str, unlocked: bool = False,
               new_turn: bool = True, deadline: float | None = None) -> dict:
    """Run one graph invocation. Returns the shared chat/approve response (without route/trace)."""
    from langgraph.errors import GraphRecursionError

    with _thread_lock(thread_id):
        if new_turn:   # re-check under the lock: a concurrent request may have left an approval pending
            _check_chat_thread(thread_id)
        elif not _thread_state(thread_id)["pending"]:
            raise ApiError(400, "nothing_pending", "This conversation has no action waiting for approval.")
        try:
            from graph import DEADLINE_MESSAGE, build_graph

            graph = build_graph(checkpointer=checkpointer(), llm=build_model(d, False, unlocked),
                                max_history=MAX_HISTORY, deadline=deadline)
            config = {**_cfg(tracer, label, session_id=thread_id),
                      "configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
            result = graph.invoke(payload, config)
        except GraphRecursionError:
            _forget(thread_id)
            return {"status": "done", "thread_id": thread_id, "steps": [], "mode": "llm",
                    "reply": "I stopped because this took too many steps. Try a simpler request."}
        except Exception as e:  # noqa: BLE001
            log.warning("agent turn failed: %s", type(e).__name__)
            _forget(thread_id)
            return degraded_chat(thread_id, f" (last error: {safe_error(e)})", failed=True)
        interrupts = result.get("__interrupt__")
        if interrupts:
            _remember(thread_id, True, turn=new_turn)
            return {"status": "pending_approval", "thread_id": thread_id, "mode": "llm",
                    "approval_request": dict(interrupts[0].value), "steps": _steps(result["messages"])}
        _remember(thread_id, False, turn=new_turn)
        reply = _text(result["messages"][-1].content)
        resp = {"status": "done", "thread_id": thread_id, "mode": "llm", "reply": reply,
                "steps": _steps(result["messages"])}
        if reply == DEADLINE_MESSAGE:
            resp.update(mode="retrieval-only", notice="This request ran out of its time budget before the model could answer.")
        return resp


def degraded_chat(thread_id: str, extra: str = "", failed: bool = False, notice: str = NO_MODEL_NOTICE) -> dict:
    return {"status": "done", "thread_id": thread_id, "steps": [], "mode": "retrieval-only",
            "reply": "The agent needs a language model to decide which tools to call, and none is available right now. "
                     "The Ask section still works in retrieval-only mode.",
            "notice": notice + extra, **({"_failed": True} if failed else {})}


def _finish_chat(resp: dict, d: Decision, tracer: Tracer) -> dict:
    flush()
    resp["route"] = route_dict(d, False, resp.pop("_failed", False))
    resp["trace"] = tracer.snapshot()
    return resp


@app.post("/api/chat", dependencies=[Depends(rl_heavy)])
def chat(body: ChatIn, request: Request):
    deadline = time.monotonic() + REQUEST_DEADLINE_S
    unlocked = is_unlocked(request)
    message = clean_text(body.message, "message")
    if body.thread_id:
        if not THREAD_RE.match(body.thread_id):
            raise ApiError(400, "bad_request", "`thread_id` is not a valid id.")
        thread_id = body.thread_id
        _check_chat_thread(thread_id)
    else:
        thread_id = uuid.uuid4().hex
    tracer = Tracer()
    with tracer.span("route", "route") as sp:
        d, _ = decide(message, "auto", min_tier="default", unlocked=unlocked)
        sp["detail"] = f"tier={d.tier} provider={d.provider or 'none'} model={d.model or 'none'}"
    ok, why = llm_gate(unlocked)
    if not ok:
        _remember(thread_id, False)
        return _finish_chat(degraded_chat(thread_id, notice=why), d, tracer)
    resp = agent_turn({"messages": [{"role": "user", "content": message}]}, thread_id, d, tracer, "showcase-chat",
                      unlocked, deadline=deadline)
    return _finish_chat(resp, d, tracer)


@app.post("/api/approve", dependencies=[Depends(rl_heavy)])
def approve(body: ApproveIn, request: Request):
    from langgraph.types import Command

    deadline = time.monotonic() + REQUEST_DEADLINE_S
    unlocked = is_unlocked(request)
    thread_id = body.thread_id
    known = bool(THREAD_RE.match(thread_id)) and \
        checkpointer().get_tuple({"configurable": {"thread_id": thread_id}}) is not None
    if not known:   # also what a restarted instance answers: the UI re-sends the original message
        raise ApiError(404, "unknown_thread", "No such conversation on this server instance. Start a new chat.")
    if not _thread_state(thread_id)["pending"]:
        raise ApiError(400, "nothing_pending", "This conversation has no action waiting for approval.")
    tracer = Tracer()
    with tracer.span("route", "route") as sp:
        d, _ = decide("resume approved action", "auto", min_tier="default", unlocked=unlocked)
        d.reason = "resuming a paused agent turn at the default tier"
        sp["detail"] = f"tier={d.tier} provider={d.provider or 'none'} model={d.model or 'none'}"
    ok, why = llm_gate(unlocked)
    if not ok:
        return _finish_chat(degraded_chat(thread_id, notice=why), d, tracer)
    resp = agent_turn(Command(resume={"approved": body.approved}), thread_id, d, tracer, "showcase-approve",
                      unlocked, new_turn=False, deadline=deadline)
    return _finish_chat(resp, d, tracer)


# --------------------------------------------------------------------------- /api/evals
_EVAL_CACHE: dict[str, Any] = {"at": 0.0, "mode": None, "payload": None}
_EVAL_LOCK = threading.Lock()
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
    """Golden-set RAG eval. Retrieval-only unless EVAL_USE_LLM=1, cached 10 min, computed once under a lock, so a
    public URL cannot burn model budget."""
    t0 = time.perf_counter()
    use_llm = os.getenv("EVAL_USE_LLM") == "1" and llm_available()
    mode_now = "llm" if use_llm else "retrieval-only"
    with _EVAL_LOCK:
        if _EVAL_CACHE["payload"] and _EVAL_CACHE["mode"] == mode_now and time.time() - _EVAL_CACHE["at"] < EVAL_TTL:
            return {**_EVAL_CACHE["payload"], "cached": True, "latency_ms": int((time.perf_counter() - t0) * 1000)}
        payload = _compute_evals(use_llm)
        _EVAL_CACHE.update(at=time.time(), mode=mode_now, payload=payload)
    return {**payload, "latency_ms": int((time.perf_counter() - t0) * 1000)}


def _compute_evals(use_llm: bool) -> dict:
    ev = _load_eval_module()
    rag = _rag()
    cases = ev.DATASET["rag"]
    d, _ = decide("short lookup", "auto") if use_llm else (None, False)
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
        raw = list(pool.map(lambda c: contextvars.copy_context().run(one, c), cases))
    mode = "llm" if state["llm_ok"] and all(r["answer"] is not None for r in raw) else "retrieval-only"
    for r in raw:
        if r["answer"] is None or mode == "retrieval-only":
            r["answer"] = extractive_answer(r["docs"]) if r["source"] else "I don't know based on the policies."
    metrics = ev.deterministic_metrics([{**r, "contexts": [x.page_content for x in r["docs"]]} for r in raw])
    if mode == "retrieval-only":
        # The extractive fallback cannot refuse and always cites, so these two would be vacuous: report null.
        metrics["refusal_rate_out_of_scope"] = None
        metrics["citation_rate"] = None

    def rank_of(r: dict) -> int | None:
        """1-based position of the expected source in the retriever's ranked results (None if absent)."""
        order = [x.metadata["source"] for x in r["docs"]]
        return order.index(r["source"]) + 1 if r["source"] in order else None

    answerable = [r for r in raw if r["source"]]
    ranks = [rank_of(r) for r in answerable]
    # Hit-in-top-4 is easy on an 8-document corpus (about half the corpus is returned), so the gate uses
    # rank-aware metrics: top-1 rate and mean reciprocal rank (MRR).
    metrics["top1_rate"] = sum(k == 1 for k in ranks) / max(len(ranks), 1)
    metrics["mrr"] = sum(1 / k for k in ranks if k) / max(len(ranks), 1)
    rows = [{"question": r["question"], "expected_source": r["source"], "sources": r["sources"],
             "rank": rank_of(r) if r["source"] else None,
             "hit": (r["source"] in r["sources"]) if r["source"]
             else (ev.REFUSAL in r["answer"].lower() if mode == "llm" else None)} for r in raw]
    payload = {
        "metrics": metrics,
        "gate": {"name": "mrr", "threshold": GATE, "passed": metrics["mrr"] >= GATE},
        "rows": rows, "mode": mode,
    }
    return payload


# --------------------------------------------------------------------------- Managed Agents
APP_TAG = "gen-ai-showcase"
CONSOLE = "https://platform.claude.com/workspaces/default/sessions/{}"
MANAGED_BETA = ["managed-agents-2026-04-01"]
MANAGED_BUDGET = {"type": "limit", "max_list_cost": {"amount": "25", "currency": "USD"}}   # amount is in cents: $0.25 per session
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
def managed_start(body: StartIn, request: Request):
    require_unlocked(request)
    _require_managed()
    task = MANAGED_TASKS.get(body.task_id)
    if task is None:
        raise ApiError(400, "unknown_task", "Unknown task_id.")
    agent: dict[str, Any] = {"type": "agent", "id": os.environ["MANAGED_AGENT_ID"]}
    if os.getenv("MANAGED_AGENT_VERSION"):
        agent["version"] = os.environ["MANAGED_AGENT_VERSION"]
    if not MANAGED_BUDGET_COUNTER.take(_int_env("MANAGED_STARTS_PER_HOUR", 3)):
        raise ApiError(429, "managed_cap", "The hourly limit for background agent sessions on this instance is reached. "
                                           "Try again later.", {"Retry-After": "600"})
    try:
        client = _client()
        session = client.beta.sessions.create(
            agent=agent, environment_id=os.environ["MANAGED_ENVIRONMENT_ID"], title=f"Showcase: {task['title']}",
            budget=MANAGED_BUDGET, metadata={"app": APP_TAG, "task": body.task_id})
        client.beta.sessions.events.send(
            session_id=session.id, events=[{"type": "user.message", "content": [{"type": "text", "text": task["prompt"]}]}])
    except ApiError:
        MANAGED_BUDGET_COUNTER.refund()
        raise
    except Exception as e:  # noqa: BLE001
        MANAGED_BUDGET_COUNTER.refund()   # nothing was billed for a session that never started
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


def _too_large() -> ApiError:
    return ApiError(413, "too_large", "That file is too large to download through this demo. Open the session in the "
                                      "Claude console to get it.")


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
        if (meta.size_bytes or 0) > MAX_FILE_BYTES:
            raise _too_large()
        data = client.beta.files.download(file_id, betas=MANAGED_BETA).read()
    except ApiError:
        raise
    except Exception as e:  # noqa: BLE001
        raise _upstream(e) from e
    if len(data) > MAX_FILE_BYTES:
        raise _too_large()
    name = re.sub(r"[^A-Za-z0-9._-]", "_", meta.filename or "output.txt")[:100]
    return Response(data, media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


app.mount("/static", StaticFiles(directory=STATIC, check_dir=False), name="static")
