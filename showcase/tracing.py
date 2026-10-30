"""Span capture for the waterfall UI.

A LangChain callback handler turns chain / retriever / LLM / tool runs into flat spans with parent ids and
millisecond offsets (the `trace` shape in docs/SHOWCASE_API.md). Manual spans (route, fallback) can be added
alongside. Langfuse stays optional and separate (common.observability); this runs with no keys at all.
"""
from __future__ import annotations

import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

MAX_SPANS = 80
_CONTROL_FLOW = {"GraphInterrupt", "GraphBubbleUp", "NodeInterrupt", "ParentCommand"}
_SECRET_ENV_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def scrub(text: Any, limit: int = 200) -> str:
    """Make text safe for a public response: redact the value of any secret-looking env var, trim length."""
    out = str(text)
    for name, value in os.environ.items():
        if value and len(value) >= 8 and any(h in name.upper() for h in _SECRET_ENV_HINTS):
            out = out.replace(value, "[redacted]")
    out = re.sub(r"\b(sk|key|gsk|AIza|hf|ghp)[-_][A-Za-z0-9_\-]{12,}", "[redacted]", out)
    return out[:limit]


class Tracer(BaseCallbackHandler):
    """Collects spans for one request. Create one per request and pass it in `config["callbacks"]`."""

    raise_error = False
    run_inline = True

    def __init__(self) -> None:
        self.trace_id = uuid.uuid4().hex[:12]
        self._t0 = time.perf_counter()
        self._lock = threading.RLock()
        self._spans: list[dict] = []
        self._by_run: dict[Any, str] = {}      # LangChain run_id -> span id
        self._parent_of: dict[Any, Any] = {}   # run_id -> parent run_id (also for runs we do not record)
        self._started: dict[str, float] = {}   # span id -> perf_counter at start

    # ------------------------------------------------------------------ core
    def _now_ms(self) -> int:
        return int((time.perf_counter() - self._t0) * 1000)

    def _resolve_parent(self, parent_run_id: Any) -> str | None:
        p = parent_run_id
        while p is not None:
            if p in self._by_run:
                return self._by_run[p]
            p = self._parent_of.get(p)
        return None

    def _open(self, run_id: Any, parent_run_id: Any, name: str, kind: str, detail: str = "") -> str | None:
        with self._lock:
            self._parent_of[run_id] = parent_run_id
            if len(self._spans) >= MAX_SPANS:
                return None
            sid = f"s{len(self._spans) + 1}"
            self._spans.append({
                "id": sid, "parent_id": self._resolve_parent(parent_run_id), "name": name, "kind": kind,
                "start_ms": self._now_ms(), "duration_ms": None, "status": "ok",
                "tokens_in": None, "tokens_out": None, "detail": detail,
            })
            self._by_run[run_id] = sid
            self._started[sid] = time.perf_counter()
            return sid

    def _span(self, run_id: Any) -> dict | None:
        sid = self._by_run.get(run_id)
        return next((s for s in self._spans if s["id"] == sid), None) if sid else None

    def _close(self, run_id: Any, error: BaseException | None = None, detail: str | None = None,
               tokens: tuple[int | None, int | None] | None = None) -> dict | None:
        with self._lock:
            span = self._span(run_id)
            if span is None:
                return None
            span["duration_ms"] = int((time.perf_counter() - self._started[span["id"]]) * 1000)
            if error is not None and type(error).__name__ in _CONTROL_FLOW:
                span["detail"] = (span["detail"] + " | paused for human approval").strip(" |")
            elif error is not None:
                span["status"] = "error"
                span["detail"] = f"{type(error).__name__}: {scrub(error, 120)}"
            if detail:
                span["detail"] = (span["detail"] + " | " + detail).strip(" |")
            if tokens:
                span["tokens_in"], span["tokens_out"] = tokens
            return span

    @staticmethod
    def _name(serialized: dict | None, kwargs: dict, default: str) -> str:
        if kwargs.get("name"):
            return str(kwargs["name"])
        if serialized:
            if serialized.get("name"):
                return str(serialized["name"])
            ident = serialized.get("id")
            if isinstance(ident, list) and ident:
                return str(ident[-1])
        return default

    # ------------------------------------------------------------------ chains
    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, **kwargs):
        if tags and "langsmith:hidden" in tags:
            self._parent_of[run_id] = parent_run_id  # keep the ancestry, hide the span
            return
        self._open(run_id, parent_run_id, self._name(serialized, kwargs, "chain"), "chain")

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self._close(run_id)

    def on_chain_error(self, error, *, run_id, **kwargs):
        self._close(run_id, error)

    # ------------------------------------------------------------------ retriever
    def on_retriever_start(self, serialized, query, *, run_id, parent_run_id=None, **kwargs):
        self._open(run_id, parent_run_id, self._name(serialized, kwargs, "retriever"), "retriever",
                   f"query={scrub(query, 80)!r}")

    def on_retriever_end(self, documents, *, run_id, **kwargs):
        self._close(run_id, detail=f"{len(documents)} chunks")

    def on_retriever_error(self, error, *, run_id, **kwargs):
        self._close(run_id, error)

    # ------------------------------------------------------------------ llm
    def _llm_start(self, serialized, run_id, parent_run_id, kwargs):
        meta = kwargs.get("metadata") or {}
        provider = meta.get("provider") or meta.get("ls_provider") or ""
        model = meta.get("ls_model_name") or (kwargs.get("invocation_params") or {}).get("model") or ""
        name = f"llm:{provider}" if provider else self._name(serialized, kwargs, "llm")
        self._open(run_id, parent_run_id, name, "llm", f"model={model}" if model else "")

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        self._llm_start(serialized, run_id, parent_run_id, kwargs)

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
        self._llm_start(serialized, run_id, parent_run_id, kwargs)

    def on_llm_end(self, response, *, run_id, **kwargs):
        tin = tout = None
        try:
            for gen in (response.generations or [[]])[0]:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None)
                if usage:
                    tin, tout = usage.get("input_tokens"), usage.get("output_tokens")
                    break
            if tin is None and getattr(response, "llm_output", None):
                u = response.llm_output.get("token_usage") or response.llm_output.get("usage") or {}
                tin = u.get("prompt_tokens", u.get("input_tokens"))
                tout = u.get("completion_tokens", u.get("output_tokens"))
        except Exception:  # noqa: BLE001  tracing must never break a request
            pass
        self._close(run_id, tokens=(tin, tout))

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        span = self._close(run_id, error)
        if span is None:
            return
        who = span["name"].removeprefix("llm:") or "provider"
        self.add_span("fallback", "fallback", f"{who} failed ({type(error).__name__}); chain moves to the next provider",
                      parent_id=span["parent_id"], status="error")

    # ------------------------------------------------------------------ tools
    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
        self._open(run_id, parent_run_id, self._name(serialized, kwargs, "tool"), "tool", scrub(input_str, 100))

    def on_tool_end(self, output, *, run_id, **kwargs):
        self._close(run_id)

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._close(run_id, error)

    # ------------------------------------------------------------------ manual spans
    def add_span(self, name: str, kind: str, detail: str = "", duration_ms: int = 0, parent_id: str | None = None,
                 status: str = "ok", start_ms: int | None = None) -> str | None:
        with self._lock:
            if len(self._spans) >= MAX_SPANS:
                return None
            sid = f"s{len(self._spans) + 1}"
            self._spans.append({
                "id": sid, "parent_id": parent_id, "name": name, "kind": kind,
                "start_ms": self._now_ms() if start_ms is None else start_ms, "duration_ms": duration_ms,
                "status": status, "tokens_in": None, "tokens_out": None, "detail": detail,
            })
            return sid

    @contextmanager
    def span(self, name: str, kind: str, detail: str = ""):
        """Time a block as a manual span, e.g. the routing decision."""
        start = self._now_ms()
        t = time.perf_counter()
        holder: dict[str, str] = {"detail": detail}
        status = "ok"
        try:
            yield holder
        except BaseException:
            status = "error"
            raise
        finally:
            self.add_span(name, kind, holder["detail"], int((time.perf_counter() - t) * 1000), status=status,
                          start_ms=start)

    def snapshot(self) -> dict:
        """The `trace` object for the response. Spans are sorted by start time; open spans end now."""
        with self._lock:
            now = time.perf_counter()
            spans = []
            for s in self._spans:
                s = dict(s)
                if s["duration_ms"] is None:
                    s["duration_ms"] = int((now - self._started.get(s["id"], now)) * 1000)
                spans.append(s)
        spans.sort(key=lambda s: (s["start_ms"], int(s["id"][1:])))
        return {"trace_id": self.trace_id, "total_ms": self._now_ms(), "spans": spans}
