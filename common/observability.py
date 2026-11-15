"""Langfuse tracing that switches itself on when keys are present.

    from common.observability import trace_config, flush
    chain.invoke(inputs, config=trace_config("rag-query", session_id="demo"))
    flush()   # short scripts must flush before exit - Langfuse sends asynchronously

With LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY unset, everything here is a no-op, so the
projects run the same with or without observability.
"""
from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def get_callbacks() -> list[Any]:
    if not enabled():
        return []
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]


def trace_config(run_name: str, session_id: str | None = None, user_id: str | None = None,
                 tags: list[str] | None = None, **extra: Any) -> dict[str, Any]:
    """RunnableConfig for LangChain/LangGraph .invoke(). Session/user/tags show up as
    filters in the Langfuse UI (via langfuse_* metadata keys read by the CallbackHandler)."""
    metadata: dict[str, Any] = {}
    if session_id:
        metadata["langfuse_session_id"] = session_id
    if user_id:
        metadata["langfuse_user_id"] = user_id
    if tags:
        metadata["langfuse_tags"] = tags
    return {"callbacks": get_callbacks(), "run_name": run_name, "metadata": metadata, **extra}


def flush() -> None:
    if enabled():
        from langfuse import get_client

        get_client().flush()
