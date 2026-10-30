"""Tests for the shared building blocks: provider registry, fallback chain, router, BM25 retriever.

    LLM_PROVIDER=fake pytest -q tests/test_core.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import providers as reg  # noqa: E402
from common import router  # noqa: E402
from common.llm import get_llm, get_llm_chain, structured  # noqa: E402

ALL_KEYS = {k for p in reg.PROVIDERS.values() for k in p.env_keys} | {
    "AI_GATEWAY_API_KEY", "VERCEL_OIDC_TOKEN", "LLM_MODEL"}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ALL_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "fake")


def test_no_keys_means_nothing_available():
    assert reg.available() == []
    assert all(not d["configured"] for d in reg.describe())


def test_available_follows_fallback_order(monkeypatch):
    for k in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "ZAI_API_KEY"):
        monkeypatch.setenv(k, "x")
    assert reg.available() == ["anthropic", "groq", "zai"]


def test_gateway_needs_token(monkeypatch):
    assert not reg.is_configured("gateway")
    monkeypatch.setenv("VERCEL_OIDC_TOKEN", "t")
    assert reg.is_configured("gateway")


def test_cloudflare_needs_both_vars(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "x")
    assert not reg.is_configured("cloudflare")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    assert reg.is_configured("cloudflare")


def test_describe_never_leaks_key_values(monkeypatch):
    for p in reg.PROVIDERS.values():
        for k in p.env_keys:
            monkeypatch.setenv(k, "sk-super-secret-value")
    monkeypatch.setenv("VERCEL_OIDC_TOKEN", "sk-super-secret-value")
    assert "sk-super-secret-value" not in json.dumps(reg.describe())


def test_model_override_env(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "my-model")
    assert reg.PROVIDERS["groq"].model_for("quick") == "my-model"


def test_claude_tiers_map_to_haiku_sonnet_opus(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert get_llm(provider="anthropic", tier="quick").model == "claude-haiku-5-5"
    assert get_llm(provider="anthropic", tier="default").model == "claude-sonnet-5-5"
    assert get_llm(provider="anthropic", tier="complex").model == "claude-opus-5-5"


def test_gateway_uses_gateway_base_url(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "k")
    llm = get_llm(provider="gateway", tier="default")
    assert llm.model == "anthropic/claude-sonnet-5-5"
    assert "ai-gateway.vercel.sh" in llm.anthropic_api_url


@pytest.mark.parametrize("name", [n for n, p in reg.PROVIDERS.items() if p.kind == "openai" and p.name != "cloudflare"])
def test_openai_compatible_providers_build(monkeypatch, name):
    for k in reg.PROVIDERS[name].env_keys:
        monkeypatch.setenv(k, "x")
    llm = get_llm(provider=name)
    assert llm.openai_api_base.rstrip("/") == reg.PROVIDERS[name].base_url.rstrip("/")


def test_unconfigured_provider_raises_clear_error():
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        get_llm(provider="groq")


def test_chain_has_fallbacks_and_prefers_requested(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "GROQ_API_KEY", "ZAI_API_KEY"):
        monkeypatch.setenv(k, "x")
    chain = get_llm_chain(prefer="zai")
    assert chain.runnable.metadata["provider"] == "zai"
    assert [f.metadata["provider"] for f in chain.fallbacks] == ["anthropic", "groq"]


def test_chain_falls_through_on_error():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.runnables import RunnableLambda

    def boom(_):
        raise RuntimeError("429 rate limited")

    chain = RunnableLambda(boom).with_fallbacks([FakeListChatModel(responses=["from fallback"])])
    assert chain.invoke("hi").content == "from fallback"


def test_structured_handles_fallback_chain(monkeypatch):
    from pydantic import BaseModel

    class R(BaseModel):
        a: int

    for k in ("ANTHROPIC_API_KEY", "GROQ_API_KEY"):
        monkeypatch.setenv(k, "x")
    wrapped = structured(get_llm_chain(), R)
    assert hasattr(wrapped, "fallbacks") and len(wrapped.fallbacks) == 1


# ------------------------------------------------------------------ router
@pytest.mark.parametrize("question,tier", [
    ("How many sick days do I get?", "quick"),
    ("hi", "quick"),
    ("Compare the leave and expense policies and tell me where they conflict", "complex"),
    ("Why is the hotel cap lower outside metro cities?", "complex"),
    ("Raise a ticket for my broken VPN", "default"),
    ("What is the password rotation period and also who do I report a lost laptop to?", "default"),
])
def test_router_tiers(question, tier):
    assert router.route(question).tier == tier


def test_router_picks_configured_provider_and_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    d = router.route("How many sick days do I get?")
    assert (d.provider, d.model) == ("anthropic", "claude-haiku-5-5")
    d2 = router.route("Compare leave and expense rules and flag any conflicts between them")
    assert d2.model == "claude-opus-5-5"
    assert d.est_cost_usd() < d2.est_cost_usd()


def test_router_without_providers_still_decides():
    d = router.route("What is the hotel cap?")
    assert d.tier in {"quick", "default", "complex"} and d.provider == ""


def test_router_pin_ignored_when_not_configured(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert router.route("hello there friend", prefer="gemini").provider == "groq"
    assert router.route("hello there friend", prefer="groq").provider == "groq"


# ------------------------------------------------------------------ BM25
def _rag():
    path = ROOT / "projects" / "02_rag_basics" / "rag_pipeline.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("rag_pipeline_core", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rag_pipeline_core"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("question,source", [
    ("What is the hotel cap per night in metro cities?", "expense_policy.md"),
    ("How many unused leave days can I carry forward?", "leave_policy.md"),
    ("What are the password rotation rules?", "it_security_policy.md"),
    ("What faithfulness score is needed before go-live?", "ai_usage_guidelines.md"),
    ("What is the referral bonus?", "benefits.md"),
])
def test_bm25_finds_the_right_document(question, source):
    rag = _rag()
    docs = rag.get_retriever(backend="bm25").invoke(question)
    assert source in [d.metadata["source"] for d in docs]


def test_bm25_defaults_to_four_results():
    # The "no heavy imports" guarantee is checked in a clean subprocess in tests/test_showcase.py,
    # because other tests in this process legitimately import chromadb/faiss.
    rag = _rag()
    assert rag.get_retriever(backend="bm25").k == 4
