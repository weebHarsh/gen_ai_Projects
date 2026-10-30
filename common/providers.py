"""Provider registry: Claude (native + via Vercel AI Gateway) and many free/cheap OpenAI-compatible APIs.

Every entry says which env var(s) enable it, so a provider simply doesn't exist until its key is
present. Model ids are best-known defaults and each can be overridden with <NAME>_MODEL
(e.g. GROQ_MODEL=...); after you add keys, `python -m common.providers --discover` calls each
provider's GET /models so you can pick current ids instead of trusting this table.

Tiers (used by common/router.py):  quick -> cheap/fast,  default -> balanced,  complex -> most capable.
"""
from __future__ import annotations

import os
import re
import sys
from contextvars import ContextVar
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

CLAUDE_TIERS = {"quick": "claude-haiku-5-5", "default": "claude-sonnet-5-5", "complex": "claude-opus-5-5"}


# If a gateway slug is not in the catalog, the gateway itself retries these documented slugs (no extra round trip from us).
GATEWAY_FALLBACK_MODELS = {"quick": ["anthropic/claude-sonnet-5"], "default": ["anthropic/claude-sonnet-5"],
                           "complex": ["anthropic/claude-opus-5"]}


@dataclass(frozen=True)
class Provider:
    name: str
    label: str
    env_keys: tuple[str, ...]            # all must be set
    kind: str = "openai"                 # openai | anthropic | gateway
    base_url: str = ""
    models: dict = field(default_factory=dict)   # tier -> model id (best known; override via env)
    free: bool = True
    cn: bool = False
    signup: str = ""
    note: str = ""

    def model_for(self, tier: str = "default") -> str:
        override = os.getenv(f"{self.name.upper()}_MODEL")
        if override:
            return override
        return self.models.get(tier) or self.models.get("default") or next(iter(self.models.values()), "")

    @property
    def configured(self) -> bool:
        return all(os.getenv(k) for k in self.env_keys)


def _p(name, label, env, base_url, model, **kw) -> Provider:
    models = model if isinstance(model, dict) else {"default": model}
    return Provider(name=name, label=label, env_keys=tuple(env), base_url=base_url, models=models, **kw)


PROVIDERS: dict[str, Provider] = {p.name: p for p in [
    Provider("anthropic", "Claude (Anthropic API)", ("ANTHROPIC_API_KEY",), kind="anthropic",
             models=CLAUDE_TIERS, free=False, signup="platform.claude.com"),
    # AI Gateway authenticates with an API key, or with the OIDC token Vercel injects at runtime.
    # Gateway slugs use a dot for decimal versions (anthropic/claude-opus-5.5), unlike Anthropic's own ids.
    Provider("gateway", "Claude via Vercel AI Gateway", (), kind="gateway", base_url="https://ai-gateway.vercel.sh",
             models={t: "anthropic/" + re.sub(r"-(\d+)-(\d+)$", r"-\1.\2", m) for t, m in CLAUDE_TIERS.items()},
             free=False, signup="vercel.com", note="needs credits on the Vercel team"),
    _p("gemini", "Google Gemini", ["GEMINI_API_KEY"], "https://generativelanguage.googleapis.com/v1beta/openai/",
       {"quick": "gemini-2.5-flash-lite", "default": "gemini-2.5-flash", "complex": "gemini-2.5-pro"},
       signup="aistudio.google.com/apikey", note="not available in EU/UK"),
    _p("groq", "Groq", ["GROQ_API_KEY"], "https://api.groq.com/openai/v1",
       {"quick": "llama-3.1-8b-instant", "default": "llama-3.3-70b-versatile"}, signup="console.groq.com/keys"),
    _p("openrouter", "OpenRouter (:free models)", ["OPENROUTER_API_KEY"], "https://openrouter.ai/api/v1",
       "deepseek/deepseek-chat-v3.1:free", signup="openrouter.ai/keys"),
    _p("zai", "Zhipu Z.ai (GLM)", ["ZAI_API_KEY"], "https://api.z.ai/api/paas/v4",
       "glm-4.7-flash", cn=True, signup="z.ai"),
    _p("siliconflow", "SiliconFlow (Qwen/GLM/DeepSeek)", ["SILICONFLOW_API_KEY"], "https://api.siliconflow.com/v1",
       "Qwen/Qwen3-8B", cn=True, signup="cloud.siliconflow.com", note="use the international site (no China ID)"),
    _p("mistral", "Mistral", ["MISTRAL_API_KEY"], "https://api.mistral.ai/v1",
       {"quick": "mistral-small-latest", "default": "mistral-small-latest", "complex": "mistral-large-latest"},
       signup="console.mistral.ai"),
    _p("github", "GitHub Models", ["GITHUB_MODELS_TOKEN"], "https://models.github.ai/inference",
       "openai/gpt-4.1-mini", signup="github.com/marketplace/models", note="PAT with models:read"),
    _p("nvidia", "NVIDIA NIM", ["NVIDIA_API_KEY"], "https://integrate.api.nvidia.com/v1",
       "meta/llama-3.3-70b-instruct", signup="build.nvidia.com"),
    _p("cerebras", "Cerebras", ["CEREBRAS_API_KEY"], "https://api.cerebras.ai/v1",
       "llama-3.3-70b", signup="cloud.cerebras.ai", note="free tier status unverified"),
    _p("cohere", "Cohere", ["COHERE_API_KEY"], "https://api.cohere.ai/compatibility/v1",
       "command-r-plus", signup="dashboard.cohere.com", note="trial key, non-commercial"),
    _p("huggingface", "Hugging Face Inference", ["HF_TOKEN"], "https://router.huggingface.co/v1",
       "Qwen/Qwen2.5-72B-Instruct", signup="huggingface.co/settings/tokens"),
    _p("dashscope", "Alibaba Qwen (Model Studio)", ["DASHSCOPE_API_KEY"],
       "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "qwen-plus", cn=True,
       signup="modelstudio.console.alibabacloud.com", note="trial credits, not permanent"),
    _p("deepseek", "DeepSeek", ["DEEPSEEK_API_KEY"], "https://api.deepseek.com", "deepseek-chat", free=False,
       cn=True, signup="platform.deepseek.com", note="very cheap, not free"),
    _p("moonshot", "Moonshot Kimi", ["MOONSHOT_API_KEY"], "https://api.moonshot.ai/v1", "kimi-k2-turbo-preview",
       cn=True, signup="platform.moonshot.ai", note="free allowance unverified"),
    _p("minimax", "MiniMax", ["MINIMAX_API_KEY"], "https://api.minimax.io/v1", "MiniMax-M2", cn=True,
       signup="platform.minimax.io", note="free allowance unverified"),
    _p("cloudflare", "Cloudflare Workers AI", ["CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID"],
       "", "@cf/meta/llama-3.3-70b-instruct-fp8-fast", signup="dash.cloudflare.com"),
]}

# Preference order when building a fallback chain from whatever keys exist.
FALLBACK_ORDER = ["anthropic", "gateway", "gemini", "groq", "zai", "siliconflow", "openrouter", "cerebras",
                  "mistral", "github", "nvidia", "dashscope", "huggingface", "moonshot", "minimax", "deepseek",
                  "cohere", "cloudflare"]


# A per-request Vercel OIDC token that the web layer has already verified (never a raw header, never os.environ).
request_oidc: ContextVar[str | None] = ContextVar("request_oidc", default=None)


def gateway_token() -> str | None:
    """AI_GATEWAY_API_KEY first, then the OIDC token Vercel injects into the environment at runtime, then a
    per-request verified token (ContextVar). Nothing here ever reads a request header."""
    return os.getenv("AI_GATEWAY_API_KEY") or os.getenv("VERCEL_OIDC_TOKEN") or request_oidc.get()


def is_configured(name: str) -> bool:
    p = PROVIDERS[name]
    return bool(gateway_token()) if p.kind == "gateway" else p.configured


def available() -> list[str]:
    """Names of providers usable right now, in fallback order."""
    return [n for n in FALLBACK_ORDER if is_configured(n)]


def order_for(unlocked: bool) -> list[str]:
    """Configured providers in the order a request of this access level should try them.

    Unlocked (presenter): FALLBACK_ORDER, Claude first. Public: free providers first, then the capped Vercel
    gateway, then the direct Anthropic key (a stranger's traffic should hit the owner's billed key last),
    then any other paid provider.
    """
    if unlocked:
        return available()
    free = [n for n in FALLBACK_ORDER if PROVIDERS[n].free and n != "gateway"]
    paid = [n for n in FALLBACK_ORDER if n not in free and n not in ("gateway", "anthropic")]
    return [n for n in [*free, "gateway", "anthropic", *paid] if is_configured(n)]


def describe() -> list[dict]:
    """Safe for an HTTP response: never includes keys or key values."""
    return [{
        "name": p.name, "label": p.label, "configured": is_configured(p.name), "free": p.free, "cn": p.cn,
        "env": list(p.env_keys), "signup": p.signup, "note": p.note,
        "models": {t: p.model_for(t) for t in (p.models or {"default": ""})},
    } for p in PROVIDERS.values()]


def discover() -> None:
    """List models for every configured OpenAI-compatible provider (run after adding keys)."""
    import httpx

    for name in available():
        p = PROVIDERS[name]
        if p.kind != "openai" or not p.base_url:
            continue
        key = os.environ[p.env_keys[0]]
        try:
            r = httpx.get(p.base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {key}"}, timeout=15)
            ids = sorted(m.get("id", "") for m in r.json().get("data", []))
            print(f"\n== {name} ({r.status_code}) default={p.model_for()} ==\n  " + "\n  ".join(ids[:40]))
        except Exception as e:  # noqa: BLE001
            print(f"\n== {name}: FAILED {type(e).__name__}: {e}")


if __name__ == "__main__":
    if "--discover" in sys.argv:
        discover()
    else:
        for d in describe():
            print(f"{'*' if d['configured'] else ' '} {d['name']:12s} {d['label']}")
