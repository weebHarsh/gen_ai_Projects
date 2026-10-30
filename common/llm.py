"""One place to choose the LLM for every project.

    from common.llm import get_llm, structured
    llm = get_llm()                      # provider from LLM_PROVIDER
    parser_llm = structured(llm, MyPydanticModel)

LLM_PROVIDER = anthropic | openai | azure | do | fake
    anthropic -> Claude via langchain-anthropic (default model claude-opus-5-5)
    openai    -> OpenAI via langchain-openai
    azure     -> Azure OpenAI (LLM_MODEL = deployment name)
    do        -> DigitalOcean AI Platform serverless inference (OpenAI-compatible API)
    fake      -> offline scripted model, no key needed; used by tests/test_smoke.py
"""
from __future__ import annotations

import json
import os
from typing import Any, Sequence

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

load_dotenv()

DEFAULT_MODELS = {
    "openai": "gpt-5-mini",
    "do": "llama3.3-70b-instruct",
}


def provider() -> str:
    return os.getenv("LLM_PROVIDER", "anthropic").strip().lower()


def get_llm(provider: str | None = None, model: str | None = None, tier: str | None = None,
            **kwargs: Any) -> BaseChatModel:
    """Return a chat model.

    provider: a name from common.providers.PROVIDERS, or openai | azure | do | fake. Default: LLM_PROVIDER.
    tier:     quick | default | complex, picks the provider's model for that tier (Claude: Haiku/Sonnet/Opus).
    model:    explicit model id; wins over tier. LLM_MODEL env var also wins over tier when set.
    kwargs pass through to the underlying model class.
    """
    from common import providers as reg

    p = (provider or globals()["provider"]()).strip().lower()
    model = model or os.getenv("LLM_MODEL") or None

    if p == "fake":
        return FakeChatModel()
    if p == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # No temperature: current Claude models reject sampling params; tune with effort instead.
        return ChatAnthropic(model=model or reg.PROVIDERS["anthropic"].model_for(tier or "complex"),
                             max_tokens=kwargs.pop("max_tokens", 16000), **kwargs)
    if p == "gateway":
        from langchain_anthropic import ChatAnthropic

        token = kwargs.pop("api_key", None) or reg.gateway_token()
        if not token:
            raise RuntimeError("gateway needs AI_GATEWAY_API_KEY or a Vercel OIDC token")
        gw_tier = tier or "default"
        extra = {}
        if not model:  # explicit model ids are used as given; defaults get the gateway's own fallback list
            extra["model_kwargs"] = {"extra_body": {"providerOptions": {"gateway": {
                "models": reg.GATEWAY_FALLBACK_MODELS.get(gw_tier, [])}}}}
        return ChatAnthropic(model=model or reg.PROVIDERS["gateway"].model_for(gw_tier),
                             base_url="https://ai-gateway.vercel.sh", api_key=token,
                             max_tokens=kwargs.pop("max_tokens", 4000), **extra, **kwargs)
    if p == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model or DEFAULT_MODELS["openai"], **kwargs)
    if p == "azure":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(azure_deployment=model or "", **kwargs)
    if p == "do":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or DEFAULT_MODELS["do"],
            api_key=os.environ["DO_MODEL_ACCESS_KEY"],
            base_url=os.getenv("DO_INFERENCE_BASE_URL", "https://inference.do-ai.run/v1"),
            **kwargs,
        )
    if p in reg.PROVIDERS:  # every other registry entry speaks the OpenAI chat-completions protocol
        from langchain_openai import ChatOpenAI

        spec = reg.PROVIDERS[p]
        if not reg.is_configured(p):
            raise RuntimeError(f"{p}: set {', '.join(spec.env_keys)}")
        base_url = spec.base_url
        if p == "cloudflare":
            base_url = f"https://api.cloudflare.com/client/v4/accounts/{os.environ['CLOUDFLARE_ACCOUNT_ID']}/ai/v1"
        return ChatOpenAI(model=model or spec.model_for(tier or "default"), api_key=os.environ[spec.env_keys[0]],
                          base_url=base_url, max_tokens=kwargs.pop("max_tokens", 2000), **kwargs)
    raise ValueError(f"Unknown LLM_PROVIDER={p!r}")


def get_llm_chain(tier: str = "default", prefer: str | None = None, max_providers: int = 4,
                  names: Sequence[str] | None = None, **kwargs: Any):
    """A model that falls through to the next configured provider on any error (quota, 429, outage).

    Order = common.providers.FALLBACK_ORDER (Claude first, then free tiers), `prefer` first if set.
    `names` replaces that list (already filtered to configured providers), e.g. providers.order_for(unlocked).
    Falls back to the FakeChatModel only if nothing is configured and LLM_PROVIDER=fake.
    """
    from common import providers as reg

    names = list(names) if names is not None else reg.available()
    if prefer and prefer in names:
        names = [prefer] + [n for n in names if n != prefer]
    if not names:
        if provider() == "fake":
            return FakeChatModel()
        raise RuntimeError("no LLM provider configured")
    models = [get_llm(provider=n, tier=tier, **dict(kwargs)) for n in names[:max_providers]]
    for n, m in zip(names, models):
        m.metadata = {**(m.metadata or {}), "provider": n}
    return models[0].with_fallbacks(models[1:]) if len(models) > 1 else models[0]


def structured(llm: BaseChatModel, schema: type[BaseModel]):
    """llm.with_structured_output with the method that works per provider.

    Claude Opus 5.5 / Sonnet 5.5 reject *forced* tool_choice, so for Anthropic we use the
    native JSON-schema output mode (output_config.format) instead of a forced tool call.
    Fallback chains structure each model separately and re-chain them.
    """
    if hasattr(llm, "runnable") and hasattr(llm, "fallbacks"):  # RunnableWithFallbacks
        return structured(llm.runnable, schema).with_fallbacks([structured(f, schema) for f in llm.fallbacks])
    if isinstance(llm, FakeChatModel):
        return llm.with_structured_output(schema)
    cls = type(llm).__name__
    native_json = cls in {"ChatAnthropic", "AzureChatOpenAI"} or (
        cls == "ChatOpenAI" and "api.openai.com" in str(getattr(llm, "openai_api_base", "") or "api.openai.com"))
    return llm.with_structured_output(schema, method="json_schema" if native_json else "function_calling")


# --------------------------------------------------------------------------- fake model
def _dummy_value(annotation: Any) -> Any:
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())
    if origin is list:
        return [_dummy_value(args[0])] if args else []
    if str(origin) == "typing.Literal" or type(annotation).__name__ == "_LiteralGenericAlias":
        return args[0]
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return dummy_instance(annotation)
    return {str: "fake", int: 1, float: 0.5, bool: True}.get(annotation, None)


def dummy_instance(schema: type[BaseModel]) -> BaseModel:
    """Build a valid instance of a Pydantic model with placeholder values."""
    values = {}
    for name, field in schema.model_fields.items():
        if not field.is_required():
            continue
        values[name] = _dummy_value(field.annotation)
    return schema.model_validate(values)


class FakeChatModel(BaseChatModel):
    """Deterministic offline model so every project can run without an API key.

    - With tools bound and no tool result yet: calls the first tool, filling string args
      with the user's last message.
    - Otherwise: echoes a short 'FAKE ANSWER' that quotes the last message.
    """

    tools: list[dict] = []

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeChatModel":
        return FakeChatModel(tools=[convert_to_openai_tool(t) for t in tools])

    def with_structured_output(self, schema: Any, **kwargs: Any):  # type: ignore[override]
        return RunnableLambda(lambda _input: dummy_instance(schema))

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        last = messages[-1]
        text = last.content if isinstance(last.content, str) else json.dumps(last.content)
        already_used_tool = any(isinstance(m, ToolMessage) for m in messages)
        if self.tools and not already_used_tool:
            fn = self.tools[0]["function"]
            props = fn.get("parameters", {}).get("properties", {})
            args = {
                k: (text[:200] if v.get("type") == "string" else 1)
                for k, v in props.items()
            }
            msg = AIMessage(content="", tool_calls=[{"name": fn["name"], "args": args, "id": "call_fake_1"}])
        else:
            msg = AIMessage(content=f"FAKE ANSWER (offline mode). Last input: {text[:300]}")
        return ChatResult(generations=[ChatGeneration(message=msg)])
