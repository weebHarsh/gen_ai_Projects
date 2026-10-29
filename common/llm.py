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
    "anthropic": "claude-opus-5-5",
    "openai": "gpt-5-mini",
    "do": "llama3.3-70b-instruct",
}


def provider() -> str:
    return os.getenv("LLM_PROVIDER", "anthropic").strip().lower()


def get_llm(**kwargs: Any) -> BaseChatModel:
    """Return a chat model for the configured provider. kwargs pass through to the model class."""
    p = provider()
    model = os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(p, "")

    if p == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # No temperature: current Claude models reject sampling params; tune with effort instead.
        return ChatAnthropic(model=model, max_tokens=16000, **kwargs)
    if p == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, **kwargs)
    if p == "azure":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(azure_deployment=model, **kwargs)
    if p == "do":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=os.environ["DO_MODEL_ACCESS_KEY"],
            base_url=os.getenv("DO_INFERENCE_BASE_URL", "https://inference.do-ai.run/v1"),
            **kwargs,
        )
    if p == "fake":
        return FakeChatModel()
    raise ValueError(f"Unknown LLM_PROVIDER={p!r}")


def structured(llm: BaseChatModel, schema: type[BaseModel]):
    """llm.with_structured_output with the method that works per provider.

    Claude Opus 5.5 / Sonnet 5.5 reject *forced* tool_choice, so for Anthropic we use the
    native JSON-schema output mode (output_config.format) instead of a forced tool call.
    """
    if isinstance(llm, FakeChatModel):
        return llm.with_structured_output(schema)
    method = "json_schema" if provider() in {"anthropic", "openai", "azure"} else "function_calling"
    return llm.with_structured_output(schema, method=method)


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
