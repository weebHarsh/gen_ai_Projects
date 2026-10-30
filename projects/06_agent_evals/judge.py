"""Judge-model helpers shared by the eval scripts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from langchain_anthropic import ChatAnthropic  # noqa: E402

DATASET = json.loads((Path(__file__).parent / "golden_dataset.json").read_text())


class NoSamplingChatAnthropic(ChatAnthropic):
    """Ragas sets temperature=0.01 on the judge, but current Claude models (Opus 5.5,
    Sonnet 5.5, ...) reject sampling params with a 400. Strip them from every request."""

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        for key in ("temperature", "top_p", "top_k"):
            payload.pop(key, None)
            payload.get("extra_body", {}).pop(key, None)
        return payload


def ragas_judge(model: str = "claude-opus-5-5"):
    """A Claude judge wrapped for Ragas. Use a strong model for judging; it's a small dataset."""
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(NoSamplingChatAnthropic(model=model, max_tokens=8000),
                               bypass_temperature=True)
