"""Cost-aware model routing: easy questions go to a small model, hard ones to the strongest.

    from common.router import route
    decision = route("Compare the leave and expense policies and flag conflicts")
    llm = decision.llm()          # Haiku / Sonnet / Opus (or a free provider if Claude isn't configured)

Why it matters (interview): most traffic is easy. Sending everything to the biggest model wastes
money and latency; sending everything to the smallest hurts quality on the hard 10%. A router
sends each request to the cheapest tier that can do it, and the trace records why.

Routing is a cheap deterministic heuristic by default (no extra LLM call, instant, testable).
ROUTER=llm adds a Haiku classifier for ambiguous cases, with the heuristic as the fallback.
Tiers:  quick (Haiku 5.5)  <  default (Sonnet 5.5)  <  complex (Opus 5.5).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from common import providers

# Rough list prices in USD per million tokens (input/output) for the cost estimate shown in the UI.
PRICE = {"quick": (0.10, 0.50), "default": (2.00, 10.00), "complex": (4.00, 20.00)}

HARD = re.compile(r"\b(compare|contrast|trade-?off|why|explain|analy[sz]e|design|conflict|inconsisten|"
                  r"summari[sz]e all|across|pros and cons|recommend|risk|impact|draft|write a|plan)\b", re.I)
ACTION = re.compile(r"\b(raise|create|open|file|submit|approve)\b.*\b(ticket|request|claim)\b|\bticket\b", re.I)
LOOKUP = re.compile(r"^(what|how many|how much|when|who|which|where|is|are|do|does|can)\b", re.I)


@dataclass
class Decision:
    tier: str
    reason: str
    signals: dict = field(default_factory=dict)
    provider: str = ""
    model: str = ""

    def llm(self, **kwargs):
        from common.llm import get_llm

        return get_llm(provider=self.provider or None, tier=self.tier, **kwargs)

    def est_cost_usd(self, in_tokens: int = 1500, out_tokens: int = 250) -> float:
        pin, pout = PRICE[self.tier]
        return (in_tokens * pin + out_tokens * pout) / 1_000_000


def heuristic(question: str) -> tuple[str, str, dict]:
    q = question.strip()
    words = len(q.split())
    signals = {"words": words, "multi_part": q.count("?") > 1 or " and " in q.lower() and words > 14,
               "hard_words": bool(HARD.search(q)), "action": bool(ACTION.search(q)), "lookup": bool(LOOKUP.match(q))}
    if signals["hard_words"] and (words >= 8 or signals["multi_part"]):
        return "complex", "open-ended / comparative question needs reasoning across sources", signals
    if signals["action"]:
        return "default", "tool-using request (creates something, needs reliable tool calls)", signals
    if signals["multi_part"] or words > 25:
        return "default", "multi-step question", signals
    if words <= 3 and not signals["hard_words"]:
        return "quick", "greeting / very short message", signals
    if signals["lookup"] and words <= 14 and not signals["hard_words"]:
        return "quick", "short single-fact lookup", signals
    return "default", "general question", signals


def route(question: str, prefer: str | None = None) -> Decision:
    """Pick a tier, then the best configured provider for it. `prefer` pins a provider by name."""
    tier, reason, signals = heuristic(question)
    if os.getenv("ROUTER", "heuristic") == "llm" and signals["words"] > 6:
        try:  # ambiguous cases only: ask the cheap model (never fatal)
            from common.llm import get_llm

            out = get_llm(tier="quick").invoke(
                "Classify the difficulty of answering this question from company policy documents. "
                "Reply with exactly one word: quick (single fact lookup), default (needs a few facts), "
                f"or complex (compare/analyse/draft).\n\nQuestion: {question}").content
            word = str(out).strip().lower().split()[0].strip(".,")
            if word in PRICE:
                tier, reason = word, f"classifier chose {word}"
        except Exception:  # noqa: BLE001
            pass
    name = prefer if prefer and providers.is_configured(prefer) else next(iter(providers.available()), "")
    model = providers.PROVIDERS[name].model_for(tier) if name else ""
    return Decision(tier=tier, reason=reason, signals=signals, provider=name, model=model)
