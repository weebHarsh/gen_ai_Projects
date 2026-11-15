"""05 - AI observability with Langfuse: traces, sessions, users, tags, scores, prompt management.

Set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL in .env first, then:

    python projects/05_observability_langfuse/main.py trace-rag        # LangChain callback tracing
    python projects/05_observability_langfuse/main.py trace-agent      # multi-step LangGraph trace
    python projects/05_observability_langfuse/main.py custom-spans     # @observe + manual spans for non-LangChain code
    python projects/05_observability_langfuse/main.py prompt-mgmt      # versioned prompts fetched at runtime
    python projects/05_observability_langfuse/main.py feedback         # attach a user-feedback score to a trace
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "projects" / "02_rag_basics"))
sys.path.insert(0, str(ROOT / "projects" / "03_langgraph_agent"))

from common.observability import enabled, flush, trace_config  # noqa: E402

QUESTIONS = [
    "How many unused leave days can I carry forward?",
    "What is the hotel cap per night in metro cities?",
    "Which models are approved for confidential data?",
]


def trace_rag() -> None:
    """One trace per question, grouped into one session, tagged for filtering."""
    from rag_pipeline import answer

    for q in QUESTIONS:
        cfg = trace_config("rag-query", session_id="obs-demo-session", user_id="harsh",
                           tags=["project-05", "rag"])
        print(f"Q: {q}\nA: {answer(q, config=cfg)['answer'][:200]}\n")


def trace_agent() -> None:
    """Agent traces show the full tree: LLM call -> tool call -> retriever -> LLM call."""
    from graph import build_graph, run_turn

    graph = build_graph()
    print(run_turn(graph, "I've worked 7 months, taken 5 days, carried 3. What's my leave balance?",
                   thread_id="obs-agent-1", auto_approve=True))


def custom_spans() -> None:
    """For code that isn't LangChain (raw SDK calls, business logic), use @observe and spans."""
    from langfuse import get_client, observe

    langfuse = get_client()

    @observe(name="pii-mask")
    def mask_pii(text: str) -> str:
        return text.replace("harsh@example.com", "<EMAIL>")

    @observe(name="ticket-pipeline")
    def pipeline(text: str) -> str:
        clean = mask_pii(text)
        with langfuse.start_as_current_observation(name="classify", as_type="generation",
                                                   model="rule-based-v1", input=clean) as gen:
            time.sleep(0.1)
            label = "P2" if "team" in clean else "P3"
            gen.update(output=label, usage_details={"input": len(clean.split()), "output": 1})
        langfuse.score_current_trace(name="rule_confidence", value=0.7)
        return label

    print("label:", pipeline("VPN down for my whole team, contact harsh@example.com"))


def prompt_mgmt() -> None:
    """Prompts live in Langfuse, versioned and labelled (production/staging); code fetches by label.
    Non-engineers can edit prompts and you can roll back without a deploy."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langfuse import get_client

    from common.llm import get_llm

    langfuse = get_client()
    langfuse.create_prompt(
        name="helpdesk-triage",
        type="chat",
        prompt=[{"role": "system", "content": "Classify the IT ticket as P1-P4. Answer with the label only."},
                {"role": "user", "content": "{{ticket}}"}],
        labels=["production"],
    )
    prompt = langfuse.get_prompt("helpdesk-triage", type="chat", label="production")
    lc_prompt = ChatPromptTemplate.from_messages(prompt.get_langchain_prompt())
    lc_prompt.metadata = {"langfuse_prompt": prompt}  # links traces to the prompt version
    chain = lc_prompt | get_llm() | StrOutputParser()
    print(chain.invoke({"ticket": "Payments API down for all customers"}, config=trace_config("prompt-mgmt")))


def feedback() -> None:
    """Thumbs up/down from your UI becomes a score on the trace -> filter bad answers, build datasets."""
    from langfuse import get_client

    from common.llm import get_llm

    langfuse = get_client()
    with langfuse.start_as_current_observation(name="chat-turn", as_type="span") as span:
        reply = get_llm().invoke("In one line: what is RAG?", config=trace_config("feedback-demo"))
        span.update(output=reply.content)
        trace_id = langfuse.get_current_trace_id()
    langfuse.create_score(name="user_feedback", value=1, data_type="NUMERIC", trace_id=trace_id,
                          comment="thumbs up from UI")
    print("trace:", langfuse.get_trace_url(trace_id=trace_id))


DEMOS = {"trace-rag": trace_rag, "trace-agent": trace_agent, "custom-spans": custom_spans,
         "prompt-mgmt": prompt_mgmt, "feedback": feedback}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("demo", choices=list(DEMOS))
    args = parser.parse_args(argv)
    if not enabled() and args.demo not in {"trace-rag", "trace-agent"}:
        sys.exit("Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env first (see README).")
    if not enabled():
        print("[warn] Langfuse keys not set: running without tracing.\n")
    DEMOS[args.demo]()
    flush()


if __name__ == "__main__":
    main()
