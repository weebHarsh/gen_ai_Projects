"""Evaluate the project-03 agent: did it pick the right tools, and was the answer helpful?

    python projects/06_agent_evals/eval_agent.py              # tool check + DeepEval metrics
    python projects/06_agent_evals/eval_agent.py --no-judge   # tool check only (offline OK)

Agent evals look at the *trajectory* (which tools, in what order, with what args), not just
the final text. Here:
  - tool_correctness (deterministic): expected tools ⊆ tools actually called, and no tools
    called when none were expected
  - DeepEval ToolCorrectnessMetric: the same idea, reported in DeepEval's format
  - DeepEval GEval "Helpfulness": LLM-as-judge with explicit evaluation steps
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "projects" / "03_langgraph_agent"))
sys.path.insert(0, str(Path(__file__).parent))

from graph import build_graph, run_turn  # noqa: E402
from judge import DATASET  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from common.observability import flush  # noqa: E402


def run_agent() -> list[dict]:
    graph = build_graph()
    results = []
    for i, case in enumerate(DATASET["agent"]):
        thread = f"eval-agent-{i}"
        final = run_turn(graph, case["input"], thread_id=thread, auto_approve=True)
        messages = graph.get_state({"configurable": {"thread_id": thread}}).values["messages"]
        called = [tc["name"] for m in messages if isinstance(m, AIMessage) for tc in m.tool_calls]
        results.append({**case, "output": final, "tools_called": called})
    return results


def tool_correctness(r: dict) -> float:
    expected, called = set(r["expected_tools"]), set(r["tools_called"])
    if not expected:
        return float(not called)
    return float(expected <= called)


def deepeval_metrics(results: list[dict]) -> None:
    from deepeval import evaluate
    from deepeval.metrics import GEval, ToolCorrectnessMetric
    from deepeval.models import AnthropicModel
    from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall

    judge = AnthropicModel(model="claude-opus-5-5")
    helpfulness = GEval(
        name="Helpfulness",
        evaluation_steps=[
            "Check the output directly addresses the user's request.",
            "Penalise invented policy facts; reward citations like [file.md#n].",
            "If an action was requested (e.g. a ticket), check the output confirms it with an id.",
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=judge, threshold=0.6,
    )
    cases = [
        LLMTestCase(input=r["input"], actual_output=r["output"],
                    tools_called=[ToolCall(name=t) for t in r["tools_called"]],
                    expected_tools=[ToolCall(name=t) for t in r["expected_tools"]])
        for r in results
    ]
    evaluate(test_cases=cases, metrics=[ToolCorrectnessMetric(model=judge), helpfulness])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args(argv)

    results = run_agent()
    flush()
    print("\n=== Agent tool-use eval ===")
    for r in results:
        print(f"{tool_correctness(r):.0f}  expected={r['expected_tools']}  called={r['tools_called']}  | {r['input'][:50]}")
    score = sum(map(tool_correctness, results)) / len(results)
    print(f"tool_correctness: {score:.2f}")

    if not args.no_judge and os.getenv("LLM_PROVIDER", "anthropic") != "fake":
        deepeval_metrics(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
