"""Evaluate the project-02 RAG pipeline on the golden dataset.

    python projects/06_agent_evals/eval_rag.py              # deterministic + Ragas LLM-judge metrics
    python projects/06_agent_evals/eval_rag.py --no-judge   # deterministic only (free, offline OK)

Two layers:
  1. Deterministic (no LLM): did retrieval return the expected source file? Did the bot
     refuse when it should? Cheap, exact, run on every commit.
  2. Ragas LLM-as-judge: faithfulness (is every claim supported by the retrieved context?),
     context precision/recall (did we retrieve the right chunks?), factual correctness vs reference.
Exits non-zero if faithfulness < 0.85 (Acme AI policy) -> usable as a CI quality gate.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "projects" / "02_rag_basics"))
sys.path.insert(0, str(Path(__file__).parent))

from judge import DATASET  # noqa: E402
from rag_pipeline import answer  # noqa: E402

from common.observability import flush, trace_config  # noqa: E402

FAITHFULNESS_GATE = 0.85
REFUSAL = "i don't know"


def run_pipeline() -> list[dict]:
    rows = []
    for case in DATASET["rag"]:
        out = answer(case["question"], config=trace_config("06-eval-rag", tags=["eval"]))
        rows.append({**case, **out})
    return rows


def deterministic_metrics(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["source"]]
    unanswerable = [r for r in rows if not r["source"]]
    hit = sum(r["source"] in r["sources"] for r in answerable) / len(answerable)
    refused = sum(REFUSAL in r["answer"].lower() for r in unanswerable) / max(len(unanswerable), 1)
    cited = sum("[" in r["answer"] and ".md#" in r["answer"] for r in answerable) / len(answerable)
    return {"source_hit_rate": hit, "refusal_rate_out_of_scope": refused, "citation_rate": cited}


def ragas_metrics(rows: list[dict]) -> dict:
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (FactualCorrectness, Faithfulness, LLMContextPrecisionWithReference,
                               LLMContextRecall)

    from judge import ragas_judge

    dataset = EvaluationDataset.from_list([
        {"user_input": r["question"], "response": r["answer"],
         "retrieved_contexts": r["contexts"], "reference": r["reference"]}
        for r in rows if r["source"]  # out-of-scope rows are checked deterministically
    ])
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), LLMContextPrecisionWithReference(), LLMContextRecall(),
                 FactualCorrectness(mode="recall")],
        llm=ragas_judge(),
    )
    df = result.to_pandas()
    df.to_csv(Path(__file__).parent / "ragas_results.csv", index=False)
    return {c: float(df[c].mean()) for c in df.columns if df[c].dtype.kind == "f"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-judge", action="store_true", help="skip Ragas LLM-judge metrics")
    args = parser.parse_args(argv)

    rows = run_pipeline()
    scores = deterministic_metrics(rows)
    use_judge = not args.no_judge and os.getenv("LLM_PROVIDER", "anthropic") != "fake"
    if use_judge:
        scores |= ragas_metrics(rows)
    flush()

    print("\n=== RAG eval ===")
    for name, value in scores.items():
        print(f"{name:32s} {value:.2f}")
    if use_judge and scores.get("faithfulness", 0) < FAITHFULNESS_GATE:
        print(f"\nFAIL: faithfulness below the {FAITHFULNESS_GATE} go-live gate")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
