"""Run the golden dataset as a Langfuse *experiment*: results, traces and scores in one UI,
comparable run-over-run (e.g. prompt v1 vs v2, Opus vs Haiku, chunk 600 vs 1200).

    python projects/06_agent_evals/langfuse_experiment.py --run-name "chunk-600"

Needs LANGFUSE_* keys. Creates (or reuses) the dataset "acme-rag-golden" in Langfuse.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "projects" / "02_rag_basics"))
sys.path.insert(0, str(Path(__file__).parent))

from judge import DATASET  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from rag_pipeline import answer  # noqa: E402

from common.llm import get_llm, structured  # noqa: E402
from common.observability import enabled  # noqa: E402

DATASET_NAME = "acme-rag-golden"


class Verdict(BaseModel):
    correct: bool = Field(description="Does the answer state the same facts as the reference?")
    reason: str


def task(*, item, **_):
    """What's being evaluated: the RAG pipeline. Langfuse traces each call automatically."""
    q = item["input"] if isinstance(item, dict) else item.input
    return answer(q["question"] if isinstance(q, dict) else q)


def source_hit(*, output, metadata, **_):
    from langfuse import Evaluation

    expected = (metadata or {}).get("source")
    if not expected:
        return Evaluation(name="source_hit", value=1.0, comment="out-of-scope question")
    return Evaluation(name="source_hit", value=float(expected in output["sources"]))


def llm_judge_correctness(*, input, output, expected_output, **_):
    from langfuse import Evaluation

    verdict = structured(get_llm(), Verdict).invoke(
        f"Question: {input}\nReference answer: {expected_output}\nCandidate answer: {output['answer']}\n"
        "Is the candidate factually consistent with the reference? Extra correct detail is fine."
    )
    return Evaluation(name="correctness", value=float(verdict.correct), comment=verdict.reason)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", default="baseline")
    args = parser.parse_args(argv)
    if not enabled():
        sys.exit("Set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY first.")

    from langfuse import get_client

    langfuse = get_client()
    try:
        dataset = langfuse.get_dataset(DATASET_NAME)
    except Exception:  # first run: upload the golden set
        langfuse.create_dataset(name=DATASET_NAME, description="Acme policy Q&A golden set")
        for case in DATASET["rag"]:
            langfuse.create_dataset_item(dataset_name=DATASET_NAME, input={"question": case["question"]},
                                         expected_output=case["reference"], metadata={"source": case["source"]})
        dataset = langfuse.get_dataset(DATASET_NAME)

    result = dataset.run_experiment(
        name="rag-quality", run_name=args.run_name,
        task=task, evaluators=[source_hit, llm_judge_correctness], max_concurrency=4,
    )
    print(result.format())
    langfuse.flush()


if __name__ == "__main__":
    main()
