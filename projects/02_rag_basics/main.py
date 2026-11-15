"""02 - RAG over the Acme knowledge base.

    python projects/02_rag_basics/main.py inspect                 # see chunks + retrieval, no LLM call
    python projects/02_rag_basics/main.py ask "How many days of annual leave do I get?"
    python projects/02_rag_basics/main.py ask "..." --store faiss  # swap the vector store
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_pipeline import answer, get_retriever, load_documents, split_documents  # noqa: E402

from common.observability import flush, trace_config  # noqa: E402


def inspect(question: str, store: str) -> None:
    docs = load_documents()
    chunks = split_documents(docs)
    print(f"{len(docs)} documents -> {len(chunks)} chunks")
    print("Sample chunk metadata:", chunks[0].metadata)
    print(f"\nTop chunks for: {question!r}")
    for d in get_retriever(store).invoke(question):
        print(f"- {d.metadata['source']}#{d.metadata['chunk_id']}: {d.page_content[:90]!r}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["inspect", "ask"])
    parser.add_argument("question", nargs="?", default="How many leave days can I carry forward?")
    parser.add_argument("--store", choices=["chroma", "faiss"], default="chroma")
    args = parser.parse_args(argv)

    if args.command == "inspect":
        inspect(args.question, args.store)
        return
    result = answer(args.question, args.store, config=trace_config("02-rag", session_id="rag-demo"))
    print(result["answer"])
    print("\nSources:", ", ".join(result["sources"]))
    flush()


if __name__ == "__main__":
    main()
