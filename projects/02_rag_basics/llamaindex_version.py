"""The same RAG pipeline in LlamaIndex, to compare with LangChain.

    pip install -r projects/02_rag_basics/requirements-llamaindex.txt
    python projects/02_rag_basics/llamaindex_version.py "What is the hotel cap in metro cities?"

LlamaIndex is data/index-first: loaders, node parsers, index and query engine are
first-class objects. LangChain is composition-first (Runnables you wire together).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def main(question: str) -> None:
    from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
    from llama_index.core.node_parser import MarkdownNodeParser
    from llama_index.embeddings.fastembed import FastEmbedEmbedding
    from llama_index.llms.anthropic import Anthropic

    Settings.llm = Anthropic(model=os.getenv("LLM_MODEL") or "claude-opus-5-5", max_tokens=4096)
    Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

    docs = SimpleDirectoryReader(str(ROOT / "data" / "acme_kb"), exclude=["README.md"]).load_data()
    nodes = MarkdownNodeParser().get_nodes_from_documents(docs)
    index = VectorStoreIndex(nodes)                     # in-memory vector store
    engine = index.as_query_engine(similarity_top_k=4)  # retriever + response synthesizer

    response = engine.query(question)
    print(response)
    print("\nSources:", sorted({n.metadata.get("file_name") for n in response.source_nodes}))


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]) or "What is the hotel cap in metro cities?")
