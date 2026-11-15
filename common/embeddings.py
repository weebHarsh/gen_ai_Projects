"""Embeddings switch. Claude has no embeddings endpoint, so RAG needs a separate embedder.

EMBEDDINGS_PROVIDER = fastembed (default, local CPU, free) | openai | fake (offline tests)
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

load_dotenv()


def get_embeddings() -> Embeddings:
    p = os.getenv("EMBEDDINGS_PROVIDER", "fastembed").lower()
    if os.getenv("LLM_PROVIDER", "").lower() == "fake":
        p = "fake"
    if p == "fastembed":
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

        # ~130 MB ONNX model, downloaded once, runs on CPU. 384-dim vectors.
        return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    if p == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-small")
    if p == "fake":
        return DeterministicFakeEmbedding(size=384)
    raise ValueError(f"Unknown EMBEDDINGS_PROVIDER={p!r}")
