"""Reusable RAG pipeline: load -> chunk -> embed -> index -> retrieve -> answer with citations.

Projects 03 (agent tool), 06 (evals) and 08 (API) import this module, so the RAG they
measure and deploy is exactly the one you learn here.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from common.embeddings import get_embeddings
from common.llm import get_llm

KB_DIR = ROOT / "data" / "acme_kb"


def load_documents(kb_dir: Path = KB_DIR) -> list[Document]:
    docs = []
    for path in sorted(kb_dir.glob("*.md")):
        if path.name == "README.md":
            continue
        docs.append(Document(page_content=path.read_text(), metadata={"source": path.name}))
    return docs


def split_documents(docs: list[Document], chunk_size: int = 600, chunk_overlap: int = 80) -> list[Document]:
    """Two-stage split: by markdown headers first (keeps sections intact and records the
    section title as metadata), then by size for long sections.

    Each chunk then gets a *contextual header* ("Leave Policy > Sick leave") prepended, so a
    chunk like "12 days per year" still says what it is about when embedded on its own.
    Header-only fragments (a title line with no body) are dropped.
    """
    header_splitter = MarkdownHeaderTextSplitter([("#", "title"), ("##", "section")], strip_headers=True)
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Document] = []
    for doc in docs:
        for part in header_splitter.split_text(doc.page_content):
            part.metadata = {**doc.metadata, **part.metadata}
            for piece in size_splitter.split_documents([part]):
                if len(piece.page_content.strip()) < 40:
                    continue
                header = " > ".join(v for k in ("title", "section") if (v := piece.metadata.get(k)))
                piece.page_content = f"{header}\n{piece.page_content}"
                chunks.append(piece)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def build_vectorstore(chunks: list[Document], store: str = "chroma") -> VectorStore:
    embeddings = get_embeddings()
    if store == "chroma":
        from langchain_chroma import Chroma

        # In-memory collection; pass persist_directory=".chroma" to keep it on disk.
        return Chroma.from_documents(chunks, embeddings, collection_name="acme_kb")
    if store == "faiss":
        from langchain_community.vectorstores import FAISS

        return FAISS.from_documents(chunks, embeddings)
    raise ValueError(f"Unknown store {store!r}")


@lru_cache(maxsize=2)
def get_retriever(store: str = "chroma", k: int = 4):
    """MMR = Maximal Marginal Relevance: relevant *and* diverse chunks (less duplication)."""
    vs = build_vectorstore(split_documents(load_documents()), store)
    return vs.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": 12})


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata['source']}#{d.metadata.get('chunk_id')}]\n{d.page_content}" for d in docs
    )


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about Acme Corp policies using ONLY the context.\n"
     "Cite sources inline like [leave_policy.md#3]. If the context does not contain the "
     "answer, say \"I don't know based on the policies.\" Do not guess."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def answer(question: str, store: str = "chroma", config: dict | None = None) -> dict:
    """Returns {'answer', 'contexts', 'sources'} - the shape evals (project 06) need."""
    docs = get_retriever(store).invoke(question, config=config)
    chain = RAG_PROMPT | get_llm() | StrOutputParser()
    text = chain.invoke({"context": format_context(docs), "question": question}, config=config)
    return {
        "answer": text,
        "contexts": [d.page_content for d in docs],
        "sources": sorted({d.metadata["source"] for d in docs}),
    }
