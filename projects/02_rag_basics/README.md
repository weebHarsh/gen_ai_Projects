# 02 · RAG: Retrieval-Augmented Generation (90 min)

**You'll learn:** loading, chunking, embeddings, vector stores (Chroma, FAISS), MMR retrieval, grounded answers with citations, and the same pipeline in LlamaIndex.

```
data/acme_kb/*.md ─► load ─► split (by headers, then size) ─► embed (BGE-small) ─► Chroma / FAISS
                                                                                     │
question ─► embed ─► MMR top-k ─► prompt(context + question + "cite or say I don't know") ─► Claude ─► answer [file#chunk]
```

## Concepts in 60 seconds
| Concept | What to say |
|---|---|
| Why RAG | Gives the model private, fresh knowledge without retraining, cuts hallucination, gives citations and access control. |
| Chunking | Split on document structure first (headers), then size (~400–800 tokens, 10–15% overlap). Add a **contextual header** (doc > section) to every chunk. Bad chunking is the #1 cause of bad RAG. |
| Embeddings | Claude has no embeddings API. Use Voyage AI (Anthropic's recommendation), OpenAI `text-embedding-3-small`, or local BGE via FastEmbed (free, CPU). The same model must embed both documents and queries. |
| Vector DBs | **Chroma**: dev/local, metadata filters. **FAISS**: in-memory library, very fast, no server. **Pinecone**: managed serverless. **Weaviate**: hybrid search (BM25 + vectors), self-host or cloud. **pgvector**: when you already run Postgres. |
| Retrieval | Similarity top-k, **MMR** (diversity), **hybrid** (BM25 + dense), **re-ranking** (cross-encoder / Cohere / Voyage rerank), metadata filters (department, date, ACL). |
| Grounding | Instruct "answer only from context, cite, else say I don't know". Measure it with **faithfulness** (project 06). |
| Advanced | Query rewriting / HyDE, parent-document retrieval, multi-query, agentic RAG (retrieval as a tool, project 03), GraphRAG. |

## Run
```bash
python projects/02_rag_basics/main.py inspect "carry forward leave"       # see chunks, no LLM call
python projects/02_rag_basics/main.py ask "How many unused leave days can I carry forward?"
python projects/02_rag_basics/main.py ask "What's the hotel cap in metro cities?" --store faiss
# LlamaIndex version
pip install -r projects/02_rag_basics/requirements-llamaindex.txt
python projects/02_rag_basics/llamaindex_version.py "What is the hotel cap in metro cities?"
```
The first run downloads the BGE-small model (~130 MB) from HuggingFace.

## Things to try
1. Ask something not in the KB ("What is the CEO's salary?"). It should say *I don't know*.
2. Change `chunk_size` to 200 and then 1500, re-run `inspect`, and see how retrieval changes.
3. Switch `search_type="mmr"` to `"similarity"` in `get_retriever`.
4. Add metadata filtering: only search `it_*.md` files for IT questions.

## LangChain vs LlamaIndex (interview answer)
LlamaIndex is **data/index-first**: rich loaders, node parsers, index types and query engines, so it's quickest for document Q&A. LangChain is **composition-first**: Runnables, tools and agents, with LangGraph for orchestration. They're commonly combined: a LlamaIndex retriever used as a tool inside a LangGraph agent.

## Interview questions
- *Your RAG gives wrong answers. How do you debug?* Separate retrieval from generation. Check whether the right chunk was retrieved (context recall/precision). If not: chunking, embeddings, hybrid search, re-ranking. If it was: the prompt, the model, and faithfulness. Use traces (project 05) and evals (project 06).
- *How do you handle document-level permissions?* Store ACL metadata on each chunk and filter at query time by the user's groups. Never filter after generation.
- *How do you keep the index fresh?* Incremental ingestion keyed on doc hash/updated_at, delete-then-upsert by doc id, scheduled re-index.
