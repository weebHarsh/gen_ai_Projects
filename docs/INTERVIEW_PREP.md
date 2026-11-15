# Interview Prep: GenAI Engineer (Python · LLMs · RAG · Agentic AI · Automation)

## 1. Your 60-second project pitch (built from this repo)
> "I built an **enterprise HR/IT helpdesk assistant** for a fictional company knowledge base. It has a **RAG pipeline** (structure-aware chunking with contextual headers, BGE embeddings, Chroma with MMR, cited answers) and a **LangGraph agent** with three tools: policy search, a leave calculator and a ServiceDesk REST API. The ticket tool pauses for **human approval** using LangGraph `interrupt`. Everything is traced in **Langfuse**, self-hosted on a DigitalOcean Droplet, with sessions, cost and prompt versions. I evaluate it with a **golden dataset**: Ragas faithfulness and context recall for RAG, and DeepEval tool-correctness and G-Eval for the agent. Faithfulness ≥ 0.85 is a CI gate. It's served as a **FastAPI** service on **DO App Platform**, so UiPath or Power Automate can call it over REST. I also built the same research-and-write task with **CrewAI, AutoGen and a LangGraph supervisor** to compare multi-agent styles, and ran a long-running version on **Claude Managed Agents** with a custom tool that keeps private data on my side."

Have numbers ready from your own runs: eval scores, p95 latency and cost per query from Langfuse.

## 2. System design: "Design an enterprise knowledge assistant for 10k employees"
1. **Requirements**: users, document sources (SharePoint, Confluence, PDFs), ACLs, languages, latency target (< 3 s to first token), and compliance (PII, data residency).
2. **Ingestion**: connectors → parse (tables, PDFs) → structure-aware chunking + contextual headers → embeddings → vector DB with metadata (source, dept, ACL groups, updated_at). Incremental re-index by content hash.
3. **Retrieval**: hybrid (BM25 + dense) → ACL filter by the user's groups → rerank → top-k with MMR.
4. **Generation**: grounded prompt with citations and an "I don't know" path, structured output where needed, streaming.
5. **Agentic layer** (when actions are needed): LangGraph with tools behind least-privilege service accounts, human approval for writes, step limits.
6. **Guardrails**: input (prompt injection, PII), output (groundedness, toxicity), rate limits.
7. **Observability & evals**: Langfuse traces, cost and latency dashboards, golden set in CI, online LLM-judge on sampled traffic, feedback buttons feeding back into the dataset.
8. **Scale & cost**: semantic caching, prompt caching, model routing (small model for classification, large for synthesis), batch for offline jobs.

## 3. Rapid-fire Q&A (≈40)
**LLM basics**
1. *Tokens and context window?* Text units the model processes. The context window caps input + output. Cost and latency scale with tokens.
2. *Temperature / top-p?* Sampling randomness. Note: current Claude models (Opus/Sonnet 5.5) reject sampling params; you control behaviour via prompts and **effort**.
3. *Hallucination: causes and fixes?* The model fills gaps from its priors. Fixes: grounding (RAG), "say I don't know", citations, constrained outputs, faithfulness evals, tool use for facts and arithmetic.
4. *Fine-tune vs RAG vs prompting?* Prompt first. RAG for knowledge (fresh, private, citable). Fine-tune for style/format or a smaller, cheaper model at scale, not for adding facts.
5. *Structured output?* Native JSON-schema mode + Pydantic validation + retry. Avoid regex parsing.
6. *Prompt caching?* Reuse a stable prefix (system prompt, tools, documents) across calls for big cost and latency savings. Keep volatile content at the end.

**RAG**
7. *Chunk size?* 300–800 tokens with 10–15% overlap as a start. Split on structure first. Tune with context-recall evals.
8. *Embeddings for Claude?* Claude has no embeddings API. Use Voyage (recommended by Anthropic), OpenAI or local BGE.
9. *Similarity vs MMR vs hybrid?* MMR adds diversity. Hybrid adds keyword precision for IDs, codes and names. A reranker boosts precision@k.
10. *Evaluate retrieval separately?* Yes: context precision/recall, hit rate, MRR. Generation: faithfulness, answer relevancy.
11. *Multi-hop questions?* Query decomposition, agentic RAG (retrieval as a tool, iterate), GraphRAG for entity-relationship data.
12. *Tables/PDFs?* Layout-aware parsers, keep tables as markdown/HTML chunks, add table summaries as extra chunks.
13. *Access control?* ACL metadata per chunk, filter at query time. Never rely on the LLM to hide data.
14. *Vector DB choice?* Chroma (dev), FAISS (in-process speed), pgvector (already on Postgres), Pinecone (managed serverless), Weaviate (hybrid, self-host).

**Agents**
15. *What makes something an agent?* An LLM choosing actions in a loop with tools and state until a goal is met.
16. *ReAct?* Reason → act → observe loop. Modern tool-calling APIs implement it natively.
17. *Why LangGraph over a plain agent executor?* Explicit state machine, cycles, branching, persistence (checkpointer), streaming, HITL (`interrupt`), time travel, production control.
18. *`create_agent` vs StateGraph?* `create_agent` (LangChain 1.0) is the standard tool loop with middleware (HITL, summarization, PII, limits). Use StateGraph for custom topologies.
19. *Memory types?* Short-term (thread state via checkpointer), long-term (store / vector DB per user), episodic (past runs), semantic (facts).
20. *Tool design best practices?* Few, well-named tools with clear descriptions and typed args. Return concise, citable results. Make writes idempotent. Least privilege.
21. *Stop runaway agents?* Recursion/step limits, tool-call limits, timeouts, budgets (e.g. a Managed Agents session budget), loop detection in traces.
22. *Multi-agent: when?* Different tools/prompts/context per subtask, or parallelism. Otherwise a single agent is cheaper and simpler.
23. *CrewAI vs AutoGen vs LangGraph?* Role/task crew vs conversational group chat vs explicit graph. AutoGen is now succeeded by Microsoft Agent Framework.
24. *MCP?* Model Context Protocol: an open standard for exposing tools and data to any agent/LLM client. Write a tool server once, use it everywhere.
25. *Claude Managed Agents?* Anthropic-hosted harness and sandbox: agent (versioned config) + environment + session + events. Good for long-running, file-producing, async jobs. Not ZDR-eligible.

**Observability & evals**
26. *What do you trace?* Every LLM call (prompt, output, model, tokens, cost, latency), tool calls, retriever results, user/session ids, errors.
27. *Key production metrics?* p50/p95 latency, time-to-first-token, cost per request/user, error and refusal rate, tool failure rate, eval scores, feedback rate.
28. *Langfuse vs LangSmith?* Langfuse: MIT, self-hostable, OTel, framework-agnostic. LangSmith: deepest LangGraph integration, SaaS (self-host is enterprise-only).
29. *Offline vs online evals?* Offline: golden set before release (CI gate). Online: judge a sample of production traces continuously and alert on drops.
30. *LLM-as-judge risks?* Bias (position, verbosity, self-preference) and variance. Use rubrics, binary scales, references, a different judge model, and human calibration.
31. *Agent evals?* Final answer + trajectory (right tools/order/args) + safety (approvals, injection) + cost/latency per task.
32. *How do you build a golden dataset?* Start with 20–50 real questions with SME answers. Add failures from production traces. Version it in Langfuse/git.

**Production, cloud & integration**
33. *Deploying an LLM service?* Containerised API, secrets manager, autoscaling, persistent checkpointer/vector store, rate limits, retries/backoff, timeouts, health checks, CI eval gate.
34. *Azure OpenAI vs OpenAI vs Bedrock vs Vertex?* Same models behind enterprise controls: VNet/private endpoints, data residency, compliance, billing on the existing cloud contract. Abstract providers behind one interface (`common/llm.py`).
35. *Cost reduction?* Prompt caching, smaller models for routing/classification, shorter context (better retrieval), batch APIs, semantic cache, cap output tokens.
36. *Security?* Prompt-injection defences, least-privilege tools, human approval for writes, PII masking in logs, output validation, secrets out of prompts.
37. *RPA + GenAI?* RPA handles deterministic UI/system steps, GenAI handles judgement (classify, extract, decide). Integrate over REST (UiPath HTTP Request, Power Automate HTTP/custom connector) and route approvals to Action Center or Teams.
38. *How do you handle LLM API outages?* Retries with backoff, model fallback (middleware or the provider's fallback), graceful degradation, queue for async jobs.
39. *Streaming?* Server-sent events to the UI. LangGraph `stream_mode="messages"` for tokens or `"updates"` for node progress.
40. *Versioning prompts?* Prompt management (Langfuse) with labels (production/staging), linked to traces and evals, so you can roll back without a deploy.

## 4. Questions to ask them
- What's the split between GenAI app work and RPA/automation work on the team?
- Which LLM platforms are in production today (Azure OpenAI / Bedrock / Claude), and what's the governance process?
- How do you evaluate and monitor GenAI solutions after go-live?
- What does a typical client engagement look like from PoC to production?
