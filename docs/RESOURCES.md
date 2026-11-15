# Curated Resources (official docs first, then courses, then deep reads)

⏱ = do it today (1-day sprint), 📚 = after the interview.

## Foundations: LLMs & prompt engineering
- ⏱ [Claude prompt engineering guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview): structure, examples, XML tags, structured output
- ⏱ [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents): workflows vs agents and the core patterns (most-cited agent article in interviews)
- 📚 [Anthropic courses (GitHub)](https://github.com/anthropics/courses): prompt engineering, tool use, real-world prompting
- 📚 [OpenAI Cookbook](https://cookbook.openai.com): practical recipes, most of them provider-agnostic
- 📚 Chip Huyen, *AI Engineering* (O'Reilly, 2025): the best single book for this role

## RAG
- ⏱ [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval): contextual chunk headers + hybrid search + rerank (project 02 uses the header idea)
- ⏱ [LangChain RAG docs](https://docs.langchain.com/oss/python/langchain/rag)
- ⏱ [LlamaIndex docs](https://docs.llamaindex.ai): loaders, node parsers, query engines
- 📚 [Eugene Yan: Patterns for building LLM-based systems](https://eugeneyan.com/writing/llm-patterns/)
- Vector DBs: [Chroma](https://docs.trychroma.com) · [FAISS](https://github.com/facebookresearch/faiss) · [Pinecone](https://docs.pinecone.io) · [Weaviate](https://docs.weaviate.io) · [pgvector](https://github.com/pgvector/pgvector)

## Agents & orchestration
- ⏱ [LangGraph docs](https://docs.langchain.com/oss/python/langgraph/overview): state, nodes, edges, persistence, `interrupt`
- ⏱ [LangChain agents (`create_agent`, middleware)](https://docs.langchain.com/oss/python/langchain/agents)
- 📚 [LangGraph: custom agentic RAG](https://docs.langchain.com/oss/python/langgraph/agentic-rag)
- ⏱ [LangChain Academy: Intro to LangGraph (free)](https://academy.langchain.com/courses/intro-to-langgraph)
- 📚 [CrewAI docs](https://docs.crewai.com) · [AutoGen AgentChat](https://microsoft.github.io/autogen/stable/) · [Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/)
- 📚 [DeepLearning.AI short courses](https://www.deeplearning.ai/short-courses/): "AI Agents in LangGraph", "Multi AI Agent Systems with crewAI", "AI Agentic Design Patterns with AutoGen", "Evaluating AI Agents"
- 📚 [Model Context Protocol](https://modelcontextprotocol.io): the standard for connecting agents to tools and data

## Claude Managed Agents
- ⏱ [Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) → [Quickstart](https://platform.claude.com/docs/en/managed-agents/quickstart) → [Sessions](https://platform.claude.com/docs/en/managed-agents/sessions)
- 📚 [Tools & permission policies](https://platform.claude.com/docs/en/managed-agents/tools) · [Environments](https://platform.claude.com/docs/en/managed-agents/choosing-an-environment) · [Scheduled deployments](https://platform.claude.com/docs/en/managed-agents/scheduled-deployments)

## Observability
- ⏱ [Langfuse docs](https://langfuse.com/docs): tracing, sessions, prompt management, datasets, scores
- ⏱ [Langfuse + LangChain/LangGraph integration](https://langfuse.com/integrations/frameworks/langchain)
- 📚 [Langfuse self-hosting](https://langfuse.com/self-hosting) · [LangSmith](https://docs.smith.langchain.com) · [Arize Phoenix](https://arize.com/docs/phoenix) · [Opik](https://www.comet.com/docs/opik/)
- Landscape comparison: [`OBSERVABILITY_LANDSCAPE.md`](OBSERVABILITY_LANDSCAPE.md)

## Evals
- ⏱ [Hamel Husain: Your AI product needs evals](https://hamel.dev/blog/posts/evals/): the mindset interviewers look for
- ⏱ [Ragas docs](https://docs.ragas.io): faithfulness, context precision/recall
- ⏱ [DeepEval docs](https://deepeval.com/docs/getting-started): pytest-style tests, agent metrics, G-Eval
- 📚 [Anthropic: Define success criteria and build evaluations](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests)

## Cloud LLM platforms (named in the JD)
- [Anthropic Claude API](https://platform.claude.com/docs) · [OpenAI API](https://platform.openai.com/docs) · [Azure AI Foundry / Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-foundry/) · [AWS Bedrock](https://docs.aws.amazon.com/bedrock/) · [Google Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs)

## DigitalOcean
- ⏱ [App Platform docs](https://docs.digitalocean.com/products/app-platform/) · [App spec reference](https://docs.digitalocean.com/products/app-platform/reference/app-spec/)
- 📚 [DigitalOcean AI Platform (formerly Gradient AI)](https://docs.digitalocean.com/products/ai-platform/): agents, knowledge bases, serverless inference
- 📚 [Droplets](https://docs.digitalocean.com/products/droplets/) · [Managed PostgreSQL](https://docs.digitalocean.com/products/databases/postgresql/)

## Intelligent automation / RPA
- 📚 [UiPath docs](https://docs.uipath.com): HTTP Request activity, Action Center, AI Center
- 📚 [Power Automate docs](https://learn.microsoft.com/en-us/power-automate/): HTTP action, custom connectors, approvals
- 📚 [Automation Anywhere docs](https://docs.automationanywhere.com) · [Blue Prism docs](https://docs.blueprism.com)

## Reading order for today (≈ 90 min total, spread across the day)
1. Building effective agents (20 min) → 2. Contextual Retrieval (10 min) → 3. LangGraph overview (20 min) → 4. Your AI product needs evals (20 min) → 5. Managed Agents overview (10 min) → 6. Langfuse tracing quickstart (10 min)
