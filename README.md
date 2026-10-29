# GenAI Projects: 1-Day Interview Sprint

Hands-on mini projects for a **GenAI Engineer** role (Python · LLMs · prompt engineering · RAG · agentic AI · LangChain/LangGraph/LlamaIndex/CrewAI/AutoGen · vector DBs · REST/RPA integration), plus **AI observability**, **agent evals**, **Claude Managed Agents** and **deployment on DigitalOcean**.

All eight projects share one fictional company (**Acme Corp**) and one knowledge base (`data/acme_kb/`), so each builds on the last: prompt → RAG → agent → multi-agent → traces → evals → managed agent → deployed API.

**Live tracker + decision log:** https://claude.ai/artifact/6A9JXoaGLsyMuh7DhYMdP3 (tick off study blocks, keep notes, log ADRs; your progress is saved)

## The day (≈10.5 h, P0 first; drop P1 if short on time)
| Time | Block | Project | Priority |
|---|---|---|---|
| 0:00–0:30 | Setup + smoke tests | this README | P0 |
| 0:30–1:15 | Prompt engineering & structured output | [01_prompt_engineering](projects/01_prompt_engineering/) | P0 |
| 1:15–2:45 | RAG (Chroma/FAISS, citations, LlamaIndex) | [02_rag_basics](projects/02_rag_basics/) | P0 |
| 2:45–4:15 | LangGraph agent (tools, memory, approvals) | [03_langgraph_agent](projects/03_langgraph_agent/) | P0 |
| 4:15–5:15 | Observability with Langfuse | [05_observability_langfuse](projects/05_observability_langfuse/) | P0 |
| 5:15–6:15 | Evals (Ragas, DeepEval, Langfuse experiments) | [06_agent_evals](projects/06_agent_evals/) | P0 |
| 6:15–7:00 | Claude Managed Agents | [07_claude_managed_agents](projects/07_claude_managed_agents/) | P1 |
| 7:00–8:00 | Multi-agent: CrewAI vs AutoGen vs LangGraph | [04_multi_agent](projects/04_multi_agent/) | P1 |
| 8:00–9:15 | Deploy on DigitalOcean + RPA integration | [08_deploy_digitalocean](projects/08_deploy_digitalocean/) | P1 |
| 9:15–10:30 | Interview prep: pitch, system design, Q&A | [docs/INTERVIEW_PREP.md](docs/INTERVIEW_PREP.md) | P0 |

## Setup (30 min)
```bash
git clone https://github.com/weebHarsh/gen_ai_Projects && cd gen_ai_Projects
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add ANTHROPIC_API_KEY (and Langfuse keys for project 05)

# 1. Offline check: no keys, no network, ~5 s
LLM_PROVIDER=fake pytest -q tests/test_smoke.py

# 2. First live call
python projects/01_prompt_engineering/main.py zero_shot
```
Each project's README follows the same pattern: **what you'll learn → concepts in 60 seconds → run → things to try → interview questions**.

Extra environments (to keep dependency pins isolated):
- Evals: `pip install -r projects/06_agent_evals/requirements-evals.txt`
- CrewAI/AutoGen: a separate venv with `projects/04_multi_agent/requirements-agents.txt`
- LlamaIndex: `projects/02_rag_basics/requirements-llamaindex.txt`

## Switching models
Every project calls `common/llm.py → get_llm()`. Change provider with one env var:

| `LLM_PROVIDER` | Needs | Default model |
|---|---|---|
| `anthropic` (default) | `ANTHROPIC_API_KEY` | `claude-opus-5-5` (set `LLM_MODEL=claude-haiku-5-5` for cheap experiments) |
| `openai` | `OPENAI_API_KEY` | `gpt-5-mini` |
| `azure` | `AZURE_OPENAI_*`, `LLM_MODEL`=deployment | — |
| `do` | `DO_MODEL_ACCESS_KEY` (DigitalOcean serverless inference) | `llama3.3-70b-instruct` |
| `fake` | nothing | offline scripted model for tests |

## Repo map
```
common/            llm.py (provider switch) · embeddings.py · observability.py (Langfuse, no-op without keys)
data/acme_kb/      8 fictional policy docs = the RAG corpus and eval ground truth
projects/01..08    the mini projects (each with README + runnable CLI)
docs/              RESOURCES.md · INTERVIEW_PREP.md · OBSERVABILITY_LANDSCAPE.md
tests/             offline smoke tests for every project
```

## Gotchas found while building this (October 2026), good interview stories
- **Claude Opus 5.5 / Sonnet 5.5 reject forced `tool_choice` and sampling params** (`temperature`). Structured output therefore uses JSON-schema mode (`common/llm.structured`), and the Ragas judge strips `temperature` (`projects/06_agent_evals/judge.py`).
- **`ragas 0.4.3` breaks with `langchain-community 0.4.2`**, so the evals requirements pin `<0.4.2`. `langchain-community` itself is being sunset in favour of standalone integration packages.
- **DeepEval's `ToolCorrectnessMetric` defaults to an OpenAI judge.** Pass `model=` explicitly, or it fails without `OPENAI_API_KEY`.
- **AutoGen is in maintenance mode.** Microsoft Agent Framework is its successor. AutoGen also needs `model_info` for model ids it doesn't know.
- **DigitalOcean renamed "Gradient AI Platform" to "DigitalOcean AI Platform"** (April 2026). Serverless inference is OpenAI-compatible at `https://inference.do-ai.run/v1`.
