# 04 · Multi-Agent: CrewAI vs AutoGen vs LangGraph (60 min)

**You'll learn:** the three mainstream multi-agent styles by building the **same task** in each: research Acme's remote-work policy → write a new-joiner FAQ → review it.

| File | Framework | Style | Runs in |
|---|---|---|---|
| `langgraph_supervisor.py` | LangGraph | Supervisor graph with explicit state, routing and step limit | main venv |
| `crewai_crew.py` | CrewAI 1.x | Role-based: Agent(role, goal, backstory) + Task + sequential Process | agents venv |
| `autogen_team.py` | AutoGen AgentChat 0.7 | Conversational: agents take turns in a team chat until a termination condition | agents venv |

```bash
python projects/04_multi_agent/langgraph_supervisor.py
# CrewAI / AutoGen pin many deps; use a separate venv
python -m venv .venv-agents && .venv-agents/bin/pip install -r projects/04_multi_agent/requirements-agents.txt
.venv-agents/bin/python projects/04_multi_agent/crewai_crew.py
.venv-agents/bin/python projects/04_multi_agent/autogen_team.py
```

## How to compare them (interview answer)
| | LangGraph | CrewAI | AutoGen / MS Agent Framework |
|---|---|---|---|
| Mental model | State machine / graph | A team of roles with tasks | A group chat |
| Control | Highest: explicit edges, state, limits | Medium: process types (sequential, hierarchical) | Lower: emergent turn-taking; termination conditions |
| Persistence & HITL | Built in (checkpointer, `interrupt`) | Basic (memory, human input) | Basic (UserProxy / handoffs) |
| Best for | Production workflows that need determinism, auditability and resumability | Fast prototypes of role-based pipelines | Research, code-gen and debate-style collaboration |
| 2026 note | v1.x stable, no breaking changes until 2.0 | 1.x, native LLM providers (no LiteLLM required) | AutoGen is in maintenance; successor is **Microsoft Agent Framework** (merged with Semantic Kernel) |

## When is multi-agent worth it?
Use it when sub-tasks need **different tools, prompts or context windows**, or can run **in parallel**. Otherwise one agent with good tools is cheaper and easier to debug. Every hand-off loses context and adds latency and cost. Always set step/message limits (`MAX_STEPS`, `MaxMessageTermination`, `max_iter`).

## Things to try
1. In the supervisor, add a `reviewer` node and make the supervisor route to it before FINISH.
2. In CrewAI, switch to `Process.hierarchical` with a `manager_llm`.
3. Trace the LangGraph version in Langfuse (project 05) and count LLM calls per run.

## Interview questions
- *Supervisor vs swarm/handoff?* Supervisor: one router controls everything (predictable). Handoff/swarm: agents transfer control directly (more flexible, harder to debug).
- *How do agents share state?* LangGraph: a typed shared state with reducers (`operator.add` on `log`). CrewAI: task `context`. AutoGen: the shared chat transcript.
- *How do you prevent agents agreeing on a hallucination?* Ground workers in tools/RAG, add a reviewer that checks against sources, and evaluate faithfulness (project 06).
