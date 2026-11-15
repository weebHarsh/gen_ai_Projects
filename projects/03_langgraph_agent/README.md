# 03 · LangGraph Agent: Tools, Memory, Human-in-the-Loop (90 min)

**You'll learn:** StateGraph, nodes/edges, conditional routing, ToolNode, checkpointers (memory per thread), `interrupt()` for approvals, and LangChain 1.x `create_agent` with middleware.

```
START ─► agent (Claude + tools) ──tool_calls?──► tools (search_policies | leave_balance | create_it_ticket)
            ▲   │ no                                   │
            │   ▼                                      │ create_it_ticket ─► interrupt() ─► human y/n ─► REST API
            │  END                                     │
            └──────────────────────────────────────────┘
```

## Files
| File | What it shows |
|---|---|
| `tools.py` | 3 tools: RAG (read), calculator (deterministic), REST ticket API (write + approval) |
| `graph.py` | Hand-built `StateGraph`. Run it to print a Mermaid diagram. |
| `main.py` | Chat CLI with thread memory and an approval prompt (`Command(resume=...)`) |
| `create_agent_version.py` | Same agent via `create_agent` + `HumanInTheLoopMiddleware` + `ModelCallLimitMiddleware` |

## Run
```bash
python projects/03_langgraph_agent/main.py
# you> I've worked 7 months, taken 5 days and carried 3. What's my balance?
# you> How many of those can I carry into next year?          <- uses memory
# you> My whole team's VPN is down, raise a ticket             <- approval prompt
python projects/03_langgraph_agent/graph.py                    # Mermaid diagram
python projects/03_langgraph_agent/create_agent_version.py "Raise a P2 ticket: VPN down for 6 people"
```

## Concepts in 60 seconds
| Concept | What to say |
|---|---|
| Agent vs chain | A chain is a fixed sequence. An agent lets the LLM choose the next action (tool) in a loop until done. Use agents only when the path can't be specified in advance. |
| ReAct loop | Reason → act (tool) → observe → repeat. Here it's the `agent ⇄ tools` cycle. |
| Why LangGraph | Explicit state machine: cycles, branching, parallel nodes, persistence, streaming, human-in-the-loop, time travel. Production control that a black-box agent lacks. |
| Checkpointer | Saves state after every step, keyed by `thread_id`. Gives conversation memory, crash recovery, and pause/resume. `InMemorySaver` for dev, `PostgresSaver`/`RedisSaver` for prod. |
| interrupt() | Pauses the graph and saves state. You resume with `Command(resume=value)`. Use it for approvals of irreversible actions (Acme's AI policy requires this). |
| create_agent | LangChain 1.0's standard agent (replaced `create_react_agent`), built on LangGraph. Middleware hooks cover HITL, summarization, PII redaction, call limits, retries and model fallback. |
| Guardrails | Tool allow-lists, call limits, input/output validation, approval steps, least-privilege API keys, timeouts. |

## Things to try
1. Add a `get_ticket_status(ticket_id)` tool.
2. Swap `InMemorySaver` for `SqliteSaver` (`pip install langgraph-checkpoint-sqlite`) and restart the CLI with the same `--thread`: memory survives.
3. Add `SummarizationMiddleware` to the `create_agent` version for long chats.
4. Make a router node that sends HR questions and IT questions to different system prompts.

## Interview questions
- *How do you stop an agent looping forever or overspending?* Recursion/step limits (`recursion_limit`, `ModelCallLimitMiddleware`), tool call limits, timeouts, budget tracking in traces.
- *How do you add memory?* Short-term: checkpointer per thread. Long-term: a store (LangGraph `Store`, vector DB) keyed by user, written by the agent deliberately.
- *When would you NOT use an agent?* When the steps are known: use a workflow/chain. It's cheaper, faster, more predictable and easier to test.
