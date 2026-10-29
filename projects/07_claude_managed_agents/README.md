# 07 · Claude Managed Agents (45 min)

**You'll learn:** Anthropic's hosted agent harness (beta `managed-agents-2026-04-01`): agents, environments, sessions, events; custom tools served from your own code; sandbox outputs; session budgets; and when to choose it over a self-built LangGraph agent.

## The four concepts
| Concept | What it is | In this project |
|---|---|---|
| **Agent** | Persisted, *versioned* config: model, system prompt, tools, MCP servers, skills | `config.py → AGENT`, created once by `setup_agent.py` |
| **Environment** | Where tools run: Anthropic cloud sandbox (networking policy, packages) or self-hosted | `config.py → ENVIRONMENT` (limited egress, pip allowed) |
| **Session** | One running agent instance with its own container, filesystem and history | `run_session.py`, pinned to an agent version, with a $2 budget |
| **Events** | SSE stream both ways: `user.message`, `agent.message`, `agent.custom_tool_use`, `session.status_idle`… | the `run()` loop |

## Run
```bash
pip install anthropic python-dotenv      # already in requirements.txt
python projects/07_claude_managed_agents/setup_agent.py --print   # inspect config, no API call
python projects/07_claude_managed_agents/setup_agent.py           # creates env + agent, saves IDs
python projects/07_claude_managed_agents/run_session.py           # streams the run, downloads outputs/
```
Open the printed Console link to watch the session's transcript, tool calls and cost live.

## What happens
1. Claude, running in Anthropic's harness, plans the task.
2. It calls **`search_acme_policies`**, a *custom tool*. The session pauses and your machine runs the project-02 RAG pipeline and sends back the excerpts. Private data stays on your side.
3. It uses the **built-in sandbox tools** (bash, file write) to create `new_joiner_faq.md` and `key_numbers.csv` in `/mnt/session/outputs/`.
4. The script downloads them to `projects/07_claude_managed_agents/outputs/`.

## Managed Agents vs LangGraph (decision rule)
| Choose **Managed Agents** when | Choose **LangGraph (self-hosted)** when |
|---|---|
| Long-running, async tasks (minutes to hours), file and code work in a sandbox | You need custom control flow: branches, parallel nodes, multi-agent graphs |
| You don't want to build or operate the loop, sandbox, compaction and caching | Multi-provider (OpenAI/Azure/Bedrock/DO) or on-prem model requirement |
| Versioned agent configs, scheduled runs, built-in session budgets | Zero Data Retention or HIPAA is required (Managed Agents is not ZDR-eligible: sessions are stateful) |
| Fast path to production for a Claude-only team | Deep integration with existing Python services and your own checkpointer |

Many teams use both: LangGraph for the synchronous chat/RAG product, Managed Agents for background "do a job and hand me files" work.

## Production notes
- **Create agents once** (setup), store IDs, and **update** to get a new version; sessions pin a version. Anthropic also offers `ant apply` to keep agent configs as files in git.
- **Credentials**: use **vaults** (`environment_variable` credentials are substituted at egress and never visible in the sandbox), or keep secrets host-side behind a custom tool as done here.
- **Least privilege**: limited networking, web tools off unless needed, `auto` permission policy plus human approval for anything that asks; deny when unattended.
- **Cost guardrails**: `budget` on the session (a hard cap at list price; the session pauses at `budget_reached`).
- **Reconnects**: SSE has no replay. On reconnect, fetch the event list, dedupe by id, and answer any pending tool call, or the session waits forever.

## Things to try
1. Run with `--task "Compare the leave and expense policies and write a risks.md"`.
2. Edit the system prompt in `config.py`, run `setup_agent.py --update`, and compare v1 vs v2 sessions in the Console.
3. Set `--budget-cents 20` and watch the session pause at `budget_reached`.
