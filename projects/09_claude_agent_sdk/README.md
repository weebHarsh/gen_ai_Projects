# 09 · Claude Agent SDK (45 min)

**You'll learn:** how to embed the Claude Code agent harness in your own Python program: `query()`, `ClaudeAgentOptions`, built-in file tools, permission modes, a `can_use_tool` callback plus `PreToolUse` hook to fence writes, in-process custom tools (SDK MCP server), and subagents on a cheaper model.

## Concepts
| Concept | What it is | In this project |
|---|---|---|
| **`query(prompt=, options=)`** | Async generator that runs the agent loop and streams messages | `run()` in `main.py` |
| **`ClaudeAgentOptions`** | `model`, `system_prompt`, `tools`, `allowed_tools`, `permission_mode`, `cwd`, `max_turns`, `mcp_servers`, `agents`, `hooks`, `can_use_tool`, `setting_sources` | `build_options()` |
| **Messages** | `AssistantMessage` (`TextBlock`, `ToolUseBlock`) then one final `ResultMessage` (`total_cost_usd`, `num_turns`, `usage`) | printed as text, `[tool] ...`, `[done] ...` |
| **Permissions** | `tools` = what exists; `allowed_tools` = auto-approved; `permission_mode` (`dontAsk` denies the rest); `can_use_tool` = your approval code | `ask` (read-only), `review` (write only REVIEW.md) |
| **Custom tool** | `@tool` + `create_sdk_mcp_server`, runs in your process; exposed as `mcp__<server>__<tool>` | `search_acme_policies` over project 02 BM25 |
| **Subagent** | `agents={"name": AgentDefinition(description, prompt, tools, model)}`, spawned by the `Agent` tool | `file-finder` on Haiku, lead on Opus |

Models: `claude-opus-5-5` (complex), `claude-sonnet-5-5` (default), `claude-haiku-5-5` (cheap); override with `MODEL_COMPLEX/DEFAULT/CHEAP`. Auth: `ANTHROPIC_API_KEY` (env or `.env`).

## Run
```bash
pip install claude-agent-sdk python-dotenv
export ANTHROPIC_API_KEY=sk-ant-...
python projects/09_claude_agent_sdk/main.py ask "How does the RETRIEVER switch work?"
python projects/09_claude_agent_sdk/main.py review              # writes ./REVIEW.md only
python projects/09_claude_agent_sdk/main.py custom-tool "How much parental leave do I get?"
python projects/09_claude_agent_sdk/main.py team "Where is the BM25 retriever defined?"
python projects/09_claude_agent_sdk/main.py ask "anything" --dry-run   # print options, no API call
```
Output: streamed text, `[tool] Grep(pattern='RETRIEVER')` lines, and a final `[done] turns=.. cost=$..` line.

## Which one should I use?
| | Messages API tool runner | **Claude Agent SDK** | Managed Agents (project 07) |
|---|---|---|---|
| What you get | A loop helper around the API; you write every tool | The Claude Code **harness** as a library | Harness **plus** a hosted sandbox |
| Who hosts the loop | Your code | **You** (it spawns the Claude Code runtime as a subprocess) | Anthropic |
| Built-in file/bash tools | None | Read, Write, Edit, Grep, Glob, Bash, ... on **your** machine/container | Yes, inside Anthropic's sandbox |
| Where tools execute | Your process | Your host filesystem (you own the blast radius) | Anthropic's container |
| Runtime needs | Any Python process | A normal long-lived process with a filesystem; **not serverless** | None on your side |
| Custom tools | Functions you pass | `@tool` in-process MCP, or external MCP | Custom tools served from your code |
| Subagents, hooks, permissions | DIY | Built in | Built in (server side) |
| Choose when | Simple tool use, full control | Coding/repo/file agents inside your infra | Long async jobs, no infra to run |

## Not deployed on the Vercel demo
This project is **not** part of the Vercel demo: the SDK launches the Claude Code runtime as a child process and its tools read and write a real filesystem, which serverless functions do not provide (short-lived, read-only, size-limited, no persistent subprocess). Run it locally, in a container, or on a VM/Droplet (see project 08 for that deployment pattern).

## Safety notes
- `ask` exposes only `Read/Grep/Glob` and `dontAsk`, so any other tool is denied.
- `review` keeps `Write` out of `allowed_tools` on purpose: an allowed tool is auto-approved before `can_use_tool` runs. The callback and a `PreToolUse` hook both reject any path except `<repo>/REVIEW.md`. Treat this as a guardrail, not a sandbox: for hard isolation add the SDK's `sandbox` option or run in a container.
- `max_turns` caps cost; `ResultMessage.total_cost_usd` reports it. `max_budget_usd` is also available.

## Verified vs. unverified
Option construction, the tool/server definitions, the write guard and `--dry-run` are tested offline (`tests/test_agent_sdk_project.py`) against the installed SDK (0.2.x). Real agent runs need an API key and were not exercised in the sandbox this was written in.

## Things to try
1. Run `ask` with and without `--dry-run`; change `allowed_tools` to add `Bash` and see what the agent can now do (and why you probably should not).
2. In `review`, ask the agent (via a prompt tweak) to also write `NOTES.md` and watch the deny message.
3. Remove `agents` from `team` and compare cost: the Haiku file-finder should cut tokens on the expensive model.
4. Add a second `@tool` (e.g. `list_policies`) to the `acme` server and allow it.

## Interview questions
1. What is the difference between `tools`, `allowed_tools`, `disallowed_tools` and `permission_mode`?
2. Why does `can_use_tool` not fire for tools in `allowed_tools`, and when do you use a `PreToolUse` hook instead?
3. How is an SDK MCP server different from a stdio MCP server? What are the trade-offs?
4. Why can't the Agent SDK run in a Vercel function, and what would you deploy it on?
5. Agent SDK vs Managed Agents vs a hand-rolled tool loop: where does each run, and who secures the filesystem?
6. How do subagents save money, and what context do they not share with the parent?
