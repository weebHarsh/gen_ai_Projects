"""Project 09: Claude Agent SDK - a "repo analyst" CLI.

The Agent SDK is the Claude Code harness (agent loop, Read/Grep/Glob/Write/Bash tools, permissions,
subagents, hooks) as a Python library. You host it; it spawns the bundled Claude Code runtime as a
subprocess, so it needs a normal long-lived process (not serverless).

    python projects/09_claude_agent_sdk/main.py ask "How does the RETRIEVER switch work?"
    python projects/09_claude_agent_sdk/main.py review
    python projects/09_claude_agent_sdk/main.py custom-tool "How many days of parental leave?"
    python projects/09_claude_agent_sdk/main.py team "Where is the BM25 retriever defined?"
    python projects/09_claude_agent_sdk/main.py ask "..." --dry-run     # no API call

Auth: ANTHROPIC_API_KEY in the environment (or .env). Model: ClaudeAgentOptions.model.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    query,
    tool,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_COMPLEX = os.getenv("MODEL_COMPLEX", "claude-opus-5-5")
MODEL_DEFAULT = os.getenv("MODEL_DEFAULT", "claude-sonnet-5-5")
MODEL_CHEAP = os.getenv("MODEL_CHEAP", "claude-haiku-5-5")

READ_ONLY_TOOLS = ["Read", "Grep", "Glob"]

ASK_PROMPT = (
    "You are a read-only code analyst for this repository. Answer questions about how the code "
    "works. Use Grep/Glob to find code and Read to confirm it. Cite every claim as path:line. "
    "If you cannot find evidence, say so instead of guessing. Never modify files."
)
REVIEW_PROMPT = (
    "You are a senior reviewer. Explore the repository (README, projects/*, common/, tests/) and "
    "write ONE file, REVIEW.md in the repository root, with sections: Architecture, Data flow, "
    "Risks (ranked, each with path:line), Suggested next steps. You may not write any other file."
)
POLICY_PROMPT = (
    "You answer questions about Acme Corp policies. Get every policy fact from the "
    "search_acme_policies tool and cite it like [leave_policy.md#3]. If the excerpts do not "
    "contain the answer, say you don't know. Never invent policy details."
)
TEAM_PROMPT = (
    "You are the lead analyst. Delegate file discovery to the 'file-finder' subagent (cheap and "
    "fast), then read the files it returns yourself and answer with path:line citations."
)


# --------------------------------------------------------------------------- permissions
def _is_review_path(path: str, cwd: Path = REPO_ROOT) -> bool:
    p = Path(path)
    p = p if p.is_absolute() else cwd / p
    return os.path.normpath(p) == str(cwd / "REVIEW.md")


async def review_can_use_tool(name: str, tool_input: dict[str, Any], context: Any):
    """can_use_tool callback: only Write to <repo>/REVIEW.md; read tools pass; all else denied.

    Caveat from the SDK: tools matched by `allowed_tools` are auto-approved BEFORE this callback
    runs (the SDK warns with CanUseToolShadowedWarning), so Write is deliberately NOT in
    allowed_tools. Read/Grep/Glob are, so they never reach here.
    """
    if name == "Write" and _is_review_path(str(tool_input.get("file_path", ""))):
        return PermissionResultAllow()
    return PermissionResultDeny(message="Only Read/Grep/Glob and Write to REVIEW.md are allowed.")


async def review_pretool_hook(input_data: dict[str, Any], tool_use_id: str | None, context: Any):
    """PreToolUse hook = second, harder layer: runs for every call regardless of allow rules."""
    if input_data.get("tool_name") == "Write":
        path = str(input_data.get("tool_input", {}).get("file_path", ""))
        if not _is_review_path(path):
            return {"hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"Writes are restricted to REVIEW.md (got {path!r}).",
            }}
    return {}


# --------------------------------------------------------------------------- custom tool
@tool(
    "search_acme_policies",
    "Search Acme Corp's internal HR/IT/expense/security/AI policies. "
    "Returns excerpts with [file#chunk] citation ids.",
    {"query": str},
)
async def search_acme_policies(args: dict[str, Any]) -> dict[str, Any]:
    """In-process tool: runs in YOUR Python process, reusing project 02's RAG retrieval (BM25)."""
    os.environ["RETRIEVER"] = "bm25"
    sys.path.insert(0, str(REPO_ROOT / "projects" / "02_rag_basics"))
    from rag_pipeline import format_context, retrieve  # lazy: heavy imports, not needed for dry-run

    docs = await asyncio.to_thread(retrieve, args["query"])
    return {"content": [{"type": "text", "text": format_context(docs)}]}


def policy_server():
    return create_sdk_mcp_server(name="acme", version="1.0.0", tools=[search_acme_policies])


# --------------------------------------------------------------------------- options
def build_options(command: str) -> ClaudeAgentOptions:
    base: dict[str, Any] = dict(cwd=str(REPO_ROOT), model=MODEL_DEFAULT, max_turns=15)
    if command == "ask":
        return ClaudeAgentOptions(
            **base, system_prompt=ASK_PROMPT, tools=READ_ONLY_TOOLS, allowed_tools=READ_ONLY_TOOLS,
            # dontAsk: anything not pre-approved is denied instead of prompting (no human here).
            permission_mode="dontAsk", setting_sources=[])
    if command == "review":
        return ClaudeAgentOptions(
            **{**base, "model": MODEL_COMPLEX, "max_turns": 30}, system_prompt=REVIEW_PROMPT,
            tools=[*READ_ONLY_TOOLS, "Write"], allowed_tools=READ_ONLY_TOOLS,  # Write NOT allowed here
            permission_mode="default", can_use_tool=review_can_use_tool, setting_sources=[],
            hooks={"PreToolUse": [HookMatcher(matcher="Write|Edit|MultiEdit",
                                              hooks=[review_pretool_hook])]})
    if command == "custom-tool":
        return ClaudeAgentOptions(
            **{**base, "max_turns": 6}, system_prompt=POLICY_PROMPT, tools=[],  # no built-ins at all
            mcp_servers={"acme": policy_server()}, allowed_tools=["mcp__acme__search_acme_policies"],
            permission_mode="dontAsk", setting_sources=[])
    if command == "team":
        return ClaudeAgentOptions(
            **{**base, "model": MODEL_COMPLEX}, system_prompt=TEAM_PROMPT,
            # The subagent-spawning tool is "Agent" in current runtimes ("Task" in older ones): list both.
            tools=[*READ_ONLY_TOOLS, "Agent", "Task"], allowed_tools=[*READ_ONLY_TOOLS, "Agent", "Task"],
            permission_mode="dontAsk", setting_sources=[],
            agents={"file-finder": AgentDefinition(
                description="Finds which files are relevant to a question. Use first, before reading.",
                prompt="Locate relevant files with Glob/Grep. Return a short list of paths with a "
                       "one-line reason each. Do not analyse the code.",
                tools=["Glob", "Grep"], model=MODEL_CHEAP, maxTurns=8)})
    raise ValueError(command)


def print_options(command: str, opts: ClaudeAgentOptions, prompt: str) -> None:
    print(f"[dry-run] command={command} (no API call made)")
    print(f"  model:           {opts.model}")
    print(f"  tools:           {opts.tools}")
    print(f"  allowed_tools:   {opts.allowed_tools}")
    print(f"  permission_mode: {opts.permission_mode}")
    print(f"  cwd:             {opts.cwd}")
    print(f"  max_turns:       {opts.max_turns}")
    print(f"  mcp_servers:     {list(opts.mcp_servers) if isinstance(opts.mcp_servers, dict) else opts.mcp_servers}")
    print(f"  can_use_tool:    {opts.can_use_tool.__name__ if opts.can_use_tool else None}")
    print(f"  hooks:           {list(opts.hooks) if opts.hooks else None}")
    print(f"  subagents:       { {k: v.model for k, v in (opts.agents or {}).items()} or None }")
    print(f"  system_prompt:   {str(opts.system_prompt)[:100]}...")
    print(f"  prompt:          {prompt[:100]}")


# --------------------------------------------------------------------------- run
def brief(args: dict[str, Any], n: int = 80) -> str:
    text = ", ".join(f"{k}={str(v)[:50]!r}" for k, v in args.items())
    return text[:n]


async def run(prompt: str, opts: ClaudeAgentOptions) -> int:
    async for msg in query(prompt=prompt, options=opts):
        if isinstance(msg, AssistantMessage):
            who = "  (subagent)" if msg.parent_tool_use_id else ""
            for block in msg.content:
                if isinstance(block, TextBlock):
                    print(block.text)
                elif isinstance(block, ToolUseBlock):
                    print(f"[tool]{who} {block.name}({brief(block.input)})")
        elif isinstance(msg, ResultMessage):
            cost = f"${msg.total_cost_usd:.4f}" if msg.total_cost_usd is not None else "n/a"
            print(f"\n[done] turns={msg.num_turns} cost={cost} subtype={msg.subtype} "
                  f"duration={msg.duration_ms / 1000:.1f}s")
            return 1 if msg.is_error else 0
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Repo analyst built on the Claude Agent SDK")
    ap.add_argument("--dry-run", action="store_true", help="build and print options; no API call")
    sub = ap.add_subparsers(dest="command", required=True)
    for name, default in (("ask", None), ("custom-tool", "How many days of parental leave do employees get?"),
                          ("team", "Where is the BM25 retriever defined and how is it selected?")):
        p = sub.add_parser(name)
        p.add_argument("question", nargs="?" if default else None, default=default)
        p.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS)
    sub.add_parser("review").add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    prompt = getattr(args, "question", None) or "Review this repository and write REVIEW.md."
    opts = build_options(args.command)
    if args.dry_run:  # accepted before or after the subcommand
        print_options(args.command, opts, prompt)
        return 0
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set (use --dry-run to inspect options offline).", file=sys.stderr)
        return 2
    return asyncio.run(run(prompt, opts))


if __name__ == "__main__":
    sys.exit(main())
