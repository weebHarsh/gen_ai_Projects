"""Agent + environment definitions for Claude Managed Agents (beta: managed-agents-2026-04-01).

Kept as plain data so you can read them, diff them and version them like infrastructure.
"""
from __future__ import annotations

import os

ENVIRONMENT = {
    "name": "acme-helpdesk-env",
    "config": {
        "type": "cloud",  # Anthropic-hosted sandbox; "self_hosted" runs tools in your own infra
        # Deny-by-default egress. The agent can pip install, nothing else.
        "networking": {"type": "limited", "allow_package_managers": True},
    },
}

SEARCH_TOOL = {
    # A *custom* tool: the agent asks, YOUR code runs it (here: the project-02 RAG pipeline on
    # your machine) and returns the result. Private data never has to be uploaded to the sandbox.
    "type": "custom",
    "name": "search_acme_policies",
    "description": "Search Acme Corp's internal HR/IT/expense/security/AI policies. "
                   "Returns policy excerpts with [file#chunk] citation ids.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "What to look up"}},
        "required": ["query"],
    },
}

AGENT = {
    "name": "Acme Policy Analyst",
    "model": os.getenv("LLM_MODEL") or "claude-opus-5-5",
    "system": (
        "You are Acme Corp's policy analyst. Get every policy fact from search_acme_policies and "
        "cite it like [leave_policy.md#3]. Write deliverables as files in /mnt/session/outputs/. "
        "Never invent policy details."
    ),
    "tools": [
        {
            # Built-in sandbox tools: bash, file read/write/edit, glob, grep, web search/fetch.
            "type": "agent_toolset_20260401",
            "default_config": {"enabled": True, "permission_policy": {"type": "auto"}},
            # The job doesn't need the web: keep web tools off (least privilege).
            "configs": [
                {"name": "web_fetch", "enabled": False},
                {"name": "web_search", "enabled": False},
            ],
        },
        SEARCH_TOOL,
    ],
}

DEFAULT_TASK = (
    "Create a new-joiner handbook page. Research leave, hybrid work, expenses and IT security "
    "with search_acme_policies, then write /mnt/session/outputs/new_joiner_faq.md with 8 Q&As "
    "(each with a citation) and /mnt/session/outputs/key_numbers.csv with columns "
    "topic,rule,value,source."
)
