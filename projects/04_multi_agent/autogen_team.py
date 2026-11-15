"""AutoGen (AgentChat 0.7): conversational agents that take turns in a team chat.

    pip install -r projects/04_multi_agent/requirements-agents.txt
    python projects/04_multi_agent/autogen_team.py

Note: Microsoft has merged AutoGen and Semantic Kernel into the **Microsoft Agent Framework**
(`pip install agent-framework`); AutoGen 0.7 is in maintenance mode. The concepts (agents,
teams, termination conditions) carry over, which makes this a useful interview talking point.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
KB = ROOT / "data" / "acme_kb"


async def read_policy(filename: str) -> str:
    """Read an Acme policy file, e.g. remote_work_policy.md or onboarding.md."""
    path = KB / Path(filename).name
    return path.read_text() if path.exists() else f"No such policy: {filename}"


async def main() -> None:
    client = AnthropicChatCompletionClient(
        model=os.getenv("LLM_MODEL") or "claude-opus-5-5",
        max_tokens=4096,
        # AutoGen needs capabilities declared for model ids it doesn't know yet.
        model_info={"vision": True, "function_calling": True, "json_output": True,
                    "family": "claude", "structured_output": True, "multiple_system_messages": False},
    )
    researcher = AssistantAgent(
        "researcher", model_client=client, tools=[read_policy],
        system_message="Read remote_work_policy.md and onboarding.md and list the key facts with sources.",
    )
    writer = AssistantAgent(
        "writer", model_client=client,
        system_message="Write a 5-question FAQ (max 150 words) for new joiners using only the researcher's facts.",
    )
    reviewer = AssistantAgent(
        "reviewer", model_client=client,
        system_message="Check the FAQ against the facts. Fix issues, output the final FAQ, then say APPROVED.",
    )
    team = RoundRobinGroupChat(
        [researcher, writer, reviewer],
        termination_condition=TextMentionTermination("APPROVED") | MaxMessageTermination(8),
    )
    await Console(team.run_stream(task="Create a remote-work FAQ for Acme new joiners."))
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
