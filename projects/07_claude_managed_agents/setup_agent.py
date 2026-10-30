"""One-time setup: create the environment and the agent, save their IDs.

    python projects/07_claude_managed_agents/setup_agent.py            # creates, writes managed_agent_ids.json
    python projects/07_claude_managed_agents/setup_agent.py --update   # new agent *version* after editing config.py
    python projects/07_claude_managed_agents/setup_agent.py --print    # show the config, no API call

Agents are persistent, versioned objects. Create once, then reference the ID from every run.
Don't call agents.create() in your request path (you'd pile up orphaned agents).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
load_dotenv(HERE.parents[1] / ".env")

from config import AGENT, ENVIRONMENT  # noqa: E402

IDS_FILE = HERE / "managed_agent_ids.json"  # git-ignored


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update", action="store_true", help="push config.py changes as a new agent version")
    parser.add_argument("--print", action="store_true", help="print config only")
    args = parser.parse_args(argv)
    if args.print:
        print(json.dumps({"environment": ENVIRONMENT, "agent": AGENT}, indent=2))
        return

    import anthropic

    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY or an `ant auth login` profile
    ids = json.loads(IDS_FILE.read_text()) if IDS_FILE.exists() else {}

    if args.update:
        agent = client.beta.agents.update(agent_id=ids["agent_id"], version=ids["agent_version"], **AGENT)
        ids["agent_version"] = agent.version
        print(f"Updated agent {agent.id} -> version {agent.version}")
    else:
        if "environment_id" not in ids:
            env = client.beta.environments.create(**ENVIRONMENT)
            ids["environment_id"] = env.id
            print("Created environment", env.id)
        if "agent_id" not in ids:
            agent = client.beta.agents.create(**AGENT)
            ids |= {"agent_id": agent.id, "agent_version": agent.version}
            print(f"Created agent {agent.id} (version {agent.version})")
    IDS_FILE.write_text(json.dumps(ids, indent=2))
    print(f"Saved IDs to {IDS_FILE.name}")


if __name__ == "__main__":
    main()
