"""03 - Chat with the LangGraph helpdesk agent (memory + human approval).

    python projects/03_langgraph_agent/main.py                       # interactive chat
    python projects/03_langgraph_agent/main.py --once "How many sick days do I get?"
    python projects/03_langgraph_agent/main.py --once "VPN is down for my team, raise a ticket" --auto-approve

Try in one chat session: "I've worked 7 months and taken 5 days, I carried 3 days. Balance?"
then "And how many can I carry into next year?" - the second question relies on memory.
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.observability import flush  # noqa: E402
from graph import build_graph, run_turn  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", help="ask a single question and exit")
    parser.add_argument("--auto-approve", action="store_true", help="approve tool calls without asking")
    parser.add_argument("--thread", default=None, help="thread id (reuse to continue a conversation)")
    args = parser.parse_args(argv)

    graph = build_graph()
    thread_id = args.thread or f"cli-{uuid.uuid4().hex[:8]}"
    approve = True if args.auto_approve else None
    if args.once:
        print(run_turn(graph, args.once, thread_id, approve))
    else:
        print(f"Helpdesk agent (thread {thread_id}). Ctrl+C to quit.")
        try:
            while True:
                print("\nagent>", run_turn(graph, input("\nyou> "), thread_id, approve))
        except (KeyboardInterrupt, EOFError):
            pass
    flush()


if __name__ == "__main__":
    main()
