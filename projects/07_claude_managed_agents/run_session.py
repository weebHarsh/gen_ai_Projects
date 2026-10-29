"""Start a Managed Agents session, stream its events, serve the custom RAG tool, download outputs.

    python projects/07_claude_managed_agents/run_session.py
    python projects/07_claude_managed_agents/run_session.py --task "Summarise the expense policy into /mnt/session/outputs/expenses.md"

Flow:  session.create -> open SSE stream -> send user.message -> loop over events:
         agent.message            -> print text
         agent.custom_tool_use    -> run our RAG locally, send user.custom_tool_result
         agent.tool_use (ask)     -> human approves/denies built-in tool calls
         session.status_idle      -> done (unless it's waiting on us)
       -> list + download files the agent wrote to /mnt/session/outputs/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "projects" / "02_rag_basics"))
load_dotenv(ROOT / ".env")

from config import DEFAULT_TASK  # noqa: E402

OUTPUT_DIR = HERE / "outputs"


def search_acme_policies(query: str) -> str:
    from rag_pipeline import format_context, get_retriever

    return format_context(get_retriever().invoke(query))


def approve(event, interactive: bool) -> bool:
    """Built-in tool calls the server couldn't auto-approve land here. Deny when unattended."""
    if not interactive:
        return False
    print(f"\n[approval] {event.name} {json.dumps(getattr(event, 'input', {}))[:300]}")
    return input("allow? (y/n) ").strip().lower().startswith("y")


def run(client, session_id: str, task: str, interactive: bool) -> None:
    # Stream-first: open the stream, THEN send, so no early events are missed.
    with client.beta.sessions.events.stream(session_id=session_id) as stream:
        client.beta.sessions.events.send(
            session_id=session_id,
            events=[{"type": "user.message", "content": [{"type": "text", "text": task}]}],
        )
        for event in stream:
            if event.type == "agent.message":
                for block in event.content:
                    if block.type == "text":
                        print(block.text, end="", flush=True)
            elif event.type == "agent.tool_use":
                print(f"\n  [sandbox tool] {event.name}", flush=True)
                if event.evaluated_permission == "ask":
                    client.beta.sessions.events.send(session_id=session_id, events=[{
                        "type": "user.tool_confirmation", "tool_use_id": event.id,
                        "result": "allow" if approve(event, interactive) else "deny",
                    }])
            elif event.type == "agent.custom_tool_use":
                print(f"\n  [custom tool] {event.name}({event.input})", flush=True)
                result = (search_acme_policies(**event.input) if event.name == "search_acme_policies"
                          else f"Unknown tool {event.name}")
                client.beta.sessions.events.send(session_id=session_id, events=[{
                    "type": "user.custom_tool_result", "custom_tool_use_id": event.id,
                    "content": [{"type": "text", "text": result}],
                }])
            elif event.type == "session.status_idle":
                if event.stop_reason.type != "requires_action":  # requires_action = waiting on us
                    print(f"\n--- idle ({event.stop_reason.type}) ---")
                    return
            elif event.type == "session.status_terminated":
                print("\n--- terminated ---")
                return


def download_outputs(client, session_id: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for attempt in range(3):  # outputs are indexed a second or two after idle
        files = client.beta.files.list(scope_id=session_id, betas=["managed-agents-2026-04-01"]).data
        if files:
            break
        time.sleep(2)
    for f in files:
        client.beta.files.download(f.id).write_to_file(OUTPUT_DIR / f.filename)
        print(f"downloaded {f.filename} ({f.size_bytes} bytes) -> {OUTPUT_DIR.relative_to(ROOT)}/")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--budget-cents", default="200", help="hard spend cap for the session (list price, USD cents)")
    parser.add_argument("--unattended", action="store_true", help="deny any tool call that needs approval")
    args = parser.parse_args(argv)

    import anthropic

    ids_file = HERE / "managed_agent_ids.json"
    if not ids_file.exists():
        sys.exit("Run setup_agent.py first.")
    ids = json.loads(ids_file.read_text())

    client = anthropic.Anthropic()
    session = client.beta.sessions.create(
        agent={"type": "agent", "id": ids["agent_id"], "version": ids["agent_version"]},  # pin the version
        environment_id=ids["environment_id"],
        title="Acme new-joiner FAQ",
        budget={"type": "limit", "max_list_cost": {"amount": args.budget_cents, "currency": "USD"}},
    )
    print(f"Session {session.id}\nWatch it live: https://platform.claude.com/workspaces/default/sessions/{session.id}\n")
    run(client, session.id, args.task, interactive=not args.unattended)
    download_outputs(client, session.id)
    usage = client.beta.sessions.retrieve(session_id=session.id).usage
    print("usage:", usage)


if __name__ == "__main__":
    main()
