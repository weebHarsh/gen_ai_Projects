"""CrewAI: role-based agents (role/goal/backstory) executing Tasks in a Process.

    pip install -r projects/04_multi_agent/requirements-agents.txt   # separate venv recommended
    python projects/04_multi_agent/crewai_crew.py

Same task in all three frameworks: research Acme's remote-work policy and write a
new-joiner FAQ, then review it.
"""
from __future__ import annotations

import os
from pathlib import Path

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
KB = ROOT / "data" / "acme_kb"


@tool("read_policy")
def read_policy(filename: str) -> str:
    """Read an Acme policy file. Available: leave_policy.md, remote_work_policy.md,
    expense_policy.md, it_security_policy.md, it_support.md, onboarding.md, benefits.md,
    ai_usage_guidelines.md"""
    path = KB / Path(filename).name  # .name blocks path traversal
    return path.read_text() if path.exists() else f"No such policy: {filename}"


llm = LLM(model=f"anthropic/{os.getenv('LLM_MODEL') or 'claude-opus-5-5'}", max_tokens=4096)

researcher = Agent(
    role="HR Policy Researcher",
    goal="Extract exact, citable facts from Acme policies",
    backstory="You never state a fact you didn't read in a policy file, and you cite the file.",
    tools=[read_policy], llm=llm, max_iter=5,
)
writer = Agent(
    role="Internal Comms Writer",
    goal="Turn policy facts into a friendly, accurate FAQ for new joiners",
    backstory="You write short, scannable FAQs. You never add facts that weren't in the research.",
    llm=llm,
)
reviewer = Agent(
    role="Compliance Reviewer",
    goal="Catch any claim in the FAQ not supported by the research notes",
    backstory="Strict but brief. You return the corrected FAQ.",
    llm=llm,
)

research = Task(
    description="Research Acme's hybrid/remote work policy and onboarding rules relevant to new joiners.",
    expected_output="Bullet list of facts, each ending with its source file.",
    agent=researcher,
)
write = Task(
    description="Write a 5-question FAQ (max 150 words) for new joiners using only the research.",
    expected_output="Markdown FAQ.", agent=writer, context=[research],
)
review = Task(
    description="Check every claim in the FAQ against the research. Fix or remove unsupported claims.",
    expected_output="Final markdown FAQ plus a one-line review note.", agent=reviewer, context=[research, write],
)

if __name__ == "__main__":
    crew = Crew(agents=[researcher, writer, reviewer], tasks=[research, write, review],
                process=Process.sequential, verbose=True)
    print(crew.kickoff())
