# Responsible AI Usage Guidelines (AI-POL-001)
Effective: 1 June 2026. Owner: AI Council.

- All production LLM applications must log prompts and responses to the approved observability platform (Langfuse) with PII masked.
- RAG applications must show source citations for every answer.
- Every AI agent that can take actions (create tickets, send email, update records) must have a **human approval step** for irreversible actions.
- New GenAI use cases are reviewed by the AI Council; turnaround is **10 business days**.
- Evaluation: a use case must reach **faithfulness >= 0.85** on its golden dataset before going live.
- Models approved for confidential data: Claude (via company workspace), Azure OpenAI (company tenant). Public chatbots are not approved.
