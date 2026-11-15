# 5-minute live demo script

**Live site:** https://genai-showcase-weebharshs-projects.vercel.app  ·  **Code:** https://github.com/weebHarsh/gen_ai_Projects  ·  **Tracker:** https://claude.ai/artifact/6A9JXoaGLsyMuh7DhYMdP3

## Before the interview (10 minutes, once)
1. Add the unlock code and any keys in Vercel: Project `genai-showcase` → Settings → Environment Variables (Production). Names are in `docs/PROVIDERS.md`. At minimum `DEMO_UNLOCK` (pick your own code, 12+ characters). Then redeploy (Deployments → Redeploy) so the variables apply.
2. In the Anthropic console, put the API key in a dedicated workspace with a **monthly spend limit**. It is the only hard cost cap; the in-app limits are best-effort.
3. Open the site 5 minutes early and click Ask once, so the serverless function is warm.
4. Press **Unlock** in the header and enter your code. The pill should read "presenter unlocked".
5. Have a fallback ready: a local copy runs with `pip install -r requirements-full.txt` then `ANTHROPIC_API_KEY=... DEMO_UNLOCK=x python -m uvicorn showcase.server:app --port 8099`. If the model call fails live, the site falls back to retrieval-only with a banner. That is a feature: say so.

## The 5 minutes

**0:00 · Frame it (20 s)**
"This is a company-policy assistant I built to show the whole lifecycle: retrieval, an agent with human approval, cost-aware model routing, tracing, and evals. It's live and deployed on Vercel."

**0:20 · Ask: RAG with citations and routing (60 s)**
- Click *How many unused leave days can I carry forward?* → answer with citation chips; click a chip to highlight the retrieved chunk.
- Point at the **Routing decision** card: "A short lookup went to Haiku, the cheapest tier. The router records why."
- Click *Compare the leave and expense policies and tell me where they conflict* → "A comparative question goes up a tier. Public visitors stop at Sonnet; as presenter I get Opus." (Cost control, said in one sentence.)
- Click *What is the CEO salary?* → "It says it doesn't know. Grounded answers or an honest refusal, never a guess."

**1:20 · Trace (40 s)**
Open **Trace**. "Every request records spans for routing, retrieval, each model call and any provider fallback, with timings and tokens. With Langfuse keys set, the same trace is also sent there."

**2:00 · Agent with human approval (60 s)**
Agent tab → *My whole team's VPN is down, raise a P2 ticket* → the **approval card** appears. "The agent can read policies and do arithmetic freely, but a write action pauses for a person. That's LangGraph's interrupt, and the state is checkpointed so it can resume." Click Approve → ticket id. Mention: "If the server forgets the thread, the UI replays the request, because serverless instances are ephemeral."

**3:00 · Evals, honestly (50 s)**
Evals tab → Run. "Hit-in-top-4 would read 100%, but on an 8-document corpus that is nearly automatic. So I gate on mean reciprocal rank: 0.87, with the right document first for 7 of 9 questions. The two misses are password rules and office days, which is exactly where I'd add hybrid search and a reranker. Offline, Ragas and DeepEval add faithfulness and tool-correctness, and a faithfulness gate runs in CI."

**3:50 · Models and resilience (40 s)**
Models tab. "One interface, 18 providers: Claude, Gemini, Groq, Chinese open models like GLM and Qwen, and more. A provider exists only when its key is set. If one is rate-limited, the call falls through to the next." If unlocked, run **Compare** on one question across 2–3 providers.

**4:30 · Close (30 s)**
"Next I'd add Postgres for agent state and a managed vector store with hybrid retrieval, run the eval gate on every PR, and put Managed Agents behind the same approval flow." (If unlocked and configured: show **Background agent** producing files.)

## Likely follow-ups
| Question | Short answer |
|---|---|
| Why BM25 on the hosted demo? | Serverless cold starts and bundle limits rule out a local embedding model. BM25 is strong on exact terms; the vector version runs locally and in the Docker/HF builds. |
| How do you stop a public demo burning money? | Public tier clamp, no compare or background agent without the unlock code, a global hourly cap and kill switch, trimmed history, short timeouts, and a spend limit on the key. |
| What if the model call fails? | Degrades to retrieval-only with citations and says so. |
| Agent vs workflow? | Fixed steps are a workflow. The agent loop is for when the next step depends on tool results; it has step limits and approval for writes. |
| Managed Agents vs your LangGraph agent? | Managed Agents hosts the loop and a sandbox for long, file-producing jobs; LangGraph gives full control and any provider. I use both (ADR-07). |
| How do you evaluate an agent? | Final answer, trajectory (right tools in order), safety (approvals, injection), plus cost and latency per task. |
