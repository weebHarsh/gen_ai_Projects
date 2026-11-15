# LLM Observability & Evaluation Landscape (Oct 2026)

Rough split: **tracing-first** tools (Langfuse, LangSmith, Phoenix, Opik) vs **evaluation-first** tools (Braintrust, Confident AI/DeepEval, Galileo). Most now do both, so pick by your stack, hosting needs and team.

| Platform | Leans toward | Open source / self-host | Notes |
|---|---|---|---|
| **Langfuse** | Tracing + prompts + datasets | MIT, self-host free (Docker/Helm); Cloud paid tiers from ~$29/mo | OTel-native, framework-agnostic, strong LangChain/LangGraph integration. **Chosen for this repo.** |
| **LangSmith** | Tracing + evals for LangChain/LangGraph | SaaS; self-host is an enterprise add-on | Best-in-class LangGraph debugging (Studio). Lock-in to the LangChain ecosystem. |
| **Arize Phoenix / Arize AX** | Tracing + evals, ML observability heritage | Phoenix OSS (self-host); AX paid, from ~$50/mo | OpenInference/OTel instrumentation; strong for RAG and embedding analysis. |
| **Opik (Comet)** | Tracing + evals + prompt optimisation | Apache-2.0, self-host | Good LLM-judge metrics library. Fits teams already on Comet. |
| **Braintrust** | Evaluation-first (experiments, scorers, CI) | SaaS (hybrid deploy for enterprise) | Excellent eval UX and regression testing in CI. |
| **Confident AI (DeepEval)** | Evaluation-first | DeepEval OSS; platform SaaS | pytest-style LLM tests; agent metrics (tool correctness, task completion). Used in project 06. |
| **Datadog LLM Observability** | APM-integrated monitoring | SaaS | Best if you already run Datadog. Multi-agent teams often pair it with a dedicated tracing tool. |
| **MLflow** | Experiment tracking + GenAI tracing/evals | Apache-2.0 | Natural for Databricks shops. |
| **Galileo** | Evals + guardrails (hallucination detection) | SaaS | Proprietary evaluator models ("Luna"). |
| **Fiddler** | Enterprise monitoring, explainability, guardrails | SaaS / on-prem | Regulated industries. |
| **Raindrop** | Production signals from user behaviour | SaaS | Detects issues (frustration, refusals) in live traffic. |

## How to choose (decision rule)
1. **Data residency / must self-host?** → Langfuse, Phoenix or Opik (all OSS).
2. **All-in on LangGraph and fine with SaaS?** → LangSmith.
3. **Eval rigour and CI gates are the priority?** → Braintrust or Confident AI, alongside a tracer.
4. **Already on Datadog?** → Datadog LLM Obs for ops metrics, plus Langfuse/LangSmith for deep agent traces.

## Interview soundbite
> "I separate *observability* (what happened: traces, cost, latency) from *evaluation* (was it good: offline golden sets, online LLM-as-judge, user feedback). I used Langfuse for tracing and prompt management because it's open source, OTel-native and self-hostable, and DeepEval/Ragas for evaluation, pushing scores back onto Langfuse traces so quality and cost live in one place."
