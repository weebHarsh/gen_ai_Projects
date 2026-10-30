# 05 · AI Observability with Langfuse (60 min)

**You'll learn:** traces, spans and generations, sessions/users/tags, cost and latency, `@observe` for non-LangChain code, prompt management, user-feedback scores, and how to run Langfuse yourself on DigitalOcean.

## Why Langfuse here
Easy to set up (Cloud free tier in 2 minutes), the most to learn (tracing, prompts, datasets, evals and scores in one tool), and production-ready (MIT-licensed, OpenTelemetry-based, self-hostable on a $24 Droplet). See `docs/OBSERVABILITY_LANDSCAPE.md` for the comparison with LangSmith, Arize Phoenix/AX, Opik, Braintrust, Datadog and others.

## Setup (pick one)
**A. Langfuse Cloud (2 min, recommended for the 1-day sprint)**
1. Sign up at https://cloud.langfuse.com → create a project → Settings → API keys.
2. Put `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL=https://cloud.langfuse.com` in `.env`.

**B. Self-host on a DigitalOcean Droplet (production path, 30 min)**
See `projects/08_deploy_digitalocean/langfuse_droplet_setup.sh`. Ubuntu Droplet with at least 4 GB RAM (8 GB is comfortable), Docker Compose, then point `LANGFUSE_BASE_URL` at `http://<droplet-ip>:3000`.

## Run
```bash
python projects/05_observability_langfuse/main.py trace-rag      # 3 traces in 1 session
python projects/05_observability_langfuse/main.py trace-agent    # agent tree: LLM -> tool -> retriever -> LLM
python projects/05_observability_langfuse/main.py custom-spans   # @observe, manual generation, trace score
python projects/05_observability_langfuse/main.py prompt-mgmt    # versioned prompt pulled by label
python projects/05_observability_langfuse/main.py feedback       # thumbs-up score linked to a trace
```
Every other project is already instrumented via `common/observability.py`. Once the keys are set, projects 01–04 and 08 send traces automatically.

## What to look at in the UI
| Screen | What to notice |
|---|---|
| Traces | Nested tree. Click the retriever span to see *which chunks* were used; click the generation to see prompt, output, tokens and cost. |
| Sessions | `obs-demo-session` groups a conversation. Agent threads use `thread_id` as the session. |
| Users | Per-user cost and usage (`user_id`). |
| Prompts | `helpdesk-triage` with versions and labels; traces link to the version used. |
| Scores | `user_feedback` and eval scores (project 06). Filter traces by low score to find failures. |
| Dashboards | Latency p50/p95, cost per model, error rate. |

## Core ideas (interview answer)
- **Trace** = one request end-to-end. **Observation/span** = a step inside it. **Generation** = an LLM call with model, tokens and cost.
- What to monitor in production: latency (p95, time-to-first-token), cost per request/user, error and refusal rate, tool-call failures, retrieval hit quality, user feedback, eval scores on sampled traffic, and drift.
- **PII**: mask before export (Langfuse supports masking functions). Acme's AI policy requires masked logging.
- **OpenTelemetry**: Langfuse v3+ is OTel-native, so any OTel GenAI instrumentation can send to it, and you can also export to Datadog etc.
- **Online evals**: run LLM-as-judge on a sample of production traces and alert when scores drop.

## Things to try
1. Run project 03 with a 3-turn chat and find the session in Langfuse.
2. Create a second version of `helpdesk-triage` in the UI, label it `production`, re-run `prompt-mgmt`. No code change.
3. Add a dashboard widget for cost by `tags`.
