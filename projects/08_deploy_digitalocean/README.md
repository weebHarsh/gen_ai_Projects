# 08 · Deploy to DigitalOcean: API, Observability Stack, Enterprise/RPA Integration (75 min)

**You'll learn:** wrapping RAG and an agent in a REST API (FastAPI), human approval over HTTP, containerising, DO App Platform CI/CD, self-hosting Langfuse on a Droplet, DO AI Platform serverless inference, and how RPA bots call GenAI services.

```
 UiPath / Power Automate / Teams bot / web UI
                │  HTTPS + X-API-Key
                ▼
  DO App Platform ── acme-genai-api (Docker: FastAPI + LangGraph + Chroma + BGE embeddings)
                │            │                    │
                │            ├── Claude API (or DO serverless inference: LLM_PROVIDER=do)
                │            └── /servicedesk/tickets  (mock ServiceNow/Jira, via TICKET_API_URL)
                ▼
  DO Droplet ── Langfuse (web, worker, Postgres, ClickHouse, Redis, MinIO)  ← traces, scores, prompts
```

## 1. Run the API locally (10 min)
```bash
python projects/08_deploy_digitalocean/app.py         # http://localhost:8080/docs (Swagger UI)
curl -s localhost:8080/ask -H 'content-type: application/json' -d '{"question":"Hotel cap in metro cities?"}'
curl -s localhost:8080/chat -H 'content-type: application/json' -d '{"message":"VPN down for my team, raise a ticket"}'
# -> {"status":"pending_approval","thread_id":"api-1a2b3c4d", ...}
curl -s localhost:8080/approve -H 'content-type: application/json' -d '{"thread_id":"api-1a2b3c4d","approved":true}'
```

## 2. Container (10 min)
```bash
docker build -f projects/08_deploy_digitalocean/Dockerfile -t acme-genai .
docker run -p 8080:8080 --env-file .env acme-genai
```

## 3. Deploy the API on App Platform (20 min)
```bash
# one-time: install doctl, then `doctl auth init` with a DO API token
doctl apps create --spec projects/08_deploy_digitalocean/app.yaml
# set the SECRET env values (ANTHROPIC_API_KEY, APP_API_KEY, LANGFUSE_*) in the dashboard -> app -> Settings
```
`deploy_on_push: true` gives you CI/CD: every push to `main` rebuilds and redeploys. The health check hits `/health`.

## 4. Self-host Langfuse on a Droplet (25 min)
```bash
# Ubuntu 24.04, at least 4 GB RAM, BLR1 region
scp projects/08_deploy_digitalocean/langfuse_droplet_setup.sh root@<ip>:~
ssh root@<ip> 'bash langfuse_droplet_setup.sh'
```
The script installs Docker, downloads Langfuse's official compose file, **generates every secret** (the defaults in the compose file are public and must be replaced), opens only ports 22/3000/9090, and waits for the health check. Then set `LANGFUSE_BASE_URL=http://<ip>:3000` in the app.

## 5. Optional: DigitalOcean AI Platform (formerly "Gradient AI")
- **Serverless inference**: OpenAI-compatible endpoint `https://inference.do-ai.run/v1` with a *model access key*. Set `LLM_PROVIDER=do`, `DO_MODEL_ACCESS_KEY=...`, `LLM_MODEL=llama3.3-70b-instruct` (or any id from `GET /v1/models`). Same code, different model: that's what `common/llm.py` is for.
- **Managed agents + knowledge bases**: point-and-click RAG agents with guardrails and multi-agent routing, managed by the `gradient` CLI/ADK. Good for a quick internal bot; less control than this repo's LangGraph code.

## 6. Enterprise & RPA integration (interview talking points)
| Tool | How it calls this API |
|---|---|
| **UiPath** | `HTTP Request` activity (UiPath.WebAPI) → POST `/ask` with JSON body + `X-API-Key` header → `Deserialize JSON` → use `answer`. For approvals, put the `pending_approval` payload into an **Action Center** task, then call `/approve`. |
| **Power Automate** | `HTTP` action (premium) or a **custom connector** built from the OpenAPI spec at `/openapi.json`. Use a Teams "approval" action between `/chat` and `/approve`. |
| **Automation Anywhere** | `REST Web Service` package → POST `/chat`. |
| **Blue Prism** | Web API service definition imported from `/openapi.json`. |

The pattern: **RPA does deterministic UI/system steps; the GenAI service does judgement** (classify, extract, answer, decide the next tool), behind a versioned REST contract with auth, traces and evals. Don't embed prompts inside RPA bots, because you lose observability and testing.

## Production checklist
- [ ] Secrets in App Platform encrypted env vars, never in the image
- [ ] `PostgresSaver` checkpointer (DO Managed PostgreSQL) instead of `InMemorySaver` before running more than one instance
- [ ] Persist the vector index (Chroma server, pgvector on Managed PostgreSQL, or a managed vector DB) instead of rebuilding on startup
- [ ] Rate limiting and request size limits; timeouts on LLM calls; retries with backoff
- [ ] Langfuse tracing with PII masking; alerts on p95 latency, error rate and cost per day
- [ ] Eval gate (project 06) in CI before deploy
