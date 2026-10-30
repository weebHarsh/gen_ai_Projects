# Showcase API contract (hosted demo, FastAPI on Vercel)

This is the contract between `showcase/server.py` (backend) and `showcase/static/index.html` (UI).
Both are built against it independently. Change it here first, then in the code.

All routes are same-origin JSON. No CORS. Errors are always `{"error": "<code>", "message": "<human text>"}` with
HTTP 400 (bad input), 404 (unknown session/thread), 429 (rate limit, with `Retry-After`), 503 (feature not configured).

## Shared shapes

```jsonc
// route: why this model was chosen (common/router.py -> Decision)
"route": {"tier": "quick|default|complex", "provider": "anthropic", "model": "claude-haiku-5-5",
          "reason": "short single-fact lookup", "est_cost_usd": 0.0004, "pinned": false}

// trace: spans captured by showcase/tracing.py, for the waterfall UI
"trace": {"trace_id": "abc123", "total_ms": 1840, "spans": [
  {"id": "s1", "parent_id": null, "name": "route", "kind": "route|chain|retriever|llm|tool|fallback",
   "start_ms": 0, "duration_ms": 3, "status": "ok|error", "tokens_in": null, "tokens_out": null,
   "detail": "tier=quick model=claude-haiku-5-5"} ]}

// mode: "llm" when a model answered; "retrieval-only" when no provider is configured or every provider failed
"mode": "llm|retrieval-only", "notice": "optional banner text, present when degraded"
```

`contexts` items: `{"source": "expense_policy.md", "chunk_id": 5, "text": "..."}`.

## Routes

| Route | Request | Response |
|---|---|---|
| `GET /` | - | static `index.html` |
| `GET /api/health` | - | `{"status":"ok","retriever":"bm25","mode":"llm\|retrieval-only","providers":["anthropic",...]}` |
| `GET /api/providers` | - | `{"providers":[common.providers.describe()...],"available":["anthropic",...]}` (never any key value) |
| `POST /api/ask` | `{"question": str(<=500), "provider": "auto"\|<name>}` | `{"answer","sources":[str],"contexts":[...],"route","mode","notice?","latency_ms","trace"}` |
| `POST /api/compare` | `{"question": str, "providers": [<name>] (1-4)}` | `{"results":[{"provider","model","answer","latency_ms","tokens_out?","error?"}]}` |
| `POST /api/chat` | `{"message": str(<=500), "thread_id"?: str}` | `{"status":"done\|pending_approval","thread_id","reply?","approval_request?":{"action","args","question"},"steps":[{"type":"tool","name","args"}],"route","mode","notice?","trace"}` |
| `POST /api/approve` | `{"thread_id": str, "approved": bool}` | same shape as `/api/chat` |
| `GET /api/evals` | - | `{"metrics":{"source_hit_rate","refusal_rate_out_of_scope","citation_rate"},"gate":{"name":"source_hit_rate","threshold":0.8,"passed":bool},"rows":[{"question","expected_source","sources":[str],"hit":bool}],"mode","latency_ms"}` |
| `GET /api/managed/status` | - | `{"configured": bool, "tasks":[{"id","title","description"}], "hint?": str}` |
| `POST /api/managed/start` | `{"task_id": str}` | `{"session_id","console_url"}` (503 `not_configured` unless ANTHROPIC_API_KEY + MANAGED_AGENT_ID + MANAGED_ENVIRONMENT_ID are set) |
| `GET /api/managed/events?session_id=&after=` | cursor `after` = last event id seen | `{"events":[{"id","type","text?","tool?","at"}],"cursor":"...","done":bool,"files":[{"id","name","size"}]}`; serves `agent.custom_tool_use` for `search_acme_policies` during the poll |
| `GET /api/managed/file?session_id=&file_id=` | - | file bytes (`Content-Disposition: attachment`) |

## Behaviour rules (backend)

- **Retriever**: `RETRIEVER` defaults to `bm25` in the showcase (set before importing rag_pipeline).
- **Routing**: `provider:"auto"` -> `common.router.route(question)` picks the tier; the model comes from `get_llm_chain(tier=...)`
  so a rate-limited provider falls through to the next configured one. A pinned provider skips routing (`route.pinned=true`).
  Agent turns (`/api/chat`) use at least tier `default` because tool calling must be reliable.
- **Degradation**: no provider configured, or every provider raises -> do not 500. Return top-2 retrieved passages as an extractive
  answer with citations, `mode:"retrieval-only"` and a `notice`. `/api/chat` degrades to a plain message saying the agent needs a model.
- **Guardrails** (public URL, metered LLM): input <= 500 chars, control characters stripped; per-IP token bucket (30 req/min overall,
  8/min on `/api/ask`, `/api/chat`, `/api/compare`, `/api/managed/start`); LLM `max_tokens` <= 1000 for demo calls; agent `recursion_limit` 12;
  `provider` and `task_id` validated against allow-lists; nothing user-supplied is fetched or executed; responses never contain secrets.
- **Tracing**: every LLM-using route returns `trace`; spans include `route`, `retriever`, `llm` (with token counts when the provider reports
  them), `tool`, and a `fallback` span when a provider failed and the chain moved on. If Langfuse keys exist, also send to Langfuse via
  `common.observability.trace_config`.
- **Vercel OIDC**: if request header `x-vercel-oidc-token` is present, set `os.environ["VERCEL_OIDC_TOKEN"]` so the `gateway` provider authenticates.
- **Threads**: agent state lives in an in-memory checkpointer; an unknown `thread_id` starts a fresh thread (the UI shows a note).
- Static assets are served from `showcase/static/`.

## UI requirements

Single page, sections: **Ask (RAG)**, **Agent** (chat + approve/reject card), **Models** (provider list with configured badges,
compare mode, routing explanation), **Trace** (waterfall of the latest request), **Evals** (run + table + gate), **Background agent**
(Managed Agents: task picker, live event feed, file downloads, graceful "not configured" state), **Architecture** (Mermaid diagram + links).
Light/dark by tokens, works at 400px width, no horizontal page scroll, keyboard accessible, no external assets except Google Fonts and
cdnjs (Mermaid). Every state has a designed empty/loading/error treatment. Show the degraded-mode banner when `mode == "retrieval-only"`.
