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
| `GET /api/evals` | - | `{"metrics":{"mrr","top1_rate","source_hit_rate","refusal_rate_out_of_scope","citation_rate"},"gate":{"name":"mrr","threshold":0.8,"passed":bool},"rows":[{"question","expected_source","sources":[str],"rank":int\|null,"hit":bool}],"mode","latency_ms","cached?"}`. Rank-based metrics gate the result because hit-in-top-4 is trivially high on an 8-document corpus. |
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

---

# v2: access control, cost caps and resilience (from the security review; binding)

## Access levels
- **Public** (default): `provider:"auto"` tiers are clamped to `MAX_TIER` (env, default `default`, so Opus is never reached by strangers).
  `/api/compare` and `/api/managed/start` return `403 {"error":"locked"}`. `/api/ask`, `/api/chat`, `/api/approve`, `/api/evals` work.
- **Unlocked** (presenter): request header `x-demo-unlock: <code>` equal to env `DEMO_UNLOCK` (`hmac.compare_digest`). Lifts the tier clamp
  (Opus allowed), enables compare and managed start. If `DEMO_UNLOCK` is unset, nobody is unlocked (fail closed) and `/api/health` says so.
- `GET /api/health` additionally returns `"access": {"unlock_configured": bool, "unlocked": bool, "max_tier_public": "default"}` (never the code).
- `POST /api/unlock {"code": str}` -> `200 {"unlocked": true}` or `403 {"error":"locked"}`; counts against the heavy bucket (anti brute force).
- The UI keeps the code in `sessionStorage` only and sends it as `x-demo-unlock` on every API call once entered.

## Global cost controls (per instance, in memory)
- Kill switch: env `DEMO_LLM_ENABLED=0` makes `llm_available()` false (everything degrades to retrieval-only).
- Hourly cap: env `DEMO_LLM_CALLS_PER_HOUR` (default 300) counts every LLM-backed request (ask/chat/approve/compare each count their calls);
  beyond it, requests degrade to retrieval-only with a notice ("hourly demo budget reached"). Unlocked requests are exempt from the cap.
- Agent: only the last 8 messages (and never splitting a tool_use from its tool_result) are sent to the model; max 10 user turns per thread
  (then `409 {"error":"thread_full"}`); `recursion_limit` 8; `max_tokens` <= 800.
- Managed start: env `MANAGED_STARTS_PER_HOUR` (default 3) per instance; session budget amount `"25"` (= $0.25); locked unless unlocked.
- Provider order for PUBLIC requests puts free providers and the capped gateway BEFORE the direct Anthropic key; unlocked requests keep
  Claude first. (`common/providers.py` exposes `order_for(unlocked: bool)`.)

## Credentials and headers
- NEVER copy `x-vercel-oidc-token` into `os.environ`. If supported at all, parse it per request into a `contextvars.ContextVar` and pass it
  explicitly to `get_llm(..., api_key=...)`, only when `os.getenv("VERCEL") == "1"`, the JWT payload `iss` starts with `https://oidc.vercel.com/`,
  and its `project_id` equals env `VERCEL_PROJECT_ID`. Otherwise ignore the header. `is_configured("gateway")` depends only on `AI_GATEWAY_API_KEY`
  or `VERCEL`-verified OIDC, never on a raw request header. Recommended path: set `AI_GATEWAY_API_KEY`.
- Client IP for rate limiting: on Vercel (`VERCEL=1`) use `x-vercel-forwarded-for` then `x-real-ip`; elsewhere use `request.client.host`. Never trust client `X-Forwarded-For`.

## Chat/approval robustness
- Thread ids: server-issued `uuid4().hex`; client-supplied ids must match `^[0-9a-f]{32}$` (unknown but well-formed ids start a fresh thread, so an
  instance restart does not 404 the chat).
- `/api/chat` on a thread with a pending approval -> `409 {"error":"approval_pending"}`. After a `GraphRecursionError` or any agent exception,
  delete that thread's checkpoint so it cannot be left with a dangling tool call.
- `/api/approve` on an unknown thread -> `404 {"error":"unknown_thread"}` (the UI re-sends the original message and shows a fresh approval card).
- LLM calls: `timeout=10`, `max_retries=0`, at most 2 providers in a chain; a per-request deadline of 45 s (after which the route returns a
  degraded answer); `maxDuration` stays 60.

## Evals
- `/api/evals` is retrieval-only by default (`EVAL_USE_LLM=1` to also call a model); guarded by a lock so concurrent cold requests compute once.
  In retrieval-only mode `refusal_rate_out_of_scope` and `citation_rate` are `null` (the extractive fallback cannot refuse and always cites).
  The UI hides tiles whose value is null.

## Misc
- `/api/managed/file`: 413 `{"error":"too_large"}` above 4,000,000 bytes (message points to the console).
- Compare and other error strings return only `type(e).__name__` and an HTTP status when known, never provider messages.
- Control characters stripped from inputs include bidi/zero-width: U+200B-U+200F, U+202A-U+202E, U+2066-U+2069.
- `route` carries `"failed": true` and the response `mode:"retrieval-only"` when the chosen provider errored.
- UI: remove the Swagger link (`/docs` is disabled); describe Langfuse as optional; replace the example prompt "Check the status of ticket INC-1042".
