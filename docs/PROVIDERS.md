# LLM providers: how to get keys and wire them in

This is the how-to for the repo owner. It covers which services to register for, where each key goes, and how
the app picks a model. The source of truth is `common/providers.py` (the `PROVIDERS` registry). Free-tier limits
change often, so treat every number here as a starting point and re-check it on the provider's page (section f).

## a) How keys flow

- **Local development:** copy `.env.example` to `.env` and fill in the values you have. `.env` is git-ignored.
  Blank values mean the provider is disabled, so you can add keys one at a time.
- **Hosted demo (Vercel):** add the same variable names in Vercel: Project Settings → Environment Variables,
  for both **Production** and **Preview**. Redeploy after changing them.
- **Never paste a key into chat, a ticket, a commit, or a screenshot.** Chat logs persist. If a key is exposed,
  revoke it in the provider's dashboard and create a new one.
- Keys are never returned by the API. `GET /api/providers` (once the showcase is live) returns names and
  `configured: true/false` only.

## b) Provider table

Every provider in the registry, in `FALLBACK_ORDER`. "Default models" are the registry's best-known ids for each
tier (quick / default / complex); a single id means the same model is used for every tier. "China-based: Yes" is
the registry's `cn` flag. Caveats come from the registry `note` field and from the plan, marked "(verify)" where
the source did not confirm them.

| Provider (registry name) | What it's good for | Signup | Env var(s) | OpenAI-compatible base URL | Default models by tier | Free / paid | China-based | Caveats |
|---|---|---|---|---|---|---|---|---|
| `anthropic` (Claude, Anthropic API) | Claude Haiku, Sonnet and Opus. Native SDK, also used for Managed Agents | platform.claude.com | `ANTHROPIC_API_KEY` | None (native SDK) | quick: `claude-haiku-5-5`; default: `claude-sonnet-5-5`; complex: `claude-opus-5-5` | Paid | No | Uses your Anthropic credits. |
| `gateway` (Claude via Vercel AI Gateway) | Claude through Vercel, so the hosted app needs no separate Anthropic key | vercel.com (already connected for the project) | `AI_GATEWAY_API_KEY`, or `VERCEL_OIDC_TOKEN` (injected by Vercel at runtime) | `https://ai-gateway.vercel.sh` | quick: `anthropic/claude-haiku-5-5`; default: `anthropic/claude-sonnet-5-5`; complex: `anthropic/claude-opus-5-5` | Paid (credits) | No | Plan sets a $5/month project budget cap. |
| `gemini` (Google Gemini) | Google Flash models (fast, low cost) | aistudio.google.com/apikey | `GEMINI_API_KEY` | `https://generativelanguage.googleapis.com/v1beta/openai/` | quick: `gemini-2.5-flash-lite`; default: `gemini-2.5-flash`; complex: `gemini-2.5-pro` | Free tier on several Flash models (limits shown in AI Studio) | No | Not available in the EU or UK. |
| `groq` (Groq) | Very fast open models (Llama, Qwen, Kimi K2, gpt-oss) | console.groq.com/keys | `GROQ_API_KEY` | `https://api.groq.com/openai/v1` | quick: `llama-3.1-8b-instant`; default: `llama-3.3-70b-versatile` | Free tier (about 30 RPM and 1K requests/day per model, verify) | No | Tight free rate limits. |
| `openrouter` (OpenRouter) | One key for many models. About 28 `:free` models (DeepSeek, Qwen, GLM, Llama) | openrouter.ai/keys | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` | default: `deepseek/deepseek-chat-v3.1:free` | Free `:free` models; a $10 top-up raises the daily cap to about 1K requests (verify) | No | |
| `zai` (Zhipu Z.ai, GLM) | GLM models, including free Flash variants | z.ai (international site) | `ZAI_API_KEY` | `https://api.z.ai/api/paas/v4` | default: `glm-4.7-flash` | Flash models free with no expiry (as reported, verify) | Yes | 1 concurrent request on the free tier (verify). |
| `siliconflow` (SiliconFlow, Qwen / GLM / DeepSeek) | Small free Qwen, GLM and DeepSeek models with high RPM | cloud.siliconflow.com | `SILICONFLOW_API_KEY` | `https://api.siliconflow.com/v1` | default: `Qwen/Qwen3-8B` | Free small models | Yes | Use the international site. The .cn site requires a Chinese ID. |
| `mistral` (Mistral) | Mistral Small and Large | console.mistral.ai | `MISTRAL_API_KEY` | `https://api.mistral.ai/v1` | quick and default: `mistral-small-latest`; complex: `mistral-large-latest` | Free plan / monthly credits, no card (verify) | No | Limits may change. |
| `github` (GitHub Models) | OpenAI, Llama, DeepSeek and Mistral models through a GitHub account | github.com/marketplace/models | `GITHUB_MODELS_TOKEN` (a personal access token with `models:read`) | `https://models.github.ai/inference` | default: `openai/gpt-4.1-mini` | Free tier for GitHub users, low daily caps (verify) | No | Create a token with `models:read`. |
| `nvidia` (NVIDIA NIM) | Many open models, including DeepSeek, Qwen and Kimi | build.nvidia.com | `NVIDIA_API_KEY` | `https://integrate.api.nvidia.com/v1` | default: `meta/llama-3.3-70b-instruct` | Free credits (verify) | No | Credit amount may change. |
| `cerebras` (Cerebras) | Llama 3.3 70B | cloud.cerebras.ai | `CEREBRAS_API_KEY` | `https://api.cerebras.ai/v1` | default: `llama-3.3-70b` | Free tier status unverified in 2026 sources | No | Do not rely on a free allowance until verified. |
| `cohere` (Cohere) | Command R+ | dashboard.cohere.com | `COHERE_API_KEY` | `https://api.cohere.ai/compatibility/v1` | default: `command-r-plus` | Trial key: about 1,000 calls/month, 20 RPM (verify) | No | Trial key is non-commercial only. |
| `huggingface` (Hugging Face Inference Providers) | Many open models through one router | huggingface.co/settings/tokens | `HF_TOKEN` | `https://router.huggingface.co/v1` | default: `Qwen/Qwen2.5-72B-Instruct` | Small monthly free credits (verify) | No | Credits are small and monthly. |
| `dashscope` (Alibaba Qwen, Model Studio intl) | Qwen models | modelstudio.console.alibabacloud.com | `DASHSCOPE_API_KEY` | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` | default: `qwen-plus` | Trial credits (about 1M tokens for 90 days, verify) | Yes | Trial credits, not permanent. |
| `deepseek` (DeepSeek) | DeepSeek chat at a very low price | platform.deepseek.com | `DEEPSEEK_API_KEY` | `https://api.deepseek.com` | default: `deepseek-chat` | Paid, very cheap (top up about $2). Free via OpenRouter `:free` models | Yes | Not free. |
| `moonshot` (Moonshot Kimi) | Kimi K2 models | platform.moonshot.ai | `MOONSHOT_API_KEY` | `https://api.moonshot.ai/v1` | default: `kimi-k2-turbo-preview` | Free allowance unverified | Yes | Free allowance unverified. Kimi is also reachable through Groq and OpenRouter. |
| `minimax` (MiniMax) | MiniMax-M2 | platform.minimax.io | `MINIMAX_API_KEY` | `https://api.minimax.io/v1` | default: `MiniMax-M2` | Free allowance unverified | Yes | Free allowance unverified. |
| `cloudflare` (Cloudflare Workers AI) | Llama models on Cloudflare's edge | dash.cloudflare.com (AI) | `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` | None in the registry. Plan: `https://api.cloudflare.com/client/v4/accounts/<id>/ai/v1` | default: `@cf/meta/llama-3.3-70b-instruct-fp8-fast` | Free: about 10,000 neurons/day (verify) | No | Needs both variables. The base URL is not in the registry, so `--discover` skips it. |

## c) Registration order

Do the six starred providers first. This takes about 15 minutes in total.

1. **Anthropic** (★): platform.claude.com, create an API key. Paid, uses your credits.
2. **Google Gemini** (★): aistudio.google.com/apikey. Not available in the EU or UK.
3. **Groq** (★): console.groq.com/keys.
4. **OpenRouter** (★): openrouter.ai/keys. Use the `:free` models.
5. **Z.ai** (★): z.ai (international site).
6. **SiliconFlow** (★): cloud.siliconflow.com (international site, not the .cn site).

The Vercel AI Gateway needs no registration because Vercel is already connected. Only set its budget.

Optional, when you want more coverage: Mistral, GitHub Models, NVIDIA NIM, Hugging Face, Cloudflare Workers AI.

**Skip for now:**
- Cerebras, Cohere, DashScope, Moonshot and MiniMax. Their free allowances are unverified, or they are trials or non-commercial. Add them later if you want more models.
- DeepSeek. It is cheap but not free. You can reach its models for free through OpenRouter `:free`.
- Baidu ERNIE, Tencent Hunyuan, ByteDance Doubao and China Mobile. These need a Chinese phone number or real-name ID. Most of their models are reachable through OpenRouter, SiliconFlow (international) or NVIDIA instead. They are not in the registry.

## d) After you add keys

```bash
# List every provider and mark the ones that are configured (*)
python -m common.providers

# Call GET /models on each configured OpenAI-compatible provider and list its current model ids
python -m common.providers --discover
```

Use the `--discover` output to pick current model ids, since the defaults in the table can go stale.

To override a model, set `<NAME>_MODEL`, where NAME is the registry key in upper case. Examples:

```bash
GROQ_MODEL=llama-3.3-70b-versatile
HUGGINGFACE_MODEL=Qwen/Qwen2.5-72B-Instruct
ZAI_MODEL=glm-4.7-flash
```

The variable name uses the registry key, not the env var. So the Hugging Face override is `HUGGINGFACE_MODEL`, not
`HF_MODEL`. An override replaces the model for **every** tier of that provider, so a multi-tier provider such as
`GEMINI_MODEL` or `ANTHROPIC_MODEL` ends up using one model for all tiers.

## e) How routing works

Routing has two layers. The router (`common/router.py`) first sorts each question into a tier: **quick**, **default** or
**complex**. With `ROUTER=heuristic` (the default) this is a deterministic rule set with no extra LLM call. With
`ROUTER=llm`, a small model classifies ambiguous questions, and the heuristic is the fallback. For Claude, the tiers map to
Haiku 5.5 (quick), Sonnet 5.5 (default) and Opus 5.5 (complex). Each free provider has one model per tier, or one model for
all tiers, as shown in the table.

The fallback chain is built from whichever keys are present, in the order of `FALLBACK_ORDER` in `common/providers.py`:
Anthropic, Vercel AI Gateway, Gemini, Groq, Z.ai, SiliconFlow, OpenRouter, Cerebras, Mistral, GitHub Models, NVIDIA,
DashScope, Hugging Face, Moonshot, MiniMax, DeepSeek, Cohere, Cloudflare. If a provider fails or hits its rate limit, the
next configured provider in that order takes over. Providers without a key are skipped.

## f) Limits change often

Free tiers, rate limits, trial credits and model ids change without notice. The numbers in this document were reported in
2026 and are marked "(verify)" where they came from a secondary source. Re-check each one before you rely on it:

- The provider's own pricing or limits page, linked from its signup host in the table.
- `python -m common.providers --discover` for the model ids each key can actually see right now.
- The provider's dashboard for your account's current limits.

If a provider starts failing, check its status page and dashboard first, then remove or replace its key rather than
debugging the app.
