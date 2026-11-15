# Progress checkpoint (resume from here after a context reset)

**State:** all 9 projects, the Vercel showcase, docs, tracker and draft PR #1 are delivered. 145 tests pass
(`LLM_PROVIDER=fake RETRIEVER=bm25 .venv/bin/python -m pytest -q tests -p no:warnings`).

- Live site: https://genai-showcase-weebharshs-projects.vercel.app (production deploy READY, commit 24a0392)
- Tracker: https://claude.ai/artifact/6A9JXoaGLsyMuh7DhYMdP3
- Branch: `claude/peaceful-mendel-p39864`; draft PR #1 -> `main`
- Plan: `/root/.claude/plans/deep-forging-perlis.md`

## Waiting on the user
1. In Vercel (Settings -> Environment Variables, Production): set `DEMO_UNLOCK` (12+ chars) and any provider keys (names in `docs/PROVIDERS.md`), then redeploy.
2. Put the Anthropic key in a workspace with a monthly spend limit (the only hard cap).
3. Register for free providers (priority list in `docs/PROVIDERS.md`) and say "keys added".
4. Decide when to merge PR #1.

## Next step once keys are added
Redeploy production, check `/api/health` and `/api/providers`, run one live call per provider, one Managed Agents session, report pass/fail.

## Not yet verified live
Any model call on the live site, free-provider model ids, Managed Agents against the real API, Mermaid rendering.

## Session settings
Autocompact is set to trigger at ~65% context via `.claude/settings.json` (`CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`).
