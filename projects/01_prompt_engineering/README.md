# 01 · Prompt Engineering & Structured Output (45 min)

**You'll learn:** prompt anatomy, zero/few-shot, structured output with Pydantic, LCEL chains, prompt-injection defence.

## Concepts in 60 seconds
| Concept | What to say in an interview |
|---|---|
| Prompt anatomy | Role, task, context, constraints, output format. Put stable instructions in the **system** message, variable data in the **human** message. |
| Few-shot | Examples steer format and edge cases more reliably than longer instructions. Keep them diverse and short. |
| Structured output | Use a schema (Pydantic / JSON Schema) and the provider's native structured-output mode instead of regex-parsing text. Field descriptions *are* prompt. |
| LCEL | `prompt | llm | parser` builds a `Runnable` that supports `.invoke/.batch/.stream` plus callbacks (tracing) for free. |
| Chain-of-thought | Modern Claude models think adaptively; you control depth with **effort**, not "think step by step" tricks. Ask for reasoning in a separate field only when you need to show it. |
| Prompt injection | Untrusted text can carry instructions. Defences: delimit it (`<ticket>` tags), state it's data, least-privilege tools, human approval for actions, output validation. |

## Run
```bash
python projects/01_prompt_engineering/main.py all          # needs ANTHROPIC_API_KEY
LLM_PROVIDER=fake python projects/01_prompt_engineering/main.py all   # offline
```

## Things to try (15 min)
1. Remove the "A team blocked = P2" rule from `structured_output` and see if the priority changes.
2. Change `LLM_PROVIDER` to `openai` or `do` and compare answers. Only one env var changes.
3. Add a `sentiment` field to `Triage`.
4. Run with Langfuse keys set (project 05) and look at the prompt and token cost of each demo.

## Interview questions
- *How do you make LLM output reliably machine-readable?* Native structured output plus Pydantic validation, and retry on validation error.
- *Zero-shot vs few-shot vs fine-tuning?* Start zero-shot. Add examples when format or edge cases fail. Fine-tune only for style/format at scale, or for latency/cost, after evals show prompting has plateaued.
- *How do you defend against prompt injection?* Treat retrieved and user text as data, use delimiters, least-privilege tools, human approval for irreversible actions, and output filters. You can't fully "prompt it away".
- *Why are there no temperature settings here?* Current Claude models (Opus 5.5 / Sonnet 5.5) reject sampling parameters; you tune with `effort` instead.
