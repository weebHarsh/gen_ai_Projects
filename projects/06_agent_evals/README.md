# 06 · Evals: RAG Metrics, Agent Trajectories, LLM-as-Judge (60 min)

**You'll learn:** golden datasets, deterministic vs LLM-judge metrics, Ragas RAG metrics, DeepEval agent metrics (tool correctness, G-Eval), Langfuse experiments, and evals as a CI quality gate.

## Setup
```bash
pip install -r requirements-full.txt -r projects/06_agent_evals/requirements-evals.txt
```
> **Dependency gotcha (real, Oct 2026):** `ragas 0.4.x` imports a module removed in `langchain-community 0.4.2`, so the evals requirements pin `langchain-community<0.4.2`. Pin versions in production and test upgrades in CI.

## Run
```bash
python projects/06_agent_evals/eval_rag.py --no-judge       # deterministic only, free
python projects/06_agent_evals/eval_rag.py                  # + Ragas (faithfulness, context precision/recall, factual correctness)
python projects/06_agent_evals/eval_agent.py                # tool-use trajectory + DeepEval ToolCorrectness + G-Eval
python projects/06_agent_evals/langfuse_experiment.py --run-name baseline   # results + scores in the Langfuse UI
```
`eval_rag.py` exits with code 1 if faithfulness < 0.85 (Acme's go-live bar), so you can drop it into GitHub Actions as a gate.

## The eval pyramid (interview answer)
| Layer | Examples here | Cost | When |
|---|---|---|---|
| Deterministic checks | source hit rate, refusal on out-of-scope, citation present, tool called | free, exact | every commit |
| LLM-as-judge (offline) | Ragas faithfulness, context precision/recall, G-Eval helpfulness, correctness vs reference | $ | every PR / prompt change |
| Online evals | judge a sample of production traces in Langfuse; user feedback scores | $$ | continuously |
| Human review | annotation queues on low-scoring traces → new golden items | $$$ | weekly |

## RAG metrics cheat-sheet
| Metric | Question it answers | Fix when low |
|---|---|---|
| **Context recall** | Did retrieval find everything needed for the reference answer? | chunking, embeddings, k, hybrid search |
| **Context precision** | Are the relevant chunks ranked at the top? | re-ranker, MMR, metadata filters |
| **Faithfulness** | Is every claim in the answer supported by retrieved context? | prompt ("only from context"), model, temperature/effort |
| **Answer relevancy / factual correctness** | Does it answer the question and match the reference? | prompt, retrieval |

## Agent eval cheat-sheet
- **Final-response evals**: was the outcome correct and helpful?
- **Trajectory evals**: right tools, right order, right arguments, no unnecessary calls (`tools_called` vs `expected_tools`).
- **Single-step evals**: given this state, was the next action right? (unit tests for the router)
- **Safety evals**: refuses out-of-scope, asks approval before write actions, resists prompt injection.
- Track **cost and latency per task** alongside quality. A "better" agent that costs 3× may not be better.

## LLM-as-judge pitfalls
Position and verbosity bias, self-preference (judge favours its own model family), and non-determinism. Mitigate with explicit rubrics/evaluation steps, binary or low-cardinality scales, reference answers, a different judge model, and periodic agreement checks against human labels.

## Things to try
1. Change `chunk_size` in project 02 to 200, re-run `eval_rag.py`, and compare context recall.
2. Run `langfuse_experiment.py --run-name haiku` with `LLM_MODEL=claude-haiku-5-5` and compare runs side by side in Langfuse.
3. Add three adversarial agent cases (prompt injection, ticket without asking) to `golden_dataset.json`.
