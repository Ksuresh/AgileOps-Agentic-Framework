# Reviewer-requested LLM runs — plan fixed before running (2026-10-22)

Cases: the frozen 32 + 16 selective-AEI cases. Code: frozen `agentic_experiments/hybrid.py` and `agentic_runtime.py`;
only runtime settings or coordinator text change. Model gpt-5.6-luna. Scoring: action in the case's admissible set.

| Arm | What | Runs | Est. calls | Reviewer |
| --- | --- | --- | --- | --- |
| HY | Hybrid AAF, reasoning none | frozen + 4 new | ~470 | R2-3, R4-3 |
| AO | Agentic-only, reasoning none | frozen + 2 new | ~500 | R2-3, R4-3 |
| B3 | Always-on Agentic + deterministic final authority: deterministic AAF on the evidence after the tools the always-on agents requested in each AO run | from AO runs | 0 | R2-1, R4-1 |
| B6 | Agentic-only, reasoning "medium", AAF governance rules added to the coordinator prompt | 3 | ~750 | R4-2, R6-2 |
| PV1, PV2 | Agentic-only coordinator with two rewordings of its prompt, re-run on the frozen agent outputs | 1 each | 96 | R5-1 |

Reported for every arm: correct cases per run; majority-of-runs correct per case; run-level accuracy; cases with the same
action in all runs; per stratum; schema and evidence-reference failures; tokens and calls. Paired comparisons use the
majority-of-runs result. Budget guards: probe call first; caps 2,600 calls and 4M tokens; 180 s per-call timeout;
each arm checkpointed when it finishes.
