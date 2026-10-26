# Frontier-model deciders on the outcome studies (frozen before any GPT run on outcome-study data)

## Why
In Studies 14-16 the LLM-decides baselines (B5, B6) use Qwen3-8B. A reviewer can fairly ask whether a frontier model,
given the same evidence, would match AAF. This run adds GPT-5.6 Luna deciders on the same decider inputs. No case,
measurement, oracle, rule, prompt or threshold changes; only the model behind the LLM arms differs.

## What runs (`outcome_study/decide_gpt.py`, workflow `outcome-decisions-gpt.yml`)
- Inputs: the committed `<study>/decisions/variants.json` of Studies 14, 15 and 16 (replicate-1 snapshots; full-evidence
  and withheld-evidence variants). Outcomes are never read.
- Arms, one run per variant (cost cap; Studies 10/10b already measure run-to-run variation):
  B5-GPT (LLM decides, reasoning none), B6-GPT (LLM decides, reasoning medium, coordinator given AAF's written policy),
  AAF-14-GPT (rule set v1.4 decides; GPT is the investigator). B3-GPT / B3-14-GPT: v1.0 / v1.4 rules re-deciding on
  B5-GPT's retrieved evidence (no model calls).
- Same bounded runtime, prompts, tools and budget guard as Studies 6-8. Per-run caps on calls and tokens; a run stopped by
  a cap is resumed with identical settings and completed rows are not rerun.

## Hypotheses (exact McNemar, O1, fault cases; per study and pooled over Studies 14-16)
- **G1 (primary, pooled):** AAF-14 (as pre-specified in each study, Qwen investigator) vs B5-GPT and vs B6-GPT; Holm over 2.
  Per-study results are reported unadjusted.
- **G2 (model independence on measured outcomes):** action agreement between AAF-14-GPT and AAF-14 (Qwen) on all variants;
  O1 accuracy of each. Expected: near-identical on full-evidence variants (the rules decide).
- **G3 (secondary):** on withheld-evidence variants, AAF-14-GPT vs B1-14 and B2-14 (fetch rule).
- **Descriptive:** GPT calls and tokens per variant per arm; O2 accuracy; security-priority and low-risk-release counts.
Any comparison may go against AAF; all are reported. Analysis: `outcome_study/analyze_gpt.py`, committed with this file.

## Execution notes (appended after the runs)
- Study 14: 933 calls, 665,572 tokens; Study 15: 1,238 calls, 882,241 tokens; Study 16: 857 calls, 612,581 tokens. No cap was reached and
  every variant has a decision for each arm.
- After the first data, `analyze_gpt.py` was changed twice: to list only the deciders present in a study, and to read the LLM single-domain
  experts and the v1.4 re-decision from the Qwen decision records for the paper's combined statistics. Neither change affects a GPT decision.
