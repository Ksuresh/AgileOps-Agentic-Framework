# B6 with AAF's full v1.4 policy (committed before any run)

## Why
In Studies 14-16, the "LLM decides with AAF's rules" configuration (B6, B6-GPT) gave its coordinator the written v1.0
governance policy (`revision_2026/run_llm_arms.py`, `GOVERNANCE_RULES`). It did not contain the two rules added in rule set
v1.4, release regression (R1) and security over cost (R2), which the ablation shows carry most of AAF's advantage. The
comparison of AAF with B6 therefore changed both who decides and which policy is applied (reviewer comments R1-1 and R4-1,
second round). This study gives the LLM decider the complete v1.4 policy so that only the decision maker differs.

## Arms (all other settings as B6 in Studies 14-16)
- **B6-14**: B6 with the complete v1.4 policy appended to the coordinator prompt: the v1.0 policy plus R1 and R2 in v1.4's
  priority order (`outcome_study/decide_b6v14.py`, `GOVERNANCE_RULES_V14`). Only the policy differs from B6.
- **B6-14E**: B6-14, and the coordinator also receives the raw evidence packet that AAF's rules read (after any tool result
  the agents retrieved). In B6 the coordinator sees only the agents' outputs, so it can apply R1 only if an agent reports the
  pre-release baseline; B6-14E removes that information gap as well.
- Models: GPT-5.6 Luna (Responses API, reasoning medium, one run per variant, as B6-GPT) and Qwen3-8B (Ollama 0.40.9,
  Q4_K_M, thinking on, temperature 0.7, three runs per variant, majority action, as B6). Decider labels: B6-14, B6-14E
  (Qwen3-8B); B6-14-GPT, B6-14E-GPT (GPT-5.6 Luna).
- Inputs: the committed decider inputs of each study (`<study>/decisions/variants.json`), full-evidence and withheld-evidence
  variants; outcomes are never read by the deciders. The agent prompts, coordinator prompt, tools, validation and outcome
  oracle are unchanged. Rule set v1.4 is unchanged.

## Hypotheses (fault cases of Studies 14-16, n = 104, O1, pooled; a missing decision counts as incorrect)
- **H2b-Q**: AAF (the pre-specified Qwen3-8B investigator) vs B6-14 and B6-14E; exact McNemar on paired case correctness,
  Holm over the two comparisons.
- **H2b-G**: AAF vs B6-14-GPT and B6-14E-GPT; exact McNemar, Holm over two.
- Family level, as for H1/H2: sign test over families of the same design on different services of one study, and cluster
  bootstrap 95% interval.
Any comparison may go against AAF, including no difference; every result is reported. The existing B6 results are kept and
labelled as using the v1.0 policy.

## Descriptive
Per-study counts; O2; B6-14 vs B6 (effect of the policy) and B6-14E vs B6-14 (effect of the evidence); agreement of each
arm's action with the v1.4 rules' own action (rule-following); GPT vs Qwen agreement per arm; withheld-evidence variants;
LLM calls, tokens and invalid outputs.

## Not covered
The prompt-injection test (Study 11b) and the three-model replication (Studies 10, 10b) are not rerun for B6-14; the paper
states this.

## Execution
GitHub Actions workflow `outcome-b6v14.yml`, control file `outcome_study/B6V14_RUN`. GPT runs stop at a per-study cap
(1,500 calls, 4,000,000 tokens). Runs are resumable; resumed runs use identical settings and are recorded below as
execution notes. Analysis: `python outcome_study/analyze_b6v14.py` (committed with this protocol, before any data).

## Execution notes (appended after the runs)
- GPT-5.6 Luna (run 42733599958, 2026-11-10): one launch per study; 232 of 232 decisions per arm without errors;
  780, 1,030 and 684 calls (Studies 14, 15, 16), below the cap.
- Qwen3-8B: 1,392 of 1,392 decisions (3 runs x 2 arms x 232 variants) without errors. The first run (42733683542) used a
  job-level time limit only; 5 Study-14 shards were stopped by it before committing, and their decisions were run again from
  the start by the next run (no committed decision was rerun or replaced). That run was then cancelled so that the
  remaining shards could restart under a step-level time limit that commits completed decisions first (workflow change
  70c336f). Further resumes with identical settings: runs 42775925839, 42897579056, 42933582155, 42947153761, 42966763286
  (control-file commits 1260fb2, 6fbe26d, 2955002, 2803e8a, ab3d425; `SKIP` lists shards still running in another run).
- About 30-38% of runs in every arm had at least one LLM output that failed the runtime's schema or evidence-reference
  checks (`analysis_b6v14.json`, `usage`); as for B6, such outputs are discarded by the runtime and the coordinator's final
  action is scored as given.
- No deviation from the pre-specified analysis.
