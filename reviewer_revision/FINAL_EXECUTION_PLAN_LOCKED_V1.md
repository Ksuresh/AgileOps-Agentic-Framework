# FINAL EXECUTION PLAN — LOCKED v1.0

**Status:** Final project execution plan; no further planning iterations. **Scientific preregistration status:** NOT YET FROZEN. The immutable public tag/OSF registration, AAF and B2 hashes, expert materials, and precision verification must be completed as P1 implementation tasks **before** confirmatory scenario authoring. Do not conflate locking this execution plan with registering the study.

## Objective and sequencing
Produce the earliest scientifically defensible IEEE Access revision, without assuming selective Hybrid AAF superiority. P1 protocol freeze precedes P2 confirmatory scenario authoring; P3 comparator implementation and P6 manuscript corrections can proceed in parallel. P4 experiments require adjudicated frozen benchmark and comparator verification; P5 statistical analysis follows P4; P7 final submission gate follows evidence and manuscript audit. External incident data ingestion remains a parallel non-gating activity.

## P1 — fixed statistical specification
- **Primary:** B4 versus B2 on combined A+I, one independently authored synthetic case per independent family. Stochastic arms B3/B4/B5/B6 (optional B7) correct when >=3 of 5 runs choose a frozen admissible action; deterministic B1/B2 run once and must pass identical-input duplicate checks.
- **Primary estimand:** paired case-level correctness difference B4 minus B2, percentage points. **95% Tango paired-proportions score confidence interval**, R `PropCIs::scoreci.mp`; record paired 2x2 discordance table and **exact McNemar test** via R `exact2x2` package. Pin R/package versions and code, verify function invocation and parameter order before freezing software manifest. No repeated-run pseudo-replication.
- **SESOI:** +/-10 percentage points as a prospective operational assumption (~one additional correct governance action per ten uncertain cases); author approval and prospective expert input required, not presented as an empirically measured PM preference.
- **Outcome A:** d >= +10pp and CI lower bound > 0; evidence of positive effect with practically relevant point estimate, **not** proof the true effect exceeds 10pp.
- **Outcome B:** 0 < d < +10pp and CI lower bound > 0; positive, below practical point-estimate threshold.
- **Outcome C:** all other cases, including CI touching/crossing zero and d=0; inconclusive, never evidence of equivalence.
- **Outcome D:** d <= -10pp and CI upper bound < 0; negative effect with practically relevant point estimate.
- **Outcome E:** -10pp < d < 0 and CI upper bound < 0; negative, below practical point-estimate threshold.
- **Architectural comparator:** B4 minus B6 action agreement on **A+I**, using the same majority vote, Tango CI and A–E outcome classification. **Only A or B permits a narrowly worded claim of superior action agreement over B6**; C/D/E prohibit a hybrid architectural-superiority claim. If optional B7 is run, apply the same restriction against B7. No overall architectural superiority claim without directly measured and reported cost, reliability and containment.
- **Three prespecified key secondary tests (Holm correction):** (1) B4 vs B3 total tokens **over C+A+I+M**; (2) B4 vs B3 end-to-end wall-clock latency **over C+A+I+M**; (3) B4 vs B6 case-level action agreement **on A+I**. For (1)/(2) use paired **10,000-resample case-level bootstrap** CI, retaining zero paired differences; define case-level aggregation of five stochastic runs as mean per case for continuous endpoints, and fix bootstrap RNG seed and percentile/BCa choice in analysis code before run. For (3) use Tango CI and exact McNemar. Declare associated p-value procedure for (1)/(2) before claiming Holm-adjusted significance (bootstrap CI alone is not a p-value). Report uncertainty and avoid interpreting Holm-adjusted significance as practical importance.
- **M stratum:** descriptive unsupported claims, hallucinated evidence references, unsafe actions, and deterministic containment by arm; all four strata reported separately.

## P2 — synthetic benchmark and expert process
Preferred independent families: C30/A60/I60/M30 (total 180, primary n=120); minimum contingency C20/A40/I40/M20 (total 120, primary n=80), selected prospectively based on capacity/precision, not observed results. I exactly 50% simple mapped-null and 50% nontrivial tool-selection, report separately. Include A/M cases not solvable by null-field mapping. No author runs B4 on draft confirmatory scenarios. Do not pool 48 exploratory cases. Freeze provenance and effective independent n.

Two external independent experts (PM/Delivery and SRE/DevOps) blindly author admissible sets and primary actions; third independent adjudicator resolves **both** set and action disagreements. Obtain consent, COI, compensation/acknowledgment preferences and independence declarations. Report Cohen kappa, Jaccard, chance agreement, set sizes, effective n. Send batches on rolling schedule only **after** protocol registration. Preferred 180-case workload is ~15–30 hours **per expert**, excluding adjudication.

**Precision warning to authors:** at primary n=80 with ~15% discordance, 95% paired CI half-width is approximately 8–9pp (scenario-dependent); small positive differences may yield Outcome C. This is an acceptable predeclared result. Compute and archive actual scenario-based expected widths for n=80/120 and multiple discordance rates before final preregistration.

## P3 — comparators
B1 deterministic; B2 strong no-LLM mapped retrieval (same z(I), evidence, tools, budget as B4, independent map); B3 always-on LLM advice with deterministic authority; B4 selective Hybrid; B5 original agentic-only; B6 reasoning-enabled agentic-only with governance policy; B7 external ReAct optional. Freeze exact code, model snapshots, prompts, temperatures, tools, budgets, seeds, logging. No case-specific tuning against confirmatory benchmark.

## P4/P5 — runs and inference
Only after frozen benchmark/adjudication and tested arms: execute 5 repeats for stochastic arms, 1 for deterministic plus determinism checks. Log actions, tool traces, token use, wall-clock latency, cost, unsupported claims, failures, and settings. Analyze primary paired differences, three key secondary tests, strata and I subtypes, M containment, uncertainty, costs, failure modes and negative results. Preserve all runs, including errors.

## P6 — manuscript/reviewer responses
Begin immediately. Revise unsupported title/abstract claims; define delta_min, RER, B(a), tau and utility weights; correct effective n (Study 1 30 templates; RCAEval 26 fault-service pairs); restrict RCAEval to ingestion feasibility; report PM-interface exact-match 0.37/0.54, post-hoc rescoring, five-record limitation, unsupported-claim score 0.281, baseline zero action matches, four unhelpful overrides and oracle circularity; document model/compute/runtime/cost; update recent related work, limitations, AI-use disclosure, reviewer matrix and cover letter. No claim that old Hybrid 41/48 beats D2 41/48.

## P7 — definition of done / resubmission gate
(1) Public immutable preregistration before confirmatory authoring; (2) versioned AAF/B2 and fair comparator configurations; (3) independent blinded expert adjudication; (4) effective n/provenance verified; (5) complete reproducible runs and prespecified statistics; (6) all negative/inconclusive results and limitations reported; (7) all six reviewer responses traced to manuscript changes/evidence; (8) title/abstract/tables/conclusions internally consistent; (9) AI, authorship and COI disclosures complete; (10) editorial decision mechanism checked; (11) human authors approve final submission.

## Immediate execution priority
1. Recruit independent experts and reserve sufficient time.
2. Create blinded expert assessment pack (scenario form, permitted action vocabulary, consent, COI, compensation/acknowledgment, disagreement rubric).
3. Finish P1 implementation: code hashes, precision calculation, test software manifest, preregistration publication.
4. Implement comparator arms and manuscript edits in parallel.
**Do not iterate this action plan further; manage changes, if truly necessary, as explicitly timestamped preregistration deviations.**
