# P1 protocol lock addendum v0.3 — prospective decisions

**Status: proposed, NOT FROZEN.** This addendum supersedes conflicting draft language in PREREGISTRATION_AMENDMENT_V02.md only once signed off, code SHA recorded, and a public immutable tag/OSF timestamp is available **before** new scenario authoring or any confirmatory B4 execution. The existing 48 cases are development-only.

## Endpoint and effect threshold
**Primary estimand:** Difference in case-level agreement (B4 minus B2) on the **combined A+I** confirmatory cases. B4 is correct for a case when **at least 3 of its 5 independent runs** select an action in the adjudicated admissible set; B2 is correct if its single deterministic decision is admissible. Report paired percentage-point difference and a 95% paired CI; exact paired-binomial/McNemar-based CI for discordant pairs where feasible, with family-cluster bootstrap sensitivity and exact discordance counts. Never count runs as independent cases. Report B4 per-run mean and within-case variance as secondary descriptive outcomes. Report overall and stratum-specific effective n (unique scenario families/templates), not merely generated cases or repeated runs.

**Smallest effect of interest (SESOI):** 10 percentage points in either direction, provisional until expert/policy justification is recorded **before** benchmark construction. Interpretation uses the 95% CI against +10pp, 0, and -10pp; a positive point estimate alone does not establish meaningful benefit. If CI crosses meaningful-benefit and no-benefit regions, report inconclusive. No equivalence/noninferiority claim without a separately justified, powered design.

**Planned sample size (proposal, not yet frozen):** C=30, A=60, I=60, M=30 independent scenario families, 180 total; primary A+I n=120. This is a planning target contingent on feasibility and independence verification, not a claim of achieved power. Before freezing, run a paired-discordance scenario analysis and simulation to report the expected 95% CI width under several discordance rates (e.g., 5%, 10%, 20%) and sample sizes (A+I 60, 120, 200). If resources force fewer cases, amend **before** observing outcomes, report the resulting wider uncertainty, and do not imply precision not achieved.

## Benchmark and oracle
Scenarios are **synthetic**, even when external experts validate them; do not imply operational prevalence or real-world generalizability. Include A and M cases where the relevant evidence source **cannot** be inferred solely from a null/missing field; predefine a scenario-design rubric to avoid inadvertently making B2 fail by construction. Authors who draft scenarios must not run B4 on drafts, inspect B4 outputs, or revise cases based on model performance. Separate authoring from confirmatory execution; record provenance, template families, exclusions, and frozen split manifest.

Two blinded external experts independently generate primary action and **admissible action sets from scratch**, with no author labels or model predictions. **Disagreements on either primary action OR admissible set are resolved by Expert 3**, using a documented decision rationale and prespecified permitted-action vocabulary. Do not use intersection/union as a hidden fallback. Freeze the final Expert-3-adjudicated set before any confirmatory run. Report pre-resolution primary-action Cohen kappa and set-membership Jaccard, final set-size distribution, chance-level agreement under a stated null model, and effective independent n. At least one expert must have SRE/DevOps expertise and another PM/Delivery expertise. Obtain written consent, record any compensation/acknowledgment, conflicts of interest, department independence, and non-coauthor status.

## Comparisons and analysis
Primary: B4 vs B2 on A+I, as above.
Key secondary family (Holm across **three** tests): B4 vs B3 paired total token use; B4 vs B3 paired end-to-end wall-clock latency; B4 vs B6 paired case-level action agreement. Use paired cluster bootstrap CIs and a prespecified two-sided paired permutation or signed-rank test for continuous cost/latency measures (with zero/tie handling declared in analysis code); exact McNemar for B4 vs B6 binary case outcomes. All other comparisons are descriptive, with uncertainty and no significance claims.
M stratum: report descriptive unsupported-claim rate, hallucinated evidence references, unsafe actions, containment by deterministic authority, and failure rates, per arm and per case family. CLEAR, A, I, M all reported separately.

## Prespecified interpretation rules
- **A (incremental value):** B4 has a positive paired effect over B2; claim meaningful improvement only if its CI supports an effect at/above the +10pp SESOI and harms/costs are transparently reported. Otherwise characterize as suggestive/inconclusive.
- **B (no demonstrated incremental advantage):** B4 and B2 estimates overlap practically important positive/negative effects, or the interval is near zero; do **not** claim equality from nonsignificance. Narrow to authority separation, auditability and bounded containment **only where directly measured**.
- **C (inferiority):** B4 performs worse; disclose the negative result, failures and costs. Claim meaningful disadvantage only if CI supports a difference at/below -10pp.
- **D (strong agentic comparator):** If B6 or optional B7 matches/beats B4, do **not** claim hybrid architectural superiority. Any remaining contribution must be justified by measured containment, auditability, cost or reliability, not asserted.
All interpretations are subject to uncertainty and predeclared endpoints; no post-hoc case selection or oracle revision.

## Freeze and submission gates
1. Record frozen AAF rules SHA, B2 independently authored trigger-to-tool map SHA, protocol SHA, comparator prompts/model snapshots, budget and executor settings, benchmark-authoring rubric, sample-size/precision calculation, and planned grouping.
2. Publish timestamped **public Git tag or OSF preregistration** before new confirmatory scenario authoring; cite URL and date in reviewer response. Do not label a mutable branch draft as preregistered.
3. Check formal IEEE editorial decision category and resubmission mechanism.
4. Before resubmission: every reviewer comment answered; frozen protocol and independent adjudication complete; comparator runs reproducible; statistics and negative results reported; manuscript claims reconciled with data; AI assistance, author contributions and conflicts disclosed; final human-author approval obtained.

**External dataset policy:** RCAEval is ingestion-feasibility evidence only for this revision; M2 real-data work is parallel, not a blocker.
