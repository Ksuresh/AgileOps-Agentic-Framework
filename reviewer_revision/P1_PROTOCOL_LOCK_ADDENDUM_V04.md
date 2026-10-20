# P1 Protocol Lock Addendum v0.4 — decisions requiring prospective freeze

**Status: committed proposal, NOT a frozen or registered protocol.** This document supersedes conflicting design proposals in v0.2/v0.3 once authors approve and the full preregistration is immutably tagged before confirmatory scenario authoring. Historical 48 cases remain exploratory development data.

## 1. Primary endpoint, estimand, and interpretation
Primary comparison B4 (selective Hybrid) minus B2 (deterministic mapped retrieval), **A+I combined**. The independent analysis unit is **one synthetic scenario per independently authored scenario family**; no family contributes multiple cases to the primary estimate. Templates, seeds, or paraphrases derived from the same parent are not additional independent families. Report counts of genuinely distinct templates/families and any provenance overlap.

Each stochastic arm (B3, B4, B5, B6, and B7 if run) is scored **correct on a case if >=3 of 5 runs** select a member of the expert-frozen admissible action set. B1 and B2 are evaluated once and undergo a two-run identical-input determinism check (same final action and tool-call trace; investigate any discrepancy before confirmatory execution). Report run-level means and within-case variance separately.

Primary estimand is the mean of paired binary B4-minus-B2 correctness differences across A+I independent families, in percentage points. Report exact paired discordance counts and 95% paired difference CI using a prespecified exact or score-based paired-binomial method, naming the method and implementation before freeze; report family-aware bootstrap as robustness only if any residual grouping remains. No repeated-run pseudo-replication.

**Smallest effect of practical interest (SESOI): +10 percentage points**, a *provisional operational design assumption* representing approximately one additional correct governance decision per ten uncertain cases, large enough to warrant investigation of extra LLM cost and operational risk. This is **not** an empirically established PM preference; expert feedback should be obtained prospectively, before freeze. Negative meaningful threshold is -10pp.

**Exhaustive outcome classification** using primary point estimate d and 95% CI [L,U]:
- **A, evidence of practically meaningful improvement:** d >= +10pp **and** L > 0. State that the interval may still include effects smaller than +10pp; do not claim the minimum effect is statistically established unless L >= +10pp.
- **B, positive but below practical threshold:** 0 < d < +10pp and L > 0; report a positive effect with uncertain practical importance. Example d=+6pp, CI [+1,+11] belongs here.
- **C, inconclusive or near-zero:** all results not covered by A, B, D, or E, including a CI crossing zero despite d >= +10pp; never call nonsignificance equivalence.
- **D, evidence of practically meaningful disadvantage:** d <= -10pp and U < 0. Do not claim minimum harm is statistically established unless U <= -10pp.
- **E, negative but below practical threshold:** -10pp < d < 0 and U < 0; report negative effect of uncertain practical importance.
Boundary d=0, L=0 or U=0, and any other ambiguous cases default to C. The outcome classification is descriptive and does not imply that a confidence interval proves an effect exceeds the SESOI.
**Architectural override:** if B6 or optional B7 matches or exceeds B4, do not claim hybrid architectural superiority regardless of primary classification. Auditability, containment, and cost claims require separate direct measurements.

## 2. Synthetic benchmark composition and precision gate
**Preferred target:** 180 independent families: CLEAR 30, AMBIGUOUS 60, INCOMPLETE 60, MISLEADING 30; primary A+I n=120.
**Minimum viable contingency:** 120 independent families: CLEAR 20, AMBIGUOUS 40, INCOMPLETE 40, MISLEADING 20; primary A+I n=80. The minimum is a **feasibility contingency, not a powered confirmatory guarantee**. Freeze a decision rule for choosing target versus minimum **before viewing model outcomes**, using expert capacity and a prospective precision simulation. Report expected paired CI widths for n=80 and n=120 under discordance rates 5%, 10%, and 20% before final lock. If minimum not reached, do not claim a confirmatory benchmark.

**INCOMPLETE composition:** exactly 50% mapped-null retrieval cases (the missing field unambiguously maps to a tool) and 50% nontrivial investigation cases (missing or conflicting evidence does not uniquely imply the useful tool). Thus preferred I=30+30; minimum I=20+20. Both I subgroups reported separately. AMBIGUOUS and MISLEADING strata also contain cases where tool selection is not implied by a null field. Neither subtype is designed to guarantee a winner; cases must pass expert plausibility review.

All cases are explicitly **synthetic**. No author who constructs a confirmatory case may inspect B4 outputs on that case or change it based on model results. Record provenance, source templates, exclusions and effective independent n; do not pool the old 48.

## 3. Expert workflow and realistic scheduling
Author scenarios in prospectively defined batches and send rolling batches to blinded experts after the rubric and protocol freeze. Two independent experts (PM/Delivery and SRE/DevOps) write primary actions and admissible sets from scratch, blind to author sets and model outputs. Third independent adjudicator resolves both action and set disagreements, documenting rationale. Obtain consent, compensation/acknowledgment choice, conflicts and independence declaration. Report kappa for primary action and Jaccard for admissible-set membership, as well as chance-level agreement and final set sizes.

At 5–10 minutes each, 180 cases require roughly 15–30 hours **per expert**, excluding disagreements and authoring. Accordingly the prior Days 4–10 expert completion assumption is unrealistic. Begin booking reviewers immediately, deliver rolling batches, and set the actual experimental start by completed adjudication rather than an artificial day count. Do not run confirmatory arms on unadjudicated cases.

## 4. Remaining P1 freeze checklist
- Record immutable AAF rules SHA and blindly authored B2 map SHA; use same z(I) trigger, tools, evidence, and tool budget.
- Record exact paired CI method/software, precision simulations, case-family provenance rule, tool/LLM configuration, and reproducible seeds.
- Specify paired continuous-outcome test for B4/B3 tokens and latency and paired binary test for B4/B6 agreement; Holm correction across those three key secondary tests. MISLEADING unsupported-claim/containment outcomes descriptive.
- Obtain author approval of operational 10pp SESOI justification and chosen target/minimum contingency before freeze.
- Publish public immutable Git tag or OSF preregistration with timestamp **before P2 authoring** and cite in reviewer response.
- Resubmission gate: all reviewer requests mapped to evidence or limitation; expert adjudication complete; runs and statistical reports reproducible; manuscript claims match outcomes; disclosures and human author signoff.

RCAEval remains ingestion-feasibility-only for this revision. External real-incident acquisition continues in parallel and is not a submission prerequisite.
