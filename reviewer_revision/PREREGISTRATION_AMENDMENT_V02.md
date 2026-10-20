# IEEE Access reviewer revision: preregistration amendment v0.2

**Status: prospective design amendment, not executed or retrospectively preregistered.** The existing 48 cases and their observed D0/D1/D2/Hybrid outcomes are exploratory development data. This document must be frozen with a commit hash **before** creating new confirmatory cases or modifying AAF decision rules. Any subsequent deviations require timestamped amendment and disclosure. The final frozen protocol must also record the AAF ruleset commit SHA and all comparator implementation hashes.

## Primary estimand and analysis
- **Primary comparison:** B4 selective Hybrid versus B2 deterministic plus rule-based evidence retrieval on **I + A** strata of the new, independently adjudicated benchmark.
- **Primary endpoint:** per-case proportion of runs whose final governed action belongs to the frozen expert-adjudicated admissible set, with a paired B4-minus-B2 difference. For stochastic B4, average correctness across its prespecified five runs per case; B2 is evaluated once. Aggregate by independent incident/template family, not by repeated runs. Report paired point estimate and cluster-aware 95% CI; additionally report case-level discordance using a prespecified binary summary (majority-correct over five runs; ties cannot occur with five runs), with exact McNemar descriptive p-value. No significance-based superiority claim absent adequate power.
- **Estimation-first framing:** perform a prospective McNemar power/sensitivity calculation using plausible discordance rates, explicitly acknowledge that small effects are likely underpowered. Do **not** claim noninferiority unless a clinically/operationally defensible margin, design, and sufficient sample size are frozen before execution.
- **Key secondary comparisons:** B4 vs B3 on total LLM tokens and wall-clock latency; B4 vs B6 on final governed-action agreement. Use Holm correction across the resulting three key secondary tests (tokens, latency, agreement). Other outcomes/strata/arms are descriptive/exploratory, with estimates and uncertainty.
- Report CLEAR, AMBIGUOUS, INCOMPLETE, MISLEADING strata separately, including zero/negative effects and uncertainty.

## Frozen comparator definitions
B1 deterministic AAF; B2 deterministic AAF plus a **blindly authored** field/trigger-to-tool retrieval map without LLM; B3 always-on LLM advice under deterministic final authority; B4 selective Hybrid; B5 agentic-only current settings; B6 agentic-only with reasoning enabled and governance policy in prompt. B7 external ReAct-style agent and second model family are **optional exploratory extensions**, not prerequisites for the primary result. Document omissions explicitly.

B2 must use the same z(I) trigger, accessible evidence, tool set, tool-call budget, and deterministic final governance as B4. Derive B2 mapping from published trigger/policy semantics, **without inspecting new benchmark outcomes or B4 recoveries**. Freeze its mapping before confirmatory execution; disclose authorship and any prior knowledge from the development 48.

Freeze AAF arbitration rules and their commit SHA before constructing new benchmark cases. Prior refinements informed by the 48 development cases must be disclosed. No tuning against new held-out labels. Never pool the historical 48 with the confirmatory cohort.

## Repeat and resource protocol
Use five independent runs per case for stochastic LLM arms, with fixed declared seed where supported and recorded model snapshot, reasoning effort, temperature, prompts, tool schema, and execution settings. Score stochastic arms by **per-case mean across five runs** for the primary estimand, and additionally report majority-correct and within-case variance. Deterministic B1/B2 run once with a reproducibility spot check. Report tokens, LLM calls, tool calls, failures, end-to-end wall-clock latency, and monetary cost per case and per run. No pseudo-replication.

## Independent expert benchmark
At least two raters blind to model predictions and **author labels** independently write scenario framing assessments, primary action, and admissible action sets **from scratch**. Include at least one SRE/DevOps expert and one PM/Delivery expert. Raters must not be co-authors or from the same department; disclose expertise, independence and written consent. Resolve disagreements with a named third independent adjudicator or a prespecified documented consensus procedure. Report Cohen's kappa on primary action, set-membership Jaccard agreement (and Krippendorff alpha where applicable), disagreement counts, and admissible-set sizes. Freeze final labels before model evaluation. Include realistic ambiguous/misleading scenarios where tool choice is not reducible to a missing-field lookup. Record provenance and template/fault-family grouping. Obtain prospective power estimates before choosing sample size.

## Prespecified interpretation
1. If B4 exceeds B2 with credible evidence, report bounded incremental benefit and costs.
2. If B4 and B2 are indistinguishable, **do not** claim equivalence or superiority; evaluate authority separation, auditability and containment only if measured.
3. If B4 underperforms B2, report the negative result and its causes.
4. If B6 or B7 matches/exceeds B4, **do not** claim hybrid architectural superiority; any claim narrows to independently measured auditability, containment and cost. Distinguish architecture-level comparison from isolating a single component.

## Manuscript and response requirements
- Define delta_min, RER, B(a); justify tau, utility weights and consensus penalty; boundary-case sensitivity.
- Effective independent n: Study 1 has 30 templates, not 120 independent scenarios; RCAEval has 26 fault-service pairs, not 78 independent cases. Report admissible-set sizes and chance-level agreement.
- **RCAEval is ingestion feasibility only** in this revision; remove unsupported selectivity and governance-oracle claims.
- PM interface: main-text exact matches 0.37/0.54, post-hoc rescoring, five-record limitation, 0.281 unsupported-claim interpretation, and baseline zero-match explanation.
- Explain four unhelpful overrides, circularity of author oracle, and put oracle limitations adjacent to main results.
- State that Hybrid vs agentic-only compares whole architectures. Report exact model snapshots, temperatures, prompts, compute, cost, and runtime in one reproducibility section.
- Update related work with recent agentic SE/AIOps studies and a measurable differentiation table.
- Revise title and abstract to match evidence, include scalability, failures, telemetry gaps, and policy maintenance; simplify figures and prose.
- Disclose AI assistance in editing, code and analysis as applicable, following journal policy; prepare cover letter and reviewer-by-reviewer response matrix.
- Check formal editorial decision type and journal submission instructions. Do not infer category or a deadline from reviewer wording.
- Confirm author contributions and acknowledgments with the human authors; AI tools are not authors.

**Operational gate:** no confirmatory data generation or runs until frozen AAF SHA, B2 mapping, expert instructions, endpoint and aggregation, split/grouping protocol, and this amendment are committed. This document itself is a draft until those hashes and details are filled.
