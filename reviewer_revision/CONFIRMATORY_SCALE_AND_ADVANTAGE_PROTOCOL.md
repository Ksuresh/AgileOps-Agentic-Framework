# Confirmatory experiment design: scale and falsifiable Agentic advantage

Status: protocol proposal only; no held-out data, governance labels, or confirmatory outcomes collected.

## Hypothesis and estimand
Primary question: Does selective Hybrid investigation improve **independently adjudicated governance-action correctness** over the strongest prespecified deterministic retrieval baseline at a matched tool/evidence budget?

Null: paired correctness difference Hybrid minus D2 <= 0. Alternative: > 0. Report two-sided confidence intervals and avoid claiming benefit unless evidence supports it. Prespecify whether the inferential test is one-sided and its multiplicity treatment *before* test access.

## Cohorts and power
- Source inventory goal: >=1,000 eligible incident candidates across RCAEval (735 indexed), UCI incident history and other independently sourced datasets; do not count incident event rows as independent cases.
- Initial expert-labeling budget: 300-500 distinct incidents; actual sample size to follow a simulation-based paired/clustered power analysis using plausible discordant rates, not guessed from accuracy alone.
- Keep repeated injections of the same (system, fault, root-cause service) family in one split. Prefer external-system testing; document transportability limits.
- Primary benchmark: natural distribution of eligible cases, inclusion criteria locked before evaluating arms. Separate deliberately enriched stress suite, not pooled into primary effect estimate.

## Arms and parity
D0 (no retrieval), D1 (fixed missing-field mapping), D2 (adaptive deterministic retrieval), Hybrid selective, Hybrid always-on, Agentic-only. Freeze prompts, code revisions, model versions, tools, stopping rules and maximum tokens/calls. Every tool returns an independently curated response, not oracle-dependent simulator behavior. Report accuracy at matched tool-call budgets and at matched monetary/latency budgets; show accuracy-cost frontier.

## Preregistered analyses
- Primary: paired case-level governance action correctness, expert-adjudicated.
- Secondary: tool-selection utility, missing-evidence resolution, abstention/escalation calibration, latency, cost, tool failures, explanation faithfulness.
- Paired discordance table and McNemar-style inference when independent; clustered bootstrap/permutation by incident family/system for repeated cases; confidence intervals and effect sizes.
- Stratify prospectively by evidence regime: clear, missing, contradictory, multi-step dependency, misleading/stale, failure/distractor, cross-domain. Report stratum prevalence, not only success rates.
- Analyze subgroup interactions as exploratory unless independently powered. Do not create adversarial cases selected for Hybrid wins.
- Blinded dual governance expert adjudication, disagreements resolved by independent third reviewer; record agreement and label uncertainty.

## Falsification and publication claims
A null or negative result on the primary endpoint must be reported. If Agentic arms are no better than D2, test only prespecified secondary advantages (e.g. lower cost through selective trigger, better abstention, auditability) and frame paper claims accordingly. Architecture is a design proposal, not evidence of superiority.

## Engineering next steps
1. Repair UCI metadata retrieval: inspect live official endpoint and validate response schema, not an assumed API URL.
2. Identify RCAEval actual telemetry storage outside the Hugging Face compact index, then acquire a small audited sample.
3. Build per-tool independent response fixtures and held-out case mapping.
4. Obtain expert governance labels and freeze data splits, power and analysis plan.
5. Run all six arms, publish reproducible paired results.
