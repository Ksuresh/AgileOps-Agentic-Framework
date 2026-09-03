# Post-review revision plan (2026)

This branch repairs the experimental design before manuscript resubmission.

## Scientific positioning

The revised manuscript should describe AAF as an **explainable, evidence-grounded multi-domain governance decision-support framework**. The four domain analyzers use deterministic evidence rules. The LLM is used only for constrained explanation generation and is not claimed to perform autonomous planning or action selection.

Rename **Re-Grounded Agentic Reasoning (RAR)** in the manuscript to **Re-Grounded Evidence Reasoning (RER)**. Code identifiers may be migrated separately to avoid breaking frozen results.

## Validation strategy

The revised controlled study does **not** use the scenario generator's embedded designer labels as independent ground truth. Instead, it uses **pre-specified experimental oracles** tied to controlled interventions.

The benchmark contains 30 base experimental conditions. Four deterministic seeded/noisy variants (default seeds 42-45) yield 120 executions. The base condition, causal domain, affected domains, admissible governance action(s), and oracle rationale are specified independently in `benchmark/oracle_specs.py` before AAF execution.

Reported metrics should therefore be called **oracle agreement**, not real-world accuracy. The controlled benchmark establishes internal validity, robustness and reproducibility; it does not establish production accuracy.

## Required experiment changes

1. Use the 30 base controlled conditions and four seeded variants as the primary controlled benchmark.
2. Evaluate against the pre-specified oracle specification, not the scenario generator's `ground_truth` field.
3. Report domain-oracle agreement and action-oracle agreement with Wilson 95% confidence intervals.
4. Use paired exact McNemar tests for comparisons where predictions are available case-by-case.
5. Replace the old `without utility -> defer` ablation with a genuine dominant-domain severity baseline that does not use utility ranking.
6. Add sensitivity analyses for lambda, consensus threshold tau, minimum improvement delta, and utility weights.
7. Evaluate RER under controlled missing-evidence/noise conditions and report trigger, acceptance and decision-stability behavior.
8. Retain GPT-4o-mini explanation generation with temperature 0 and publish the complete constrained prompt. Treat explanation evaluation separately from decision correctness.
9. Reposition the PM-prompt experiment as a **prompt-derived evidence use-case/demonstration** unless an independent oracle can be specified without circular mapping.
10. Describe controlled RER evidence additions as deterministic simulated evidence enrichment, not live production retrieval.
11. Treat the current Sock Shop experiment as a three-case runtime-artifact feasibility pilot unless a larger experiment is newly generated and frozen. Distinguish raw artifacts from derived telemetry and cost proxies.

## Added on this branch

- `orchestrator/decision_baselines.py`: meaningful non-utility dominant-domain baseline.
- `evaluation/statistical_methods.py`: Wilson CI, Cohen's kappa and exact McNemar utilities. Kappa remains available but is not required by the oracle-based controlled study.
- `evaluation/export_annotation_templates.py`: optional blinded annotation exporter retained for future external validation; it is no longer required for the primary controlled study.
- `benchmark/oracle_specs.py`: pre-specified experimental oracle definitions for all 30 base conditions.
- `evaluation/run_oracle_benchmark.py`: evaluates seeded controlled executions against the oracle and reports Wilson intervals.

## Interpretation architecture

**Controlled benchmark:** intervention -> generated evidence/noise -> pre-specified oracle -> AAF execution -> oracle agreement.

**Robustness study:** controlled missing evidence/noise -> RER enrichment/re-evaluation -> decision stability/recovery.

**Comparative study:** full AAF decision policy vs dominant-domain baseline and ablations, with paired statistics.

**PM prompts:** prompt-derived evidence demonstration; do not present designer-labelled prompt matching as independent accuracy validation.

**Sock Shop:** runtime-artifact feasibility evidence; do not generalize three cases to production accuracy.

## Claims to avoid

- Do not call the deterministic domain reasoning loop an LLM-based agentic AI system.
- Do not describe designer labels as independently validated ground truth.
- Do not describe oracle agreement as real-world or production accuracy.
- Do not claim direct cloud cost measurement when replica/resource footprint is used as a proxy.
- Do not claim 20 runtime cases / 733 artifacts unless those data and results are reproducibly present in the repository.
- Do not interpret the historical `without utility` result as causal evidence for utility ranking.
