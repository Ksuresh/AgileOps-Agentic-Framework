# Post-review Revision Plan (Historical Development Record)

> **Status:** Historical planning document. This file records the deterministic/post-review stage that preceded the final Hybrid AAF manuscript. It is **not** the current reviewer-facing description of the architecture or validation package. For the current paper use `README.md`, `REPRODUCE_CURRENT_PAPER.md`, `paper_results/README.md`, and `paper_results/MANUSCRIPT_RESULT_MAP.md`.

This branch originally repaired the deterministic experimental design before the later selective-Agentic extension was frozen.

## Historical scientific positioning

At this stage AAF was described as an explainable, evidence-grounded multi-domain governance decision-support framework. The four domain analyzers used deterministic evidence rules and the LLM was used only for constrained explanation generation. The later Hybrid extension added bounded selective Agentic Evidence Investigation (AEI) while retaining deterministic final governance authority.

Earlier drafts also used **Re-Grounded Agentic Reasoning (RAR)** and later **Re-Grounded Evidence Reasoning (RER)**. The final manuscript separates these concepts more precisely:

- **AEI** — LLM-mediated Agentic Evidence Investigation and approved evidence acquisition.
- **Evidence re-grounding/reassessment** — deterministic processing after evidence acquisition.

Historical code identifiers and frozen outputs are not rewritten merely to update terminology.

## Validation strategy established in this revision stage

The revised controlled study does not use the scenario generator's embedded designer labels as independent ground truth. It uses pre-specified experimental oracles tied to controlled interventions.

The benchmark contains 30 base experimental conditions. Four deterministic seeded/noisy variants (default seeds 42–45) yield 120 executions. The base condition, causal domain, affected domains, admissible governance action(s), and oracle rationale are specified independently in `benchmark/oracle_specs.py` before AAF execution.

Reported controlled metrics are therefore **oracle agreement**, not real-world accuracy. The controlled benchmark establishes internal validity, robustness and reproducibility; it does not establish production accuracy.

## Changes introduced at this stage

1. Use the 30 base controlled conditions and four seeded variants as the primary controlled benchmark.
2. Evaluate against the pre-specified oracle specification, not the scenario generator's `ground_truth` field.
3. Report domain-oracle agreement and action-oracle agreement with Wilson 95% confidence intervals.
4. Use paired exact McNemar tests for comparisons where predictions are available case-by-case.
5. Replace the old `without utility -> defer` ablation with a genuine dominant-domain severity baseline that does not use utility ranking.
6. Add sensitivity analyses for consensus/decision parameters and utility weights.
7. Evaluate deterministic evidence enrichment/reassessment under controlled missing-evidence/noise conditions.
8. Publish the constrained explanation prompt and evaluate explanation behavior separately from decision correctness.
9. Treat PM-prompt experiments separately from governance-decision correctness.
10. Describe controlled evidence additions as simulated evidence enrichment, not live production retrieval.
11. Avoid generalizing small runtime pilots beyond the evidence actually collected.

## Components added during this stage

- `orchestrator/decision_baselines.py` — non-utility dominant-domain baseline.
- `evaluation/statistical_methods.py` — Wilson CI, Cohen's kappa and exact McNemar utilities.
- `evaluation/export_annotation_templates.py` — optional blinded annotation exporter.
- `benchmark/oracle_specs.py` — pre-specified experimental oracle definitions.
- `evaluation/run_oracle_benchmark.py` — seeded controlled execution against the oracle.

## Interpretation boundaries retained in the final work

- Do not describe designer labels as independently validated ground truth.
- Do not describe oracle agreement as real-world or production accuracy.
- Do not claim direct cloud cost measurement when replica/resource footprint is used as a proxy.
- Do not interpret historical reduced-model results as stronger causal evidence than their experimental design supports.
- Do not silently modify frozen datasets, protocols, prompts, or results after outcome inspection.

The current Hybrid architecture, selective AEI studies, prospective replication, RCAEval validation, bounded PM-interface study, and their frozen provenance are documented in the reviewer-facing files linked at the top of this document.