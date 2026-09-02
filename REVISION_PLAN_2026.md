# Post-review revision plan (2026)

This branch repairs the experimental design before manuscript resubmission.

## Scientific positioning

The revised manuscript should describe AAF as an **explainable, evidence-grounded multi-agent governance decision-support framework**. The four domain agents are deterministic evidence-reasoning agents. The LLM is used only for constrained explanation generation and is not claimed to perform autonomous planning or action selection.

Rename **Re-Grounded Agentic Reasoning (RAR)** in the manuscript to **Re-Grounded Evidence Reasoning (RER)**. Code identifiers may be migrated separately to avoid breaking frozen results.

## Required experiment changes

1. Preserve the 120 controlled scenarios, but treat their embedded labels as designer/reference labels rather than independently validated ground truth.
2. Obtain two independent blinded annotations for primary domain and governance action. Adjudicate disagreements and freeze the resulting expert gold standard.
3. Repeat the same blinded annotation procedure for the PM-prompt benchmark.
4. Report Cohen's kappa before adjudication.
5. Replace normal-approximation proportion intervals with Wilson intervals.
6. Use paired exact McNemar tests for action/domain comparisons where predictions are available case-by-case.
7. Replace the old `without utility -> defer` ablation with a genuine dominant-domain severity baseline that does not use utility ranking.
8. Add sensitivity analyses for lambda, consensus threshold tau, minimum improvement delta, and utility weights.
9. Retain GPT-4o-mini explanation generation with temperature 0 and publish the complete constrained prompt. Add a small human evaluation of groundedness, clarity, action consistency and PM usefulness.
10. Describe PM prompt conversion as **prompt-derived evidence mode**, not blind evidence mode.
11. Describe controlled RER evidence additions as deterministic simulated evidence enrichment, not live production retrieval.
12. Treat the current Sock Shop experiment as a three-case runtime-artifact feasibility pilot unless a larger experiment is newly generated and frozen. Distinguish raw artifacts from derived telemetry and cost proxies.

## Added on this branch

- `orchestrator/decision_baselines.py`: meaningful non-utility dominant-domain baseline.
- `evaluation/statistical_methods.py`: Wilson CI, Cohen's kappa and exact McNemar utilities.
- `evaluation/export_annotation_templates.py`: blinded annotation-sheet exporter that omits existing labels and AAF predictions.

## Human annotation protocol

Each annotator receives only case text/evidence. Existing expected labels and AAF predictions must remain hidden. Each annotator records primary domain, optional secondary domains, governance action, confidence (1-5), and notes. Agreement is calculated before adjudication. A final adjudicated CSV is then frozen and used for all reported accuracy/action-match results.

## Claims to avoid

- Do not call the deterministic domain reasoning loop an LLM-based agentic AI system.
- Do not describe designer labels as independently validated ground truth.
- Do not claim direct cloud cost measurement when replica/resource footprint is used as a proxy.
- Do not claim 20 runtime cases / 733 artifacts unless those data and results are reproducibly present in the repository.
- Do not interpret the historical `without utility` result as causal evidence for utility ranking.
