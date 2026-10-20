# RCAEval RE2: preregistration-oriented cohort decision (20 October 2026)

## Evidence from executed GitHub audit
The source index (SHA-256 c49a288920dbba2e8e724679a14636d5c7eb2b45426bba14007ef79a6c0ab1bb) contains 735 cases, including RE2-OB 90, RE2-SS 90, RE2-TT 90. For RE2, logs/traces: OB 90/90, SS 90/0, TT 89/90. These are *availability indicators*, not proof of evidence sufficiency. Audit: https://github.com/Ksuresh/AgileOps-Agentic-Framework/actions/runs/39414929658/job/118601525676

## Decision: candidate cohort, not held-out benchmark
Prioritize RE2 (270 indexed incidents) for *feasibility and telemetry inspection*, because it exposes heterogeneous evidence modalities. Keep RE1 (375) as metrics-only comparator and RE3 (90) as exploratory extension. Do not claim all 270 have all modalities.

## Gates before any confirmatory test
1. Inspect actual telemetry for at least one case per RE2 system and fault family. Verify real metric/log/trace file structure, timestamps, license, missingness, and per-tool availability. No model predictions.
2. Establish a documented mapping from technical telemetry to **observable governance decision contexts**; some cases may not support meaningful project governance decisions and must be excluded with reasons recorded before results.
3. Obtain independent governance action labels, with blinded dual review and disagreement resolution. RCAEval's root-cause labels are NOT governance labels.
4. Freeze action space, evidence visibility, per-tool responses (including no-result and failure), cost/budget accounting, leakage-resistant grouping, development/holdout partition and sample-size justification.
5. Ensure held-out cases do not share fault-family repetitions with tuning cases. Prefer cross-system testing where supported; report system-specific performance and evidence-modality effects.
6. Compare D0, D1, D2, Hybrid-selective, Hybrid-always and Agentic-only on identical evidence/tool budgets. No post-hoc prompt selection using held-out results.
7. Report negative findings, paired confidence intervals, clustered analysis, latency/cost, tool calls, and failure modes.

## Immediate next implementation
- Add a bounded telemetry inspection workflow for RE2 that fetches **one case per system**, checks metadata/file availability, and stores hashes and schema summaries, without publishing sensitive raw logs.
- Audit UCI Incident Management Process as a separate governance-centric source.
- Only after source feasibility is verified, design expert adjudication and freeze evaluation.

## Interpretation
The existing 48 cases are development-only. D1/D2 and original Hybrid each obtained 41/48; no incremental Agentic advantage has been demonstrated. Architectural soundness is not an empirical result.
