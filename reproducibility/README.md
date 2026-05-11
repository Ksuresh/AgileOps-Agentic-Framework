# Expanded Controlled Evaluation: 120 Scenarios

This folder contains the expanded controlled evaluation results for the AgileOps Agentic Framework (AAF).   

## Scope

The evaluation contains 120 deterministic controlled governance scenarios generated from the original scenario set using severity variation, controlled metric jitter, missing-evidence simulation, and cross-domain ambiguity.

These scenarios are intended to strengthen reproducibility and controlled comparison. They should not be interpreted as 120 independent production incidents.

## Reproduction

Run from the repository root:

```bash
python -m experiments.run_all --out results_phase1_expanded_120 --n 120 --seed 42
python -m experiments.statistical_summary --results results_phase1_expanded_120
