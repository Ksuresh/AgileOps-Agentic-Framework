# Phase 1 Controlled Scenario Evaluation

This folder contains the frozen Phase 1 controlled evaluation outputs for the AgileOps Agentic Framework.

## Evaluation Scope

Phase 1 evaluates the AAF reasoning pipeline using 30 deterministic controlled scenarios across:

- DevOps deployment failures
- SRE reliability degradations
- FinOps cost anomalies
- DevSecOps security and compliance risks
- Cross-domain governance trade-offs

The evaluation compares:

- Traditional baseline
- Single-agent LLM baseline
- AAF full pipeline
- AAF without consensus
- AAF without RAR
- AAF without utility

## Key Results

- AAF primary-domain accuracy: 0.80
- AAF action match rate: 0.80
- AAF consensus mean: 0.74
- AAF composite utility: 0.81
- AAF explainability index: 0.81

## Utility Score

The utility score is decomposed into:

- operational performance
- cost efficiency
- risk reduction

The composite utility score represents the weighted governance value of the selected action.

## RAR Note

RAR is evaluated as an uncertainty and evidence-retrieval mechanism. In this controlled evaluation, RAR detected low-consensus cases, but resolved only a limited subset.

Frozen Phase 1 RAR results:

- RAR triggered: 19 / 30
- RAR accepted: 2 / 19
- RAR unresolved: 17 / 19
- RAR trigger rate: 0.63
- RAR acceptance rate when triggered: 0.11

## Reproducibility

This run was generated with:

```bash
python -m experiments.run_all --out results_v2_final_phase1 --n 30 --seed 42
