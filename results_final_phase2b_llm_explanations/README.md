# Phase 2B LLM-Grounded Explanation Evaluation

This folder contains the frozen Phase 2B LLM-grounded explanation evaluation outputs for the AgileOps Agentic Framework.

## Evaluation Scope

Phase 2B evaluates the LLM explanation layer using structured outputs from the AAF pipeline.

The LLM is not used to make the governance decision. The flow is:

AAF structured output
-> LLM grounded explanation prompt
-> PM-facing explanation
-> consistency, evidence, hallucination-risk, XI, token, and cost scoring

## Model Configuration

- Model: gpt-4o-mini
- Temperature: 0.0
- Cases evaluated: 10
- LLM role: explanation generation only
- Decision source: deterministic AAF pipeline

## Key Results

- Domain consistency rate: 1.00
- Action consistency rate: 1.00
- Evidence coverage mean: 1.00
- Unsupported-claim risk proxy mean: 0.293
- XI mean: 0.588
- Prompt tokens: 11,164
- Completion tokens: 2,377
- Total tokens: 13,541
- Estimated cost: USD 0.0031

## Interpretation

The LLM preserved the AAF-selected domain and action across all evaluated cases. The result supports the use of an LLM as a grounded PM-facing explanation layer, while keeping decision logic under the deterministic AAF pipeline.

The lower XI score reflects the current XI scorer's reliance on explicit evidence phrase matching and trace labels. LLM explanations often paraphrase evidence, which may reduce automated XI scores despite maintaining domain/action consistency.

## Reproducibility

This run was generated with:

```bash
python -m llm_evaluation.run_llm_explanation_eval --limit 10
