# Phase 2A PM Prompt Evaluation

This folder contains the frozen Phase 2A PM prompt evaluation outputs for the AgileOps Agentic Framework.

## Evaluation Scope

Phase 2A evaluates the Project Manager-facing prompt interaction layer using 50 governance prompts.

The evaluation tests whether natural-language PM prompts can be routed into operational context and processed through the AAF pipeline.

The flow is:

PM prompt
-> prompt routing
-> telemetry construction
-> domain-agent inference
-> consensus scoring
-> RAR when needed
-> utility-based action selection
-> PM-readable explanation
-> XI scoring

## Key Results

- Number of PM prompts: 50
- Domain match rate: 0.94
- Action match rate: 0.66
- Consensus mean: 0.698
- Composite utility: 0.788
- Operational performance mean: 0.420
- Cost efficiency mean: 0.714
- Risk reduction mean: 0.560
- Explainability index: 0.819
- RAR trigger rate: 0.66
- RAR acceptance rate when triggered: 0.515

## Interpretation

This evaluation validates the PM-facing prompt workflow. It shows that the framework can convert PM governance questions into structured operational reasoning and return actionable explanations.

## Important Note

Phase 2A is a prompt evaluation, not a production telemetry validation. Realistic telemetry validation is planned separately as Phase 2B.

## Reproducibility

This run was generated with:

```bash
python run_pm_prompt_experiments.py
