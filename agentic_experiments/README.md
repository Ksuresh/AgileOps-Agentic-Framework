# Selective Agentic AAF - Experiments 2-4

The confirmatory protocol and dataset were frozen before any LLM execution. See `FROZEN_PROTOCOL.md` and `cases_v1.yaml`.

## Run

```bash
export OPENAI_API_KEY='...'
export OPENAI_MODEL='gpt-5.6-luna'
export OPENAI_REASONING_EFFORT='none'
python -m agentic_experiments.run_experiments_2_4
```

Outputs are written to `paper_results/agentic_experiments_2_4/`:

- `metadata.json` - model/configuration and frozen dataset SHA-256
- `raw_outputs.json` - deterministic, raw Agentic-only, tool, and Hybrid records
- `case_results.csv` - one row per independent case
- `summary.json` - Experiment 2-4 aggregate metrics

## Important

Do not edit `cases_v1.yaml`, the trigger, prompts, or governance policy after inspecting confirmatory model outcomes. Any later changes require a new versioned exploratory/replication dataset and must not be merged into the original confirmatory metrics.

The model never receives evaluator-only fields such as case ID, stratum, focus, oracle domains, admissible actions, expected tool, or hidden tool result. Tool results are exposed only after the model requests an allowed tool.
