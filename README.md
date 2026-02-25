# AAF Reproducibility Artifact

This repository contains the configuration, synthetic scenario generator, deterministic agent logic, orchestration (consensus + RAR), utility scoring, and an optional LLM explanation runner consistent with the manuscript:

**AgileOps Agentic Framework (AAF): A Multi-Agent and LLM-Driven Governance System for Operational Visibility in Agile-DevOps Ecosystems**  
Manuscript: Access-2025-54890

## What this artifact provides
- `config/config.yaml`: fixed thresholds (τ, Δmin), utility weights, and LLM decoding parameters.
- `scenario_generator/`: generates 30 synthetic cross-domain scenarios with controllable noise.
- `agents/`: deterministic agent logic for DevOps, SRE, FinOps, DevSecOps.
- `orchestrator/`: consensus scoring, RAR loop, action selection.
- `instrumentation/`: latency logging hooks and result summarization.
- `prompts/`: system prompt and user template (Appendix A).
- `examples/`: one fully serialized input + expected structure.

## Quickstart (synthetic evaluation only)
1. Create a Python environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Generate scenarios + run the pipeline (without calling an LLM):
```bash
python run_experiments.py --config config/config.yaml --no-llm
```

This produces:
- `results/run_<timestamp>/scenario_outputs.jsonl`
- `results/run_<timestamp>/metrics_summary.json`
- `results/run_<timestamp>/latency.csv`

## Optional: LLM explanation (Mistral via llama.cpp)
If you have a local llama.cpp binary and a compatible GGUF model:
1. Download **Mistral-7B-Instruct-v0.2** GGUF (Q4_K_M).
2. Set the path and run:
```bash
python run_experiments.py --config config/config.yaml --llm --llama_bin /path/to/llama --gguf /path/to/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

Notes:
- Decoding is configured with `temperature=0.0, top_p=1.0, max_tokens=800`.
- Minor nondeterminism can occur due to floating point behavior; outputs are expected to be structurally consistent.

## Reproducing paper tables
- Latency statistics: `python tools/summarize_latency.py results/.../latency.csv`

## License
MIT (see `LICENSE`).

# AgileOps Agentic Framework (AAF)

[![Reproducibility Artifact](https://img.shields.io/badge/Reproducibility-Available-brightgreen)](https://github.com/Ksuresh/AgileOps-Agentic-Framework)
