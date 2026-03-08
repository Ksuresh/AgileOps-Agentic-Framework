# AgileOps Agentic Framework (AAF)

[![Reproducibility Artifact](https://img.shields.io/badge/Reproducibility-Available-brightgreen)](https://github.com/Ksuresh/AgileOps-Agentic-Framework)

This repository contains the reproducibility artifact and proof-of-concept (POC) implementation for the manuscript:

**AgileOps Agentic Framework (AAF): A Multi-Agent and LLM-Driven Governance System for Operational Visibility in Agile-DevOps Ecosystems**  
Manuscript ID: **Access-2025-54890**

---

## Overview

AAF is a multi-agent governance framework designed to interpret operational telemetry across four domains:

- **DevOps**
- **SRE**
- **FinOps**
- **DevSecOps**

The framework supports:

- deterministic synthetic scenario generation
- domain-specific agent inference
- cross-agent consensus scoring
- Re-Grounded Agentic Reasoning (RAR)
- telemetry-aware utility-based action selection
- optional explanation generation
- explainability scoring

This repository is intended to make the paper’s synthetic evaluation reproducible without requiring a live Kubernetes or cloud environment.

---

## What this artifact provides

- `config/config.yaml`  
  Fixed experiment configuration including thresholds, utility weights, deterministic embeddings, and optional LLM settings.

- `scenario_generator/`  
  Generator for 30 synthetic cross-domain governance scenarios with controllable noise.

- `agents/`  
  Deterministic domain agents for:
  - DevOps
  - SRE
  - FinOps
  - DevSecOps

- `orchestrator/`  
  Logic for:
  - consensus scoring
  - RAR
  - utility-based action selection

- `metrics/`  
  Explainability index (XI) calculation.

- `llm/`  
  Deterministic explainer and optional llama.cpp integration.

- `reproducibility/`  
  Scripts to regenerate paper-style summary outputs.

- `prompts/`  
  Prompt templates aligned with the manuscript appendix.

---

## Current POC scope

This Phase 1 artifact focuses on **30 synthetic scenarios** for reproducibility.

It does **not** yet connect to real-time telemetry sources such as:

- GitHub Actions / Jenkins
- Prometheus / Grafana
- cloud billing systems
- Trivy / OPA / Falco
- Kubernetes runtime telemetry

Those real-time integrations are planned for **Phase 2**.

---

## Repository structure

```text
AgileOps-Agentic-Framework/
├── aaf/
├── agents/
├── baselines/
├── config/
├── experiments/
├── llm/
├── metrics/
├── orchestrator/
├── prompts/
├── reproducibility/
├── scenario_generator/
├── tools/
├── pipeline.py
├── run_experiments.py
├── requirements.txt
└── README.md
