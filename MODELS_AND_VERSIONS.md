# Models, tools and versions used in the experiments

All experiments run on GitHub-hosted runners (`ubuntu-latest`, 4 vCPU, 16 GB RAM, no GPU) from workflows in
`.github/workflows/`. Every run commits its raw outputs and logs back to the branch, so the files in this repository are the
exact outputs the paper reports.

## LLMs

| Use | Model | Access | Settings | Where recorded |
|---|---|---|---|---|
| Studies 6-8, PM interface, explanations | `gpt-5.6-luna` | OpenAI Responses API (repository secret `OPENAI_API_KEY`) | reasoning effort `none` (AAF, B5) or `medium` (B6); default sampling | workflow env `OPENAI_MODEL`; `revision_2026/run_llm_arms.py` |
| Study 10, 11b, 9, 12, 13, 14 deciders | `qwen3:8b` (Qwen3 8.2B, Q4_K_M, digest `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`) | Ollama `/api/chat`, JSON output | temperature 0.7, context 8192, max 3072 output tokens, no seed; `think=false` (AAF, B5) / `think=true` (B6) | `env_shard_*.txt` next to each result file |
| Study 10b | `llama3.1:8b` (Llama 3.1 8B Instruct, Q4_K_M; digest in `model_replication/results/llama3.1_8b/main/env_shard_*.txt`) | Ollama | as Qwen; no thinking mode (B6 without thinking) | same |
| Study 11 | none (deterministic fault models) | - | seeded per case/run (CRC32) | `containment_study/run_containment.py` |

Ollama: 0.40.9 for all main runs (0.9.0 in the first Study 10 smoke run only). Prompts are verbatim in
`agentic_experiments/agentic_runtime.py` (`AGENT_SYSTEM`, `COORDINATOR_SYSTEM`) and `revision_2026/run_llm_arms.py`
(`GOVERNANCE_RULES`, `COORD_B6`); Supplementary Appendix M reproduces them.

## Testbed (Studies 9, 12, 13, 14)

| Component | Version |
|---|---|
| kind | v0.24.0 (node image `kindest/node:v1.31.0`), single node |
| Google Online Boutique | commit `b9a978db9e01f4ad3dca9494a22cb9edc17548fe`, `release/kubernetes-manifests.yaml`, load generator removed |
| Stable frontend image | `us-central1-docker.pkg.dev/online-boutique-ci/microservices-demo/frontend:v0.10.6` (0 critical, 32 HIGH/MEDIUM findings) |
| Vulnerable images | `gcr.io/google-samples/microservices-demo/frontend:v0.8.0` (A), `v0.7.0` (B), `v0.6.0` (C), `v0.5.0` (D); Study 14 E, F, P1, P2 in `outcome_study14/calibration/calibration.json` |
| metrics-server | v0.7.2 (`--kubelet-insecure-tls`) |
| trivy | 0.56.2 (container image `aquasec/trivy:0.56.2`; the .deb install fell back to the container) |
| Python | 3.11 (`actions/setup-python`), packages from `requirements.txt` |

## Determinism and randomness

- Deterministic components (rules, gate, arbitration, B1, B2, siloed rule experts, fixed policies) are pure functions of the
  evidence: rerunning them gives identical actions.
- LLM components are sampled (temperature 0.7 for Ollama; API defaults for GPT); each LLM arm is run several times and the
  majority action is reported. Per-run actions are in the result files.
- Case-job order is shuffled with fixed seeds (2026 in Study 10, 9 in Studies 9-14) only to balance shards.
- Live measurements vary between runners; each outcome case is measured in two independent replicates and averaged.
