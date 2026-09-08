# Paper Results Index

This directory is the reviewer-facing index for the experiments reported in the manuscript **Cross-Domain Evidence Integration for Project Governance in Agile–DevOps: An Explainable Multi-Agent Decision-Support Framework**.

It intentionally separates three things:

1. **Experiment definitions / evidence inputs** — kept in `scenario_generator/`, `benchmark/`, `runtime_validation/`, and `experiments/`.
2. **Generated result directories** — produced locally by the commands below and ignored by git to avoid presenting reruns as the frozen historical execution.
3. **Frozen execution evidence** — preserved by the corresponding GitHub Actions artifacts and historical provenance branches/runs documented in `REPRODUCE_CURRENT_PAPER.md`.

The benchmark applications (Sock Shop and Google Online Boutique) are not called datasets. They are open-source benchmark systems on which controlled interventions generate runtime evidence.

## Experiment map

| Manuscript experiment | Evidence / input definition | Command or workflow | Generated results |
|---|---|---|---|
| Controlled oracle benchmark | `scenario_generator/`, `benchmark/oracle_specs.py`, seeds 42–45 | `python evaluation/run_oracle_benchmark.py --out paper_results/controlled/generated` | `paper_results/controlled/generated/case_results.csv`, `summary.json` |
| Cross-domain held-out comparison | frozen held-out templates in experiment code | `python experiments/run_cross_domain_heldout.py` | script-defined held-out results; see reproduction guide |
| Sock Shop runtime studies | `runtime_validation/interventions*.yaml` + measured runtime artifacts | runtime workflows under `.github/workflows/` | workflow artifacts and evaluator result directories |
| Prospective HRT-32–39 | `runtime_validation/interventions_v2_prospective.yaml` | `.github/workflows/aaf-v2-prospective-runtime.yml` | `results_v2_prospective_runtime/runtime_case_results.csv`, `runtime_summary.json` |
| Online Boutique | `runtime_validation/interventions_online_boutique.yaml` | `.github/workflows/runtime-online-boutique.yml` | `results_online_boutique_runtime/OB-*.json`, `summary.json` |
| Learned baselines | generated calibration data + `experiments/learned_baseline_hrt_template_features.csv` | `python experiments/run_learned_baselines.py` or workflow | `results_learned_baselines/learned_baseline_case_results.csv`, `learned_baseline_summary.json` |

## Important reproducibility rule

Do not edit oracle/admissible-action definitions after inspecting model/framework outputs. Runtime intervention manifests are frozen experimental specifications; evidence acquisition is separated from oracle evaluation.

For complete setup instructions, pinned benchmark revisions, and commands, use [`REPRODUCE_CURRENT_PAPER.md`](../REPRODUCE_CURRENT_PAPER.md).
