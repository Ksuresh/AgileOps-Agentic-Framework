# Reproducing the revision studies (Studies 9-14, 10b, 11, 11b)

Studies 1-8 are reproduced as described in [`REPRODUCE_CURRENT_PAPER.md`](REPRODUCE_CURRENT_PAPER.md). This file covers the
studies added in the 2026 revision. Versions, model digests and sampling settings are in
[`MODELS_AND_VERSIONS.md`](MODELS_AND_VERSIONS.md). Every study has a `PROTOCOL.md` committed before its first run; the git
history shows the order (protocol commit, then result commits by `github-actions[bot]`).

## 0. Two ways to reproduce

**A. Re-analyse the committed raw outputs (minutes, no cluster, no LLM).** Every live measurement, decision and LLM output is
committed. The analysis scripts recompute every number in the paper from them:

```bash
git clone -b text-evidence-pilot https://github.com/Ksuresh/AgileOps-Agentic-Framework.git && cd AgileOps-Agentic-Framework
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
export PYTHONPATH=$PWD
python containment_study/run_containment.py            # Study 11 (deterministic; reruns the fault models)
python model_replication/analyze.py                    # Study 10   -> model_replication/analysis.json
python model_replication/analyze_three.py              # Study 10b  -> model_replication/analysis_three_models.json
python containment_study/analyze_injection.py          # Study 11b  -> containment_study/results/injection_analysis.json
python outcome_study/analyze.py                        # Study 9    -> outcome_study/analysis.json
python outcome_study12/analyze12.py                    # Study 12   -> outcome_study12/analysis.json
python outcome_study13/analyze13.py                    # Study 13   -> outcome_study13/analysis.json
python outcome_study14/analyze14.py                    # Study 14   -> outcome_study14/analysis.json (incl. pooled H14.7)
python outcome_study/sensitivity.py                    # oracle-threshold sensitivity -> outcome_study/sensitivity.json
```

The deterministic deciders (B1, B1-12, B1-14, B2, siloed rule experts, fixed policies) can also be recomputed from the
committed snapshots without any LLM: `OUTCOME_DIR=outcome_study14 python -c "..."` is not needed; `decide.py` writes them in
shard 0 and they are pure functions of `variants.json`.

**B. Re-run the experiments (hours; GitHub Actions).** Fork the repository, enable Actions, and trigger the workflows below
(`workflow_dispatch`, or edit the control file named in each section and push). Results are committed back to the branch by
the workflow. No API key is needed for any Qwen/Llama study; Studies 6-8 need `OPENAI_API_KEY` as a repository secret.

## 1. Study 9 - outcome-based oracle on Online Boutique

| Step | Control file / workflow | Output |
|---|---|---|
| Calibration (healthy states, image selection) | `outcome_study/RUN`: `MODE=calibrate` -> `outcome-study.yml` | `outcome_study/calibration/calibration.json` |
| Outcomes, replicate 1 and 2 | `MODE=outcomes`, `NSHARDS=16`, `REP=1` (then `REP=2`) | `outcome_study/measurements/rep{1,2}/<case>.json` |
| Deciders (replicate-1 snapshots) | `outcome_study/DECIDE` -> `outcome-decisions.yml` | `outcome_study/decisions/*.jsonl`, `variants.json` |
| Analysis | `python outcome_study/analyze.py` | `outcome_study/analysis.json` |

What a run does (`.github/workflows/outcome-study.yml`): creates a kind cluster with port 30080 mapped to 8080, deploys
Online Boutique at the pinned commit, removes the load generator, installs metrics-server, exposes the frontend as a
NodePort, installs trivy, then runs `outcome_study/run_outcomes.py`. For each case it applies the previous release, the
current release, measures the incident snapshot, and for each action (`none`, `rollback`, `patch`, `mitigate`, `scale`)
rebuilds the incident, applies the action and measures the outcome (`outcome_study/testbed.py`).

To run one case locally (Linux, Docker, kind, kubectl, trivy installed): create the cluster and deploy as in the workflow's
"Cluster" step, then `CASES="OX-01" REP=local python outcome_study/run_outcomes.py`.

## 2. Study 12 - revised rules v1.2, fresh cases

Same as Study 9 with `outcome_study12/RUN`, `outcome_study12/DECIDE`, workflows `outcome-study12.yml`,
`outcome-decisions12.yml` (env `OUTCOME_DIR=outcome_study12`, `LLM_ARMS="AAF AAF-12 B5 B6"`). Rules:
`agentic_experiments/rules_v12.py`. Analysis: `python outcome_study12/analyze12.py`.

## 3. Study 13 - security priority and risk-based gating

`outcome_study13/RUN` (`MODE=calibrate`, then `MODE=outcomes`), workflow `outcome-study13.yml` (adds
`MEASURE_NONCRITICAL=1`; calibration `outcome_study/calibrate13.py`), deciders `outcome-decisions13.yml`.
Analysis: `python outcome_study13/analyze13.py`.

## 4. Study 14 - rules v1.4 on two services

`outcome_study14/RUN`, workflow `outcome-study14.yml` (adds `MEASURE_PRE_RELEASE=1`; calibration
`outcome_study/calibrate14.py`; cases carry `service: frontend | productcatalogservice`), deciders
`outcome-decisions14.yml` (`LLM_ARMS="AAF AAF-14 B5 B6"`). Rules: `agentic_experiments/rules_v14.py`.
Analysis: `python outcome_study14/analyze14.py`.

## 5. Study 10 / 10b - second and third model

Control file `model_replication/RUN` -> `model-replication-qwen.yml` (installs Ollama, pulls the model, runs
`model_replication/run_qwen.py` in shards). Study 10: `MODEL=qwen3:8b`, `RUN_LABEL=main`. Study 10b:
`MODEL=llama3.1:8b`, `NO_THINK=1`. Outputs: `model_replication/results/<model>/main/shard_*.jsonl` and `env_shard_*.txt`
(model digest, Ollama version, runner CPU/RAM). To run locally with Ollama installed:
`QWEN_MODEL=qwen3:8b NSHARDS=1 SHARD=0 RUN_LABEL=local python model_replication/run_qwen.py` (resumable).

## 6. Study 11 / 11b - LLM-failure containment

Study 11 needs no model: `python containment_study/run_containment.py`. Study 11b (prompt injection): `model_replication/RUN`
with `INJECT=1`, `RUNS_PER_ARM=1`, `RUN_LABEL=injection`.

## 7. Integrity checks

- `git log --format='%h %an %ad %s' -- outcome_study14/PROTOCOL.md outcome_study14/measurements` shows the protocol commit
  precedes every result commit.
- Each result file is written by `github-actions[bot]`; `ci_logs/` holds the run logs; `env_shard_*.txt` the model digests.
- Execution notes (bugs found and fixed during a study, with what was rerun) are appended to each `PROTOCOL.md`.
