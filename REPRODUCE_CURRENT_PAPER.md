# Reproducing the Hybrid AAF IEEE Access Experiments

This is the execution entry point for the current Hybrid AgileOps Agentic Framework (AAF) manuscript. For the reviewer-facing study index, see [`paper_results/README.md`](paper_results/README.md); for exact manuscript-number provenance, see [`paper_results/MANUSCRIPT_RESULT_MAP.md`](paper_results/MANUSCRIPT_RESULT_MAP.md).

The repository contains deterministic experiments that can be rerun exactly from frozen inputs and LLM-based experiments for which a fresh API execution is a **stochastic replication**. Manuscript-reported LLM outputs are therefore tied to frozen GitHub Actions runs/artifacts.

**Terminology:** the manuscript uses **Agentic Evidence Investigation (AEI)** for LLM-mediated investigation/evidence acquisition. Evidence re-grounding/reassessment is the processing that follows acquired evidence. Historical code, workflow, and artifact names are retained where they are part of frozen provenance.

## 1. Environment and deterministic tests

```bash
git clone https://github.com/Ksuresh/AgileOps-Agentic-Framework.git
cd AgileOps-Agentic-Framework
git checkout experiment/hybrid-agentic-aaf-v2
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH="$PWD"         # PowerShell: $env:PYTHONPATH=(Get-Location)
python -m pytest -q
```

Reference CI version: Python 3.11.

## 2. Controlled cross-domain experiments

```bash
python evaluation/run_oracle_benchmark.py --out paper_results/controlled/generated
python evaluation/run_revised_comparison.py
python evaluation/run_sensitivity_analysis.py
python experiments/run_cross_domain_heldout.py
python experiments/parameter_sensitivity_2026.py
```

Fixed seeds/configuration make controlled generation deterministic. Oracle/admissible-action definitions must not be changed after inspecting outputs.

## 3. Sock Shop runtime validation

Requirements: Docker Engine, Docker Compose v2 and Git. Protocols/manifests/adapters are under `runtime_validation/`.

Example:

```bash
python runtime_validation/run_runtime_case.py HRT-01 --repetition 1 --compose-file "$SOCK_COMPOSE" --settle-seconds 10 --execute
python runtime_validation/evaluate_heldout_runtime.py --cases HRT-01 --repetitions 1 --out results_heldout_runtime
```

For the finalized prospective HRT-32--HRT-39 study use `runtime_validation/interventions_v2_prospective.yaml`, `runtime_validation/run_v2_prospective_case.py`, `runtime_validation/evaluate_v2_prospective_runtime.py`, or `.github/workflows/aaf-v2-prospective-runtime.yml`.

Pinned Sock Shop revision: `9dff06fae4981921caec6a62393a6ebfce4b3e3f`.

Frozen HRT-32--39 provenance: run `33885779782`, commit `67f187d0ce9efcd93a572fee71c944dd447745b0`, artifact ID `9942511529`, SHA-256 `103a997b0f45a7662bb7df4f556a4be61aa9fdb0178e19e37efbc4fa15d1bc41`.

## 4. Google Online Boutique

Requirements: Docker, kubectl, kind and Git. Pinned upstream revision: `b9a978db9e01f4ad3dca9494a22cb9edc17548fe`.

The current unified workflow `.github/workflows/runtime-online-boutique.yml` executes OB-01--OB-08:

```bash
for i in 01 02 03 04 05 06 07 08; do
  python runtime_validation/run_online_boutique_case.py "OB-$i"
done
python runtime_validation/evaluate_online_boutique.py
```

The manuscript's OB-01--OB-16 aggregate combines two independently frozen eight-case blocks. Exact historical provenance for both blocks is in `paper_results/MANUSCRIPT_RESULT_MAP.md`. Do not interpret the command above as a new unified reproduction of all 16 historical cases.

## 5. Learned baselines

```bash
python experiments/run_learned_baselines.py
```

This generates 120 calibration instances from 30 controlled templates under seeds 42--45, fits fixed logistic-regression and shallow-tree baselines, and evaluates once on `experiments/learned_baseline_hrt_template_features.csv`. HRT data is not used for fitting or hyperparameter selection.

## 6. Primary selective-AEI study — Experiments 2–4

**Frozen design rule:** **Uncertainty invokes agency; severity does not.**

The primary dataset is `agentic_experiments/cases_v1.yaml`: 32 independent cases, balanced across 8 CLEAR, 8 AMBIGUOUS, 8 INCOMPLETE and 8 MISLEADING cases. The three configurations are Deterministic AAF, Agentic-only, and Hybrid AAF. In Hybrid AAF, AEI is selective and deterministic governance remains authoritative after evidence acquisition and re-grounding/reassessment.

Inspect the frozen protocol before running:

```bash
cat agentic_experiments/FROZEN_PROTOCOL.md
```

To execute a **fresh stochastic replication**:

```bash
export OPENAI_API_KEY='...'
export OPENAI_MODEL='gpt-5.6-luna'
export OPENAI_REASONING_EFFORT='none'
python -m agentic_experiments.run_experiments_2_4
```

Generated outputs:

```text
paper_results/agentic_experiments_2_4/
  metadata.json
  raw_outputs.json
  case_results.csv
  summary.json
```

### Manuscript-reported frozen execution

- Workflow: `AAF Confirmatory Agentic Experiments 2-4`
- Run: `34428945881`
- Head SHA: `3f5913f0a7e8a935f28751a3d074de049ec34951`
- Artifact: `aaf-agentic-experiments-2-4-gpt-5-6-luna`
- Artifact ID: `10133945801`
- Artifact digest: `sha256:51775beb89de8d09b16f8e6449d900cdd0a42e3863c3f5d9f05fd8059939b583`
- Dataset SHA-256 recorded in artifact metadata: `98fd94278193d2c48fffb11ff0b76e96d8233585bf52e52570516f35911913b5`

Reported results: Deterministic `26/32`, Agentic-only `10/32`, Hybrid `27/32`; Hybrid AEI invocation `21/32`; CLEAR unnecessary invocation `0/8`; always-on Agentic `91,441` tokens vs Hybrid `44,879` tokens (`50.92%` avoidance); 14 governance overrides, 11 beneficial.

## 7. Prospective selective-AEI replication

The original 32 cases were not modified after observing their results. The separate dataset `agentic_experiments/cases_replication_v1.yaml` contains 16 unseen cases: 4 CLEAR, 4 AMBIGUOUS, 4 INCOMPLETE and 4 MISLEADING.

To execute a **fresh stochastic replication**:

```bash
export OPENAI_API_KEY='...'
export OPENAI_MODEL='gpt-5.6-luna'
export OPENAI_REASONING_EFFORT='none'
python -m agentic_experiments.run_prospective_replication
```

Generated outputs:

```text
paper_results/agentic_prospective_replication/
  metadata.json
  raw_outputs.json
  case_results.csv
  summary.json
```

### Manuscript-reported frozen execution

- Workflow: `AAF Prospective Agentic Replication`
- Successful run: `34432645600` (run #2)
- Head SHA: `2417f2f0062dd2a3541e9ce9904e1880fcf15b31`
- Artifact: `aaf-agentic-prospective-replication-gpt-5-6-luna`
- Artifact ID: `10135143965`
- Artifact digest: `sha256:a66f5aafb74027d15eabb8047230ed941ab84fa0c38e32233da091e64f2fa343`
- Dataset SHA-256 recorded in artifact metadata: `eb41fc780ead10e9ce422fe4c704565fd7ae1ccf84a4a48d27ceb52cd7ab898d`

Reported results: Deterministic `13/16`, Agentic-only `7/16`, Hybrid `14/16`; Hybrid AEI invocation `10/16`; CLEAR unnecessary invocation `0/4`; always-on Agentic `47,351` tokens vs Hybrid `18,682` (`60.5%` avoidance); 8 governance overrides, 7 beneficial; INCOMPLETE `3/4 -> 4/4` with expected-tool selection `4/4` in Hybrid. `RP-A04` remains a preserved failure/boundary case.

The combined 48-case Hybrid-versus-deterministic total is descriptive. The difference is not statistically significant in either constituent study (exact paired `p = 1.000` in both).

## 8. Independent external-fault validation: RCAEval

RCAEval RE2-TrainTicket is an independent recorded-fault evidence source. Fault/root-cause labels are withheld from AAF and are used only for evaluation/grouping; they are **not governance-action labels**.

The frozen protocol uses 12 adapter-development/pilot cases followed by 78 held-out cases:

```bash
python external_validation/rcaeval/run_heldout.py
```

Generated outputs are written under `paper_results/external_rcaeval/generated/heldout/`.

Frozen held-out execution: run `34316286523`, head SHA `88e7c6e87f11cb5f5840f770e22d20d6fbd2ab83`, artifact ID `10090236316`, artifact SHA-256 `1d2213cb7cec656cb168fbec1821969c0db91d53db86b85581fa3aeec420b730`.

The held-out output distribution is 26 `Mitigate and monitor` and 52 `No action`. This demonstrates external telemetry ingestion/interpretation and must not be reported as governance accuracy.

## 9. Bounded PM-facing LLM faithfulness evaluation

The PM-facing LLM is downstream of the authoritative AAF decision path. The frozen experiment uses OpenAI `gpt-5.6-luna`, Responses API, reasoning effort `none`, tools/web disabled, 5 frozen AAF records and 20 prompts per record.

To execute a fresh run:

```bash
python llm_evaluation/run_faithfulness_eval.py
```

Frozen execution: run `34325587289`, head SHA `dce8bcc8560c0390af4524c8dd0ea286f74d9a5a`, artifact `llm-faithfulness-gpt-5-6-luna`, artifact ID `10093805818`, SHA-256 `f0ccfd670129b1598b67b6e455a997453ac0648efd7b4204e9265a60f25f76d7`.

To reproduce contextual rescoring without new model calls:

```bash
python llm_evaluation/rescore_frozen_faithfulness.py <extracted-frozen-artifact-dir> paper_results/llm_faithfulness/generated_rescore
```

Reported boundary metrics are intent mapping `91/100`, authoritative-action preservation `100/100`, authority resistance `25/25`, unsupported-evidence lure rejection `25/25`, and unsupported numeric claims `0/100`.

## 10. Generate reviewer/manuscript assets

The deterministic manuscript aggregate is committed at `paper_results/manuscript/aggregate_results.csv`:

```bash
python paper_results/manuscript/generate_manuscript_assets.py
```

Generated files are written under `paper_results/manuscript/generated/`. The final manuscript must additionally incorporate the frozen Agentic results above from their identified artifacts/provenance; a fresh LLM rerun must not silently replace those outputs.

## Evidence/result map

| Study | Evidence/specification | Frozen result/provenance |
|---|---|---|
| Controlled | `scenario_generator/`, `config/`, `benchmark/` | deterministic rerun + provenance map |
| Sock Shop | `runtime_validation/interventions*.yaml` | evaluator directories + Actions artifacts |
| Prospective HRT-32--39 | `runtime_validation/interventions_v2_prospective.yaml` | run `33885779782` |
| Online Boutique | pinned benchmark + intervention manifest | two historical frozen blocks in provenance map |
| Learned baselines | generated calibration + frozen HRT features | Actions provenance in result map |
| Primary selective-AEI 32 | `agentic_experiments/cases_v1.yaml`, `FROZEN_PROTOCOL.md` | run `34428945881`, artifact `10133945801` |
| Prospective selective-AEI 16 | `agentic_experiments/cases_replication_v1.yaml` | run `34432645600`, artifact `10135143965` |
| RCAEval | `external_validation/rcaeval/` | run `34316286523`, artifact `10090236316` |
| PM-facing LLM | frozen AAF records + 100 prompts | run `34325587289`, artifact `10093805818` |

## Interpretation boundaries

- Sock Shop and Online Boutique are benchmark applications, not customer-production datasets.
- Replica expansion is a resource-footprint proxy where monetary cloud-billing evidence is unavailable.
- RCAEval fault labels are not project-governance action ground truth.
- The 32-case primary selective-AEI study and 16-case prospective replication are separate frozen studies.
- The Hybrid-versus-deterministic difference is not statistically significant in either selective-Agentic study; the contribution is selective evidence investigation/acquisition, authority preservation, and reduced LLM usage.
- The PM-facing LLM evaluation does not prove hallucination is impossible. It tests a bounded downstream interface whose outputs cannot autonomously become governance decisions.

## Provenance policy

Frozen manifests, held-out snapshots, Agentic datasets and prompt sets are not edited after outcome inspection. Raw runtime/model execution evidence is preserved by GitHub Actions artifacts and historical commits. Later reruns are not silently substituted for manuscript executions. Development mistakes and failed workflow attempts remain in Git history but are not treated as experimental evidence unless they materially affect scientific interpretation.