# Reproducing the IEEE Access Experiments

This is the execution entry point for the current AAF manuscript. For the compact experiment/provenance index, see [`paper_results/README.md`](paper_results/README.md), and for manuscript-number provenance see [`paper_results/MANUSCRIPT_RESULT_MAP.md`](paper_results/MANUSCRIPT_RESULT_MAP.md).

## 1. Environment and deterministic tests

```bash
git clone https://github.com/Ksuresh/AgileOps-Agentic-Framework.git
cd AgileOps-Agentic-Framework
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH="$PWD"         # PowerShell: $env:PYTHONPATH=(Get-Location)
python -m pytest -q
```

Reference CI version: Python 3.11. The reviewer-packaging workflow repeats dependency installation, deterministic tests, manuscript-asset generation and package validation in a clean GitHub Actions environment.

## 2. Controlled cross-domain experiments

```bash
python evaluation/run_oracle_benchmark.py --out paper_results/controlled/generated
python evaluation/run_revised_comparison.py
python evaluation/run_sensitivity_analysis.py
python experiments/run_cross_domain_heldout.py
python experiments/parameter_sensitivity_2026.py
```

The first command writes `case_results.csv` and `summary.json` under `paper_results/controlled/generated/`. Fixed seeds/configuration make controlled generation deterministic. Oracle/admissible-action definitions must not be changed after inspecting outputs.

## 3. Sock Shop runtime validation

Requirements: Docker Engine, Docker Compose v2 and Git. Protocols/manifests/adapters are under `runtime_validation/`.

Example:

```bash
python runtime_validation/run_runtime_case.py HRT-01 --repetition 1 --compose-file "$SOCK_COMPOSE" --settle-seconds 10 --execute
python runtime_validation/evaluate_heldout_runtime.py --cases HRT-01 --repetitions 1 --out results_heldout_runtime
```

For the finalized prospective HRT-32--HRT-39 study use `runtime_validation/interventions_v2_prospective.yaml`, `runtime_validation/run_v2_prospective_case.py`, `runtime_validation/evaluate_v2_prospective_runtime.py`, or `.github/workflows/aaf-v2-prospective-runtime.yml`.

Pinned Sock Shop revision for that study: `9dff06fae4981921caec6a62393a6ebfce4b3e3f`.

Frozen prospective provenance: run `33885779782`, commit `67f187d0ce9efcd93a572fee71c944dd447745b0`, artifact ID `9942511529`, SHA-256 `103a997b0f45a7662bb7df4f556a4be61aa9fdb0178e19e37efbc4fa15d1bc41`.

## 4. Google Online Boutique

Requirements: Docker, kubectl, kind and Git. Pinned upstream revision: `b9a978db9e01f4ad3dca9494a22cb9edc17548fe`.

The current unified workflow `.github/workflows/runtime-online-boutique.yml` deploys the pinned benchmark and executes the first frozen block, OB-01--OB-08.

```bash
for i in 01 02 03 04 05 06 07 08; do
  python runtime_validation/run_online_boutique_case.py "OB-$i"
done
python runtime_validation/evaluate_online_boutique.py
```

Generated files are under `results_online_boutique_runtime/`.

The manuscript's OB-01--OB-16 aggregate combines two independently frozen eight-case blocks (OB-01--OB-08 and OB-09--OB-16). The current unified runner executes one eight-case block; exact provenance for both historical blocks is preserved in `paper_results/MANUSCRIPT_RESULT_MAP.md`. Do not interpret the command above as a new single-run reproduction of all 16 historical cases.

## 5. Learned baselines

```bash
python experiments/run_learned_baselines.py
```

This generates 120 calibration instances from 30 controlled templates under seeds 42--45, fits fixed logistic-regression and shallow-tree baselines, and evaluates once on the frozen HRT feature snapshot `experiments/learned_baseline_hrt_template_features.csv`. HRT data is not used for fitting or hyperparameter selection.

Reference workflow: `.github/workflows/learned-baselines.yml`. Frozen reference run: `33851740601`, commit `c1222b81c2ab070263fde44358dac15da179c220`.

## 6. Independent external-fault validation: RCAEval

The external study uses RCAEval RE2-TrainTicket as an independent recorded-fault evidence source. Fault/root-cause labels are withheld from AAF and are used only for evaluation/grouping; they are not governance-action labels.

The implementation and frozen protocol are committed under `external_validation/rcaeval/`. The split is 12 adapter-development/pilot cases followed by 78 held-out cases. To execute the held-out study:

```bash
python external_validation/rcaeval/run_heldout.py
```

Generated outputs are written under `paper_results/external_rcaeval/generated/heldout/`. The successful frozen held-out execution used run `34316286523`, head SHA `88e7c6e87f11cb5f5840f770e22d20d6fbd2ab83`, artifact ID `10090236316`, artifact SHA-256 `1d2213cb7cec656cb168fbec1821969c0db91d53db86b85581fa3aeec420b730`.

The historical experiment branch is retained for provenance, but the reviewer-ready repository now contains the RCAEval adapter, frozen protocol and held-out runner needed to inspect and rerun the study.

## 7. Bounded LLM faithfulness evaluation

The optional LLM is outside the authoritative AAF decision path. The frozen experiment uses OpenAI `gpt-5.6-luna`, Responses API, reasoning effort `none`, tools disabled, 5 frozen AAF records and 20 prompts per record.

To execute a new run, set `OPENAI_API_KEY` in the environment (or repository Actions secret) and run:

```bash
python llm_evaluation/run_faithfulness_eval.py
```

Reference workflow: `.github/workflows/llm-faithfulness.yml`.

Frozen execution: run `34325587289`, head SHA `dce8bcc8560c0390af4524c8dd0ea286f74d9a5a`, artifact `llm-faithfulness-gpt-5-6-luna`, artifact ID `10093805818`, SHA-256 `f0ccfd670129b1598b67b6e455a997453ac0648efd7b4204e9265a60f25f76d7`.

The original artifact contains `faithfulness_outputs.jsonl`. To reproduce the corrected contextual scoring without making any new model calls:

```bash
python llm_evaluation/rescore_frozen_faithfulness.py <extracted-frozen-artifact-dir> paper_results/llm_faithfulness/generated_rescore
```

See [`paper_results/llm_faithfulness/README.md`](paper_results/llm_faithfulness/README.md) for metrics and interpretation boundaries.

## 8. Generate reviewer/manuscript assets

The canonical manuscript aggregate is committed at `paper_results/manuscript/aggregate_results.csv`. Generate the reviewer-facing table, provenance CSV and validation overview directly from that input:

```bash
python paper_results/manuscript/generate_manuscript_assets.py
```

Generated files are written under `paper_results/manuscript/generated/`. The same generation and validation path is exercised by `.github/workflows/reviewer-ready-packaging.yml`.

## Evidence/result map

| Study | Evidence/specification | Result/provenance location |
|---|---|---|
| Controlled | `scenario_generator/`, `config/`, `benchmark/` | `paper_results/controlled/generated/` on rerun |
| Sock Shop | `runtime_validation/interventions*.yaml` + measured artifacts | evaluator directories + Actions artifacts |
| Prospective HRT-32--39 | `runtime_validation/interventions_v2_prospective.yaml` | frozen run/artifact identifiers above |
| Online Boutique | pinned benchmark + `runtime_validation/interventions_online_boutique.yaml` | current 8-case outputs + historical two-block provenance map |
| Learned baselines | generated calibration + frozen HRT features | `results_learned_baselines/` + Actions artifact |
| RCAEval | `external_validation/rcaeval/` + independent RE2-TrainTicket evidence | `paper_results/external_rcaeval/generated/heldout/` + frozen artifact |
| LLM faithfulness | frozen AAF decision records + 100 PM prompts | frozen run/artifact + deterministic rescoring script |
| Manuscript package | `paper_results/manuscript/aggregate_results.csv` | code-generated assets under `paper_results/manuscript/generated/` |

## Interpretation boundaries

Sock Shop and Online Boutique are benchmark applications, not customer-production datasets. Replica expansion is a resource-footprint proxy when monetary cloud-billing evidence is unavailable. RCAEval fault labels are not governance-action ground truth. The LLM experiment tests a bounded interface under a fixed model/configuration; it does not prove that hallucination is impossible.

## Provenance policy

Frozen manifests, held-out snapshots and prompt sets are committed. Raw runtime/model execution evidence is preserved by GitHub Actions artifacts and historical branches. Later reruns are not silently substituted for the execution used during manuscript development. Any post-hoc scoring correction must be separately versioned and operate on the same frozen raw outputs.

The previous pre-paper main is preserved at `archive/pre-paper-2026-main`.
