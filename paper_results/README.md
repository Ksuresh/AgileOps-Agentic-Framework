# Paper Results and Reproducibility Index

This directory is the reviewer-facing index for **Cross-Domain Evidence Integration for Project Governance in Agile–DevOps: An Explainable Multi-Agent Decision-Support Framework**.

AAF is evaluated in four complementary layers: controlled cross-domain evidence, live benchmark evidence, independent recorded fault evidence, and bounded-LLM interface faithfulness. These layers answer different questions and should not be collapsed into one accuracy number.

## Validation map

| Validation layer | What it tests | Evidence source | Execution / provenance |
|---|---|---|---|
| Controlled cross-domain | arbitration, interaction detection, ablations, sensitivity | deterministic generated scenarios and frozen oracles | `evaluation/`, `experiments/`, `benchmark/` |
| Sock Shop runtime | AAF behavior on measured live benchmark evidence | controlled interventions + runtime telemetry | `runtime_validation/`; GitHub Actions runtime workflows |
| Online Boutique runtime | second live benchmark | pinned Online Boutique + controlled interventions | `.github/workflows/runtime-online-boutique.yml` |
| External RCAEval | behavior on independently recorded fault evidence | RCAEval RE2-TrainTicket | frozen external-validation branch/run provenance |
| Bounded LLM faithfulness | whether the optional PM interface preserves AAF authority/evidence | 5 frozen AAF records x 20 PM prompts | `llm_evaluation/`; `llm_faithfulness/README.md` |

## Directory policy

```text
paper_results/
├── README.md                    this reviewer-facing index
├── controlled/                  controlled benchmark outputs when generated
├── runtime/                     canonical runtime packaging target
│   ├── sock_shop/
│   └── online_boutique/
├── external_rcaeval/            external recorded-fault validation package
├── llm_faithfulness/            frozen LLM run provenance and rescoring documentation
└── manuscript/                  generated master tables/figures (packaging target)
```

Generated rerun outputs are kept separate from frozen historical execution evidence. We do not commit a later runtime rerun and silently present it as the immutable manuscript execution. Frozen manifests, test snapshots, code, workflow run IDs, artifact IDs/digests and historical branches provide provenance; documented commands regenerate derived files.

## Study details

### 1. Controlled evidence

Run:

```bash
python evaluation/run_oracle_benchmark.py --out paper_results/controlled/generated
python evaluation/run_revised_comparison.py
python evaluation/run_sensitivity_analysis.py
python experiments/run_cross_domain_heldout.py
python experiments/parameter_sensitivity_2026.py
```

The controlled studies test cross-domain interaction handling, dominant-domain/no-interaction ablations, finalized arbitration and parameter robustness. Learned baselines are comparative baselines, not component ablations.

### 2. Sock Shop runtime

Evidence definitions are in `runtime_validation/interventions*.yaml`. Raw evidence is collected before AAF inference. The prospective HRT-32--HRT-39 study uses `interventions_v2_prospective.yaml`, `run_v2_prospective_case.py`, and `evaluate_v2_prospective_runtime.py`.

Frozen prospective workflow provenance: run `33885779782`, commit `67f187d0ce9efcd93a572fee71c944dd447745b0`, artifact `aaf-v2-prospective-sock-shop`, artifact ID `9942511529`, SHA-256 `103a997b0f45a7662bb7df4f556a4be61aa9fdb0178e19e37efbc4fa15d1bc41`.

### 3. Online Boutique runtime

Pinned upstream commit: `b9a978db9e01f4ad3dca9494a22cb9edc17548fe`.

The unified workflow currently executes OB-01--OB-08. Any manuscript result referring to a larger historical Online Boutique set must use its historical provenance rather than implying that the current eight-case workflow reproduces sixteen cases. This distinction is intentionally explicit until the larger historical path is consolidated.

### 4. Learned baselines

`experiments/run_learned_baselines.py` fits fixed logistic-regression and shallow-tree baselines on 120 generated calibration instances (30 templates x seeds 42--45) and evaluates on the frozen HRT feature snapshot. Frozen reference run: `33851740601`, commit `c1222b81c2ab070263fde44358dac15da179c220`.

### 5. External RCAEval validation

RCAEval RE2-TrainTicket supplies independent recorded fault evidence; its fault labels are withheld from AAF and used only for evaluation/grouping. It is not a governance-action ground-truth dataset. The frozen protocol uses a 12-case pilot and a 78-case held-out set.

Successful held-out run: `34316286523`, head SHA `88e7c6e87f11cb5f5840f770e22d20d6fbd2ab83`, artifact ID `10090236316`, SHA-256 `1d2213cb7cec656cb168fbec1821969c0db91d53db86b85581fa3aeec420b730`.

### 6. Bounded LLM faithfulness

See [`llm_faithfulness/README.md`](llm_faithfulness/README.md). Frozen run `34325587289` used OpenAI `gpt-5.6-luna`, Responses API, reasoning effort `none`, tools disabled, 100 prompts. Raw and system action preservation were both 100%. A deterministic contextual rescoring of the same frozen responses records 25/25 adversarial override attempts resisted and 25/25 unsupported-evidence prompts rejected. No prompts or model outputs were changed during rescoring.

## Reproducibility rule

Do not edit oracle/admissible-action definitions, frozen intervention manifests, prompt sets or evaluation thresholds after inspecting outcomes. Corrections to post-hoc scoring must be versioned separately and must operate on the same frozen raw outputs, as done for the LLM faithfulness study.

For environment setup, exact commands and pinned benchmark revisions, use [`REPRODUCE_CURRENT_PAPER.md`](../REPRODUCE_CURRENT_PAPER.md).
