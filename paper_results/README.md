# Paper Results and Reproducibility Index

This directory is the reviewer-facing evidence/provenance index for the Hybrid AgileOps Agentic Framework (AAF).

AAF is evaluated through complementary studies that answer different questions. Deterministic cross-domain evidence, live benchmark behavior, selective Agentic Evidence Investigation (AEI), prospective AEI replication, independent external telemetry, and the bounded PM-facing LLM interface must **not** be collapsed into one accuracy number.

## Validation map

| Validation layer | What it tests | Evidence source | Execution / provenance |
|---|---|---|---|
| Controlled cross-domain | arbitration, interaction detection, ablations, sensitivity | deterministic generated scenarios and frozen oracles | `evaluation/`, `experiments/`, `benchmark/` |
| Sock Shop runtime | AAF behavior on measured live benchmark evidence | controlled interventions + runtime telemetry | `runtime_validation/`; GitHub Actions runtime workflows |
| Online Boutique runtime | second live benchmark | pinned Online Boutique + controlled interventions | `.github/workflows/runtime-online-boutique.yml` |
| Primary selective-AEI | Deterministic vs Agentic-only vs Hybrid; selective invocation; evidence acquisition/re-grounding; governance overrides | 32 frozen balanced cases | `agentic_experiments/`; run `34428945881` |
| Prospective selective-AEI replication | whether frozen Protocol-v2 behavior reproduces on unseen cases | 16 separately frozen balanced cases | `agentic_experiments/`; run `34432645600` |
| External RCAEval | behavior on independently recorded fault evidence | RCAEval RE2-TrainTicket | frozen external-validation run provenance |
| Bounded PM-facing LLM | whether downstream natural-language interaction preserves AAF authority/evidence | 5 frozen AAF records x 20 PM prompts | `llm_evaluation/`; `llm_faithfulness/README.md` |

**Terminology:** AEI is the LLM-mediated investigation/evidence-acquisition step. Evidence re-grounding/reassessment is the deterministic processing that follows newly acquired evidence. Historical repository/code labels may differ where changing them would disturb frozen provenance.

## Directory policy

```text
paper_results/
├── README.md                    reviewer-facing index
├── MANUSCRIPT_RESULT_MAP.md     manuscript claim -> frozen execution provenance
├── controlled/                  controlled benchmark outputs when generated
├── runtime/                     canonical runtime packaging target
│   ├── sock_shop/
│   └── online_boutique/
├── agentic_experiments_2_4/     generated primary AEI outputs (Actions/local)
├── agentic_prospective_replication/ generated prospective outputs (Actions/local)
├── external_rcaeval/            external recorded-fault validation package
├── llm_faithfulness/            frozen PM-interface run provenance/rescoring docs
└── manuscript/                  generated master tables/figures (packaging target)
```

Generated rerun outputs are kept separate from frozen historical execution evidence. Manuscript-reported LLM raw outputs are preserved in immutable GitHub Actions artifacts and identified by run, commit, artifact ID and digest. A fresh LLM run is a stochastic replication and must not silently replace the manuscript execution.

## Study details

### 1. Controlled and cross-domain evidence

Run:

```bash
python evaluation/run_oracle_benchmark.py --out paper_results/controlled/generated
python evaluation/run_revised_comparison.py
python evaluation/run_sensitivity_analysis.py
python experiments/run_cross_domain_heldout.py
python experiments/parameter_sensitivity_2026.py
```

The controlled studies test cross-domain interaction handling, dominant-domain/no-interaction ablations, finalized arbitration and parameter robustness. The separate frozen 16-case cross-domain held-out comparison records Full AAF `16/16`, dominant-domain `12/16`, and no-interaction `11/16`; on the eight decision-critical cases Full AAF records `8/8` versus `4/8` for each reduced baseline.

### 2. Sock Shop runtime

Evidence definitions are in `runtime_validation/interventions*.yaml`. Raw evidence is collected before AAF inference. The historical HRT-01--31 aggregate is Full AAF `27/31 = 87.1%`, no-interaction `24/31 = 77.4%`, dominant-domain `14/31 = 45.2%`. The separately prospective HRT-32--39 block records finalized AAF `8/8`, initial formulation `5/8`, no-interaction `7/8`, and dominant-domain `3/8`.

Frozen HRT-32--39 provenance: run `33885779782`, commit `67f187d0ce9efcd93a572fee71c944dd447745b0`, artifact ID `9942511529`, SHA-256 `103a997b0f45a7662bb7df4f556a4be61aa9fdb0178e19e37efbc4fa15d1bc41`.

### 3. Online Boutique runtime

Pinned upstream commit: `b9a978db9e01f4ad3dca9494a22cb9edc17548fe`.

The manuscript aggregate combines two independently frozen eight-case blocks and records Full AAF `16/16`, no-interaction `12/16`, and dominant-domain `8/16` (Full vs dominant exact paired `p = 0.0078125`). The current unified workflow executes OB-01--OB-08; historical provenance for OB-09--OB-16 remains separately identified in `MANUSCRIPT_RESULT_MAP.md` and must not be represented as a new unified 16-case rerun.

### 4. Learned baselines

`experiments/run_learned_baselines.py` fits fixed logistic-regression and shallow-tree baselines on 120 generated calibration instances (30 templates x seeds 42--45) and evaluates once on the frozen 13-case HRT feature snapshot. Results: Full AAF `12/13 = 92.3%`, logistic regression `6/13 = 46.2%` (`p = 0.03125`), shallow tree `3/13 = 23.1%` (`p = 0.01171875`).

### 5. Primary selective-AEI study — 32 cases

Frozen Protocol-v2 uses 8 CLEAR, 8 AMBIGUOUS, 8 INCOMPLETE and 8 MISLEADING cases with OpenAI `gpt-5.6-luna`, reasoning effort `none`.

Results:

- Deterministic AAF: `26/32 = 81.25%`
- Agentic-only: `10/32 = 31.25%`
- Hybrid AAF: `27/32 = 84.375%`
- Hybrid AEI invocation: `21/32 = 65.625%`
- CLEAR unnecessary invocation: `0/8`
- Always-on Agentic tokens: `91,441`; Hybrid: `44,879`; avoidance: `50.92%`
- Governance overrides: `14`; beneficial overrides: `11`
- INCOMPLETE: deterministic pre-investigation `7/8` -> Hybrid after AEI/evidence re-grounding `8/8`
- MISLEADING: Agentic-only `2/8`; Hybrid `7/8`

Frozen execution: workflow `AAF Confirmatory Agentic Experiments 2-4`, run `34428945881`, head SHA `3f5913f0a7e8a935f28751a3d074de049ec34951`, artifact ID `10133945801`, digest `sha256:51775beb89de8d09b16f8e6449d900cdd0a42e3863c3f5d9f05fd8059939b583`.

The result is not interpreted as evidence that Agentic AI generally beats deterministic AAF. The governance-action gain is only `26/32 -> 27/32`; the stronger evidence concerns targeted investigation/evidence acquisition, deterministic authority control, and compute avoidance.

### 6. Prospective selective-AEI replication — 16 unseen cases

The original 32 cases were left unchanged after outcome inspection. A separate prospective dataset was frozen with 4 cases per stratum.

Results:

- Deterministic AAF: `13/16 = 81.25%`
- Agentic-only: `7/16 = 43.75%`
- Hybrid AAF: `14/16 = 87.5%`
- Hybrid AEI invocation: `10/16 = 62.5%`
- CLEAR unnecessary invocation: `0/4`
- Always-on Agentic tokens: `47,351`; Hybrid: `18,682`; avoidance: `60.5%`
- Governance overrides: `8`; beneficial overrides: `7`
- INCOMPLETE: deterministic `3/4` -> Hybrid `4/4`; expected tool selected by Hybrid `4/4`

Frozen execution: workflow `AAF Prospective Agentic Replication`, run `34432645600`, head SHA `2417f2f0062dd2a3541e9ce9904e1880fcf15b31`, artifact ID `10135143965`, digest `sha256:a66f5aafb74027d15eabb8047230ed941ab84fa0c38e32233da091e64f2fa343`.

`RP-A04` is retained as a genuine failure/boundary case.

Across the 48 primary + prospective cases, Deterministic AAF is `39/48 = 81.25%`, Agentic-only `17/48 = 35.42%`, and Hybrid AAF `41/48 = 85.42%`. This descriptive Hybrid-versus-deterministic difference is **not statistically significant in either constituent study** (exact paired `p = 1.000` in the primary and prospective studies); it must not be presented as a superiority claim.

### 7. External RCAEval validation

RCAEval RE2-TrainTicket supplies independent recorded fault evidence; its fault/root-cause labels are withheld from AAF and are used only for evaluation/grouping. They are **not** project-governance action labels. The frozen protocol uses 12 adapter-development/pilot cases and 78 held-out cases. AAF consumes mapped telemetry only.

Successful held-out run: `34316286523`, head SHA `88e7c6e87f11cb5f5840f770e22d20d6fbd2ab83`, artifact ID `10090236316`, SHA-256 `1d2213cb7cec656cb168fbec1821969c0db91d53db86b85581fa3aeec420b730`.

The 78 held-out outputs contain 26 `Mitigate and monitor` and 52 `No action` recommendations. This demonstrates external telemetry ingestion/interpretation; it must not be called governance accuracy.

### 8. Bounded PM-facing LLM faithfulness

The PM-facing LLM is downstream of governance and constrained to interpret/verbalize supplied structured AAF records. The frozen 100-prompt GPT-5.6 Luna study used reasoning `none`, tools/web disabled, and 5 frozen AAF records.

Reported boundary metrics are intent mapping `91/100`, authoritative-action preservation `100/100`, authority resistance `25/25`, unsupported-evidence lure rejection `25/25`, and unsupported numeric claims `0/100`.

We do not claim hallucination is impossible; the architectural claim is that unsupported LLM conclusions do not automatically become governance decisions.

## Reproducibility rule

Do not edit oracle/admissible-action definitions, frozen intervention manifests, Agentic datasets, selective trigger, prompt sets, tool contracts or evaluation thresholds after inspecting outcomes. The primary 32-case study and the prospective 16-case replication must remain separately identifiable. Corrections to post-hoc scoring must be separately versioned and operate on the same frozen raw outputs.

For environment setup, exact commands and pinned benchmark revisions, use [`REPRODUCE_CURRENT_PAPER.md`](../REPRODUCE_CURRENT_PAPER.md).