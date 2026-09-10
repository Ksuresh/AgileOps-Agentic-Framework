# Manuscript Result Map — Hybrid AAF

This file maps manuscript-facing AAF results to frozen execution evidence. It is the authoritative reviewer-facing provenance map for deterministic, runtime, selective Agentic Evidence Investigation (AEI), prospective replication, external-telemetry, and bounded PM-facing LLM evidence.

**Architecture boundary:** Hybrid AAF may use bounded AEI to investigate uncertainty and acquire missing evidence. Newly acquired evidence is then re-grounded/reassessed, while deterministic cross-domain governance remains authoritative for the final action. The downstream PM-facing LLM is also non-authoritative.

**Terminology boundary:** AEI denotes the LLM-mediated investigation/evidence-acquisition step. Evidence re-grounding/reassessment denotes processing after acquisition. Historical code/artifact names are preserved where they form part of frozen provenance.

## 1. Controlled cross-domain evidence

The repository retains the controlled generated-scenario evidence and the later mechanism-focused 16-case held-out comparison. The manuscript must distinguish historical controlled results from the later cross-domain held-out headline comparison rather than substitute one for the other.

Frozen cross-domain held-out comparison:

- Workflow: `AAF Cross-Domain Held-Out`
- Run: `33745199681`
- Frozen commit: `f62250a7f2bb4e1716b8fbea8565ce12608a1554`
- Artifact ID: `9889330911`
- Full AAF: `16/16`
- Dominant-domain: `12/16`
- AAF no-interaction: `11/16`
- Decision-critical subgroup: Full AAF `8/8`; dominant-domain `4/8`; no-interaction `4/8`

One-at-a-time parameter robustness from the same workflow records 100% action stability across tested active parameter settings. Historical `lambda` rows are not manuscript-active because embedding-based semantic similarity was removed.

## 2. Sock Shop HRT-01--HRT-21

Frozen block:

- Full AAF: `19/21 = 90.5%`
- No-interaction: `15/21 = 71.4%`
- Dominant-domain: `9/21 = 42.9%`
- Full vs dominant exact paired `p = 0.001953125` (reported `p = 0.0020`)

HRT-21 diagnostic repetitions are not independent templates and remain excluded from `n`.

## 3. Sock Shop HRT-22--HRT-31 extension

- Workflow: `Sock Shop Extended Frozen Validation`
- Run: `33880374847`
- Frozen commit: `6055bb6dda1dbbfd9c667985722770009e1e3c4e`
- Artifact ID: `9940876850`
- HRT-22--31: Full `8/10`, no-interaction `9/10`, dominant-domain `5/10`

Combined HRT-01--31 manuscript aggregate:

- Full AAF: `27/31 = 87.1%`
- No-interaction: `24/31 = 77.4%`
- Dominant-domain: `14/31 = 45.2%`
- Full vs dominant exact paired `p = 0.00024` for the extended aggregate as reported in the final evidence package

## 4. Prospective Sock Shop HRT-32--HRT-39

- Run: `33885779782`
- Frozen commit: `67f187d0ce9efcd93a572fee71c944dd447745b0`
- Artifact ID: `9942511529`
- Artifact SHA-256: `103a997b0f45a7662bb7df4f556a4be61aa9fdb0178e19e37efbc4fa15d1bc41`

Results:

- Finalized arbitration AAF: `8/8`
- Initial arbitration formulation: `5/8`
- No-interaction: `7/8`
- Dominant-domain: `3/8`

This is prospective mechanism validation after arbitration finalization, not a high-powered superiority study.

## 5. Google Online Boutique OB-01--OB-16

Pinned benchmark revision: `b9a978db9e01f4ad3dca9494a22cb9edc17548fe`.

OB-01--08:

- Run: `33873029932`
- Frozen commit: `a6c5718d0dc4ebf6040a2aa49934fb73c488c33a`
- Artifact ID: `9937642578`
- Full `8/8`; no-interaction `6/8`; dominant-domain `4/8`

OB-09--16:

- Run: `33880508662`
- Frozen commit: `fdf222fc6ba0079633693a38cd9762489c414d18`
- Artifact ID: `9940195074`
- Full `8/8`; no-interaction `6/8`; dominant-domain `4/8`

Combined manuscript result:

- Full AAF: `16/16 = 100%`
- No-interaction: `12/16 = 75%`
- Dominant-domain: `8/16 = 50%`
- Full vs dominant exact paired `p = 0.0078125`

The current unified runner executes the first eight-case block; do not imply that it newly reproduces all 16 historical cases in one run.

## 6. Learned comparators

Frozen held-out HRT feature snapshot results:

- Full AAF: `12/13 = 92.3%`
- Logistic regression: `6/13 = 46.2%`, exact paired `p = 0.03125`
- Shallow decision tree: `3/13 = 23.1%`, exact paired `p = 0.01171875`

The learned models are comparative baselines, not AAF component ablations.

## 7. Primary selective-AEI Protocol-v2 study — 32 cases

Dataset: `agentic_experiments/cases_v1.yaml`, balanced 8 CLEAR / 8 AMBIGUOUS / 8 INCOMPLETE / 8 MISLEADING.

Model/configuration: OpenAI `gpt-5.6-luna`, reasoning effort `none`.

**Frozen manuscript execution:**

- Workflow: `AAF Confirmatory Agentic Experiments 2-4`
- Run: `34428945881`
- Head SHA: `3f5913f0a7e8a935f28751a3d074de049ec34951`
- Artifact: `aaf-agentic-experiments-2-4-gpt-5-6-luna`
- Artifact ID: `10133945801`
- Artifact digest: `sha256:51775beb89de8d09b16f8e6449d900cdd0a42e3863c3f5d9f05fd8059939b583`
- Dataset SHA-256 from artifact metadata: `98fd94278193d2c48fffb11ff0b76e96d8233585bf52e52570516f35911913b5`

Frozen results:

- Deterministic: `26/32 = 81.25%`
- Agentic-only: `10/32 = 31.25%`
- Hybrid: `27/32 = 84.375%`
- Hybrid AEI invocation: `21/32 = 65.625%`
- CLEAR unnecessary invocation: `0/8`
- Always-on Agentic tokens: `91,441`
- Hybrid tokens: `44,879`
- Token avoidance: `50.92%`
- Governance overrides: `14`
- Beneficial overrides: `11`
- INCOMPLETE deterministic before investigation `7/8` -> Hybrid after AEI/evidence re-grounding `8/8`; one evidence-acquisition recovery
- MISLEADING Agentic-only `2/8`; Hybrid `7/8`

**Interpretation boundary:** do not claim Agentic AI generally beats deterministic AAF. The action-agreement gain is only `26/32 -> 27/32`. The stronger architectural evidence is selective AEI invocation, evidence acquisition/re-grounding, deterministic authority over Agentic proposals, and token avoidance.

The earlier 32-case run before trigger refinement remains development provenance. Protocol-v2 is the final reported selective design.

## 8. Prospective selective-AEI replication — 16 unseen cases

Dataset: `agentic_experiments/cases_replication_v1.yaml`, separately frozen after the primary 32-case outcomes were observed; 4 CLEAR / 4 AMBIGUOUS / 4 INCOMPLETE / 4 MISLEADING.

Model/configuration: OpenAI `gpt-5.6-luna`, reasoning effort `none`.

**Frozen manuscript execution:**

- Workflow: `AAF Prospective Agentic Replication`
- Successful run: `34432645600` (run #2)
- Head SHA: `2417f2f0062dd2a3541e9ce9904e1880fcf15b31`
- Artifact: `aaf-agentic-prospective-replication-gpt-5-6-luna`
- Artifact ID: `10135143965`
- Artifact digest: `sha256:a66f5aafb74027d15eabb8047230ed941ab84fa0c38e32233da091e64f2fa343`
- Dataset SHA-256 from artifact metadata: `eb41fc780ead10e9ce422fe4c704565fd7ae1ccf84a4a48d27ceb52cd7ab898d`

Frozen results:

- Deterministic: `13/16 = 81.25%`
- Agentic-only: `7/16 = 43.75%`
- Hybrid: `14/16 = 87.5%`
- CLEAR invocation: `0/4`
- AMBIGUOUS invocation: `4/4`
- INCOMPLETE invocation: `4/4`
- MISLEADING invocation: `2/4`
- Overall Hybrid AEI invocation: `10/16 = 62.5%`
- Always-on Agentic tokens: `47,351`
- Hybrid tokens: `18,682`
- Token avoidance: `60.5%`
- Governance overrides: `8`
- Beneficial overrides: `7`
- INCOMPLETE deterministic `3/4` -> Hybrid after AEI/evidence re-grounding `4/4`
- Expected tool selected by Hybrid `4/4`

`RP-A04` remains incorrect under both deterministic and Hybrid AAF and is retained as a genuine boundary/failure case.

## 9. Combined selective-Agentic architectural evidence — 48 cases

Primary 32 + prospective 16:

- Deterministic AAF: `39/48 = 81.25%`
- Agentic-only: `17/48 = 35.42%`
- Hybrid AAF: `41/48 = 85.42%`

This combined total is descriptive. The Hybrid-versus-deterministic difference is **not statistically significant in either constituent study** (exact paired `p = 1.000` in both). The prospective study instead reproduces the central architectural behavior: no unnecessary CLEAR invocation, selective AEI on uncertain/incomplete cases, targeted evidence acquisition, deterministic final authority, and substantial token avoidance versus always-on Agentic reasoning.

The two studies must remain separately identifiable in Methods/Results even when the combined descriptive total is reported.

## 10. RCAEval external telemetry validation

RCAEval RE2-TrainTicket provides independent recorded fault evidence. The protocol uses 12 adapter-development/freeze cases and 78 held-out cases. AAF consumes mapped telemetry only.

- Frozen held-out run: `34316286523`
- Head SHA: `88e7c6e87f11cb5f5840f770e22d20d6fbd2ab83`
- Artifact ID: `10090236316`
- Artifact SHA-256: `1d2213cb7cec656cb168fbec1821969c0db91d53db86b85581fa3aeec420b730`
- Held-out output distribution: `Mitigate and monitor = 26`; `No action = 52`

**Interpretation boundary:** RCAEval does not provide project-governance action ground truth. Do not call these results governance accuracy. They demonstrate ingestion and interpretation of independently recorded external telemetry.

## 11. Bounded PM-facing LLM interface

Frozen study:

- 100 prompts over 5 frozen AAF records
- OpenAI `gpt-5.6-luna`
- reasoning `none`
- tools/web disabled
- Frozen run: `34325587289`
- Head SHA: `dce8bcc8560c0390af4524c8dd0ea286f74d9a5a`
- Artifact ID: `10093805818`
- Artifact SHA-256: `f0ccfd670129b1598b67b6e455a997453ac0648efd7b4204e9265a60f25f76d7`

Reported boundary metrics:

- Intent mapping: `91/100`
- Authoritative-action preservation: `100/100`
- Authority resistance: `25/25`
- Unsupported-evidence lure rejection: `25/25`
- Unsupported numeric claims: `0/100`

The LLM is downstream of governance and constrained to interpret/verbalize structured AAF outputs. Do not claim hallucinations are eliminated. The correct architectural claim is that unsupported LLM conclusions do not automatically become governance decisions.

## Figure and result-file policy

Raw runtime and LLM execution evidence can be large and is retained in the cited GitHub Actions artifacts. Manuscript-facing tables/figures must use verified frozen outputs or deterministic reruns whose provenance is recorded here. Do not reconstruct missing case data from manuscript percentages.

Development iterations, failed workflows and stale assertions remain in Git history unless they materially affect interpretation. The scientifically relevant frozen protocols and successful runs above are the manuscript evidence.