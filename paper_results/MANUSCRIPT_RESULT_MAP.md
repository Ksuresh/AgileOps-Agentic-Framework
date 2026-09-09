# Manuscript Result Map

This file maps the manuscript-facing AAF results to the frozen execution evidence that supports them. It is intended for reviewers and future reproductions. GitHub Actions artifacts preserve the raw execution outputs; the repository keeps the experiment definitions, evaluators, and this provenance map.

## Clean-current-main validation

Current manuscript/reproducibility branch: `main`.

- Current audited commit: `d025254ba0717fc6d5bc2f86f05ce91622dcb8d4`.
- Reproducibility smoke / deterministic validation: Actions run `34238573860`, successful.
- Learned baseline reproduction: Actions run `34238573913`, successful.
- The dependency correction that made clean installation pass added the OpenAI Python SDK required by the bounded PM natural-language interface. The LLM remains outside the deterministic governance decision path.

## 1. Controlled frozen-oracle experiment

Frozen execution:

- Workflow: `Revision Experiments`
- Run: `33706624802`
- Frozen commit: `c3f81ecdd748bc4eaa8342867dcb0649a3ae1264`
- Artifact: `revision-controlled-experiments`
- Artifact ID: `9875432477`
- Artifact SHA-256: `17a7e67a368259f93e5a0f5cb7b42ce90d2f342fd0514624bc19bc360d93cd6e`

The artifact contains 120 controlled instances (30 base conditions x seeds 42--45). Under the pre-specified experimental oracle, the full controlled configuration records domain-oracle agreement `79/120 = 65.8%` and action-oracle agreement `84/120 = 70.0%`, with action Wilson 95% CI `[61.3%, 77.5%]`. These controlled-oracle values are not production accuracy claims.

The historical dominant-domain paired comparison in this artifact is also `84/120 = 70.0%`, with six utility-only wins and six baseline-only wins (`p = 1.0`). This experiment predates the later cross-domain held-out design below and should not be substituted for the held-out cross-domain headline comparison.

## 2. Frozen cross-domain held-out comparison

Frozen execution:

- Workflow: `AAF Cross-Domain Held-Out`
- Run: `33745199681`
- Frozen commit: `f62250a7f2bb4e1716b8fbea8565ce12608a1554`
- Artifact: `aaf-cross-domain-heldout-results`
- Artifact ID: `9889330911`
- Artifact SHA-256: `0cc67779d8a1d560c1f0757180f43d70d78f7a2d2b064d51e592534531093fe4`

Verified results:

| Group | n | Dominant-domain | AAF no-interaction | Full AAF |
|---|---:|---:|---:|---:|
| Single-domain | 4 | 4/4 | 4/4 | 4/4 |
| Straight compound | 4 | 4/4 | 3/4 | 4/4 |
| Decision-critical cross-domain | 8 | 4/8 | 4/8 | 8/8 |
| All held-out cases | 16 | 12/16 | 11/16 | 16/16 |

The decision-critical block is the mechanism-focused comparison: Full AAF resolves all eight frozen cases while both reduced baselines resolve four of eight.

## 3. One-at-a-time parameter robustness

Frozen execution:

- Workflow: `AAF Cross-Domain Held-Out`
- Run: `33745199681`
- Frozen commit: `f62250a7f2bb4e1716b8fbea8565ce12608a1554`
- Artifact: `aaf-parameter-sensitivity-2026`
- Artifact ID: `9889331591`
- Artifact SHA-256: `31d129c144bb164b53fb0202a056eeba2f5d445adca3c50088f3e2caae380b61`

The robustness analysis is one-at-a-time, not a joint factorial search. Across the tested `tau`, `delta_min`, and utility-weight settings, action stability versus the default configuration is `100%`. The artifact also contains historical `lambda` rows; `lambda` is no longer manuscript-active because embedding-based semantic similarity was removed. It must not be presented as an active tuned parameter. `beta` in manuscript notation corresponds to the minimum evidence-improvement threshold (`delta_min`).

## 4. Sock Shop HRT-01--HRT-21 frozen block

The independently frozen Sock Shop runtime block reports:

- Full AAF: `19/21 = 90.5%`
- AAF without interaction reasoning: `15/21 = 71.4%`
- Dominant-domain baseline: `9/21 = 42.9%`
- Full AAF vs dominant-domain exact paired test: `p = 0.001953125` (reported as `p = 0.0020`)
- Full AAF vs no-interaction exact paired test: `p = 0.21875`

HRT-21 was subsequently repeated five times only as a diagnostic. Those repetitions are not independent templates and are excluded from `n`.

## 5. Sock Shop HRT-22--HRT-31 extension

Frozen execution:

- Workflow: `Sock Shop Extended Frozen Validation`
- Run: `33880374847`
- Frozen commit: `6055bb6dda1dbbfd9c667985722770009e1e3c4e`
- Artifact: `aaf-sock-shop-extended-validation`
- Artifact ID: `9940876850`
- Artifact SHA-256: `6a5382cfe2fa696b9cee2563edd2e2e4ae1bc642a9bef3e6a5f12f5181776b9b`

Verified HRT-22--HRT-31 results:

- Full AAF: `8/10 = 80%`
- AAF without interaction reasoning: `9/10 = 90%`
- Dominant-domain baseline: `5/10 = 50%`

Combined with HRT-01--HRT-21, the manuscript aggregate is therefore:

- Full AAF: `27/31 = 87.1%`
- AAF without interaction reasoning: `24/31 = 77.4%`
- Dominant-domain baseline: `14/31 = 45.2%`
- Wilson 95% CI for Full AAF 27/31: approximately `[71.1%, 94.9%]`

The five HRT-21 diagnostic repetitions in this artifact consistently reproduce the known over-escalation (`Rollback` via generic multi-domain accumulation) and remain excluded from the independent-case denominator.

## 6. Prospective Sock Shop HRT-32--HRT-39

Frozen execution:

- Workflow run: `33885779782`
- Frozen commit: `67f187d0ce9efcd93a572fee71c944dd447745b0`
- Artifact: `aaf-v2-prospective-sock-shop`
- Artifact ID: `9942511529`
- Artifact SHA-256: `103a997b0f45a7662bb7df4f556a4be61aa9fdb0178e19e37efbc4fa15d1bc41`

Verified prospective results:

- Finalized arbitration AAF: `8/8`
- Initial arbitration formulation: `5/8`
- AAF without interaction reasoning: `7/8`
- Dominant-domain baseline: `3/8`
- Finalized-only wins vs initial: `3`
- Initial-only wins vs finalized: `0`
- Two-sided exact paired test: `p = 0.25`

This block is mechanism validation after the arbitration rule was finalized; it is not presented as a high-powered superiority study. The historical artifact label `v2` denotes the implementation artifact used for the prospective run, not a public framework version name.

## 7. Google Online Boutique OB-01--OB-16

Online Boutique was evaluated as two independently frozen eight-template blocks using the same pinned benchmark revision `b9a978db9e01f4ad3dca9494a22cb9edc17548fe`.

### OB-01--OB-08

- Workflow: `Online Boutique Frozen Second Benchmark - Harness Fix`
- Run: `33873029932` (successful final attempt)
- Frozen commit: `a6c5718d0dc4ebf6040a2aa49934fb73c488c33a`
- Artifact: `aaf-online-boutique-second-benchmark-nodeport-final`
- Artifact ID: `9937642578`
- Artifact SHA-256: `bad9df0f0931ccf3f291b563846aa80e201f7f0c7a409ae84d2aa6ed947c5ea2`
- Full AAF: `8/8`; no-interaction: `6/8`; dominant-domain: `4/8`.

### OB-09--OB-16

- Workflow: `Online Boutique Extended Frozen Validation`
- Run: `33880508662`
- Frozen commit: `fdf222fc6ba0079633693a38cd9762489c414d18`
- Artifact: `aaf-online-boutique-extended-validation`
- Artifact ID: `9940195074`
- Artifact SHA-256: `8f45d89e9b29d27d6f49462f890eb9372f414ffa92f725cceab05b29b85f7116`
- Full AAF: `8/8`; no-interaction: `6/8`; dominant-domain: `4/8`.

Combined manuscript result:

- Full AAF: `16/16 = 100%`
- AAF without interaction reasoning: `12/16 = 75%`
- Dominant-domain baseline: `8/16 = 50%`
- Full AAF vs dominant-domain exact paired test: `p = 0.0078125`
- Full AAF vs no-interaction exact paired test: `p = 0.125`

This resolves the apparent OB-01--08 versus OB-01--16 discrepancy: `main` historically contained the first eight-case manifest, while the second frozen block is preserved on the extension provenance branch and in its successful Actions artifact. Exact historical reproduction of OB-09--16 should use frozen commit `fdf222fc6ba0079633693a38cd9762489c414d18`; do not silently rewrite its freeze hashes against newer `main` code.

## 8. Learned baselines

Current-main clean reproduction:

- Workflow: `AAF Learned Baseline Reproduction`
- Run: `34238573913`
- Commit: `d025254ba0717fc6d5bc2f86f05ce91622dcb8d4`
- Artifact: `aaf-learned-baseline-results`
- Artifact ID: `10060949917`
- Artifact SHA-256: `cd2029f5634e20e9374831a8e9a49a2b545d0800b6bb4e4b9892a124e1e28768`

Training/calibration design: 30 controlled template families x four frozen seeds (`42--45`) = 120 calibration instances; evaluation is on the 13 frozen HRT templates.

Verified held-out action agreement:

- Logistic regression: `6/13 = 46.2%`
- Shallow decision tree: `3/13 = 23.1%`
- Full AAF: `12/13 = 92.3%`
- Logistic regression vs Full AAF exact paired test: `p = 0.03125`
- Shallow tree vs Full AAF exact paired test: `p = 0.01171875`

## Figure and result-file policy

Raw runtime evidence can be large and is retained in the cited GitHub Actions artifacts. Manuscript-facing tables/figures must be generated only from verified frozen outputs or from deterministic reruns whose provenance is recorded here. Do not reconstruct missing case data from manuscript percentages.

At the time this map was added, the repository still needed a canonical committed manuscript figure set. Until those files and their plotting source are added, the Actions artifacts and this map are the authoritative result provenance.
