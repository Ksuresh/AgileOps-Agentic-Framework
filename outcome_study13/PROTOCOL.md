# Study 13: Security priority and risk-based release gating (frozen before any Study 13 run)

## Question
AAF's policy treats security as a priority domain but gates on risk: critical CVEs and policy violations are material
(Patch or block release), whereas HIGH/MEDIUM findings alone do not block a release when the other domains are healthy.
Study 13 checks both behaviours on the live testbed against the outcome oracle, and contrasts them with a fixed policy
that blocks on any finding (`BlockAnyCVE`) and with LLM-decided baselines that see the same findings.

## Design
- Same testbed, actions, measurement, thresholds and oracles as Studies 9 and 12; additionally the HIGH+MEDIUM finding
  count (trivy) is measured and given to every decider as `sec.noncritical_cves` (AAF's rules do not use it).
- Images: L1, L2 = low-risk images chosen by the fixed rule in `outcome_study/calibrate13.py` (0 critical CVEs, >= 1
  HIGH/MEDIUM finding, rolls out and serves); A, C, D = the critical-CVE images of Studies 9 and 12.
- Cases (`cases.yaml`, 7): 4 low-risk releases (LR), 3 critical releases (CR); all other domains healthy. Two replicates.
- Deciders: AAF, AAF-12 (Qwen, 3 runs), B1, B1-12, B2, B5, B6 (Qwen), D-* siloed experts, BlockAnyCVE, AlwaysRollback, AlwaysNone.

## Expectations and reporting (fixed now)
- **E13.1 (low risk):** on LR cases AAF continues the release (an action that keeps the feature delivered and is O1-correct,
  i.e. the O2-correct set). Reported per decider as k/4.
- **E13.2 (critical):** on CR cases AAF selects an action that removes the critical CVEs (O1-correct). Reported as k/3.
- With 7 cases this is a descriptive check of the stated policy, not a significance test; all outcomes are reported.

## Amendment A1 (after calibration, before any outcome measurement)
The selection rule found only one low-risk image: the stable image `frontend:v0.10.6` itself (0 critical, 32 HIGH/MEDIUM
findings); the other candidate tags could not be scanned and no older tag has 0 critical CVEs. L2 is therefore undefined,
so LR-02 and LR-04 (the L2 cases) are dropped. Study 13 has 2 low-risk cases (LR-01, LR-03) and 3 critical cases.
Note: because the stable image carries 32 HIGH/MEDIUM findings, every healthy state in Studies 9 and 12 was also a
low-risk-finding state; Study 13 makes that finding visible to the deciders.

## Execution notes (appended after the study)
- One decider batch stopped because the block-on-any-finding comparison policy received a withheld (missing) CVE count; the
  policy was fixed to treat a missing count as zero and the batch was rerun (commits 83eca4f, ba43872). No AAF code changed.
