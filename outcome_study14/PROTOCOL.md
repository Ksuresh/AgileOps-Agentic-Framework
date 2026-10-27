# Study 14: Refined governance rules on fresh cases across two services (frozen before any Study 14 run)

## Origin
Study 12 showed that a release-regression rule (R1) recovers release-caused faults but over-fires when the fault predated
the release; Study 13 showed that the original rules let a cost action outrank a critical CVE and that R1 fired on latency
noise near the fixed 450 ms threshold. Studies 9, 12 and 13 are development data for Study 14.

## Revision (`agentic_experiments/rules_v14.py`), fixed now
- R1' release regression: recent release (measured from ReplicaSet history) and the release degraded reliability relative
  to its own pre-release baseline (p95 >= 450 ms and >= 2x the pre-release p95, or error rate >= 8% and >= pre-release + 5
  points), or the new pods restart (>= 3) -> Rollback to stable deployment.
- R2 security over cost (as v12): critical CVEs or a policy violation outrank cost-only actions -> Patch or block release.
- Risk-based gating (unchanged): HIGH/MEDIUM findings alone are not material; with other domains healthy the release continues.
- New measured evidence: `sre.p95_pre_release_ms`, `sre.error_rate_pre_release_pct` (45 s under the case's load before
  the release is applied), `deploy.recent_release`, `sec.noncritical_cves`.
- Development check (O1, using each case's measured previous-release state as the baseline): Study 9 32/32, Study 12 26/26,
  Study 13 5/5. The factor 2 was chosen on these data (factor 1.5 gave equal O1 and lower O2 on Study 9).
- Nothing else changes (gate, utility, interaction policy, thresholds, trigger, prompts).

## Fresh cases (`cases.yaml`, 40)
Two services: frontend (QF-*) and, for generalisation, productcatalogservice (QP-*, a backend dependency of the storefront;
faults propagate to the measured storefront latency and errors). Per service: 1 control, 2 low-risk releases (stable image,
which carries HIGH/MEDIUM but no critical findings), 4 release-caused regressions, 4 faults present before the release
(traps), 4 security-vs-cost, 2 critical CVE with other domains healthy, 1 cost-only, 2 compound. New images by the fixed
rule in `outcome_study/calibrate14.py` (frontend E, F; productcatalog P1, P2). Testbed, actions, measurement, violation
thresholds (Study 9 calibration) and oracles (O1 primary; O2 secondary: keep the feature, then lower footprint; O3 Pareto)
are unchanged. Two replicates; decisions use replicate 1 snapshots.

## Deciders
AAF-14 (v14 rules + selective investigation, Qwen3-8B), AAF (original rules + investigation), B1-14, B1-12, B1, B2,
B3 (from B5), B5, B6 (Qwen; LLM arms 3 runs, majority), siloed experts D-* (rules on one domain) and L-* (LLM domain agents),
Dominant, AlwaysRollback, AlwaysNone, BlockAnyCVE.

## Hypotheses (exact McNemar on paired per-case O1 correctness; fault cases = groups R, P, S, K, D, X, n = 34)
- **H14.1 (primary):** AAF-14 vs each of the 8 siloed experts, Holm-adjusted over 8.
- **H14.2:** AAF-14 vs AAF (original rules).
- **H14.3:** AAF-14 vs B5, B6, AlwaysRollback, Holm-adjusted over 3.
- **H14.4 (traps):** on the 8 P cases, AAF-14 vs B1-12 (the v12 rule that over-fired) and vs AAF.
- **H14.5 (security policy, descriptive):** K and S cases, critical issue removed (O1); L cases, release continued (O2).
- **H14.6 (generalisation, descriptive):** H14.1-H14.3 counts reported per service.
- **H14.7 (pooled, original AAF frozen since Study 9):** AAF (original) vs each siloed rule expert on the pooled fault cases
  of Studies 9, 12 and 14 (n = 30 + 24 + 34), Holm over 4.
Withheld-evidence variants (as in Study 9b) are secondary. Any comparison may go against AAF; all are reported.

## Calibration result (before any outcome measurement)
Frontend E = `frontend:v0.4.0` (5 critical CVEs), F = `frontend:v0.3.0` (8); productcatalogservice P1 = `v0.8.0` (4),
P2 = `v0.7.0` (5); all gcr.io, all roll out and keep the storefront serving. Stable images (v0.10.6) have 0 critical and
32 HIGH/MEDIUM findings each.

## Execution notes (appended after the study)
- Five decider batches reached the GitHub runner time limit; the remaining 24 of 864 LLM decisions were completed by a resumed
  run with identical settings (commit f411d8f). Completed decisions were not rerun.
- As in Studies 9 and 12, the comparison deciders D-*, B2 and B3 use the v1.0 domain rules; only B1-14 and AAF-14 use v1.4.
  A post hoc recomputation with v1.4 (`posthoc_v14_baselines.py`, no model calls) changes the siloed DevOps expert from 14 to
  17 of 34 (all other siloed experts unchanged; AAF-14 still better than each), and gives a v1.4 fetch rule (B2) 32 of 32
  withheld-evidence variants against 27 for AAF-14 (5 vs 0 discordant, p = 0.0625) and B3 29.
