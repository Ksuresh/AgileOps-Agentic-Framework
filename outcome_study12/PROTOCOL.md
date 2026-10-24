# Study 12: Revised governance rules tested on fresh outcome-measured cases (frozen before any Study 12 run)

## Origin
Study 9 (outcome-based oracle) showed two gaps in AAF's rules: (1) with no signal that a release had just been deployed,
AAF patched or mitigated release-caused regressions that only a rollback fixed (7 compound cases); (2) cost actions were
chosen while critical CVEs were present (2 cases). Study 9 is therefore **development data** for Study 12, as Study 7 was
for Study 8.

## Revision (`agentic_experiments/rules_v12.py`), fixed now
- New measured evidence field `deploy.recent_release`: the two newest frontend ReplicaSets differ in image, environment or
  resources, and the newest is younger than 30 minutes (`testbed.release_info`).
- R1 release regression: recent release and (reliability material or restart loops >= 3) -> Rollback to stable deployment.
- R2 security over cost: security material and the selected action is Scale adjustment, Review scaling policy or
  No action -> Patch or block release.
- All other rules, thresholds, weights, the selective trigger, the investigation and the prompts are unchanged.
- Development checks (reported, not hypotheses): on Study 9's 30 fault cases the revised deterministic rules choose an
  O1-correct action in 30 (unrevised: 21); on the 96 Study 6/7 cases (no release field, so only R2 can act) one action
  changes (RP-M04, Scale adjustment -> Patch or block release; correct -> incorrect on the authors' key).

## Fresh test cases (`cases.yaml`, 26)
2 controls; 7 release-caused regressions; 7 cases where the fault predates the release, so a rollback cannot fix it (these
test whether R1 over-applies); 5 security-vs-cost cases; 5 single-domain cases. New CPU budgets (30m, 50m), loads (60, 80
clients), replica counts (2, 4, 5, 6, 7) and two new vulnerable images (C = frontend v0.6.0, D = v0.5.0; next in the Study 9
selection order, 6 critical CVEs each). Same service (frontend), testbed, actions, measurement, violation thresholds
(from Study 9's calibration, unchanged) and oracle (O1 primary; O2, O3 secondary). Two replicates.

## Deciders
AAF-12 (revised rules + selective investigation, Qwen3-8B), AAF (Study 9 rules), B1-12 (revised rules, no LLM), B1, B2,
B3 (from B5), B5, B6 (Qwen; 3 runs each LLM arm, majority), siloed experts D-* and L-*, Dominant, AlwaysRollback, AlwaysNone.

## Hypotheses (O1; exact McNemar on paired per-case correctness)
- **H12.1 (primary):** AAF-12 is correct more often than AAF on the 24 fault cases (two-sided, alpha 0.05).
- **H12.2:** AAF-12 vs. each siloed expert (8) on the 24 fault cases, Holm-adjusted.
- **H12.3:** AAF-12 vs. B5, B6, AlwaysRollback on the 24 fault cases, Holm-adjusted.
- **H12.4 (over-application):** on the 7 predating-fault cases, AAF-12's correctness vs. AAF's (reported with counts).
- Withheld-evidence variants (as in Study 9b) reported as secondary.
AAF-12 can lose any comparison, including to the unrevised AAF; all results will be reported.
