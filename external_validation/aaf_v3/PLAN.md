# AAF v3 improvement round — pre-registered plan

Status: committed 2026-10-21, **before** AAF v3 is designed and before any holdout data below is inspected beyond
metadata (case counts, fault types, column names). Changes go in the amendment table with reasons.

## Why
The external studies found two weaknesses in AAF v1 (frozen): (1) fixed absolute thresholds miss faults that are
large relative to the system's own normal level (RCAEval RE2: detection 34.9%, CPU 2/43); (2) on free-text evidence
AAF almost never acts, because responders rarely write the numbers its thresholds need (postmortems: 35/40
escalations, 1/40 correct). v3 targets exactly these two and nothing else.

## Data split
| Role | Data | Status |
| --- | --- | --- |
| Development (design and tuning allowed) | RCAEval RE2 (270 cases, fault + control windows); the 40 labelled postmortems and their run outputs; text-pilot case sets | already seen |
| Holdout H1 | RCAEval RE1 (375 cases: OB, SS, TT) | never run |
| Holdout H2 | RCAEval RE3 (90 code-level-fault cases) | never run |
| Holdout H3 | GitLab production incidents continuing the seeded order of A3 from GL-058 (fetch further batches in the same order if needed) until 40 cases are included | not screened; no arm run |

## Allowed v3 changes (designed on development data only)
1. **Change-aware materiality:** a signal is also material when it departs from the same system's own pre-window
   baseline by a fixed ratio, in addition to the v1 absolute thresholds.
2. **Categorical text evidence:** the L+ boolean fields (outage, error spike, latency spike, resource exhaustion,
   change preceded impact), possibly with one further field if development cases show a need, mapped into AAF evidence.
3. **Refined change→rollback rule:** fix the false Rollbacks seen on development postmortems (e.g. require the text to
   link the change to the impact).
Weights, action set, gate structure, tiers and checker v2 stay frozen. v3 is frozen (git commit + SHA-256 of the
changed files) before any holdout is touched; the holdout runs record that hash.

## Holdout outcomes
- **H1/H2 (RCAEval):** windows and controls exactly as `rcaeval_governance/PROTOCOL.md` (fault at inject time,
  control at inject−300 s). Admissible: resource faults {Mitigate, Scale}; network faults {Mitigate}; RE3 code-level
  faults {Rollback, Block release and fix pipeline, Patch or block release, Mitigate}; controls {No action}. Any other
  fault type is reported separately. Arms: AAF v1, AAF v3, change-detection baseline (≥1.5×). Primary: v3 vs v1 on
  admissible rate (paired exact McNemar); secondary: v3 vs baseline, detection on fault windows, false alarms on
  control windows (Wilson 95%).
- **H3 (postmortems):** same screening rules, cut rule, rubric (incl. A6 Patch set), notes-only evidence, encrypted
  handling, AI second rater (A8 prompt) before arms run. Arms: M0 (majority of H3), AAF v1 reader (L), AAF v3 reader,
  AI-only (A). Primary: v3 vs v1 and v3 vs M0 (paired exact McNemar, Holm); secondary: v3 vs A, non-decision rate,
  wrong committed actions.
- Results are reported whatever they are. Development-set numbers for v3 are reported only as development numbers.

## Amendments
| ID | Date | Change | Reason |
| --- | --- | --- | --- |
| A1 | 2026-10-21 | From the metadata probe (column names and fault labels only, `probe_report/summary.json`): RE1 fault types are cpu, mem, disk, delay, loss (mapped as in the plan); RE3 faults f1–f5 are code-level faults (admissible set as in the plan). RE1-OB has service latency only as `<svc>_latency` (no `_latency-90`); for any case without `_latency-90` columns, `_latency` columns are used as the latency family for all arms (v1, v3, baseline) and the case is flagged `latency_columns_renamed`; RE1-OB is also reported separately because this latency is not a p90. | Format difference only; without it every arm is blind to latency on RE1-OB. |
| A2 | 2026-10-21 | **AAF v3 frozen** at commit 5835d66 (SHA-256 prefixes: accf395133843041 text_v3.py;217d95d95b993353 rcaeval_v3.py;51436c547cbb6351 run_text_v3.py;13cde9fd15f4f0df run_rcaeval_v3.py). Development numbers (not results): RE2 admissible v1 67.6% / v3 97.6% / baseline 89.1%, false alarms on controls v1 1.9% / v3 1.9% / baseline 20.7%; 40 dev postmortems agreement M0 47.5% / L 2.5% / v3 iteration 1 35.0% / v3 iteration 2 (frozen) 45.0% (AI-only arm A from the earlier study: 42.5%). Two text iterations were run on development data; no further tuning. Holdout GitLab screening starts only after this commit. | Freeze before holdout, as planned. |
| A3 | 2026-10-21 | Holdout H3 screening complete (after the v3 freeze, before any holdout arm run): GL-058 to GL-180 screened in seeded order (GL-151 onward from the second encrypted batch, `screening/gitlab_151_300.*`, whose order reproduces batch 1), 40 included (Mitigate 17, Rollback 15, Patch 5, Scale 3), 83 excluded with reasons; `holdout/labels_rater1_holdout.csv`, frozen `holdout/evidence_hashes.json`. Same rules as the development set. M0 for H3 is therefore "Mitigate and monitor". | As planned. |
