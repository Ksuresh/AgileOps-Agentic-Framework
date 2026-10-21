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
