# AAF v3 improvement round — results

Plan: `PLAN.md` (committed before design; amendments A1–A4). v3 was designed on development data only (RCAEval RE2,
the 40 development postmortems) and frozen (A2) before any holdout was touched. Holdout numbers below are the results;
development numbers are listed only for comparison.

## H1/H2 — RCAEval holdout (telemetry side: change-aware materiality)

Run `RE1_RE3_20261021T123732Z` (GitHub Actions 39600613441), 0 errors. Every case has a fault window and a control
window 300 s earlier. Admissible: resource faults {Mitigate, Scale}; network {Mitigate}; code-level {Rollback, Block
release, Patch, Mitigate}; controls {No action}.

| Suite | Arm | Admissible (all windows) | Detection (fault) | False alarm (control) |
| --- | --- | --- | --- | --- |
| RE1 (375 cases) | AAF v1 | 73.2% (549/750) | 49.6% | 3.2% |
| | **AAF v3** | **94.8% (711/750)** | 94.1% | 4.5% |
| | change baseline (≥1.5×) | 85.1% (638/750) | 98.1% | 28.0% |
| RE3 (90 cases) | AAF v1 | 50.6% (91/180) | 2.2% | 1.1% |
| | **AAF v3** | **96.7% (174/180)** | 100% | 6.7% |
| | change baseline | 84.4% (152/180) | 100% | 31.1% |

Primary (paired exact McNemar, Holm over the two suites): v3 vs v1 — RE1 167 vs 5, p = 8e-43; RE3 88 vs 5, p = 1e-20.
Secondary: v3 vs baseline — RE1 88 vs 15, p = 1e-13; RE3 22 vs 0, p = 5e-7.

Limits: false alarms rose from 1.9% on development (RE2) to 4.5% (RE1) and 6.7% (RE3), concentrated in Train Ticket
(RE1-TT 8.8%, RE3-TT 20.0%); RE1-OB latency is a mean, not p90 (A1), and the baseline is slightly better there
(96.0% vs 94.4%); network-loss detection is the weakest fault type (80.0%). RCAEval measures detection plus a broad
action class, not the choice between Rollback and Mitigate.

## H3 — postmortem holdout (text side: categorical evidence + suspected-cause rule)

Run `holdout/20261021T124517Z` (GitHub Actions 39601225562): 40 new GitLab incidents (GL-058..GL-180, screened after
the freeze), all evidence hashes matched, 0 failed calls, 5 runs per arm. Label distribution: Mitigate 17, Rollback 15,
Patch 5, Scale 3. Answer-key check by the blind AI second rater: 31/40 agreement, κ = 0.67 (0.75 where both include).

| Arm | Agreement with responders | Wilson 95% | Modal non-decision | Wrong committed |
| --- | --- | --- | --- | --- |
| M0 majority ("Mitigate") | 17/40 (42.5%) | 28.5–57.8 | – | – |
| AAF v1 reader (L) | 0/40 (0%) | 0–8.8 | 37/40 | 3 |
| **AAF v3** | **17/40 (42.5%)** | 28.5–57.8 | 9/40 | 14 |
| AI-only (A) | 12/40 (30.0%) | 18.1–45.4 | 13/40 | 15 |

Primary (Holm): v3 vs v1 17 vs 0, p = 3e-5 (**v3 better**); v3 vs M0 10 vs 10, p = 1.0 (**no difference**).
Secondary: v3 vs AI-only 9 vs 4, p = 0.27 (not significant). By action: Rollback v3 10/15 vs AI-only 4/15;
Mitigate 7/17 each; Patch 0/5 for every arm; Scale v3 0/3, AI-only 1/3.

Sensitivity to the answer key: agreed cases only (31): v3 15, M0 13, A 11, L 0; rater-2 labels (37 included): v3 16,
M0 16, A 12, L 0. v3 vs v1 stays p < 1e-4; v3 vs M0 and v3 vs A stay non-significant.

Development comparison (not results): on the 40 development postmortems v3 reached 18/40 vs M0 19/40 — the holdout
reproduces the development picture.

## What this supports

- **Measured telemetry:** change-aware materiality fixes AAF's main weakness and generalises to new systems' fault
  suites and to code-level faults: v3 is the most accurate arm on both holdouts, with false alarms far below the change
  baseline but above v1 and the development estimate.
- **Free text:** qualitative facts with verbatim quotes lift AAF from never acting (0/40) to the level of the majority
  baseline (17/40); it now commits on 31/40 cases and gets Rollback right more often than AI-only. It is **not** better
  than always answering "Mitigate", and it is not significantly better than AI-only. Text-only first-action prediction
  remains hard for every arm.
