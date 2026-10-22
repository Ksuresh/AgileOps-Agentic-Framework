# Provenance timeline (Access-2026-46118 revision)

Built from git history and GitHub Actions run records (`gh run view`). "Frozen" = the dataset/protocol file was
committed before the run and its SHA-256 is unchanged in the current repository. All times UTC unless marked IST.

## Original manuscript studies

| Study | Dataset / protocol commit (first and only) | Run | Run head = frozen commit | Dataset hash unchanged |
| --- | --- | --- | --- | --- |
| Controlled held-out (Study 1) | f62250a (2026-09-03 10:36) | 33745199681, 2026-09-03 10:36 | yes | — |
| Sock Shop HRT-22–31 | f7abd6b (2026-09-04 13:50) | 33880374847, 2026-09-04 13:51 | yes (6055bb6) | yes |
| Online Boutique OB-01–08 | a6c5718 (2026-09-04 12:29) | 33873029932, 2026-09-04 12:29 | yes | — |
| Online Boutique OB-09–16 | 121567d (2026-09-04 13:51) | 33880508662, 2026-09-04 13:52 | yes (fdf222f) | yes |
| Prospective HRT-32–39 | 742e380 (2026-09-04 14:46) | 33885779782, 2026-09-04 14:47 | yes (67f187d) | yes |
| RCAEval RE2-TT (original) | 88e7c6e (2026-09-09 05:47) | 34316286523, 2026-09-09 05:47 | yes | — |
| Selective AEI primary (32) | cases f04bb19 (2026-09-10 01:27); protocol 79a39cd (01:26) | 34428945881, 2026-09-10 02:18 | yes (3f5913f) | yes (98fd9427…) |
| Selective AEI replication (16) | cases d3f8b27 (2026-09-10 02:55) | 34432645600, 2026-09-10 03:15 | yes (2417f2f) | yes (eb41fc78…) |

**Disclosure — primary selective-AEI study.** After the 32 cases and the protocol were committed, a development run
(34427322163, 01:54) was executed on the same cases, and the selective trigger was then refined twice (86ecac5 "Fix
selective trigger and preserve deterministic final authority", c697724 "Refine selective trigger to ambiguity rather
than severity") before the reported run 34428945881 (02:18). The trigger was therefore not fully prospective for the
primary 32 cases. Between the primary run and the 16-case replication, no framework code changed (only the replication
dataset, runner and workflow were added), so the replication is the prospective test of the final trigger.

## Revision studies (this resubmission)

| Study | Protocol committed (before any run) | Run(s) | Results committed | Amendments |
| --- | --- | --- | --- | --- |
| RCAEval RE2 with healthy controls | cd23afc, 2026-10-21 10:01 | 39581484587 | 8363eb7, 10:04 | none |
| AAF-CA (change-aware) dev/holdout | ed530d0, 2026-10-21 11:56 (before design) | dev on RE2; holdout 39600613441 | 824362a, 12:46 | A1–A4 in PLAN.md (A1 before holdout; A4 runner error capture) |
| Claim-level PM-interface re-test (E1) | 41a84a4, 2026-10-21 12:56 | 39603634523 | 63cbae8, 13:25 | A1–A4 in PROTOCOL.md (later variants exploratory) |
| Reviewer-requested LLM arms | 8818492, 2026-10-22 06:48 | 39736227167 | 2b19364, 07:04 | none |
| Study 7: 48 new cases, unflagged gaps | protocol, cases, sealed author key 627177e, 2026-10-22 12:24; practitioner labels e69ead3 (before run) | 39784044792, 13:07 | 67a1889, 13:13; analysis script committed before results | A1 (schema accepts `stale`, decision-neutral), A2 (labels are AI-assisted) |
| Study 8: investigator fix, 36 fresh cases | protocol, cases, sealed key, hybrid_v2.py bf61833, 2026-10-22 13:23 | 39786446655, 13:24 | f28eda9, 13:38; analysis a6994d0 before results | none |
| Deterministic analyses (auto-fetch, per-stratum, boundary, chance, rescoring) | scripts in revision_2026/ | local, no model calls | see git log | — |

## Automated audit

`python revision_2026/audit_provenance.py` re-derives the table above from git history and the GitHub Actions run records: for each study it checks that every frozen file was committed before the run started, that the run's head commit holds the same file content as today (or the original text followed only by appended, documented amendments), that results were committed after the run, and that the run succeeded. Output: `revision_2026/results/provenance_audit.json`. All ten audited studies pass; the E1 protocol is the one file with appended amendments.
