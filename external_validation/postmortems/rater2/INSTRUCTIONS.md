# Rater-2 instructions (independent check of the answer key)

You are the second, independent rater. Please do **not** open `labels_rater1*.csv`, `evidence_hashes.json`,
`results/` or any run output in this repository before you have finished: the check only counts if you are
blind to rater 1 and to the arms. The sheet `rater2_sheet.csv` lists 80 public incident reports in random order
(a mix of documents rater 1 included and excluded; you are not told which).

For each URL (for GitLab issues, read the issue's comments in time order and ignore the description):

1. **Include?** Write `yes` only if all three hold, otherwise `no` and the reason:
   - there is a timeline of timestamped or clearly ordered entries;
   - at least one observation of the affected system is written down **before** the first human response action;
   - the first human response action is stated explicitly and fits one of the four rows below.
   Exclude: security breaches; data loss by human error; hardware, power, cooling or facility failures; purely
   upstream-provider outages; incidents whose first effective action was automatic; narrative-only documents.
2. **First action quote:** copy the few words that state the first human response action.
3. **Action label** (one of four):

| Responders' first action | Label |
| --- | --- |
| Revert, roll back, or disable/undo a change deployed shortly before the impact | Rollback |
| Halt or pause deploys/releases, release freeze, ship a hotfix or patch | Patch |
| Add capacity, scale out or up | Scale |
| Restart, failover, reroute or shed traffic, rate-limit, disable a feature not just deployed, run a repair procedure | Mitigate |

Use `notes` for anything ambiguous. Expect about 1–2 minutes per document; return the filled CSV.
Agreement (Cohen's kappa) is computed for inclusion on all 80 and for the label on documents both raters include;
disagreements are resolved by discussion and logged.
