# Study 15: Frozen rule set v1.4 on fresh cases across three services (frozen before calibration and before any Study 15 run)

## Purpose
Study 14 tested rule set v1.4 once, on 34 fault cases across two services. Three questions were left with too little data:
(i) AAF-14 against Qwen deciding with tools (B6) was 34 vs 29 with 5 vs 0 discordant cases (p = 0.0625); (ii) the
risk-based release gate (low-risk findings with healthy other domains should not stop the release) had four cases; (iii)
the evaluation covered two services. Study 15 is a confirmatory replication with a larger, fresh case set and a third service.
No rule, threshold, prompt, model, trigger, utility weight or oracle changes. Studies 9-14 are not reused as test data.

## What is frozen
- Rules: `agentic_experiments/rules_v14.py` exactly as used in Study 14 (commit of this protocol). No edits until Study 15 is analysed.
- Testbed, actions, measurement, violation thresholds (Study 9 calibration: p95 > 816 ms or errors > 1%; critical CVEs above
  stable or a secret in config; footprint > 300m) and oracles (O1 primary; O2 secondary: keep the feature, then lower
  footprint; O3 Pareto) as in Study 14.
- Language model and runtime: Qwen3-8B Q4_K_M via Ollama 0.40.9, temperature 0.7, context 8192, prompts unchanged; three runs
  per LLM arm and variant, majority action.

## Fresh cases (`cases.yaml`, 57; 42 fault cases)
Three services, 19 cases each: frontend (UF-*), productcatalogservice (UP-*) and, new, currencyservice (UC-*). The latter two
are reached through the storefront home page, so their faults show up in the measured storefront latency and errors.
Per service: 1 control (C); 4 low-risk releases (L01-L02 stable image with the feature; L03-L04 an image with more
HIGH/MEDIUM findings than stable and no extra critical CVE); 3 release-caused regressions (R); 3 faults present before the
release (P, rollback cannot fix them); 3 security vs cost (S); 2 security issue with other domains healthy (K, one critical
CVE, one exposed secret); 1 cost only (D); 2 compound (X). Fault groups for the primary analysis: R, P, S, K, D, X
(14 per service, 42 in all). The case file is written now; three quantities in it are symbolic and are filled by the fixed
calibration rules below, before any outcome is measured.

## Calibration rules (`outcome_study/calibrate15.py`; no action outcomes measured)
1. Vulnerable images per service (frontend G, H; productcatalogservice P3, P4; currencyservice C1, C2): candidate tags
   v0.9.0 down to v0.1.0, gcr.io then us-central1-docker.pkg.dev; keep images that scan, carry >= 1 critical CVE more than
   the stable image, roll out and keep the storefront serving; first two tags not used in Studies 9-14. Only if fewer than
   two qualify are previously used tags considered, in the same order.
2. Low-risk image per service (NF, NP, NC): tags v0.10.5 down to v0.9.0; no more critical CVEs than stable, more HIGH/MEDIUM
   findings than stable, rolls out and serves; first that qualifies. If none qualifies, L03-L04 of that service use the
   stable image (recorded in the measurement file as `lowrisk_fallback`).
3. Degradation level per service (`PERF`): CPU limit candidates 50m, 30m, 20m, 10m, 5m, largest first, at 1 replica under
   normal load; the largest that violates the reliability threshold. If none violates, 5m (recorded). This replaces the
   fixed 30m/50m of Study 14, where the productcatalogservice degradation did not violate the threshold.
4. The invalid-port release is applied once per service and its rollout and storefront errors are recorded (descriptive).

## Deciders
AAF-14 (v1.4 rules + selective investigation, Qwen3-8B); B1-14 (v1.4 rules, no investigation); B2-14 (v1.4 rules + fetch
every missing field); B3 and B3-14 (v1.0 / v1.4 rules re-deciding on B5's retrieved evidence); B5, B6 (LLM decides; B6 with
thinking and tools); siloed rule experts with v1.4 domain rules (D-DevOps-14, D-SRE-14, D-FinOps-14, D-Sec-14) and, for
continuity with Studies 9-14, with v1.0 domain rules (D-*); LLM domain agents (L-*, the domain agents' proposals in the B5
traces, as in Study 14); B1, B1-12, B2, Dominant, AlwaysRollback, AlwaysNone, BlockAnyCVE. LLM arms: AAF-14, B5, B6 (the original-rule
hybrid AAF was tested in Studies 9-14 and is not rerun).

## Hypotheses (exact McNemar on paired per-case O1 correctness, fault cases n = 42 unless stated)
- **H15.1 (primary):** AAF-14 vs each siloed expert: D-DevOps-14, D-SRE-14, D-FinOps-14, D-Sec-14, L-DevOps, L-SRE, L-FinOps,
  L-Sec; Holm over 8.
- **H15.2:** AAF-14 vs B5, B6 and AlwaysRollback; Holm over 3.
- **H15.3 (pooled confirmation, Studies 14 + 15, n = 34 + 42 = 76):** AAF-14 vs B6, B5, AlwaysRollback and D-Sec (v1.0 rules,
  available in both studies); Holm over 4.
- **H15.4 (risk-based release gate):** on the 12 L cases, the share where AAF-14's action is O2-correct (the release
  continues with the lower footprint), with a Wilson 95% interval; McNemar against B5, B6, AlwaysRollback and L-Sec,
  unadjusted (descriptive). Pooled with Study 14's L cases (n = 16) as well.
- **H15.5 (security priority, descriptive):** on S, K and security-carrying R/X cases, the share where the chosen action
  removes the critical CVE or secret (O1).
- **H15.6 (generalisation, descriptive):** H15.1-H15.2 counts per service; currencyservice is new.
- **H15.7 (withheld evidence, secondary):** on the withheld-evidence variants (as Study 14), AAF-14 vs B1-14 and vs B2-14.
  Study 14 found the fetch rule at least as accurate as the LLM investigator; this is a replication, not a test we expect to win.
- **Descriptive:** number of discriminating cases (where not every action is O1-correct) and the H15.1-H15.2 counts on them;
  O1-set agreement between the two measurement replicates; LLM calls and tokens per case.
Any comparison may go against AAF; every pre-specified comparison is reported.

## Execution
Calibration (one job), then two measurement replicates (sharded; decisions use replicate 1 snapshots), then deciders
(sharded). Runs that hit runner limits are resumed with identical settings; completed results are not rerun. Deviations are
appended below as execution notes.
