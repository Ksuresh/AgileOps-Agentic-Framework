# Frozen RCAEval External Validation Protocol

This file freezes the external-validation protocol before the untouched held-out RCAEval cases are executed.

## Dataset and split

- Dataset: RCAEval, RE2-TrainTicket (`RE2-TT`)
- Total cases: 90
- Adapter-development pilot: first two lexicographically sorted cases from each of the six fault classes (12 cases total)
- Held-out evaluation: all remaining 78 cases
- Pilot cases are excluded from held-out result denominators.

## Information-flow rule

AAF receives only source telemetry and the injection-time boundary. RCAEval fault labels and root-cause service annotations are audit metadata only and are appended after the AAF decision has been produced.

## Frozen adapter policy

Decision-active mappings:

1. RCAEval CPU percentage -> AAF SRE `saturation_pct`, only when the post-injection 5-minute median is at least 1.5x the pre-injection 5-minute median.
2. RCAEval service p90 latency -> conservative lower-bound proxy for AAF `p95_latency_ms`, converted to milliseconds and only when the post-injection 5-minute median is at least 1.5x the pre-injection median.

Auxiliary evidence retained for audit/explanation but not coerced into unrelated AAF fields:

- memory
- disk I/O
- socket activity
- available raw logs / error-like log activity

No RCAEval-specific changes are made to shared AAF severity thresholds, materiality logic, interaction arbitration, utility weights, or action selection.

## AAF configuration

- Utility weights: `(0.45, 0.20, 0.35)`
- Shared AAF decision logic remains unchanged from the manuscript configuration.
- The external adapter is dataset-specific and does not modify Sock Shop, Online Boutique, or controlled-experiment pipelines.

## Reporting rule

The held-out experiment reports behavior stratified by fault class and observed AAF action. Because RCAEval does not provide governance-action ground truth, injected fault labels are not treated as action labels and no fabricated action-accuracy metric is reported. The external study is a generalization/behavioral validation of AAF under independently generated fault evidence.

The pilot and held-out runs preserve full case-level JSON, a CSV summary, and a machine-readable aggregate summary under `paper_results/external_rcaeval/generated/` via GitHub Actions artifacts.
