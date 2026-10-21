# RCAEval RE2 governance study — results (pre-registered, deterministic)

Protocol: `PROTOCOL.md` (committed before the run, cd23afc). Run: GitHub Actions on `text-evidence-pilot`, results commit 8363eb7. No LLM calls. No amendments.

## Primary (held-out: 258 cases → 258 fault windows + 258 healthy control windows)

| Arm | Admissible, all windows | Detection on fault windows | False alarm on healthy windows |
| --- | --- | --- | --- |
| AAF (frozen adapter and thresholds) | 343/516 = 66.5% [62.3, 70.4] | 90/258 = 34.9% [29.3, 40.9] | 5/258 = 1.9% [0.8, 4.5] |
| Change-detection baseline (≥1.5× post/pre) | 466/516 = 90.3% [87.5, 92.6] | 256/258 = 99.2% [97.2, 99.8] | 48/258 = 18.6% [14.3, 23.8] |

Paired exact McNemar on all held-out windows: AAF-only correct 43, baseline-only correct 166, p < 1e-6. Wilson 95% CIs in brackets. Fault-pair majority (effective n = 90 service-fault pairs): AAF 35/90, baseline 90/90.

## By fault (held-out, 43 cases each)

| Fault | AAF detection | Baseline detection | AAF false alarm | Baseline false alarm |
| --- | --- | --- | --- | --- |
| cpu | 2/43 | 43/43 | 0/43 | 9/43 |
| mem | 19/43 | 43/43 | 0/43 | 8/43 |
| disk | 6/43 | 43/43 | 3/43 | 11/43 |
| socket | 11/43 | 41/43 | 0/43 | 4/43 |
| delay | 26/43 | 43/43 | 1/43 | 5/43 |
| loss | 26/43 | 43/43 | 1/43 | 11/43 |

## By system (held-out)

| System | AAF admissible / detection / false alarm | Baseline admissible / detection / false alarm |
| --- | --- | --- |
| Online Boutique | 72.2% / 44.4% / 0.0% | 98.9% / 97.8% / 0.0% |
| Sock Shop | 62.8% / 26.7% / 1.1% | 96.1% / 100% / 7.8% |
| Train Ticket | 64.1% / 33.3% / 5.1% | 73.7% / 100% / 52.6% |

## Adapter-development cases (12 RE2-TT cases, reported separately)

AAF 22/24 = 91.7% admissible (detection 10/12, false alarm 0/12); baseline 15/24 = 62.5% (false alarm 8/12). AAF's performance on the cases used to build its adapter does not carry over to held-out cases.

## Why AAF misses resource faults

On CPU-fault windows the CPU value AAF receives is 10–80% (median 40%); AAF's saturation thresholds are 85% (elevated) and 90% (severe). The CPU stress is clearly visible as a relative change (e.g. 0.4% → 20%, 14% → 70%) but never crosses the absolute thresholds, so AAF observes. The earlier manuscript's "no action on 12 of 13 CPU faults" was therefore missed detection caused by absolute thresholds, not evidence-based selectivity — as reviewers R4, R5 and R6 suspected.

## Interpretation

- Under the pre-registered proxy oracle, the simple baseline is substantially better overall. AAF is highly specific (few false alarms) but insensitive on this telemetry.
- The baseline is sensitive but noisy, especially on Train Ticket (52.6% false alarms), so neither arm is a good governance policy as is; the trade-off is the finding.
- The gap between development and held-out cases is a concrete overfitting signal for the frozen adapter.
- Any change (e.g. relative-change-aware materiality) must be a separately labelled exploratory analysis with its own held-out evaluation, not a retroactive fix.
