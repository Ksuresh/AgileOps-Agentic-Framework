# Pre-registered protocol: RCAEval RE2 governance study (deterministic, no LLM)

Status: **committed before any run of this study** (2026-10-21). Changes after the first run go in an amendment table at the end, with reasons.

## Question

On independently generated fault telemetry, how often does AAF take an admissible governance action during injected faults, and how often does it correctly stay silent during the healthy period before each fault? How does it compare with a simple change-detection threshold baseline?

This answers reviewer requests R4.5 (simple threshold baseline; explain "no action" on CPU/disk faults), R5 (proxy oracle from root-cause annotations) and R6 (healthy controls, or reframe).

## Data

- RCAEval RE2 via the public Hugging Face copy `phamquiluan/RCAEval`: 270 cases = 3 systems (Online Boutique, Sock Shop, Train Ticket) × 5 root-cause services × 6 faults (cpu, mem, disk, socket, delay, loss) × 3 repetitions.
- Each case has 12 minutes of normal operation before `inject_time` and about 12 minutes after.
- Only `metrics.parquet` and `inject_time` are used. Fault and root-cause labels are used **only** to score, after the decision.

## Windows (units of observation)

| Window | Definition | n |
| --- | --- | --- |
| Fault | Adapter called with the true `inject_time`: compares [t−300 s, t) with [t, t+300 s) | 270 |
| Control | Adapter called with `inject_time − 300 s`: compares [t−600 s, t−300 s) with [t−300 s, t), both healthy | 270 |

Effective independent units: 90 service-fault pairs (3 repetitions each) for fault windows; 270 control windows share systems and services and are reported with that caveat.

## Proxy oracle (admissible actions)

| Window / fault | Admissible actions |
| --- | --- |
| Control (healthy) | No action (observe) |
| cpu, mem, disk, socket (resource faults) | Mitigate and monitor; Scale adjustment |
| delay, loss (network faults) | Mitigate and monitor |

Rollback, release blocking and patching are never admissible: RE2 contains no deployment or security evidence. "Escalate for evidence/human review" is not admissible in the primary analysis; a lenient secondary analysis counts any non-observe action as detection.

## Arms

1. **AAF (as published for RCAEval):** the frozen adapter `external_validation/rcaeval/adapter.py` and the frozen decision path of `run_heldout.py` (utility weights (0.45, 0.20, 0.35), `choose_action_details` then `apply_interaction_policy_v2`). No threshold, adapter or policy change.
2. **Change-detection baseline:** "Mitigate and monitor" if the largest post/pre median ratio across all latency-90, cpu or mem metrics (any service, label-blind, same windows) is ≥ 1.5; otherwise "No action (observe)". This uses the adapter's own 1.5× gate but skips AAF's absolute thresholds.

## Outcomes

- **Primary:** admissible-action agreement over all 540 windows, per arm; paired exact McNemar AAF vs baseline.
- **Secondary:** detection rate on fault windows (any non-observe action); false-alarm rate on control windows; agreement by fault type and by system; service-fault-pair majority (≥2 of 3 repetitions) for effective-n reporting; Wilson 95% CIs.
- **Split:** the 12 RE2-TT cases used to develop the adapter are reported separately; the other 258 cases (all of RE2-OB and RE2-SS, 78 of RE2-TT) are the held-out set used for primary estimates.
- **Descriptive diagnostic:** for CPU-fault windows, the post-injection CPU value AAF received versus its saturation thresholds (85% / 90%), to explain any "no action".

## Interpretation rules

- Agreement is with this pre-specified proxy oracle, not with expert judgement.
- A low AAF detection rate is reported as a finding about AAF's absolute thresholds on this telemetry, not tuned away. Any threshold or adapter change would be a separate, labelled exploratory analysis.
- No LLM is used in this study.

## Amendments

| ID | Date | Change | Reason |
| --- | --- | --- | --- |
