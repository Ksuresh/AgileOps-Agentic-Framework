# Pre-registered protocol: real-postmortem governance study

Status: **committed before the full candidate pool is screened or any arm is run** (2026-10-21). Only the 20-case feasibility screen (`screening/sample_20.json`) was read beforehand; those 20 are excluded from the study set. Later changes go in the amendment table with reasons.

## Question

When the only evidence is what real responders wrote in their incident timeline before they acted, how often does each arm choose the action the responders actually took first, compared with a majority-class baseline?

The answer key comes from the original responders, not from the authors or recruited experts.

## Sources and data handling

- **Pool:** the community list `danluu/post-mortems` — the 100 documents that passed the keyword probe (≥1 action type, ≥2 evidence types, ≥3 timestamps; `probe_report/danluu_metadata.csv`), minus the 20 screening documents.
- **Extension (only if fewer than 40 cases are included):** a seeded random sample of GitLab's public production incidents (`gitlab-com/gl-infra/production`, label `incident`), using issue notes for the timeline. Same rules.
- **Copyright:** postmortem text is never committed in plain form. The repo holds URLs, fetch hashes, cut positions, labels and short rationales; text is fetched at run time and stored only encrypted.

## Screening (inclusion and exclusion)

Include a document only if **all** hold:
1. It contains a timeline with timestamped or clearly ordered entries.
2. At least one observation of the affected system is recorded **before** the first human response action.
3. The first human response action is stated explicitly and maps to the rubric below.

Exclude: security breaches and compromises; data loss caused by human error (e.g. a deleted database); hardware, power, cooling or facility failures; incidents caused purely by an upstream provider's outage; incidents whose first effective action was automatic (no human decision); documents with no usable timeline (narrative only).

## Cut rule (no hindsight)

The evidence given to every arm is **only** the timeline entries strictly before the first human response action, in their original order. Summaries, root-cause sections, impact statements and anything written after the decision point are never shown. Each case records the cut as the index of the first excluded timeline entry.

## Answer key: action-mapping rubric

| Responders' first action | AAF action |
| --- | --- |
| Revert, roll back, or disable/undo a change deployed shortly before the impact | Rollback to stable deployment |
| Halt or pause deploys/releases, release freeze, ship a hotfix or patch | Patch or block release |
| Add capacity, scale out or up | Scale adjustment |
| Restart, failover, reroute or shed traffic, rate-limit, disable a feature not just deployed, run a repair procedure | Mitigate and monitor |

Two raters apply the rubric independently, blind to all arm outputs: rater 1 is Claude (AI-assisted screening, disclosed), rater 2 is a human author or colleague. Report Cohen's κ for inclusion and for the action label; disagreements are resolved by discussion and logged. The admissible set is the single mapped action.

## Arms

| Arm | Description |
| --- | --- |
| M0 Majority baseline | Always the most frequent action in the final case set |
| R Regex reader → AAF | `text_evidence_pilot/regex_reader.py` with checker v2, frozen AAF rules |
| L LLM reader v2 → AAF (primary) | Frozen reader prompt and checker v2 from the text-evidence pilot, frozen AAF rules |
| A AI-only (B6-style) | One LLM call with the evidence text, the governance action list and AAF's governance policy in the prompt, reasoning enabled; chooses the action |
| L+ LLM reader + categorical fields → AAF-ext (secondary) | As L, plus the categorical fields and one added rule below |

Model for L, L+ and A: gpt-5.6-luna (same as the pilot); A uses reasoning effort "medium". Five runs per case for LLM arms; a case is correct if at least 3 of 5 runs give the answer-key action.

### Secondary extension (L+), fixed now

Readers may additionally fill boolean fields, each with a verbatim quote checked by checker v2:

| Field | Meaning | Mapping into AAF evidence |
| --- | --- | --- |
| `sre.outage_present` | the service was down or unavailable | `availability_pct` = 94 |
| `sre.error_spike` | errors/failures rose sharply | `error_rate_pct` = 12 |
| `sre.latency_spike` | latency/slowness rose sharply | `p95_latency_ms` = 800 |
| `sre.resource_exhaustion` | CPU, memory, disk or connections ran out | `saturation_pct` = 90 |
| `deploy.change_preceded_impact` | a deployment, release or config change started shortly before the impact | used by the added rule |

Added rule (tier "specific causal"): if `deploy.change_preceded_impact` is true and an SRE outcome is material, prefer "Rollback to stable deployment". Nothing else in AAF changes.

## Outcomes

- **Primary:** agreement with the responders' first action, arm L vs M0 and arm L vs A (paired exact McNemar, Wilson 95% CIs).
- **Secondary:** all arms per action type; L+ vs L; rate of "No action" / "Escalate" outputs (insufficient structured evidence); field-level extraction statistics.
- **Known limits stated in advance:** AAF reaches Rollback only via restart-burst evidence and Scale only via joint reliability+cost evidence, so arm L is expected to under-select Rollback and Scale on real timelines. Agreement is with what responders did, not necessarily the best action.

## Amendments

| ID | Date | Change | Reason |
| --- | --- | --- | --- |
| A1 | 2026-10-21 | Operational definitions, fixed before any arm run: (a) "timeline" = `timeline_extract.ordered_entries` output with at least 3 entries (newest-first status pages are reversed deterministically); (b) each included case records a `start` index so that summary paragraphs before the first real timeline entry are never shown; evidence = entries[start:cut]; (c) `timeline_sha256` detects page changes between screening and run time (changed pages are re-checked, not silently used). | The protocol's "timestamped or clearly ordered entries" needed a reproducible definition; reversal and start index prevent wrong ordering and hindsight leakage observed during screening. |
| A2 | 2026-10-21 | Rater-1 screening of the 80-document pool complete (`labels_rater1.csv`): 21 included (Mitigate 10, Rollback 8, Scale 2, Patch 1), 59 excluded with reasons. Since 21 < 40, the pre-registered GitLab extension is triggered. | As specified in "Sources". |

