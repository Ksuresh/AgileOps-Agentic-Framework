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
| A3 | 2026-10-21 | GitLab extension specification, fixed before fetching: population = closed issues in `gitlab-com/gl-infra/production` with labels `incident` and `severity::1` or `severity::2`, created 2020-01-01 to 2025-12-31; order = shuffled with seed 20261021; screen in that order until 40 cases are included in total or 150 GitLab incidents have been screened. Timeline = the issue's human (non-system) notes in creation order, each rendered as "YYYY-MM-DD HH:MM UTC: <note text>"; the issue title and description are never used (descriptions are often completed after the incident). Same inclusion, cut and rubric rules; same encrypted handling. | Triggered by A2. Notes are real-time responder communication, which avoids hindsight better than the description. |
| A4 | 2026-10-21 | GitLab notes are fetched through the public GraphQL API instead of the REST notes endpoint. Population, seed, sample and rendering unchanged. | The REST notes endpoint now returns 401 without authentication (first fetch returned zero notes for all 150 incidents; no content was read). |
| A5 | 2026-10-21 | Rater-1 screening of the GitLab extension complete (`labels_rater1_gitlab.csv`): GL-001 to GL-057 screened in seeded order, 19 included (Mitigate 9, Rollback 8, Patch 2), 38 excluded with reasons; screening stopped at 40 included in total, as specified in A3. Final study set: 40 cases (Mitigate 19, Rollback 16, Patch 3, Scale 2); M0 therefore always answers "Mitigate and monitor". Labels committed before any arm is run. | As specified in A3. |
| A6 | 2026-10-21 | Implementation details fixed in `run_study.py` before any arm run: (a) the rubric row "halt deploys / hotfix" is admissible as either of AAF's two release-halting labels, "Patch or block release" or "Block release and fix pipeline" (all other rows: the single mapped action); (b) the evidence is one document `TIMELINE` = entries[start:cut] joined by newlines, scope = "the service or system affected by the incident" with no environment, starting from all 16 schema fields missing; (c) frozen `evidence_hashes.json` (SHA-256 of each case's evidence, hashes only) is checked at run time; a case whose re-fetched evidence differs is excluded from analysis and listed, not silently used; (d) L+: categorical fields are mapped only into SRE fields the reader did not fill numerically, and the added Rollback rule yields to a hard-override interaction; (e) arm A prompt (action list + AAF policy in prose) is fixed in code and its SHA-256 recorded; an answer outside the action list counts as incorrect; (f) raw run logs contain verbatim quotes and are committed only encrypted. An offline mock run (no LLM, regex output as stand-in) checked the pipeline; its numbers are not results. | The protocol did not fix these mechanics; (a) because AAF's action set has two labels for halting a release. |
| A7 | 2026-10-21 | Rater-2 reliability is assessed on a blinded sample instead of all 137 screened documents: all 40 included cases plus 40 randomly drawn excluded documents (seed 20261022), shuffled (`rater2/rater2_sheet.csv`; the item-to-case key is held back, its SHA-256 is in `rater2/key_sha256.txt`). Cohen's kappa for inclusion on the 80 and for the action label on the documents both raters include. Rater 2 records the first-action quote rather than entry indices. | Keeps the human workload to about two hours while covering every case in the study set. |
| A8 | 2026-10-21 | No human second rater is available (the human rater returned 9 of 80 items and then declined). Answer-key reliability is instead checked by an independent AI rater (`rater2_ai.py`): a different prompt from rater 1, blind to rater-1 labels and to all arm outputs, shown the timeline window entries[start:cut+4] of each of the 40 included cases, 3 calls per case (majority), reasoning effort high. Report label agreement and Cohen's kappa (exclude as its own category), first-action-entry agreement, and the confusion table. Both raters are AI and this is disclosed; it measures rubric reproducibility, not human agreement. Inclusion reliability on excluded documents is not assessed. The human rater's 9 returned items (`rater2/returned_round1.csv`) are reported descriptively; one (GL-052) shows an action recorded only in the issue description, outside the notes-only evidence rule (A3), and is reported as a limitation. | Replaces the human-rater plan in A7. |
