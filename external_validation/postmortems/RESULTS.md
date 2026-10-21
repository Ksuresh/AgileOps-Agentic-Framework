# Results: real-postmortem governance study

Run `20261021T111420Z` (GitHub Actions run 39589896845), protocol `PROTOCOL.md` with amendments A1–A7, all committed
before the run. 40 cases, all 40 re-fetched timelines matched the frozen evidence hashes, 0 failed LLM calls,
5 runs per LLM arm. Model gpt-5.6-luna (reader effort none, arm A effort medium). The answer key is rater 1 only;
rater-2 reliability (A7) is pending.

## Primary outcome: agreement with the responders' first action

| Arm | Correct | % | Wilson 95% |
| --- | --- | --- | --- |
| M0 majority ("Mitigate and monitor") | 19/40 | 47.5 | 32.9–62.5 |
| R regex reader → AAF | 0/40 | 0.0 | 0.0–8.8 |
| **L LLM reader v2 → AAF (primary)** | **1/40** | **2.5** | 0.4–12.9 |
| A AI-only (policy in prompt) | 17/40 | 42.5 | 28.5–57.8 |
| L+ reader + categorical fields → AAF-ext | 7/40 | 17.5 | 8.7–32.0 |

Pre-registered tests (exact McNemar, Holm over the two):
- **L vs M0:** L right where M0 wrong 1, M0 right where L wrong 19; p = 4.0e-5 (Holm 6.1e-5). L is **worse** than the majority baseline.
- **L vs A:** 0 vs 16; p = 3.1e-5 (Holm 6.1e-5). L is **worse** than AI-only.

Secondary: L+ vs L 6 vs 0, p = 0.031 (categorical fields help); A vs M0 10 vs 12, p = 0.83 (AI-only is not
distinguishable from the majority baseline); L+ vs A 0 vs 10, p = 0.002; R vs M0 0 vs 19, p = 3.8e-6.

By answer-key action (M0 / R / L / A / L+): Mitigate 19/19, 0/19, 0/19, 7/19, 2/19; Rollback 0/16, 0, 0, 8/16, 4/16;
Patch 0/3, 0, 1/3, 2/3, 1/3; Scale 0/2 in every arm. Pattern is the same for the danluu (21) and GitLab (19) sources.

## Why: AAF abstains on text-only real evidence

Modal output was "Escalate for evidence/human review" or "No action" in R 40/40, **L 35/40**, L+ 25/40 and A 7/40.
Real timelines written before the first action rarely contain the numbers AAF's materiality thresholds need
(p95 ms, error-rate %, saturation %, restart counts). The LLM reader accepted a numeric/boolean schema field in only a
handful of cases, and checker v2 rejected nothing in L because almost nothing was proposed. The most frequent accepted
field was `deploy.pipeline_failed` (GitLab QA/deploy failures), which routes AAF to "Block release and fix pipeline":
correct for GL-017, wrong for five Rollback/Mitigate cases. This is the limitation stated in advance in PROTOCOL.md
("arm L is expected to under-select Rollback and Scale"), but its size was larger than expected: AAF almost never acts.

The categorical extension (L+) recovered some decisions: the added rule fired in 40 runs and gave Rollback in 4/16
Rollback cases (PM-019, PM-022, PM-034, PM-066), but it also fired wrongly on Mitigate cases with a preceding change
(PM-016, PM-068, partly GL-003).

## Exploratory (not pre-registered): what happens when an arm does commit

| Arm | Committed to an action (modal) | Of those, correct | Wrong committed actions |
| --- | --- | --- | --- |
| L | 5/40 | 1 | 4 (10% of cases) |
| L+ | 15/40 | 7 | 8 (20%) |
| A | 33/40 | 18 | 15 (38%) |

AAF's low accuracy is mostly abstention, not wrong actions: it commits rarely and is wrong on 4 of 40 cases, while
AI-only commits on 33 and is wrong on 15. Neither is good decision support here: AI-only's accuracy is at the
majority-baseline level and AAF mostly hands the case back to a human. Abstention is scored as incorrect by the
pre-registered rule; this table is descriptive only.

## What this supports and does not support

- Does **not** support: AAF (as-is) as a decision-support method when the only evidence is responders' free text.
- Supports: AAF's design requires quantitative telemetry; without it, it escalates rather than guessing (35/40).
- Supports: adding qualitative evidence types (L+) moves AAF in the right direction (p = 0.031) but stays well below
  AI-only and the majority baseline; any future categorical rule needs its own pre-registered evaluation on new cases.
- AI-only with the policy in the prompt was not better than always answering "Mitigate"; on real incidents
  neither arm reproduces responders' first actions well.
- Limits: single rater-1 answer key (rater 2 pending); n = 40 with few Patch/Scale cases; agreement is with what
  responders did, not necessarily what was best; evidence stops at the first action, so it is sparse by design.

## Answer-key reliability (amendment A8) and sensitivity

Independent AI rater (different prompt, blind to rater 1 and to arm outputs, 3 calls per case, run 39594417801):
label agreement **33/40, Cohen's κ = 0.71** (exclude as a category); 33/39 and κ = 0.75 where both include; same
first-action entry 35/39. Disagreements: PM-019 (Rollback→Mitigate), PM-045 (Mitigate→exclude), PM-064 and GL-017
(Patch→Mitigate), GL-008 (Rollback→Mitigate), GL-009 and GL-023 (Rollback→Patch). Three of these (GL-008, GL-017,
GL-023) were already flagged ambiguous by rater 1. Both raters are AI; this shows the rubric is reproducible, not
that humans would agree. The human rater's 9 returned items are in `rater2/returned_round1.csv` (5/9 agreement on
inclusion; GL-052's true first action, a canary drain, is recorded only in the issue description, outside the
notes-only rule).

Conclusions do not depend on the disputed cases:

| Answer key | n | M0 | R | L | A | L+ | L vs M0 | L vs A | A vs M0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rater 1 (primary) | 40 | 19 | 0 | 1 | 17 | 7 | 1 vs 19, p = 4e-5 | 0 vs 16, p = 3e-5 | 10 vs 12, p = 0.83 |
| Agreed cases only | 33 | 18 | 0 | 0 | 13 | 4 | 0 vs 18, p = 8e-6 | 0 vs 13, p = 2e-4 | 7 vs 12, p = 0.36 |
| Rater-2 labels | 39 | 22 | 0 | 0 | 13 | 4 | 0 vs 22, p = 5e-7 | 0 vs 13, p = 2e-4 | 7 vs 16, p = 0.09 |
