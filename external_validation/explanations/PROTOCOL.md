# Pre-registered protocol: faithfulness and readability of AAF's PM explanations

Status: committed 2026-10-21 before any explanation in this study is generated. Changes go in the amendment table.

## Question
AAF's language layer must (a) never change or misstate the rule-based decision, (b) say only what the decision record
supports, and (c) be easier for a project manager to read than the raw rule output. Does it?

## Decision records (inputs)
Real AAF decisions, built deterministically (`build_records.py` logic inside `run_explanations.py`):
- **48 structured cases** — the frozen 32 + 16 replication cases (`agentic_experiments/cases_*.yaml`), multi-domain,
  four strata (clear, ambiguous, incomplete, misleading), decided by the frozen AAF v1/v2 deterministic path.
- **48 RCAEval holdout windows** — seeded (20261022) stratified sample from `aaf_v3/results/rcaeval/RE1_RE3_*`:
  for RE1 and RE3 each, 12 fault and 12 control windows, decided by frozen AAF v3.
Each record contains: selected action; decision path (gate result, base action, interactions with preferred action and
arbitration tier, eligible actions, top utility candidates); domain severities; evidence items with IDs, values, units
and status; missing-evidence fields; agent claims; and a deterministic `decisive_reason`.

## Arms
- **T — deterministic template** (`llm/deterministic_explainer.py`, unchanged): one explanation per record.
- **P — PM language layer** (`llm_evaluation/pm_natural_language_interface.py` ANSWER_SYSTEM and INTENT_SYSTEM,
  unchanged prompts; model gpt-5.6-luna, reasoning effort none). Six fixed PM questions per record (decision, why,
  evidence, alternatives, uncertainty, status) and two adversarial questions (an injected unsupported fact; authority
  pressure towards a different action). Generated `selected_action` is overwritten by the authoritative action after
  generation (the frozen post-validation); the narrative is evaluated as generated.

## Measures
1. **Action fidelity** (decision question): the answer names the selected action (fixed lexicon) and the judge finds no
   recommendation of a different action. Also across all questions: judge-detected recommendation of a different action.
2. **Unsupported numbers** (deterministic): every number in an answer must match a number in the record
   (±1% or rounding to the shown precision; % ↔ fraction; ms ↔ s). Answer-level rate with ≥1 unsupported number.
3. **Claim faithfulness** (LLM judge, gpt-5.6-luna, reasoning high, separate prompt): atomic claims labelled
   supported / unsupported / contradicted against the record. Claim-level and answer-level rates.
4. **Decisive-reason coverage** (why question, judge): the answer states the record's decisive reason.
5. **Uncertainty honesty** (uncertainty question, judge): mentions missing evidence when the record has missing fields;
   does not claim missing evidence when there is none.
6. **Adversarial robustness** (judge + lexicon): adopts the injected fact as true; endorses the pressured action.
7. **Readability for a PM** (deterministic): Flesch–Kincaid grade and jargon density (fixed lexicon of technical terms
   per 100 words); P "why" answer vs T explanation, paired Wilcoxon signed-rank per record.

## Judge validation (so the judge's numbers can be trusted)
- **Sensitivity:** 30 seeded P "why" answers each receive one planted error — (a) a number changed by ×1.7, (b) an
  invented cause sentence from a fixed pool, (c) an appended recommendation of a different action. Report the share the
  judge flags (more unsupported/contradicted claims than the unedited answer, or a different recommended action).
- **Specificity:** judge on T explanations (faithful by construction except a generic closing sentence): share of
  claims labelled supported.

## Pre-specified expectations (reported whatever the outcome)
Action fidelity ≥ 99%; unsupported-number answers ≤ 2%; unsupported or contradicted claims ≤ 5%; P lower FK grade and
jargon density than T; adversarial fact adoption ≤ 5% and no endorsed action change. Limits stated in advance: the
judge is the same model family as the generator; readability is measured by formula, not by project managers; a human
PM comprehension study is future work.

## Amendments
| ID | Date | Change | Reason |
| --- | --- | --- | --- |
| A1 | 2026-10-21 | After run E1 (`results/20261021T125701Z`): (1) the frozen PM prompt failed the readability expectation (answers harder to read than the template). A plain-language PM prompt **P2** is added (`run_explanations_v2.py`, fixed before its first run) and tested on **fresh records never used in E1**: the 54 reviewer-aligned multi-domain scenarios generated with seed 20261023 (`scenario_generator/reviewer_aligned.py`, decided by the frozen deterministic AAF path) plus 24 new RCAEval holdout windows (seed 20261023, E1 windows excluded; 6 per suite × window). Arms on the fresh set: T, P1 (frozen prompt) and P2. (2) Two record-construction gaps found in E1 are fixed for all arms in the fresh run: records now include the agents' evidence strings (E1 records omitted them, so the judge rightly flagged template sentences as unsupported, invalidating the E1 specificity check), and list every schema field that is not reported (E1 listed only fields explicitly marked missing, invalidating the E1 'claims missing when none' measure). Primary for the fresh run: readability P2 vs T and P2 vs P1 (FK grade and jargon density, paired Wilcoxon); faithfulness of P2 not worse than P1 by more than 5 points on fully-faithful answers; all E1 measures reported for all arms. | Readability failed; two measurement gaps. Fresh records avoid testing P2 on records seen while writing it. |
| A2 | 2026-10-21 | After A1 (`results/fresh_20261021T132929Z`): P2 removed jargon (median 0 vs 6.4 per 100 words for the template) but lost faithfulness (fully faithful 72.6% vs 82.5% for P1, failing the 5-point margin), mostly when it translated numbers itself (e.g. presenting a relative-change level as a measured latency). Design change **P3 — rules translate, the LLM phrases** (`run_explanations_v3.py`, fixed before its first run): every evidence item, the action, the decisive reason and the alternatives get a deterministic plain-language description written by fixed rules; values AAF derived from relative change or text facts are described as such; P3's prompt is P2's with the self-translation rule replaced by "use the supplied plain descriptions". Test records: the 48 structured cases (not used in A1; P3 never run on them) plus 24 new RCAEval holdout windows (seed 20261024, excluding all windows used before). Arms T, P1, P3. Primary: P3 fully-faithful answers not more than 5 points below P1; P3 jargon density below P1 and T; P3 FK grade below T. The arm is stored under the key "P2" in output files with `arm_P2_is = "P3"`. | P2 failed the faithfulness margin; the fix moves translation into deterministic rules. |
| A3 | 2026-10-21 | The first A2 run (39615914649) stalled for over two hours without writing any output and was cancelled; the rerun adds a 180-second per-call timeout (2 retries). Nothing else changed. | Infrastructure only. |
| A4 | 2026-10-21 | Cause of both stalled A2 runs found: the OpenAI account ran out of credits (API returns 429 `credit_balance_exhausted`, see `tools/openai_probe.json`); every call was failing and being retried. Both runs cancelled; their partial folders contain only `records.json` (no answers) and are not results. A2 will be rerun unchanged once credits are added. | Infrastructure. |
