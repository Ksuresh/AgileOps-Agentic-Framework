# W1 Blind 40-Case Practitioner Audit — working protocol
Status: DRAFT, NOT RECRUITED, NOT EXECUTED.

## Recruitment: WEEK 1
Recruit one experienced PM/Delivery practitioner and one experienced SRE/DevOps practitioner, neither involved in the AAF design, benchmark authoring, oracle labels or analysis. Auditors must NOT be briefed on study hypotheses or B4 design. Collect professional role/experience band, consent, conflicts, availability and appropriate acknowledgment preferences. No names or personal contact details needed in the public research repository.

## Sampling
After the frozen benchmark is authored, draw 40 cases using a prespecified RNG seed: 10 each CLEAR, AMBIGUOUS, INCOMPLETE, MISLEADING; within INCOMPLETE, 5 mapped-null and 5 nontrivial evidence selection. One case per independent scenario family. Audit is prospective and blind to system outputs and oracle labels.

## Questionnaire for each case
- Scenario ID, context, observations, uncertainties, decision authority, evidence availability.
- Choose one preferred action from frozen vocabulary.
- Choose all admissible actions.
- Confidence: high / medium / low.
- Short rationale; indicate whether information is insufficient or scenario ambiguous.
- No study hypotheses, B4 details, oracle labels or AI outputs shown.

## Analysis
Report between-practitioner exact agreement and Cohen's kappa for primary actions; each practitioner vs oracle agreement; pairwise Jaccard of admissible sets (define treatment of empty sets); disagreement categories: policy ambiguity, missing evidence, role-based interpretation, unclear scenario, policy-oracle discrepancy, other. Disagreement is NOT automatically oracle error. Show confusion matrices and confidence intervals as descriptive where feasible.

## Confirmatory integrity
Never change frozen oracle answers or remove cases from primary analysis based on audit findings. Prespecify any secondary sensitivity analysis on substantively disputed audited cases; disclose that this is only a subset, not a substitute for independent validation of the full benchmark.

## Recruitment fallback and scientific cost
Recruit actively in week 1; record attempts and availability. If unavailable, preregistered contingency may omit audit, but R4/R5 concerns about author-written answer keys remain materially unresolved. The response letter must acknowledge this and not claim independent practitioner validation.

## Execution status
[ ] Auditors recruited and screened
[ ] Consent/COI documented
[ ] Questionnaire piloted for timing
[ ] Sampling seed frozen
[ ] Benchmark frozen
[ ] Auditors complete 40 independent judgments each
[ ] Agreement and disagreement analysis archived
