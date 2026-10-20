# W1 Independent Policy Oracle — working specification (NOT FROZEN)
Date: 2026-10-20. Status: development only; NOT preregistered, no confirmatory cases authored.

## Purpose
Define a reference decision procedure separate from AAF and all comparator outputs. This is not a copy of AAF rules, not a practitioner-validated oracle, and not a claim that public frameworks uniquely prescribe governance decisions.

## Source provenance and policy taxonomy
1. Reliability and release risk: Google SRE error-budget and incident-management principles. Record exact source URL, document version/access date, quoted proposition (briefly), and the study-specific action rule derived from it.
2. Delivery metrics: DORA software delivery performance measures. Metrics are indicators, not automatic release/escalation instructions.
3. Security: NIST SSDF secure-development practices. Operational escalation thresholds require separate declared assumptions.
4. Cost: FinOps Foundation framework for cost accountability. Project-specific thresholds are study assumptions.

No reference policy rule may cite a source generically without a verifiable section-level citation. Until citations and policy-owner review are completed, all rules are *candidates*.

## Canonical action vocabulary (draft)
PROCEED, INVESTIGATE, DEFER, ESCALATE, CONTAIN, REQUEST_EVIDENCE, INDETERMINATE. These are study abstractions; their exact operational meaning and exclusivity must be frozen before cases are authored.

## Rule record schema
rule_id; version; decision_scope; external_source_url; external_source_section; source_principle; source_support_type [direct|inferred|none]; study_assumption; required_observations; missing_evidence_behavior; conflicting_evidence_behavior; authority_constraint; admissible_actions; prohibited_actions; rationale; counterexample; oracle_test_ids; independent_verifier; review_status.

## Adjudication principles
- Do not infer evidence that is absent. Missing decision-critical evidence may require REQUEST_EVIDENCE or INVESTIGATE.
- Explicitly define when DEFER versus ESCALATE versus CONTAIN is required; do not let vague language decide ground truth.
- When multiple actions are permissible, preserve the full admissible set.
- If requirements conflict or fail to determine an action, label INDETERMINATE under a frozen rule; count and report these cases.
- No reference action may be derived by executing B1/B2/B4 or consulting their outputs.
- Implement oracle tests in a separate module; consistency checks cannot establish practitioner validity.
- Include counterexamples and boundary tests for every rule.

## Independence and leakage controls
Keep oracle authorship and implementation separate from B2 map and B4 policy code. Hash and register source documents, policy specification, action schema, tests, and scenario-generation instructions. Scenario authors must not inspect confirmatory B4 outputs. Historical 48 cases are exploratory only and may inform development, not confirmatory outcome claims.

## Immediate blockers before freeze
[ ] Verify and capture exact section-level external guidance citations.
[ ] Write and independently inspect actionable rules and boundary conditions.
[ ] Freeze action vocabulary and admissible-set conflict precedence.
[ ] Confirm rule-to-scenario coverage and indeterminate handling.
[ ] Confirm statistical implementation and prospective sample size choice.
[ ] Verify verbatim six reviewer reports.
[ ] Obtain author approval and immutable registration.
