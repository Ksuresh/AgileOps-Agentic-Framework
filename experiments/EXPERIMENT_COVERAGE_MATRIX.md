# AAF Experiment Coverage Matrix

## Purpose

This audit preserves the existing 30 controlled scenarios as the base benchmark and classifies them against the paper's actual governance claims. The objective is not to replace the existing experiment set, but to identify the minimum additional cases needed to test cross-domain release governance, conflicting evidence, safe abstention/escalation, and PM-facing explanation.

## Locked experiment groups

- **G1 Single-domain control**: one operational domain is materially responsible; AAF should be comparable to simpler baselines.
- **G2 Cross-domain/compound**: two or more domains contain material evidence relevant to the release/governance decision; this is the primary efficacy group.
- **G3 Conflicting-governance evidence**: domains support competing actions or priorities; tests coordination and governance trade-off handling.
- **G4 Incomplete/ambiguous evidence**: evidence is missing, suppressed, noisy, or insufficient; tests RER/escalation and unsupported-action avoidance.
- **G5 Healthy/no-action**: no material anomaly; tests false intervention and abstention.
- **G6 PM explanation/interaction**: PM-level questions over evidence-grounded decisions; evaluated separately from decision accuracy.

## Audit of existing TC-01 to TC-30

| ID | Existing scenario | Primary | Existing secondary evidence | Group | Keep? | Audit note |
|---|---|---|---|---|---|---|
| TC-01 | bad_image_tag | DevOps | SRE | G2 | Yes | Release fault with reliability impact; useful DevOps->SRE causal chain. |
| TC-02 | configuration_drift | DevOps | SRE | G2 | Yes | Strong release-governance case. |
| TC-03 | failed_pipeline_gate | DevOps | none | G1 | Yes | Clean single-domain control. |
| TC-04 | container_restart_after_release | DevOps | SRE, FinOps | G2 | Yes | Strong 3-domain compound case. |
| TC-05 | autoscaling_cost_spike | FinOps | SRE stable | G1 | Yes | FinOps control with SRE context but no competing material SRE signal. |
| TC-06 | over_provisioned_resources | FinOps | none | G1 | Yes | Clean FinOps control. |
| TC-07 | unused_capacity | FinOps | SRE stable | G1 | Yes | FinOps control. |
| TC-08 | cost_increase_after_scaling_policy | FinOps | DevOps | G2 | Yes | FinOps + configuration-governance interaction. |
| TC-09 | critical_vulnerability | DevSecOps | none | G1 | Yes | Clean security control. |
| TC-10 | policy_as_code_violation | DevSecOps | DevOps | G2 | Yes | Security + pipeline gate. |
| TC-11 | iam_drift_detected | DevSecOps | none | G1 | Yes | Clean security control. |
| TC-12 | compliance_control_failure | DevSecOps | none | G1 | Yes | Clean compliance control. |
| TC-13 | latency_spike | SRE | none | G1 | Yes | Clean SRE control. |
| TC-14 | error_rate_spike | SRE | none | G1 | Yes | Clean SRE control. |
| TC-15 | resource_saturation | SRE | FinOps | G2 | Yes | Reliability-capacity-cost interaction. |
| TC-16 | availability_drop | SRE | none | G1 | Yes | Clean SRE control. |
| TC-17 | audit_log_gap | DevSecOps | none | G1 | Yes | Compliance control. |
| TC-18 | iam_policy_mismatch | DevSecOps | none | G1 | Yes | Compliance/security control. |
| TC-19 | release_evidence_missing | DevSecOps | DevOps | G2/G4 candidate | Yes | Name implies missing evidence, but current telemetry encodes violations rather than true provenance-level evidence absence; retain but do not count as G4 until evidence suppression is explicit. |
| TC-20 | capacity_exhaustion | SRE | FinOps | G2 | Yes | Strong SRE-FinOps case. |
| TC-21 | unnecessary_scale_out | FinOps | SRE stable | G2 | Yes | Cross-domain context; useful opposite-direction control to TC-20. |
| TC-22 | high_cpu_trend | SRE | FinOps | G2 | Yes | Reliability/cost interaction. |
| TC-23 | failed_deployment_caused_incident | DevOps | SRE | G2 | Yes | Strong causal release incident. |
| TC-24 | cascading_service_errors | SRE | FinOps | G2 | Yes | Reliability with cost consequence. |
| TC-25 | security_policy_triggered_incident | DevSecOps | SRE | G2 | Yes | Security/reliability compound condition. |
| TC-26 | policy_version_drift | DevSecOps | none | G1 | Yes | Security/policy control. |
| TC-27 | missed_policy_update_in_pipeline | DevOps | DevSecOps | G2 | Yes | Strong pipeline-security governance case. |
| TC-28 | budget_pressure_with_stable_slo | FinOps | SRE stable | G3 candidate | Yes | Existing trade-off case; should be explicitly tagged as governance conflict/context rather than generic cross-domain. |
| TC-29 | performance_risk_requires_capacity | SRE | FinOps | G3 candidate | Yes | Opposing performance vs cost pressure; strong governance trade-off case. |
| TC-30 | release_change_triggered_multi_signal_anomaly | DevOps | SRE, FinOps | G2 | Yes | Strong 3-domain release-governance case. |

## Existing coverage summary

Approximate coverage before adding any new controlled cases:

- **G1 single-domain controls:** 13
- **G2 cross-domain/compound:** 15
- **G3 explicit conflicting-governance cases:** 2 credible candidates (TC-28, TC-29), but they need explicit conflict labels and frozen admissible-action rationale
- **G4 incomplete/ambiguous evidence:** effectively 0 explicit base cases; current random `_missing` flag is not sufficient as a dedicated benchmark because the base scenario truth remains unchanged and the evidence absence is not independently stratified
- **G5 healthy/no-action:** 0 controlled base cases
- **G6 PM interaction:** existing PM prompt and LLM-evaluation assets exist, but are not currently integrated into the main reviewer-aligned benchmark

The existing 30 cases therefore already provide substantial single-domain and cross-domain coverage. The main gaps are **explicit conflict, evidence insufficiency, healthy abstention, and PM release-governance interrogation**.

## Minimum additional controlled cases

The following additions are sufficient to balance the benchmark without reinventing the scenario suite.

### A. Healthy/no-action controls (6 new cases)

| Proposed ID | Case | Expected governance behavior |
|---|---|---|
| TC-31 | healthy release, all four domains normal | No action (observe) |
| TC-32 | successful deployment with benign CPU fluctuation | No action (observe) |
| TC-33 | modest cost increase within policy and SLO stable | No action (observe) |
| TC-34 | low-severity CVE below release-blocking policy | No action (observe) or policy-compliant observe, frozen before run |
| TC-35 | transient latency below materiality threshold | No action (observe) |
| TC-36 | clean rollback marker with no active degradation | No action (observe) |

### B. Explicit incomplete/ambiguous evidence cases (8 new cases)

| Proposed ID | Case | Expected governance behavior |
|---|---|---|
| TC-37 | deployment anomaly but release metadata unavailable | Escalate for evidence/human review |
| TC-38 | SRE degradation indication but latency/error sources unavailable | Escalate unless remaining provenance-backed evidence is independently sufficient |
| TC-39 | cost anomaly proxy without billing/resource-history evidence | Escalate or qualified FinOps recommendation according to frozen evidence rule |
| TC-40 | security alert without scanner provenance | Escalate; do not block solely on unsupported claim |
| TC-41 | DevOps + SRE compound with deployment evidence suppressed | Test whether remaining SRE evidence changes action without fabricated root cause |
| TC-42 | SRE + FinOps compound with cost evidence suppressed | Reliability decision may continue; FinOps claim must abstain |
| TC-43 | contradictory duplicated metric sources | Escalate/flag evidence conflict |
| TC-44 | 25% evidence drop + 10% numeric jitter | Robustness/noise case aligned with runtime RT-20 |

### C. Explicit cross-domain governance conflicts (10 new cases)

| Proposed ID | Competing evidence | Governance question |
|---|---|---|
| TC-45 | SRE latency high + FinOps cost already high | Scale temporarily vs contain cost? |
| TC-46 | DevOps deployment fault + SRE severe degradation | Roll back vs mitigate in place? |
| TC-47 | DevSecOps critical release blocker + SRE healthy | Block despite healthy service? |
| TC-48 | DevSecOps warning below blocker + SRE severe outage | Restore service first vs security remediation? |
| TC-49 | DevOps healthy + FinOps excessive scale + SRE healthy | Continue release but correct scaling policy? |
| TC-50 | DevOps config drift + FinOps spike + SRE healthy | Roll back config vs cost-only adjustment? |
| TC-51 | SRE saturation + FinOps budget cap | Capacity vs budget trade-off. |
| TC-52 | security policy violation + failed pipeline | Which blocker should drive PM release decision? |
| TC-53 | deployment rollback reduces reliability risk but increases temporary cost | Rollback vs cost acceptance. |
| TC-54 | multi-domain mild warnings, none individually material | Observe vs over-intervene; coordination must not manufacture severity. |

This brings the controlled base set to **54 cases**, while preserving every existing scenario. With 5 frozen perturbation variants per case, the main controlled benchmark becomes **270 evaluations**.

## Runtime experiment mapping

The existing RT-01 to RT-20 catalogue already covers the core reviewer-aligned groups:

- **Healthy:** RT-01
- **Single-domain SRE:** RT-02, RT-03, RT-07, RT-08, RT-11
- **Single-domain DevOps:** RT-05, RT-06
- **Single-domain FinOps:** RT-09, RT-10
- **Single-domain DevSecOps:** RT-14, RT-15, RT-16
- **Cross-domain/compound:** RT-04, RT-12, RT-13, RT-17, RT-18
- **Incomplete/noisy evidence:** RT-19, RT-20

The runtime design therefore does **not** need a new catalogue. It needs execution support, realistic instrumentation, and increased repetitions.

### Proposed Sock Shop execution target

- Execute every RT case that can be implemented without fabricating evidence.
- Minimum **5 repetitions** per executable intervention.
- Preserve raw logs, process state, resource metrics, deployment/config evidence, and scanner/policy evidence where genuinely collected.
- Report unsupported RT cases explicitly rather than simulating unavailable modalities.

Expected target: roughly **70-100 real runtime trials**, depending on which RT interventions are technically executable with authentic evidence collection.

## Second runtime workload

Use **Google Online Boutique** as a confirmatory workload, not as a replacement for Sock Shop. Re-run a smaller frozen subset corresponding to:

1. healthy baseline
2. service stop / availability loss
3. CPU or resource pressure
4. excessive scaling / resource footprint
5. high request load
6. SRE + FinOps compound
7. deployment + reliability compound
8. incomplete evidence variant

Target: 8 conditions x 5 repetitions = **40 confirmatory runtime trials**.

## Public operational dataset validation

A public labeled AIOps dataset should be used as an **external operational-data validation study**, with a strict scope rule: only evaluate domains actually represented by the dataset. Do not synthesize FinOps/DevSecOps fields merely to fit AAF.

Candidate priority:

1. AIOps Challenge 2020 or another public labeled distributed-system incident dataset with accessible metrics/logs/traces and failure labels.
2. GAIA/MicroSS as secondary operational benchmark if its available modalities map cleanly to AAF evidence.

This study validates ingestion, evidence attribution, SRE/operational reasoning, and grounded explanation on externally produced data; it is not used to claim four-domain coverage if the dataset lacks those domains.

## PM-facing explanation experiment

Use the existing PM prompt and LLM explanation assets rather than building a new interface. For each selected governance case, compare three information conditions:

1. raw telemetry/alerts
2. structured AAF governance output without LLM explanation
3. evidence-grounded natural-language AAF explanation

Evaluate:

- answer correctness to PM governance question
- evidence coverage/traceability
- unsupported-claim rate
- domain/action consistency
- concise decision rationale completeness
- optional human comprehension/time-to-decision only if a defensible evaluator protocol is available

Core PM questions should include:

- Can I proceed with the release?
- What is blocking the release?
- Why is rollback/mitigation/scale adjustment recommended?
- Which domain is driving the decision?
- What evidence supports the recommendation?
- What evidence is missing?

## Frozen comparison set

Primary comparisons:

1. AAF full
2. dominant-domain deterministic baseline
3. isolated-domain decision baseline
4. AAF without consensus
5. AAF without RER
6. AAF without utility/action coordination
7. LLM-only/single-agent baseline only where implemented with a reproducible fixed model/prompt configuration

## Primary metrics

Report results **stratified by G1-G5**, not only aggregate accuracy.

- domain attribution accuracy / macro-F1
- admissible-action accuracy
- false intervention rate on healthy controls
- unsafe abstention rate on material incidents
- correct escalation rate under insufficient evidence
- unsupported-decision rate
- evidence provenance/coverage
- cross-domain gain over dominant-domain and isolated-domain baselines
- paired McNemar tests for action correctness where appropriate
- Wilson confidence intervals for proportions
- effect sizes and bootstrap confidence intervals where suitable

## Decision rule before execution

The benchmark specification, scenario group, evidence sources, oracle/admissible actions, thresholds, and comparison baselines must be frozen before viewing final experiment outcomes. Implementation bugs may be fixed transparently, but scenario oracles and thresholds must not be changed post hoc to improve AAF scores.
