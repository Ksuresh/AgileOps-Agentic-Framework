from __future__ import annotations

"""Frozen oracles for the reviewer-aligned controlled benchmark.

These specifications are independent of AAF predictions. Existing TC-01..TC-30
oracles are imported unchanged; TC-31..TC-54 are frozen before benchmark execution.
"""

from typing import Any, Dict

from benchmark.oracle_specs import ORACLE_SPECS as ORIGINAL_ORACLE_SPECS

ADDITIONAL_ORACLE_SPECS: Dict[str, Dict[str, Any]] = {
    "TC-31": {"condition": "healthy release", "causal_domain": None, "affected_domains": [], "admissible_actions": ["No action (observe)"], "basis": "All four operational domains remain within predefined materiality bounds."},
    "TC-32": {"condition": "benign CPU fluctuation", "causal_domain": None, "affected_domains": [], "admissible_actions": ["No action (observe)"], "basis": "CPU/saturation variation remains below the SRE materiality threshold."},
    "TC-33": {"condition": "cost increase within policy", "causal_domain": None, "affected_domains": [], "admissible_actions": ["No action (observe)"], "basis": "Cost increase remains below FinOps materiality with stable service health."},
    "TC-34": {"condition": "non-blocking security context", "causal_domain": None, "affected_domains": [], "admissible_actions": ["No action (observe)"], "basis": "No critical vulnerability or policy/compliance breach is present."},
    "TC-35": {"condition": "transient latency below threshold", "causal_domain": None, "affected_domains": [], "admissible_actions": ["No action (observe)"], "basis": "Latency variation remains below the predefined reliability intervention threshold."},
    "TC-36": {"condition": "clean rollback marker without degradation", "causal_domain": None, "affected_domains": [], "admissible_actions": ["No action (observe)"], "basis": "A rollback marker alone is historical context; no active material operational signal is present."},

    "TC-37": {"condition": "deployment indication without release evidence", "causal_domain": None, "affected_domains": ["DevOps"], "admissible_actions": ["Escalate for evidence/human review"], "basis": "All deployment evidence sources required for attribution are explicitly unavailable."},
    "TC-38": {"condition": "reliability indication without SRE sources", "causal_domain": None, "affected_domains": ["SRE"], "admissible_actions": ["Escalate for evidence/human review"], "basis": "Latency, error, saturation and availability evidence are explicitly unavailable."},
    "TC-39": {"condition": "cost indication without FinOps provenance", "causal_domain": None, "affected_domains": ["FinOps"], "admissible_actions": ["Escalate for evidence/human review"], "basis": "Billing/resource-history evidence is unavailable; a FinOps intervention is not evidence-supported."},
    "TC-40": {"condition": "security indication without scanner provenance", "causal_domain": None, "affected_domains": ["DevSecOps"], "admissible_actions": ["Escalate for evidence/human review"], "basis": "Security/policy evidence sources are unavailable, so blocking action cannot be justified solely from an unsupported alert."},
    "TC-41": {"condition": "compound incident with deployment evidence suppressed", "causal_domain": "SRE", "affected_domains": ["SRE"], "admissible_actions": ["Mitigate and monitor"], "basis": "Direct SRE degradation remains observable while deployment causality is unavailable and must not be invented."},
    "TC-42": {"condition": "reliability incident with cost evidence suppressed", "causal_domain": "SRE", "affected_domains": ["SRE"], "admissible_actions": ["Mitigate and monitor", "Scale adjustment"], "basis": "Reliability degradation remains independently evidenced; no FinOps causal claim is admissible without cost evidence."},
    "TC-43": {"condition": "conflicting duplicated availability sources", "causal_domain": None, "affected_domains": ["SRE"], "admissible_actions": ["Escalate for evidence/human review"], "basis": "Two explicitly represented sources disagree materially; autonomous intervention should be withheld pending evidence reconciliation."},
    "TC-44": {"condition": "partial noisy SRE-FinOps evidence", "causal_domain": "SRE", "affected_domains": ["SRE", "FinOps"], "admissible_actions": ["Scale adjustment", "Mitigate and monitor"], "basis": "Despite one suppressed metric, remaining provenance-backed saturation/latency and FinOps context are sufficient for a reliability-led capacity response."},

    "TC-45": {"condition": "high latency under elevated cost", "causal_domain": "SRE", "affected_domains": ["SRE", "FinOps"], "admissible_actions": ["Scale adjustment", "Mitigate and monitor"], "basis": "Reliability is materially degraded; cost pressure constrains but does not negate the need for service recovery."},
    "TC-46": {"condition": "deployment fault with severe reliability degradation", "causal_domain": "DevOps", "affected_domains": ["DevOps", "SRE"], "admissible_actions": ["Rollback to stable deployment"], "basis": "A directly evidenced deployment/configuration fault precedes severe SRE degradation, making rollback the release-governance action."},
    "TC-47": {"condition": "critical security blocker with healthy runtime", "causal_domain": "DevSecOps", "affected_domains": ["DevSecOps", "SRE"], "admissible_actions": ["Patch or block release"], "basis": "Critical security evidence is release-blocking even when reliability is healthy."},
    "TC-48": {"condition": "severe outage with no blocking security evidence", "causal_domain": "SRE", "affected_domains": ["SRE", "DevSecOps"], "admissible_actions": ["Mitigate and monitor"], "basis": "Service restoration is the material governance concern because no security blocker is present."},
    "TC-49": {"condition": "healthy release with excessive scale", "causal_domain": "FinOps", "affected_domains": ["FinOps", "SRE", "DevOps"], "admissible_actions": ["Review scaling policy", "Scale adjustment"], "basis": "Runtime and deployment are healthy; excessive resource footprint is the material governance issue."},
    "TC-50": {"condition": "configuration drift with cost spike", "causal_domain": "DevOps", "affected_domains": ["DevOps", "FinOps"], "admissible_actions": ["Rollback to stable deployment", "Review scaling policy"], "basis": "Configuration drift is directly evidenced and causally prior to the FinOps symptom."},
    "TC-51": {"condition": "capacity pressure under budget constraint", "causal_domain": "SRE", "affected_domains": ["SRE", "FinOps"], "admissible_actions": ["Scale adjustment", "Mitigate and monitor"], "basis": "Reliability pressure is material; budget constraint must be reflected in the capacity action rather than suppressing it."},
    "TC-52": {"condition": "simultaneous security and pipeline blockers", "causal_domain": "DevSecOps", "affected_domains": ["DevSecOps", "DevOps"], "admissible_actions": ["Patch or block release", "Block release and fix pipeline"], "basis": "Both release gates are blocking; either explicit blocking remediation action is admissible, with security treated as the primary governance domain."},
    "TC-53": {"condition": "rollback reduces reliability risk at temporary cost", "causal_domain": "DevOps", "affected_domains": ["DevOps", "SRE", "FinOps"], "admissible_actions": ["Rollback to stable deployment"], "basis": "The directly evidenced deployment/configuration issue plus reliability impact justifies rollback despite temporary cost consequences."},
    "TC-54": {"condition": "multi-domain mild warnings below materiality", "causal_domain": None, "affected_domains": ["SRE", "FinOps"], "admissible_actions": ["No action (observe)"], "basis": "Multiple weak signals remain individually and jointly below predefined materiality; coordination must not manufacture severity."},
}

REVIEWER_ORACLE_SPECS: Dict[str, Dict[str, Any]] = {
    **ORIGINAL_ORACLE_SPECS,
    **ADDITIONAL_ORACLE_SPECS,
}


def get_reviewer_oracle(case_id: str) -> Dict[str, Any]:
    if case_id not in REVIEWER_ORACLE_SPECS:
        raise KeyError(f"No frozen reviewer-aligned oracle for {case_id}")
    return REVIEWER_ORACLE_SPECS[case_id].copy()
