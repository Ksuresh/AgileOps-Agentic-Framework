from __future__ import annotations

from typing import Any, Dict


# Oracle definitions are specified independently of AAF outputs and agent rules.
# They are derived from the experimental intervention represented by each base
# condition. Evaluation code must not infer an oracle from AAF predictions.
ORACLE_SPECS: Dict[str, Dict[str, Any]] = {
    "TC-01": {"condition": "bad deployment artifact/image", "causal_domain": "DevOps", "affected_domains": ["DevOps", "SRE"], "admissible_actions": ["Rollback to stable deployment"], "basis": "Injected deployment artifact failure with post-release reliability impact."},
    "TC-02": {"condition": "deployment configuration drift", "causal_domain": "DevOps", "affected_domains": ["DevOps", "SRE"], "admissible_actions": ["Rollback to stable deployment"], "basis": "Injected configuration drift and restart instability following deployment."},
    "TC-03": {"condition": "failed release pipeline gate", "causal_domain": "DevOps", "affected_domains": ["DevOps"], "admissible_actions": ["Block release and fix pipeline"], "basis": "Injected pre-release pipeline/artifact gate failure without runtime degradation."},
    "TC-04": {"condition": "post-release restart instability", "causal_domain": "DevOps", "affected_domains": ["DevOps", "SRE", "FinOps"], "admissible_actions": ["Rollback to stable deployment"], "basis": "Injected post-release restart loops with reliability degradation."},
    "TC-05": {"condition": "autoscaling cost spike", "causal_domain": "FinOps", "affected_domains": ["FinOps"], "admissible_actions": ["Scale adjustment"], "basis": "Injected excessive scale-out/cost increase while reliability remains healthy."},
    "TC-06": {"condition": "over-provisioned resources", "causal_domain": "FinOps", "affected_domains": ["FinOps"], "admissible_actions": ["Scale adjustment"], "basis": "Injected resource-request growth and cost increase without reliability need."},
    "TC-07": {"condition": "unused capacity", "causal_domain": "FinOps", "affected_domains": ["FinOps"], "admissible_actions": ["Scale adjustment"], "basis": "Injected capacity/cost excess with low saturation."},
    "TC-08": {"condition": "cost increase after scaling policy", "causal_domain": "FinOps", "affected_domains": ["FinOps", "DevOps"], "admissible_actions": ["Review scaling policy", "Scale adjustment"], "basis": "Injected policy-linked cost increase requiring governance review or adjustment."},
    "TC-09": {"condition": "critical vulnerability", "causal_domain": "DevSecOps", "affected_domains": ["DevSecOps"], "admissible_actions": ["Patch or block release"], "basis": "Injected critical vulnerability before release."},
    "TC-10": {"condition": "policy-as-code violation", "causal_domain": "DevSecOps", "affected_domains": ["DevSecOps", "DevOps"], "admissible_actions": ["Patch or block release"], "basis": "Injected security policy violation at a release gate."},
    "TC-11": {"condition": "IAM drift", "causal_domain": "DevSecOps", "affected_domains": ["DevSecOps"], "admissible_actions": ["Patch or block release"], "basis": "Injected identity/access-control drift."},
    "TC-12": {"condition": "compliance control failure", "causal_domain": "DevSecOps", "affected_domains": ["DevSecOps"], "admissible_actions": ["Patch or block release"], "basis": "Injected compliance/policy control failure."},
    "TC-13": {"condition": "latency degradation", "causal_domain": "SRE", "affected_domains": ["SRE"], "admissible_actions": ["Mitigate and monitor"], "basis": "Injected latency degradation without deployment or security cause."},
    "TC-14": {"condition": "error-rate degradation", "causal_domain": "SRE", "affected_domains": ["SRE"], "admissible_actions": ["Mitigate and monitor"], "basis": "Injected elevated service error rate."},
    "TC-15": {"condition": "resource saturation", "causal_domain": "SRE", "affected_domains": ["SRE", "FinOps"], "admissible_actions": ["Scale adjustment", "Mitigate and monitor"], "basis": "Injected resource saturation and latency pressure requiring capacity action."},
    "TC-16": {"condition": "availability degradation", "causal_domain": "SRE", "affected_domains": ["SRE"], "admissible_actions": ["Mitigate and monitor"], "basis": "Injected availability and error-rate degradation."},
    "TC-17": {"condition": "audit evidence gap", "causal_domain": "DevSecOps", "affected_domains": ["DevSecOps"], "admissible_actions": ["Patch or block release"], "basis": "Injected compliance evidence gap before governance approval."},
    "TC-18": {"condition": "IAM policy mismatch", "causal_domain": "DevSecOps", "affected_domains": ["DevSecOps"], "admissible_actions": ["Patch or block release"], "basis": "Injected IAM/compliance mismatch."},
    "TC-19": {"condition": "release evidence missing", "causal_domain": "DevSecOps", "affected_domains": ["DevSecOps", "DevOps"], "admissible_actions": ["Block release and fix pipeline", "Patch or block release"], "basis": "Injected missing compliance/release evidence; release should not proceed."},
    "TC-20": {"condition": "capacity exhaustion", "causal_domain": "SRE", "affected_domains": ["SRE", "FinOps"], "admissible_actions": ["Scale adjustment", "Mitigate and monitor"], "basis": "Injected saturation/latency pressure requiring capacity intervention."},
    "TC-21": {"condition": "unnecessary scale-out", "causal_domain": "FinOps", "affected_domains": ["FinOps", "SRE"], "admissible_actions": ["Scale adjustment"], "basis": "Injected high replica/cost footprint while reliability remains healthy."},
    "TC-22": {"condition": "high CPU/saturation trend", "causal_domain": "SRE", "affected_domains": ["SRE", "FinOps"], "admissible_actions": ["Scale adjustment", "Mitigate and monitor"], "basis": "Injected saturation and latency pressure."},
    "TC-23": {"condition": "deployment-caused incident", "causal_domain": "DevOps", "affected_domains": ["DevOps", "SRE"], "admissible_actions": ["Rollback to stable deployment"], "basis": "Injected failed deployment followed by production incident signals."},
    "TC-24": {"condition": "cascading service errors", "causal_domain": "SRE", "affected_domains": ["SRE", "FinOps"], "admissible_actions": ["Mitigate and monitor"], "basis": "Injected service error/latency cascade without deployment or security cause."},
    "TC-25": {"condition": "security policy incident", "causal_domain": "DevSecOps", "affected_domains": ["DevSecOps", "SRE"], "admissible_actions": ["Patch or block release"], "basis": "Injected security-policy/IAM incident with operational impact."},
    "TC-26": {"condition": "policy version drift", "causal_domain": "DevSecOps", "affected_domains": ["DevSecOps"], "admissible_actions": ["Patch or block release"], "basis": "Injected policy/compliance drift."},
    "TC-27": {"condition": "missed policy update in pipeline", "causal_domain": "DevOps", "affected_domains": ["DevOps", "DevSecOps"], "admissible_actions": ["Block release and fix pipeline", "Patch or block release"], "basis": "Injected pipeline/configuration failure that prevents policy propagation."},
    "TC-28": {"condition": "budget pressure with stable SLO", "causal_domain": "FinOps", "affected_domains": ["FinOps", "SRE"], "admissible_actions": ["Review scaling policy", "Scale adjustment"], "basis": "Injected cost pressure while reliability remains within target."},
    "TC-29": {"condition": "performance risk requires capacity", "causal_domain": "SRE", "affected_domains": ["SRE", "FinOps"], "admissible_actions": ["Scale adjustment", "Mitigate and monitor"], "basis": "Injected latency/saturation pressure where capacity is the experimental intervention."},
    "TC-30": {"condition": "release-triggered multi-signal anomaly", "causal_domain": "DevOps", "affected_domains": ["DevOps", "SRE", "FinOps"], "admissible_actions": ["Rollback to stable deployment"], "basis": "Injected release/configuration change followed by reliability and cost symptoms."},
}


def get_oracle(case_id: str) -> Dict[str, Any]:
    if case_id not in ORACLE_SPECS:
        raise KeyError(f"No pre-specified oracle for {case_id}")
    return ORACLE_SPECS[case_id].copy()
