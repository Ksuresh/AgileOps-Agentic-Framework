"""Study 12 governance revision (designed on Study 9 development data only; frozen before the Study 12 test cases ran).

Two post-arbitration rules are added on top of the unchanged deterministic assessment (hybrid.deterministic_assessment):

R1 release regression: if the deployment changed recently (deploy.recent_release, a measured comparison of the two newest
   ReplicaSet pod specs) and reliability is material or the new pods restart (restart_loops >= 3), select
   "Rollback to stable deployment". Rationale from Study 9: patching or mitigating a faulty release left the regression in place.
R2 security precedence over cost: if security is material (critical CVEs or a policy violation) and the selected action
   addresses only cost or nothing ("Scale adjustment", "Review scaling policy", "No action (observe)"), select
   "Patch or block release". Rationale from Study 9: cost actions left critical CVEs in production.

Everything else (gate, utility, interaction policy, thresholds, trigger) is unchanged.
"""
from __future__ import annotations

import copy
from typing import Any

from evidence.schema import bool_evidence, numeric_evidence
from orchestrator.severity import severity_scores

from agentic_experiments import hybrid as H

COST_ONLY = {"Scale adjustment", "Review scaling policy", "No action (observe)"}


def deterministic_assessment_v12(evidence: dict[str, Any]) -> dict[str, Any]:
    det = H.deterministic_assessment(evidence)
    sev = severity_scores(evidence)
    deploy = evidence.get("deploy", {}) or {}
    rr = bool_evidence(deploy, "recent_release")
    rl = numeric_evidence(deploy, "restart_loops")
    restarting = rl.usable and float(rl.value) >= 3
    action, rule = det["selected_action"], None
    if rr.usable and rr.value and (sev["reliability"] > 0 or restarting):
        action, rule = "Rollback to stable deployment", "R1_release_regression"
    elif sev["security"] > 0 and action in COST_ONLY:
        action, rule = "Patch or block release", "R2_security_over_cost"
    if rule:
        det = copy.deepcopy(det)
        det["v11_action"] = det["selected_action"]
        det["selected_action"] = action
        det["v12_rule"] = rule
        det["governance"] = dict(det["governance"], selected_action=action)
    return det


def run_hybrid_v12(case: dict[str, Any], runtime) -> dict[str, Any]:
    """AAF with the v12 rules: identical selective investigation, final action from deterministic_assessment_v12."""
    original = H.deterministic_assessment
    H.deterministic_assessment = deterministic_assessment_v12
    try:
        return H.run_hybrid(case, runtime)
    finally:
        H.deterministic_assessment = original
