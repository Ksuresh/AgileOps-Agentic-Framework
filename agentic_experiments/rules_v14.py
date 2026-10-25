"""Study 14 governance revision (designed on Studies 9, 12 and 13 as development data; frozen before Study 14 ran).

Compared with v12 (rules_v12.py), R1 now requires that the release made reliability worse than it was before the release:
R1' release regression: deploy.recent_release and the release degraded reliability relative to the pre-release baseline
    (p95 >= 450 ms and p95 >= 2 x pre-release p95, or error rate >= 8% and >= pre-release error rate + 5 points) or the new
    pods restart (restart_loops >= 3) -> "Rollback to stable deployment". A fault that was already present before the
    release (baseline already degraded) therefore does not trigger a rollback.
R2  security over cost (unchanged from v12): security material and the selected action is Scale adjustment, Review scaling
    policy or No action -> "Patch or block release". Non-critical findings are not material (unchanged), so a release with
    only HIGH/MEDIUM findings and healthy other domains continues.
Pre-release baselines are measured evidence fields sre.p95_pre_release_ms and sre.error_rate_pre_release_pct. If they are
missing, R1' falls back to v12's R1 condition. Development check (Studies 9/12/13, O1): 32/32, 26/26, 5/5.
"""
from __future__ import annotations

import copy
from typing import Any

from evidence.schema import bool_evidence, numeric_evidence
from orchestrator.severity import severity_scores

from agentic_experiments import hybrid as H

_BASE_ASSESSMENT = H.deterministic_assessment
COST_ONLY = {"Scale adjustment", "Review scaling policy", "No action (observe)"}


def _num(block: dict, field: str) -> float | None:
    ev = numeric_evidence(block, field)
    return float(ev.value) if ev.usable else None


def deterministic_assessment_v14(evidence: dict[str, Any]) -> dict[str, Any]:
    det = _BASE_ASSESSMENT(evidence)
    sev = severity_scores(evidence)
    deploy, sre = evidence.get("deploy", {}) or {}, evidence.get("sre", {}) or {}
    rr = bool_evidence(deploy, "recent_release")
    restarts = _num(deploy, "restart_loops") or 0.0
    p95, err = _num(sre, "p95_latency_ms"), _num(sre, "error_rate_pct")
    p95_0, err_0 = _num(sre, "p95_pre_release_ms"), _num(sre, "error_rate_pre_release_pct")
    if p95_0 is None and err_0 is None:
        regression = sev["reliability"] > 0
    else:
        regression = bool((p95 is not None and p95_0 is not None and p95 >= 450 and p95 >= 2 * p95_0)
                          or (err is not None and err_0 is not None and err >= 8 and err >= err_0 + 5))
    action, rule = det["selected_action"], None
    if rr.usable and rr.value and (regression or restarts >= 3):
        action, rule = "Rollback to stable deployment", "R1p_release_regression"
    elif sev["security"] > 0 and action in COST_ONLY:
        action, rule = "Patch or block release", "R2_security_over_cost"
    if rule:
        det = copy.deepcopy(det)
        det["v11_action"] = det["selected_action"]
        det["selected_action"] = action
        det["v14_rule"] = rule
        det["governance"] = dict(det["governance"], selected_action=action)
    return det


def run_hybrid_v14(case: dict[str, Any], runtime) -> dict[str, Any]:
    original = H.deterministic_assessment
    H.deterministic_assessment = deterministic_assessment_v14
    try:
        return H.run_hybrid(case, runtime)
    finally:
        H.deterministic_assessment = original
