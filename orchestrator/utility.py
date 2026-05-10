from __future__ import annotations

from typing import Any, Dict, Tuple


def utility_score(
    performance_score: float,
    cost_efficiency_score: float,
    risk_reduction_score: float,
    w: Tuple[float, float, float],
) -> float:
    """
    Composite utility:
        U(a) = w_perf * performance_score
             + w_cost * cost_efficiency_score
             + w_risk * risk_reduction_score

    All components are normalized to [0, 1], where higher is better.
    """
    w_perf, w_cost, w_risk = w
    return (
        w_perf * performance_score
        + w_cost * cost_efficiency_score
        + w_risk * risk_reduction_score
    )


def _severity_scores(telemetry: Dict[str, Any]) -> Dict[str, float]:
    deploy = telemetry.get("deploy", {}) or {}
    sre = telemetry.get("sre", {}) or {}
    finops = telemetry.get("finops", {}) or {}
    sec = telemetry.get("sec", {}) or {}

    restart_loops = int(float(deploy.get("restart_loops", 0) or 0))
    pipeline_failed = bool(deploy.get("pipeline_failed", False))
    config_drift = bool(deploy.get("config_drift", False))
    rollback_marker = bool(deploy.get("rollback_marker", False))
    artifact_mismatch = bool(deploy.get("artifact_mismatch", False))

    p95 = float(sre.get("p95_latency_ms", 0.0) or 0.0)
    err = float(sre.get("error_rate_pct", 0.0) or 0.0)
    sat = float(sre.get("saturation_pct", 0.0) or 0.0)
    availability = float(sre.get("availability_pct", 99.9) or 99.9)

    cost_spike = float(finops.get("cost_spike_pct", 0.0) or 0.0)
    hpa_scale = int(float(finops.get("hpa_scale_to", 0) or 0))
    cpu_inc = float(finops.get("cpu_request_increase_pct", 0.0) or 0.0)
    mem_inc = float(finops.get("memory_request_increase_pct", 0.0) or 0.0)

    critical_cves = int(float(sec.get("critical_cves", 0) or 0))
    policy_violation = bool(sec.get("policy_violation", False))
    iam_drift = bool(sec.get("iam_drift", False))
    compliance_gap = bool(sec.get("compliance_gap", False))

    deployment = 0.0
    if pipeline_failed:
        deployment += 0.35
    if config_drift:
        deployment += 0.25
    if rollback_marker:
        deployment += 0.25
    if artifact_mismatch:
        deployment += 0.25
    if restart_loops >= 12:
        deployment += 0.25
    elif restart_loops >= 6:
        deployment += 0.10

    reliability = 0.0
    if p95 >= 800:
        reliability += 0.35
    elif p95 >= 450:
        reliability += 0.25
    if err >= 12:
        reliability += 0.35
    elif err >= 8:
        reliability += 0.25
    if sat >= 90:
        reliability += 0.25
    elif sat >= 85:
        reliability += 0.15
    if availability < 99.0:
        reliability += 0.25

    cost = 0.0
    if cost_spike >= 35:
        cost += 0.40
    elif cost_spike >= 22:
        cost += 0.30
    elif cost_spike >= 8:
        cost += 0.15
    if hpa_scale >= 14:
        cost += 0.25
    elif hpa_scale >= 11:
        cost += 0.15
    if cpu_inc >= 50:
        cost += 0.20
    if mem_inc >= 40:
        cost += 0.20

    security = 0.0
    if critical_cves >= 2:
        security += 0.40
    elif critical_cves == 1:
        security += 0.30
    if policy_violation:
        security += 0.25
    if iam_drift:
        security += 0.20
    if compliance_gap:
        security += 0.20

    return {
        "deployment": min(1.0, deployment),
        "reliability": min(1.0, reliability),
        "cost": min(1.0, cost),
        "security": min(1.0, security),
    }


def _action_components(telemetry: Dict[str, Any]) -> Dict[str, Tuple[float, float, float]]:
    """
    Returns:
        action -> (performance_score, cost_efficiency_score, risk_reduction_score)

    Higher is better for all three components.
    """
    s = _severity_scores(telemetry)

    deployment = s["deployment"]
    reliability = s["reliability"]
    cost = s["cost"]
    security = s["security"]

    return {
        "Rollback to stable deployment": (
            min(1.0, 0.35 + 0.40 * deployment + 0.25 * reliability),
            0.65,
            min(1.0, 0.35 + 0.35 * deployment + 0.10 * security),
        ),
        "Block release and fix pipeline": (
            min(1.0, 0.25 + 0.45 * deployment),
            0.75,
            min(1.0, 0.40 + 0.30 * deployment + 0.20 * security),
        ),
        "Mitigate and monitor": (
            min(1.0, 0.30 + 0.35 * reliability),
            0.68,
            min(1.0, 0.30 + 0.20 * reliability),
        ),
        "Scale adjustment": (
            min(1.0, 0.32 + 0.38 * reliability + 0.28 * cost),
            min(1.0, 0.48 + 0.40 * cost),
            min(1.0, 0.32 + 0.25 * reliability),
        ),
        "Review scaling policy": (
            min(1.0, 0.22 + 0.25 * cost),
            min(1.0, 0.60 + 0.35 * cost),
            min(1.0, 0.32 + 0.18 * cost),
        ),
        "Patch or block release": (
            min(1.0, 0.22 + 0.25 * security),
            0.70,
            min(1.0, 0.48 + 0.48 * security),
        ),
        "No action (observe)": (
            0.20,
            0.95,
            max(0.05, 0.30 - 0.20 * max(deployment, reliability, cost, security)),
        ),
    }


def _dominant_signal(severities: Dict[str, float]) -> str:
    return max(severities, key=lambda k: severities[k])


def _action_fit_bonus(action: str, severities: Dict[str, float]) -> float:
    """
    Governance action-fit adjustment.

    This rewards actions that fit the dominant operational signal and
    penalizes generic actions when a specific governance action is required.
    """
    deployment = severities["deployment"]
    reliability = severities["reliability"]
    cost = severities["cost"]
    security = severities["security"]

    dominant = _dominant_signal(severities)
    bonus = 0.0

    # ------------------------------------------------------------
    # Deployment governance
    # ------------------------------------------------------------
    if dominant == "deployment":
        if action == "Rollback to stable deployment":
            bonus += 0.13
        elif action == "Block release and fix pipeline":
            bonus += 0.13
        elif action == "Mitigate and monitor":
            bonus -= 0.12

    # Pipeline gate / release evidence cases:
    # high deployment signal but limited reliability impact should block release,
    # not rollback.
    if deployment >= 0.55 and reliability < 0.35:
        if action == "Block release and fix pipeline":
            bonus += 0.08
        if action == "Rollback to stable deployment":
            bonus -= 0.05

    # Deployment incident with reliability impact should rollback.
    if deployment >= 0.50 and reliability >= 0.35:
        if action == "Rollback to stable deployment":
            bonus += 0.10
        if action == "Block release and fix pipeline":
            bonus -= 0.03
        if action == "Mitigate and monitor":
            bonus -= 0.08

    # ------------------------------------------------------------
    # Reliability governance
    # ------------------------------------------------------------
    if dominant == "reliability":
        if action == "Mitigate and monitor":
            bonus += 0.10
        if action == "Scale adjustment" and reliability >= 0.55:
            bonus += 0.08
        if action == "Rollback to stable deployment" and deployment < 0.40:
            bonus -= 0.08

    # Capacity/resource cases should scale, not only monitor.
    if reliability >= 0.55 and cost >= 0.15:
        if action == "Scale adjustment":
            bonus += 0.18
        if action == "Mitigate and monitor":
            bonus -= 0.08

    # Pure reliability degradation without cost pressure should mitigate.
    if reliability >= 0.55 and cost < 0.15 and deployment < 0.35:
        if action == "Mitigate and monitor":
            bonus += 0.10
        if action == "Scale adjustment":
            bonus -= 0.10

    # ------------------------------------------------------------
    # Cost governance
    # ------------------------------------------------------------
    if dominant == "cost":
        if action == "Scale adjustment":
            bonus += 0.22
        elif action == "Review scaling policy":
            bonus += 0.13
        elif action == "Mitigate and monitor":
            bonus -= 0.18

    # High autoscaling / over-provisioning should scale-adjust.
    if cost >= 0.45:
        if action == "Scale adjustment":
            bonus += 0.10
        if action == "Review scaling policy":
            bonus -= 0.04
        if action == "Mitigate and monitor":
            bonus -= 0.12

    # Moderate cost with stable reliability should review policy.
    if 0.25 <= cost < 0.45 and reliability < 0.30:
        if action == "Review scaling policy":
            bonus += 0.12
        if action == "Scale adjustment":
            bonus -= 0.04

    # ------------------------------------------------------------
    # Security / compliance governance
    # ------------------------------------------------------------
    if dominant == "security":
        if action == "Patch or block release":
            bonus += 0.24
        elif action in {"Rollback to stable deployment", "Mitigate and monitor"}:
            bonus -= 0.15

    # Any meaningful security/compliance signal should strongly prefer patch/block.
    if security >= 0.20:
        if action == "Patch or block release":
            bonus += 0.16
        if action == "Rollback to stable deployment":
            bonus -= 0.10
        if action == "Mitigate and monitor":
            bonus -= 0.12

    return bonus


def choose_action_details(
    telemetry: Dict[str, Any],
    w: Tuple[float, float, float],
) -> Dict[str, Any]:
    severities = _severity_scores(telemetry)
    components = _action_components(telemetry)

    best_action = None
    best_utility = float("-inf")
    best_components = (0.0, 0.0, 0.0)

    candidates = []

    for action, (perf, cost_eff, risk_red) in components.items():
        base_utility = utility_score(perf, cost_eff, risk_red, w)
        fit_bonus = _action_fit_bonus(action, severities)
        final_utility = base_utility + fit_bonus

        candidate = {
            "action": action,
            "performance_score": round(float(perf), 4),
            "cost_efficiency_score": round(float(cost_eff), 4),
            "risk_reduction_score": round(float(risk_red), 4),
            "base_utility": round(float(base_utility), 4),
            "action_fit_bonus": round(float(fit_bonus), 4),
            "utility": round(float(final_utility), 4),
        }
        candidates.append(candidate)

        if final_utility > best_utility:
            best_action = action
            best_utility = final_utility
            best_components = (perf, cost_eff, risk_red)

    candidates.sort(key=lambda x: x["utility"], reverse=True)

    return {
        "selected_action": str(best_action),
        "best_utility": round(float(best_utility), 4),
        "performance_score": round(float(best_components[0]), 4),
        "cost_efficiency_score": round(float(best_components[1]), 4),
        "risk_reduction_score": round(float(best_components[2]), 4),
        "candidates": candidates[:3],
    }


def choose_action(
    telemetry: Dict[str, Any],
    w: Tuple[float, float, float],
) -> Tuple[str, float]:
    details = choose_action_details(telemetry, w)
    return details["selected_action"], float(details["best_utility"])
