from __future__ import annotations

from typing import Dict, Any, Tuple


def utility_score(perf_gain: float, cost_delta: float, risk: float, w: Tuple[float, float, float]) -> float:
    """
    Utility function:
        U(a) = w_perf * perf_gain - w_cost * cost_delta - w_risk * risk
    """
    w_perf, w_cost, w_risk = w
    return (w_perf * perf_gain) - (w_cost * cost_delta) - (w_risk * risk)


def _build_action_profiles(telemetry: Dict[str, Any]) -> Dict[str, Tuple[float, float, float]]:
    """
    Build telemetry-aware action profiles.

    Returns:
        action -> (perf_gain, cost_delta, risk)

    Notes:
    - Higher perf_gain is better.
    - Higher cost_delta is worse.
    - Higher risk is worse.
    """
    deploy = telemetry.get("deploy", {}) or {}
    sre = telemetry.get("sre", {}) or {}
    finops = telemetry.get("finops", {}) or {}
    sec = telemetry.get("sec", {}) or {}

    restart_loops = int(deploy.get("restart_loops", 0))
    pipeline_failed = bool(deploy.get("pipeline_failed", False))
    config_drift = bool(deploy.get("config_drift", False))

    p95 = float(sre.get("p95_latency_ms", 0.0))
    err = float(sre.get("error_rate_pct", 0.0))
    sat = float(sre.get("saturation_pct", 0.0))

    cost_spike = float(finops.get("cost_spike_pct", 0.0))
    hpa_scale = int(finops.get("hpa_scale_to", 0))

    critical_cves = int(sec.get("critical_cves", 0))
    policy_violation = bool(sec.get("policy_violation", False))
    iam_drift = bool(sec.get("iam_drift", False))

    security_severity = 0.0
    if critical_cves > 0:
        security_severity += 0.5
    if policy_violation:
        security_severity += 0.3
    if iam_drift:
        security_severity += 0.2

    reliability_severity = 0.0
    if p95 >= 450:
        reliability_severity += 0.35
    if err >= 8.0:
        reliability_severity += 0.35
    if sat >= 85:
        reliability_severity += 0.20
    if restart_loops >= 12:
        reliability_severity += 0.10

    deployment_severity = 0.0
    if pipeline_failed:
        deployment_severity += 0.4
    if config_drift:
        deployment_severity += 0.3
    if restart_loops >= 12:
        deployment_severity += 0.3

    cost_severity = 0.0
    if cost_spike >= 22:
        cost_severity += 0.6
    elif cost_spike >= 8:
        cost_severity += 0.3

    if hpa_scale >= 11:
        cost_severity += 0.3
    elif hpa_scale >= 7:
        cost_severity += 0.1

    overall_pressure = min(
        1.0,
        security_severity + reliability_severity + deployment_severity + cost_severity
    )

    actions: Dict[str, Tuple[float, float, float]] = {
        "Rollback to stable deployment": (
            min(1.0, 0.45 + deployment_severity + 0.5 * reliability_severity),
            0.20,
            max(0.05, 0.25 + 0.20 * security_severity),
        ),
        "Mitigate and monitor": (
            min(1.0, 0.40 + 0.7 * reliability_severity + 0.2 * deployment_severity),
            0.10,
            max(0.05, 0.15 + 0.10 * overall_pressure),
        ),
        "Scale adjustment": (
            min(1.0, 0.30 + 0.8 * cost_severity + 0.3 * reliability_severity),
            min(1.0, 0.15 + 0.6 * cost_severity),
            max(0.05, 0.10 + 0.10 * security_severity),
        ),
        "Patch or block release": (
            min(1.0, 0.25 + 0.3 * security_severity),
            0.15,
            max(0.0, 0.60 - 0.45 * security_severity),
        ),
        "No action (observe)": (
            max(0.0, 0.20 - 0.10 * overall_pressure),
            0.00,
            min(1.0, 0.05 + 0.60 * overall_pressure),
        ),
    }

    return actions


def choose_action(telemetry: Dict[str, Any], w: Tuple[float, float, float]) -> Tuple[str, float]:
    """
    Select the action with maximum utility under telemetry-aware profiles.
    """
    actions = _build_action_profiles(telemetry)

    best_action = None
    best_utility = float("-inf")

    for action, (perf_gain, cost_delta, risk) in actions.items():
        u = utility_score(perf_gain, cost_delta, risk, w)
        if u > best_utility:
            best_utility = u
            best_action = action

    return str(best_action), float(best_utility)
