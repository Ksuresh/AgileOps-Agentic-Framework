from __future__ import annotations

from typing import Dict, Any, Tuple


ACTIONS = [
    "Rollback to stable deployment",
    "Block release and fix pipeline",
    "Mitigate and monitor",
    "Scale adjustment",
    "Review scaling policy",
    "Patch or block release",
    "No action (observe)",
]


def utility_score(perf_gain: float, cost_delta: float, risk: float, w: Tuple[float, float, float]) -> float:
    w_perf, w_cost, w_risk = w
    return (w_perf * perf_gain) - (w_cost * cost_delta) - (w_risk * risk)


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


def _build_action_profiles(telemetry: Dict[str, Any]) -> Dict[str, Tuple[float, float, float]]:
    scores = _severity_scores(telemetry)

    deployment = scores["deployment"]
    reliability = scores["reliability"]
    cost = scores["cost"]
    security = scores["security"]

    # Utility profiles are intentionally governance-oriented.
    # perf_gain, cost_delta, risk
    return {
        "Rollback to stable deployment": (
            0.35 + 0.45 * deployment + 0.25 * reliability,
            0.15,
            0.20 + 0.10 * security,
        ),
        "Block release and fix pipeline": (
            0.25 + 0.55 * deployment + 0.15 * security,
            0.10,
            0.15,
        ),
        "Mitigate and monitor": (
            0.25 + 0.55 * reliability,
            0.08,
            0.20,
        ),
        "Scale adjustment": (
            0.25 + 0.35 * reliability + 0.45 * cost,
            0.20 + 0.30 * cost,
            0.15,
        ),
        "Review scaling policy": (
            0.15 + 0.60 * cost,
            0.05,
            0.12,
        ),
        "Patch or block release": (
            0.20 + 0.55 * security,
            0.12,
            0.08,
        ),
        "No action (observe)": (
            0.10,
            0.00,
            0.50 * max(deployment, reliability, cost, security),
        ),
    }


def choose_action(telemetry: Dict[str, Any], w: Tuple[float, float, float]) -> Tuple[str, float]:
    actions = _build_action_profiles(telemetry)

    best_action = None
    best_utility = float("-inf")

    for action, (perf_gain, cost_delta, risk) in actions.items():
        u = utility_score(perf_gain, cost_delta, risk, w)
        if u > best_utility:
            best_action = action
            best_utility = u

    return str(best_action), float(best_utility)
