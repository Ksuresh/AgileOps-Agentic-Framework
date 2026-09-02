from __future__ import annotations

from typing import Any, Dict, Tuple


DOMAIN_ACTION_MAP = {
    "deployment": "Rollback to stable deployment",
    "reliability": "Mitigate and monitor",
    "cost": "Scale adjustment",
    "security": "Patch or block release",
}


def _severity_scores(telemetry: Dict[str, Any]) -> Dict[str, float]:
    """Independent, non-utility severity scores used only for the baseline.

    This deliberately avoids the multi-objective utility function and its
    action-fit bonuses. It provides a meaningful comparator for testing whether
    utility-based action ranking adds value over a simple dominant-domain policy.
    """
    deploy = telemetry.get("deploy", {}) or {}
    sre = telemetry.get("sre", {}) or {}
    finops = telemetry.get("finops", {}) or {}
    sec = telemetry.get("sec", {}) or {}

    deployment = 0.0
    deployment += 0.35 if bool(deploy.get("pipeline_failed", False)) else 0.0
    deployment += 0.25 if bool(deploy.get("config_drift", False)) else 0.0
    deployment += 0.25 if bool(deploy.get("rollback_marker", False)) else 0.0
    deployment += 0.25 if bool(deploy.get("artifact_mismatch", False)) else 0.0
    restart_loops = int(float(deploy.get("restart_loops", 0) or 0))
    deployment += 0.25 if restart_loops >= 12 else (0.10 if restart_loops >= 6 else 0.0)

    p95 = float(sre.get("p95_latency_ms", 0.0) or 0.0)
    err = float(sre.get("error_rate_pct", 0.0) or 0.0)
    sat = float(sre.get("saturation_pct", 0.0) or 0.0)
    availability = float(sre.get("availability_pct", 99.9) or 99.9)
    reliability = 0.0
    reliability += 0.35 if p95 >= 800 else (0.25 if p95 >= 450 else 0.0)
    reliability += 0.35 if err >= 12 else (0.25 if err >= 8 else 0.0)
    reliability += 0.25 if sat >= 90 else (0.15 if sat >= 85 else 0.0)
    reliability += 0.25 if availability < 99.0 else 0.0

    cost_spike = float(finops.get("cost_spike_pct", 0.0) or 0.0)
    hpa_scale = int(float(finops.get("hpa_scale_to", 0) or 0))
    cpu_inc = float(finops.get("cpu_request_increase_pct", 0.0) or 0.0)
    mem_inc = float(finops.get("memory_request_increase_pct", 0.0) or 0.0)
    cost = 0.0
    cost += 0.40 if cost_spike >= 35 else (0.30 if cost_spike >= 22 else (0.15 if cost_spike >= 8 else 0.0))
    cost += 0.25 if hpa_scale >= 14 else (0.15 if hpa_scale >= 11 else 0.0)
    cost += 0.20 if cpu_inc >= 50 else 0.0
    cost += 0.20 if mem_inc >= 40 else 0.0

    critical_cves = int(float(sec.get("critical_cves", 0) or 0))
    security = 0.0
    security += 0.40 if critical_cves >= 2 else (0.30 if critical_cves == 1 else 0.0)
    security += 0.25 if bool(sec.get("policy_violation", False)) else 0.0
    security += 0.20 if bool(sec.get("iam_drift", False)) else 0.0
    security += 0.20 if bool(sec.get("compliance_gap", False)) else 0.0

    return {
        "deployment": min(1.0, deployment),
        "reliability": min(1.0, reliability),
        "cost": min(1.0, cost),
        "security": min(1.0, security),
    }


def choose_dominant_domain_action(telemetry: Dict[str, Any]) -> Tuple[str, float, str]:
    """Choose an action using only the strongest domain severity.

    Returns (action, severity, dominant_domain). This is a deliberately simple
    non-utility baseline for ablation/comparison experiments.
    """
    severities = _severity_scores(telemetry)
    dominant = max(severities, key=severities.get)
    severity = float(severities[dominant])

    if severity <= 0.0:
        return "No action (observe)", 0.0, dominant

    action = DOMAIN_ACTION_MAP[dominant]

    # A deployment-only pipeline-gate condition is better represented as a hold
    # than a rollback. This remains a simple deterministic policy and does not
    # use utility ranking.
    deploy = telemetry.get("deploy", {}) or {}
    sre = telemetry.get("sre", {}) or {}
    if dominant == "deployment":
        pipeline_failed = bool(deploy.get("pipeline_failed", False))
        artifact_mismatch = bool(deploy.get("artifact_mismatch", False))
        reliability_impact = (
            float(sre.get("p95_latency_ms", 0.0) or 0.0) >= 450
            or float(sre.get("error_rate_pct", 0.0) or 0.0) >= 8
            or float(sre.get("availability_pct", 99.9) or 99.9) < 99.0
        )
        if (pipeline_failed or artifact_mismatch) and not reliability_impact:
            action = "Block release and fix pipeline"

    return action, severity, dominant
