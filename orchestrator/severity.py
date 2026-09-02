from __future__ import annotations

from typing import Any, Dict

from evidence.schema import bool_evidence, numeric_evidence


def severity_scores(telemetry: Dict[str, Any]) -> Dict[str, float]:
    deploy = telemetry.get("deploy", {}) or {}
    sre = telemetry.get("sre", {}) or {}
    finops = telemetry.get("finops", {}) or {}
    sec = telemetry.get("sec", {}) or {}

    deployment = 0.0
    for field, weight in (
        ("pipeline_failed", 0.35),
        ("config_drift", 0.25),
        ("rollback_marker", 0.25),
        ("artifact_mismatch", 0.25),
    ):
        ev = bool_evidence(deploy, field)
        if ev.usable and ev.value:
            deployment += weight
    restart = numeric_evidence(deploy, "restart_loops")
    if restart.usable:
        value = int(float(restart.value))
        deployment += 0.25 if value >= 12 else (0.10 if value >= 6 else 0.0)

    reliability = 0.0
    p95 = numeric_evidence(sre, "p95_latency_ms")
    err = numeric_evidence(sre, "error_rate_pct")
    sat = numeric_evidence(sre, "saturation_pct")
    availability = numeric_evidence(sre, "availability_pct")
    if p95.usable:
        value = float(p95.value)
        reliability += 0.35 if value >= 800 else (0.25 if value >= 450 else 0.0)
    if err.usable:
        value = float(err.value)
        reliability += 0.35 if value >= 12 else (0.25 if value >= 8 else 0.0)
    if sat.usable:
        value = float(sat.value)
        reliability += 0.25 if value >= 90 else (0.15 if value >= 85 else 0.0)
    if availability.usable and float(availability.value) < 99.0:
        reliability += 0.25

    cost = 0.0
    spike = numeric_evidence(finops, "cost_spike_pct")
    hpa = numeric_evidence(finops, "hpa_scale_to")
    cpu = numeric_evidence(finops, "cpu_request_increase_pct")
    mem = numeric_evidence(finops, "memory_request_increase_pct")
    if spike.usable:
        value = float(spike.value)
        cost += 0.40 if value >= 35 else (0.30 if value >= 22 else (0.15 if value >= 8 else 0.0))
    if hpa.usable:
        value = int(float(hpa.value))
        cost += 0.25 if value >= 14 else (0.15 if value >= 11 else 0.0)
    if cpu.usable and float(cpu.value) >= 50:
        cost += 0.20
    if mem.usable and float(mem.value) >= 40:
        cost += 0.20

    security = 0.0
    cves = numeric_evidence(sec, "critical_cves")
    if cves.usable:
        value = int(float(cves.value))
        security += 0.40 if value >= 2 else (0.30 if value == 1 else 0.0)
    for field, weight in (("policy_violation", 0.25), ("iam_drift", 0.20), ("compliance_gap", 0.20)):
        ev = bool_evidence(sec, field)
        if ev.usable and ev.value:
            security += weight

    return {
        "deployment": min(1.0, deployment),
        "reliability": min(1.0, reliability),
        "cost": min(1.0, cost),
        "security": min(1.0, security),
    }
