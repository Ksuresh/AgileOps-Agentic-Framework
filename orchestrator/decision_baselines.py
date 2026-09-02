from __future__ import annotations

from typing import Any, Dict, Tuple

from evidence.schema import bool_evidence, numeric_evidence
from orchestrator.severity import severity_scores


DOMAIN_ACTION_MAP = {
    "deployment": "Rollback to stable deployment",
    "reliability": "Mitigate and monitor",
    "cost": "Scale adjustment",
    "security": "Patch or block release",
}


def choose_dominant_domain_action(telemetry: Dict[str, Any]) -> Tuple[str, float, str]:
    """Simple non-utility comparator based on the strongest provenance-aware domain severity."""
    severities = severity_scores(telemetry)
    dominant = max(severities, key=severities.get)
    severity = float(severities[dominant])
    if severity <= 0.0:
        return "No action (observe)", 0.0, dominant

    action = DOMAIN_ACTION_MAP[dominant]
    if dominant == "deployment":
        deploy = telemetry.get("deploy", {}) or {}
        sre = telemetry.get("sre", {}) or {}
        pipeline = bool_evidence(deploy, "pipeline_failed")
        artifact = bool_evidence(deploy, "artifact_mismatch")
        p95 = numeric_evidence(sre, "p95_latency_ms")
        err = numeric_evidence(sre, "error_rate_pct")
        avail = numeric_evidence(sre, "availability_pct")
        reliability_impact = (
            (p95.usable and float(p95.value) >= 450)
            or (err.usable and float(err.value) >= 8)
            or (avail.usable and float(avail.value) < 99.0)
        )
        gate_failure = (pipeline.usable and bool(pipeline.value)) or (artifact.usable and bool(artifact.value))
        if gate_failure and not reliability_impact:
            action = "Block release and fix pipeline"
    return action, severity, dominant
