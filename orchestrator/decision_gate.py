from __future__ import annotations

"""Pre-decision materiality gate.

The gate is deliberately independent of experiment labels. It answers a
question the earlier pipeline skipped: is there enough provenance-backed
operational evidence to justify an active governance intervention at all?

- no usable evidence -> escalate for evidence acquisition/human review
- usable evidence but no threshold breach in any domain -> abstain/observe
- at least one material domain signal -> allow action selection
"""

from typing import Any, Dict

from evidence.schema import get_evidence
from orchestrator.severity import severity_scores

FIELDS = {
    "deploy": ["pipeline_failed", "config_drift", "rollback_marker", "artifact_mismatch", "restart_loops"],
    "sre": ["p95_latency_ms", "error_rate_pct", "saturation_pct", "availability_pct"],
    "finops": ["cost_spike_pct", "hpa_scale_to", "cpu_request_increase_pct", "memory_request_increase_pct"],
    "sec": ["critical_cves", "policy_violation", "iam_drift", "compliance_gap"],
}


def materiality_gate(telemetry: Dict[str, Any]) -> Dict[str, Any]:
    severities = severity_scores(telemetry)
    usable = 0
    measured = 0
    proxy = 0
    missing = 0

    for block_name, fields in FIELDS.items():
        block = telemetry.get(block_name, {}) or {}
        for field in fields:
            ev = get_evidence(block, field)
            if ev.usable:
                usable += 1
                if ev.status == "measured":
                    measured += 1
                elif ev.status == "proxy":
                    proxy += 1
            elif ev.status == "missing":
                missing += 1

    active_domains = [name for name, score in severities.items() if float(score) > 0.0]
    max_severity = max(severities.values()) if severities else 0.0

    if usable == 0:
        decision = "escalate"
        reason = "No usable measured or proxy evidence is available for a defensible governance decision."
    elif not active_domains:
        decision = "observe"
        reason = "Usable evidence is available, but no domain-specific materiality threshold is breached."
    else:
        decision = "act"
        reason = "At least one provenance-backed domain signal breaches its pre-defined materiality threshold."

    return {
        "decision": decision,
        "reason": reason,
        "active_domains": active_domains,
        "severities": severities,
        "max_severity": float(max_severity),
        "usable_evidence_fields": usable,
        "measured_evidence_fields": measured,
        "proxy_evidence_fields": proxy,
        "missing_evidence_fields": missing,
    }
