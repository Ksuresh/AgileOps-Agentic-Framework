from __future__ import annotations

"""Evidence-driven cross-domain governance interactions.

This module is deliberately independent of experiment labels/oracles. It takes
only provenance-backed telemetry/severity and expresses governance interactions
that cannot be represented by a dominant-domain policy alone.

The interaction layer is an architectural hypothesis to be ablated in held-out
experiments; it is not an oracle lookup table.
"""

from typing import Any, Dict, List

from evidence.schema import numeric_evidence, bool_evidence
from orchestrator.severity import severity_scores


def _material_sre_outcome(telemetry: Dict[str, Any]) -> bool:
    sre = telemetry.get("sre", {}) or {}
    p95 = numeric_evidence(sre, "p95_latency_ms")
    err = numeric_evidence(sre, "error_rate_pct")
    avail = numeric_evidence(sre, "availability_pct")
    return bool(
        (p95.usable and float(p95.value) >= 450.0)
        or (err.usable and float(err.value) >= 8.0)
        or (avail.usable and float(avail.value) < 99.0)
    )


def _restart_burst(telemetry: Dict[str, Any]) -> bool:
    deploy = telemetry.get("deploy", {}) or {}
    count = numeric_evidence(deploy, "restart_burst_count")
    window = numeric_evidence(deploy, "restart_window_seconds")
    if count.usable and window.usable:
        c = float(count.value)
        w = max(float(window.value), 1.0)
        # General temporal-instability rule: at least 3 transitions with an
        # average inter-restart interval <=30 s. This is not tied to RT-04's
        # exact count of five.
        return c >= 3.0 and (w / max(c - 1.0, 1.0)) <= 30.0
    return False


def _release_gate_failure(telemetry: Dict[str, Any]) -> bool:
    deploy = telemetry.get("deploy", {}) or {}
    pipeline = bool_evidence(deploy, "pipeline_failed")
    artifact = bool_evidence(deploy, "artifact_mismatch")
    config = bool_evidence(deploy, "config_drift")
    return bool(
        (pipeline.usable and bool(pipeline.value))
        or (artifact.usable and bool(artifact.value))
        or (config.usable and bool(config.value))
    )


def interaction_state(telemetry: Dict[str, Any]) -> Dict[str, Any]:
    s = severity_scores(telemetry)
    interactions: List[Dict[str, Any]] = []

    restart_burst = _restart_burst(telemetry)
    sre_outcome = _material_sre_outcome(telemetry)
    gate_failure = _release_gate_failure(telemetry)

    # Deployment x reliability: temporal instability after a release is more
    # informative than either a restart count or a reliability metric alone.
    if restart_burst and (sre_outcome or s["reliability"] > 0.0):
        interactions.append({
            "name": "deployment_reliability_causal_chain",
            "domains": ["deployment", "reliability"],
            "strength": 0.90,
            "preferred_action": "Rollback to stable deployment",
            "rationale": "restart burst co-occurs with material reliability evidence",
        })
    elif restart_burst:
        interactions.append({
            "name": "deployment_temporal_instability",
            "domains": ["deployment"],
            "strength": 0.72,
            "preferred_action": "Rollback to stable deployment",
            "rationale": "repeated process transitions form a short-window restart burst",
        })

    # Deployment x security: a technically healthy service does not make a
    # failed release/security gate acceptable.
    if gate_failure and s["security"] > 0.0:
        interactions.append({
            "name": "release_security_override",
            "domains": ["deployment", "security"],
            "strength": 0.95,
            "preferred_action": "Patch or block release",
            "rationale": "release-integrity and security evidence jointly block continuation",
        })

    # Reliability x cost/resource efficiency: when both are material, immediate
    # scale correction is preferred over a purely advisory FinOps response.
    if s["reliability"] > 0.0 and s["cost"] > 0.0:
        interactions.append({
            "name": "reliability_resource_tradeoff",
            "domains": ["reliability", "cost"],
            "strength": min(0.95, 0.55 + 0.25 * s["reliability"] + 0.25 * s["cost"]),
            "preferred_action": "Scale adjustment",
            "rationale": "material reliability and resource-efficiency evidence require a joint scaling decision",
        })

    # Multi-domain weak-signal accumulation. This is intentionally conservative:
    # at least three non-zero domains are required and the summed evidence must
    # exceed a joint threshold before active governance is recommended.
    active = [name for name, value in s.items() if float(value) > 0.0]
    if len(active) >= 3 and sum(float(s[x]) for x in active) >= 0.75:
        preferred = "Patch or block release" if "security" in active else (
            "Rollback to stable deployment" if "deployment" in active else "Mitigate and monitor"
        )
        interactions.append({
            "name": "multi_domain_accumulation",
            "domains": active,
            "strength": min(0.95, 0.55 + 0.15 * len(active)),
            "preferred_action": preferred,
            "rationale": "multiple locally plausible domain signals jointly cross a governance threshold",
        })

    interactions.sort(key=lambda x: float(x["strength"]), reverse=True)
    return {
        "severities": s,
        "interactions": interactions,
        "active": bool(interactions),
        "dominant_interaction": interactions[0] if interactions else None,
    }


def apply_interaction_policy(telemetry: Dict[str, Any], base_action: str) -> Dict[str, Any]:
    state = interaction_state(telemetry)
    dominant = state["dominant_interaction"]
    if dominant is None:
        return {
            "selected_action": base_action,
            "interaction_applied": False,
            "interaction_state": state,
        }
    return {
        "selected_action": str(dominant["preferred_action"]),
        "interaction_applied": True,
        "interaction_state": state,
    }
