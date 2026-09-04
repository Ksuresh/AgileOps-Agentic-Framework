from __future__ import annotations

"""AAF v2 cross-domain arbitration.

Design principle: generic accumulation is supporting evidence, not an automatic
command override. Hard governance constraints and specific causal interactions
are evaluated before generic accumulation. The module never reads experiment
labels or oracle actions.

This is intentionally separate from cross_domain.py so AAF-v1 remains frozen
and reproducible.
"""

from typing import Any, Dict, List

from evidence.schema import bool_evidence, numeric_evidence
from orchestrator.severity import severity_scores


# Lexicographic arbitration tiers. These are semantic classes, not learned or
# benchmark-tuned weights.
TIER_HARD_OVERRIDE = 3
TIER_SPECIFIC_CAUSAL = 2
TIER_GENERIC_SUPPORT = 1


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
    if not (count.usable and window.usable):
        return False
    c = float(count.value)
    w = max(float(window.value), 1.0)
    return c >= 3.0 and (w / max(c - 1.0, 1.0)) <= 30.0


def _release_gate_failure(telemetry: Dict[str, Any]) -> bool:
    deploy = telemetry.get("deploy", {}) or {}
    for field in ("pipeline_failed", "artifact_mismatch", "config_drift"):
        ev = bool_evidence(deploy, field)
        if ev.usable and bool(ev.value):
            return True
    return False


def _interaction(
    name: str,
    domains: List[str],
    strength: float,
    preferred_action: str,
    rationale: str,
    tier: int,
    causal_support: str,
) -> Dict[str, Any]:
    return {
        "name": name,
        "domains": domains,
        "strength": round(float(strength), 4),
        "preferred_action": preferred_action,
        "rationale": rationale,
        "arbitration_tier": int(tier),
        "causal_support": causal_support,
    }


def interaction_state_v2(telemetry: Dict[str, Any]) -> Dict[str, Any]:
    s = severity_scores(telemetry)
    interactions: List[Dict[str, Any]] = []

    restart_burst = _restart_burst(telemetry)
    sre_outcome = _material_sre_outcome(telemetry)
    gate_failure = _release_gate_failure(telemetry)

    # Hard release/security governance constraint.
    if gate_failure and s["security"] > 0.0:
        interactions.append(_interaction(
            "release_security_override",
            ["deployment", "security"],
            0.95,
            "Patch or block release",
            "release-integrity and security evidence jointly block continuation",
            TIER_HARD_OVERRIDE,
            "direct_joint_evidence",
        ))

    # Specific deployment -> reliability causal pattern.
    if restart_burst and (sre_outcome or s["reliability"] > 0.0):
        interactions.append(_interaction(
            "deployment_reliability_causal_chain",
            ["deployment", "reliability"],
            0.90,
            "Rollback to stable deployment",
            "restart burst co-occurs with material reliability evidence",
            TIER_SPECIFIC_CAUSAL,
            "temporal_plus_outcome",
        ))
    elif restart_burst:
        interactions.append(_interaction(
            "deployment_temporal_instability",
            ["deployment"],
            0.72,
            "Rollback to stable deployment",
            "repeated process transitions form a short-window restart burst",
            TIER_SPECIFIC_CAUSAL,
            "temporal_process_evidence",
        ))

    # Reliability-resource interaction remains a specific actionable relation.
    if s["reliability"] > 0.0 and s["cost"] > 0.0:
        interactions.append(_interaction(
            "reliability_resource_tradeoff",
            ["reliability", "cost"],
            min(0.95, 0.55 + 0.25 * s["reliability"] + 0.25 * s["cost"]),
            "Scale adjustment",
            "material reliability and resource-efficiency evidence require a joint scaling decision",
            TIER_SPECIFIC_CAUSAL,
            "joint_material_evidence",
        ))

    # Generic accumulation is retained for detection/traceability but is not an
    # automatic command override when a specific interaction or an already
    # evidence-supported utility action exists.
    active = [name for name, value in s.items() if float(value) > 0.0]
    if len(active) >= 3 and sum(float(s[x]) for x in active) >= 0.75:
        interactions.append(_interaction(
            "multi_domain_accumulation",
            active,
            min(0.95, 0.55 + 0.15 * len(active)),
            "Patch or block release" if "security" in active else (
                "Rollback to stable deployment" if "deployment" in active else "Mitigate and monitor"
            ),
            "multiple locally plausible domain signals jointly cross the accumulation threshold",
            TIER_GENERIC_SUPPORT,
            "co_occurrence_only",
        ))

    interactions.sort(
        key=lambda x: (int(x["arbitration_tier"]), float(x["strength"])),
        reverse=True,
    )
    return {
        "severities": s,
        "interactions": interactions,
        "active": bool(interactions),
        "dominant_interaction": interactions[0] if interactions else None,
        "arbitration_rule": "hard override > specific causal interaction > generic accumulation",
    }


def apply_interaction_policy_v2(telemetry: Dict[str, Any], base_action: str) -> Dict[str, Any]:
    """Apply v2 arbitration without letting generic accumulation erase a valid base action."""
    state = interaction_state_v2(telemetry)
    interactions = state["interactions"]
    if not interactions:
        return {
            "selected_action": base_action,
            "interaction_applied": False,
            "interaction_state": state,
            "base_action": base_action,
            "arbitration_reason": "no_cross_domain_interaction",
        }

    top = interactions[0]
    tier = int(top["arbitration_tier"])

    # Hard constraints always govern. Specific causal interactions can govern.
    # Generic accumulation is traceable context only and never overrides an
    # already evidence-supported base action by itself.
    if tier >= TIER_SPECIFIC_CAUSAL:
        selected = str(top["preferred_action"])
        return {
            "selected_action": selected,
            "interaction_applied": selected != base_action,
            "interaction_state": state,
            "base_action": base_action,
            "arbitration_reason": "hard_override" if tier == TIER_HARD_OVERRIDE else "specific_causal_interaction",
        }

    return {
        "selected_action": base_action,
        "interaction_applied": False,
        "interaction_state": state,
        "base_action": base_action,
        "arbitration_reason": "generic_accumulation_support_only",
    }
