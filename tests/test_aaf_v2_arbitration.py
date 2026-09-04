from __future__ import annotations

from llm_evaluation.pm_natural_language_interface import _validate_authority
from orchestrator.cross_domain_v2 import apply_interaction_policy_v2, interaction_state_v2
from orchestrator.utility import choose_action_details


def _ev(status: str = "measured"):
    return {"status": status, "source": "test", "note": ""}


def _base():
    return {
        "deploy": {"_evidence": {}},
        "sre": {"_evidence": {}},
        "finops": {"_evidence": {}},
        "sec": {"_evidence": {}},
    }


def test_generic_three_domain_accumulation_does_not_override_specific_scaling_action():
    telemetry = _base()
    telemetry["deploy"].update({
        "config_drift": True,
        "_evidence": {"config_drift": _ev()},
    })
    telemetry["sre"].update({
        "p95_latency_ms": 900.0,
        "error_rate_pct": 14.0,
        "availability_pct": 98.0,
        "_evidence": {
            "p95_latency_ms": _ev(),
            "error_rate_pct": _ev(),
            "availability_pct": _ev(),
        },
    })
    telemetry["finops"].update({
        "cost_spike_pct": 40.0,
        "_evidence": {"cost_spike_pct": _ev()},
    })

    base = choose_action_details(telemetry, (0.4, 0.3, 0.3))["selected_action"]
    assert base == "Scale adjustment"

    state = interaction_state_v2(telemetry)
    names = {x["name"] for x in state["interactions"]}
    assert "reliability_resource_tradeoff" in names
    assert "multi_domain_accumulation" in names

    result = apply_interaction_policy_v2(telemetry, base)
    assert result["selected_action"] == "Scale adjustment"
    assert result["arbitration_reason"] == "specific_causal_interaction"


def test_restart_plus_reliability_remains_a_specific_rollback_case():
    telemetry = _base()
    telemetry["deploy"].update({
        "restart_burst_count": 4,
        "restart_window_seconds": 45.0,
        "_evidence": {
            "restart_burst_count": _ev(),
            "restart_window_seconds": _ev(),
        },
    })
    telemetry["sre"].update({
        "p95_latency_ms": 1100.0,
        "error_rate_pct": 20.0,
        "availability_pct": 95.0,
        "_evidence": {
            "p95_latency_ms": _ev(),
            "error_rate_pct": _ev(),
            "availability_pct": _ev(),
        },
    })
    base = choose_action_details(telemetry, (0.4, 0.3, 0.3))["selected_action"]
    result = apply_interaction_policy_v2(telemetry, base)
    assert result["selected_action"] == "Rollback to stable deployment"
    assert result["interaction_state"]["dominant_interaction"]["name"] == "deployment_reliability_causal_chain"


def test_release_security_constraint_is_highest_tier():
    telemetry = _base()
    telemetry["deploy"].update({
        "config_drift": True,
        "_evidence": {"config_drift": _ev()},
    })
    telemetry["sec"].update({
        "critical_cves": 1,
        "policy_violation": True,
        "_evidence": {
            "critical_cves": _ev(),
            "policy_violation": _ev(),
        },
    })
    base = choose_action_details(telemetry, (0.4, 0.3, 0.3))["selected_action"]
    result = apply_interaction_policy_v2(telemetry, base)
    assert result["selected_action"] == "Patch or block release"
    assert result["arbitration_reason"] == "hard_override"


def test_pm_llm_cannot_change_authoritative_action_field():
    out = _validate_authority(
        {"answer": "A generated explanation.", "selected_action": "Rollback to stable deployment"},
        "Scale adjustment",
    )
    assert out["selected_action"] == "Scale adjustment"
