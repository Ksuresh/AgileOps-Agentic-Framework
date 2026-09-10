from __future__ import annotations

from agentic_experiments.agentic_runtime import EVALUATOR_ONLY, domain_evidence_packet, strip_evaluator_fields
from agentic_experiments.hybrid import deterministic_assessment, selective_trigger


def test_evaluator_fields_are_never_model_visible():
    case = {
        "id": "SECRET-ID", "stratum": "INCOMPLETE", "focus": "secret-focus",
        "evidence": {"sre": {"p95_latency_ms": 900}},
        "oracle_domains": ["SRE"], "admissible_actions": ["Mitigate and monitor"],
        "expected_agentic": True, "expected_tool": "get_error_rate_history",
        "tool_result": {"id": "SECRET-TOOL"}, "misleading_signal": "secret",
        "available_tools": ["get_error_rate_history"],
    }
    visible = strip_evaluator_fields(case)
    for key in EVALUATOR_ONLY:
        assert key not in visible
    assert visible["evidence"]["sre"]["p95_latency_ms"] == 900
    assert visible["available_tools"] == ["get_error_rate_history"]


def test_specialist_agent_sees_only_assigned_domain_raw_evidence():
    evidence = {
        "deploy": {"config_drift": True},
        "sre": {"p95_latency_ms": 900},
        "finops": {"cost_spike_pct": 30},
        "sec": {"critical_cves": 2},
    }
    packet = domain_evidence_packet(evidence, "SRE")
    assert set(packet) == {"sre"}
    assert packet["sre"]["p95_latency_ms"]["id"] == "EV-sre-p95_latency_ms"


def test_clear_healthy_case_does_not_trigger_agentic_reasoning():
    evidence = {
        "deploy": {"pipeline_failed": False, "artifact_mismatch": False, "config_drift": False},
        "sre": {"p95_latency_ms": 210, "error_rate_pct": 0.8, "saturation_pct": 44, "availability_pct": 99.95},
        "finops": {"cost_spike_pct": 2, "hpa_scale_to": 4},
        "sec": {"critical_cves": 0, "policy_violation": False},
    }
    det = deterministic_assessment(evidence)
    trigger = selective_trigger(evidence, det)
    assert trigger["invoke"] is False


def test_near_boundary_case_triggers_without_oracle():
    evidence = {
        "sre": {"p95_latency_ms": 470, "error_rate_pct": 7.7, "saturation_pct": 84, "availability_pct": 99.1},
        "deploy": {"pipeline_failed": False},
        "finops": {"cost_spike_pct": 4},
        "sec": {"policy_violation": False},
    }
    det = deterministic_assessment(evidence)
    trigger = selective_trigger(evidence, det)
    assert trigger["invoke"] is True
    assert "clustered_boundary_ambiguity" in trigger["reasons"]


def test_explicit_missing_evidence_triggers_without_oracle():
    evidence = {
        "deploy": {
            "restart_burst_count": None,
            "restart_window_seconds": None,
            "_evidence": {
                "restart_burst_count": {"status": "missing"},
                "restart_window_seconds": {"status": "missing"},
            },
        },
        "sre": {"p95_latency_ms": 920, "error_rate_pct": 13, "availability_pct": 98.1},
    }
    det = deterministic_assessment(evidence)
    trigger = selective_trigger(evidence, det)
    assert trigger["invoke"] is True
    assert "decision_relevant_evidence_missing" in trigger["reasons"]
