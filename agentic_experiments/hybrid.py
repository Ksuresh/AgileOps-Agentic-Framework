from __future__ import annotations

import copy
from typing import Any

from agents.devops import DevOpsAgent
from agents.sre import SREAgent
from agents.finops import FinOpsAgent
from agents.devsecops import DevSecOpsAgent
from orchestrator.consensus import consensus_score
from orchestrator.cross_domain_v2 import apply_interaction_policy_v2, interaction_state_v2
from orchestrator.decision_gate import materiality_gate
from orchestrator.utility import choose_action_details

from agentic_experiments.agentic_runtime import (
    BoundedAgenticRuntime,
    evidence_packet,
    merge_tool_result,
    packet_ids,
    total_call_usage,
)

AGENTS = [DevOpsAgent(), SREAgent(), FinOpsAgent(), DevSecOpsAgent()]
WEIGHTS = (0.4, 0.3, 0.3)
TAU = 0.75


def deterministic_assessment(evidence: dict[str, Any]) -> dict[str, Any]:
    outputs = [a.infer(evidence) for a in AGENTS]
    claims = [x.claim for x in outputs]
    confs = [float(x.confidence) for x in outputs]
    consensus, _ = consensus_score(claims, confs, lam=0.5)
    gate = materiality_gate(evidence)
    if gate["decision"] == "observe":
        base_action = "No action (observe)"
        governed = {"selected_action": base_action, "base_action": base_action, "interaction_state": interaction_state_v2(evidence)}
    elif gate["decision"] == "escalate":
        base_action = "Escalate for evidence/human review"
        governed = {"selected_action": base_action, "base_action": base_action, "interaction_state": interaction_state_v2(evidence)}
    else:
        utility = choose_action_details(evidence, WEIGHTS)
        base_action = str(utility["selected_action"])
        governed = apply_interaction_policy_v2(evidence, base_action)
    return {
        "agents": [x.__dict__ for x in outputs],
        "consensus": float(consensus),
        "gate": gate,
        "base_action": base_action,
        "selected_action": str(governed["selected_action"]),
        "governance": governed,
    }


def missing_fields(evidence: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for domain, block in evidence.items():
        if not isinstance(block, dict):
            continue
        meta = block.get("_evidence", {}) or {}
        for field, item in meta.items():
            if isinstance(item, dict) and str(item.get("status", "")).lower() == "missing":
                out.append(f"{domain}.{field}")
    return sorted(out)


def _numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _borderline_cluster(evidence: dict[str, Any]) -> bool:
    """Detect ambiguity, not severity.

    A single value at a threshold is not enough to invoke Agentic reasoning. A
    domain must contain a cluster of near-boundary indicators without a clearly
    severe indicator that already resolves the direction. This prevents clear
    high-severity cases from being misclassified as ambiguous simply because a
    value equals a frozen threshold.
    """
    specs = {
        "sre": {
            "near": {
                "p95_latency_ms": [(450.0, 25.0), (800.0, 40.0)],
                "error_rate_pct": [(8.0, 0.5), (12.0, 0.75)],
                "saturation_pct": [(85.0, 2.0), (90.0, 2.0)],
                "availability_pct": [(99.0, 0.15)],
            },
            "strong": {
                "p95_latency_ms": lambda x: x >= 850,
                "error_rate_pct": lambda x: x >= 13,
                "saturation_pct": lambda x: x >= 92,
                "availability_pct": lambda x: x < 98.5,
            },
        },
        "finops": {
            "near": {
                "cost_spike_pct": [(22.0, 2.0), (35.0, 2.0)],
                "hpa_scale_to": [(11.0, 1.0), (14.0, 1.0)],
                "cpu_request_increase_pct": [(50.0, 3.0)],
                "memory_request_increase_pct": [(40.0, 3.0)],
            },
            "strong": {
                "cost_spike_pct": lambda x: x >= 38,
                "hpa_scale_to": lambda x: x >= 15,
                "cpu_request_increase_pct": lambda x: x >= 55,
                "memory_request_increase_pct": lambda x: x >= 45,
            },
        },
    }

    for domain, spec in specs.items():
        block = evidence.get(domain, {}) or {}
        near_count = 0
        strong = False
        for field, intervals in spec["near"].items():
            x = _numeric(block.get(field))
            if x is None:
                continue
            if any(abs(x - threshold) <= margin for threshold, margin in intervals):
                near_count += 1
        for field, predicate in spec["strong"].items():
            x = _numeric(block.get(field))
            if x is not None and predicate(x):
                strong = True
                break
        if near_count >= 2 and not strong:
            return True

    deploy = evidence.get("deploy", {}) or {}
    rc = _numeric(deploy.get("restart_burst_count"))
    rw = _numeric(deploy.get("restart_window_seconds"))
    if rc is not None and rw is not None and 2 <= rc <= 4 and 55 <= rw <= 80:
        return True

    return False


def selective_trigger(evidence: dict[str, Any], det: dict[str, Any]) -> dict[str, Any]:
    """Protocol-v2 selective trigger: uncertainty invokes agency, severity does not."""
    reasons: list[str] = []
    missing = missing_fields(evidence)
    if missing:
        reasons.append("decision_relevant_evidence_missing")

    state = det["governance"].get("interaction_state") or interaction_state_v2(evidence)
    interactions = state.get("interactions", []) or []
    actions = {str(x.get("preferred_action")) for x in interactions if x.get("preferred_action")}
    competing = len(actions) >= 2
    if competing:
        reasons.append("competing_cross_domain_actions")

    low_readiness = det["gate"].get("decision") == "act" and float(det["consensus"]) < TAU
    if low_readiness:
        reasons.append("decision_readiness_below_tau")

    if _borderline_cluster(evidence):
        reasons.append("clustered_boundary_ambiguity")

    deploy = evidence.get("deploy", {}) or {}
    rc = deploy.get("restart_burst_count")
    rw = deploy.get("restart_window_seconds")
    if (rc is None) != (rw is None):
        reasons.append("temporal_relation_incomplete")

    if det["gate"].get("decision") == "escalate":
        reasons.append("insufficient_evidence_for_governed_action")

    return {"invoke": bool(reasons), "reasons": sorted(set(reasons)), "missing_fields": missing}


def _relevant_agents(evidence: dict[str, Any], det: dict[str, Any]) -> list[str]:
    relevant: list[str] = []
    for output in det["agents"]:
        claim = str(output.get("claim", "")).lower()
        conf = float(output.get("confidence", 0.0) or 0.0)
        if conf >= 0.45 and "no material" not in claim and "no anomaly" not in claim:
            relevant.append(str(output["agent_type"]))
    for agent, block in (("DevOps", "deploy"), ("SRE", "sre"), ("FinOps", "finops"), ("DevSecOps", "sec")):
        if block in evidence and agent not in relevant:
            if any(x.startswith(block + ".") for x in missing_fields(evidence)):
                relevant.append(agent)
    return relevant or ["DevOps", "SRE", "FinOps", "DevSecOps"]


def _execute_requested_tool(case: dict[str, Any], requested: list[str]) -> tuple[dict[str, Any] | None, str | None]:
    available = list(case.get("available_tools", []) or [])
    for tool in requested:
        if tool in available:
            if tool == case.get("expected_tool"):
                return copy.deepcopy(case.get("tool_result")), tool
            return {"id": f"TOOL-NORESULT-{tool}", "status": "no_additional_evidence"}, tool
    return None, None


def run_agentic_only(case: dict[str, Any], runtime: BoundedAgenticRuntime) -> dict[str, Any]:
    evidence = copy.deepcopy(case["evidence"])
    available = list(case.get("available_tools", []) or [])
    calls = []
    outputs = []
    tool_events = []
    for agent in ["DevOps", "SRE", "FinOps", "DevSecOps"]:
        call = runtime.domain_agent(agent, evidence, available)
        calls.append(call)
        requested = list(call.output.get("requested_tools", []) or [])
        result, tool = _execute_requested_tool(case, requested)
        if result is not None and tool is not None:
            tool_events.append({"agent": agent, "tool": tool, "result": result})
            if tool == case.get("expected_tool"):
                evidence = merge_tool_result(evidence, result)
                second = runtime.domain_agent(agent, evidence, available, [result])
                calls.append(second)
                call = second
        outputs.append(call.output)
    known = packet_ids(evidence_packet(evidence)) | {str(x["result"].get("id")) for x in tool_events if isinstance(x.get("result"), dict) and x["result"].get("id")}
    coord = runtime.coordinator(outputs, known)
    calls.append(coord)
    return {
        "selected_action": str(coord.output.get("selected_action", "Escalate for evidence/human review")),
        "agent_outputs": outputs,
        "coordinator": coord.output,
        "tool_events": tool_events,
        "usage": total_call_usage(calls),
        "latency_ms": sum(x.latency_ms for x in calls),
        "all_schema_valid": all(x.schema_valid for x in calls),
        "all_evidence_refs_valid": all(x.valid_evidence_refs for x in calls),
        "call_audit": [x.audit_record() for x in calls],
    }


def run_hybrid(case: dict[str, Any], runtime: BoundedAgenticRuntime) -> dict[str, Any]:
    evidence = copy.deepcopy(case["evidence"])
    det_before = deterministic_assessment(evidence)
    trigger = selective_trigger(evidence, det_before)
    if not trigger["invoke"]:
        return {
            "selected_action": det_before["selected_action"], "deterministic_before": det_before,
            "deterministic_after": det_before, "trigger": trigger, "agentic_invoked": False,
            "agent_outputs": [], "tool_events": [], "call_audit": [],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, "latency_ms": 0.0,
            "governance_override": False,
        }

    available = list(case.get("available_tools", []) or [])
    calls = []
    outputs = []
    tool_events = []
    for agent in _relevant_agents(evidence, det_before):
        call = runtime.domain_agent(agent, evidence, available)
        calls.append(call)
        requested = list(call.output.get("requested_tools", []) or [])
        result, tool = _execute_requested_tool(case, requested)
        if result is not None and tool is not None:
            tool_events.append({"agent": agent, "tool": tool, "result": result})
            if tool == case.get("expected_tool"):
                evidence = merge_tool_result(evidence, result)
                second = runtime.domain_agent(agent, evidence, available, [result])
                calls.append(second)
                call = second
        outputs.append(call.output)

    known = packet_ids(evidence_packet(evidence)) | {str(x["result"].get("id")) for x in tool_events if isinstance(x.get("result"), dict) and x["result"].get("id")}
    coord = runtime.coordinator(outputs, known)
    calls.append(coord)
    agent_proposal = str(coord.output.get("selected_action", "Escalate for evidence/human review"))

    det_after = deterministic_assessment(evidence)
    final_action = str(det_after["selected_action"])

    return {
        "selected_action": final_action,
        "agent_proposal": agent_proposal,
        "deterministic_before": det_before,
        "deterministic_after": det_after,
        "trigger": trigger,
        "agentic_invoked": True,
        "agent_outputs": outputs,
        "tool_events": tool_events,
        "usage": total_call_usage(calls),
        "latency_ms": sum(x.latency_ms for x in calls),
        "all_schema_valid": all(x.schema_valid for x in calls),
        "all_evidence_refs_valid": all(x.valid_evidence_refs for x in calls),
        "governance_override": final_action != agent_proposal,
        "governance": det_after["governance"],
        "call_audit": [x.audit_record() for x in calls],
    }
