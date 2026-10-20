"""Reviewer comparator D1: deterministic missing-evidence retrieval.

No LLM. Tool selection sees only case evidence and available_tools.
The frozen simulator's hidden expected_tool/tool_result are accessed ONLY by
the common executor after tool selection, never by the selection policy.
"""
from __future__ import annotations

import copy
from typing import Any

from agentic_experiments.agentic_runtime import merge_tool_result
from agentic_experiments.hybrid import deterministic_assessment, missing_fields

# Fixed field-to-tool semantics; no case IDs, strata, or expected-tool labels.
FIELD_TOOL_MAP: dict[str, str] = {
    "deploy.restart_burst_count": "get_restart_history",
    "deploy.restart_window_seconds": "get_restart_history",
    "deploy.pipeline_failed": "get_pipeline_history",
    "deploy.artifact_mismatch": "get_deployment_history",
    "deploy.config_drift": "get_deployment_history",
    "sre.error_rate_pct": "get_error_rate_history",
    "sre.p95_latency_ms": "get_latency_history",
    "sre.saturation_pct": "get_saturation_history",
    "finops.hpa_scale_to": "get_replica_history",
    "finops.cost_spike_pct": "get_cost_proxy_history",
    "sec.policy_violation": "get_policy_gate_status",
    "sec.critical_cves": "get_vulnerability_details",
}

def select_tools(evidence: dict[str, Any], available_tools: list[str], max_calls: int = 1) -> list[str]:
    """Choose tools from observed missing-field metadata, never evaluator fields."""
    if max_calls < 0:
        raise ValueError("max_calls must be nonnegative")
    allowed = set(available_tools)
    selected: list[str] = []
    for field in missing_fields(evidence):
        tool = FIELD_TOOL_MAP.get(field)
        if tool in allowed and tool not in selected:
            selected.append(tool)
        if len(selected) >= max_calls:
            break
    return selected[:max_calls]

def simulated_tool_execution(case: dict[str, Any], tool: str) -> dict[str, Any]:
    """Shared frozen simulator contract, with hidden evaluator data isolated here."""
    if tool not in (case.get("available_tools") or []):
        raise ValueError("Tool not available")
    if tool == case.get("expected_tool"):
        return copy.deepcopy(case.get("tool_result")) or {
            "id": "TOOL-EMPTY-" + tool, "status": "no_additional_evidence"
        }
    return {"id": "TOOL-NORESULT-" + tool, "status": "no_additional_evidence"}

def run_deterministic_retrieval(case: dict[str, Any], max_calls: int = 1) -> dict[str, Any]:
    evidence = copy.deepcopy(case["evidence"])
    before = deterministic_assessment(evidence)
    tools = select_tools(evidence, list(case.get("available_tools") or []), max_calls)
    events: list[dict[str, Any]] = []
    for tool in tools:
        result = simulated_tool_execution(case, tool)
        events.append({"tool": tool, "result": result})
        if result.get("status") != "no_additional_evidence":
            evidence = merge_tool_result(evidence, result)
    after = deterministic_assessment(evidence)
    return {
        "selected_action": after["selected_action"],
        "deterministic_before": before,
        "deterministic_after": after,
        "requested_tools": tools,
        "tool_events": events,
        "missing_fields_before": missing_fields(case["evidence"]),
        "missing_fields_after": missing_fields(evidence),
        "llm_calls": 0,
        "llm_tokens": 0,
    }
