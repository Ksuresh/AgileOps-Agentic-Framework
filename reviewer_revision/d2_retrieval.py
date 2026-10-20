"""D2 deterministic multi-tool evidence retrieval (exploratory implementation).

This is a stronger, budgeted non-LLM comparator, not a confirmatory result.
Selection depends only on case-visible evidence and the available tool catalog.
The legacy expected_tool simulator is used only AFTER selection; held-out
evaluation must replace it with independent per-tool responses.
"""
from __future__ import annotations

import copy
from typing import Any

from agentic_experiments.agentic_runtime import merge_tool_result
from agentic_experiments.hybrid import deterministic_assessment, missing_fields
from reviewer_revision.deterministic_retrieval import FIELD_TOOL_MAP, simulated_tool_execution

# Domain fallback is intentionally broad and deterministic; does not inspect oracle.
DOMAIN_FALLBACK: dict[str, tuple[str, ...]] = {
    "deploy": ("get_deployment_history", "get_pipeline_history", "get_restart_history"),
    "sre": ("get_error_rate_history", "get_latency_history", "get_saturation_history"),
    "finops": ("get_cost_proxy_history", "get_replica_history"),
    "sec": ("get_policy_gate_status", "get_vulnerability_details"),
}


def rank_tools(evidence: dict[str, Any], available_tools: list[str]) -> list[str]:
    """Rank exact mapped tools first, then domain fallbacks, with stable ties."""
    missing = missing_fields(evidence)
    available = set(available_tools)
    ranked: list[str] = []
    def add(tool: str) -> None:
        if tool in available and tool not in ranked:
            ranked.append(tool)
    for field in missing:
        mapped = FIELD_TOOL_MAP.get(field)
        if mapped:
            add(mapped)
    for domain in sorted({field.split(".", 1)[0] for field in missing}):
        for tool in DOMAIN_FALLBACK.get(domain, ()):
            add(tool)
    return ranked


def run_d2(case: dict[str, Any], max_calls: int = 2) -> dict[str, Any]:
    if max_calls < 0:
        raise ValueError("max_calls must be nonnegative")
    evidence = copy.deepcopy(case["evidence"])
    before = deterministic_assessment(evidence)
    events: list[dict[str, Any]] = []
    requested: list[str] = []
    for tool in rank_tools(evidence, list(case.get("available_tools") or [])):
        if len(requested) >= max_calls or not missing_fields(evidence):
            break
        result = simulated_tool_execution(case, tool)
        requested.append(tool)
        events.append({"tool": tool, "result": result})
        if result.get("status") != "no_additional_evidence":
            evidence = merge_tool_result(evidence, result)
    after = deterministic_assessment(evidence)
    return {
        "selected_action": after["selected_action"],
        "deterministic_before": before,
        "deterministic_after": after,
        "requested_tools": requested,
        "tool_events": events,
        "missing_fields_before": missing_fields(case["evidence"]),
        "missing_fields_after": missing_fields(evidence),
        "llm_calls": 0,
        "llm_tokens": 0,
    }
