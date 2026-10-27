"""Investigator fixes for Hybrid AAF (Study 8, pre-registered in external_validation/investigator_fix/PROTOCOL.md).

The frozen Hybrid (hybrid.py) is unchanged. Study 7 showed its selective trigger fires on unflagged evidence gaps but
its investigator does not look outside the domains that already show a material signal: it invokes only "relevant"
agents and each agent may request only its own domain's tools. Two fixes keep the same trigger and the same
deterministic final authority:

HY2-F  trigger-gated rule fetch: when the trigger fires, call every available approved read-only tool once, merge the
       results, recompute deterministically. No LLM in the investigation.
HY2-L  trigger-gated wide LLM investigator: when the trigger fires, invoke all four domain agents (reasoning effort set
       by the runtime), each still seeing only its own domain's evidence but allowed to request ANY available approved
       tool, with a short addendum on stale and absent evidence; every requested allowed tool is executed; the
       requesting agent is re-run on the results; the coordinator proposes; deterministic governance decides.
Tool semantics are unchanged: a tool returns the case's recorded result only if it is the evidence-bearing tool.
"""
from __future__ import annotations

import copy
from typing import Any

from agentic_experiments import agentic_runtime as AR
from agentic_experiments.hybrid import deterministic_assessment, selective_trigger

ADDENDUM = """
Additional investigation guidance:
- Evidence whose status is "stale" may be out of date. If your assessment depends on it, request the tool that refreshes it.
- If your domain has no current evidence at all, request the tool that would provide it.
- allowed_tools may include tools from other domains; request any of them when it could explain or rule out a cause of
  the incident. Request only tools from allowed_tools.
"""


def _tool_result(case: dict[str, Any], tool: str) -> dict[str, Any]:
    if tool == case.get("expected_tool"):
        return copy.deepcopy(case.get("tool_result"))
    return {"id": f"TOOL-NORESULT-{tool}", "status": "no_additional_evidence"}


def run_hybrid_fetch(case: dict[str, Any]) -> dict[str, Any]:
    evidence = copy.deepcopy(case["evidence"])
    det_before = deterministic_assessment(evidence)
    trigger = selective_trigger(evidence, det_before)
    tools: list[str] = []
    if trigger["invoke"]:
        for tool in case.get("available_tools") or []:
            tools.append(tool)
            if tool == case.get("expected_tool"):
                evidence = AR.merge_tool_result(evidence, _tool_result(case, tool))
    det_after = deterministic_assessment(evidence)
    return {"selected_action": str(det_after["selected_action"]), "trigger": trigger, "agentic_invoked": False,
            "investigation_invoked": bool(trigger["invoke"]), "tool_events": [{"tool": t} for t in tools],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, "call_audit": [], "latency_ms": 0.0}


class WideRuntimeMixin:
    """Domain agents may request any available tool; agent instructions get the addendum."""

    def domain_agent(self, agent_type, evidence, available_tools, tool_results_seen=None):  # noqa: D401
        packet = AR.domain_evidence_packet(evidence, agent_type)
        allowed = sorted(set(available_tools))
        payload = {"agent_type": agent_type, "domain_block": AR.DOMAIN_BLOCKS[agent_type], "incident_evidence": packet,
                   "tool_results_seen": tool_results_seen or [], "allowed_tools": allowed, "allowed_actions": AR.ACTIONS}
        try:
            obj, raw, usage, latency = self._call(AR.AGENT_SYSTEM + ADDENDUM, payload)
            required = {"agent_type", "claim", "confidence", "evidence_ids", "proposed_action", "uncertainty",
                        "needs_more_evidence", "requested_tools", "rationale_summary"}
            requested = [str(x) for x in (obj.get("requested_tools") or [])]
            schema_valid = bool(required.issubset(obj) and obj.get("agent_type") == agent_type and obj.get("proposed_action") in AR.ACTIONS
                                and AR._in_unit_interval(obj.get("confidence")) and AR._in_unit_interval(obj.get("uncertainty"))
                                and isinstance(obj.get("evidence_ids"), list) and isinstance(obj.get("requested_tools"), list)
                                and set(requested).issubset(set(allowed)))
            known = AR.packet_ids(packet) | {str(tr.get("id")) for tr in (tool_results_seen or []) if tr.get("id")}
            valid_refs = {str(x) for x in (obj.get("evidence_ids") or [])}.issubset(known)
            return AR.AgentCall(obj, raw, usage, latency, valid_refs, schema_valid, "domain_agent", agent_type)
        except Exception as exc:
            return AR.AgentCall({"agent_type": agent_type, "_error": type(exc).__name__, "claim": "model_call_failed"}, str(exc),
                                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, 0.0, False, False, "domain_agent", agent_type)


def run_hybrid_wide(case: dict[str, Any], runtime) -> dict[str, Any]:
    """`runtime` must be a WideRuntimeMixin runtime."""
    evidence = copy.deepcopy(case["evidence"])
    det_before = deterministic_assessment(evidence)
    trigger = selective_trigger(evidence, det_before)
    if not trigger["invoke"]:
        return {"selected_action": det_before["selected_action"], "trigger": trigger, "agentic_invoked": False, "tool_events": [],
                "agent_outputs": [], "call_audit": [], "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "latency_ms": 0.0, "governance_override": False}
    available = list(case.get("available_tools") or [])
    calls, outputs, tool_events, seen = [], [], [], set()
    for agent in ["DevOps", "SRE", "FinOps", "DevSecOps"]:
        call = runtime.domain_agent(agent, evidence, available)
        calls.append(call)
        requested = [t for t in (call.output.get("requested_tools") or []) if t in available and t not in seen]
        results = []
        for tool in requested:
            seen.add(tool)
            res = _tool_result(case, tool)
            tool_events.append({"agent": agent, "tool": tool, "result": res})
            results.append(res)
            if tool == case.get("expected_tool"):
                evidence = AR.merge_tool_result(evidence, res)
        if results:
            call = runtime.domain_agent(agent, evidence, available, results)
            calls.append(call)
        outputs.append(call.output)
    known = AR.packet_ids(AR.evidence_packet(evidence)) | {str(t["result"].get("id")) for t in tool_events if t["result"].get("id")}
    coord = runtime.coordinator(outputs, known)
    calls.append(coord)
    proposal = str(coord.output.get("selected_action", "Escalate for evidence/human review"))
    det_after = deterministic_assessment(evidence)
    final = str(det_after["selected_action"])
    return {"selected_action": final, "agent_proposal": proposal, "trigger": trigger, "agentic_invoked": True, "agent_outputs": outputs,
            "tool_events": tool_events, "usage": AR.total_call_usage(calls), "latency_ms": sum(x.latency_ms for x in calls),
            "all_schema_valid": all(x.schema_valid for x in calls), "all_evidence_refs_valid": all(x.valid_evidence_refs for x in calls),
            "governance_override": final != proposal, "call_audit": [x.audit_record() for x in calls]}
