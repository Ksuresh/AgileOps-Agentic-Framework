from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable

from openai import OpenAI

ACTIONS = [
    "Rollback to stable deployment",
    "Block release and fix pipeline",
    "Mitigate and monitor",
    "Scale adjustment",
    "Review scaling policy",
    "Patch or block release",
    "No action (observe)",
    "Escalate for evidence/human review",
]

DOMAIN_BLOCKS = {
    "DevOps": "deploy",
    "SRE": "sre",
    "FinOps": "finops",
    "DevSecOps": "sec",
}

DOMAIN_TOOLS = {
    "DevOps": {"get_restart_history", "get_pipeline_history", "get_deployment_history"},
    "SRE": {"get_error_rate_history", "get_latency_history", "get_saturation_history", "get_restart_history"},
    "FinOps": {"get_replica_history", "get_cost_proxy_history"},
    "DevSecOps": {"get_policy_gate_status", "get_vulnerability_details"},
}

EVALUATOR_ONLY = {
    "id", "stratum", "focus", "oracle_domains", "admissible_actions", "expected_agentic",
    "expected_tool", "tool_result", "misleading_signal", "notes",
}

AGENT_SYSTEM = """You are one bounded operational domain agent in a project-governance experiment.
Assess only your assigned operational domain using the incident evidence supplied to you.
You may request read-only evidence tools from the allowed list when information needed for a justified assessment is missing.
Never invent telemetry, causal links, incidents, business impact, vulnerabilities, deployment history, or tool results.
Never use knowledge outside the supplied incident evidence and tool results.
Return only strict JSON with these keys:
agent_type, claim, confidence, evidence_ids, proposed_action, uncertainty, needs_more_evidence, requested_tools, rationale_summary.
confidence and uncertainty must be numbers in [0,1]. proposed_action must be one of the supplied allowed actions.
evidence_ids must contain only IDs visible in your assigned-domain evidence or tool results. requested_tools must contain only supplied allowed tools.
rationale_summary must be a concise audit rationale, not private chain-of-thought.
If evidence is insufficient, state that and request the smallest relevant tool set rather than guessing.
"""

COORDINATOR_SYSTEM = """You are a bounded multi-agent coordinator for a project-governance experiment.
You receive structured outputs from domain agents. Select one governance action from the supplied allowed-action list based only on the cited evidence and agent outputs.
Do not invent evidence or infer hidden labels. Treat each domain-agent proposal as advisory.
Return only strict JSON with keys selected_action, confidence, supporting_agents, evidence_ids, rationale_summary.
rationale_summary must be concise and auditable, not private chain-of-thought.
"""


def _json_obj(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("No JSON object in model output")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Model output is not a JSON object")
    return value


def _usage(response: Any) -> dict[str, int]:
    u = getattr(response, "usage", None)
    return {
        "input_tokens": int(getattr(u, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(u, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(u, "total_tokens", 0) or 0),
    }


def _sum_usage(items: Iterable[dict[str, int]]) -> dict[str, int]:
    out = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for item in items:
        for key in out:
            out[key] += int(item.get(key, 0) or 0)
    return out


def strip_evaluator_fields(case: dict[str, Any]) -> dict[str, Any]:
    return {k: copy.deepcopy(v) for k, v in case.items() if k not in EVALUATOR_ONLY}


def _status(block: dict[str, Any], field: str) -> str:
    meta = (block.get("_evidence", {}) or {}).get(field, {}) or {}
    if "status" in meta:
        return str(meta["status"])
    return "measured" if field in block and block.get(field) is not None else "missing"


def evidence_packet(evidence: dict[str, Any]) -> dict[str, Any]:
    """Convert raw case evidence into stable, citable model-visible evidence IDs."""
    packet: dict[str, Any] = {}
    for domain, block in evidence.items():
        if not isinstance(block, dict) or domain.startswith("_"):
            continue
        fields: dict[str, Any] = {}
        for field, value in block.items():
            if field.startswith("_"):
                continue
            eid = f"EV-{domain}-{field}"
            fields[field] = {"id": eid, "value": value, "status": _status(block, field)}
        packet[domain] = fields
    return packet


def domain_evidence_packet(evidence: dict[str, Any], agent_type: str) -> dict[str, Any]:
    """Expose only the assigned domain's raw evidence to a specialist agent."""
    full = evidence_packet(evidence)
    block = DOMAIN_BLOCKS[agent_type]
    return {block: copy.deepcopy(full.get(block, {}))}


def packet_ids(packet: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for block in packet.values():
        if not isinstance(block, dict):
            continue
        for item in block.values():
            if isinstance(item, dict) and item.get("id"):
                ids.add(str(item["id"]))
    return ids


def merge_tool_result(evidence: dict[str, Any], tool_result: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(evidence)
    for domain, block in tool_result.items():
        if domain in {"id", "status"} or not isinstance(block, dict):
            continue
        dst = out.setdefault(domain, {})
        for field, value in block.items():
            dst[field] = value
            meta = dst.setdefault("_evidence", {}).setdefault(field, {})
            meta.update({"status": "measured", "source": str(tool_result.get("id", "tool_result"))})
    return out


@dataclass
class AgentCall:
    output: dict[str, Any]
    raw: str
    usage: dict[str, int]
    latency_ms: float
    valid_evidence_refs: bool
    schema_valid: bool
    call_type: str
    agent_type: str | None = None

    def audit_record(self) -> dict[str, Any]:
        return asdict(self)


def _in_unit_interval(value: Any) -> bool:
    try:
        x = float(value)
        return 0.0 <= x <= 1.0
    except (TypeError, ValueError):
        return False


class BoundedAgenticRuntime:
    def __init__(self, model: str, reasoning_effort: str = "none"):
        self.client = OpenAI()
        self.model = model
        self.reasoning_effort = reasoning_effort

    def _call(self, instructions: str, payload: dict[str, Any]) -> tuple[dict[str, Any], str, dict[str, int], float]:
        t0 = time.perf_counter()
        response = self.client.responses.create(
            model=self.model,
            reasoning={"effort": self.reasoning_effort},
            instructions=instructions,
            input=json.dumps(payload, ensure_ascii=False, sort_keys=True),
        )
        latency = (time.perf_counter() - t0) * 1000.0
        raw = response.output_text or ""
        return _json_obj(raw), raw, _usage(response), latency

    def domain_agent(
        self,
        agent_type: str,
        evidence: dict[str, Any],
        available_tools: list[str],
        tool_results_seen: list[dict[str, Any]] | None = None,
    ) -> AgentCall:
        packet = domain_evidence_packet(evidence, agent_type)
        allowed = sorted(set(available_tools) & DOMAIN_TOOLS.get(agent_type, set()))
        payload = {
            "agent_type": agent_type,
            "domain_block": DOMAIN_BLOCKS[agent_type],
            "incident_evidence": packet,
            "tool_results_seen": tool_results_seen or [],
            "allowed_tools": allowed,
            "allowed_actions": ACTIONS,
        }
        try:
            obj, raw, usage, latency = self._call(AGENT_SYSTEM, payload)
            required = {
                "agent_type", "claim", "confidence", "evidence_ids", "proposed_action", "uncertainty",
                "needs_more_evidence", "requested_tools", "rationale_summary",
            }
            requested = [str(x) for x in (obj.get("requested_tools") or [])]
            schema_valid = bool(
                required.issubset(obj)
                and obj.get("agent_type") == agent_type
                and obj.get("proposed_action") in ACTIONS
                and _in_unit_interval(obj.get("confidence"))
                and _in_unit_interval(obj.get("uncertainty"))
                and isinstance(obj.get("evidence_ids"), list)
                and isinstance(obj.get("requested_tools"), list)
                and set(requested).issubset(set(allowed))
            )
            known = packet_ids(packet)
            for tr in tool_results_seen or []:
                tid = tr.get("id")
                if tid:
                    known.add(str(tid))
            refs = {str(x) for x in (obj.get("evidence_ids") or [])}
            valid_refs = refs.issubset(known)
            return AgentCall(obj, raw, usage, latency, valid_refs, schema_valid, "domain_agent", agent_type)
        except Exception as exc:
            return AgentCall(
                {"agent_type": agent_type, "_error": type(exc).__name__, "claim": "model_call_failed"},
                str(exc), {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, 0.0, False, False,
                "domain_agent", agent_type,
            )

    def coordinator(self, agent_outputs: list[dict[str, Any]], known_evidence_ids: set[str]) -> AgentCall:
        payload = {"agent_outputs": agent_outputs, "allowed_actions": ACTIONS}
        try:
            obj, raw, usage, latency = self._call(COORDINATOR_SYSTEM, payload)
            required = {"selected_action", "confidence", "supporting_agents", "evidence_ids", "rationale_summary"}
            schema_valid = bool(
                required.issubset(obj)
                and obj.get("selected_action") in ACTIONS
                and _in_unit_interval(obj.get("confidence"))
                and isinstance(obj.get("supporting_agents"), list)
                and isinstance(obj.get("evidence_ids"), list)
            )
            refs = {str(x) for x in (obj.get("evidence_ids") or [])}
            return AgentCall(obj, raw, usage, latency, refs.issubset(known_evidence_ids), schema_valid, "coordinator", None)
        except Exception as exc:
            return AgentCall(
                {"_error": type(exc).__name__, "selected_action": "Escalate for evidence/human review"},
                str(exc), {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, 0.0, False, False,
                "coordinator", None,
            )


def total_call_usage(calls: Iterable[AgentCall]) -> dict[str, int]:
    return _sum_usage(call.usage for call in calls)
