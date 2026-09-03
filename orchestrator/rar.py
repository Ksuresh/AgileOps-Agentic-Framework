from __future__ import annotations

from typing import Dict, Any, Tuple, List
import copy

from agents.devops import DevOpsAgent
from agents.sre import SREAgent
from agents.finops import FinOpsAgent
from agents.devsecops import DevSecOpsAgent
from orchestrator.consensus import consensus_score

AGENTS = [DevOpsAgent(), SREAgent(), FinOpsAgent(), DevSecOpsAgent()]
DOMAIN_KEYS = ["deploy", "sre", "finops", "sec"]


def _run_agents(telemetry: Dict[str, Any], lam: float = 0.5) -> Tuple[List[Any], List[str], List[float], float]:
    outputs = [agent.infer(telemetry) for agent in AGENTS]
    claims = [output.claim for output in outputs]
    confidences = [float(output.confidence) for output in outputs]
    score, _ = consensus_score(claims, confidences, lam=lam)
    return outputs, claims, confidences, float(score)


def _field_status(block: Dict[str, Any], field: str) -> str:
    meta = (block.get("_evidence") or {}).get(field)
    if isinstance(meta, dict):
        return str(meta.get("status", "missing")).lower()
    if field in block and block.get(field) is not None:
        return "provided"
    return "missing"


def _missing_domains(telemetry: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    for domain in DOMAIN_KEYS:
        block = telemetry.get(domain, {}) or {}
        if block.get("_missing") is True:
            missing.append(domain); continue
        evidence = block.get("_evidence")
        if isinstance(evidence, dict) and evidence and all(str(v.get("status", "missing")).lower() in {"missing", "not_applicable"} for v in evidence.values() if isinstance(v, dict)):
            missing.append(domain)
    return missing


def _re_evaluate_available_evidence(telemetry: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """RER: re-evaluate only evidence already present with auditable provenance.

    No numeric or boolean operational value is fabricated. If evidence is
    missing, RER records the unresolved gap and leaves it missing. An external
    retrieval adapter may populate new measured/proxy evidence before this
    function is called; that evidence must carry source/status metadata.
    """
    enriched = copy.deepcopy(telemetry)
    notes: List[str] = []
    for domain in DOMAIN_KEYS:
        block = enriched.get(domain, {}) or {}
        meta = block.get("_evidence")
        if isinstance(meta, dict):
            usable = [field for field, item in meta.items() if isinstance(item, dict) and str(item.get("status", "missing")).lower() not in {"missing", "not_applicable"}]
            unresolved = [field for field, item in meta.items() if isinstance(item, dict) and str(item.get("status", "missing")).lower() == "missing"]
            if usable:
                notes.append(f"{domain}: re-evaluated {len(usable)} provenance-backed evidence field(s)")
            if unresolved:
                notes.append(f"{domain}: {len(unresolved)} evidence field(s) remain unresolved")
        elif block.get("_missing") is True:
            notes.append(f"{domain}: legacy domain-level evidence remains missing; no value synthesized")
    return enriched, notes


def re_ground(telemetry: Dict[str, Any], tau: float = 0.65, delta_min: float = 0.05, lam: float = 0.5) -> Dict[str, Any]:
    initial_outputs, _, _, s_before = _run_agents(telemetry, lam=lam)
    result: Dict[str, Any] = {
        "rar_triggered": False, "rar_accepted": False, "rer_triggered": False, "rer_accepted": False,
        "escalated": False, "iterations": 0, "consensus_before": float(s_before), "consensus_after": float(s_before),
        "missing_domains": _missing_domains(telemetry), "evidence_added": [], "updated_telemetry": telemetry,
        "updated_agent_outputs": [o.__dict__ for o in initial_outputs], "rar_notes": [], "rer_notes": [],
    }
    if s_before >= tau:
        result["rer_notes"].append("RER not triggered: consensus above threshold"); result["rar_notes"] = list(result["rer_notes"]); return result

    result["rer_triggered"] = True; result["rar_triggered"] = True; result["iterations"] = 1
    enriched, notes = _re_evaluate_available_evidence(telemetry)
    updated_outputs, _, _, s_after = _run_agents(enriched, lam=lam)
    result["consensus_after"] = float(s_after); result["updated_telemetry"] = enriched
    result["updated_agent_outputs"] = [o.__dict__ for o in updated_outputs]
    result["rer_notes"] = notes; result["rar_notes"] = list(notes)

    improvement = float(s_after) - float(s_before)
    # Re-evaluation alone is accepted only if it actually resolves the rule.
    # Otherwise the framework escalates for external evidence acquisition or human review.
    if s_after >= tau or improvement >= delta_min:
        result["rer_accepted"] = True; result["rar_accepted"] = True
        result["rer_notes"].append(f"RER accepted: consensus changed from {s_before:.3f} to {s_after:.3f}")
    else:
        result["escalated"] = True
        result["rer_notes"].append(f"RER escalation: no provenance-backed evidence resolved low consensus ({s_before:.3f} -> {s_after:.3f})")
    result["rar_notes"] = list(result["rer_notes"])
    return result


def re_ground_telemetry(telemetry: Dict[str, Any], tau: float = 0.65, delta_min: float = 0.05, lam: float = 0.5) -> Tuple[Dict[str, Any], float, bool]:
    result = re_ground(telemetry=telemetry, tau=tau, delta_min=delta_min, lam=lam)
    return result.get("updated_telemetry", telemetry), float(result.get("consensus_after", 0.0)), bool(result.get("rer_accepted", False))
