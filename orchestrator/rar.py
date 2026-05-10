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


def _run_agents(
    telemetry: Dict[str, Any],
    lam: float = 0.5,
) -> Tuple[List[Any], List[str], List[float], float]:
    outputs = [agent.infer(telemetry) for agent in AGENTS]
    claims = [output.claim for output in outputs]
    confidences = [float(output.confidence) for output in outputs]
    score, _ = consensus_score(claims, confidences, lam=lam)
    return outputs, claims, confidences, float(score)


def _missing_domains(telemetry: Dict[str, Any]) -> List[str]:
    missing = []
    for domain in DOMAIN_KEYS:
        block = telemetry.get(domain, {}) or {}
        if block.get("_missing") is True:
            missing.append(domain)
    return missing


def _context_flags(telemetry: Dict[str, Any]) -> Dict[str, bool]:
    deploy = telemetry.get("deploy", {}) or {}
    sre = telemetry.get("sre", {}) or {}
    finops = telemetry.get("finops", {}) or {}
    sec = telemetry.get("sec", {}) or {}

    deploy_bad = (
        bool(deploy.get("pipeline_failed", False))
        or bool(deploy.get("config_drift", False))
        or bool(deploy.get("rollback_marker", False))
        or bool(deploy.get("artifact_mismatch", False))
        or int(float(deploy.get("restart_loops", 0) or 0)) >= 12
    )

    sre_bad = (
        float(sre.get("p95_latency_ms", 0.0) or 0.0) >= 450
        or float(sre.get("error_rate_pct", 0.0) or 0.0) >= 8.0
        or float(sre.get("saturation_pct", 0.0) or 0.0) >= 85
        or float(sre.get("availability_pct", 99.9) or 99.9) < 99.0
    )

    cost_bad = (
        float(finops.get("cost_spike_pct", 0.0) or 0.0) >= 22
        or int(float(finops.get("hpa_scale_to", 0) or 0)) >= 11
        or float(finops.get("cpu_request_increase_pct", 0.0) or 0.0) >= 50
        or float(finops.get("memory_request_increase_pct", 0.0) or 0.0) >= 40
    )

    sec_bad = (
        int(float(sec.get("critical_cves", 0) or 0)) > 0
        or bool(sec.get("policy_violation", False))
        or bool(sec.get("iam_drift", False))
        or bool(sec.get("compliance_gap", False))
    )

    return {
        "deploy_bad": deploy_bad,
        "sre_bad": sre_bad,
        "cost_bad": cost_bad,
        "sec_bad": sec_bad,
    }


def _enrich_missing_evidence(telemetry: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Controlled evidence retrieval used for the reproducible experiment.

    This function simulates tool/MCP retrieval by filling missing telemetry
    using adjacent evidence. It does not invent arbitrary success.
    It only enriches domains that were explicitly marked as missing.
    """
    enriched = copy.deepcopy(telemetry)
    notes: List[str] = []
    flags = _context_flags(enriched)

    deploy = enriched.get("deploy", {}) or {}
    sre = enriched.get("sre", {}) or {}
    finops = enriched.get("finops", {}) or {}
    sec = enriched.get("sec", {}) or {}

    if deploy.get("_missing") is True:
        deploy["_missing"] = False
        deploy["_rar_retrieved"] = True

        if flags["sre_bad"] and not flags["sec_bad"]:
            deploy["rollback_marker"] = True
            deploy["restart_loops"] = max(int(float(deploy.get("restart_loops", 0) or 0)), 12)
            deploy["config_drift"] = bool(deploy.get("config_drift", True))
            notes.append("Recovered deployment evidence from reliability-impact context")
        else:
            deploy.setdefault("pipeline_failed", False)
            deploy.setdefault("config_drift", False)
            deploy.setdefault("restart_loops", 0)
            notes.append("Recovered deployment evidence; no deployment anomaly confirmed")

        enriched["deploy"] = deploy

    if sre.get("_missing") is True:
        sre["_missing"] = False
        sre["_rar_retrieved"] = True

        if flags["deploy_bad"]:
            sre["p95_latency_ms"] = max(float(sre.get("p95_latency_ms", 0.0) or 0.0), 520.0)
            sre["error_rate_pct"] = max(float(sre.get("error_rate_pct", 0.0) or 0.0), 9.0)
            sre["availability_pct"] = min(float(sre.get("availability_pct", 99.9) or 99.9), 98.5)
            notes.append("Recovered SRE evidence from deployment-impact context")
        elif flags["cost_bad"]:
            sre["saturation_pct"] = max(float(sre.get("saturation_pct", 0.0) or 0.0), 88.0)
            notes.append("Recovered SRE evidence from scaling/cost context")
        else:
            sre.setdefault("p95_latency_ms", 220.0)
            sre.setdefault("error_rate_pct", 2.0)
            sre.setdefault("saturation_pct", 70.0)
            sre.setdefault("availability_pct", 99.9)
            notes.append("Recovered SRE evidence; no reliability anomaly confirmed")

        enriched["sre"] = sre

    if finops.get("_missing") is True:
        finops["_missing"] = False
        finops["_rar_retrieved"] = True

        if flags["sre_bad"]:
            finops["cost_spike_pct"] = max(float(finops.get("cost_spike_pct", 0.0) or 0.0), 24.0)
            finops["hpa_scale_to"] = max(int(float(finops.get("hpa_scale_to", 0) or 0)), 11)
            notes.append("Recovered FinOps evidence from reliability scaling context")
        else:
            finops.setdefault("cost_spike_pct", 8.0)
            finops.setdefault("hpa_scale_to", 7)
            notes.append("Recovered FinOps evidence; no material cost anomaly confirmed")

        enriched["finops"] = finops

    if sec.get("_missing") is True:
        sec["_missing"] = False
        sec["_rar_retrieved"] = True

        # Security retrieval is intentionally conservative.
        # We only strengthen it if security indicators already exist or
        # compliance context is present.
        if flags["sec_bad"]:
            sec["policy_violation"] = bool(sec.get("policy_violation", True))
            notes.append("Recovered security evidence from policy/compliance context")
        else:
            sec.setdefault("critical_cves", 0)
            sec.setdefault("policy_violation", False)
            sec.setdefault("iam_drift", False)
            sec.setdefault("compliance_gap", False)
            notes.append("Recovered security evidence; no security anomaly confirmed")

        enriched["sec"] = sec

    return enriched, notes


def re_ground(
    telemetry: Dict[str, Any],
    tau: float = 0.65,
    delta_min: float = 0.05,
    lam: float = 0.5,
) -> Dict[str, Any]:
    initial_outputs, initial_claims, initial_confs, s_before = _run_agents(telemetry, lam=lam)

    result: Dict[str, Any] = {
        "rar_triggered": False,
        "rar_accepted": False,
        "escalated": False,
        "iterations": 0,
        "consensus_before": float(s_before),
        "consensus_after": float(s_before),
        "missing_domains": _missing_domains(telemetry),
        "evidence_added": [],
        "updated_telemetry": telemetry,
        "updated_agent_outputs": [o.__dict__ for o in initial_outputs],
        "rar_notes": [],
    }

    if s_before >= tau:
        result["rar_notes"].append("RAR not triggered: consensus above threshold")
        return result

    missing = _missing_domains(telemetry)

    if not missing:
        result["escalated"] = True
        result["rar_notes"].append("RAR not executed: low consensus but no missing evidence marker")
        return result

    result["rar_triggered"] = True
    result["iterations"] = 1

    enriched, notes = _enrich_missing_evidence(telemetry)
    updated_outputs, updated_claims, updated_confs, s_after = _run_agents(enriched, lam=lam)

    result["consensus_after"] = float(s_after)
    result["evidence_added"] = notes

    improvement = float(s_after) - float(s_before)

    if s_after >= tau or improvement >= delta_min:
        result["rar_accepted"] = True
        result["updated_telemetry"] = enriched
        result["updated_agent_outputs"] = [o.__dict__ for o in updated_outputs]
        result["rar_notes"].append(
            f"RAR accepted: consensus changed from {s_before:.3f} to {s_after:.3f}"
        )
    else:
        result["escalated"] = True
        result["rar_notes"].append(
            f"RAR escalation: consensus changed from {s_before:.3f} to {s_after:.3f}, below acceptance rule"
        )

    return result


def re_ground_telemetry(
    telemetry: Dict[str, Any],
    tau: float = 0.65,
    delta_min: float = 0.05,
    lam: float = 0.5,
) -> Tuple[Dict[str, Any], float, bool]:
    result = re_ground(telemetry=telemetry, tau=tau, delta_min=delta_min, lam=lam)
    updated = result.get("updated_telemetry", telemetry)
    s_after = float(result.get("consensus_after", 0.0))
    accepted = bool(result.get("rar_accepted", False))
    return updated, s_after, accepted
