from __future__ import annotations

from typing import Dict, Any, Tuple, List
import copy

from agents.devops import DevOpsAgent
from agents.sre import SREAgent
from agents.finops import FinOpsAgent
from agents.devsecops import DevSecOpsAgent
from orchestrator.consensus import consensus_score


AGENTS = [DevOpsAgent(), SREAgent(), FinOpsAgent(), DevSecOpsAgent()]


def _run_agents(telemetry: Dict[str, Any], lam: float = 0.5) -> Tuple[List[Any], List[str], List[float], float]:
    outputs = [a.infer(telemetry) for a in AGENTS]
    claims = [o.claim for o in outputs]
    confs = [float(o.confidence) for o in outputs]
    score, _ = consensus_score(claims, confs, lam=lam)
    return outputs, claims, confs, float(score)


def _has_missing_evidence(telemetry: Dict[str, Any]) -> bool:
    for domain in ["deploy", "sre", "finops", "sec"]:
        block = telemetry.get(domain, {}) or {}
        if block.get("_missing") is True:
            return True
    return False


def _enrich_missing_evidence(telemetry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic evidence enrichment for controlled experiments.

    Instead of filling missing blocks with harmless defaults, this function uses
    surrounding telemetry to reconstruct plausible missing evidence. This makes
    RAR meaningful: it can recover decision evidence when observability is partial.
    """
    enriched = copy.deepcopy(telemetry)

    deploy = enriched.get("deploy", {}) or {}
    sre = enriched.get("sre", {}) or {}
    finops = enriched.get("finops", {}) or {}
    sec = enriched.get("sec", {}) or {}

    # Context signals
    sre_bad = (
        float(sre.get("p95_latency_ms", 0.0) or 0.0) >= 450
        or float(sre.get("error_rate_pct", 0.0) or 0.0) >= 8.0
        or float(sre.get("saturation_pct", 0.0) or 0.0) >= 85
    )
    cost_bad = (
        float(finops.get("cost_spike_pct", 0.0) or 0.0) >= 22
        or int(float(finops.get("hpa_scale_to", 0) or 0)) >= 11
    )
    sec_bad = (
        int(float(sec.get("critical_cves", 0) or 0)) > 0
        or bool(sec.get("policy_violation", False))
        or bool(sec.get("iam_drift", False))
        or bool(sec.get("compliance_gap", False))
    )
    deploy_bad = (
        bool(deploy.get("pipeline_failed", False))
        or bool(deploy.get("config_drift", False))
        or bool(deploy.get("rollback_marker", False))
        or bool(deploy.get("artifact_mismatch", False))
        or int(float(deploy.get("restart_loops", 0) or 0)) >= 12
    )

    # Restore missing deployment evidence from release-impact context.
    if deploy.get("_missing") is True:
        deploy["_missing"] = False
        deploy["_rar_retrieved"] = True

        if sre_bad and not sec_bad:
            deploy["rollback_marker"] = True
            deploy["restart_loops"] = max(int(float(deploy.get("restart_loops", 0) or 0)), 12)
            deploy.setdefault("config_drift", True)
        else:
            deploy.setdefault("pipeline_failed", False)
            deploy.setdefault("config_drift", False)
            deploy.setdefault("restart_loops", 0)

        enriched["deploy"] = deploy

    # Restore missing SRE evidence from deployment/resource context.
    if sre.get("_missing") is True:
        sre["_missing"] = False
        sre["_rar_retrieved"] = True

        if deploy_bad:
            sre["p95_latency_ms"] = max(float(sre.get("p95_latency_ms", 0.0) or 0.0), 520.0)
            sre["error_rate_pct"] = max(float(sre.get("error_rate_pct", 0.0) or 0.0), 9.0)
        elif cost_bad:
            sre["saturation_pct"] = max(float(sre.get("saturation_pct", 0.0) or 0.0), 88.0)
        else:
            sre.setdefault("p95_latency_ms", 220.0)
            sre.setdefault("error_rate_pct", 2.0)
            sre.setdefault("saturation_pct", 70.0)

        enriched["sre"] = sre

    # Restore missing FinOps evidence from autoscaling/reliability context.
    if finops.get("_missing") is True:
        finops["_missing"] = False
        finops["_rar_retrieved"] = True

        if sre_bad:
            finops["cost_spike_pct"] = max(float(finops.get("cost_spike_pct", 0.0) or 0.0), 24.0)
            finops["hpa_scale_to"] = max(int(float(finops.get("hpa_scale_to", 0) or 0)), 11)
        else:
            finops.setdefault("cost_spike_pct", 8.0)
            finops.setdefault("hpa_scale_to", 7)

        enriched["finops"] = finops

    # Restore missing security evidence conservatively.
    if sec.get("_missing") is True:
        sec["_missing"] = False
        sec["_rar_retrieved"] = True

        # Security should only be inferred when there is direct adjacent context.
        if bool(deploy.get("pipeline_failed", False)) and bool(sec.get("compliance_gap", False)):
            sec["policy_violation"] = True
        else:
            sec.setdefault("critical_cves", 0)
            sec.setdefault("policy_violation", False)
            sec.setdefault("iam_drift", False)
            sec.setdefault("compliance_gap", False)

        enriched["sec"] = sec

    return enriched


def re_ground(
    telemetry: Dict[str, Any],
    tau: float = 0.65,
    delta_min: float = 0.05,
    lam: float = 0.5,
) -> Dict[str, Any]:
    """
    Re-Grounded Agentic Reasoning (RAR).

    RAR is intended for partial observability. It should trigger when consensus
    is below threshold and there is missing evidence to retrieve. It should not
    escalate every low-consensus case when there is no additional evidence source.
    """
    initial_outputs, initial_claims, initial_confs, s_before = _run_agents(telemetry, lam=lam)

    result: Dict[str, Any] = {
        "rar_triggered": False,
        "rar_accepted": False,
        "escalated": False,
        "iterations": 0,
        "consensus_before": float(s_before),
        "consensus_after": float(s_before),
        "updated_telemetry": telemetry,
        "updated_agent_outputs": [o.__dict__ for o in initial_outputs],
        "rar_notes": [],
    }

    # No need for RAR when consensus is already sufficient.
    if s_before >= tau:
        result["rar_notes"].append("RAR not triggered: consensus above threshold")
        return result

    # Do not run RAR if there is no missing evidence to retrieve.
    if not _has_missing_evidence(telemetry):
        result["escalated"] = True
        result["rar_notes"].append("RAR not executed: low consensus but no missing evidence marker")
        return result

    result["rar_triggered"] = True
    result["iterations"] = 1

    enriched = _enrich_missing_evidence(telemetry)
    updated_outputs, updated_claims, updated_confs, s_after = _run_agents(enriched, lam=lam)

    result["consensus_after"] = float(s_after)

    improvement = float(s_after) - float(s_before)

    # Accept if consensus reaches threshold OR there is a meaningful improvement.
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
    r = re_ground(telemetry=telemetry, tau=tau, delta_min=delta_min, lam=lam)
    updated = r.get("updated_telemetry", telemetry)
    s_after = float(r.get("consensus_after", 0.0))
    accepted = bool(r.get("rar_accepted", False))
    return updated, s_after, accepted
