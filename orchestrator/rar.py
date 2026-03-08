from __future__ import annotations

from typing import Dict, Any, Tuple, List
import copy

from agents.devops import DevOpsAgent
from agents.sre import SREAgent
from agents.finops import FinOpsAgent
from agents.devsecops import DevSecOpsAgent
from orchestrator.consensus import consensus_score


AGENTS = [DevOpsAgent(), SREAgent(), FinOpsAgent(), DevSecOpsAgent()]


def _run_agents(telemetry: Dict[str, Any]) -> Tuple[List[Any], List[str], List[float], float]:
    """
    Run all agents on the current telemetry and compute consensus from
    their actual claims/confidences.
    """
    outputs = [a.infer(telemetry) for a in AGENTS]
    claims = [o.claim for o in outputs]
    confs = [float(o.confidence) for o in outputs]
    score, _ = consensus_score(claims, confs, lam=0.5)
    return outputs, claims, confs, float(score)


def _enrich_missing_evidence(telemetry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic synthetic evidence enrichment for reproducibility.
    This simulates retrieval of additional evidence in the synthetic testbed.
    """
    enriched = copy.deepcopy(telemetry)

    for domain in ["deploy", "sre", "finops", "sec"]:
        block = enriched.get(domain, {}) or {}

        if block.get("_missing") is True:
            block["_missing"] = False
            block["_rar_retrieved"] = True

            if domain == "deploy":
                block.setdefault("pipeline_failed", False)
                block.setdefault("config_drift", False)
                block.setdefault("restart_loops", 0)

            elif domain == "sre":
                block.setdefault("p95_latency_ms", 220.0)
                block.setdefault("error_rate_pct", 2.0)
                block.setdefault("saturation_pct", 70.0)

            elif domain == "finops":
                block.setdefault("cost_spike_pct", 8.0)
                block.setdefault("hpa_scale_to", 7)

            elif domain == "sec":
                block.setdefault("critical_cves", 0)
                block.setdefault("policy_violation", False)
                block.setdefault("iam_drift", False)

            enriched[domain] = block

    return enriched


def re_ground(
    telemetry: Dict[str, Any],
    tau: float = 0.75,
    delta_min: float = 0.15,
    lam: float = 0.5,
) -> Dict[str, Any]:
    """
    Re-Grounded Agentic Reasoning (RAR) for the synthetic reproducibility artifact.

    Behavior:
    - compute consensus from actual agent outputs
    - if consensus < tau, enrich missing telemetry deterministically
    - rerun agents and recompute consensus
    - accept if improvement >= delta_min
    """
    initial_outputs = [a.infer(telemetry) for a in AGENTS]
    initial_claims = [o.claim for o in initial_outputs]
    initial_confs = [float(o.confidence) for o in initial_outputs]
    s_before, _ = consensus_score(initial_claims, initial_confs, lam=lam)

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

    if s_before >= tau:
        return result

    result["rar_triggered"] = True
    result["iterations"] = 1

    enriched = _enrich_missing_evidence(telemetry)

    updated_outputs = [a.infer(enriched) for a in AGENTS]
    updated_claims = [o.claim for o in updated_outputs]
    updated_confs = [float(o.confidence) for o in updated_outputs]
    s_after, _ = consensus_score(updated_claims, updated_confs, lam=lam)

    result["consensus_after"] = float(s_after)

    if (s_after - s_before) >= delta_min:
        result["rar_accepted"] = True
        result["updated_telemetry"] = enriched
        result["updated_agent_outputs"] = [o.__dict__ for o in updated_outputs]
        result["rar_notes"].append("RAR accepted: consensus improved after evidence enrichment")
    else:
        result["escalated"] = True
        result["rar_notes"].append("RAR escalation: insufficient consensus improvement")

    return result


def re_ground_telemetry(
    telemetry: Dict[str, Any],
    tau: float = 0.75,
    delta_min: float = 0.15,
    lam: float = 0.5,
) -> Tuple[Dict[str, Any], float, bool]:
    """
    Convenience wrapper used by the pipeline.

    Returns:
      updated_telemetry, consensus_after, accepted
    """
    r = re_ground(telemetry=telemetry, tau=tau, delta_min=delta_min, lam=lam)
    updated = r.get("updated_telemetry", telemetry)
    s_after = float(r.get("consensus_after", 0.0))
    accepted = bool(r.get("rar_accepted", False))
    return updated, s_after, accepted
