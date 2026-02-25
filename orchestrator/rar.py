from __future__ import annotations
from typing import Dict, Any, Callable
import copy
import time


def re_ground(
    telemetry: Dict[str, Any],
    compute_consensus: Callable[[Dict[str, Any]], float],
    tau: float,
    delta_min: float
) -> Dict[str, Any]:
    """
    Re-Grounded Agentic Reasoning (RAR)

    Trigger condition:
        S(I) < tau

    Acceptance condition:
        S'(I) - S(I) >= delta_min

    If not satisfied, escalation flag is returned.
    """

    result = {
        "rar_triggered": False,
        "rar_accepted": False,
        "escalated": False,
        "latency_ms": 0.0,
        "updated_telemetry": telemetry,
        "consensus_before": None,
        "consensus_after": None
    }

    # Compute initial consensus
    s_before = compute_consensus(telemetry)
    result["consensus_before"] = s_before

    if s_before >= tau:
        # No RAR needed
        return result

    # RAR Triggered
    result["rar_triggered"] = True
    start_time = time.time()

    enriched = copy.deepcopy(telemetry)

    # Simulate evidence enrichment
    for domain in ["deploy", "sre", "finops", "sec"]:
        if enriched.get(domain, {}).get("_missing"):
            enriched[domain]["_missing"] = False
            enriched[domain]["_rar_note"] = "Additional evidence retrieved during RAR"

    # Recompute consensus after enrichment
    s_after = compute_consensus(enriched)
    result["consensus_after"] = s_after

    latency = (time.time() - start_time) * 1000
    result["latency_ms"] = latency

    # Accept only if improvement >= delta_min
    if (s_after - s_before) >= delta_min:
        result["rar_accepted"] = True
        result["updated_telemetry"] = enriched
    else:
        result["escalated"] = True

    return result
