from __future__ import annotations
from typing import Dict, Any, Tuple, List
import copy
import time

from orchestrator.consensus import consensus_score


def _extract_claims_and_confidences(telemetry: Dict[str, Any]) -> Tuple[List[str], List[float]]:
    """
    Extract (claim, confidence) for each domain from the telemetry envelope.
    Expected minimal shape per domain:
        telemetry["deploy"]["claim"], telemetry["deploy"]["confidence"]
        telemetry["sre"]["claim"], telemetry["sre"]["confidence"]
        telemetry["finops"]["claim"], telemetry["finops"]["confidence"]
        telemetry["sec"]["claim"], telemetry["sec"]["confidence"]

    Falls back to empty/neutral defaults if fields are missing.
    """
    domains = ["deploy", "sre", "finops", "sec"]
    claims: List[str] = []
    confidences: List[float] = []

    for d in domains:
        block = telemetry.get(d, {}) or {}
        claims.append(str(block.get("claim", f"{d}:no-claim")))
        try:
            confidences.append(float(block.get("confidence", 0.5)))
        except Exception:
            confidences.append(0.5)

    return claims, confidences


def re_ground(
    telemetry: Dict[str, Any],
    tau: float = 0.75,
    delta_min: float = 0.15,
    lam: float = 0.5
) -> Dict[str, Any]:
    """
    Re-Grounded Agentic Reasoning (RAR), consistent with the manuscript.

    Trigger:
        S(I) < tau

    Acceptance:
        S'(I) - S(I) >= delta_min

    Behavior:
      - Enriches missing/low-evidence fields deterministically (synthetic testbed).
      - Recomputes consensus after enrichment.
      - If improvement is insufficient, escalates (no low-confidence narrative).
      - Returns RAR overhead latency in ms for trade-off reporting.
    """
    result: Dict[str, Any] = {
        "rar_triggered": False,
        "rar_accepted": False,
        "escalated": False,
        "iterations": 0,
        "latency_ms": 0.0,
        "consensus_before": None,
        "consensus_after": None,
        "updated_telemetry": telemetry,
        "rar_notes": []
    }

    # Consensus before
    claims, confs = _extract_claims_and_confidences(telemetry)
    s_before, _ = consensus_score(claims, confs, lam=lam)
    result["consensus_before"] = s_before

    if s_before >= tau:
        return result  # no RAR needed

    result["rar_triggered"] = True
    start = time.time()

    enriched = copy.deepcopy(telemetry)
    result["iterations"] = 1

    # --- Evidence enrichment (synthetic testbed) ---
    # In your prototype, "missing evidence" is signaled via `_missing: true`.
    # RAR simulates retrieval by filling evidence blocks and clearing `_missing`.
    for k in ["deploy", "sre", "finops", "sec"]:
        block = enriched.get(k, {}) or {}

        if block.get("_missing") is True:
            block["_missing"] = False
            block["_rar_note"] = "Additional evidence retrieved during RAR"
            # Provide a deterministic evidence stub so downstream explanation can cite it
            block.setdefault("evidence", {})
            block["evidence"].setdefault("source", "synthetic-evidence-store")
            block["evidence"].setdefault("items", [])
            block["evidence"]["items"].append(
                {"type": "rar_retrieval", "detail": f"Evidence refreshed for domain={k}"}
            )
            enriched[k] = block
            result["rar_notes"].append(f"{k}: evidence refreshed")

    # Consensus after
    claims2, confs2 = _extract_claims_and_confidences(enriched)
    s_after, _ = consensus_score(claims2, confs2, lam=lam)
    result["consensus_after"] = s_after

    result["latency_ms"] = (time.time() - start) * 1000.0

    # Accept / Escalate
    if (s_after - s_before) >= delta_min:
        result["rar_accepted"] = True
        result["updated_telemetry"] = enriched
    else:
        result["escalated"] = True
        # Keep updated_telemetry as original to avoid generating overconfident narratives
        result["updated_telemetry"] = telemetry

    return result
