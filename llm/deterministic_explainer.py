from __future__ import annotations

from typing import Any, Dict, List


NEGATIVE_MARKERS = [
    "no material",
    "no anomaly",
    "no issue",
    "within expected",
]


def _is_negative_claim(claim: str) -> bool:
    c = (claim or "").lower()
    return any(m in c for m in NEGATIVE_MARKERS)


def _select_primary_agent(agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    active = []
    for agent in agents:
        claim = str(agent.get("claim", ""))
        if not _is_negative_claim(claim):
            active.append(agent)

    candidates = active if active else agents
    return max(candidates, key=lambda a: float(a.get("confidence", 0.0) or 0.0))


def generate_explanation(payload: Dict[str, Any]) -> str:
    incident_id = payload.get("incident_id", "unknown")
    agents: List[Dict[str, Any]] = payload.get("agents", []) or []
    consensus = float(payload.get("consensus_score", 0.0) or 0.0)
    rar_triggered = bool(payload.get("rar_triggered", False))
    action = payload.get("recommended_action", "No action available")
    utility = float(payload.get("utility_score", 0.0) or 0.0)

    primary = _select_primary_agent(agents)
    primary_agent = primary.get("agent_type", "Unknown Agent")
    primary_claim = primary.get("claim", "No claim available")

    confidence_label = "High" if consensus >= 0.75 else "Medium" if consensus >= 0.55 else "Low"

    lines = []

    lines.append("1. What Happened:")
    lines.append(f"{primary_agent}: {primary_claim}")
    lines.append("")

    lines.append("2. Why It Happened:")
    for agent in agents:
        agent_type = agent.get("agent_type", "Unknown Agent")
        claim = agent.get("claim", "No claim provided")
        confidence = float(agent.get("confidence", 0.0) or 0.0)
        evidence = agent.get("evidence", []) or []

        lines.append(f"{agent_type}: {claim}. Confidence={confidence:.2f}.")
        for ev in evidence[:3]:
            lines.append(f"{agent_type} evidence: {ev}.")
    lines.append("")

    lines.append("3. Impact Across Domains:")
    for agent in agents:
        agent_type = agent.get("agent_type", "Unknown Agent")
        evidence = agent.get("evidence", []) or []
        if evidence:
            lines.append(f"{agent_type} impact evidence: {evidence[0]}.")
    lines.append("")

    lines.append("4. Recommended Action:")
    lines.append(
        f"{action}. This recommendation was selected using the utility model "
        f"with utility score {utility:.2f}."
    )
    lines.append("")

    lines.append("5. Confidence Level:")
    lines.append(
        f"{confidence_label} confidence. Consensus score={consensus:.2f}. "
        f"Re-grounding was {'triggered' if rar_triggered else 'not triggered'}."
    )
    lines.append("")

    lines.append("6. Project Manager Interpretation:")
    lines.append(
        "Review the listed evidence, confirm delivery impact with the owning team, "
        "and proceed with the recommended governance action if it aligns with release priorities."
    )

    return "\n".join(lines)
