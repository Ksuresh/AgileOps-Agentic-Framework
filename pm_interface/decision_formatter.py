from __future__ import annotations

from typing import Dict, Any, List


def _top_domains(agents: List[Dict[str, Any]]) -> List[str]:
    ranked = sorted(
        agents,
        key=lambda x: float(x.get("confidence", 0.0)),
        reverse=True,
    )
    domains = []
    for agent in ranked:
        claim = str(agent.get("claim", "")).lower()
        if claim.startswith("no "):
            continue
        domains.append(agent.get("agent_type", "Unknown"))
    return domains[:2] if domains else ["General Operations"]


def format_pm_decision(
    pm_prompt: str,
    route: Dict[str, Any],
    result: Dict[str, Any],
    explanation: str,
) -> str:
    top_domains = _top_domains(result.get("agents", []))
    decision = result.get("recommended_action", "No action (observe)")
    consensus = float(result.get("consensus_score", 0.0))
    rarity = "Yes" if result.get("rar_triggered", False) else "No"

    confidence = "Low"
    if consensus >= 0.75:
        confidence = "High"
    elif consensus >= 0.55:
        confidence = "Medium"

    return f"""PM Request:
{pm_prompt}

Recommended Governance Decision:
{decision}

Primary Domains Impacted:
- {chr(10).join(top_domains)}

Consensus Score:
{consensus:.2f}

RAR Triggered:
{rarity}

Confidence:
{confidence}

Why:
{explanation}
"""
