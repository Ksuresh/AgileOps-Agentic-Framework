from __future__ import annotations

from typing import Dict, List
import re


def compute_xi(explanation: str, payload: Dict) -> Dict[str, float]:
    r = _readability_norm(explanation)
    e = _evidence_clarity(explanation, payload)
    t = _traceability(explanation, payload)
    xi = 0.4 * r + 0.4 * e + 0.2 * t
    return {
        "readability": r,
        "evidence_clarity": e,
        "traceability": t,
        "xi": xi,
    }


def _readability_norm(text: str) -> float:
    sentences = [s.strip() for s in re.split(r"[.!?]\s+|\n+", text) if s.strip()]
    if not sentences:
        return 0.0

    words = [w for w in re.findall(r"[A-Za-z0-9']+", text)]
    avg_len = (len(words) / len(sentences)) if sentences else 50.0

    r = 1.0 - (avg_len - 8.0) / 22.0
    return max(0.0, min(1.0, r))


def _evidence_terms(payload: Dict) -> List[str]:
    terms: List[str] = []

    for agent in payload.get("agents", []):
        for ev in agent.get("evidence") or []:
            if isinstance(ev, str) and ev.strip():
                terms.append(ev.strip().lower())

                # Also include shorter token-level evidence anchors.
                for token in re.findall(r"[A-Za-z][A-Za-z\-]+", ev.lower()):
                    if len(token) >= 6:
                        terms.append(token)

    return sorted(set(terms))


def _evidence_clarity(text: str, payload: Dict) -> float:
    terms = _evidence_terms(payload)
    if not terms:
        return 0.0

    text_l = text.lower()

    hits = 0
    for term in terms:
        if term in text_l:
            hits += 1

    # Cap at 1.0. Requiring all token anchors is too strict, so normalize
    # against a moderate evidence target.
    target = min(8, max(3, len(terms) // 3))
    return max(0.0, min(1.0, hits / target))


def _traceability(text: str, payload: Dict) -> float:
    agent_types = [a.get("agent_type") for a in payload.get("agents", []) if a.get("agent_type")]
    if not agent_types:
        return 0.0

    text_l = text.lower()

    agent_hits = sum(1 for t in agent_types if str(t).lower() in text_l)
    score_agents = agent_hits / len(agent_types)

    has_consensus = "consensus" in text_l
    has_recommendation = "recommended action" in text_l or "recommendation" in text_l
    has_evidence_label = "evidence" in text_l

    structure_score = (
        (1.0 if has_consensus else 0.0)
        + (1.0 if has_recommendation else 0.0)
        + (1.0 if has_evidence_label else 0.0)
    ) / 3.0

    return 0.6 * score_agents + 0.4 * structure_score
