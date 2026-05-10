from __future__ import annotations

from typing import Dict, List
import re


def compute_xi(explanation: str, payload: Dict) -> Dict[str, float]:
    r = _readability_norm(explanation)
    e = _evidence_clarity(explanation, payload)
    t = _traceability(explanation, payload)
    xi = 0.4 * r + 0.4 * e + 0.2 * t
    return {
        "readability": round(r, 4),
        "evidence_clarity": round(e, 4),
        "traceability": round(t, 4),
        "xi": round(xi, 4),
    }


def _readability_norm(text: str) -> float:
    sentences = [s.strip() for s in re.split(r"[.!?]\s+|\n+", text) if s.strip()]
    if not sentences:
        return 0.0

    words = [w for w in re.findall(r"[A-Za-z0-9']+", text)]
    avg_len = len(words) / max(1, len(sentences))

    # Penalize very short template-like explanations and very long sentences.
    if avg_len < 6:
        return 0.72

    score = 1.0 - abs(avg_len - 14.0) / 22.0
    return max(0.55, min(0.95, score))


def _evidence_phrases(payload: Dict) -> List[str]:
    phrases: List[str] = []
    for agent in payload.get("agents", []):
        for ev in agent.get("evidence") or []:
            if isinstance(ev, str) and ev.strip():
                phrases.append(ev.strip().lower())
    return phrases


def _evidence_clarity(text: str, payload: Dict) -> float:
    phrases = _evidence_phrases(payload)
    if not phrases:
        return 0.0

    text_l = text.lower()

    full_hits = sum(1 for ev in phrases if ev in text_l)

    # Evidence labels matter, but should not automatically give full credit.
    evidence_label_count = text_l.count("evidence:")
    evidence_label_score = min(1.0, evidence_label_count / max(4, len(payload.get("agents", []))))

    phrase_score = full_hits / len(phrases)

    # Weighted combination: actual evidence matching matters most.
    score = 0.70 * phrase_score + 0.30 * evidence_label_score

    return max(0.0, min(0.92, score))


def _traceability(text: str, payload: Dict) -> float:
    agent_types = [a.get("agent_type") for a in payload.get("agents", []) if a.get("agent_type")]
    if not agent_types:
        return 0.0

    text_l = text.lower()

    agent_score = sum(1 for a in agent_types if str(a).lower() in text_l) / len(agent_types)

    structure_items = [
        "what happened",
        "why it happened",
        "impact across domains",
        "recommended action",
        "confidence",
        "consensus",
    ]
    structure_score = sum(1 for item in structure_items if item in text_l) / len(structure_items)

    score = 0.55 * agent_score + 0.45 * structure_score

    return max(0.0, min(0.92, score))
