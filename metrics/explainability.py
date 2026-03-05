"""Explainability Index (XI) computation.

XI = 0.4 R + 0.4 E + 0.2 T

This implementation is intentionally simple, deterministic, and self-contained.
It is meant to support repeatable experiments. If you later adopt a different
readability library, keep the normalization and weights consistent.
"""

from __future__ import annotations

from typing import Dict
import re


def compute_xi(explanation: str, payload: Dict) -> Dict[str, float]:
    r = _readability_norm(explanation)
    e = _evidence_clarity(explanation, payload)
    t = _traceability(explanation, payload)
    xi = 0.4 * r + 0.4 * e + 0.2 * t
    return {"readability": r, "evidence_clarity": e, "traceability": t, "xi": xi}


def _readability_norm(text: str) -> float:
    """Normalize a crude readability estimate to [0,1]."""
    # Average sentence length proxy: shorter tends to be easier.
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", text) if s.strip()]
    if not sentences:
        return 0.0
    words = [w for w in re.findall(r"[A-Za-z0-9']+", text)]
    avg_len = (len(words) / len(sentences)) if sentences else 50.0
    # Map avg sentence length 8..28 to 1..0 linearly, clamp.
    r = 1.0 - (avg_len - 8.0) / 20.0
    return max(0.0, min(1.0, r))


def _evidence_clarity(text: str, payload: Dict) -> float:
    """Fraction of sentences that include at least one evidence snippet."""
    evidence_phrases = []
    for a in payload.get("agents", []):
        for ev in a.get("evidence") or []:
            if ev and isinstance(ev, str):
                evidence_phrases.append(ev.strip())
    if not evidence_phrases:
        return 0.0

    sentences = [s.strip() for s in re.split(r"\n+", text) if s.strip()]
    if not sentences:
        return 0.0
    hits = 0
    for s in sentences:
        if any(ev[:20] in s for ev in evidence_phrases):
            hits += 1
    return hits / len(sentences)


def _traceability(text: str, payload: Dict) -> float:
    """Fraction of sections that explicitly reference an agent type."""
    agent_types = [a.get("agent_type") for a in payload.get("agents", [])]
    agent_types = [t for t in agent_types if t]
    if not agent_types:
        return 0.0
    sections = [s.strip() for s in text.split("\n\n") if s.strip()]
    if not sections:
        return 0.0
    traced = 0
    for sec in sections:
        if any(t in sec for t in agent_types):
            traced += 1
    return traced / len(sections)
