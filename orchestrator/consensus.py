from __future__ import annotations

from typing import List, Tuple


NEGATIVE_MARKERS = [
    "no material",
    "no anomaly",
    "no issue",
    "no security",
    "no reliability",
    "no deployment",
    "no cost",
    "within expected",
]


PRIMARY_MARKERS = [
    "likely primary",
    "primary operational cause",
    "failure is the likely",
    "degradation is the likely",
    "issue is the likely",
]


POSSIBLE_MARKERS = [
    "possible",
    "indicates",
    "incomplete",
]


def _is_negative_claim(claim: str) -> bool:
    c = (claim or "").lower()
    return any(m in c for m in NEGATIVE_MARKERS)


def _is_primary_claim(claim: str) -> bool:
    c = (claim or "").lower()
    return any(m in c for m in PRIMARY_MARKERS)


def _is_possible_claim(claim: str) -> bool:
    c = (claim or "").lower()
    return any(m in c for m in POSSIBLE_MARKERS)


def confidence_alignment(a: float, b: float) -> float:
    return max(0.0, 1.0 - abs(float(a) - float(b)))


def consensus_score(
    claims: List[str],
    confidences: List[float],
    lam: float = 0.5,
) -> Tuple[float, List[List[float]]]:
    """
    Evidence-aware consensus for governance interpretation.

    The earlier version compared every claim semantically, including
    'no issue' claims. That made consensus artificially low in scenarios
    where only one or two domains were actually relevant.

    This version treats consensus as decision readiness:
    - strong primary evidence should produce high consensus
    - multiple competing primary claims should reduce consensus
    - missing/possible claims should produce moderate uncertainty
    - negative/no-issue claims should not dominate
    """

    n = len(claims)
    if n == 0:
        return 0.0, []
    if n == 1:
        return float(confidences[0]), [[1.0]]

    pair = [[0.0] * n for _ in range(n)]

    active = []
    primary = []
    possible = []

    for i, claim in enumerate(claims):
        conf = float(confidences[i])
        neg = _is_negative_claim(claim)

        if _is_primary_claim(claim) and not neg:
            primary.append((i, conf))
            active.append((i, conf))
        elif _is_possible_claim(claim) and not neg:
            possible.append((i, conf))
            active.append((i, conf))
        elif not neg and conf >= 0.45:
            active.append((i, conf))

    if not active:
        # All agents report no material issue. This is stable consensus.
        avg_conf = sum(float(c) for c in confidences) / len(confidences)
        score = min(0.80, max(0.60, avg_conf))
        for i in range(n):
            for j in range(n):
                pair[i][j] = 1.0 if i == j else score
        return score, pair

    primary_count = len(primary)
    active_conf = [conf for _, conf in active]
    avg_active_conf = sum(active_conf) / len(active_conf)

    # Clear single-primary case: enough for governance decision.
    if primary_count == 1:
        score = max(0.70, min(0.92, avg_active_conf + 0.10))

    # Two primary claims can be legitimate cross-domain evidence,
    # but it is less clear than one primary root cause.
    elif primary_count == 2:
        primary_confs = [conf for _, conf in primary]
        alignment = confidence_alignment(primary_confs[0], primary_confs[1])
        score = min(0.82, 0.55 + 0.25 * alignment)

    # More than two primary claims means competing root causes.
    elif primary_count > 2:
        score = 0.50

    # No primary, only possible/incomplete evidence.
    else:
        score = min(0.65, max(0.45, avg_active_conf))

    # Penalize incomplete evidence slightly.
    if possible:
        score -= min(0.10, 0.03 * len(possible))

    score = max(0.0, min(1.0, score))

    for i in range(n):
        for j in range(n):
            if i == j:
                pair[i][j] = 1.0
            else:
                pair[i][j] = score

    return score, pair
