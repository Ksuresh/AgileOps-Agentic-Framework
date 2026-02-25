from __future__ import annotations
from typing import List, Tuple
from aaf.embeddings import embed_claims, cosine_sim

def confidence_alignment(a: float, b: float) -> float:
    return 1.0 - abs(a - b)

def consensus_score(claims: List[str], confidences: List[float], lam: float = 0.5) -> Tuple[float, List[List[float]]]:
    vecs = embed_claims(claims)
    n = len(claims)
    if n < 2:
        return 1.0, [[1.0]]

    total = 0.0
    cnt = 0
    pair = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            s = cosine_sim(vecs[i], vecs[j])
            g = confidence_alignment(confidences[i], confidences[j])
            v = lam*s + (1.0-lam)*g
            pair[i][j] = v
            total += v
            cnt += 1
    return total / cnt, pair
