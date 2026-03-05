"""Lightweight, fully local embeddings for claim similarity.

The paper uses semantic similarity between agent claims as part of consensus.
For repository reproducibility (no external APIs/models), we implement a
hashing-based bag-of-words embedding.

If you later want stronger semantics, replace `embed_claims` with a
sentence-transformer model, but keep the interface stable.
"""

from __future__ import annotations

from typing import List
import math
import re
import hashlib

_DIM = 256


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[A-Za-z0-9]+", (text or "").lower()) if t]


def _hash(token: str) -> int:
    h = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little")


def embed_claims(claims: List[str]) -> List[List[float]]:
    vecs: List[List[float]] = []
    for c in claims:
        v = [0.0] * _DIM
        toks = _tokenize(c)
        for t in toks:
            idx = _hash(t) % _DIM
            v[idx] += 1.0
        # L2 normalize
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        v = [x / norm for x in v]
        vecs.append(v)
    return vecs


def cosine_sim(a: List[float], b: List[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))
