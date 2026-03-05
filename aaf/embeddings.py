# aaf/embeddings.py
from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "sentence-transformers is required for embeddings. "
        "Install it via: pip install sentence-transformers"
    ) from e


@lru_cache(maxsize=1)
def _get_model(model_name: str) -> SentenceTransformer:
    """
    Cache the embedding model so we don't reload it per scenario.
    """
    return SentenceTransformer(model_name)


def embed_claims(claims: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Embed a list of textual claims into vectors.

    Returns:
        embeddings: shape (len(claims), dim) as float32
    """
    model = _get_model(model_name)
    emb = model.encode(
        claims,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return emb.astype(np.float32)


def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Cosine similarity mapped to [0, 1].
    """
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)

    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0.0:
        return 0.0

    cos = float(np.dot(v1, v2) / denom)
    cos = max(-1.0, min(1.0, cos))
    sim01 = (cos + 1.0) / 2.0
    return max(0.0, min(1.0, sim01))
