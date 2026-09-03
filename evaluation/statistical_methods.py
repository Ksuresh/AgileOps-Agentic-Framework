from __future__ import annotations

import math
from typing import Iterable, Tuple


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) / n) + z2 / (4.0 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def cohens_kappa(labels_a: Iterable[str], labels_b: Iterable[str]) -> float:
    """Cohen's kappa for two categorical annotators."""
    a = [str(x) for x in labels_a]
    b = [str(x) for x in labels_b]
    if len(a) != len(b) or not a:
        raise ValueError("Annotator label vectors must have equal non-zero length")

    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    categories = sorted(set(a) | set(b))
    expected = 0.0
    for c in categories:
        pa = sum(x == c for x in a) / n
        pb = sum(x == c for x in b) / n
        expected += pa * pb
    if math.isclose(expected, 1.0):
        return 1.0
    return (observed - expected) / (1.0 - expected)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value using the binomial distribution.

    b: cases correct under A and incorrect under B
    c: cases incorrect under A and correct under B
    """
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)
