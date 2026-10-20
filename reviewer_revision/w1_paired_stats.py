"""Paired sign-flip test and paired bootstrap for case-level continuous outcomes.

Development utility; freeze permutation count, seed, missingness policy and
failure handling before confirmatory runs. No model runs here.
"""
from __future__ import annotations
import math
import random
from statistics import mean
from typing import Sequence


def paired_inference(
    arm_a: Sequence[float],
    arm_b: Sequence[float],
    *,
    seed: int = 20261020,
    n_permutations: int = 99999,
    n_bootstrap: int = 10000,
) -> dict:
    if len(arm_a) != len(arm_b) or len(arm_a) < 2:
        raise ValueError("paired arms require equal length >= 2")
    if n_permutations < 1 or n_bootstrap < 100:
        raise ValueError("invalid resampling budget")
    if not all(math.isfinite(float(v)) for v in (*arm_a, *arm_b)):
        raise ValueError("nonfinite or missing outcomes: handle prospectively")
    diffs = [float(a) - float(b) for a, b in zip(arm_a, arm_b)]
    n = len(diffs)
    observed = mean(diffs)
    rng = random.Random(seed)
    # Conditional sign-flip randomization on paired differences.
    # Exact enumeration when <=16 nonzero differences; otherwise Monte Carlo
    # with +1 numerator/denominator adjustment to avoid zero p-values.
    nonzero = [x for x in diffs if x != 0]
    k = len(nonzero)
    if k <= 16:
        extreme = 0
        total = 1 << k
        for mask in range(total):
            val = sum(x if (mask >> j) & 1 else -x for j, x in enumerate(nonzero)) / n
            extreme += abs(val) >= abs(observed) - 1e-12
        p_value = extreme / total
        method = "exact_sign_flip"
    else:
        extreme = 0
        for _ in range(n_permutations):
            val = sum(x if rng.getrandbits(1) else -x for x in nonzero) / n
            extreme += abs(val) >= abs(observed) - 1e-12
        p_value = (extreme + 1) / (n_permutations + 1)
        method = "monte_carlo_sign_flip"
    boots = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_bootstrap)
    )
    lo = boots[int(0.025 * n_bootstrap)]
    hi = boots[min(n_bootstrap - 1, math.ceil(0.975 * n_bootstrap) - 1)]
    return {
        "n_cases": n,
        "paired_mean_difference": observed,
        "percentile_bootstrap_95ci": [lo, hi],
        "two_sided_sign_flip_p": p_value,
        "test": method,
        "n_nonzero_pairs": k,
        "bootstrap_draws": n_bootstrap,
        "permutation_draws": 0 if method == "exact_sign_flip" else n_permutations,
        "seed": seed,
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    if not p_values or any(not 0 <= p <= 1 for p in p_values):
        raise ValueError("p-values must be in [0,1]")
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    result = [0.0] * len(p_values)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[idx]))
        result[idx] = running
    return result
