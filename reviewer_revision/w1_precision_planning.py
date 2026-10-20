"""Prospective paired-proportion precision exploration, not a confirmatory analysis.

Simulates paired B4/B2 binary correctness under prespecified discordance rates.
The reported interval is an approximate normal paired-difference interval for
PLANNING ONLY; the confirmatory interval remains the preregistered Tango score CI.
No model outputs or confirmatory cases are accessed.
"""
from __future__ import annotations
import json
import math
import random
from statistics import mean, median

SEED = 20261020
N_SIM = 10000
N_VALUES = (80, 120)
DISCORDANCE = (0.05, 0.10, 0.15, 0.20)
TRUE_DIFFERENCE = (0.0, 0.05, 0.10)
Z = 1.959963984540054


def simulate(n: int, discordance: float, difference: float, rng: random.Random) -> dict:
    assert abs(difference) <= discordance
    plus = (discordance + difference) / 2
    minus = (discordance - difference) / 2
    widths, covers_zero = [], []
    for _ in range(N_SIM):
        diffs = []
        for _ in range(n):
            u = rng.random()
            diffs.append(1 if u < plus else (-1 if u < plus + minus else 0))
        d = mean(diffs)
        # Unbiased sample variance of paired {-1,0,1} differences.
        var = sum((x - d) ** 2 for x in diffs) / (n - 1)
        half = Z * math.sqrt(var / n)
        widths.append(half * 100)
        covers_zero.append((d - half) <= 0 <= (d + half))
    widths.sort()
    return {
        "n": n,
        "discordance": discordance,
        "true_difference": difference,
        "median_approx_95ci_halfwidth_pp": round(median(widths), 2),
        "p10_halfwidth_pp": round(widths[int(.10 * N_SIM)], 2),
        "p90_halfwidth_pp": round(widths[int(.90 * N_SIM)], 2),
        "fraction_intervals_containing_zero": round(mean(covers_zero), 3),
    }


def main() -> None:
    rng = random.Random(SEED)
    results = [simulate(n, q, d, rng) for n in N_VALUES
               for q in DISCORDANCE for d in TRUE_DIFFERENCE if d <= q]
    print(json.dumps({
        "status": "prospective_planning_simulation_only",
        "seed": SEED,
        "simulations_per_cell": N_SIM,
        "interval": "approximate paired normal CI; NOT confirmatory Tango score CI",
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
