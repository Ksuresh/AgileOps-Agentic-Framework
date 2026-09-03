from __future__ import annotations

"""One-at-a-time parameter sensitivity analysis for the controlled AAF evaluation.

Robustness analysis only: scenarios, seed, noise model, decision rules, and
labels remain fixed. One parameter family is varied at a time around the
pre-specified default; no held-out outcome is used for tuning.
"""

import copy
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from scenario_generator.generate import generate_scenarios
from pipeline import run_pipeline
from experiments.scoring import score_primary_domain_accuracy, score_action_match

OUT = Path("results_parameter_sensitivity_2026")
DEFAULT = {
    "lambda": 0.50,
    "tau": 0.65,
    "delta_min": 0.05,
    "utility_weights": (0.40, 0.30, 0.30),
}
GRIDS = {
    "lambda": [0.30, 0.40, 0.50, 0.60, 0.70],
    "tau": [0.55, 0.60, 0.65, 0.70, 0.75],
    "delta_min": [0.03, 0.05, 0.10, 0.15],
    "utility_weights": [
        (0.40, 0.30, 0.30),
        (0.50, 0.25, 0.25),
        (0.30, 0.35, 0.35),
        (0.35, 0.40, 0.25),
        (0.35, 0.25, 0.40),
        (0.45, 0.35, 0.20),
        (0.45, 0.20, 0.35),
    ],
}


def fixed_scenarios() -> List[Dict[str, Any]]:
    return generate_scenarios(seed=42, noise={
        "missing_evidence_prob": 0.20,
        "contradiction_prob": 0.10,
        "metric_jitter_pct": 0.05,
    })


def configured(base: List[Dict[str, Any]], family: str, value: Any) -> List[Dict[str, Any]]:
    out = copy.deepcopy(base)
    for sc in out:
        sc["lam"] = DEFAULT["lambda"]
        sc.setdefault("thresholds", {})["tau_consensus"] = DEFAULT["tau"]
        sc["thresholds"]["delta_min"] = DEFAULT["delta_min"]
        sc["thresholds"]["max_rar_loops"] = 2
        sc["utility_weights"] = DEFAULT["utility_weights"]
        if family == "lambda":
            sc["lam"] = float(value)
        elif family == "tau":
            sc["thresholds"]["tau_consensus"] = float(value)
        elif family == "delta_min":
            sc["thresholds"]["delta_min"] = float(value)
        elif family == "utility_weights":
            sc["utility_weights"] = tuple(float(x) for x in value)
        else:
            raise ValueError(family)
    return out


def run_rows(scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [asdict(run_pipeline(sc, mode="aaf_full")) for sc in scenarios]


def selected_action(row: Dict[str, Any]) -> str:
    return str(row.get("utility", {}).get("selected_action", ""))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = fixed_scenarios()
    default_rows = run_rows(configured(base, "lambda", DEFAULT["lambda"]))
    default_actions = [selected_action(r) for r in default_rows]
    results = []

    for family, values in GRIDS.items():
        for value in values:
            rows = run_rows(configured(base, family, value))
            actions = [selected_action(r) for r in rows]
            stability = sum(a == b for a, b in zip(default_actions, actions)) / len(actions)
            rar_rate = sum(bool(r.get("rar", {}).get("triggered")) for r in rows) / len(rows)
            domain_stats = score_primary_domain_accuracy(rows)
            action_stats = score_action_match(rows)
            results.append({
                "parameter_family": family,
                "value": "/".join(f"{x:.2f}" for x in value) if family == "utility_weights" else f"{float(value):.2f}",
                "n_scenarios": len(rows),
                "domain_match": round(float(domain_stats["accuracy"]), 6),
                "action_match": round(float(action_stats["action_match_rate"]), 6),
                "action_stability_vs_default": round(stability, 6),
                "rar_trigger_rate": round(rar_rate, 6),
            })

    with (OUT / "parameter_sensitivity.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    summary = {
        "design": "one-at-a-time robustness analysis on fixed deterministic controlled scenarios; not parameter tuning",
        "seed": 42,
        "n_scenarios": len(base),
        "default_parameters": {"lambda": 0.5, "tau": 0.65, "delta_min": 0.05, "utility_weights": [0.4, 0.3, 0.3]},
        "results": results,
    }
    (OUT / "parameter_sensitivity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
