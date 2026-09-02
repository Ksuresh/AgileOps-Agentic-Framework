from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from benchmark.oracle_specs import get_oracle
from pipeline import run_pipeline
from scenario_generator.generate import generate_scenarios
from evaluation.statistical_methods import wilson_interval


OUT_DIR = Path("results_revision_sensitivity")


def _evaluate(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    domain_hits: List[bool] = []
    action_hits: List[bool] = []
    consensus: List[float] = []
    rer_triggered: List[bool] = []
    for case in cases:
        r = run_pipeline(case, mode="aaf_full")
        oracle = get_oracle(case["oracle_case_id"])
        domain_hits.append(str(r.predicted_primary_domain or "").lower() == str(oracle["causal_domain"]).lower())
        action_hits.append(r.utility.get("selected_action") in set(oracle["admissible_actions"]))
        consensus.append(float(r.consensus_score))
        rer_triggered.append(bool(r.rar.get("triggered")))
    n = len(cases)
    d_lo, d_hi = wilson_interval(sum(domain_hits), n)
    a_lo, a_hi = wilson_interval(sum(action_hits), n)
    return {
        "n": n,
        "evaluation_reference": "pre_specified_experimental_oracle",
        "domain_oracle_agreement": sum(domain_hits) / n,
        "domain_ci_low": d_lo,
        "domain_ci_high": d_hi,
        "action_oracle_agreement": sum(action_hits) / n,
        "action_ci_low": a_lo,
        "action_ci_high": a_hi,
        "mean_consensus": sum(consensus) / n,
        "rer_trigger_rate": sum(rer_triggered) / n,
    }


def _cases() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seed in (42, 43, 44, 45):
        for sc in generate_scenarios(seed=seed, noise={"missing_evidence_prob": 0.20, "metric_jitter_pct": 0.05}):
            c = copy.deepcopy(sc)
            c["oracle_case_id"] = sc["scenario_id"]
            c["scenario_id"] = f"S{seed}-{sc['scenario_id']}"
            rows.append(c)
    return rows


def _weight_grid() -> List[Tuple[float, float, float]]:
    return [
        (0.40, 0.30, 0.30),
        (0.60, 0.20, 0.20),
        (0.20, 0.60, 0.20),
        (0.20, 0.20, 0.60),
        (0.50, 0.40, 0.10),
        (0.50, 0.10, 0.40),
        (0.10, 0.45, 0.45),
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = _cases()
    rows: List[Dict[str, Any]] = []

    def run(label: str, value: str, mutate) -> None:
        cases = copy.deepcopy(base)
        for c in cases:
            mutate(c)
        rows.append({"parameter": label, "value": value, **_evaluate(cases)})

    for lam in (0.25, 0.50, 0.75):
        run("lambda", str(lam), lambda c, x=lam: c.__setitem__("lam", x))

    for tau in (0.55, 0.65, 0.75, 0.85):
        def set_tau(c, x=tau):
            c.setdefault("thresholds", {})["tau_consensus"] = x
        run("tau_consensus", str(tau), set_tau)

    for delta in (0.00, 0.05, 0.10, 0.15):
        def set_delta(c, x=delta):
            c.setdefault("thresholds", {})["delta_min"] = x
        run("delta_min", str(delta), set_delta)

    for weights in _weight_grid():
        run("utility_weights", json.dumps(weights), lambda c, x=weights: c.__setitem__("utility_weights", x))

    fields = list(rows[0].keys())
    with (OUT_DIR / "sensitivity_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "sensitivity_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"configurations": len(rows), "evaluation_reference": "pre_specified_experimental_oracle"}, indent=2))


if __name__ == "__main__":
    main()
