from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from benchmark.oracle_specs import get_oracle
from pipeline import run_pipeline
from scenario_generator.generate import generate_scenarios
from evaluation.statistical_methods import mcnemar_exact, wilson_interval


OUT_DIR = Path("results_revision_comparison")


def cases_120() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for seed in (42, 43, 44, 45):
        for sc in generate_scenarios(seed=seed, noise={"missing_evidence_prob": 0.20, "metric_jitter_pct": 0.05}):
            c = copy.deepcopy(sc)
            c["oracle_case_id"] = sc["scenario_id"]
            c["scenario_id"] = f"S{seed}-{sc['scenario_id']}"
            cases.append(c)
    return cases


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail: List[Dict[str, Any]] = []
    full_hits: List[bool] = []
    baseline_hits: List[bool] = []

    for case in cases_120():
        full = run_pipeline(copy.deepcopy(case), mode="aaf_full")
        baseline = run_pipeline(copy.deepcopy(case), mode="aaf_no_utility")
        oracle = get_oracle(case["oracle_case_id"])
        admissible = set(oracle["admissible_actions"])
        full_action = full.utility.get("selected_action")
        base_action = baseline.utility.get("selected_action")
        full_ok = full_action in admissible
        base_ok = base_action in admissible
        full_hits.append(full_ok)
        baseline_hits.append(base_ok)
        detail.append({
            "scenario_id": case.get("scenario_id"),
            "oracle_case_id": case["oracle_case_id"],
            "oracle_causal_domain": oracle["causal_domain"],
            "oracle_admissible_actions": json.dumps(oracle["admissible_actions"]),
            "utility_action": full_action,
            "baseline_action": base_action,
            "utility_oracle_match": full_ok,
            "baseline_oracle_match": base_ok,
        })

    n = len(detail)
    full_n = sum(full_hits)
    base_n = sum(baseline_hits)
    full_ci = wilson_interval(full_n, n)
    base_ci = wilson_interval(base_n, n)
    b = sum(f and not g for f, g in zip(full_hits, baseline_hits))
    c = sum((not f) and g for f, g in zip(full_hits, baseline_hits))

    summary = {
        "n": n,
        "evaluation_reference": "pre_specified_experimental_oracle",
        "utility_action_oracle_agreement": full_n / n,
        "utility_action_oracle_wilson_95": full_ci,
        "dominant_domain_baseline_action_oracle_agreement": base_n / n,
        "dominant_domain_baseline_wilson_95": base_ci,
        "mcnemar_b_utility_only_correct": b,
        "mcnemar_c_baseline_only_correct": c,
        "mcnemar_exact_two_sided_p": mcnemar_exact(b, c),
        "interpretation_warning": "Controlled-oracle agreement; do not interpret as production accuracy.",
    }

    with (OUT_DIR / "paired_action_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail[0].keys()))
        writer.writeheader()
        writer.writerows(detail)
    (OUT_DIR / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
