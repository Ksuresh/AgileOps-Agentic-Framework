from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from pipeline import run_pipeline
from scenario_generator.generate import generate_scenarios
from evaluation.statistical_methods import mcnemar_exact, wilson_interval


OUT_DIR = Path("results_revision_comparison")


def norm_action(action: str | None) -> str:
    a = str(action or "").lower()
    if "rollback" in a:
        return "rollback"
    if "patch" in a or "block" in a:
        return "patch_block"
    if "scale" in a:
        return "scale"
    if "mitigate" in a or "monitor" in a:
        return "mitigate_monitor"
    if "review" in a:
        return "review"
    if "observe" in a or "no action" in a:
        return "observe"
    return a.strip()


def cases_120() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for seed in (42, 43, 44, 45):
        for sc in generate_scenarios(seed=seed, noise={"missing_evidence_prob": 0.20, "metric_jitter_pct": 0.05}):
            c = copy.deepcopy(sc)
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
        expected = (case.get("ground_truth") or {}).get("expected_action")
        full_ok = norm_action(full.utility.get("selected_action")) == norm_action(expected)
        base_ok = norm_action(baseline.utility.get("selected_action")) == norm_action(expected)
        full_hits.append(full_ok)
        baseline_hits.append(base_ok)
        detail.append({
            "scenario_id": case.get("scenario_id"),
            "designer_reference_action": expected,
            "utility_action": full.utility.get("selected_action"),
            "baseline_action": baseline.utility.get("selected_action"),
            "utility_correct": full_ok,
            "baseline_correct": base_ok,
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
        "label_status": "designer_reference_labels_pending_independent_expert_adjudication",
        "utility_action_match": full_n / n,
        "utility_action_match_wilson_95": full_ci,
        "dominant_domain_baseline_action_match": base_n / n,
        "dominant_domain_baseline_wilson_95": base_ci,
        "mcnemar_b_utility_only_correct": b,
        "mcnemar_c_baseline_only_correct": c,
        "mcnemar_exact_two_sided_p": mcnemar_exact(b, c),
        "interpretation_warning": "Do not report as final accuracy until blinded expert labels are frozen.",
    }

    with (OUT_DIR / "paired_action_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail[0].keys()))
        writer.writeheader()
        writer.writerows(detail)
    (OUT_DIR / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
