from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from benchmark.oracle_specs import get_oracle
from evaluation.statistical_methods import wilson_interval
from pipeline import run_pipeline
from scenario_generator.generate import generate_scenarios


DEFAULT_SEEDS = (42, 43, 44, 45)


def _normalize_domain(value: Any) -> str:
    return str(value or "").strip().lower()


def _evaluate_case(case: Dict[str, Any], seed: int, mode: str) -> Dict[str, Any]:
    result = run_pipeline(case, mode=mode)  # type: ignore[arg-type]
    oracle = get_oracle(case["scenario_id"])
    predicted_domain = result.predicted_primary_domain
    predicted_action = result.utility.get("selected_action")

    domain_match = _normalize_domain(predicted_domain) == _normalize_domain(oracle["causal_domain"])
    action_match = predicted_action in oracle["admissible_actions"]

    return {
        "case_id": case["scenario_id"],
        "seed": seed,
        "condition": oracle["condition"],
        "oracle_causal_domain": oracle["causal_domain"],
        "oracle_affected_domains": oracle["affected_domains"],
        "oracle_admissible_actions": oracle["admissible_actions"],
        "oracle_basis": oracle["basis"],
        "predicted_primary_domain": predicted_domain,
        "recommended_action": predicted_action,
        "domain_oracle_match": domain_match,
        "action_oracle_match": action_match,
        "consensus_score": result.consensus_score,
        "rer_triggered": bool(result.rar.get("triggered")),
        "rer_accepted": bool(result.rar.get("accepted")) if result.rar.get("triggered") else False,
        "decision_policy": result.utility.get("selection_method", "unknown"),
        "label_source": "pre_specified_experimental_oracle",
    }


def run_benchmark(seeds: Iterable[int] = DEFAULT_SEEDS, mode: str = "aaf_full") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for case in generate_scenarios(seed=seed):
            rows.append(_evaluate_case(case, seed, mode))
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    domain_successes = sum(bool(r["domain_oracle_match"]) for r in rows)
    action_successes = sum(bool(r["action_oracle_match"]) for r in rows)
    domain_ci = wilson_interval(domain_successes, n)
    action_ci = wilson_interval(action_successes, n)
    triggered = sum(bool(r["rer_triggered"]) for r in rows)
    accepted = sum(bool(r["rer_accepted"]) for r in rows)
    return {
        "n": n,
        "base_conditions": len({r["case_id"] for r in rows}),
        "seeds": sorted({int(r["seed"]) for r in rows}),
        "evaluation_reference": "pre_specified_experimental_oracle",
        "domain_oracle_agreement": domain_successes / n if n else 0.0,
        "domain_oracle_wilson_95": domain_ci,
        "action_oracle_agreement": action_successes / n if n else 0.0,
        "action_oracle_wilson_95": action_ci,
        "rer_trigger_rate": triggered / n if n else 0.0,
        "rer_acceptance_when_triggered": accepted / triggered if triggered else 0.0,
        "interpretation": (
            "Agreement is measured against pre-specified controlled-experiment oracles. "
            "It must not be interpreted as production accuracy or independently observed real-world ground truth."
        ),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    cooked = []
    for row in rows:
        r = dict(row)
        r["oracle_affected_domains"] = json.dumps(r["oracle_affected_domains"])
        r["oracle_admissible_actions"] = json.dumps(r["oracle_admissible_actions"])
        cooked.append(r)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cooked[0].keys()))
        writer.writeheader()
        writer.writerows(cooked)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the controlled AAF benchmark against pre-specified experimental oracles.")
    parser.add_argument("--out", default="results_revision_oracle_benchmark")
    parser.add_argument("--seeds", default="42,43,44,45")
    parser.add_argument("--mode", default="aaf_full", choices=["aaf_full", "aaf_no_consensus", "aaf_no_rar", "aaf_no_utility"])
    args = parser.parse_args()

    seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())
    rows = run_benchmark(seeds=seeds, mode=args.mode)
    summary = summarize(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "case_results.csv", rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
