from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from benchmark.reviewer_oracle_specs import get_reviewer_oracle
from evaluation.statistical_methods import wilson_interval
from pipeline import run_pipeline
from scenario_generator.reviewer_aligned import generate_reviewer_aligned_scenarios

DEFAULT_SEEDS = (42, 43, 44, 45, 46)


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _row(case: Dict[str, Any], seed: int, mode: str) -> Dict[str, Any]:
    result = run_pipeline(case, mode=mode)  # type: ignore[arg-type]
    oracle = get_reviewer_oracle(case["scenario_id"])
    pred_domain = result.predicted_primary_domain
    pred_action = result.utility.get("selected_action")
    expected_domain = oracle.get("causal_domain")
    return {
        "case_id": case["scenario_id"],
        "seed": seed,
        "group": case.get("experiment_group", "unknown"),
        "condition": oracle["condition"],
        "oracle_causal_domain": expected_domain,
        "oracle_affected_domains": oracle["affected_domains"],
        "oracle_admissible_actions": oracle["admissible_actions"],
        "oracle_basis": oracle["basis"],
        "predicted_primary_domain": pred_domain,
        "recommended_action": pred_action,
        "domain_oracle_match": _norm(pred_domain) == _norm(expected_domain),
        "action_oracle_match": pred_action in oracle["admissible_actions"],
        "consensus_score": result.consensus_score,
        "rer_triggered": bool(result.rar.get("triggered")),
        "rer_accepted": bool(result.rar.get("accepted")) if result.rar.get("triggered") else False,
        "rer_escalated": bool(result.rar.get("escalated")) if result.rar.get("triggered") else False,
        "decision_policy": result.utility.get("selection_method", "unknown"),
        "label_source": "frozen_reviewer_aligned_oracle",
    }


def run(seeds: Iterable[int], mode: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        cases = generate_reviewer_aligned_scenarios(seed=seed)
        if len(cases) != 54:
            raise RuntimeError(f"Expected 54 frozen base cases, got {len(cases)}")
        for case in cases:
            rows.append(_row(case, seed, mode))
    return rows


def _summary_for(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    d = sum(bool(r["domain_oracle_match"]) for r in rows)
    a = sum(bool(r["action_oracle_match"]) for r in rows)
    trig = sum(bool(r["rer_triggered"]) for r in rows)
    acc = sum(bool(r["rer_accepted"]) for r in rows)
    esc = sum(bool(r["rer_escalated"]) for r in rows)
    return {
        "n": n,
        "base_cases": len({r["case_id"] for r in rows}),
        "domain_agreement": d / n if n else 0.0,
        "domain_wilson_95": wilson_interval(d, n),
        "action_agreement": a / n if n else 0.0,
        "action_wilson_95": wilson_interval(a, n),
        "rer_trigger_rate": trig / n if n else 0.0,
        "rer_acceptance_when_triggered": acc / trig if trig else 0.0,
        "rer_escalation_when_triggered": esc / trig if trig else 0.0,
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group"])].append(row)
    return {
        "evaluation_reference": "frozen_reviewer_aligned_oracles",
        "overall": _summary_for(rows),
        "by_group": {group: _summary_for(group_rows) for group, group_rows in sorted(grouped.items())},
        "interpretation": (
            "Primary efficacy must be interpreted by experiment group. G1 is a control; G2/G3 test cross-domain governance; "
            "G4 tests evidence insufficiency/escalation; G5 tests healthy abstention. Aggregate accuracy alone is not the primary claim."
        ),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    cooked = []
    for row in rows:
        r = dict(row)
        r["oracle_affected_domains"] = json.dumps(r["oracle_affected_domains"])
        r["oracle_admissible_actions"] = json.dumps(r["oracle_admissible_actions"])
        cooked.append(r)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cooked[0].keys()))
        writer.writeheader(); writer.writerows(cooked)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen 54-case reviewer-aligned AAF benchmark.")
    parser.add_argument("--out", default="results_reviewer_aligned")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--mode", default="aaf_full", choices=["aaf_full", "aaf_no_consensus", "aaf_no_rar", "aaf_no_utility"])
    args = parser.parse_args()
    seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())
    rows = run(seeds=seeds, mode=args.mode)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "case_results.csv", rows)
    summary = summarize(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
