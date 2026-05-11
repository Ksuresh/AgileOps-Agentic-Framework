"""Reproducible experiment runner for the AAF paper."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List

from scenario_generator.generate import generate_scenarios
from pipeline import run_pipeline

from experiments.scoring import (
    score_primary_domain_accuracy,
    score_action_match,
    compute_latency_stats,
    compute_rar_stats,
    compute_utility_stats,
    compute_utility_component_stats,
    compute_xi_stats,
    compute_consensus_stats,
)

from baselines.traditional import run_traditional_baseline
from baselines.single_agent_llm import run_single_agent_llm_baseline


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _dump_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summarize_baseline(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "accuracy": score_primary_domain_accuracy(rows),
        "action_match": score_action_match(rows),
    }


def _summarize_aaf(rows: List[Dict[str, Any]], include_rar: bool = False) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "accuracy": score_primary_domain_accuracy(rows),
        "action_match": score_action_match(rows),
        "consensus": compute_consensus_stats(rows),
        "utility": compute_utility_stats(rows),
        "utility_components": compute_utility_component_stats(rows),
        "xi": compute_xi_stats(rows),
    }

    if include_rar:
        summary["rar"] = compute_rar_stats(rows)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results", help="Output folder")
    parser.add_argument("--n", type=int, default=30, help="Number of scenarios")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    args = parser.parse_args()

    _ensure_dir(args.out)

    # The generator returns the original 30 controlled scenarios by default.
    # When --n is greater than 30, target_n expands the controlled scenario set
    # using deterministic severity and cross-domain variants.
    scenarios_all: List[Dict[str, Any]] = generate_scenarios(
        seed=args.seed,
        noise={
            "missing_evidence_prob": 0.20,
            "contradiction_prob": 0.10,
            "metric_jitter_pct": 0.05,
            "target_n": max(args.n, 30),
        },
    )
    scenarios = scenarios_all[: args.n]

    if len(scenarios) < args.n:
        raise ValueError(
            f"Requested {args.n} scenarios, but generator returned only {len(scenarios)}."
        )

    # Persist scenario definitions for reproducibility.
    with open(os.path.join(args.out, "scenarios.json"), "w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=2)

    # Baselines
    trad_rows = run_traditional_baseline(scenarios, seed=args.seed)
    llm_rows = run_single_agent_llm_baseline(scenarios, seed=args.seed)

    # AAF full and ablations
    aaf_rows: List[Dict[str, Any]] = []
    aaf_no_consensus: List[Dict[str, Any]] = []
    aaf_no_rar: List[Dict[str, Any]] = []
    aaf_no_utility: List[Dict[str, Any]] = []

    for scenario in scenarios:
        aaf_rows.append(asdict(run_pipeline(scenario, mode="aaf_full")))
        aaf_no_consensus.append(asdict(run_pipeline(scenario, mode="aaf_no_consensus")))
        aaf_no_rar.append(asdict(run_pipeline(scenario, mode="aaf_no_rar")))
        aaf_no_utility.append(asdict(run_pipeline(scenario, mode="aaf_no_utility")))

    # Persist raw rows.
    _dump_jsonl(os.path.join(args.out, "traditional.jsonl"), trad_rows)
    _dump_jsonl(os.path.join(args.out, "single_agent_llm.jsonl"), llm_rows)
    _dump_jsonl(os.path.join(args.out, "aaf_full.jsonl"), aaf_rows)
    _dump_jsonl(os.path.join(args.out, "aaf_no_consensus.jsonl"), aaf_no_consensus)
    _dump_jsonl(os.path.join(args.out, "aaf_no_rar.jsonl"), aaf_no_rar)
    _dump_jsonl(os.path.join(args.out, "aaf_no_utility.jsonl"), aaf_no_utility)

    # Aggregate metrics.
    summary = {
        "experiment": {
            "n_requested": args.n,
            "n_evaluated": len(scenarios),
            "seed": args.seed,
            "scenario_source": "controlled_scenario_variants",
            "note": (
                "Expanded scenarios are deterministic controlled variants, "
                "not independent production incidents."
            ),
        },
        "parameters": {
            "missing_evidence_prob": 0.20,
            "contradiction_prob": 0.10,
            "metric_jitter_pct": 0.05,
            "target_n": max(args.n, 30),
        },
        "traditional": _summarize_baseline(trad_rows),
        "single_agent_llm": _summarize_baseline(llm_rows),
        "aaf_full": {
            **_summarize_aaf(aaf_rows, include_rar=True),
            "latency": compute_latency_stats(aaf_rows),
        },
        "aaf_no_consensus": _summarize_aaf(aaf_no_consensus),
        "aaf_no_rar": _summarize_aaf(aaf_no_rar, include_rar=True),
        "aaf_no_utility": _summarize_aaf(aaf_no_utility),
    }

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    from experiments.report_tables import write_tables

    write_tables(args.out, summary)

    print(f"Wrote results to: {args.out}")
    print(f"Evaluated scenarios: {len(scenarios)}")


if __name__ == "__main__":
    main()
