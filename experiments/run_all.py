"""Reproducible experiment runner for the AAF paper.

This script is intentionally *self-contained* and deterministic:
- Uses the existing synthetic scenario generator (30 scenarios by default).
- Runs: (i) Traditional baseline (simulated PM protocol), (ii) Single-agent LLM
  baseline (deterministic summarizer), (iii) AAF full, (iv) Ablations.
- Writes per-scenario JSONL + aggregated CSV/Markdown tables.

Why deterministic?
The paper emphasizes fixed parameters, repeatability, and no learning.

Run:
  python -m experiments.run_all --out results
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Dict, Any, List

from scenario_generator.generate import generate_scenarios
from pipeline import run_pipeline

from experiments.scoring import (
    score_primary_domain_accuracy,
    compute_latency_stats,
    compute_rar_stats,
    compute_utility_stats,
    compute_xi_stats,
)
from baselines.traditional import run_traditional_baseline
from baselines.single_agent_llm import run_single_agent_llm_baseline


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results", help="Output folder")
    ap.add_argument("--n", type=int, default=30, help="Number of synthetic scenarios")
    ap.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    args = ap.parse_args()

    _ensure_dir(args.out)

    # The bundled generator produces 30 scenarios deterministically.
    scenarios_all: List[Dict[str, Any]] = generate_scenarios(
        seed=args.seed,
        noise={
            "missing_evidence_prob": 0.20,
            "contradiction_prob": 0.10,
            "metric_jitter_pct": 0.05,
        },
    )
    scenarios = scenarios_all[: args.n]

    # --- Run baselines
    trad_rows = run_traditional_baseline(scenarios, seed=args.seed)
    llm_rows = run_single_agent_llm_baseline(scenarios, seed=args.seed)

    # --- Run AAF (full + ablations)
    aaf_rows = []
    aaf_no_consensus = []
    aaf_no_rar = []
    aaf_no_utility = []

    for sc in scenarios:
        aaf_rows.append(asdict(run_pipeline(sc, mode="aaf_full")))
        aaf_no_consensus.append(asdict(run_pipeline(sc, mode="aaf_no_consensus")))
        aaf_no_rar.append(asdict(run_pipeline(sc, mode="aaf_no_rar")))
        aaf_no_utility.append(asdict(run_pipeline(sc, mode="aaf_no_utility")))

    # Persist raw rows (JSONL)
    def dump_jsonl(name: str, rows: List[Dict[str, Any]]) -> None:
        p = os.path.join(args.out, f"{name}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump_jsonl("traditional", trad_rows)
    dump_jsonl("single_agent_llm", llm_rows)
    dump_jsonl("aaf_full", aaf_rows)
    dump_jsonl("aaf_no_consensus", aaf_no_consensus)
    dump_jsonl("aaf_no_rar", aaf_no_rar)
    dump_jsonl("aaf_no_utility", aaf_no_utility)

    # --- Aggregate metrics
    summary = {
        "traditional": {
            "accuracy": score_primary_domain_accuracy(trad_rows),
        },
        "single_agent_llm": {
            "accuracy": score_primary_domain_accuracy(llm_rows),
        },
        "aaf_full": {
            "accuracy": score_primary_domain_accuracy(aaf_rows),
            "rar": compute_rar_stats(aaf_rows),
            "latency": compute_latency_stats(aaf_rows),
            "utility": compute_utility_stats(aaf_rows),
            "xi": compute_xi_stats(aaf_rows),
        },
        "aaf_no_consensus": {"accuracy": score_primary_domain_accuracy(aaf_no_consensus)},
        "aaf_no_rar": {"accuracy": score_primary_domain_accuracy(aaf_no_rar)},
        "aaf_no_utility": {"accuracy": score_primary_domain_accuracy(aaf_no_utility)},
    }

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    from experiments.report_tables import write_tables

    write_tables(args.out, summary)

    print(f"Wrote results to: {args.out}")


if __name__ == "__main__":
    main()
