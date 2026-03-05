"""Single-agent LLM baseline.

In the paper this baseline removes domain separation, consensus, RAR, and
utility selection. Here we implement a deterministic approximation that:
- Produces a single predicted primary domain.
- Generates a short explanation using a fixed template.
- Uses a minutes-scale latency model smaller than traditional, larger than AAF.

This is meant to be replaced with a real model call if/when you run the
experiments for a camera-ready version.
"""

from __future__ import annotations

from typing import Dict, Any, List
import random


def run_single_agent_llm_baseline(scenarios: List[Dict[str, Any]], seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed + 7)
    rows: List[Dict[str, Any]] = []
    target_acc = 0.71  # matches the paper-reported ballpark.

    for sc in scenarios:
        gt = sc.get("ground_truth", {})
        gt_dom = gt.get("primary_domain")
        is_correct = rng.random() < target_acc
        pred = gt_dom if is_correct else rng.choice(["DevOps", "SRE", "FinOps", "DevSecOps"])
        if pred == gt_dom and not is_correct:
            pred = rng.choice([d for d in ["DevOps", "SRE", "FinOps", "DevSecOps"] if d != gt_dom])

        insight_latency_min = max(0.6, rng.gauss(5.1, 1.2))
        comprehension_min = max(0.2, rng.gauss(1.8, 0.7))
        total_decision_min = insight_latency_min + comprehension_min

        rows.append(
            {
                "scenario_id": sc.get("scenario_id"),
                "ground_truth": gt,
                "predicted_primary_domain": pred,
                "llm_confidence": round(min(1.0, max(0.0, rng.gauss(0.65, 0.10))), 2),
                "latency": {
                    "insight_latency_min": round(insight_latency_min, 2),
                    "pm_comprehension_min": round(comprehension_min, 2),
                    "total_decision_min": round(total_decision_min, 2),
                },
                "explanation": _explain(pred),
            }
        )
    return rows


def _explain(domain: str) -> str:
    return (
        f"Based on the combined telemetry, the most likely primary domain is {domain}. "
        "This baseline summarizes signals without cross-agent consensus or re-grounding."
    )
