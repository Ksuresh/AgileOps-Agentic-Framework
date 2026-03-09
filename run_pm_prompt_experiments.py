from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import yaml

from pipeline import run_once
from llm.deterministic_explainer import generate_explanation
from metrics.explainability import compute_xi
from pm_interface.prompt_router import route_prompt
from simulation.prompt_to_telemetry import build_telemetry_from_prompt_context


ROOT = Path(__file__).resolve().parent
PROMPT_LIBRARY_PATH = ROOT / "prompts" / "pm_prompt_library.yaml"
RESULTS_DIR = ROOT / "results"


DEFAULT_THRESHOLDS = {
    "tau_consensus": 0.75,
    "delta_min": 0.15,
    "max_rar_loops": 2,
}
DEFAULT_WEIGHTS = (0.4, 0.3, 0.3)
DEFAULT_LAMBDA = 0.5


def load_prompt_library() -> list[dict]:
    if not PROMPT_LIBRARY_PATH.exists():
        raise FileNotFoundError(f"Prompt library not found: {PROMPT_LIBRARY_PATH}")

    data = yaml.safe_load(PROMPT_LIBRARY_PATH.read_text(encoding="utf-8")) or {}
    prompts = data.get("prompts", [])
    if not isinstance(prompts, list):
        raise ValueError("Prompt library format invalid: 'prompts' must be a list")
    return prompts


def main() -> None:
    prompts = load_prompt_library()

    outdir = RESULTS_DIR / f"phase2_prompts_{int(time.time())}"
    outdir.mkdir(parents=True, exist_ok=True)

    outputs_path = outdir / "phase2_prompt_outputs.jsonl"
    csv_path = outdir / "phase2_prompt_metrics.csv"
    summary_path = outdir / "phase2_prompt_summary.json"

    rows = []

    with outputs_path.open("w", encoding="utf-8") as f:
        for item in prompts:
            prompt_id = item.get("id")
            title = item.get("title")
            category = item.get("category")
            priority = item.get("priority")
            prompt_text = item.get("prompt", "")
            expected_domains = item.get("expected_domains", [])
            expected_action = item.get("expected_action")

            route = route_prompt(prompt_text)
            telemetry = build_telemetry_from_prompt_context(route)

            result = run_once(
                telemetry=telemetry,
                thresholds=DEFAULT_THRESHOLDS,
                lam=DEFAULT_LAMBDA,
                w=DEFAULT_WEIGHTS,
            )

            payload = {
                "incident_id": prompt_id,
                "agents": result["agents"],
                "consensus_score": result["consensus_score"],
                "rar_triggered": result["rar_triggered"],
                "recommended_action": result["recommended_action"],
                "utility_score": result["utility_score"],
            }

            explanation = generate_explanation(payload)
            xi = compute_xi(explanation, payload)

            actual_domains = [a.get("agent_type") for a in result.get("agents", [])]
            actual_action = result.get("recommended_action")

            domain_overlap = sorted(set(expected_domains).intersection(set(actual_domains)))
            action_match = actual_action == expected_action

            record = {
                "id": prompt_id,
                "title": title,
                "category": category,
                "priority": priority,
                "pm_goal": item.get("pm_goal"),
                "prompt": prompt_text,
                "expected_domains": expected_domains,
                "expected_action": expected_action,
                "route": route,
                "telemetry": telemetry,
                "agents": result.get("agents", []),
                "consensus_score": result.get("consensus_score"),
                "rar_triggered": result.get("rar_triggered"),
                "rar_loops": result.get("rar_loops"),
                "recommended_action": actual_action,
                "utility_score": result.get("utility_score"),
                "domain_overlap": domain_overlap,
                "action_match": action_match,
                "explanation": explanation,
                "xi": xi,
            }

            f.write(json.dumps(record) + "\n")

            rows.append(
                {
                    "id": prompt_id,
                    "title": title,
                    "category": category,
                    "priority": priority,
                    "expected_action": expected_action,
                    "recommended_action": actual_action,
                    "action_match": action_match,
                    "consensus_score": result.get("consensus_score"),
                    "rar_triggered": result.get("rar_triggered"),
                    "rar_loops": result.get("rar_loops"),
                    "utility_score": result.get("utility_score"),
                    "xi_score": xi.get("xi"),
                    "expected_domain_count": len(expected_domains),
                    "actual_domain_count": len(actual_domains),
                    "domain_overlap_count": len(domain_overlap),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    summary = {
        "n_prompts": int(len(df)),
        "categories": sorted(df["category"].dropna().unique().tolist()),
        "priorities": sorted(df["priority"].dropna().unique().tolist()),
        "action_match_rate": float(df["action_match"].mean()) if len(df) else 0.0,
        "rar_trigger_rate": float(df["rar_triggered"].mean()) if len(df) else 0.0,
        "avg_rar_loops": float(df["rar_loops"].mean()) if len(df) else 0.0,
        "consensus_mean": float(df["consensus_score"].mean()) if len(df) else 0.0,
        "consensus_p50": float(df["consensus_score"].quantile(0.50)) if len(df) else 0.0,
        "consensus_p95": float(df["consensus_score"].quantile(0.95)) if len(df) else 0.0,
        "utility_mean": float(df["utility_score"].mean()) if len(df) else 0.0,
        "xi_mean": float(df["xi_score"].mean()) if len(df) else 0.0,
        "action_distribution": df["recommended_action"].value_counts().to_dict(),
        "category_breakdown": (
            df.groupby("category")
            .agg(
                prompts=("id", "count"),
                action_match_rate=("action_match", "mean"),
                rar_trigger_rate=("rar_triggered", "mean"),
                consensus_mean=("consensus_score", "mean"),
                xi_mean=("xi_score", "mean"),
            )
            .round(4)
            .reset_index()
            .to_dict(orient="records")
        ),
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {outputs_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
