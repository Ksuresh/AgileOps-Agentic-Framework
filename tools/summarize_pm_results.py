from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize Phase 2 PM prompt experiment CSV results")
    ap.add_argument("csv_path", help="Path to phase2_prompt_metrics.csv")
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    summary = {
        "n_prompts": int(len(df)),
        "action_match_rate": float(df["action_match"].mean()) if len(df) else 0.0,
        "rar_trigger_rate": float(df["rar_triggered"].mean()) if len(df) else 0.0,
        "avg_rar_loops": float(df["rar_loops"].mean()) if len(df) else 0.0,
        "consensus_mean": float(df["consensus_score"].mean()) if len(df) else 0.0,
        "consensus_min": float(df["consensus_score"].min()) if len(df) else 0.0,
        "consensus_max": float(df["consensus_score"].max()) if len(df) else 0.0,
        "utility_mean": float(df["utility_score"].mean()) if len(df) else 0.0,
        "xi_mean": float(df["xi_score"].mean()) if len(df) else 0.0,
        "action_distribution": df["recommended_action"].value_counts().to_dict(),
        "priority_breakdown": (
            df.groupby("priority")
            .agg(
                prompts=("id", "count"),
                action_match_rate=("action_match", "mean"),
                rar_trigger_rate=("rar_triggered", "mean"),
                consensus_mean=("consensus_score", "mean"),
            )
            .round(4)
            .reset_index()
            .to_dict(orient="records")
        ),
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

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
