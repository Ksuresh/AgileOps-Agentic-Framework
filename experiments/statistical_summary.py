from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_expected_domain(row: Dict[str, Any]) -> str:
    return row.get("ground_truth", {}).get("primary_domain") or row.get("expected_primary_domain")


def get_predicted_domain(row: Dict[str, Any]) -> str:
    return (
        row.get("predicted_primary_domain")
        or row.get("primary_domain")
        or row.get("predicted_domain")
        or row.get("domain")
    )


def get_expected_action(row: Dict[str, Any]) -> str:
    gt = row.get("ground_truth", {})
    return gt.get("expected_action") or gt.get("recommended_action") or row.get("expected_action")


def get_selected_action(row: Dict[str, Any]) -> str:
    return (
        row.get("selected_action")
        or row.get("recommended_action")
        or row.get("action")
    )


def binary_metric_ci(successes: int, n: int) -> Dict[str, float]:
    if n == 0:
        return {"rate": 0.0, "n": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "se": 0.0}

    p = successes / n
    se = math.sqrt((p * (1 - p)) / n)
    low = max(0.0, p - 1.96 * se)
    high = min(1.0, p + 1.96 * se)
    return {
        "rate": p,
        "n": float(n),
        "successes": float(successes),
        "se": se,
        "ci95_low": low,
        "ci95_high": high,
    }


def domain_correct(row: Dict[str, Any]) -> bool:
    return get_expected_domain(row) == get_predicted_domain(row)


def action_correct(row: Dict[str, Any]) -> bool:
    expected = get_expected_action(row)
    selected = get_selected_action(row)
    return bool(expected and selected and expected == selected)


def summarize_method(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    domain_success = sum(1 for r in rows if domain_correct(r))
    action_rows = [r for r in rows if get_expected_action(r) and get_selected_action(r)]
    action_success = sum(1 for r in action_rows if action_correct(r))

    return {
        "domain_accuracy_ci": binary_metric_ci(domain_success, n),
        "action_match_ci": binary_metric_ci(action_success, len(action_rows)),
    }


def paired_difference_ci(
    rows_a: List[Dict[str, Any]],
    rows_b: List[Dict[str, Any]],
    metric: str,
) -> Dict[str, float]:
    if len(rows_a) != len(rows_b):
        raise ValueError("Paired rows must have same length.")

    diffs = []
    for a, b in zip(rows_a, rows_b):
        if metric == "domain":
            diffs.append((1 if domain_correct(a) else 0) - (1 if domain_correct(b) else 0))
        elif metric == "action":
            diffs.append((1 if action_correct(a) else 0) - (1 if action_correct(b) else 0))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    n = len(diffs)
    mean = sum(diffs) / n
    if n <= 1:
        se = 0.0
    else:
        var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
        se = math.sqrt(var / n)

    return {
        "mean_difference": mean,
        "n": float(n),
        "se": se,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Results folder")
    args = parser.parse_args()

    results_dir = Path(args.results)

    methods = {
        "traditional": load_jsonl(results_dir / "traditional.jsonl"),
        "single_agent_llm": load_jsonl(results_dir / "single_agent_llm.jsonl"),
        "aaf_full": load_jsonl(results_dir / "aaf_full.jsonl"),
        "aaf_no_consensus": load_jsonl(results_dir / "aaf_no_consensus.jsonl"),
        "aaf_no_rar": load_jsonl(results_dir / "aaf_no_rar.jsonl"),
        "aaf_no_utility": load_jsonl(results_dir / "aaf_no_utility.jsonl"),
    }

    summary = {
        "methods": {name: summarize_method(rows) for name, rows in methods.items()},
        "paired_differences_vs_aaf_full": {
            "domain_vs_traditional": paired_difference_ci(methods["aaf_full"], methods["traditional"], "domain"),
            "domain_vs_single_agent": paired_difference_ci(methods["aaf_full"], methods["single_agent_llm"], "domain"),
            "action_vs_no_utility": paired_difference_ci(methods["aaf_full"], methods["aaf_no_utility"], "action"),
        },
    }

    out_json = results_dir / "statistical_summary.json"
    out_md = results_dir / "statistical_summary.md"

    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Statistical Summary\n")
    lines.append("## Method Metrics with 95% Confidence Intervals\n")
    lines.append("| Method | Domain Accuracy | 95% CI | Action Match | 95% CI |")
    lines.append("|---|---:|---:|---:|---:|")

    for name, data in summary["methods"].items():
        da = data["domain_accuracy_ci"]
        am = data["action_match_ci"]
        lines.append(
            f"| {name} | {da['rate']:.3f} | [{da['ci95_low']:.3f}, {da['ci95_high']:.3f}] "
            f"| {am['rate']:.3f} | [{am['ci95_low']:.3f}, {am['ci95_high']:.3f}] |"
        )

    lines.append("\n## Paired Difference Estimates\n")
    lines.append("| Comparison | Mean Difference | 95% CI |")
    lines.append("|---|---:|---:|")

    for name, data in summary["paired_differences_vs_aaf_full"].items():
        lines.append(
            f"| {name} | {data['mean_difference']:.3f} | "
            f"[{data['ci95_low']:.3f}, {data['ci95_high']:.3f}] |"
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
