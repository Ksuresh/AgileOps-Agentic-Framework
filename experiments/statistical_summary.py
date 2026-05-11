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


def get_expected_domain(row: Dict[str, Any]) -> str | None:
    gt = row.get("ground_truth", {})
    return (
        gt.get("primary_domain")
        or row.get("expected_primary_domain")
        or row.get("primary_domain_expected")
    )


def get_predicted_domain(row: Dict[str, Any]) -> str | None:
    return (
        row.get("predicted_primary_domain")
        or row.get("primary_domain")
        or row.get("predicted_domain")
        or row.get("domain")
    )


def get_expected_action(row: Dict[str, Any]) -> str | None:
    gt = row.get("ground_truth", {})
    return (
        gt.get("expected_action")
        or gt.get("recommended_action")
        or row.get("expected_action")
        or row.get("expected_recommended_action")
    )


def get_selected_action(row: Dict[str, Any]) -> str | None:
    utility = row.get("utility") or {}
    utility_decision = row.get("utility_decision") or {}
    decision = row.get("decision") or {}
    governance_decision = row.get("governance_decision") or {}

    return (
        row.get("selected_action")
        or row.get("recommended_action")
        or row.get("action")
        or utility.get("selected_action")
        or utility.get("recommended_action")
        or utility.get("action")
        or utility_decision.get("selected_action")
        or utility_decision.get("recommended_action")
        or utility_decision.get("action")
        or decision.get("selected_action")
        or decision.get("recommended_action")
        or decision.get("action")
        or governance_decision.get("selected_action")
        or governance_decision.get("recommended_action")
        or governance_decision.get("action")
    )


def domain_correct(row: Dict[str, Any]) -> bool:
    expected = get_expected_domain(row)
    predicted = get_predicted_domain(row)
    return bool(expected and predicted and expected == predicted)


def action_correct(row: Dict[str, Any]) -> bool:
    expected = get_expected_action(row)
    selected = get_selected_action(row)
    return bool(expected and selected and expected == selected)


def binary_metric_ci(successes: int, n: int) -> Dict[str, float]:
    """
    Normal approximation 95% CI for a binary proportion.

    For this controlled evaluation, this is used as an interpretable
    uncertainty estimate rather than a claim of population-level inference.
    """
    if n == 0:
        return {
            "rate": 0.0,
            "n": 0.0,
            "successes": 0.0,
            "se": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
        }

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


def summarize_method(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)

    domain_success = sum(1 for row in rows if domain_correct(row))
    action_evaluable_rows = [
        row for row in rows if get_expected_action(row) and get_selected_action(row)
    ]
    action_success = sum(1 for row in action_evaluable_rows if action_correct(row))

    return {
        "domain_accuracy_ci": binary_metric_ci(domain_success, n),
        "action_match_ci": binary_metric_ci(action_success, len(action_evaluable_rows)),
        "action_evaluable_n": float(len(action_evaluable_rows)),
    }


def paired_difference_ci(
    rows_a: List[Dict[str, Any]],
    rows_b: List[Dict[str, Any]],
    metric: str,
) -> Dict[str, float]:
    if len(rows_a) != len(rows_b):
        raise ValueError("Paired rows must have the same length.")

    diffs: List[int] = []

    for a, b in zip(rows_a, rows_b):
        if metric == "domain":
            a_score = 1 if domain_correct(a) else 0
            b_score = 1 if domain_correct(b) else 0
        elif metric == "action":
            a_score = 1 if action_correct(a) else 0
            b_score = 1 if action_correct(b) else 0
        else:
            raise ValueError(f"Unknown metric: {metric}")

        diffs.append(a_score - b_score)

    n = len(diffs)
    mean = sum(diffs) / n if n else 0.0

    if n <= 1:
        se = 0.0
    else:
        variance = sum((d - mean) ** 2 for d in diffs) / (n - 1)
        se = math.sqrt(variance / n)

    return {
        "mean_difference": mean,
        "n": float(n),
        "se": se,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
    }


def load_required_results(results_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    required_files = {
        "traditional": "traditional.jsonl",
        "single_agent_llm": "single_agent_llm.jsonl",
        "aaf_full": "aaf_full.jsonl",
        "aaf_no_consensus": "aaf_no_consensus.jsonl",
        "aaf_no_rar": "aaf_no_rar.jsonl",
        "aaf_no_utility": "aaf_no_utility.jsonl",
    }

    missing = [
        filename
        for filename in required_files.values()
        if not (results_dir / filename).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required result files: "
            + ", ".join(missing)
            + f". Re-run: python -m experiments.run_all --out {results_dir} --n 120 --seed 42"
        )

    return {
        method: load_jsonl(results_dir / filename)
        for method, filename in required_files.items()
    }


def write_markdown_report(summary: Dict[str, Any], out_path: Path) -> None:
    lines: List[str] = []

    lines.append("# Statistical Summary\n")
    lines.append("## Method Metrics with 95% Confidence Intervals\n")
    lines.append("| Method | Domain Accuracy | 95% CI | Action Match | 95% CI | Action N |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    method_labels = {
        "traditional": "Traditional Baseline",
        "single_agent_llm": "Single-Agent Baseline",
        "aaf_full": "AAF Full",
        "aaf_no_consensus": "AAF w/o Consensus",
        "aaf_no_rar": "AAF w/o RAR",
        "aaf_no_utility": "AAF w/o Utility",
    }

    for method_name, data in summary["methods"].items():
        label = method_labels.get(method_name, method_name)
        da = data["domain_accuracy_ci"]
        am = data["action_match_ci"]

        lines.append(
            f"| {label} | "
            f"{da['rate']:.3f} | [{da['ci95_low']:.3f}, {da['ci95_high']:.3f}] | "
            f"{am['rate']:.3f} | [{am['ci95_low']:.3f}, {am['ci95_high']:.3f}] | "
            f"{int(data['action_evaluable_n'])} |"
        )

    lines.append("\n## Paired Difference Estimates\n")
    lines.append("| Comparison | Mean Difference | 95% CI |")
    lines.append("|---|---:|---:|")

    comparison_labels = {
        "domain_vs_traditional": "AAF Full domain accuracy minus Traditional Baseline",
        "domain_vs_single_agent": "AAF Full domain accuracy minus Single-Agent Baseline",
        "domain_vs_no_consensus": "AAF Full domain accuracy minus AAF w/o Consensus",
        "domain_vs_no_rar": "AAF Full domain accuracy minus AAF w/o RAR",
        "action_vs_no_utility": "AAF Full action match minus AAF w/o Utility",
    }

    for comparison_name, data in summary["paired_differences_vs_aaf_full"].items():
        label = comparison_labels.get(comparison_name, comparison_name)
        lines.append(
            f"| {label} | "
            f"{data['mean_difference']:.3f} | "
            f"[{data['ci95_low']:.3f}, {data['ci95_high']:.3f}] |"
        )

    lines.append(
        "\nNote: Confidence intervals are descriptive uncertainty estimates for the "
        "controlled evaluation set and should not be interpreted as claims of "
        "production-scale generalization.\n"
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Results folder")
    args = parser.parse_args()

    results_dir = Path(args.results)
    methods = load_required_results(results_dir)

    summary = {
        "methods": {
            name: summarize_method(rows)
            for name, rows in methods.items()
        },
        "paired_differences_vs_aaf_full": {
            "domain_vs_traditional": paired_difference_ci(
                methods["aaf_full"], methods["traditional"], "domain"
            ),
            "domain_vs_single_agent": paired_difference_ci(
                methods["aaf_full"], methods["single_agent_llm"], "domain"
            ),
            "domain_vs_no_consensus": paired_difference_ci(
                methods["aaf_full"], methods["aaf_no_consensus"], "domain"
            ),
            "domain_vs_no_rar": paired_difference_ci(
                methods["aaf_full"], methods["aaf_no_rar"], "domain"
            ),
            "action_vs_no_utility": paired_difference_ci(
                methods["aaf_full"], methods["aaf_no_utility"], "action"
            ),
        },
    }

    out_json = results_dir / "statistical_summary.json"
    out_md = results_dir / "statistical_summary.md"

    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_report(summary, out_md)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
