"""Generate CSV/Markdown tables from experiment summary."""

from __future__ import annotations

from typing import Dict, Any
import os


def _get(d: Dict[str, Any], path: list[str], default: float = 0.0) -> float:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    try:
        return float(cur)
    except Exception:
        return default


def write_tables(out_dir: str, summary: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Accuracy comparison
    lines = ["| Method | Primary Domain Accuracy | Action Match Rate |", "|---|---:|---:|"]
    methods = [
        ("Traditional", "traditional"),
        ("Single-Agent LLM", "single_agent_llm"),
        ("AAF (Full)", "aaf_full"),
        ("AAF (No Consensus)", "aaf_no_consensus"),
        ("AAF (No RAR)", "aaf_no_rar"),
        ("AAF (No Utility)", "aaf_no_utility"),
    ]
    for label, key in methods:
        acc = _get(summary, [key, "accuracy", "accuracy"])
        act = _get(summary, [key, "action_match", "action_match_rate"])
        lines.append(f"| {label} | {acc:.2f} | {act:.2f} |")

    with open(os.path.join(out_dir, "table_accuracy.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # RAR table
    rar = summary.get("aaf_full", {}).get("rar", {})
    rar_lines = [
        "| Metric | Value |",
        "|---|---:|",
        f"| RAR Triggered | {rar.get('triggered', 0):.0f} |",
        f"| RAR Accepted | {rar.get('accepted', 0):.0f} |",
        f"| RAR Unresolved | {rar.get('unresolved', 0):.0f} |",
        f"| Trigger Rate | {rar.get('trigger_rate', 0):.2f} |",
        f"| Acceptance Rate When Triggered | {rar.get('acceptance_rate_when_triggered', 0):.2f} |",
        f"| Avg Consensus Improvement | {rar.get('avg_consensus_improvement', 0):.3f} |",
    ]
    with open(os.path.join(out_dir, "table_rar.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rar_lines) + "\n")

    # Consensus / utility / XI summary
    lines = [
        "| Method | Consensus Mean | Utility Mean | XI Mean |",
        "|---|---:|---:|---:|",
    ]
    for label, key in methods[2:]:
        c = _get(summary, [key, "consensus", "mean"])
        u = _get(summary, [key, "utility", "mean"])
        xi = _get(summary, [key, "xi", "mean"])
        lines.append(f"| {label} | {c:.2f} | {u:.2f} | {xi:.2f} |")

    with open(os.path.join(out_dir, "table_framework_metrics.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Actual performance gains from current run
    trad_acc = _get(summary, ["traditional", "accuracy", "accuracy"])
    aaf_acc = _get(summary, ["aaf_full", "accuracy", "accuracy"])
    trad_action = _get(summary, ["traditional", "action_match", "action_match_rate"])
    aaf_action = _get(summary, ["aaf_full", "action_match", "action_match_rate"])
    aaf_xi = _get(summary, ["aaf_full", "xi", "mean"])
    aaf_utility = _get(summary, ["aaf_full", "utility", "mean"])

    gains_md = [
        "| Metric | Traditional | AAF | Difference |",
        "|---|---:|---:|---:|",
        f"| Primary Domain Accuracy | {trad_acc:.2f} | {aaf_acc:.2f} | {aaf_acc - trad_acc:+.2f} |",
        f"| Action Match Rate | {trad_action:.2f} | {aaf_action:.2f} | {aaf_action - trad_action:+.2f} |",
        f"| XI | N/A | {aaf_xi:.2f} | N/A |",
        f"| Utility Score | N/A | {aaf_utility:.2f} | N/A |",
    ]
    with open(os.path.join(out_dir, "table_performance_gains.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(gains_md) + "\n")

    # Latency summary
    lat = summary["aaf_full"].get("latency", {}).get("total", {})
    with open(os.path.join(out_dir, "latency_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"TOTAL mean(ms): {lat.get('mean', 0.0):.2f}\n")
        f.write(f"TOTAL p50(ms): {lat.get('p50', 0.0):.2f}\n")
        f.write(f"TOTAL p95(ms): {lat.get('p95', 0.0):.2f}\n")
        f.write(f"TOTAL p99(ms): {lat.get('p99', 0.0):.2f}\n")
