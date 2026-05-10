"""Generate CSV/Markdown tables from experiment summary."""

from __future__ import annotations

from typing import Any, Dict
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


def _write(path: str, lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_tables(out_dir: str, summary: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)

    methods = [
        ("Traditional", "traditional"),
        ("Single-Agent LLM", "single_agent_llm"),
        ("AAF (Full)", "aaf_full"),
        ("AAF (No Consensus)", "aaf_no_consensus"),
        ("AAF (No RAR)", "aaf_no_rar"),
        ("AAF (No Utility)", "aaf_no_utility"),
    ]

    aaf_methods = methods[2:]

    # ------------------------------------------------------------------
    # Table 1: Accuracy and action selection
    # ------------------------------------------------------------------
    lines = [
        "| Method | Primary Domain Accuracy | Action Match Rate |",
        "|---|---:|---:|",
    ]

    for label, key in methods:
        acc = _get(summary, [key, "accuracy", "accuracy"])
        act = _get(summary, [key, "action_match", "action_match_rate"])
        lines.append(f"| {label} | {acc:.2f} | {act:.2f} |")

    _write(os.path.join(out_dir, "table_accuracy.md"), lines)

    # ------------------------------------------------------------------
    # Table 2: RAR summary for full AAF
    # ------------------------------------------------------------------
    rar = summary.get("aaf_full", {}).get("rar", {})
    rar_lines = [
        "| Metric | Value |",
        "|---|---:|",
        f"| RAR Triggered | {float(rar.get('triggered', 0.0)):.0f} |",
        f"| RAR Accepted | {float(rar.get('accepted', 0.0)):.0f} |",
        f"| RAR Unresolved | {float(rar.get('unresolved', 0.0)):.0f} |",
        f"| Trigger Rate | {float(rar.get('trigger_rate', 0.0)):.2f} |",
        f"| Acceptance Rate When Triggered | {float(rar.get('acceptance_rate_when_triggered', 0.0)):.2f} |",
        f"| Avg Consensus Improvement | {float(rar.get('avg_consensus_improvement', 0.0)):.3f} |",
    ]

    _write(os.path.join(out_dir, "table_rar.md"), rar_lines)

    # ------------------------------------------------------------------
    # Table 3: Framework-level metrics
    # ------------------------------------------------------------------
    framework_lines = [
        "| Method | Consensus Mean | Utility Mean | XI Mean |",
        "|---|---:|---:|---:|",
    ]

    for label, key in aaf_methods:
        consensus = _get(summary, [key, "consensus", "mean"])
        utility = _get(summary, [key, "utility", "mean"])
        xi = _get(summary, [key, "xi", "mean"])
        framework_lines.append(f"| {label} | {consensus:.2f} | {utility:.2f} | {xi:.2f} |")

    _write(os.path.join(out_dir, "table_framework_metrics.md"), framework_lines)

    # ------------------------------------------------------------------
    # Table 4: Utility component breakdown
    # ------------------------------------------------------------------
    utility_lines = [
        "| Method | Performance | Cost Efficiency | Risk Reduction | Composite Utility |",
        "|---|---:|---:|---:|---:|",
    ]

    for label, key in aaf_methods:
        perf = _get(summary, [key, "utility_components", "performance_mean"])
        cost = _get(summary, [key, "utility_components", "cost_efficiency_mean"])
        risk = _get(summary, [key, "utility_components", "risk_reduction_mean"])
        util = _get(summary, [key, "utility_components", "utility_mean"])
        utility_lines.append(f"| {label} | {perf:.2f} | {cost:.2f} | {risk:.2f} | {util:.2f} |")

    _write(os.path.join(out_dir, "table_utility_components.md"), utility_lines)

    # ------------------------------------------------------------------
    # Table 5: Explainability component breakdown
    # ------------------------------------------------------------------
    xi_lines = [
        "| Method | Readability | Evidence Clarity | Traceability | XI |",
        "|---|---:|---:|---:|---:|",
    ]

    for label, key in aaf_methods:
        readability = _get(summary, [key, "xi", "readability_mean"])
        evidence = _get(summary, [key, "xi", "evidence_clarity_mean"])
        traceability = _get(summary, [key, "xi", "traceability_mean"])
        xi = _get(summary, [key, "xi", "mean"])
        xi_lines.append(f"| {label} | {readability:.2f} | {evidence:.2f} | {traceability:.2f} | {xi:.2f} |")

    _write(os.path.join(out_dir, "table_xi_components.md"), xi_lines)

    # ------------------------------------------------------------------
    # Table 6: Actual performance gains from current run
    # ------------------------------------------------------------------
    trad_acc = _get(summary, ["traditional", "accuracy", "accuracy"])
    llm_acc = _get(summary, ["single_agent_llm", "accuracy", "accuracy"])
    aaf_acc = _get(summary, ["aaf_full", "accuracy", "accuracy"])

    trad_action = _get(summary, ["traditional", "action_match", "action_match_rate"])
    llm_action = _get(summary, ["single_agent_llm", "action_match", "action_match_rate"])
    aaf_action = _get(summary, ["aaf_full", "action_match", "action_match_rate"])

    aaf_xi = _get(summary, ["aaf_full", "xi", "mean"])
    aaf_utility = _get(summary, ["aaf_full", "utility_components", "utility_mean"])

    gains_lines = [
        "| Metric | Traditional | Single-Agent LLM | AAF | AAF vs Traditional | AAF vs Single-Agent |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Primary Domain Accuracy | {trad_acc:.2f} | {llm_acc:.2f} | {aaf_acc:.2f} | "
            f"{aaf_acc - trad_acc:+.2f} | {aaf_acc - llm_acc:+.2f} |"
        ),
        (
            f"| Action Match Rate | {trad_action:.2f} | {llm_action:.2f} | {aaf_action:.2f} | "
            f"{aaf_action - trad_action:+.2f} | {aaf_action - llm_action:+.2f} |"
        ),
        f"| Explainability Index | N/A | N/A | {aaf_xi:.2f} | N/A | N/A |",
        f"| Composite Utility | N/A | N/A | {aaf_utility:.2f} | N/A | N/A |",
    ]

    _write(os.path.join(out_dir, "table_performance_gains.md"), gains_lines)

    # ------------------------------------------------------------------
    # Latency summary
    # ------------------------------------------------------------------
    lat = summary.get("aaf_full", {}).get("latency", {}).get("total", {})
    latency_lines = [
        f"TOTAL mean(ms): {float(lat.get('mean', 0.0)):.2f}",
        f"TOTAL p50(ms): {float(lat.get('p50', 0.0)):.2f}",
        f"TOTAL p95(ms): {float(lat.get('p95', 0.0)):.2f}",
        f"TOTAL p99(ms): {float(lat.get('p99', 0.0)):.2f}",
    ]

    _write(os.path.join(out_dir, "latency_summary.txt"), latency_lines)
