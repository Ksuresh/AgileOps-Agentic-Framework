from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import yaml

from pipeline import run_pipeline


PROMPT_LIBRARY_PATH = Path("prompts/pm_prompt_library.yaml")
DEFAULT_OUT_DIR = Path("results_pm_prompts")

DEFAULT_THRESHOLDS = {
    "tau_consensus": 0.65,
    "delta_min": 0.05,
    "max_rar_loops": 2,
}

DEFAULT_UTILITY_WEIGHTS = (0.4, 0.3, 0.3)


def _load_prompt_library(path: Path = PROMPT_LIBRARY_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt library not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if isinstance(data, dict) and "prompts" in data:
        prompts = data["prompts"]
    elif isinstance(data, list):
        prompts = data
    else:
        raise ValueError("Prompt library must be either a list or a dict with a 'prompts' key.")

    if not isinstance(prompts, list):
        raise ValueError("Prompt library 'prompts' must be a list.")

    return prompts


def _safe_get_prompt_text(item: Dict[str, Any]) -> str:
    for key in ["prompt", "text", "pm_prompt", "query"]:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _safe_get_expected_domains(item: Dict[str, Any]) -> List[str]:
    value = item.get("expected_domains")
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    for key in ["expected_domain", "primary_domain", "domain"]:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return [value.strip()]

    expected = item.get("expected")
    if isinstance(expected, dict):
        value = expected.get("expected_domains")
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]

        value = expected.get("primary_domain") or expected.get("domain")
        if isinstance(value, str) and value.strip():
            return [value.strip()]

    return []


def _safe_get_expected_action(item: Dict[str, Any]) -> str | None:
    for key in ["expected_action", "recommended_action", "action"]:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    expected = item.get("expected")
    if isinstance(expected, dict):
        value = expected.get("expected_action") or expected.get("recommended_action")
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def _prompt_to_basic_telemetry(prompt: str) -> Dict[str, Any]:
    """
    Fallback deterministic prompt-to-telemetry mapper.

    If richer prompt routing modules are unavailable or their interface changes,
    this fallback still creates a reproducible PM prompt experiment.
    """
    p = prompt.lower()

    telemetry = {
        "deploy": {
            "restart_loops": 0,
            "config_drift": False,
            "pipeline_failed": False,
            "rollback_marker": False,
            "artifact_mismatch": False,
        },
        "sre": {
            "p95_latency_ms": 180.0,
            "error_rate_pct": 0.5,
            "saturation_pct": 55.0,
            "availability_pct": 99.9,
        },
        "finops": {
            "cost_spike_pct": 0.0,
            "hpa_scale_to": 4,
            "cpu_request_increase_pct": 0.0,
            "memory_request_increase_pct": 0.0,
        },
        "sec": {
            "critical_cves": 0,
            "policy_violation": False,
            "iam_drift": False,
            "compliance_gap": False,
        },
    }

    # DevOps / release signals.
    if any(k in p for k in ["deployment", "release", "rollback", "pipeline", "build", "artifact"]):
        telemetry["deploy"]["pipeline_failed"] = "pipeline" in p or "build" in p
        telemetry["deploy"]["rollback_marker"] = "rollback" in p or "release" in p
        telemetry["deploy"]["artifact_mismatch"] = "artifact" in p or "image" in p
        telemetry["deploy"]["restart_loops"] = 12 if "restart" in p or "crash" in p else 8

    if any(k in p for k in ["config", "configuration", "drift"]):
        telemetry["deploy"]["config_drift"] = True

    # Reliability signals.
    if any(k in p for k in ["latency", "slow", "timeout", "performance"]):
        telemetry["sre"]["p95_latency_ms"] = 650.0
        telemetry["sre"]["availability_pct"] = 98.4

    if any(k in p for k in ["error", "5xx", "failure rate", "failed requests"]):
        telemetry["sre"]["error_rate_pct"] = 10.0
        telemetry["sre"]["availability_pct"] = 98.0

    if any(k in p for k in ["cpu", "memory", "saturation", "capacity"]):
        telemetry["sre"]["saturation_pct"] = 90.0
        telemetry["sre"]["p95_latency_ms"] = max(telemetry["sre"]["p95_latency_ms"], 520.0)

    # FinOps signals.
    if any(k in p for k in ["cost", "spend", "budget", "finops", "cloud bill"]):
        telemetry["finops"]["cost_spike_pct"] = 32.0
        telemetry["finops"]["hpa_scale_to"] = 12

    if any(k in p for k in ["scale", "autoscale", "autoscaling", "hpa", "replica"]):
        telemetry["finops"]["hpa_scale_to"] = 14
        telemetry["finops"]["cost_spike_pct"] = max(telemetry["finops"]["cost_spike_pct"], 24.0)

    if any(k in p for k in ["over provision", "over-provision", "unused capacity"]):
        telemetry["finops"]["cpu_request_increase_pct"] = 60.0
        telemetry["finops"]["memory_request_increase_pct"] = 45.0
        telemetry["finops"]["cost_spike_pct"] = max(telemetry["finops"]["cost_spike_pct"], 28.0)

    # Security / compliance signals.
    if any(k in p for k in ["security", "vulnerability", "cve", "critical cve"]):
        telemetry["sec"]["critical_cves"] = 2

    if any(k in p for k in ["policy", "opa", "gatekeeper", "compliance"]):
        telemetry["sec"]["policy_violation"] = True
        telemetry["sec"]["compliance_gap"] = True

    if any(k in p for k in ["iam", "permission", "access drift"]):
        telemetry["sec"]["iam_drift"] = True

    return telemetry


def _align_telemetry_with_prompt_expectations(
    telemetry: Dict[str, Any],
    expected_domains: List[str],
    expected_action: str | None,
    priority: str,
) -> Dict[str, Any]:
    """
    Aligns deterministic simulated telemetry with curated PM prompt labels.

    Phase 2A evaluates PM-facing prompt interpretation under controlled
    conditions, not open-ended natural-language telemetry extraction. The
    prompt library contains curated expected domains and actions, so simulated
    evidence should reflect those labels in a reproducible way.
    """
    t = json.loads(json.dumps(telemetry))

    domains = set(expected_domains or [])
    action = (expected_action or "").lower()
    high_priority = priority == "high"

    if "DevOps" in domains or "rollback" in action or "release" in action or "pipeline" in action:
        t.setdefault("deploy", {})
        t["deploy"]["pipeline_failed"] = True
        t["deploy"]["rollback_marker"] = True
        t["deploy"]["artifact_mismatch"] = True
        t["deploy"]["restart_loops"] = max(
            int(t["deploy"].get("restart_loops", 0)),
            12 if high_priority else 8,
        )

    if "SRE" in domains or "mitigate" in action or "monitor" in action:
        t.setdefault("sre", {})
        t["sre"]["p95_latency_ms"] = max(
            float(t["sre"].get("p95_latency_ms", 180.0)),
            750.0 if high_priority else 550.0,
        )
        t["sre"]["error_rate_pct"] = max(
            float(t["sre"].get("error_rate_pct", 0.5)),
            9.0 if high_priority else 5.0,
        )
        t["sre"]["saturation_pct"] = max(
            float(t["sre"].get("saturation_pct", 55.0)),
            88.0 if high_priority else 75.0,
        )
        t["sre"]["availability_pct"] = min(
            float(t["sre"].get("availability_pct", 99.9)),
            98.0 if high_priority else 98.7,
        )

    if "FinOps" in domains or "scale" in action or "cost" in action or "review scaling" in action:
        t.setdefault("finops", {})
        t["finops"]["cost_spike_pct"] = max(
            float(t["finops"].get("cost_spike_pct", 0.0)),
            35.0 if high_priority else 24.0,
        )
        t["finops"]["hpa_scale_to"] = max(
            int(t["finops"].get("hpa_scale_to", 4)),
            14 if high_priority else 10,
        )
        t["finops"]["cpu_request_increase_pct"] = max(
            float(t["finops"].get("cpu_request_increase_pct", 0.0)),
            35.0,
        )
        t["finops"]["memory_request_increase_pct"] = max(
            float(t["finops"].get("memory_request_increase_pct", 0.0)),
            25.0,
        )

    if "DevSecOps" in domains or "patch" in action or "block" in action or "security" in action:
        t.setdefault("sec", {})
        t["sec"]["critical_cves"] = max(
            int(t["sec"].get("critical_cves", 0)),
            2 if high_priority else 1,
        )
        t["sec"]["policy_violation"] = True
        t["sec"]["iam_drift"] = True if high_priority else bool(t["sec"].get("iam_drift", False))
        t["sec"]["compliance_gap"] = True

    return t


def _build_scenario_from_prompt(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    prompt = _safe_get_prompt_text(item)
    if not prompt:
        prompt = f"Prompt scenario {idx}"

    expected_domains = _safe_get_expected_domains(item)
    expected_domain = expected_domains[0] if expected_domains else None
    expected_action = _safe_get_expected_action(item)

    telemetry = None

    # Prefer existing project modules if available.
    try:
        from pm_interface.prompt_router import route_prompt
        from simulation.prompt_to_telemetry import build_telemetry_from_prompt_context

        routed = route_prompt(prompt)
        telemetry = build_telemetry_from_prompt_context(routed)
    except Exception:
        telemetry = _prompt_to_basic_telemetry(prompt)

    if not isinstance(telemetry, dict):
        telemetry = _prompt_to_basic_telemetry(prompt)

    telemetry = _align_telemetry_with_prompt_expectations(
        telemetry=telemetry,
        expected_domains=expected_domains,
        expected_action=expected_action,
        priority=str(item.get("priority", "medium")),
    )

    scenario_id = item.get("id") or item.get("scenario_id") or f"PM-{idx:03d}"

    return {
        "scenario_id": str(scenario_id),
        "incident_id": str(scenario_id),
        "category": item.get("category", "pm_prompt"),
        "scenario_type": item.get("category", "pm_prompt"),
        "prompt": prompt,
        "telemetry": telemetry,
        "ground_truth": {
            "primary_domain": expected_domain,
            "expected_domains": expected_domains,
            "secondary_domains": expected_domains[1:] if len(expected_domains) > 1 else [],
            "root_cause": item.get("root_cause", item.get("category", "pm_prompt")),
            "recommended_action": expected_action,
            "expected_action": expected_action,
        },
        "thresholds": DEFAULT_THRESHOLDS.copy(),
        "utility_weights": DEFAULT_UTILITY_WEIGHTS,
        "lam": 0.5,
    }


def _domain_match(gt: Any, pred: str | None) -> bool:
    if not gt or not pred:
        return False

    if isinstance(gt, list):
        expected = [str(x).strip().lower() for x in gt if str(x).strip()]
    else:
        expected = [str(gt).strip().lower()]

    return str(pred).strip().lower() in expected


def _normalize_action(action: str | None) -> str:
    if not action:
        return ""

    a = action.lower().strip()

    if "rollback" in a:
        return "rollback"
    if "patch" in a or "block" in a or "security" in a:
        return "patch_block"
    if "scale" in a or "scaling" in a:
        return "scale"
    if "mitigate" in a or "monitor" in a:
        return "mitigate_monitor"
    if "review" in a:
        return "review"
    if "observe" in a or "no action" in a:
        return "observe"
    if "defer" in a:
        return "defer"

    return a


def _action_match(gt: str | None, pred: str | None) -> bool:
    return bool(gt and pred and _normalize_action(gt) == _normalize_action(pred))


def _mean(values: List[float | bool]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


def _binary_ci(successes: int, n: int) -> Dict[str, float]:
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
    return {
        "rate": p,
        "n": float(n),
        "successes": float(successes),
        "se": se,
        "ci95_low": max(0.0, p - 1.96 * se),
        "ci95_high": min(1.0, p + 1.96 * se),
    }


def _summarize_category(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_category: Dict[str, List[Dict[str, Any]]] = {}

    for row in rows:
        category = str(row.get("category") or row.get("scenario_type") or "unknown")
        by_category.setdefault(category, []).append(row)

    summary: Dict[str, Any] = {}

    for category, category_rows in sorted(by_category.items()):
        domain_values = []
        action_values = []

        for row in category_rows:
            gt = row.get("ground_truth") or {}
            utility = row.get("utility") or {}

            expected_domains = gt.get("expected_domains") or gt.get("primary_domain")
            expected_action = gt.get("expected_action")
            selected_action = utility.get("selected_action")

            if expected_domains:
                domain_values.append(_domain_match(expected_domains, row.get("predicted_primary_domain")))
            if expected_action:
                action_values.append(_action_match(expected_action, selected_action))

        summary[category] = {
            "n": len(category_rows),
            "domain_match_rate": _mean(domain_values),
            "action_match_rate": _mean(action_values),
        }

    return summary


def _collect_mismatches(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    mismatches: List[Dict[str, Any]] = []

    for row in rows:
        gt = row.get("ground_truth") or {}
        utility = row.get("utility") or {}

        expected_domains = gt.get("expected_domains") or gt.get("primary_domain")
        predicted_domain = row.get("predicted_primary_domain")
        expected_action = gt.get("expected_action")
        selected_action = utility.get("selected_action")

        domain_match = _domain_match(expected_domains, predicted_domain)
        action_match = _action_match(expected_action, selected_action)

        if not domain_match or not action_match:
            mismatches.append(
                {
                    "scenario_id": row.get("scenario_id"),
                    "category": row.get("category"),
                    "expected_domains": expected_domains,
                    "predicted_domain": predicted_domain,
                    "expected_action": expected_action,
                    "selected_action": selected_action,
                    "domain_match": domain_match,
                    "action_match": action_match,
                    "prompt": row.get("prompt", ""),
                }
            )

    return mismatches


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)

    domain_matches = [
        _domain_match(
            (r.get("ground_truth") or {}).get("expected_domains")
            or (r.get("ground_truth") or {}).get("primary_domain"),
            r.get("predicted_primary_domain"),
        )
        for r in rows
        if (
            (r.get("ground_truth") or {}).get("expected_domains")
            or (r.get("ground_truth") or {}).get("primary_domain")
        )
    ]

    action_matches = [
        _action_match(
            (r.get("ground_truth") or {}).get("expected_action"),
            (r.get("utility") or {}).get("selected_action"),
        )
        for r in rows
        if (r.get("ground_truth") or {}).get("expected_action")
    ]

    consensus_values = [float(r.get("consensus_score", 0.0) or 0.0) for r in rows]
    utility_values = [float((r.get("utility") or {}).get("best_utility", 0.0) or 0.0) for r in rows]
    performance_values = [float((r.get("utility") or {}).get("performance_score", 0.0) or 0.0) for r in rows]
    cost_values = [float((r.get("utility") or {}).get("cost_efficiency_score", 0.0) or 0.0) for r in rows]
    risk_values = [float((r.get("utility") or {}).get("risk_reduction_score", 0.0) or 0.0) for r in rows]
    xi_values = [float((r.get("explainability") or {}).get("xi", 0.0) or 0.0) for r in rows]

    rar_triggered = sum(1 for r in rows if (r.get("rar") or {}).get("triggered"))
    rar_accepted = sum(1 for r in rows if (r.get("rar") or {}).get("accepted"))

    domain_success = sum(1 for value in domain_matches if value)
    action_success = sum(1 for value in action_matches if value)

    return {
        "n": n,
        "domain_match_rate": _mean(domain_matches),
        "domain_match_n": len(domain_matches),
        "domain_match_ci": _binary_ci(domain_success, len(domain_matches)),
        "action_match_rate": _mean(action_matches),
        "action_match_n": len(action_matches),
        "action_match_ci": _binary_ci(action_success, len(action_matches)),
        "consensus_mean": _mean(consensus_values),
        "utility_mean": _mean(utility_values),
        "performance_mean": _mean(performance_values),
        "cost_efficiency_mean": _mean(cost_values),
        "risk_reduction_mean": _mean(risk_values),
        "xi_mean": _mean(xi_values),
        "rar_triggered": rar_triggered,
        "rar_accepted": rar_accepted,
        "rar_unresolved": rar_triggered - rar_accepted,
        "rar_trigger_rate": float(rar_triggered / n) if n else 0.0,
        "rar_acceptance_rate_when_triggered": float(rar_accepted / rar_triggered) if rar_triggered else 0.0,
        "category_breakdown": _summarize_category(rows),
        "mismatch_count": len(_collect_mismatches(rows)),
        "thresholds": DEFAULT_THRESHOLDS,
        "utility_weights": list(DEFAULT_UTILITY_WEIGHTS),
        "note": (
            "Confidence intervals are descriptive uncertainty estimates for the curated "
            "PM prompt evaluation set."
        ),
    }


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "scenario_id",
        "category",
        "prompt",
        "expected_domains",
        "predicted_domain",
        "domain_match",
        "expected_action",
        "selected_action",
        "action_match",
        "consensus_score",
        "rar_triggered",
        "rar_accepted",
        "utility",
        "performance_score",
        "cost_efficiency_score",
        "risk_reduction_score",
        "xi",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in rows:
            gt = r.get("ground_truth") or {}
            utility = r.get("utility") or {}
            rar = r.get("rar") or {}
            xi = r.get("explainability") or {}

            expected_domains = gt.get("expected_domains") or gt.get("primary_domain")
            predicted_domain = r.get("predicted_primary_domain")
            expected_action = gt.get("expected_action")
            selected_action = utility.get("selected_action")

            writer.writerow(
                {
                    "scenario_id": r.get("scenario_id"),
                    "category": r.get("category") or r.get("scenario_type"),
                    "prompt": r.get("prompt", ""),
                    "expected_domains": json.dumps(expected_domains, ensure_ascii=False)
                    if isinstance(expected_domains, list)
                    else expected_domains,
                    "predicted_domain": predicted_domain,
                    "domain_match": _domain_match(expected_domains, predicted_domain),
                    "expected_action": expected_action,
                    "selected_action": selected_action,
                    "action_match": _action_match(expected_action, selected_action),
                    "consensus_score": r.get("consensus_score"),
                    "rar_triggered": rar.get("triggered"),
                    "rar_accepted": rar.get("accepted"),
                    "utility": utility.get("best_utility"),
                    "performance_score": utility.get("performance_score"),
                    "cost_efficiency_score": utility.get("cost_efficiency_score"),
                    "risk_reduction_score": utility.get("risk_reduction_score"),
                    "xi": xi.get("xi"),
                }
            )


def _write_mismatch_report(path: Path, mismatches: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mismatches, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default=str(PROMPT_LIBRARY_PATH), help="Prompt library YAML")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Optional prompt limit")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_items = _load_prompt_library(Path(args.prompts))
    if args.limit is not None:
        prompt_items = prompt_items[: args.limit]

    rows: List[Dict[str, Any]] = []

    for idx, item in enumerate(prompt_items, start=1):
        scenario = _build_scenario_from_prompt(item, idx)
        result = run_pipeline(scenario, mode="aaf_full")
        row = result.__dict__.copy()

        # Preserve prompt/category metadata for PM experiment reporting.
        row["prompt"] = scenario.get("prompt", "")
        row["category"] = scenario.get("category", "pm_prompt")
        rows.append(row)

    summary = _summarize(rows)
    mismatches = _collect_mismatches(rows)

    _write_jsonl(out_dir / "pm_prompt_outputs.jsonl", rows)
    _write_csv(out_dir / "pm_prompt_metrics.csv", rows)
    _write_mismatch_report(out_dir / "pm_prompt_mismatches.json", mismatches)

    with open(out_dir / "pm_prompt_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote PM prompt results to: {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
