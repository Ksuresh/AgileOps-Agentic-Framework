from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

from pipeline import run_pipeline


PROMPT_LIBRARY_PATH = Path("prompts/pm_prompt_library.yaml")
OUT_DIR = Path("results_pm_prompts")

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
        raise ValueError(
            "Prompt library must be either a list or a dict with a 'prompts' key."
        )

    if not isinstance(prompts, list):
        raise ValueError("Prompt library 'prompts' must be a list.")

    return prompts


def _safe_get_prompt_text(item: Dict[str, Any]) -> str:
    for key in ["prompt", "text", "pm_prompt", "query"]:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _safe_get_expected_domain(item: Dict[str, Any]) -> str | None:
    for key in ["expected_domain", "primary_domain", "domain"]:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    expected = item.get("expected")
    if isinstance(expected, dict):
        value = expected.get("primary_domain") or expected.get("domain")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


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

    If the repository's richer prompt routing modules are unavailable or their
    interface changes, this fallback still creates a reproducible PM prompt
    experiment. It intentionally maps only explicit prompt symptoms.
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

    # DevOps / release signals
    if any(k in p for k in ["deployment", "release", "rollback", "pipeline", "build", "artifact"]):
        telemetry["deploy"]["pipeline_failed"] = "pipeline" in p or "build" in p
        telemetry["deploy"]["rollback_marker"] = "rollback" in p or "release" in p
        telemetry["deploy"]["artifact_mismatch"] = "artifact" in p or "image" in p
        telemetry["deploy"]["restart_loops"] = 12 if "restart" in p or "crash" in p else 8

    if any(k in p for k in ["config", "configuration", "drift"]):
        telemetry["deploy"]["config_drift"] = True

    # Reliability signals
    if any(k in p for k in ["latency", "slow", "timeout", "performance"]):
        telemetry["sre"]["p95_latency_ms"] = 650.0
        telemetry["sre"]["availability_pct"] = 98.4

    if any(k in p for k in ["error", "5xx", "failure rate", "failed requests"]):
        telemetry["sre"]["error_rate_pct"] = 10.0
        telemetry["sre"]["availability_pct"] = 98.0

    if any(k in p for k in ["cpu", "memory", "saturation", "capacity"]):
        telemetry["sre"]["saturation_pct"] = 90.0
        telemetry["sre"]["p95_latency_ms"] = max(telemetry["sre"]["p95_latency_ms"], 520.0)

    # FinOps signals
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

    # Security / compliance signals
    if any(k in p for k in ["security", "vulnerability", "cve", "critical cve"]):
        telemetry["sec"]["critical_cves"] = 2

    if any(k in p for k in ["policy", "opa", "gatekeeper", "compliance"]):
        telemetry["sec"]["policy_violation"] = True
        telemetry["sec"]["compliance_gap"] = True

    if any(k in p for k in ["iam", "permission", "access drift"]):
        telemetry["sec"]["iam_drift"] = True

    return telemetry


def _build_scenario_from_prompt(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    prompt = _safe_get_prompt_text(item)
    if not prompt:
        prompt = f"Prompt scenario {idx}"

    expected_domain = _safe_get_expected_domain(item)
    expected_action = _safe_get_expected_action(item)

    telemetry = None

    # Prefer existing project modules if available.
    try:
        from pm_interface.prompt_router import route_prompt
        from simulation.prompt_to_telemetry import prompt_to_telemetry

        routed = route_prompt(prompt)
        telemetry = prompt_to_telemetry(routed)
    except Exception:
        telemetry = _prompt_to_basic_telemetry(prompt)

    if not isinstance(telemetry, dict):
        telemetry = _prompt_to_basic_telemetry(prompt)

    scenario_id = item.get("id") or item.get("scenario_id") or f"PM-{idx:03d}"

    scenario = {
        "scenario_id": str(scenario_id),
        "incident_id": str(scenario_id),
        "category": item.get("category", "pm_prompt"),
        "scenario_type": item.get("category", "pm_prompt"),
        "prompt": prompt,
        "telemetry": telemetry,
        "ground_truth": {
            "primary_domain": expected_domain,
            "secondary_domains": item.get("secondary_domains", []),
            "root_cause": item.get("root_cause", item.get("category", "pm_prompt")),
            "recommended_action": expected_action,
            "expected_action": expected_action,
        },
        "thresholds": DEFAULT_THRESHOLDS.copy(),
        "utility_weights": DEFAULT_UTILITY_WEIGHTS,
        "lam": 0.5,
    }

    return scenario


def _domain_match(gt: str | None, pred: str | None) -> bool:
    if not gt or not pred:
        return False
    return gt.strip().lower() == pred.strip().lower()


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


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)

    domain_matches = [
        _domain_match(
            (r.get("ground_truth") or {}).get("primary_domain"),
            r.get("predicted_primary_domain"),
        )
        for r in rows
        if (r.get("ground_truth") or {}).get("primary_domain")
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
    xi_values = [float((r.get("explainability") or {}).get("xi", 0.0) or 0.0) for r in rows]

    rar_triggered = sum(1 for r in rows if (r.get("rar") or {}).get("triggered"))
    rar_accepted = sum(1 for r in rows if (r.get("rar") or {}).get("accepted"))

    def mean(values: List[float | bool]) -> float:
        if not values:
            return 0.0
        return float(sum(float(v) for v in values) / len(values))

    return {
        "n": n,
        "domain_match_rate": mean(domain_matches),
        "domain_match_n": len(domain_matches),
        "action_match_rate": mean(action_matches),
        "action_match_n": len(action_matches),
        "consensus_mean": mean(consensus_values),
        "utility_mean": mean(utility_values),
        "xi_mean": mean(xi_values),
        "rar_triggered": rar_triggered,
        "rar_accepted": rar_accepted,
        "rar_trigger_rate": float(rar_triggered / n) if n else 0.0,
        "rar_acceptance_rate_when_triggered": float(rar_accepted / rar_triggered) if rar_triggered else 0.0,
        "thresholds": DEFAULT_THRESHOLDS,
        "utility_weights": DEFAULT_UTILITY_WEIGHTS,
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
        "expected_domain",
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

            expected_domain = gt.get("primary_domain")
            predicted_domain = r.get("predicted_primary_domain")
            expected_action = gt.get("expected_action")
            selected_action = utility.get("selected_action")

            writer.writerow(
                {
                    "scenario_id": r.get("scenario_id"),
                    "category": r.get("category") or r.get("scenario_type"),
                    "prompt": r.get("prompt", ""),
                    "expected_domain": expected_domain,
                    "predicted_domain": predicted_domain,
                    "domain_match": _domain_match(expected_domain, predicted_domain),
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prompt_items = _load_prompt_library()
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

    _write_jsonl(OUT_DIR / "pm_prompt_outputs.jsonl", rows)
    _write_csv(OUT_DIR / "pm_prompt_metrics.csv", rows)

    with open(OUT_DIR / "pm_prompt_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote PM prompt results to: {OUT_DIR}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
