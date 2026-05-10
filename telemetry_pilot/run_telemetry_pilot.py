from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from pipeline import run_pipeline


RAW_DIR = Path("telemetry_pilot/raw")
OUT_DIR = Path("results_phase3_telemetry_pilot")

DEFAULT_THRESHOLDS = {
    "tau_consensus": 0.65,
    "delta_min": 0.05,
    "max_rar_loops": 2,
}

DEFAULT_UTILITY_WEIGHTS = (0.4, 0.3, 0.3)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _count_lines_containing(text: str, terms: List[str]) -> int:
    count = 0
    for line in text.lower().splitlines():
        if any(term.lower() in line for term in terms):
            count += 1
    return count


def _container_count(text: str, keyword: str) -> int:
    count = 0
    for line in text.lower().splitlines():
        if keyword.lower() in line and line.strip():
            count += 1
    return count


def _base_telemetry() -> Dict[str, Any]:
    return {
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


def _case(
    scenario_id: str,
    title: str,
    telemetry: Dict[str, Any],
    primary_domain: str,
    expected_domains: List[str],
    expected_action: str,
    root_cause: str,
    raw_artifacts: List[str],
) -> Dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "incident_id": scenario_id,
        "category": "phase3_sock_shop_telemetry_pilot",
        "scenario_type": title,
        "source": "sock_shop_runtime_artifacts",
        "raw_artifacts": raw_artifacts,
        "telemetry": telemetry,
        "ground_truth": {
            "primary_domain": primary_domain,
            "expected_domains": expected_domains,
            "secondary_domains": expected_domains[1:] if len(expected_domains) > 1 else [],
            "root_cause": root_cause,
            "expected_action": expected_action,
            "recommended_action": expected_action,
        },
        "thresholds": DEFAULT_THRESHOLDS.copy(),
        "utility_weights": DEFAULT_UTILITY_WEIGHTS,
        "lam": 0.5,
    }


def build_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # T01 baseline
    # ------------------------------------------------------------------
    t01_dir = RAW_DIR / "T01_baseline"
    t01_ps = _read_text(t01_dir / "docker_ps.txt")
    t01_stats = _read_text(t01_dir / "docker_stats.txt")
    t01_logs = _read_text(t01_dir / "front_end_logs.txt")

    t01 = _base_telemetry()
    baseline_errors = _count_lines_containing(t01_logs, ["error", "exception", "failed", "timeout"])
    running_containers = len([line for line in t01_ps.splitlines() if line.strip()]) - 1

    t01["sre"]["availability_pct"] = 99.9 if baseline_errors == 0 else 99.2
    t01["sre"]["error_rate_pct"] = min(5.0, baseline_errors * 0.5)
    t01["finops"]["hpa_scale_to"] = max(4, min(8, running_containers // 2 if running_containers > 0 else 4))

    cases.append(
        _case(
            scenario_id="P3-T01",
            title="sock_shop_baseline_runtime_health",
            telemetry=t01,
            primary_domain="DevOps",
            expected_domains=["DevOps"],
            expected_action="No action (observe)",
            root_cause="baseline_runtime_health",
            raw_artifacts=[
                str(t01_dir / "docker_ps.txt"),
                str(t01_dir / "compose_ps.txt"),
                str(t01_dir / "docker_stats.txt"),
                str(t01_dir / "front_end_logs.txt"),
            ],
        )
    )

    # ------------------------------------------------------------------
    # T02 service degradation: catalogue stopped
    # ------------------------------------------------------------------
    t02_dir = RAW_DIR / "T02_service_degradation"
    t02_ps = _read_text(t02_dir / "docker_ps_a.txt")
    t02_compose = _read_text(t02_dir / "compose_ps.txt")
    t02_front_logs = _read_text(t02_dir / "front_end_logs.txt")
    t02_catalogue_logs = _read_text(t02_dir / "catalogue_logs.txt")

    t02 = _base_telemetry()

    stopped_or_exited = _count_lines_containing(t02_ps + "\n" + t02_compose, ["exited", "stopped"])
    service_errors = _count_lines_containing(
        t02_front_logs + "\n" + t02_catalogue_logs,
        ["error", "exception", "failed", "timeout", "connection refused", "unavailable"],
    )

    if stopped_or_exited > 0:
        t02["deploy"]["restart_loops"] = 12
        t02["deploy"]["rollback_marker"] = True

    t02["sre"]["p95_latency_ms"] = 650.0 if service_errors > 0 or stopped_or_exited > 0 else 300.0
    t02["sre"]["error_rate_pct"] = 9.0 if service_errors > 0 or stopped_or_exited > 0 else 3.0
    t02["sre"]["availability_pct"] = 98.5 if stopped_or_exited > 0 else 99.2

    cases.append(
        _case(
            scenario_id="P3-T02",
            title="sock_shop_catalogue_service_degradation",
            telemetry=t02,
            primary_domain="SRE",
            expected_domains=["SRE", "DevOps"],
            expected_action="Mitigate and monitor",
            root_cause="catalogue_service_unavailable",
            raw_artifacts=[
                str(t02_dir / "docker_ps_a.txt"),
                str(t02_dir / "compose_ps.txt"),
                str(t02_dir / "docker_stats.txt"),
                str(t02_dir / "front_end_logs.txt"),
                str(t02_dir / "catalogue_logs.txt"),
            ],
        )
    )

    # ------------------------------------------------------------------
    # T03 resource scaling / cost proxy
    # ------------------------------------------------------------------
    t03_dir = RAW_DIR / "T03_resource_scaling"
    t03_ps = _read_text(t03_dir / "docker_ps.txt")
    t03_compose = _read_text(t03_dir / "compose_ps.txt")
    t03_stats = _read_text(t03_dir / "docker_stats.txt")
    t03_logs = _read_text(t03_dir / "front_end_logs.txt")

    t03 = _base_telemetry()

    frontend_count = max(
        _container_count(t03_ps, "front"),
        _container_count(t03_compose, "front"),
    )

    # Cost proxy: extra frontend replicas increase cost.
    if frontend_count >= 3:
        t03["finops"]["cost_spike_pct"] = 35.0
        t03["finops"]["hpa_scale_to"] = frontend_count
        t03["finops"]["cpu_request_increase_pct"] = 50.0
        t03["finops"]["memory_request_increase_pct"] = 40.0
    else:
        t03["finops"]["cost_spike_pct"] = 18.0
        t03["finops"]["hpa_scale_to"] = max(4, frontend_count)

    t03["sre"]["p95_latency_ms"] = 240.0
    t03["sre"]["error_rate_pct"] = 1.0
    t03["sre"]["availability_pct"] = 99.8

    cases.append(
        _case(
            scenario_id="P3-T03",
            title="sock_shop_frontend_resource_scaling_cost_proxy",
            telemetry=t03,
            primary_domain="FinOps",
            expected_domains=["FinOps"],
            expected_action="Scale adjustment",
            root_cause="resource_scaling_cost_proxy",
            raw_artifacts=[
                str(t03_dir / "docker_ps.txt"),
                str(t03_dir / "compose_ps.txt"),
                str(t03_dir / "docker_stats.txt"),
                str(t03_dir / "front_end_logs.txt"),
            ],
        )
    )

    return cases


def _domain_match(expected: List[str] | str | None, predicted: str | None) -> bool:
    if not expected or not predicted:
        return False

    if isinstance(expected, list):
        expected_values = [str(x).lower() for x in expected]
    else:
        expected_values = [str(expected).lower()]

    return str(predicted).lower() in expected_values


def _normalize_action(action: str | None) -> str:
    if not action:
        return ""
    a = str(action).lower()

    if "rollback" in a:
        return "rollback"
    if "patch" in a or "block" in a:
        return "patch_block"
    if "scale" in a:
        return "scale"
    if "mitigate" in a or "monitor" in a:
        return "mitigate_monitor"
    if "review" in a:
        return "review"
    if "observe" in a or "no action" in a:
        return "observe"

    return a


def _action_match(expected: str | None, predicted: str | None) -> bool:
    return bool(expected and predicted and _normalize_action(expected) == _normalize_action(predicted))


def _mean(values: List[float | bool]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "scenario_id",
        "scenario_type",
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
            expected_action = gt.get("expected_action")
            predicted_domain = r.get("predicted_primary_domain")
            selected_action = utility.get("selected_action")

            writer.writerow(
                {
                    "scenario_id": r.get("scenario_id"),
                    "scenario_type": r.get("scenario_type"),
                    "expected_domains": json.dumps(expected_domains),
                    "predicted_domain": predicted_domain,
                    "domain_match": _domain_match(expected_domains, predicted_domain),
                    "expected_action": expected_action,
                    "selected_action": selected_action,
                    "action_match": _action_match(expected_action, selected_action),
                    "consensus_score": r.get("consensus_score"),
                    "rar_triggered": rar.get("triggered"),
                    "rar_accepted": rar.get("accepted"),
                    "utility": utility.get("best_utility"),
                    "xi": xi.get("xi"),
                }
            )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cases = build_cases()
    rows: List[Dict[str, Any]] = []

    for case in cases:
        result = run_pipeline(case, mode="aaf_full")
        row = result.__dict__.copy()
        row["source"] = case.get("source")
        row["raw_artifacts"] = case.get("raw_artifacts", [])
        rows.append(row)

    domain_matches = [
        _domain_match(
            (r.get("ground_truth") or {}).get("expected_domains"),
            r.get("predicted_primary_domain"),
        )
        for r in rows
    ]

    action_matches = [
        _action_match(
            (r.get("ground_truth") or {}).get("expected_action"),
            (r.get("utility") or {}).get("selected_action"),
        )
        for r in rows
    ]

    summary = {
        "n": len(rows),
        "benchmark": "Sock Shop",
        "pilot_type": "local runtime artifact telemetry pilot",
        "domain_match_rate": _mean(domain_matches),
        "action_match_rate": _mean(action_matches),
        "consensus_mean": _mean([float(r.get("consensus_score", 0.0) or 0.0) for r in rows]),
        "utility_mean": _mean([float((r.get("utility") or {}).get("best_utility", 0.0) or 0.0) for r in rows]),
        "xi_mean": _mean([float((r.get("explainability") or {}).get("xi", 0.0) or 0.0) for r in rows]),
        "rar_triggered": sum(1 for r in rows if (r.get("rar") or {}).get("triggered")),
        "rar_accepted": sum(1 for r in rows if (r.get("rar") or {}).get("accepted")),
        "raw_artifact_root": str(RAW_DIR),
        "cases": [r.get("scenario_id") for r in rows],
    }

    _write_jsonl(OUT_DIR / "telemetry_pilot_outputs.jsonl", rows)
    _write_csv(OUT_DIR / "telemetry_pilot_metrics.csv", rows)

    with open(OUT_DIR / "telemetry_pilot_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote telemetry pilot results to: {OUT_DIR}")


if __name__ == "__main__":
    main()
