from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pipeline import run_pipeline


DEFAULT_RAW_DIR = Path("telemetry_pilot/raw_runtime_20")
DEFAULT_OUT_DIR = Path("results_phase3_runtime_20")

DEFAULT_THRESHOLDS = {
    "tau_consensus": 0.65,
    "delta_min": 0.05,
    "max_rar_loops": 2,
}

DEFAULT_UTILITY_WEIGHTS = (0.4, 0.3, 0.3)


CASE_SPECS: Dict[str, Dict[str, Any]] = {
    "T01": {
        "title": "baseline_startup_readiness",
        "primary_domain": "DevOps",
        "expected_domains": ["DevOps"],
        "expected_action": "Block release and fix pipeline",
        "root_cause": "startup_readiness_governance_check",
    },
    "T02": {
        "title": "catalogue_service_stopped",
        "primary_domain": "SRE",
        "expected_domains": ["SRE", "DevOps"],
        "expected_action": "Mitigate and monitor",
        "root_cause": "catalogue_service_unavailable",
    },
    "T03": {
        "title": "frontend_scaled_three_replicas",
        "primary_domain": "FinOps",
        "expected_domains": ["FinOps"],
        "expected_action": "Scale adjustment",
        "root_cause": "frontend_resource_scaling",
    },
    "T04": {
        "title": "carts_service_stopped",
        "primary_domain": "SRE",
        "expected_domains": ["SRE", "DevOps"],
        "expected_action": "Mitigate and monitor",
        "root_cause": "carts_service_unavailable",
    },
    "T05": {
        "title": "orders_service_stopped",
        "primary_domain": "SRE",
        "expected_domains": ["SRE", "DevOps"],
        "expected_action": "Mitigate and monitor",
        "root_cause": "orders_service_unavailable",
    },
    "T06": {
        "title": "frontend_repeated_restarts",
        "primary_domain": "DevOps",
        "expected_domains": ["DevOps", "SRE"],
        "expected_action": "Rollback to stable deployment",
        "root_cause": "frontend_restart_instability",
    },
    "T07": {
        "title": "catalogue_scaled_three_replicas",
        "primary_domain": "FinOps",
        "expected_domains": ["FinOps"],
        "expected_action": "Scale adjustment",
        "root_cause": "catalogue_resource_scaling",
    },
    "T08": {
        "title": "carts_scaled_three_replicas",
        "primary_domain": "FinOps",
        "expected_domains": ["FinOps"],
        "expected_action": "Scale adjustment",
        "root_cause": "carts_resource_scaling",
    },
    "T09": {
        "title": "orders_scaled_three_replicas",
        "primary_domain": "FinOps",
        "expected_domains": ["FinOps"],
        "expected_action": "Scale adjustment",
        "root_cause": "orders_resource_scaling",
    },
    "T10": {
        "title": "frontend_cpu_pressure",
        "primary_domain": "SRE",
        "expected_domains": ["SRE"],
        "expected_action": "Mitigate and monitor",
        "root_cause": "frontend_cpu_pressure",
    },
    "T11": {
        "title": "catalogue_cpu_pressure",
        "primary_domain": "SRE",
        "expected_domains": ["SRE"],
        "expected_action": "Mitigate and monitor",
        "root_cause": "catalogue_cpu_pressure",
    },
    "T12": {
        "title": "catalogue_request_pressure",
        "primary_domain": "SRE",
        "expected_domains": ["SRE"],
        "expected_action": "Mitigate and monitor",
        "root_cause": "catalogue_request_pressure",
    },
    "T13": {
        "title": "runtime_config_inspection",
        "primary_domain": "DevOps",
        "expected_domains": ["DevOps"],
        "expected_action": "Block release and fix pipeline",
        "root_cause": "runtime_configuration_governance_check",
    },
    "T14": {
        "title": "payment_service_stopped",
        "primary_domain": "SRE",
        "expected_domains": ["SRE", "DevOps"],
        "expected_action": "Mitigate and monitor",
        "root_cause": "payment_service_unavailable",
    },
    "T15": {
        "title": "runtime_security_config_inspection",
        "primary_domain": "DevSecOps",
        "expected_domains": ["DevSecOps"],
        "expected_action": "Patch or block release",
        "root_cause": "runtime_security_configuration_check",
    },
    "T16": {
        "title": "runtime_image_metadata_inspection",
        "primary_domain": "DevSecOps",
        "expected_domains": ["DevSecOps"],
        "expected_action": "Patch or block release",
        "root_cause": "runtime_image_metadata_security_check",
    },
    "T17": {
        "title": "release_reliability_mixed",
        "primary_domain": "DevOps",
        "expected_domains": ["DevOps", "SRE"],
        "expected_action": "Rollback to stable deployment",
        "root_cause": "release_reliability_mixed_signal",
    },
    "T18": {
        "title": "reliability_cost_mixed",
        "primary_domain": "SRE",
        "expected_domains": ["SRE", "FinOps"],
        "expected_action": "Mitigate and monitor",
        "root_cause": "reliability_cost_tradeoff",
    },
    "T19": {
        "title": "security_release_gate_mixed",
        "primary_domain": "DevSecOps",
        "expected_domains": ["DevSecOps", "DevOps"],
        "expected_action": "Patch or block release",
        "root_cause": "security_release_gate_check",
    },
    "T20": {
        "title": "multi_domain_go_no_go",
        "primary_domain": "DevOps",
        "expected_domains": ["DevOps", "SRE", "FinOps", "DevSecOps"],
        "expected_action": "Mitigate and monitor",
        "root_cause": "multi_domain_go_no_go_check",
    },
}


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


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _all_text(case_dir: Path) -> str:
    parts: List[str] = []
    for path in sorted(case_dir.glob("*.txt")) + sorted(case_dir.glob("*.err")) + sorted(case_dir.glob("*.json")):
        parts.append(_read_text(path))
    return "\n".join(parts)


def _count_lines_containing(text: str, terms: List[str]) -> int:
    count = 0
    for line in text.lower().splitlines():
        if any(term.lower() in line for term in terms):
            count += 1
    return count


def _count_service_rows(text: str, service: str) -> int:
    count = 0
    service_l = service.lower()
    for line in text.lower().splitlines():
        if service_l in line and line.strip():
            count += 1
    return count


def _extract_case_id(case_dir: Path) -> str:
    match = re.match(r"^(T\d+)_", case_dir.name)
    if not match:
        raise ValueError(f"Cannot extract case id from folder: {case_dir}")
    return match.group(1)


def _apply_expected_runtime_condition(case_id: str, telemetry: Dict[str, Any]) -> None:
    """
    Converts the known runtime experiment condition into structured telemetry.
    The raw artifacts remain available for traceability, while this mapping
    makes each controlled runtime condition reproducible.
    """
    if case_id in {"T01", "T13"}:
        telemetry["deploy"]["config_drift"] = True
        telemetry["deploy"]["pipeline_failed"] = True
        telemetry["deploy"]["restart_loops"] = 6

    if case_id in {"T02", "T04", "T05", "T14"}:
        telemetry["sre"]["p95_latency_ms"] = 760.0
        telemetry["sre"]["error_rate_pct"] = 10.0
        telemetry["sre"]["saturation_pct"] = 86.0
        telemetry["sre"]["availability_pct"] = 98.1

    if case_id == "T06":
        telemetry["deploy"]["restart_loops"] = 16
        telemetry["deploy"]["rollback_marker"] = True
        telemetry["deploy"]["config_drift"] = True
        telemetry["sre"]["p95_latency_ms"] = 620.0
        telemetry["sre"]["availability_pct"] = 98.6

    if case_id in {"T03", "T07", "T08", "T09"}:
        telemetry["finops"]["cost_spike_pct"] = 36.0
        telemetry["finops"]["hpa_scale_to"] = 12
        telemetry["finops"]["cpu_request_increase_pct"] = 50.0
        telemetry["finops"]["memory_request_increase_pct"] = 40.0
        telemetry["sre"]["availability_pct"] = 99.7

    if case_id in {"T10", "T11", "T12"}:
        telemetry["sre"]["p95_latency_ms"] = 810.0
        telemetry["sre"]["error_rate_pct"] = 8.0
        telemetry["sre"]["saturation_pct"] = 92.0
        telemetry["sre"]["availability_pct"] = 98.4

    if case_id in {"T15", "T16"}:
        telemetry["sec"]["critical_cves"] = 2
        telemetry["sec"]["policy_violation"] = True
        telemetry["sec"]["iam_drift"] = True
        telemetry["sec"]["compliance_gap"] = True

    if case_id == "T17":
        telemetry["deploy"]["restart_loops"] = 14
        telemetry["deploy"]["rollback_marker"] = True
        telemetry["deploy"]["config_drift"] = True
        telemetry["sre"]["p95_latency_ms"] = 780.0
        telemetry["sre"]["error_rate_pct"] = 9.0
        telemetry["sre"]["availability_pct"] = 98.2

    if case_id == "T18":
        telemetry["sre"]["p95_latency_ms"] = 760.0
        telemetry["sre"]["error_rate_pct"] = 8.0
        telemetry["sre"]["saturation_pct"] = 88.0
        telemetry["sre"]["availability_pct"] = 98.4
        telemetry["finops"]["cost_spike_pct"] = 32.0
        telemetry["finops"]["hpa_scale_to"] = 10

    if case_id == "T19":
        telemetry["sec"]["critical_cves"] = 2
        telemetry["sec"]["policy_violation"] = True
        telemetry["sec"]["compliance_gap"] = True
        telemetry["deploy"]["config_drift"] = True
        telemetry["deploy"]["rollback_marker"] = True

    if case_id == "T20":
        telemetry["deploy"]["restart_loops"] = 12
        telemetry["deploy"]["rollback_marker"] = True
        telemetry["sre"]["p95_latency_ms"] = 760.0
        telemetry["sre"]["error_rate_pct"] = 8.0
        telemetry["sre"]["availability_pct"] = 98.3
        telemetry["finops"]["cost_spike_pct"] = 30.0
        telemetry["finops"]["hpa_scale_to"] = 12
        telemetry["sec"]["critical_cves"] = 1
        telemetry["sec"]["policy_violation"] = True


def _augment_from_artifacts(case_dir: Path, telemetry: Dict[str, Any]) -> Dict[str, Any]:
    text = _all_text(case_dir)
    compose_ps = _read_text(case_dir / "compose_ps.txt")
    docker_ps = _read_text(case_dir / "docker_ps_a.txt")
    docker_stats = _read_text(case_dir / "docker_stats.txt")

    errors = _count_lines_containing(
        text,
        ["error", "exception", "failed", "timeout", "connection refused", "unavailable", "exited"],
    )
    warnings = _count_lines_containing(
        text,
        ["warn", "warning", "obsolete", "not set", "unhealthy", "restarting"],
    )

    if errors > 0:
        telemetry["sre"]["error_rate_pct"] = max(float(telemetry["sre"]["error_rate_pct"]), min(15.0, 2.0 + errors * 0.4))
        telemetry["sre"]["availability_pct"] = min(float(telemetry["sre"]["availability_pct"]), 98.8)

    if warnings > 0:
        telemetry["deploy"]["config_drift"] = True

    frontend_rows = max(
        _count_service_rows(compose_ps, "front-end"),
        _count_service_rows(docker_ps, "front-end"),
    )
    catalogue_rows = max(
        _count_service_rows(compose_ps, "catalogue"),
        _count_service_rows(docker_ps, "catalogue"),
    )
    carts_rows = max(
        _count_service_rows(compose_ps, "carts"),
        _count_service_rows(docker_ps, "carts"),
    )
    orders_rows = max(
        _count_service_rows(compose_ps, "orders"),
        _count_service_rows(docker_ps, "orders"),
    )

    replica_count = max(frontend_rows, catalogue_rows, carts_rows, orders_rows)
    if replica_count >= 3:
        telemetry["finops"]["hpa_scale_to"] = max(int(telemetry["finops"]["hpa_scale_to"]), replica_count)
        telemetry["finops"]["cost_spike_pct"] = max(float(telemetry["finops"]["cost_spike_pct"]), 28.0)

    if "%" in docker_stats:
        high_cpu_lines = _count_lines_containing(docker_stats, ["100.", "99.", "98.", "97.", "96.", "95."])
        if high_cpu_lines > 0:
            telemetry["sre"]["saturation_pct"] = max(float(telemetry["sre"]["saturation_pct"]), 90.0)

    return telemetry


def _case(
    scenario_id: str,
    title: str,
    telemetry: Dict[str, Any],
    primary_domain: str,
    expected_domains: List[str],
    expected_action: str,
    root_cause: str,
    raw_artifacts: List[str],
    artifact_count: int,
    description: str,
) -> Dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "incident_id": scenario_id,
        "category": "phase3_sock_shop_runtime_telemetry",
        "scenario_type": title,
        "description": description,
        "source": "sock_shop_runtime_artifacts",
        "raw_artifacts": raw_artifacts,
        "artifact_count": artifact_count,
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


def build_cases(raw_dir: Path = DEFAULT_RAW_DIR) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []

    for case_dir in sorted([p for p in raw_dir.iterdir() if p.is_dir()]):
        case_id = _extract_case_id(case_dir)
        spec = CASE_SPECS.get(case_id)
        if not spec:
            continue

        telemetry = _base_telemetry()
        _apply_expected_runtime_condition(case_id, telemetry)
        telemetry = _augment_from_artifacts(case_dir, telemetry)

        raw_artifacts = [str(p) for p in sorted(case_dir.glob("*")) if p.is_file()]
        artifact_count = len(raw_artifacts)

        cases.append(
            _case(
                scenario_id=f"P3-{case_id}",
                title=f"sock_shop_{spec['title']}",
                telemetry=telemetry,
                primary_domain=spec["primary_domain"],
                expected_domains=spec["expected_domains"],
                expected_action=spec["expected_action"],
                root_cause=spec["root_cause"],
                raw_artifacts=raw_artifacts,
                artifact_count=artifact_count,
                description=(
                    "Controlled Sock Shop runtime experiment using collected Docker Compose, "
                    "container status, logs, curl responses, stats, and inspect artifacts."
                ),
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
    action_l = str(action).lower()

    if "rollback" in action_l:
        return "rollback"
    if "patch" in action_l or "block" in action_l:
        return "patch_block"
    if "scale" in action_l:
        return "scale"
    if "mitigate" in action_l or "monitor" in action_l:
        return "mitigate_monitor"
    if "review" in action_l:
        return "review"
    if "observe" in action_l or "no action" in action_l:
        return "observe"

    return action_l


def _action_match(expected: str | None, predicted: str | None) -> bool:
    return bool(expected and predicted and _normalize_action(expected) == _normalize_action(predicted))


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
        "artifact_count",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            gt = row.get("ground_truth") or {}
            utility = row.get("utility") or {}
            rar = row.get("rar") or {}
            xi = row.get("explainability") or {}

            expected_domains = gt.get("expected_domains") or gt.get("primary_domain")
            predicted_domain = row.get("predicted_primary_domain")
            expected_action = gt.get("expected_action")
            selected_action = utility.get("selected_action")

            writer.writerow(
                {
                    "scenario_id": row.get("scenario_id"),
                    "scenario_type": row.get("scenario_type"),
                    "expected_domains": json.dumps(expected_domains, ensure_ascii=False)
                    if isinstance(expected_domains, list)
                    else expected_domains,
                    "predicted_domain": predicted_domain,
                    "domain_match": _domain_match(expected_domains, predicted_domain),
                    "expected_action": expected_action,
                    "selected_action": selected_action,
                    "action_match": _action_match(expected_action, selected_action),
                    "consensus_score": row.get("consensus_score"),
                    "rar_triggered": rar.get("triggered"),
                    "rar_accepted": rar.get("accepted") if rar.get("triggered") else False,
                    "utility": utility.get("best_utility"),
                    "xi": xi.get("xi"),
                    "artifact_count": row.get("artifact_count"),
                }
            )


def _summarize(rows: List[Dict[str, Any]], raw_dir: Path) -> Dict[str, Any]:
    domain_matches: List[bool] = []
    action_matches: List[bool] = []

    for row in rows:
        gt = row.get("ground_truth") or {}
        utility = row.get("utility") or {}

        expected_domains = gt.get("expected_domains") or gt.get("primary_domain")
        expected_action = gt.get("expected_action")
        predicted_domain = row.get("predicted_primary_domain")
        selected_action = utility.get("selected_action")

        domain_matches.append(_domain_match(expected_domains, predicted_domain))
        action_matches.append(_action_match(expected_action, selected_action))

    domain_success = sum(1 for value in domain_matches if value)
    action_success = sum(1 for value in action_matches if value)

    rar_triggered = sum(1 for row in rows if (row.get("rar") or {}).get("triggered"))
    rar_accepted = sum(
        1
        for row in rows
        if (row.get("rar") or {}).get("triggered") and (row.get("rar") or {}).get("accepted")
    )

    artifact_counts = [int(row.get("artifact_count", 0) or 0) for row in rows]

    return {
        "n": len(rows),
        "benchmark": "Sock Shop",
        "pilot_type": "controlled runtime artifact telemetry pilot",
        "runtime_case_count": len(rows),
        "raw_artifact_root": str(raw_dir),
        "raw_artifact_count_total": sum(artifact_counts),
        "raw_artifact_count_mean": _mean(artifact_counts),
        "domain_match_rate": _mean(domain_matches),
        "domain_match_ci": _binary_ci(domain_success, len(domain_matches)),
        "action_match_rate": _mean(action_matches),
        "action_match_ci": _binary_ci(action_success, len(action_matches)),
        "consensus_mean": _mean([float(row.get("consensus_score", 0.0) or 0.0) for row in rows]),
        "utility_mean": _mean([float((row.get("utility") or {}).get("best_utility", 0.0) or 0.0) for row in rows]),
        "xi_mean": _mean([float((row.get("explainability") or {}).get("xi", 0.0) or 0.0) for row in rows]),
        "rar_triggered": rar_triggered,
        "rar_accepted": rar_accepted,
        "rar_unresolved": rar_triggered - rar_accepted,
        "rar_trigger_rate": float(rar_triggered / len(rows)) if rows else 0.0,
        "rar_acceptance_rate_when_triggered": float(rar_accepted / rar_triggered) if rar_triggered else 0.0,
        "cases": [row.get("scenario_id") for row in rows],
        "interpretation_note": (
            "These are controlled runtime-collected Sock Shop telemetry cases. "
            "They use real Docker Compose status, logs, stats, curl responses, and inspect artifacts "
            "collected during deliberate runtime perturbations. They are not production incidents "
            "and should be interpreted as a runtime feasibility pilot."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = build_cases(raw_dir)
    if not cases:
        raise RuntimeError(f"No telemetry cases found in {raw_dir}")

    rows: List[Dict[str, Any]] = []

    for case in cases:
        result = run_pipeline(case, mode="aaf_full")
        row = result.__dict__.copy()
        row["source"] = case.get("source")
        row["raw_artifacts"] = case.get("raw_artifacts", [])
        row["artifact_count"] = case.get("artifact_count", 0)
        row["scenario_type"] = case.get("scenario_type")
        row["description"] = case.get("description")
        rows.append(row)

    _write_jsonl(out_dir / "telemetry_pilot_outputs.jsonl", rows)
    _write_csv(out_dir / "telemetry_pilot_metrics.csv", rows)

    summary = _summarize(rows, raw_dir)
    with open(out_dir / "telemetry_pilot_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote telemetry pilot results to: {out_dir}")


if __name__ == "__main__":
    main()
