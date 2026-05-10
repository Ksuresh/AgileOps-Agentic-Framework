from __future__ import annotations

import copy
import json
import random
from typing import Any, Dict, List


DEFAULT_THRESHOLDS = {
    "tau_consensus": 0.65,
    "delta_min": 0.05,
    "max_rar_loops": 2,
}

DEFAULT_UTILITY_WEIGHTS = (0.4, 0.3, 0.3)


def _scenario(
    scenario_id: str,
    category: str,
    primary_domain: str,
    root_cause: str,
    expected_action: str,
    telemetry: Dict[str, Any],
    secondary_domains: List[str] | None = None,
    priority: str = "medium",
) -> Dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "incident_id": scenario_id,
        "category": category,
        "scenario_type": category,
        "priority": priority,
        "telemetry": telemetry,
        "ground_truth": {
            "primary_domain": primary_domain,
            "secondary_domains": secondary_domains or [],
            "root_cause": root_cause,
            "recommended_action": expected_action,
            "expected_action": expected_action,
        },
        "thresholds": copy.deepcopy(DEFAULT_THRESHOLDS),
        "utility_weights": DEFAULT_UTILITY_WEIGHTS,
        "lam": 0.5,
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


def _apply_missing_or_noise(
    scenario: Dict[str, Any],
    rng: random.Random,
    missing_p: float,
    jitter_pct: float,
) -> Dict[str, Any]:
    """
    Adds controlled noise without changing the ground truth.
    Missing evidence is used to test RAR.
    """
    s = copy.deepcopy(scenario)
    telemetry = s["telemetry"]

    # Controlled metric jitter.
    for domain in ["sre", "finops"]:
        for key, value in telemetry[domain].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                jitter = 1.0 + rng.uniform(-jitter_pct, jitter_pct)
                telemetry[domain][key] = round(float(value) * jitter, 3)

    # Mark one evidence block as partially missing.
    if rng.random() < missing_p:
        primary = s["ground_truth"]["primary_domain"]
        domain_map = {
            "DevOps": "deploy",
            "SRE": "sre",
            "FinOps": "finops",
            "DevSecOps": "sec",
        }
        missing_domain = domain_map.get(primary, rng.choice(["deploy", "sre", "finops", "sec"]))
        telemetry[missing_domain]["_missing"] = True

    return s


def generate_scenarios(seed: int = 42, noise: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """
    Generates 30 deterministic governance scenarios.

    Design principle:
    - Each scenario has a clear primary domain.
    - Secondary impacts may exist, but they do not override the root cause.
    - Ground truth is assigned by scenario design, not by random threshold priority.
    """
    noise = noise or {}
    rng = random.Random(seed)

    missing_p = float(noise.get("missing_evidence_prob", 0.20))
    jitter_pct = float(noise.get("metric_jitter_pct", 0.05))

    scenarios: List[Dict[str, Any]] = []

    def add(
        category: str,
        primary_domain: str,
        root_cause: str,
        expected_action: str,
        modifications: Dict[str, Dict[str, Any]],
        secondary_domains: List[str] | None = None,
        priority: str = "medium",
    ) -> None:
        idx = len(scenarios) + 1
        telemetry = _base_telemetry()
        for domain, fields in modifications.items():
            telemetry[domain].update(fields)

        sc = _scenario(
            scenario_id=f"TC-{idx:02d}",
            category=category,
            primary_domain=primary_domain,
            root_cause=root_cause,
            expected_action=expected_action,
            telemetry=telemetry,
            secondary_domains=secondary_domains,
            priority=priority,
        )
        scenarios.append(_apply_missing_or_noise(sc, rng, missing_p, jitter_pct))

    # ------------------------------------------------------------------
    # 1. Deployment failures - DevOps primary
    # ------------------------------------------------------------------
    add(
        "deployment_failure",
        "DevOps",
        "bad_image_tag",
        "Rollback to stable deployment",
        {
            "deploy": {"pipeline_failed": True, "rollback_marker": True, "artifact_mismatch": True},
            "sre": {"error_rate_pct": 8.0, "p95_latency_ms": 450.0},
        },
        ["SRE"],
        "high",
    )
    add(
        "deployment_failure",
        "DevOps",
        "configuration_drift",
        "Rollback to stable deployment",
        {
            "deploy": {"config_drift": True, "restart_loops": 12, "rollback_marker": True},
            "sre": {"error_rate_pct": 6.0, "p95_latency_ms": 390.0},
        },
        ["SRE"],
        "high",
    )
    add(
        "deployment_failure",
        "DevOps",
        "failed_pipeline_gate",
        "Block release and fix pipeline",
        {
            "deploy": {"pipeline_failed": True, "artifact_mismatch": True},
        },
        [],
        "medium",
    )
    add(
        "deployment_failure",
        "DevOps",
        "container_restart_after_release",
        "Rollback to stable deployment",
        {
            "deploy": {"restart_loops": 18, "rollback_marker": True},
            "sre": {"error_rate_pct": 10.0, "p95_latency_ms": 520.0},
            "finops": {"cost_spike_pct": 12.0, "hpa_scale_to": 7},
        },
        ["SRE", "FinOps"],
        "high",
    )

    # ------------------------------------------------------------------
    # 2. Cost anomalies - FinOps primary
    # ------------------------------------------------------------------
    add(
        "cost_anomaly",
        "FinOps",
        "autoscaling_cost_spike",
        "Scale adjustment",
        {
            "finops": {"cost_spike_pct": 40.0, "hpa_scale_to": 16},
            "sre": {"p95_latency_ms": 210.0, "error_rate_pct": 1.0},
        },
        [],
        "high",
    )
    add(
        "cost_anomaly",
        "FinOps",
        "over_provisioned_resources",
        "Scale adjustment",
        {
            "finops": {"cost_spike_pct": 28.0, "cpu_request_increase_pct": 60.0, "memory_request_increase_pct": 45.0},
        },
        [],
        "medium",
    )
    add(
        "cost_anomaly",
        "FinOps",
        "unused_capacity",
        "Scale adjustment",
        {
            "finops": {"cost_spike_pct": 22.0, "hpa_scale_to": 11},
            "sre": {"saturation_pct": 42.0},
        },
        [],
        "medium",
    )
    add(
        "cost_anomaly",
        "FinOps",
        "cost_increase_after_scaling_policy",
        "Review scaling policy",
        {
            "finops": {"cost_spike_pct": 35.0, "hpa_scale_to": 14},
            "deploy": {"config_drift": True},
        },
        ["DevOps"],
        "high",
    )

    # ------------------------------------------------------------------
    # 3. Security violations - DevSecOps primary
    # ------------------------------------------------------------------
    add(
        "security_violation",
        "DevSecOps",
        "critical_vulnerability",
        "Patch or block release",
        {
            "sec": {"critical_cves": 2},
        },
        [],
        "high",
    )
    add(
        "security_violation",
        "DevSecOps",
        "policy_as_code_violation",
        "Patch or block release",
        {
            "sec": {"policy_violation": True},
            "deploy": {"pipeline_failed": True},
        },
        ["DevOps"],
        "high",
    )
    add(
        "security_violation",
        "DevSecOps",
        "iam_drift_detected",
        "Patch or block release",
        {
            "sec": {"iam_drift": True},
        },
        [],
        "high",
    )
    add(
        "security_violation",
        "DevSecOps",
        "compliance_control_failure",
        "Patch or block release",
        {
            "sec": {"compliance_gap": True, "policy_violation": True},
        },
        [],
        "medium",
    )

    # ------------------------------------------------------------------
    # 4. Reliability degradations - SRE primary
    # ------------------------------------------------------------------
    add(
        "reliability_degradation",
        "SRE",
        "latency_spike",
        "Mitigate and monitor",
        {
            "sre": {"p95_latency_ms": 900.0, "error_rate_pct": 4.0, "saturation_pct": 80.0},
        },
        [],
        "high",
    )
    add(
        "reliability_degradation",
        "SRE",
        "error_rate_spike",
        "Mitigate and monitor",
        {
            "sre": {"p95_latency_ms": 420.0, "error_rate_pct": 14.0},
        },
        [],
        "high",
    )
    add(
        "reliability_degradation",
        "SRE",
        "resource_saturation",
        "Scale adjustment",
        {
            "sre": {"saturation_pct": 94.0, "p95_latency_ms": 650.0},
            "finops": {"cost_spike_pct": 10.0, "hpa_scale_to": 7},
        },
        ["FinOps"],
        "high",
    )
    add(
        "reliability_degradation",
        "SRE",
        "availability_drop",
        "Mitigate and monitor",
        {
            "sre": {"availability_pct": 97.5, "error_rate_pct": 9.0, "p95_latency_ms": 550.0},
        },
        [],
        "high",
    )

    # ------------------------------------------------------------------
    # 5. Compliance preparation - DevSecOps primary
    # ------------------------------------------------------------------
    add(
        "compliance_preparation",
        "DevSecOps",
        "audit_log_gap",
        "Patch or block release",
        {
            "sec": {"compliance_gap": True},
        },
        [],
        "medium",
    )
    add(
        "compliance_preparation",
        "DevSecOps",
        "iam_policy_mismatch",
        "Patch or block release",
        {
            "sec": {"iam_drift": True, "compliance_gap": True},
        },
        [],
        "medium",
    )
    add(
        "compliance_preparation",
        "DevSecOps",
        "release_evidence_missing",
        "Block release and fix pipeline",
        {
            "sec": {"compliance_gap": True},
            "deploy": {"pipeline_failed": True},
        },
        ["DevOps"],
        "medium",
    )

    # ------------------------------------------------------------------
    # 6. Resource scaling - SRE / FinOps mix
    # ------------------------------------------------------------------
    add(
        "resource_scaling",
        "SRE",
        "capacity_exhaustion",
        "Scale adjustment",
        {
            "sre": {"saturation_pct": 92.0, "p95_latency_ms": 700.0},
            "finops": {"hpa_scale_to": 11, "cost_spike_pct": 18.0},
        },
        ["FinOps"],
        "high",
    )
    add(
        "resource_scaling",
        "FinOps",
        "unnecessary_scale_out",
        "Scale adjustment",
        {
            "finops": {"hpa_scale_to": 16, "cost_spike_pct": 32.0},
            "sre": {"p95_latency_ms": 190.0, "error_rate_pct": 0.5, "saturation_pct": 45.0},
        },
        ["SRE"],
        "medium",
    )
    add(
        "resource_scaling",
        "SRE",
        "high_cpu_trend",
        "Scale adjustment",
        {
            "sre": {"saturation_pct": 88.0, "p95_latency_ms": 500.0},
            "finops": {"cost_spike_pct": 12.0, "hpa_scale_to": 8},
        },
        ["FinOps"],
        "medium",
    )

    # ------------------------------------------------------------------
    # 7. Incident review - DevOps/SRE primary
    # ------------------------------------------------------------------
    add(
        "incident_review",
        "DevOps",
        "failed_deployment_caused_incident",
        "Rollback to stable deployment",
        {
            "deploy": {"pipeline_failed": True, "rollback_marker": True, "restart_loops": 14},
            "sre": {"error_rate_pct": 12.0, "p95_latency_ms": 760.0},
        },
        ["SRE"],
        "high",
    )
    add(
        "incident_review",
        "SRE",
        "cascading_service_errors",
        "Mitigate and monitor",
        {
            "sre": {"error_rate_pct": 13.0, "p95_latency_ms": 680.0, "saturation_pct": 86.0},
            "finops": {"cost_spike_pct": 16.0, "hpa_scale_to": 9},
        },
        ["FinOps"],
        "high",
    )
    add(
        "incident_review",
        "DevSecOps",
        "security_policy_triggered_incident",
        "Patch or block release",
        {
            "sec": {"policy_violation": True, "iam_drift": True},
            "sre": {"error_rate_pct": 6.0, "p95_latency_ms": 410.0},
        },
        ["SRE"],
        "high",
    )

    # ------------------------------------------------------------------
    # 8. Policy propagation - DevSecOps primary
    # ------------------------------------------------------------------
    add(
        "policy_propagation",
        "DevSecOps",
        "policy_version_drift",
        "Patch or block release",
        {
            "sec": {"policy_violation": True, "compliance_gap": True},
        },
        [],
        "medium",
    )
    add(
        "policy_propagation",
        "DevOps",
        "missed_policy_update_in_pipeline",
        "Block release and fix pipeline",
        {
            "deploy": {"pipeline_failed": True, "config_drift": True},
            "sec": {"policy_violation": True},
        },
        ["DevSecOps"],
        "medium",
    )

    # ------------------------------------------------------------------
    # 9. Cost-performance simulation - FinOps/SRE trade-off
    # ------------------------------------------------------------------
    add(
        "cost_performance_simulation",
        "FinOps",
        "budget_pressure_with_stable_slo",
        "Review scaling policy",
        {
            "finops": {"cost_spike_pct": 30.0, "hpa_scale_to": 13},
            "sre": {"p95_latency_ms": 210.0, "error_rate_pct": 1.0, "saturation_pct": 58.0},
        },
        ["SRE"],
        "medium",
    )
    add(
        "cost_performance_simulation",
        "SRE",
        "performance_risk_requires_capacity",
        "Scale adjustment",
        {
            "sre": {"p95_latency_ms": 820.0, "error_rate_pct": 7.0, "saturation_pct": 90.0},
            "finops": {"cost_spike_pct": 20.0, "hpa_scale_to": 10},
        },
        ["FinOps"],
        "high",
    )

    # ------------------------------------------------------------------
    # 10. Anomaly-to-insight - cross-domain
    # ------------------------------------------------------------------
    add(
        "anomaly_to_insight",
        "DevOps",
        "release_change_triggered_multi_signal_anomaly",
        "Rollback to stable deployment",
        {
            "deploy": {"config_drift": True, "rollback_marker": True, "restart_loops": 10},
            "sre": {"p95_latency_ms": 600.0, "error_rate_pct": 9.0},
            "finops": {"cost_spike_pct": 18.0, "hpa_scale_to": 9},
            "sec": {"critical_cves": 0, "policy_violation": False},
        },
        ["SRE", "FinOps"],
        "high",
    )

    return scenarios


def save_scenarios(path: str, scenarios: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=2)
