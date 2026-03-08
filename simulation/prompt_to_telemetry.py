from __future__ import annotations

from typing import Dict, Any


def build_telemetry_from_prompt_context(route: Dict[str, Any]) -> Dict[str, Any]:
    symptoms = set(route.get("symptoms", []))
    priority = route.get("priority", "medium")

    telemetry = {
        "deploy": {
            "restart_loops": 0,
            "config_drift": False,
            "pipeline_failed": False,
        },
        "sre": {
            "p95_latency_ms": 220.0,
            "error_rate_pct": 2.0,
            "saturation_pct": 70.0,
        },
        "finops": {
            "cost_spike_pct": 8.0,
            "hpa_scale_to": 7,
        },
        "sec": {
            "critical_cves": 0,
            "policy_violation": False,
            "iam_drift": False,
        },
    }

    if "deployment_failure" in symptoms:
        telemetry["deploy"]["pipeline_failed"] = True
        telemetry["deploy"]["config_drift"] = True
        telemetry["deploy"]["restart_loops"] = 14 if priority != "low" else 8

    if "latency_spike" in symptoms:
        telemetry["sre"]["p95_latency_ms"] = 900.0 if priority == "high" else 600.0
        telemetry["sre"]["error_rate_pct"] = 14.0 if priority == "high" else 8.5
        telemetry["sre"]["saturation_pct"] = 92.0 if priority == "high" else 86.0

    if "cost_spike" in symptoms:
        telemetry["finops"]["cost_spike_pct"] = 40.0 if priority == "high" else 22.0
        telemetry["finops"]["hpa_scale_to"] = 16 if priority == "high" else 11

    if "security_violation" in symptoms:
        telemetry["sec"]["critical_cves"] = 2 if priority == "high" else 1
        telemetry["sec"]["policy_violation"] = True
        telemetry["sec"]["iam_drift"] = True if priority == "high" else False

    if "general_investigation" in symptoms:
        telemetry["deploy"]["restart_loops"] = 6
        telemetry["sre"]["p95_latency_ms"] = 450.0
        telemetry["sre"]["error_rate_pct"] = 8.0
        telemetry["finops"]["cost_spike_pct"] = 22.0

    return telemetry
