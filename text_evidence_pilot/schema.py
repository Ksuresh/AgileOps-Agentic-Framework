"""Decision-relevant evidence fields that a text reader may fill.

Taken directly from the frozen deterministic AAF modules (agents/*.py,
orchestrator/decision_gate.py, orchestrator/cross_domain_v2.py). Readers may
only output these fields; they never output governance actions.
"""
from __future__ import annotations

FIELD_SPECS: dict[str, dict[str, str]] = {
    # DevOps
    "deploy.pipeline_failed": {"type": "bool", "desc": "CI/CD pipeline for the release failed"},
    "deploy.config_drift": {"type": "bool", "desc": "configuration drift detected in the release/environment"},
    "deploy.artifact_mismatch": {"type": "bool", "desc": "deployed artifact/image digest does not match the approved build"},
    "deploy.restart_burst_count": {"type": "number", "unit": "count", "desc": "number of container/pod restarts in the burst after a deployment"},
    "deploy.restart_window_seconds": {"type": "number", "unit": "seconds", "desc": "time window (seconds) in which that restart burst happened"},
    # SRE
    "sre.p95_latency_ms": {"type": "number", "unit": "ms", "desc": "current p95 request latency in milliseconds"},
    "sre.error_rate_pct": {"type": "number", "unit": "%", "desc": "current request error rate in percent"},
    "sre.saturation_pct": {"type": "number", "unit": "%", "desc": "current resource saturation/utilisation in percent"},
    "sre.availability_pct": {"type": "number", "unit": "%", "desc": "current availability in percent"},
    # FinOps
    "finops.cost_spike_pct": {"type": "number", "unit": "%", "desc": "increase in cost or cost-proxy versus baseline, percent"},
    "finops.hpa_scale_to": {"type": "number", "unit": "replicas", "desc": "number of replicas/pods the autoscaler scaled to"},
    "finops.cpu_request_increase_pct": {"type": "number", "unit": "%", "desc": "increase in CPU requests, percent"},
    "finops.memory_request_increase_pct": {"type": "number", "unit": "%", "desc": "increase in memory requests, percent"},
    # DevSecOps
    "sec.critical_cves": {"type": "number", "unit": "count", "desc": "number of critical CVEs in the release"},
    "sec.policy_violation": {"type": "bool", "desc": "policy-as-code / release policy gate violated"},
    "sec.iam_drift": {"type": "bool", "desc": "IAM permissions drifted from the approved baseline"},
}


def split(field: str) -> tuple[str, str]:
    block, name = field.split(".", 1)
    return block, name
