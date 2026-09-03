from __future__ import annotations

"""Reviewer-aligned extension of the original 30-case controlled benchmark.

This module preserves TC-01..TC-30 unchanged and adds only the cases identified
by the frozen experiment coverage audit: healthy/no-action controls, explicit
incomplete/ambiguous evidence cases, and cross-domain governance conflicts.

The scenario oracle is NOT inferred from AAF output. Oracle definitions live in
benchmark/reviewer_oracle_specs.py and are frozen independently.
"""

import copy
import random
from typing import Any, Dict, List

from scenario_generator.generate import generate_scenarios

DEFAULT_THRESHOLDS = {"tau_consensus": 0.65, "delta_min": 0.05, "max_rar_loops": 2}
DEFAULT_UTILITY_WEIGHTS = (0.4, 0.3, 0.3)


def _ev(status: str, source: str, note: str = "") -> Dict[str, str]:
    return {"status": status, "source": source, "note": note}


def _base() -> Dict[str, Any]:
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
    case_id: str,
    group: str,
    category: str,
    telemetry: Dict[str, Any],
    primary_domain: str | None,
    expected_action: str,
    secondary_domains: List[str] | None = None,
    priority: str = "medium",
) -> Dict[str, Any]:
    return {
        "scenario_id": case_id,
        "incident_id": case_id,
        "category": category,
        "scenario_type": category,
        "experiment_group": group,
        "priority": priority,
        "telemetry": telemetry,
        "ground_truth": {
            "primary_domain": primary_domain,
            "secondary_domains": secondary_domains or [],
            "expected_action": expected_action,
            "recommended_action": expected_action,
        },
        "thresholds": copy.deepcopy(DEFAULT_THRESHOLDS),
        "utility_weights": DEFAULT_UTILITY_WEIGHTS,
        "lam": 0.5,
    }


def _set_missing(block: Dict[str, Any], fields: List[str], source: str, note: str) -> None:
    meta = block.setdefault("_evidence", {})
    for field in fields:
        block[field] = None
        meta[field] = _ev("missing", source, note)


def _set_proxy(block: Dict[str, Any], field: str, value: Any, source: str, note: str) -> None:
    block[field] = value
    block.setdefault("_evidence", {})[field] = _ev("proxy", source, note)


def _additional_base_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []

    # G5: healthy / no-action controls.
    t = _base(); cases.append(_case("TC-31", "G5", "healthy_release", t, None, "No action (observe)"))
    t = _base(); t["sre"]["saturation_pct"] = 62.0; cases.append(_case("TC-32", "G5", "benign_cpu_fluctuation", t, None, "No action (observe)"))
    t = _base(); t["finops"]["cost_spike_pct"] = 8.0; cases.append(_case("TC-33", "G5", "cost_within_policy", t, None, "No action (observe)"))
    t = _base(); t["sec"]["critical_cves"] = 0; cases.append(_case("TC-34", "G5", "non_blocking_security_context", t, None, "No action (observe)"))
    t = _base(); t["sre"]["p95_latency_ms"] = 240.0; cases.append(_case("TC-35", "G5", "transient_latency_below_threshold", t, None, "No action (observe)"))
    t = _base(); t["deploy"]["rollback_marker"] = True; cases.append(_case("TC-36", "G5", "clean_rollback_marker", t, None, "No action (observe)"))

    # G4: explicit evidence insufficiency / ambiguity. Missingness is represented
    # in the provenance schema, not by hidden oracle-dependent flags.
    t = _base(); t["deploy"]["config_drift"] = None
    _set_missing(t["deploy"], ["pipeline_failed", "config_drift", "rollback_marker", "artifact_mismatch", "restart_loops"], "release metadata source unavailable", "Deployment anomaly cannot be attributed without release evidence.")
    cases.append(_case("TC-37", "G4", "missing_release_metadata", t, None, "Escalate for evidence/human review"))

    t = _base(); _set_missing(t["sre"], ["p95_latency_ms", "error_rate_pct", "saturation_pct", "availability_pct"], "SRE telemetry source unavailable", "Reliability indication cannot be validated from source telemetry.")
    cases.append(_case("TC-38", "G4", "missing_sre_sources", t, None, "Escalate for evidence/human review"))

    t = _base(); _set_missing(t["finops"], ["cost_spike_pct", "hpa_scale_to", "cpu_request_increase_pct", "memory_request_increase_pct"], "billing/resource history unavailable", "Cost anomaly lacks a provenance-backed resource or billing source.")
    cases.append(_case("TC-39", "G4", "missing_finops_sources", t, None, "Escalate for evidence/human review"))

    t = _base(); _set_missing(t["sec"], ["critical_cves", "policy_violation", "iam_drift", "compliance_gap"], "security scanner provenance unavailable", "Security alert is unsupported by scanner/policy evidence.")
    cases.append(_case("TC-40", "G4", "missing_security_provenance", t, None, "Escalate for evidence/human review"))

    t = _base(); t["sre"].update({"p95_latency_ms": 760.0, "error_rate_pct": 10.0, "availability_pct": 98.0})
    _set_missing(t["deploy"], ["pipeline_failed", "config_drift", "rollback_marker", "artifact_mismatch", "restart_loops"], "deployment evidence suppressed", "Do not infer deployment causality from SRE symptoms alone.")
    cases.append(_case("TC-41", "G4", "compound_missing_deployment_evidence", t, "SRE", "Mitigate and monitor"))

    t = _base(); t["sre"].update({"p95_latency_ms": 780.0, "error_rate_pct": 8.0, "saturation_pct": 91.0})
    _set_missing(t["finops"], ["cost_spike_pct", "hpa_scale_to", "cpu_request_increase_pct", "memory_request_increase_pct"], "cost evidence suppressed", "Reliability decision may continue; no FinOps causal claim is admissible.")
    cases.append(_case("TC-42", "G4", "compound_missing_cost_evidence", t, "SRE", "Mitigate and monitor"))

    t = _base();
    _set_proxy(t["sre"], "availability_pct", 99.9, "source-A process proxy", "Conflicts with independent availability source.")
    t["sre"].setdefault("_evidence_conflicts", {})["availability_pct"] = {"source_b_value": 96.0, "source_b": "source-B synthetic controlled measurement"}
    cases.append(_case("TC-43", "G4", "conflicting_metric_sources", t, None, "Escalate for evidence/human review"))

    t = _base(); t["sre"].update({"p95_latency_ms": 720.0, "error_rate_pct": 7.0, "saturation_pct": 88.0}); t["finops"].update({"cost_spike_pct": 24.0, "hpa_scale_to": 10})
    _set_missing(t["sre"], ["error_rate_pct"], "controlled 25% evidence drop", "Robustness case; remaining evidence stays provenance-backed.")
    cases.append(_case("TC-44", "G4", "partial_noisy_evidence", t, "SRE", "Scale adjustment", ["FinOps"]))

    # G3: explicit cross-domain governance conflicts.
    t = _base(); t["sre"].update({"p95_latency_ms": 850.0, "error_rate_pct": 8.0, "saturation_pct": 92.0}); t["finops"].update({"cost_spike_pct": 42.0, "hpa_scale_to": 12})
    cases.append(_case("TC-45", "G3", "latency_vs_cost", t, "SRE", "Scale adjustment", ["FinOps"], "high"))

    t = _base(); t["deploy"].update({"config_drift": True, "rollback_marker": True, "restart_loops": 12}); t["sre"].update({"p95_latency_ms": 900.0, "error_rate_pct": 14.0, "availability_pct": 97.0})
    cases.append(_case("TC-46", "G3", "rollback_vs_mitigate", t, "DevOps", "Rollback to stable deployment", ["SRE"], "high"))

    t = _base(); t["sec"]["critical_cves"] = 2
    cases.append(_case("TC-47", "G3", "security_blocker_with_healthy_service", t, "DevSecOps", "Patch or block release", ["SRE"], "high"))

    t = _base(); t["sec"]["critical_cves"] = 0; t["sre"].update({"p95_latency_ms": 950.0, "error_rate_pct": 15.0, "availability_pct": 96.5})
    cases.append(_case("TC-48", "G3", "nonblocking_security_vs_outage", t, "SRE", "Mitigate and monitor", ["DevSecOps"], "high"))

    t = _base(); t["finops"].update({"cost_spike_pct": 35.0, "hpa_scale_to": 16})
    cases.append(_case("TC-49", "G3", "healthy_release_excessive_scale", t, "FinOps", "Review scaling policy", ["SRE", "DevOps"]))

    t = _base(); t["deploy"]["config_drift"] = True; t["finops"].update({"cost_spike_pct": 38.0, "hpa_scale_to": 14})
    cases.append(_case("TC-50", "G3", "config_drift_with_cost_spike", t, "DevOps", "Rollback to stable deployment", ["FinOps"]))

    t = _base(); t["sre"].update({"p95_latency_ms": 800.0, "saturation_pct": 93.0}); t["finops"].update({"cost_spike_pct": 45.0, "hpa_scale_to": 12})
    cases.append(_case("TC-51", "G3", "capacity_vs_budget_cap", t, "SRE", "Scale adjustment", ["FinOps"], "high"))

    t = _base(); t["sec"]["policy_violation"] = True; t["deploy"]["pipeline_failed"] = True
    cases.append(_case("TC-52", "G3", "security_and_pipeline_blockers", t, "DevSecOps", "Patch or block release", ["DevOps"], "high"))

    t = _base(); t["deploy"].update({"rollback_marker": True, "config_drift": True}); t["sre"].update({"p95_latency_ms": 700.0, "error_rate_pct": 9.0}); t["finops"]["cost_spike_pct"] = 25.0
    cases.append(_case("TC-53", "G3", "rollback_reliability_vs_temporary_cost", t, "DevOps", "Rollback to stable deployment", ["SRE", "FinOps"], "high"))

    t = _base(); t["sre"].update({"p95_latency_ms": 245.0, "error_rate_pct": 1.2, "saturation_pct": 64.0}); t["finops"]["cost_spike_pct"] = 9.0
    cases.append(_case("TC-54", "G3", "multi_domain_mild_warnings", t, None, "No action (observe)", ["SRE", "FinOps"]))

    return cases


def _perturb(case: Dict[str, Any], seed: int, jitter_pct: float = 0.03) -> Dict[str, Any]:
    """Create a deterministic, label-preserving numeric perturbation.

    Missing/proxy provenance is preserved. Booleans and discrete policy fields
    are not perturbed. This yields variants without changing the frozen oracle.
    """
    out = copy.deepcopy(case)
    rng = random.Random(f"{seed}:{case['scenario_id']}")
    for block_name in ("sre", "finops"):
        block = out["telemetry"].get(block_name, {})
        evidence = block.get("_evidence", {}) or {}
        for key, value in list(block.items()):
            if key.startswith("_") or value is None or isinstance(value, bool):
                continue
            if not isinstance(value, (int, float)):
                continue
            if (evidence.get(key, {}) or {}).get("status") == "missing":
                continue
            factor = 1.0 + rng.uniform(-jitter_pct, jitter_pct)
            block[key] = round(float(value) * factor, 3)
    out["variant_seed"] = seed
    return out


def generate_reviewer_aligned_scenarios(seed: int = 42) -> List[Dict[str, Any]]:
    # Preserve the original 30 cases; disable the old random missingness because
    # G4 now contains explicit, independently specified evidence-insufficiency cases.
    original = generate_scenarios(
        seed=seed,
        noise={"missing_evidence_prob": 0.0, "metric_jitter_pct": 0.0},
    )
    for case in original:
        secondary = case.get("ground_truth", {}).get("secondary_domains", []) or []
        case["experiment_group"] = "G2" if secondary else "G1"
        if case["scenario_id"] in {"TC-28", "TC-29"}:
            case["experiment_group"] = "G3"
    all_cases = original + _additional_base_cases()
    return [_perturb(case, seed=seed) for case in all_cases]
