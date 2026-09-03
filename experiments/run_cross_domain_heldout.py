from __future__ import annotations

"""Frozen controlled validation for the cross-domain AAF hypothesis.

Important protocol property: the interaction policy was committed before this
held-out case matrix. Do not tune orchestrator/cross_domain.py against outcomes
from this file. If the policy is changed after inspecting these results, this
matrix becomes development evidence and a new held-out matrix is required.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from orchestrator.cross_domain import apply_interaction_policy
from orchestrator.decision_baselines import choose_dominant_domain_action
from orchestrator.utility import choose_action_details

OUT = Path("results_cross_domain_heldout")


def ev(status: str = "measured", source: str = "controlled held-out intervention") -> Dict[str, str]:
    return {"status": status, "source": source, "note": ""}


def telemetry(*, restart_loops=0, burst_count=None, burst_window=None,
              pipeline_failed=False, config_drift=False, artifact_mismatch=False,
              p95=120.0, error=0.2, availability=99.9, saturation=None,
              cost_spike=0.0, cves=0, policy=False) -> Dict[str, Any]:
    deploy = {
        "restart_loops": restart_loops,
        "pipeline_failed": pipeline_failed,
        "config_drift": config_drift,
        "artifact_mismatch": artifact_mismatch,
        "_evidence": {
            "restart_loops": ev(), "pipeline_failed": ev(), "config_drift": ev(),
            "artifact_mismatch": ev(), "rollback_marker": ev("missing"),
        },
    }
    if burst_count is not None:
        deploy["restart_burst_count"] = burst_count
        deploy["_evidence"]["restart_burst_count"] = ev()
    if burst_window is not None:
        deploy["restart_window_seconds"] = burst_window
        deploy["_evidence"]["restart_window_seconds"] = ev()

    sre = {
        "p95_latency_ms": p95,
        "error_rate_pct": error,
        "availability_pct": availability,
        "_evidence": {
            "p95_latency_ms": ev(), "error_rate_pct": ev(), "availability_pct": ev(),
            "saturation_pct": ev("missing"),
        },
    }
    if saturation is not None:
        sre["saturation_pct"] = saturation
        sre["_evidence"]["saturation_pct"] = ev()

    finops = {
        "cost_spike_pct": cost_spike,
        "_evidence": {
            "cost_spike_pct": ev(), "hpa_scale_to": ev("missing"),
            "cpu_request_increase_pct": ev("missing"), "memory_request_increase_pct": ev("missing"),
        },
    }
    sec = {
        "critical_cves": cves,
        "policy_violation": policy,
        "_evidence": {
            "critical_cves": ev(), "policy_violation": ev(),
            "iam_drift": ev("missing"), "compliance_gap": ev("missing"),
        },
    }
    return {"deploy": deploy, "sre": sre, "finops": finops, "sec": sec}


CASES: List[Dict[str, Any]] = [
    # Single-domain controls: AAF should not need interaction to remain competent.
    {"id":"HC-01","group":"single","oracle":["Block release and fix pipeline"],
     "telemetry":telemetry(pipeline_failed=True)},
    {"id":"HC-02","group":"single","oracle":["Mitigate and monitor"],
     "telemetry":telemetry(availability=97.5)},
    {"id":"HC-03","group":"single","oracle":["Scale adjustment","Review scaling policy"],
     "telemetry":telemetry(cost_spike=42.0)},
    {"id":"HC-04","group":"single","oracle":["Patch or block release"],
     "telemetry":telemetry(cves=2)},

    # Straight compounds: two material domains, but no special causal dependency.
    {"id":"HC-05","group":"straight_compound","oracle":["Scale adjustment","Mitigate and monitor"],
     "telemetry":telemetry(p95=520.0, cost_spike=38.0)},
    {"id":"HC-06","group":"straight_compound","oracle":["Patch or block release"],
     "telemetry":telemetry(cves=2, availability=99.9)},
    {"id":"HC-07","group":"straight_compound","oracle":["Rollback to stable deployment","Mitigate and monitor"],
     "telemetry":telemetry(config_drift=True, p95=500.0)},
    {"id":"HC-08","group":"straight_compound","oracle":["Patch or block release","Block release and fix pipeline"],
     "telemetry":telemetry(artifact_mismatch=True, cves=1)},

    # Decision-critical compounds: correct governance depends on interaction.
    {"id":"HC-09","group":"decision_critical","oracle":["Rollback to stable deployment"],
     "telemetry":telemetry(burst_count=4, burst_window=70.0, p95=470.0)},
    {"id":"HC-10","group":"decision_critical","oracle":["Rollback to stable deployment"],
     "telemetry":telemetry(burst_count=3, burst_window=45.0, error=8.5)},
    {"id":"HC-11","group":"decision_critical","oracle":["Patch or block release"],
     "telemetry":telemetry(artifact_mismatch=True, cves=1, availability=99.9)},
    {"id":"HC-12","group":"decision_critical","oracle":["Patch or block release"],
     "telemetry":telemetry(pipeline_failed=True, policy=True, p95=120.0)},
    {"id":"HC-13","group":"decision_critical","oracle":["Scale adjustment"],
     "telemetry":telemetry(p95=500.0, cost_spike=30.0)},
    {"id":"HC-14","group":"decision_critical","oracle":["Scale adjustment"],
     "telemetry":telemetry(error=9.0, cost_spike=25.0)},
    {"id":"HC-15","group":"decision_critical","oracle":["Patch or block release"],
     "telemetry":telemetry(config_drift=True, p95=500.0, cves=1)},
    {"id":"HC-16","group":"decision_critical","oracle":["Patch or block release"],
     "telemetry":telemetry(artifact_mismatch=True, error=8.5, policy=True)},
]


def run_case(case: Dict[str, Any]) -> Dict[str, Any]:
    t = case["telemetry"]
    oracle = set(case["oracle"])
    baseline_action, baseline_severity, baseline_domain = choose_dominant_domain_action(t)
    no_interaction = choose_action_details(t, (0.4, 0.3, 0.3))["selected_action"]
    full = apply_interaction_policy(t, no_interaction)
    full_action = full["selected_action"]
    dominant = full["interaction_state"].get("dominant_interaction")
    return {
        "case_id": case["id"],
        "group": case["group"],
        "oracle_actions": json.dumps(sorted(oracle)),
        "baseline_action": baseline_action,
        "baseline_match": baseline_action in oracle,
        "baseline_domain": baseline_domain,
        "baseline_severity": round(float(baseline_severity), 4),
        "aaf_no_interaction_action": no_interaction,
        "aaf_no_interaction_match": no_interaction in oracle,
        "aaf_full_action": full_action,
        "aaf_full_match": full_action in oracle,
        "interaction_applied": bool(full["interaction_applied"]),
        "dominant_interaction": None if dominant is None else dominant["name"],
    }


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": len(rows), "protocol": "frozen controlled held-out evaluation; not production accuracy"}
    for group in ["single", "straight_compound", "decision_critical", "all"]:
        subset = rows if group == "all" else [r for r in rows if r["group"] == group]
        if not subset:
            continue
        out[group] = {
            "n": len(subset),
            "dominant_domain_baseline": sum(bool(r["baseline_match"]) for r in subset) / len(subset),
            "aaf_no_interaction": sum(bool(r["aaf_no_interaction_match"]) for r in subset) / len(subset),
            "aaf_full": sum(bool(r["aaf_full_match"]) for r in subset) / len(subset),
        }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [run_case(c) for c in CASES]
    with (OUT / "case_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    summary = aggregate(rows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
