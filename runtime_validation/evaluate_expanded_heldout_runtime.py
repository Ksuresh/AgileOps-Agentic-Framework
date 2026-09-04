from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from orchestrator.cross_domain import apply_interaction_policy
from orchestrator.decision_baselines import choose_dominant_domain_action
from orchestrator.utility import choose_action_details
from runtime_validation.evidence_adapter import write_evidence
from runtime_validation.evaluate_heldout_runtime import normalize_runtime_evidence, latest_artifact
from runtime_validation.evaluate_batch3_runtime import classify_batch3_config

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("interventions_expanded_heldout.yaml")


def manifest() -> dict[str, dict[str, Any]]:
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    return {case["id"]: case for case in data["cases"]}


def evaluate(case_id: str, baseline: Path, cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    case_dir = latest_artifact(case_id, 1)
    if case_dir is None:
        raise FileNotFoundError(case_id)

    telemetry = write_evidence(case_dir, baseline_dir=baseline)
    telemetry = normalize_runtime_evidence(case_dir, telemetry)
    telemetry = classify_batch3_config(case_dir, telemetry)
    (case_dir / "expanded_telemetry_evaluated.json").write_text(
        json.dumps(telemetry, indent=2), encoding="utf-8"
    )

    oracle = set(cases[case_id]["admissible_actions"])
    baseline_action, baseline_severity, baseline_domain = choose_dominant_domain_action(telemetry)
    no_interaction = choose_action_details(telemetry, (0.4, 0.3, 0.3))["selected_action"]
    full = apply_interaction_policy(telemetry, no_interaction)
    dominant = full["interaction_state"].get("dominant_interaction")

    return {
        "case_id": case_id,
        "oracle_actions": json.dumps(sorted(oracle)),
        "baseline_action": baseline_action,
        "baseline_match": baseline_action in oracle,
        "baseline_domain": baseline_domain,
        "baseline_severity": round(float(baseline_severity), 4),
        "aaf_no_interaction_action": no_interaction,
        "aaf_no_interaction_match": no_interaction in oracle,
        "aaf_full_action": full["selected_action"],
        "aaf_full_match": full["selected_action"] in oracle,
        "interaction_applied": bool(full["interaction_applied"]),
        "dominant_interaction": None if dominant is None else dominant["name"],
        "p95_latency_ms": telemetry.get("sre", {}).get("p95_latency_ms"),
        "error_rate_pct": telemetry.get("sre", {}).get("error_rate_pct"),
        "resource_footprint_proxy_pct": telemetry.get("finops", {}).get("cost_spike_pct"),
        "config_marker": telemetry.get("_runtime_observables", {}).get("batch3_config_marker"),
        "restart_burst_count": telemetry.get("deploy", {}).get("restart_burst_count"),
    }


def main() -> None:
    cases = manifest()
    baseline = latest_artifact("HRT-01", 1)
    if baseline is None:
        raise FileNotFoundError("HRT-01 baseline required")

    ids = [f"HRT-{i}" for i in range(14, 22)]
    rows = [evaluate(case_id, baseline, cases) for case_id in ids]

    out = ROOT / "results_expanded_heldout_runtime"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "runtime_case_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "n_independent_scenario_templates": len(rows),
        "runtime_system": "Sock Shop",
        "evaluation_reference": "frozen HRT-14--HRT-21 expanded held-out oracle",
        "dominant_domain_baseline_action_oracle_agreement": sum(r["baseline_match"] for r in rows) / len(rows),
        "aaf_no_interaction_action_oracle_agreement": sum(r["aaf_no_interaction_match"] for r in rows) / len(rows),
        "aaf_full_action_oracle_agreement": sum(r["aaf_full_match"] for r in rows) / len(rows),
        "aaf_only_wins_vs_dominant": sum(r["aaf_full_match"] and not r["baseline_match"] for r in rows),
        "dominant_only_wins_vs_aaf": sum(r["baseline_match"] and not r["aaf_full_match"] for r in rows),
        "interpretation": "Independent held-out scenario-template analysis; one execution per template; no policy or threshold retuning.",
    }
    (out / "runtime_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
