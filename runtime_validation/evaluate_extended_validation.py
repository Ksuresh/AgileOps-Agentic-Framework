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
MANIFEST = Path(__file__).with_name("interventions_extended_validation.yaml")
OUT = ROOT / "results_extended_validation"


def manifest() -> dict[str, dict[str, Any]]:
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    return {case["id"]: case for case in data["cases"]}


def evaluate(case_id: str, rep: int, baseline: Path, cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    case_dir = latest_artifact(case_id, rep)
    if case_dir is None:
        raise FileNotFoundError(f"{case_id} rep {rep}")
    telemetry = write_evidence(case_dir, baseline_dir=baseline)
    telemetry = normalize_runtime_evidence(case_dir, telemetry)
    telemetry = classify_batch3_config(case_dir, telemetry)
    oracle = set(cases[case_id]["admissible_actions"])
    baseline_action, baseline_severity, baseline_domain = choose_dominant_domain_action(telemetry)
    no_int = choose_action_details(telemetry, (0.4, 0.3, 0.3))["selected_action"]
    full = apply_interaction_policy(telemetry, no_int)
    dominant = full["interaction_state"].get("dominant_interaction")
    return {
        "case_id": case_id,
        "repetition": rep,
        "analysis_role": cases[case_id].get("analysis_role", "independent_template"),
        "oracle_actions": json.dumps(sorted(oracle)),
        "baseline_action": baseline_action,
        "baseline_match": baseline_action in oracle,
        "baseline_domain": baseline_domain,
        "baseline_severity": round(float(baseline_severity), 4),
        "aaf_no_interaction_action": no_int,
        "aaf_no_interaction_match": no_int in oracle,
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

    diagnostic = [evaluate("HRT-21", rep, baseline, cases) for rep in range(1, 6)]
    independent = [evaluate(f"HRT-{i}", 1, baseline, cases) for i in range(22, 32)]
    OUT.mkdir(parents=True, exist_ok=True)

    for name, rows in (("hrt21_diagnostic_repetitions.csv", diagnostic), ("new_independent_templates.csv", independent)):
        with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader(); writer.writerows(rows)

    summary = {
        "diagnostic_repetitions": {
            "case_id": "HRT-21",
            "n_repetitions": 5,
            "not_counted_as_independent_n": True,
            "full_aaf_match_rate": sum(r["aaf_full_match"] for r in diagnostic) / 5,
            "no_interaction_match_rate": sum(r["aaf_no_interaction_match"] for r in diagnostic) / 5,
            "dominant_domain_match_rate": sum(r["baseline_match"] for r in diagnostic) / 5,
            "full_actions": [r["aaf_full_action"] for r in diagnostic],
            "dominant_interactions": [r["dominant_interaction"] for r in diagnostic],
        },
        "new_independent_templates": {
            "n": len(independent),
            "full_aaf_agreement": sum(r["aaf_full_match"] for r in independent) / len(independent),
            "no_interaction_agreement": sum(r["aaf_no_interaction_match"] for r in independent) / len(independent),
            "dominant_domain_agreement": sum(r["baseline_match"] for r in independent) / len(independent),
            "aaf_only_wins_vs_dominant": sum(r["aaf_full_match"] and not r["baseline_match"] for r in independent),
            "dominant_only_wins_vs_aaf": sum(r["baseline_match"] and not r["aaf_full_match"] for r in independent),
        },
        "interpretation": "HRT-21 repetitions are diagnostic stability checks only. HRT-22--HRT-31 are ten newly frozen independent scenario templates. No decision-policy or oracle retuning is performed after outcomes are observed.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
