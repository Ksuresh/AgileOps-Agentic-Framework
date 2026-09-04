from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from orchestrator.cross_domain import apply_interaction_policy as apply_v1
from orchestrator.cross_domain_v2 import apply_interaction_policy_v2 as apply_v2
from orchestrator.decision_baselines import choose_dominant_domain_action
from orchestrator.utility import choose_action_details
from runtime_validation.evidence_adapter import write_evidence
from runtime_validation.evaluate_heldout_runtime import normalize_runtime_evidence, latest_artifact
from runtime_validation.evaluate_batch3_runtime import classify_batch3_config

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("interventions_v2_prospective.yaml")


def _cases() -> dict[str, dict[str, Any]]:
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    return {c["id"]: c for c in data["cases"]}


def evaluate(case_id: str, baseline: Path, cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    case_dir = latest_artifact(case_id, 1)
    if case_dir is None:
        raise FileNotFoundError(case_id)
    telemetry = write_evidence(case_dir, baseline_dir=baseline)
    telemetry = normalize_runtime_evidence(case_dir, telemetry)
    telemetry = classify_batch3_config(case_dir, telemetry)
    (case_dir / "v2_telemetry_evaluated.json").write_text(json.dumps(telemetry, indent=2), encoding="utf-8")

    oracle = set(cases[case_id]["admissible_actions"])
    baseline_action, _, _ = choose_dominant_domain_action(telemetry)
    utility = choose_action_details(telemetry, (0.4, 0.3, 0.3))
    no_int = utility["selected_action"]
    v1 = apply_v1(telemetry, no_int)
    v2 = apply_v2(telemetry, no_int)

    return {
        "case_id": case_id,
        "oracle_actions": json.dumps(sorted(oracle)),
        "dominant_domain_action": baseline_action,
        "dominant_domain_match": baseline_action in oracle,
        "no_interaction_action": no_int,
        "no_interaction_match": no_int in oracle,
        "aaf_v1_action": v1["selected_action"],
        "aaf_v1_match": v1["selected_action"] in oracle,
        "aaf_v2_action": v2["selected_action"],
        "aaf_v2_match": v2["selected_action"] in oracle,
        "v1_interaction": (v1.get("interaction_state") or {}).get("dominant_interaction", {}).get("name") if (v1.get("interaction_state") or {}).get("dominant_interaction") else None,
        "v2_interaction": (v2.get("interaction_state") or {}).get("dominant_interaction", {}).get("name") if (v2.get("interaction_state") or {}).get("dominant_interaction") else None,
        "v2_arbitration_reason": v2.get("arbitration_reason"),
        "p95_latency_ms": telemetry.get("sre", {}).get("p95_latency_ms"),
        "error_rate_pct": telemetry.get("sre", {}).get("error_rate_pct"),
        "resource_footprint_proxy_pct": telemetry.get("finops", {}).get("cost_spike_pct"),
    }


def main() -> None:
    cases = _cases()
    baseline = latest_artifact("HRT-01", 1)
    if baseline is None:
        raise FileNotFoundError("HRT-01 baseline required")
    ids = [f"HRT-{i}" for i in range(32, 40)]
    rows = [evaluate(case_id, baseline, cases) for case_id in ids]

    out = ROOT / "results_v2_prospective_runtime"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "runtime_case_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    n = len(rows)
    summary = {
        "n_independent_prospective_templates": n,
        "runtime_system": "Sock Shop",
        "dominant_domain_agreement": sum(r["dominant_domain_match"] for r in rows) / n,
        "no_interaction_agreement": sum(r["no_interaction_match"] for r in rows) / n,
        "aaf_v1_agreement": sum(r["aaf_v1_match"] for r in rows) / n,
        "aaf_v2_agreement": sum(r["aaf_v2_match"] for r in rows) / n,
        "v2_only_wins_vs_v1": sum(r["aaf_v2_match"] and not r["aaf_v1_match"] for r in rows),
        "v1_only_wins_vs_v2": sum(r["aaf_v1_match"] and not r["aaf_v2_match"] for r in rows),
        "interpretation": "Prospective post-refinement validation; scenarios frozen after v2 definition and before execution.",
    }
    (out / "runtime_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
