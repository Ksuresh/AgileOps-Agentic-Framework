from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from evaluation.statistical_methods import wilson_interval
from pipeline import run_pipeline
from runtime_validation.evidence_adapter import write_evidence

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("interventions.yaml")


def load_manifest() -> Dict[str, Dict[str, Any]]:
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    return {case["id"]: case for case in data["cases"]}


def latest_artifact(case_id: str, repetition: int) -> Optional[Path]:
    base = ROOT / "runtime_validation" / "artifacts" / case_id / f"rep-{repetition}"
    dirs = sorted([p for p in base.glob("*") if p.is_dir()]) if base.exists() else []
    return dirs[-1] if dirs else None


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def enrich_with_direct_runtime_observations(case_dir: Path, telemetry: Dict[str, Any]) -> Dict[str, Any]:
    """Attach direct measurements captured during the intervention window.

    These files are produced by runtime collectors/executors without reading the
    experimental oracle. They repair a sampling limitation of post-hoc snapshots:
    transient restart events and request latency/error observations can disappear
    before the ordinary artifact collector runs.
    """
    load = _load_json(case_dir / "load_observation.json")
    if load:
        p95 = load.get("latency_p95_ms")
        error_rate = load.get("error_rate_pct")
        sre = telemetry.setdefault("sre", {})
        evidence = sre.setdefault("_evidence", {})
        if p95 is not None:
            sre["p95_latency_ms"] = float(p95)
            evidence["p95_latency_ms"] = {
                "status": "measured",
                "source": str(load.get("source", "direct HTTP load observation")),
                "note": "Measured during the active runtime load window.",
            }
        if error_rate is not None:
            sre["error_rate_pct"] = float(error_rate)
            evidence["error_rate_pct"] = {
                "status": "measured",
                "source": str(load.get("source", "direct HTTP load observation")),
                "note": "HTTP request failure percentage measured during the active load window.",
            }
        telemetry.setdefault("_runtime_observables", {})["load_observation"] = load

    temporal = _load_json(case_dir / "temporal_process_observation.json")
    if temporal:
        restarts = temporal.get("observed_restart_events")
        if restarts is not None:
            deploy = telemetry.setdefault("deploy", {})
            deploy["restart_loops"] = int(restarts)
            deploy.setdefault("_evidence", {})["restart_loops"] = {
                "status": "measured",
                "source": str(temporal.get("source", "temporal Docker process observation")),
                "note": "Observed process start-time transitions during the intervention window.",
            }
        telemetry.setdefault("_runtime_observables", {})["temporal_process_observation"] = temporal

    return telemetry


def evaluate_case(case_id: str, repetition: int, baseline_dir: Optional[Path], manifest: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    case_dir = latest_artifact(case_id, repetition)
    if case_dir is None:
        raise FileNotFoundError(f"No artifacts found for {case_id} repetition {repetition}")
    telemetry = write_evidence(case_dir, baseline_dir=baseline_dir)
    telemetry = enrich_with_direct_runtime_observations(case_dir, telemetry)
    # Persist the exact telemetry evaluated after direct-observation enrichment.
    (case_dir / "aaf_telemetry_evaluated.json").write_text(json.dumps(telemetry, indent=2), encoding="utf-8")
    scenario = {
        "scenario_id": f"{case_id}-rep-{repetition}",
        "telemetry": telemetry,
        "thresholds": {"tau_consensus": 0.65, "delta_min": 0.05, "max_rar_loops": 1},
        "lam": 0.5,
        "utility_weights": (0.4, 0.3, 0.3),
    }
    full = run_pipeline(scenario, mode="aaf_full")
    baseline = run_pipeline(scenario, mode="aaf_no_utility")
    oracle = manifest[case_id]
    oracle_domains = set(oracle.get("oracle_domains", []))
    admissible_actions = set(oracle.get("admissible_actions", []))
    predicted_domain = full.predicted_primary_domain
    domain_match = None if not oracle_domains else predicted_domain in oracle_domains
    action = full.utility.get("selected_action")
    base_action = baseline.utility.get("selected_action")
    return {
        "case_id": case_id,
        "repetition": repetition,
        "artifact_dir": str(case_dir.relative_to(ROOT)),
        "oracle_domains": json.dumps(sorted(oracle_domains)),
        "oracle_actions": json.dumps(sorted(admissible_actions)),
        "predicted_primary_domain": predicted_domain,
        "domain_oracle_match": domain_match,
        "selected_action": action,
        "action_oracle_match": action in admissible_actions,
        "baseline_action": base_action,
        "baseline_action_oracle_match": base_action in admissible_actions,
        "consensus_score": full.consensus_score,
        "rer_triggered": bool(full.rar.get("triggered")),
        "rer_accepted": bool(full.rar.get("accepted")),
        "rer_escalated": bool(full.rar.get("escalated")),
        "utility_score": full.utility.get("best_utility"),
        "measured_p95_latency_ms": telemetry.get("sre", {}).get("p95_latency_ms"),
        "measured_error_rate_pct": telemetry.get("sre", {}).get("error_rate_pct"),
        "observed_restart_events": telemetry.get("deploy", {}).get("restart_loops"),
        "evidence_schema_version": telemetry.get("_evidence_schema_version"),
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    action_hits = sum(bool(r["action_oracle_match"]) for r in rows)
    base_hits = sum(bool(r["baseline_action_oracle_match"]) for r in rows)
    domain_rows = [r for r in rows if r["domain_oracle_match"] is not None]
    domain_hits = sum(bool(r["domain_oracle_match"]) for r in domain_rows)
    return {
        "n": len(rows),
        "runtime_system": "Sock Shop",
        "evaluation_reference": "pre_specified_runtime_intervention_oracle",
        "action_oracle_agreement": action_hits / len(rows) if rows else 0.0,
        "action_oracle_wilson_95": wilson_interval(action_hits, len(rows)) if rows else (0.0, 0.0),
        "dominant_domain_baseline_action_oracle_agreement": base_hits / len(rows) if rows else 0.0,
        "domain_oracle_agreement": domain_hits / len(domain_rows) if domain_rows else None,
        "domain_oracle_wilson_95": wilson_interval(domain_hits, len(domain_rows)) if domain_rows else None,
        "rer_trigger_rate": sum(bool(r["rer_triggered"]) for r in rows) / len(rows) if rows else 0.0,
        "interpretation": "Runtime intervention-oracle agreement using preserved direct measurements; not production accuracy.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="RT-01,RT-02,RT-09")
    ap.add_argument("--repetitions", default="1,2,3")
    ap.add_argument("--out", default="results_revision_runtime_pilot")
    args = ap.parse_args()
    case_ids = [x.strip() for x in args.cases.split(",") if x.strip()]
    reps = [int(x.strip()) for x in args.repetitions.split(",") if x.strip()]
    manifest = load_manifest()
    baseline = latest_artifact("RT-01", reps[0])
    rows = [evaluate_case(cid, rep, baseline, manifest) for cid in case_ids for rep in reps]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with (out / "runtime_case_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    summary = summarize(rows)
    (out / "runtime_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
